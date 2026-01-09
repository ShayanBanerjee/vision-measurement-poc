# app_min_imu.py
# PoC: Phone planar measurement (length/width) using YOLO + FoV + camera height + IMU tilt.
# IMU is collected from a TOP-LEVEL static page (not inside Streamlit iframe) and returned via URL params.

import math
import urllib.parse
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import cv2
import streamlit as st


# -------------------------
# Geometry (simple, explainable)
# -------------------------
@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

def intrinsics_from_hfov(width: int, height: int, hfov_deg: float) -> Intrinsics:
    hfov = math.radians(float(hfov_deg))
    fx = (width / 2.0) / math.tan(hfov / 2.0)
    vfov = 2.0 * math.atan(math.tan(hfov / 2.0) * (height / width))
    fy = (height / 2.0) / math.tan(vfov / 2.0)
    return Intrinsics(width, height, fx, fy, width / 2.0, height / 2.0)

def pixel_to_unit_ray(u: float, v: float, K: Intrinsics) -> np.ndarray:
    # Camera frame: x right, y up, z forward. Pixel v down => flip sign for y.
    x = (u - K.cx) / K.fx
    y = -(v - K.cy) / K.fy
    z = 1.0
    r = np.array([x, y, z], dtype=np.float64)
    n = np.linalg.norm(r)
    return r / n if n > 0 else r

def Rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]], dtype=np.float64)

def Rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[ c, -s, 0],
                     [ s,  c, 0],
                     [ 0,  0, 1]], dtype=np.float64)

def base_R_looking_down() -> np.ndarray:
    """
    Base mapping when phone is approximately looking down at the plane:
      camera +z -> world -Y
      camera +x -> world +X
      camera +y -> world -Z
    World: X right, Y up, Z forward (arbitrary but consistent).
    """
    return np.array([[1, 0, 0],
                     [0, 0,-1],
                     [0,-1, 0]], dtype=np.float64)

def rotate_ray_to_world(ray_cam: np.ndarray, pitch_rad: float, roll_rad: float) -> np.ndarray:
    # PoC orientation: deltas around camera axes
    R0 = base_R_looking_down()
    R = R0 @ (Rz(roll_rad) @ Rx(pitch_rad))
    return R @ ray_cam

def intersect_plane_Y0(camera_height_m: float, ray_world: np.ndarray, eps: float = 1e-9) -> Optional[np.ndarray]:
    # Plane: Y=0. Camera origin: C=(0,h,0). Ray: P=C+t*r
    C = np.array([0.0, float(camera_height_m), 0.0], dtype=np.float64)
    denom = ray_world[1]
    if abs(denom) < eps:
        return None
    t = (0.0 - C[1]) / denom
    if t <= 0:
        return None
    return C + t * ray_world

def estimate_length_width_from_bbox(
    xyxy: Tuple[float, float, float, float],
    img_w: int,
    img_h: int,
    hfov_deg: float,
    camera_height_m: float,
    pitch_rad: float,
    roll_rad: float,
) -> Optional[Tuple[float, float]]:
    """
    Projects bbox corners onto plane (Y=0) and estimates planar length/width.
    Uses edge lengths of projected quad (simple, explainable).
    """
    if camera_height_m <= 0:
        return None

    x1, y1, x2, y2 = map(float, xyxy)
    K = intrinsics_from_hfov(img_w, img_h, hfov_deg)

    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]  # TL, TR, BR, BL
    ptsXZ = []

    for (u, v) in corners:
        r_cam = pixel_to_unit_ray(u, v, K)
        r_w = rotate_ray_to_world(r_cam, pitch_rad, roll_rad)
        P = intersect_plane_Y0(camera_height_m, r_w)
        if P is None:
            return None
        ptsXZ.append([P[0], P[2]])  # (X,Z) plane coords

    ptsXZ = np.array(ptsXZ, dtype=np.float64)

    def dist(i: int, j: int) -> float:
        return float(np.linalg.norm(ptsXZ[i] - ptsXZ[j]))

    top = dist(0, 1)
    right = dist(1, 2)
    bottom = dist(2, 3)
    left = dist(3, 0)

    length = max(top, bottom)
    width = max(left, right)
    if width > length:
        length, width = width, length
    return length, width


# -------------------------
# YOLO (Ultralytics) - cached
# -------------------------
@st.cache_resource
def load_yolo(model_name: str = "yolov8n.pt"):
    from ultralytics import YOLO
    return YOLO(model_name)

def pick_best_det(results) -> Optional[Dict[str, Any]]:
    """
    Pick highest-confidence detection.
    Returns dict: {xyxy, conf, cls, name}
    """
    if not results:
        return None
    r0 = results[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return None

    conf = r0.boxes.conf.detach().cpu().numpy()
    xyxy = r0.boxes.xyxy.detach().cpu().numpy()
    cls = r0.boxes.cls.detach().cpu().numpy().astype(int)

    i = int(np.argmax(conf))
    name = r0.names[int(cls[i])] if hasattr(r0, "names") else str(int(cls[i]))
    return {"xyxy": tuple(map(float, xyxy[i])), "conf": float(conf[i]), "cls": int(cls[i]), "name": name}


# -------------------------
# Query param helpers (robust across Streamlit versions)
# -------------------------
def _get_qp() -> Dict[str, Any]:
    # Streamlit new API: st.query_params behaves like a mapping
    try:
        return dict(st.query_params)  # may convert values to str
    except Exception:
        return st.experimental_get_query_params()

def qp_float(name: str) -> Optional[float]:
    qp = _get_qp()
    v = qp.get(name)
    if v is None:
        return None
    # could be list (old API) or str (new API)
    if isinstance(v, (list, tuple)):
        v = v[0] if v else None
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None

def qp_str(name: str) -> Optional[str]:
    qp = _get_qp()
    v = qp.get(name)
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        v = v[0] if v else None
    return v


# -------------------------
# IMU Capture Page (Top-level) HTML template
# -------------------------
IMU_CAPTURE_HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>IMU Capture</title>
  <style>
    body { font-family: system-ui, sans-serif; padding: 16px; line-height: 1.3; }
    button { padding: 12px 14px; border-radius: 10px; border: 1px solid #ccc; background: #fff; width: 100%; margin: 8px 0; }
    .card { padding: 12px; border: 1px solid #ddd; border-radius: 12px; margin-top: 12px; }
    code { font-family: ui-monospace, Menlo, monospace; word-break: break-all; }
    small { color: #555; }
  </style>
</head>
<body>
  <h2>IMU Capture (Top-level page)</h2>
  <small>
    This page must be opened directly (top-level), not inside an iframe.
    It reads motion sensors and redirects back to your Streamlit app with beta/gamma in the URL.
  </small>

  <div class="card">
    <div><b>Return URL</b></div>
    <code id="ret">(missing ?return=...)</code>
  </div>

  <button id="btnEnable">Enable motion sensors</button>
  <button id="btnSend" disabled>Send to Streamlit</button>

  <div class="card">
    <div>Status: <span id="status">idle</span></div>
    <div><code id="vals">alpha: -, beta: -, gamma: -</code></div>
  </div>

<script>
  const statusEl = document.getElementById("status");
  const valsEl = document.getElementById("vals");
  const btnEnable = document.getElementById("btnEnable");
  const btnSend = document.getElementById("btnSend");
  const retEl = document.getElementById("ret");

  const params = new URLSearchParams(location.search);
  const returnUrl = params.get("return") || "";
  retEl.textContent = returnUrl || "(missing ?return=...)";

  let last = {alpha:null, beta:null, gamma:null};

  function setStatus(s){ statusEl.textContent = s; }

  function onOri(e){
    last = {alpha:e.alpha, beta:e.beta, gamma:e.gamma};
    valsEl.textContent =
      `alpha: ${last.alpha?.toFixed?.(1) ?? "-"}, beta: ${last.beta?.toFixed?.(1) ?? "-"}, gamma: ${last.gamma?.toFixed?.(1) ?? "-"}`;
    btnSend.disabled = !(returnUrl && last.beta != null && last.gamma != null);
  }

  async function enableSensors(){
    try {
      // iOS requires explicit permission request; Android usually doesn't.
      if (typeof DeviceOrientationEvent !== "undefined" &&
          typeof DeviceOrientationEvent.requestPermission === "function") {
        const resp = await DeviceOrientationEvent.requestPermission();
        if (resp !== "granted") { setStatus("permission denied"); return; }
      }
      window.addEventListener("deviceorientation", onOri, true);
      setStatus("listening");
    } catch (e) {
      setStatus("error: " + e);
    }
  }

  function sendBack(){
    if (!returnUrl) return;
    const u = new URL(returnUrl);
    u.searchParams.set("beta", String(last.beta));
    u.searchParams.set("gamma", String(last.gamma));
    u.searchParams.set("alpha", String(last.alpha));
    u.searchParams.set("imu", "1");
    location.href = u.toString();
  }

  btnEnable.addEventListener("click", enableSensors);
  btnSend.addEventListener("click", sendBack);
</script>
</body>
</html>
""".strip()


# -------------------------
# Streamlit App UI
# -------------------------
st.set_page_config(page_title="Phone Measure PoC (IMU via Redirect)", layout="wide")
st.title("Phone Measure PoC — YOLO + FoV + Height + IMU (via top-level redirect)")

st.info(
    "This PoC avoids IMU-in-iframe issues. IMU is collected from a top-level IMU capture page "
    "and sent back to this app via URL query params. This works far more reliably on mobile."
)

# Sidebar controls
with st.sidebar:
    st.header("Measurement Settings")
    model_name = st.selectbox("YOLO model", ["yolov8n.pt", "yolov8s.pt"], index=0)
    conf_thres = st.slider("YOLO confidence", 0.05, 0.9, 0.25, 0.05)
    iou_thres = st.slider("YOLO IoU (NMS)", 0.1, 0.9, 0.45, 0.05)

    st.divider()
    hfov = st.slider("Horizontal FoV (deg)", 40, 120, 70, 1)
    cam_h_cm = st.slider("Camera height above plane (cm)", 10, 200, 40, 1)
    cam_h_m = cam_h_cm / 100.0

    st.divider()
    st.header("IMU Setup")
    st.caption("Preferred: set these once for your deployment.")
    app_url = st.text_input(
        "Streamlit app URL (HTTPS)",
        value=st.secrets.get("APP_URL", ""),
        placeholder="https://<your-app>.streamlit.app"
    )
    imu_base_url = st.text_input(
        "IMU capture page URL (HTTPS)",
        value=st.secrets.get("IMU_PAGE_URL", ""),
        placeholder="https://<your-username>.github.io/<repo>/imu/"
    )

    st.caption(
        "Best practice: host IMU page on GitHub Pages at /docs/imu/index.html "
        "and set IMU_PAGE_URL to the /imu/ URL."
    )

# Session state for calibration
if "imu_zero" not in st.session_state:
    st.session_state.imu_zero = None

# Read IMU values from query params
imu_beta = qp_float("beta")
imu_gamma = qp_float("gamma")
imu_alpha = qp_float("alpha")
imu_present = (imu_beta is not None and imu_gamma is not None)

# Axis tweaks
colA, colB, colC = st.columns([1, 1, 1])
with colA:
    st.subheader("IMU (from URL)")
    st.write({"beta": imu_beta, "gamma": imu_gamma, "alpha": imu_alpha, "imu_flag": qp_str("imu")})

with colB:
    inv_pitch = st.checkbox("Invert pitch", value=False)
    inv_roll = st.checkbox("Invert roll", value=False)

with colC:
    use_manual = st.checkbox("Use manual tilt sliders (fallback)", value=not imu_present)

# Buttons: open IMU page + calibrate
btn1, btn2, btn3 = st.columns([1, 1, 2])
with btn1:
    if imu_base_url and app_url:
        imu_link = f"{imu_base_url.rstrip('/')}/?return={urllib.parse.quote(app_url, safe='')}"
        st.link_button("Open IMU Capture Page", imu_link)
    else:
        st.warning("Set both APP_URL and IMU_PAGE_URL in sidebar (or Streamlit Secrets).")

with btn2:
    if st.button("Calibrate (use current IMU as zero)"):
        if not imu_present:
            st.error("No IMU in URL yet. Use 'Open IMU Capture Page' first.")
        else:
            st.session_state.imu_zero = {"beta": imu_beta, "gamma": imu_gamma}
            st.success("Calibration stored.")

with btn3:
    with st.expander("IMU Capture Page HTML (copy/paste into GitHub Pages)", expanded=False):
        st.code(IMU_CAPTURE_HTML, language="html")
        st.markdown(
            "**GitHub Pages quick setup:** put the HTML into `docs/imu/index.html`, enable Pages for `/docs`, "
            "then set `IMU_PAGE_URL` to `https://<user>.github.io/<repo>/imu/`."
        )

# Decide pitch/roll (either IMU deltas or manual sliders)
pitch_deg: Optional[float] = None
roll_deg: Optional[float] = None

if use_manual:
    st.subheader("Manual tilt (fallback)")
    pitch_deg = st.slider("Pitch delta (deg)", -45.0, 45.0, 0.0, 0.5)
    roll_deg = st.slider("Roll delta (deg)", -45.0, 45.0, 0.0, 0.5)
else:
    pitch_deg = imu_beta
    roll_deg = imu_gamma
    if st.session_state.imu_zero:
        pitch_deg = pitch_deg - st.session_state.imu_zero["beta"]
        roll_deg = roll_deg - st.session_state.imu_zero["gamma"]

# Apply inversions
if pitch_deg is not None and inv_pitch:
    pitch_deg = -pitch_deg
if roll_deg is not None and inv_roll:
    roll_deg = -roll_deg

pitch_rad = math.radians(pitch_deg) if pitch_deg is not None else None
roll_rad = math.radians(roll_deg) if roll_deg is not None else None

st.caption("Tilt used by geometry (after calibration & inversion):")
st.write({"pitch_deg": pitch_deg, "roll_deg": roll_deg, "pitch_rad": pitch_rad, "roll_rad": roll_rad})

st.divider()

# Capture image
st.subheader("Capture image (phone camera)")
img_file = st.camera_input("Take a photo of an object lying on a plane (table/bed/floor).")

if not img_file:
    st.stop()

# Decode image
file_bytes = np.asarray(bytearray(img_file.getvalue()), dtype=np.uint8)
bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
H, W = rgb.shape[:2]

# Run YOLO
model = load_yolo(model_name)
results = model.predict(rgb, conf=conf_thres, iou=iou_thres, verbose=False)
det = pick_best_det(results)

if det is None:
    st.error("No objects detected. Improve lighting or choose a clearer target.")
    st.stop()

xyxy = det["xyxy"]
conf = det["conf"]
name = det["name"]

# Visualize bbox
vis = rgb.copy()
x1, y1, x2, y2 = map(int, xyxy)
cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.putText(vis, f"{name} {conf:.2f}", (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

st.image(vis, caption="Detection bbox used for measurement", use_container_width=True)

# Measure (needs pitch/roll)
if pitch_rad is None or roll_rad is None:
    st.warning("No tilt available. Use manual sliders or run IMU capture flow.")
    st.stop()

est = estimate_length_width_from_bbox(
    xyxy=xyxy,
    img_w=W,
    img_h=H,
    hfov_deg=hfov,
    camera_height_m=cam_h_m,
    pitch_rad=pitch_rad,
    roll_rad=roll_rad,
)

if est is None:
    st.error("Geometry failed (ray/plane intersection). Try: reduce tilt, recalibrate, or adjust camera height.")
    st.stop()

length_m, width_m = est

c1, c2, c3 = st.columns(3)
c1.metric("Estimated length", f"{length_m*100:.1f} cm")
c2.metric("Estimated width", f"{width_m*100:.1f} cm")
c3.metric("YOLO confidence", f"{conf:.2f}")

st.caption(
    "PoC note: accuracy is dominated by camera-height error and bbox tightness. "
    "For a stronger demo: keep phone steady, re-calibrate if pose changes, and keep object near frame center."
)
