# app_min_imu.py
import math
import numpy as np
import streamlit as st
import cv2

from imu_component import imu_widget
from geometry_min import estimate_length_width_from_bbox

@st.cache_resource
def load_model():
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")

def best_bbox(results):
    r0 = results[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return None
    conf = r0.boxes.conf.detach().cpu().numpy()
    xyxy = r0.boxes.xyxy.detach().cpu().numpy()
    cls  = r0.boxes.cls.detach().cpu().numpy().astype(int)
    i = int(np.argmax(conf))
    return xyxy[i], float(conf[i]), int(cls[i]), r0.names[int(cls[i])]

st.set_page_config(page_title="Simple Phone Measure PoC (IMU)", layout="wide")
st.title("Simple Phone Measure PoC (YOLO + FoV + Height + IMU Tilt)")

st.write(
    "PoC assumption: object lies on a flat plane (table/bed/floor). "
    "No reference object. Height + tilt give metric scale."
)

# ---- Session state for calibration ----
if "calib" not in st.session_state:
    st.session_state.calib = {"beta0": None, "gamma0": None}

# ---- Controls ----
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    hfov = st.slider("Horizontal FoV (deg)", 40, 120, 70, 1)
    cam_h_cm = st.slider("Camera height above plane (cm)", 10, 200, 40, 1)
    cam_h_m = cam_h_cm / 100.0

with col2:
    st.subheader("IMU")
    imu = imu_widget(key="imu", poll_hz=15)

    granted = bool(imu and imu.get("granted"))
    beta = imu.get("beta") if imu else None
    gamma = imu.get("gamma") if imu else None

    st.write(f"Permission: {'OK' if granted else 'Not granted yet'}")
    st.write(f"beta (pitch proxy): {beta}")
    st.write(f"gamma (roll proxy): {gamma}")

    if st.button("Calibrate (hold phone steady)"):
        if beta is None or gamma is None:
            st.error("No IMU readings yet. Tap 'Enable motion sensors' in the IMU box first.")
        else:
            st.session_state.calib["beta0"] = float(beta)
            st.session_state.calib["gamma0"] = float(gamma)
            st.success("Calibration stored.")

with col3:
    st.subheader("Axis tweaks (PoC)")
    inv_pitch = st.checkbox("Invert pitch", value=False)
    inv_roll  = st.checkbox("Invert roll", value=False)
    use_sliders_fallback = st.checkbox("Use manual sliders (fallback)", value=False)

# ---- Compute pitch/roll (either IMU or sliders) ----
if use_sliders_fallback:
    pitch_deg = st.slider("Manual pitch delta (deg)", -30.0, 30.0, 0.0, 0.5)
    roll_deg  = st.slider("Manual roll delta (deg)",  -30.0, 30.0, 0.0, 0.5)
else:
    # IMU-based
    beta0 = st.session_state.calib.get("beta0")
    gamma0 = st.session_state.calib.get("gamma0")

    if beta is None or gamma is None:
        pitch_deg, roll_deg = 0.0, 0.0
    else:
        # If not calibrated, treat current pose as baseline (keeps things usable)
        if beta0 is None: beta0 = float(beta)
        if gamma0 is None: gamma0 = float(gamma)

        pitch_deg = float(beta) - float(beta0)
        roll_deg  = float(gamma) - float(gamma0)

    if inv_pitch: pitch_deg = -pitch_deg
    if inv_roll:  roll_deg  = -roll_deg

pitch = math.radians(pitch_deg)
roll  = math.radians(roll_deg)

st.caption(f"Using pitch_delta={pitch_deg:.2f}°, roll_delta={roll_deg:.2f}°")

st.divider()

# ---- Capture image ----
img_file = st.camera_input("Capture an image")
if not img_file:
    st.info("Take a photo of an object on a plane. For best results: steady phone + calibrated pose.")
    st.stop()

# Decode image
file_bytes = np.asarray(bytearray(img_file.getvalue()), dtype=np.uint8)
bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
H, W = rgb.shape[:2]

# YOLO detect
model = load_model()
results = model.predict(rgb, conf=0.25, iou=0.45, verbose=False)
picked = best_bbox(results)

if picked is None:
    st.error("No objects detected. Try better lighting or a clearer object.")
    st.stop()

xyxy, conf, cls_id, name = picked
x1, y1, x2, y2 = map(float, xyxy)

# Visualize bbox
vis = rgb.copy()
cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
cv2.putText(vis, f"{name} {conf:.2f}", (int(x1), max(0,int(y1)-8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
st.image(vis, caption="Detection bbox used for measurement", use_container_width=True)

# Geometry estimate
est = estimate_length_width_from_bbox(
    x1, y1, x2, y2, W, H,
    hfov_deg=hfov,
    camera_height_m=cam_h_m,
    pitch_rad=pitch,
    roll_rad=roll
)

if est is None:
    st.error("Geometry failed (ray/plane intersection). Reduce tilt, recalibrate, or adjust height.")
    st.stop()

length_m, width_m, _ptsXZ = est

c1, c2, c3 = st.columns(3)
c1.metric("Estimated length", f"{length_m*100:.1f} cm")
c2.metric("Estimated width",  f"{width_m*100:.1f} cm")
c3.metric("Detection confidence", f"{conf:.2f}")

st.caption(
    "PoC accuracy depends mainly on camera height and stable tilt. "
    "Next steps (later): segmentation instead of bbox + ARKit/ARCore depth to remove height input."
)
