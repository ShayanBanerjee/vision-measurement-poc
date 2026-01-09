import math
import uuid
import streamlit as st
from streamlit_js_eval import streamlit_js_eval  # from streamlit-js-eval

def _imu_js_once(timeout_ms: int = 1200) -> str:
    # Returns a Promise that resolves to {alpha,beta,gamma} or null
    return f"""
new Promise((resolve) => {{
  let done = false;
  const finish = (v) => {{ if (!done) {{ done = true; resolve(v); }} }};

  function handler(e) {{
    window.removeEventListener('deviceorientation', handler);
    finish({{alpha: e.alpha, beta: e.beta, gamma: e.gamma}});
  }}

  window.addEventListener('deviceorientation', handler, {{ once: true }});
  setTimeout(() => finish(null), {timeout_ms});
}})
"""

def read_imu_sample(timeout_ms: int = 1200):
    # New key every time forces the JS snippet to execute again.
    return streamlit_js_eval(
        js_expressions=_imu_js_once(timeout_ms),
        key=f"IMU_{uuid.uuid4().hex}",
    )

def deg2rad(x):
    return None if x is None else (x * math.pi / 180.0)

# --- IMU UI ---
st.subheader("IMU (PoC)")

# In Streamlit, keep component calls "stable": do not hide them in deep branches.
# We'll drive refresh using a counter + rerun.
if "imu_refresh" not in st.session_state:
    st.session_state.imu_refresh = 0
if "imu_zero" not in st.session_state:
    st.session_state.imu_zero = None
if "imu_latest" not in st.session_state:
    st.session_state.imu_latest = None

colA, colB = st.columns(2)
with colA:
    if st.button("Enable / Refresh IMU"):
        st.session_state.imu_refresh += 1
        st.rerun()
with colB:
    if st.button("Calibrate (hold phone steady)"):
        # Use the most recent sample as "zero"
        if st.session_state.imu_latest:
            st.session_state.imu_zero = st.session_state.imu_latest
        else:
            st.warning("No IMU sample available yet. Tap 'Enable / Refresh IMU' first.")

# Always execute the JS eval (key changes when refresh counter changes)
imu = read_imu_sample(timeout_ms=1200)
st.session_state.imu_latest = imu

st.caption("Raw IMU sample (degrees):")
st.write(imu)

# Fallback manual sliders if IMU returns null (common if sensors blocked)
if not imu:
    st.error("IMU not available in browser. Using manual pitch/roll sliders (demo fallback).")
    pitch_deg = st.slider("Manual pitch (deg)", -90.0, 90.0, 35.0, 0.5)
    roll_deg  = st.slider("Manual roll (deg)",  -90.0, 90.0, 0.0, 0.5)
else:
    pitch_deg = imu.get("beta")
    roll_deg  = imu.get("gamma")

# Apply calibration (subtract offsets)
if st.session_state.imu_zero and imu:
    pitch_deg = (pitch_deg - st.session_state.imu_zero.get("beta")) if pitch_deg is not None else None
    roll_deg  = (roll_deg  - st.session_state.imu_zero.get("gamma")) if roll_deg is not None else None

pitch_rad = deg2rad(pitch_deg)
roll_rad  = deg2rad(roll_deg)

st.caption("Pose used by geometry (after calibration):")
st.write({"pitch_deg": pitch_deg, "roll_deg": roll_deg, "pitch_rad": pitch_rad, "roll_rad": roll_rad})
