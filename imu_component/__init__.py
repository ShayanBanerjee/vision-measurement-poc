# imu_component/__init__.py
import os
import streamlit.components.v1 as components

_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
_imu = components.declare_component("imu_component", path=_FRONTEND_DIR)

def imu_widget(key="imu", poll_hz=15):
    """
    Returns a dict like:
      {alpha, beta, gamma, ts, granted, ua}
    or None if not ready.
    """
    return _imu(key=key, poll_hz=int(poll_hz))
