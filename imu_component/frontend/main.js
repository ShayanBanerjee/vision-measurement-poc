// imu_component/frontend/main.js
const btn = document.getElementById("btn");
const statusEl = document.getElementById("status");
const valsEl = document.getElementById("vals");

let granted = false;
let started = false;
let pollHz = 15;

let last = { alpha: null, beta: null, gamma: null, ts: null };

function setStatus(s) {
  statusEl.textContent = "Status: " + s;
}

function pushValue() {
  if (!window.Streamlit) return;

  window.Streamlit.setComponentValue({
    ...last,
    granted: granted,
    ua: navigator.userAgent
  });
  window.Streamlit.setFrameHeight(110);
}

function handleOrientation(ev) {
  last = { alpha: ev.alpha, beta: ev.beta, gamma: ev.gamma, ts: Date.now() };
  valsEl.textContent =
    "alpha: " + (last.alpha?.toFixed?.(1) ?? "-") +
    ", beta: " + (last.beta?.toFixed?.(1) ?? "-") +
    ", gamma: " + (last.gamma?.toFixed?.(1) ?? "-");
}

async function enable() {
  try {
    if (started) return;
    started = true;

    // iOS permission gate; Android usually returns true
    if (typeof DeviceOrientationEvent !== "undefined" &&
        typeof DeviceOrientationEvent.requestPermission === "function") {
      const resp = await DeviceOrientationEvent.requestPermission();
      granted = (resp === "granted");
    } else {
      granted = true;
    }

    if (!granted) {
      setStatus("permission denied");
      pushValue();
      return;
    }

    window.addEventListener("deviceorientation", handleOrientation, true);
    setStatus("listening");

    setInterval(pushValue, Math.max(50, Math.floor(1000 / pollHz)));
  } catch (e) {
    setStatus("error: " + e);
    pushValue();
  }
}

btn.addEventListener("click", enable);

function onRender(event) {
  const args = event.detail.args || {};
  if (args.poll_hz) pollHz = args.poll_hz;
  pushValue();
}

window.Streamlit.events.addEventListener(window.Streamlit.RENDER_EVENT, onRender);
window.Streamlit.setComponentReady();
window.Streamlit.setFrameHeight(110);
