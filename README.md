# Phone Measure PoC (Streamlit + YOLO + IMU via Redirect)

This repository demonstrates a **reliable mobile PoC** for estimating an object's **planar length & width** using:
- **YOLO (Ultralytics YOLOv8)** to detect an object (bounding box),
- Camera model derived from **Horizontal FoV (HFoV)**,
- User-provided **camera height above a plane** (cm),
- Phone **IMU tilt** (beta/gamma) captured from a **top-level web page** (GitHub Pages) and returned to Streamlit via URL query parameters,
- Geometry: **pixels → rays → plane intersection → centimeters**.

## Why the IMU is NOT read inside Streamlit

On many mobile browsers and WebViews, **motion/orientation APIs are blocked or unreliable inside iframes**.  
Streamlit components and HTML embeds run in iframes, which is why you may see sensor access fail.

**Best-practice PoC workaround:** collect IMU in a **top-level page** (not iframe), then redirect back to Streamlit with `beta/gamma` in the URL.

This approach is:
- simple to explain,
- reliable on Android/iOS (over HTTPS),
- easy to improve later (ARCore/ARKit depth, segmentation, etc.).

---

## Expected accuracy (PoC guidance)

- Best-case demo conditions: **~8–15%**
- Typical handheld demo: **~15–30%**
- Poor conditions (wrong height / strong tilt drift / object not on plane): **30%+**

Main error sources: camera height estimate, bbox tightness, tilt stability, and FoV mismatch.

---

## Repository structure (current)

```
phone-measure-poc/
├─ app_min_imu.py
├─ requirements.txt
├─ .gitignore
└─ docs/
   └─ imu/
      └─ index.html
```

### What each file does
- `app_min_imu.py`  
  Streamlit app: camera capture (`st.camera_input`), YOLO detection, planar measurement, IMU calibration.  
  **Reads IMU from query params** (e.g., `?beta=...&gamma=...`) returned by the IMU capture page.

- `docs/imu/index.html`  
  IMU capture page served via **GitHub Pages** (top-level). It requests motion permission (if needed), reads `DeviceOrientationEvent`, and **redirects back** to Streamlit with IMU values in URL.

---

## A) Run locally (laptop)

### 1) Setup virtual environment
#### Windows (PowerShell)
```powershell
cd phone-measure-poc
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

#### macOS / Linux
```bash
cd phone-measure-poc
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run Streamlit
```bash
streamlit run app_min_imu.py
```

Open:
- http://localhost:8501

**Note:** IMU will not work from localhost on your phone unless you are using an HTTPS tunnel. The recommended demo path is Streamlit Cloud + GitHub Pages.

---

## B) Push to GitHub

### 1) Create a GitHub repo
Suggested names:
- `phone-measure-poc`
- `camera-size-estimator-poc`
- `mobile-measurement-poc`

### 2) Push code
```bash
git init
git add .
git commit -m "Initial PoC: Streamlit + YOLO + IMU redirect"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

---

## C) Enable GitHub Pages (for IMU capture page)

This serves the IMU capture page as HTTPS:

### Steps
1. GitHub repo → **Settings** → **Pages**
2. Source: **Deploy from a branch**
3. Branch: `main`
4. Folder: `/docs`
5. Save

Your IMU page will be:
```
https://<your-username>.github.io/<your-repo>/imu/
```

---

## D) Deploy Streamlit app on Streamlit Community Cloud (HTTPS)

### Steps
1. Go to Streamlit Community Cloud → **Create app**
2. Choose your GitHub repo and branch `main`
3. Main file path: **`app_min_imu.py`**
4. Deploy

You will get:
```
https://<your-app>.streamlit.app
```

---

## E) Configure Streamlit Secrets (recommended)

In Streamlit Community Cloud:
- App → **Settings** → **Secrets**
Add:

```toml
APP_URL = "https://<your-app>.streamlit.app"
IMU_PAGE_URL = "https://<your-username>.github.io/<your-repo>/imu"
```

This lets the app generate a correct "Open IMU Capture Page" link automatically.

---

## F) Mobile demo flow (Android / iPhone)

### Recommended browser
- Android: **Chrome**
- iPhone: **Safari**

### Steps
1. Open the Streamlit app (HTTPS):  
   `https://<your-app>.streamlit.app`
2. Tap **Open IMU Capture Page** (it opens GitHub Pages top-level).
3. Tap **Enable motion sensors** and allow permissions (if prompted).
4. Tap **Send to Streamlit** (redirects back with `beta/gamma` in the URL).
5. In Streamlit, tap **Calibrate** (stores current beta/gamma as baseline).
6. Capture an image in Streamlit and read **Estimated length/width**.

### If tilt looks inverted
Use the app toggles:
- **Invert pitch**
- **Invert roll**

Axis conventions vary by device/orientation; this is normal for PoC.

---

## G) Troubleshooting

### “No IMU values in URL”
- Ensure GitHub Pages is enabled and the IMU page opens successfully.
- Ensure the IMU page has the correct return URL (Streamlit app URL).
- On iPhone, use Safari and grant motion permission explicitly.

### Camera does not open
- Must be HTTPS (Streamlit Cloud is HTTPS).
- Allow camera permissions for the site.
- Avoid in-app browsers; open in Chrome/Safari directly.

### Measurements are unstable / far off
- Object must be on a single plane.
- Re-calibrate if pose/height changes.
- Keep object near image center.
- Use consistent camera height (cm).

---

## Future upgrades (post-PoC)
1. Use **segmentation** instead of bbox (YOLO-seg) for tighter measurement.
2. Replace manual camera height with **ARCore/ARKit depth**.
3. Add device intrinsics/distortion correction.
4. Add stability/quality indicator and guided capture UX.
