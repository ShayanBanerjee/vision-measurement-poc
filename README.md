# Phone Measure PoC (Streamlit + YOLO + IMU Tilt)

This repository contains a **simple Proof of Concept (PoC)** that estimates an object’s **planar length and width** using:
- **YOLO** object detection (Ultralytics YOLOv8) to obtain a bounding box
- Approximate camera model from **Horizontal FoV (HFoV)**
- User-provided **camera height above a plane**
- Phone **IMU tilt** (DeviceOrientation: beta/gamma) via a tiny Streamlit component
- Geometry: **pixels → rays → plane intersection → centimeters**

## What this PoC is (and is not)

### Assumptions (required)
1. The target object lies on a **single plane** (table/bed/floor).
2. You enter an approximate **camera height above that plane** (e.g., 40 cm) and keep it reasonably stable.
3. You tap **Calibrate** once while holding the phone steady in your measurement pose.

### Expected accuracy (rule of thumb)
- Best-case demo conditions: ~8–15%
- Typical handheld demo: ~15–30%
- Poor conditions (wrong height, strong tilt drift, object not on plane): 30%+

Major error sources: camera height estimate, IMU tilt noise, YOLO bounding-box tightness, and FoV mismatch.

### Disclaimer
This is a **demo PoC** to validate feasibility and workflow. It is not validated for clinical measurement accuracy.

---

## Repository structure

```
.
├── app_min_imu.py
├── geometry_min.py
├── requirements.txt
└── imu_component/
    ├── __init__.py
    └── frontend/
        └── index.html
```

- `app_min_imu.py`: Streamlit app using `st.camera_input` + IMU + YOLO + measurement  
- `geometry_min.py`: FoV intrinsics, ray construction, plane intersection, length/width estimate  
- `imu_component/`: Tiny Streamlit custom component that reads `DeviceOrientationEvent`

---

## A) Run locally (laptop)

### 1) Create a virtual environment and install dependencies

#### Windows (PowerShell)
```powershell
cd <your_project_folder>
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

#### macOS / Linux
```bash
cd vision-measurement-poc
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Start the app
```bash
streamlit run app_min_imu.py
```

Open on laptop:
- http://localhost:8501

Note: IMU values won’t be meaningful on desktop. The real test is on mobile over **HTTPS**.

---

## B) Push to GitHub

### 1) Create a new repository on GitHub
Create a new repo (suggested names at the bottom of this README).

### 2) Initialize git and push
From the project root:
```bash
git init
git add .
git commit -m "Initial PoC: Streamlit + YOLO + IMU tilt measurement"
git branch -M main
git remote add origin https://github.com/ShayanBanerjee/vision-measurement-poc.git
git push -u origin main
```

---

## C) Deploy on Streamlit Community Cloud (recommended: HTTPS for mobile)

### Why Streamlit Cloud
Phone camera and motion sensors typically require **HTTPS**. Streamlit Community Cloud provides an HTTPS URL for your app.

### Deploy steps
1. Go to Streamlit Community Cloud and choose **Create app**
2. Select:
   - Repository: your GitHub repo
   - Branch: `main`
   - Main file path: `app_min_imu.py`
3. Deploy

Once deployed, you’ll get a URL like:
- `https://<your-app>.streamlit.app`

---

## D) Mobile usage (Camera + IMU)

### Recommended browsers
- iPhone: **Safari**
- Android: **Chrome**

### Step-by-step on phone
1. Open the **HTTPS** Streamlit Cloud URL on your phone.
2. In the IMU box, tap **Enable motion sensors** and allow permissions (iOS requires this explicit step).
3. Allow **Camera** permission when prompted.
4. Place the object on a **flat plane** (table/bed).
5. Hold the phone steady at your chosen height and tap **Calibrate**.
6. Set:
   - **Horizontal FoV (deg)** (start with 65–75° for many 1× phone cameras; adjust if you see consistent bias)
   - **Camera height above plane (cm)** (e.g., 30–50 cm)
7. Take a photo and read **Estimated length/width**.

### If measurements look “flipped”
Use the axis toggles in the app:
- **Invert pitch**
- **Invert roll**

This is normal in PoCs because IMU axis conventions vary across devices and orientations.

---

## E) Troubleshooting

### IMU stays null / no beta & gamma
- Confirm you are using the **HTTPS** URL (Streamlit Cloud is HTTPS).
- iPhone: use **Safari** (not an in-app browser).
- Tap **Enable motion sensors** again and accept permissions.
- If you denied permission previously, clear site permissions / website data for that domain and retry.

### Camera does not open
- Must be **HTTPS** (or localhost).
- Allow camera permissions in the browser settings for that site.
- Avoid in-app browsers; open in Safari/Chrome directly.

### YOLO is slow
- First run may download model weights.
- Use `yolov8n.pt` (smallest) for PoC speed.
- Improve lighting and keep the object clear in frame.

### Results are unstable / far off
- Ensure object is actually on a single plane.
- Re-calibrate after changing height or posture.
- Keep the object near the **center** of the image.
- Avoid ultra-wide lens / digital zoom.
- Keep camera height as constant as possible during capture.

---

## F) Next improvements (after PoC)
1. Replace bbox with **segmentation** (YOLO-seg) for tighter measurement.
2. Remove manual camera-height input via **ARKit/ARCore depth**.
3. Use device-specific intrinsics and lens distortion correction.
4. Add UX guidance (stability indicator, re-calibration prompts, quality scoring).

---

## Suggested repository / project names

**Short + clear**
- `phone-measure-poc`
- `mobile-measurement-poc`
- `smart-measure-poc`
- `camera-measure-poc`

**Medical-context friendly (still general)**
- `clinical-measurement-poc`
- `mobile-clinical-metrics`
- `care-measure-poc`

**Brandable (neutral)**
- `PocketMeasure`
- `PlaneMeasure`
- `MedMeasureLite`
