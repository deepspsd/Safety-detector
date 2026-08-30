# CCTV AI Monitoring Platform (On-Premise)

Production-grade on-premise CCTV video analytics and automated rule-engine system designed for bakery, factory, and store environments.

The platform performs RTSP camera management, object detection, persistent ByteTrack tracking, person cropping, Teachable Machine / ONNX attribute classification, spatial polygon gating, line-crossing detection, configurable state machines, evidence snapshot capture, push notifications, and live dashboard streaming.

---

## 1. Architecture Overview

```
RTSP Cameras
    │
    ▼
Camera Reader (Threaded RTSP / Reconnect)
    │
    ▼
Frame Sampling & Inference Pool (Shared YOLO Detector)
    │
    ▼
ByteTrack Tracker (Persistent Track IDs & Trajectory)
    │
    ├────────► Zone & Virtual Line Engine (ENTER/EXIT, INWARD/OUTWARD)
    │
    ▼
Person Cropping Engine (Full Person / Upper Body / Head / Custom)
    │
    ▼
Teachable Machine / ONNX Classifier Adapter (with Temporal Smoothing & Caching)
    │
    ▼
Rule Engine State Machines (Idle, Absence, Shift-Start, Camera-Standing, Dirty Floor)
    │
    ▼
Platform Event Bus (Pub/Sub: RULE_MATCHED)
    │
    ├────────► Evidence Storage (JPEG Snapshot)
    ├────────► Database (SQLite / Postgres)
    ├────────► Notification Engine (Telegram, ntfy.sh, Dashboard)
    │
    ▼
FastAPI REST + WebSockets ◄──► React TypeScript Dashboard
```

---

## 2. Installation & Quick Start

### Local Development (Windows / Linux / macOS)

1. **Prerequisites:** Python 3.11+, Node.js 18+, Tesseract OCR.
2. **Backend Setup:**
   ```bash
   cd backend
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate

   pip install -r requirements.txt
   ```
3. **Frontend Setup:**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
4. **Start Backend:**
   ```bash
   cd backend
   uvicorn main:app --reload --port 8000
   ```

---

## 3. Docker Setup (On-Premise Server)

Run the entire system in a single command using Docker Compose:

```bash
docker compose up -d --build
```

The application will be available at:
- Web Dashboard: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`

---

## 4. Environment Variables (`.env`)

Create `.env` in `backend/` or project root:

```ini
APP_ENV=development
SECRET_KEY=change-this-super-secret-key-for-production
DATABASE_URL=sqlite:///./safety.db

# YOLO Detection
YOLO_MODEL=ppe_factory_v0_cash.pt
DETECTION_CONF=0.30

# Mock Mode (runs without cameras/models)
MOCK_MODE=false

# Push Notifications
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
NTFY_TOPIC=bakery-alerts
NTFY_SERVER=https://ntfy.sh
```

---

## 5. How to Register Cameras

Cameras can be registered via:
1. **API / UI:** Navigate to `/cameras` on dashboard and enter RTSP URL or ONVIF discovery.
2. **Config file:** Edit `config/cameras.yaml`:
   ```yaml
   cameras:
     - camera_id: shop_01
       name: "Shop Front Counter"
       rtsp_url: "rtsp://admin:pass@192.168.1.101:554/Streaming/Channels/102"
       floor: "shop"
       zone_type: "shop_counter"
       enabled: true
   ```

---

## 6. How to Register Models

Models are defined in `config/models.yaml`:
- **Object Detectors:** YOLO models (`.pt` or `.onnx`) placed in `backend/`.
- **Image Classifiers:** Teachable Machine models placed in `backend/models/<model_id>.onnx` or `backend/models/<model_id>/`.

---

## 7. How to Add Teachable Machine Classifiers

1. Train a model on [Google Teachable Machine](https://teachablemachine.withgoogle.com/) (Image Project).
2. Export as **TensorFlow SavedModel** or **ONNX**.
3. Place model files in `backend/models/<model_id>/` or `backend/models/<model_id>.onnx`.
4. Register the model in `config/models.yaml`.
5. The system automatically loads it via `TeachableMachineAdapter` with zero code changes.

---

## 8. How Person Cropping Works

When a `person` is detected by YOLO:
1. Bounding box coordinates are clamped to image boundaries.
2. Configurable padding (e.g. 10%) is applied.
3. Cropper extracts ROI based on configured `crop_mode`:
   - `full_person` (entire box)
   - `upper_body` (top 65% of box)
   - `head` (top 30% of box)
   - `custom` (arbitrary slice)
4. Crop is resized to the classifier input size (e.g. 224×224).
5. Crop is dispatched to the classifier adapter.

---

## 9. Defining Polygon Zones & Line Crossing

### Polygon Zones (`config/zones.yaml` or Dashboard Zone Painter)
```yaml
zones:
  shop_01:
    - name: "shop_counter"
      polygon: [[50, 100], [250, 100], [250, 300], [50, 300]]
    - name: "cashbox"
      polygon: [[180, 150], [240, 150], [240, 220], [180, 220]]
```

### Virtual Lines
```yaml
lines:
  ground_entrance:
    - line_id: "main_entry"
      p1: [50, 350]
      p2: [350, 350]
      inward_direction: "down"
      outward_direction: "up"
```

---

## 10. Rule Engine & Supported Rules

1. **Shift Start Check:** Alerts if no person/activity is detected on a floor by scheduled start time (e.g. 08:00 AM).
2. **Employee Idle:** Tracks person trajectory; alerts if stationary beyond 5 minutes.
3. **Shop Absence:** State machine alerts if shop counter zone is unattended for > 60 seconds.
4. **Camera Blocking:** Alerts if person remains standing directly in front of camera lens for > 60 seconds.
5. **Dirty Floor:** Non-ML baseline image difference detection for contamination/spills.
6. **Cash Monitoring:** Tracks hand/cash movements around cashbox zones.
7. **Attribute Violations:** Missing uniform, hairnet/head-cap, or jewellery detection.

---

## 11. Evidence Storage

When an alert triggers:
- Frame snapshot is saved to `backend/uploads/evidence/cam_<id>/<date>/alert_<id>.jpg`.
- File path is linked to the alert case in the database.
- Dashboard displays snapshot directly on the Alerts Review page.

---

## 12. Mock Mode (Testing Without Cameras/Models)

Enable mock mode in `.env`:
```ini
MOCK_MODE=true
```
In mock mode:
- Synthetic frames with smooth moving simulated persons are generated.
- Random simulated PPE/attribute classifications are emitted.
- All rules, state machines, and dashboard endpoints can be tested without hardware.

---

## 13. Running Unit Tests

Run the test suite using Python:
```bash
cd backend
python -c "
from tests.test_pipeline import *
tc = TestPersonCropper()
tc.test_crop_modes_and_clamping()
tc.test_invalid_bbox_returns_none()
tclf = TestClassifierAdapterAndSmoothing()
tclf.test_mock_classifier_predictions()
tclf.test_classifier_cache_rate_limiting()
tclf.test_temporal_smoother_majority_voting()
ttrk = TestTrackingLayer()
ttrk.test_track_observation_history()
tzn = TestZoneEngine()
tzn.test_point_in_polygon_zone()
tzn.test_line_crossing()
trsm = TestRuleStateMachines()
trsm.test_idle_rule_state_machine()
trsm.test_absence_rule_state_machine()
trsm.test_mock_detector_generator()
print('ALL TESTS PASSED')
"
```

---

## 14. API Endpoints Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Real liveness check (inference pool, cameras, DB) |
| `GET` | `/dashboard/summary` | Top-level KPI summary |
| `GET` | `/tracks` | Live tracking snapshot across all cameras |
| `GET` | `/tracks/{camera_id}` | Tracks for specific camera |
| `GET` | `/cameras` | List configured cameras & health status |
| `GET` | `/alerts` | Paginated alert cases |
| `GET` | `/platform/models` | List registered AI models & capabilities |
| `WS` | `/ws/detect-cctv` | Real-time CCTV detection stream |