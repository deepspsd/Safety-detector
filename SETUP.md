# 🛠️ OccuSafe — Full Setup Guide

Pull this repo and have the app running in under 10 minutes.

---

## Prerequisites

| Tool | Min. Version | Install |
|---|---|---|
| Python | 3.11+ | https://python.org/downloads |
| Node.js | 18+ | https://nodejs.org |
| npm | 9+ | bundled with Node |
| Git | any | https://git-scm.com |

> **GPU (optional):** If you have an NVIDIA GPU, install PyTorch with CUDA support for much faster inference. See https://pytorch.org/get-started/locally/

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/rohith-dasari-1/safety-detection-tool.git
cd safety-detection-tool
```

---

## Step 2 — Backend Setup

### 2a. Create a virtual environment

```bash
cd backend
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2b. Install Python dependencies

```bash
pip install -r requirements.txt
```

> This installs FastAPI, uvicorn, SQLAlchemy, ultralytics (YOLOv8), OpenCV, and all other dependencies.

### 2c. Download the AI model (`ppe.pt`)

The custom PPE detection model is too large for Git. Run the automatic downloader:

```bash
python download_ppe_model.py
```

This downloads `ppe.pt` (~84 MB) from Google Drive into the `backend/` folder.

> **If the download fails:** Download manually from the link printed by the script and place `ppe.pt` in the `backend/` folder.

### 2d. Configure environment

```bash
copy .env.example .env           # Windows
# OR
cp .env.example .env             # macOS / Linux
```

Edit `.env` if needed (the defaults work for local development):

```
SECRET_KEY=change-this-to-a-long-random-string
DATABASE_URL=sqlite:///./safety.db
MIN_VIOLATION_CONF=0.40
ALERT_COOLDOWN=15
```

### 2e. Start the backend

```bash
uvicorn main:app --reload --port 8000
```

You should see:
```
✅ Custom ppe.pt model loaded successfully
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## Step 3 — Frontend Setup

Open a **new terminal** window.

```bash
cd safety-detection-tool/frontend
```

### 3a. Configure environment

```bash
copy .env.example .env           # Windows
# OR
cp .env.example .env             # macOS / Linux
```

Default `.env` contents (works for local dev):
```
VITE_API_BASE_URL=http://localhost:8000
```

### 3b. Install Node dependencies

```bash
npm install
```

### 3c. Start the frontend dev server

```bash
npm run dev
```

You should see:
```
  VITE v6.x  ready in 500ms
  ➜  Local:   http://localhost:5173/
  ➜  Network: http://192.168.X.X:5173/
```

---

## Step 4 — Open the App

Navigate to **http://localhost:5173** in your browser.

1. Click **Sign Up** and create an account (any email + password)
2. You'll be taken to the **Live Monitor** page automatically
3. Click **Start Webcam** to begin real-time PPE detection
4. Toggle the PPE filter chips (Hardhat, Vest, Mask…) to control what is monitored

---

## Accessing from Phone / Other Devices on Same Network

The dev server exposes on all network interfaces. Find your PC's local IP:

```bash
ipconfig           # Windows — look for IPv4 Address
# OR
ip addr            # Linux/macOS
```

Then open `http://<YOUR_IP>:5173` on your phone.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `uvicorn: command not found` | Activate the venv first: `venv\Scripts\activate` |
| `ppe.pt not found` | Run `python download_ppe_model.py` from inside `backend/` |
| Port 8000 blocked (network devices) | Ensure Windows Firewall allows inbound on port 8000, OR use the Vite proxy (already configured — API calls go through port 5173 automatically) |
| Webcam not starting | Allow camera permissions in your browser |
| `WebSocket failed` error | Make sure both uvicorn (8000) and Vite (5173) are running |
| Face recognition slow/missing | Install `face-recognition` separately: `pip install face-recognition` (requires cmake + dlib) |

---

## Production Build

To build the frontend for production:

```bash
cd frontend
npm run build
```

Static files will be in `frontend/dist/`. Serve them with your preferred web server (nginx, Apache) pointing the backend API to your deployed FastAPI instance.

---

## Project Commands Reference

| Command | Directory | Description |
|---|---|---|
| `uvicorn main:app --reload --port 8000` | `backend/` | Start backend dev server |
| `npm run dev` | `frontend/` | Start frontend dev server |
| `npm run build` | `frontend/` | Build production bundle |
| `python download_ppe_model.py` | `backend/` | Download AI model weights |
