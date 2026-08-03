# 🦺 OccuSafe — AI-Powered PPE Safety Detection Platform

> Real-time Personal Protective Equipment (PPE) detection using YOLOv8, FastAPI, and React.

![OccuSafe](https://img.shields.io/badge/AI-YOLOv8-blue?style=flat-square) ![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat-square) ![React](https://img.shields.io/badge/Frontend-React%2019-61DAFB?style=flat-square) ![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## ✨ Features

- 🎥 **Live Webcam Detection** — real-time PPE violation analysis via WebSocket
- 📹 **Video Upload Processing** — scan recorded footage for violations
- 🎛️ **Custom PPE Filters** — toggle which equipment types to monitor (Hardhat, Vest, Mask, Gloves, Goggles, Safety Shoes)
- 👤 **Role-Based Rules** — separate PPE requirements for Construction Worker, Doctor, Traffic Police, College, and Home roles
- 📍 **Location-Based Role Detection** — auto-set your role using GPS + OpenStreetMap
- 🔔 **Real-Time Alerts** — browser notifications + sound when violations are detected
- 📊 **Alert History Dashboard** — full log of all past violations with snapshots
- 🌙 **Dark / Light Mode** — adaptive theme

---

## 🏗️ Tech Stack

| Layer | Technology |
|---|---|
| AI Model | YOLOv8 (`ultralytics`) — custom `ppe.pt` |
| Backend | Python 3.11 · FastAPI · SQLite · SQLAlchemy |
| Real-Time | WebSocket (FastAPI) · `asyncio.gather` concurrent pipeline |
| Frontend | React 19 · Vite · Vanilla CSS |
| Auth | JWT (`python-jose`) |

---

## 🚀 Quick Start

> See **[SETUP.md](./SETUP.md)** for the full step-by-step guide.

```bash
# 1. Clone
git clone https://github.com/deepspsd/safety-detector.git
cd safety-detector

# 2. Backend
cd backend
python -m venv venv && venv\Scripts\activate      # Windows
pip install -r requirements.txt
python download_ppe_model.py                       # downloads ppe.pt
copy .env.example .env
uvicorn main:app --reload --port 8000

# 3. Frontend (new terminal)
cd frontend
copy .env.example .env
npm install
npm run dev
```

Open **http://localhost:5173** — sign up with any email to get started.

---

## 📁 Project Structure

```
safety-detection-tool/
├── backend/
│   ├── main.py                  # FastAPI app entry point
│   ├── database.py              # SQLAlchemy models & DB setup
│   ├── config.py                # Environment config (pydantic-settings)
│   ├── auth_utils.py            # JWT helpers
│   ├── requirements.txt
│   ├── download_ppe_model.py    # Auto-downloads ppe.pt from Google Drive
│   ├── routers/
│   │   ├── detection.py         # WebSocket PPE detection (concurrent pipeline)
│   │   ├── video.py             # Video upload & background job processing
│   │   ├── auth.py              # Login / signup endpoints
│   │   └── users.py             # Profile & config endpoints
│   └── services/
│       ├── yolo_service.py      # YOLOv8 inference pipeline + filter logic
│       ├── face_service.py      # Face recognition (Home role)
│       └── alert_service.py     # Alert persistence & snapshot saving
└── frontend/
    ├── index.html
    ├── vite.config.js
    └── src/
        ├── main.jsx
        ├── App.jsx
        ├── api/api.js           # Axios client (proxied via Vite in dev)
        ├── context/             # Auth, Toast, Theme context providers
        └── pages/
            ├── LiveMonitor.jsx  # Webcam / upload detection UI
            ├── Dashboard.jsx    # Alerts overview
            ├── Alerts.jsx       # Full alert log
            ├── Settings.jsx     # Camera, profile, location-based role
            └── Profile.jsx
```

---

## 🤖 AI Model

The system uses a **custom YOLOv8 model** (`ppe.pt`) trained specifically for PPE detection. The model is **not included in this repo** due to file size (~84 MB). Run the downloader:

```bash
cd backend
python download_ppe_model.py
```

**Detected classes:** `Person`, `Hardhat`, `NO-Hardhat`, `Mask`, `NO-Mask`, `Safety Vest`, `NO-Safety Vest`, `Safety Cone`, `machinery`, `vehicle`

> **Simulation mode:** If `ppe.pt` is missing, the system auto-activates a realistic simulation mode that generates synthetic violations — useful for UI testing without a GPU.

---

## 🌐 Accessing from Mobile / Other Devices

The app works on your local network. The Vite dev server exposes itself on `0.0.0.0`. Open:

```
http://<YOUR_PC_IP>:5173
```

Find your IP: `ipconfig` → IPv4 Address (e.g., `192.168.1.100`).

---

## 📄 License

MIT — feel free to use, modify, and distribute.