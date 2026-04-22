from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from database import create_tables
from config import settings
from services.yolo_service import load_model
from routers import auth, users, alerts, detection, video, faces, cctv

app = FastAPI(
    title="Safety Monitor API",
    description="Real-time AI Safety Monitoring System",
    version="1.0.0"
)

# CORS — allow browser on any device (localhost + LAN IP)
# The "*" wildcard covers all origins so mobile browsers on the same
# WiFi network can reach the API when VITE_API_BASE_URL is set to the LAN IP.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # wildcard enables mobile browser access
    allow_credentials=False,  # must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(alerts.router)
app.include_router(detection.router)
app.include_router(video.router)
app.include_router(faces.router)
app.include_router(cctv.router)

# Serve uploaded files
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")


@app.on_event("startup")
def startup():
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    os.makedirs(os.path.join(settings.UPLOAD_DIR, "snapshots"), exist_ok=True)
    create_tables()

    # ── Auto-migrate: add thumbnail_b64 to face_encodings if missing ──────────
    try:
        from sqlalchemy import text
        from database import engine
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE face_encodings ADD COLUMN thumbnail_b64 TEXT"))
            conn.commit()
        print("✅ DB migration: thumbnail_b64 column added")
    except Exception as e:
        if "duplicate column" in str(e).lower() or "already exists" in str(e).lower():
            pass  # column already there — all good
        else:
            print(f"⚠️  Migration warning: {e}")

    load_model()
    print("✅ Safety Monitor API v3.0 started")


@app.get("/")
def root():
    return {"message": "Safety Monitor API", "version": "1.0.0", "status": "running"}


@app.get("/health")
def health():
    return {"status": "ok"}
