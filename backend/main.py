from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from database import create_tables
from config import settings
from services.yolo_service import load_model
from routers import auth, users, alerts, detection, video, faces, cctv, cameras

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
app.include_router(cameras.router)

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

    # ── Auto-migrate v2: extend alerts table for multi-camera factory schema ───
    # Adds three nullable columns introduced in database.py v2.0.
    # Each ALTER is wrapped individually so a single missing column doesn't
    # abort the others.  Errors other than "already exists" are surfaced.
    _v2_alert_migrations = [
        ("camera_id",   "ALTER TABLE alerts ADD COLUMN camera_id   INTEGER REFERENCES cameras(id)"),
        ("floor",       "ALTER TABLE alerts ADD COLUMN floor        TEXT"),
        ("employee_id", "ALTER TABLE alerts ADD COLUMN employee_id  INTEGER REFERENCES employees(id)"),
    ]
    try:
        from sqlalchemy import text
        from database import engine
        with engine.connect() as conn:
            for col_name, sql in _v2_alert_migrations:
                try:
                    conn.execute(text(sql))
                    conn.commit()
                    print(f"✅ DB migration v2: alerts.{col_name} column added")
                except Exception as col_err:
                    col_msg = str(col_err).lower()
                    if "duplicate column" in col_msg or "already exists" in col_msg:
                        pass  # already migrated — skip silently
                    else:
                        print(f"⚠️  Migration v2 warning [{col_name}]: {col_err}")
    except Exception as e:
        print(f"⚠️  Migration v2 block error: {e}")

    load_model()

    # ── Start server-managed camera streams ───────────────────────────────────
    # Reads all Camera rows with status != 'offline' from the DB and
    # starts one background reader thread per camera.
    # WebSocket endpoints subscribe to these shared streams via camera_manager.
    from services import camera_manager
    _started = camera_manager.start_all()
    print(f"✅ Camera manager: {_started} camera(s) started")

    print("✅ Safety Monitor API v4.0 started (factory monitoring schema)")


@app.on_event("shutdown")
def shutdown():
    """Gracefully stop all camera reader threads on server shutdown."""
    from services import camera_manager
    camera_manager.stop_all()
    print("🛑 Safety Monitor: all camera readers stopped")


@app.get("/")
def root():
    return {"message": "Safety Monitor API", "version": "1.0.0", "status": "running"}


@app.get("/health")
def health():
    return {"status": "ok"}
