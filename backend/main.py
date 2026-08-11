from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from database import create_tables, ensure_enterprise_schema
from config import settings
from services.yolo_service import load_model
from routers import auth, users, alerts, detection, video, faces, cctv, cameras
from routers.settings import router as settings_router, baseline_router
from routers.enterprise import router as enterprise_router

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
app.include_router(enterprise_router)
app.include_router(settings_router)   # GET/PUT /settings/*
app.include_router(baseline_router)   # POST/DELETE /cameras/{id}/baseline

# ── /api prefix aggregate router (production single-origin compatibility) ─────
# The frontend (api.js) defaults baseURL to '/api' in development, which Vite
# proxies to the backend (stripping /api).  In production single-origin mode
# (FastAPI serves the built frontend), there is no Vite proxy, so API calls
# with an '/api' prefix would 404.  This aggregate router makes the backend
# respond to BOTH /api/... and /... so a single frontend build works for both
# development and production single-origin deployments.
from fastapi import APIRouter as _APIRouter
_api_router = _APIRouter(prefix="/api")
_api_router.include_router(auth.router)
_api_router.include_router(users.router)
_api_router.include_router(alerts.router)
_api_router.include_router(video.router)
_api_router.include_router(faces.router)
_api_router.include_router(cameras.router)
_api_router.include_router(enterprise_router)
_api_router.include_router(settings_router)
_api_router.include_router(baseline_router)
# NOTE: WebSocket routes (/ws/*) are intentionally NOT included here —
# the frontend connects to them directly (no /api prefix).
app.include_router(_api_router)

# Serve uploaded files
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")

# ── Serve built frontend in production ───────────────────────────────────────
# After `npm run build` the Vite output lands in frontend/dist/.
# FastAPI serves it at / so there is no need for a separate Nginx process on
# the Windows server.  In development the Vite dev server handles this instead.
_FRONTEND_DIST = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.isdir(_FRONTEND_DIST):
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles as _SF

    # Serve JS/CSS/assets
    app.mount("/assets", _SF(directory=os.path.join(_FRONTEND_DIST, "assets"), html=False), name="frontend-assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_frontend(full_path: str):
        """
        Catch-all: serve index.html for every non-API path so React Router
        client-side navigation works after a page refresh.
        Falls back to the file if it exists (e.g. favicon.ico, manifest).
        """
        file_path = os.path.join(_FRONTEND_DIST, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(_FRONTEND_DIST, "index.html"))


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
    ensure_enterprise_schema()

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

    # ── Auto-migrate v3: Alert.status column (confidence-tier routing) ────────
    try:
        from sqlalchemy import text
        from database import engine
        with engine.connect() as conn:
            try:
                conn.execute(text(
                    "ALTER TABLE alerts ADD COLUMN status TEXT NOT NULL DEFAULT 'confirmed'"
                ))
                conn.commit()
                print("✅ DB migration v3: alerts.status column added")
            except Exception as col_err:
                col_msg = str(col_err).lower()
                if "duplicate column" in col_msg or "already exists" in col_msg:
                    pass  # already migrated
                else:
                    print(f"⚠️  Migration v3 [alerts.status]: {col_err}")
    except Exception as e:
        print(f"⚠️  Migration v3 block error: {e}")

    load_model()

    # The registry is the stable contract for current and future vision models.
    # Importing these modules binds the event -> rule -> alert pipeline once.
    try:
        from database import SessionLocal
        from services import model_manager
        _mdb = SessionLocal()
        try:
            model_manager.seed_registry(_mdb)
        finally:
            _mdb.close()
        from services import rule_engine_v2, alert_engine_v2, notification_engine  # noqa: F401
        # Wire camera health events (offline / drift) → DB alerts + Telegram
        from services import camera_alert_handler
        camera_alert_handler.register()
        print("Enterprise event, rule, alert and model services ready")
    except Exception as e:
        print(f"Enterprise platform startup warning: {e}")

    # ── Seed SystemSettings defaults ──────────────────────────────────────────
    try:
        from database import SessionLocal
        from services import rule_engine
        _sdb = SessionLocal()
        try:
            rule_engine.seed_defaults(_sdb)
            print("✅ SystemSettings defaults seeded")
        finally:
            _sdb.close()
    except Exception as e:
        print(f"⚠️  SystemSettings seed warning: {e}")

    # ── Auto-migrate new tables (safe if already exist) ───────────────────────
    _v3_migrations = [
        ("system_settings",
         "CREATE TABLE IF NOT EXISTS system_settings "
         "(id INTEGER PRIMARY KEY, key TEXT UNIQUE NOT NULL, "
         "value TEXT NOT NULL, description TEXT, updated_at DATETIME)"),
        ("dirty_floor_baselines",
         "CREATE TABLE IF NOT EXISTS dirty_floor_baselines "
         "(id INTEGER PRIMARY KEY, camera_id INTEGER REFERENCES cameras(id), "
         "zone_name TEXT NOT NULL, image_path TEXT NOT NULL, uploaded_at DATETIME)"),
    ]
    try:
        from sqlalchemy import text
        from database import engine
        with engine.connect() as conn:
            for tname, sql in _v3_migrations:
                try:
                    conn.execute(text(sql))
                    conn.commit()
                    print(f"✅ DB migration v3: table '{tname}' ensured")
                except Exception as te:
                    if "already exists" in str(te).lower():
                        pass
                    else:
                        print(f"⚠️  Migration v3 [{tname}]: {te}")
    except Exception as e:
        print(f"⚠️  Migration v3 block error: {e}")

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
