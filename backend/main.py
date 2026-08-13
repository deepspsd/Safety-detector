from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from database import create_tables, ensure_enterprise_schema
from config import settings
from services.yolo_service import load_model
from routers import auth, users, alerts, detection, video, faces, cctv, cameras
from routers.settings import router as settings_router, baseline_router
from routers.enterprise import router as enterprise_router
from routers.attendance import router as attendance_router
from routers.documents import router as documents_router
from routers.workflow import router as workflow_router

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
app.include_router(attendance_router)  # GET/POST /attendance/*
app.include_router(documents_router)   # GET/POST /documents/*
app.include_router(workflow_router)    # GET /workflow/*

# ── /api prefix aggregate router (production single-origin compatibility) ──────
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
_api_router.include_router(attendance_router)
_api_router.include_router(documents_router)
_api_router.include_router(workflow_router)
app.include_router(_api_router)

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")


@app.get("/")
def root():
    return {"message": "Safety Monitor API", "version": "1.0.0", "status": "running"}


@app.get("/health")
def health():
    return {"status": "ok"}


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

    # ── Auto-migrate v4: new tables (attendance, lift events, ocr logs) ─────────
    _v4_migrations = [
        ("attendance_records",
         "CREATE TABLE IF NOT EXISTS attendance_records "
         "(id INTEGER PRIMARY KEY, employee_id INTEGER REFERENCES employees(id), "
         "user_id INTEGER REFERENCES users(id), camera_id INTEGER REFERENCES cameras(id), "
         "clock_in DATETIME NOT NULL, clock_out DATETIME, duration_seconds FLOAT, "
         "method VARCHAR(20) DEFAULT 'manual', notes VARCHAR(500))"),
        ("lift_events",
         "CREATE TABLE IF NOT EXISTS lift_events "
         "(id INTEGER PRIMARY KEY, camera_id INTEGER NOT NULL REFERENCES cameras(id), "
         "track_id INTEGER NOT NULL, employee_id INTEGER REFERENCES employees(id), "
         "event_type VARCHAR(30) NOT NULL, floor_from VARCHAR(20), floor_to VARCHAR(20), "
         "duration_sec FLOAT, timestamp DATETIME)"),
        ("invoice_logs",
         "CREATE TABLE IF NOT EXISTS invoice_logs "
         "(id INTEGER PRIMARY KEY, camera_id INTEGER REFERENCES cameras(id), "
         "employee_id INTEGER REFERENCES employees(id), direction VARCHAR(10) DEFAULT 'inward', "
         "raw_ocr_text TEXT, approved BOOLEAN NOT NULL DEFAULT 0, "
         "snapshot_b64 TEXT, ocr_available BOOLEAN NOT NULL DEFAULT 1, timestamp DATETIME NOT NULL)"),
        ("order_form_logs",
         "CREATE TABLE IF NOT EXISTS order_form_logs "
         "(id INTEGER PRIMARY KEY, camera_id INTEGER REFERENCES cameras(id), "
         "employee_id INTEGER REFERENCES employees(id), direction VARCHAR(10) DEFAULT 'outward', "
         "raw_ocr_text TEXT, approved BOOLEAN NOT NULL DEFAULT 0, "
         "snapshot_b64 TEXT, ocr_available BOOLEAN NOT NULL DEFAULT 1, timestamp DATETIME NOT NULL)"),
    ]
    try:
        from sqlalchemy import text
        from database import engine
        with engine.connect() as conn:
            for tname, sql in _v4_migrations:
                try:
                    conn.execute(text(sql))
                    conn.commit()
                    print(f"✅ DB migration v4: table '{tname}' ensured")
                except Exception as te:
                    if "already exists" in str(te).lower():
                        pass
                    else:
                        print(f"⚠️  Migration v4 [{tname}]: {te}")
    except Exception as e:
        print(f"⚠️  Migration v4 block error: {e}")

    # ── Start server-managed camera streams ───────────────────────────────────
    # Reads all Camera rows with status != 'offline' from the DB and
    # starts one background reader thread per camera.
    # WebSocket endpoints subscribe to these shared streams via camera_manager.
    # v5 camera registration schema. The tables are additive and camera fields
    # are migrated one-by-one to keep existing SQLite deployments intact.
    _v5_tables = [
        "CREATE TABLE IF NOT EXISTS camera_credentials (camera_id INTEGER PRIMARY KEY REFERENCES cameras(id), encrypted_username TEXT NOT NULL, encrypted_password TEXT NOT NULL, created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL)",
        "CREATE TABLE IF NOT EXISTS camera_streams (id INTEGER PRIMARY KEY, camera_id INTEGER NOT NULL REFERENCES cameras(id), profile_token VARCHAR(200), stream_type VARCHAR(20) NOT NULL, codec VARCHAR(40), width INTEGER, height INTEGER, fps FLOAT, encrypted_rtsp_uri TEXT NOT NULL, active BOOLEAN NOT NULL DEFAULT 1, created_at DATETIME NOT NULL)",
        "CREATE TABLE IF NOT EXISTS camera_health (id INTEGER PRIMARY KEY, camera_id INTEGER NOT NULL REFERENCES cameras(id), status VARCHAR(40) NOT NULL, fps FLOAT, bitrate_kbps FLOAT, latency_ms FLOAT, packet_loss FLOAT, last_frame_at DATETIME, reconnect_count INTEGER NOT NULL DEFAULT 0, last_error TEXT, created_at DATETIME NOT NULL)",
    ]
    _v5_columns = [
        "ALTER TABLE cameras ADD COLUMN manufacturer VARCHAR(120)",
        "ALTER TABLE cameras ADD COLUMN model VARCHAR(160)",
        "ALTER TABLE cameras ADD COLUMN ip_address VARCHAR(64)",
        "ALTER TABLE cameras ADD COLUMN onvif_endpoint VARCHAR(500)",
        "ALTER TABLE cameras ADD COLUMN discovery_id VARCHAR(100)",
        "ALTER TABLE cameras ADD COLUMN preferred_stream VARCHAR(20) NOT NULL DEFAULT 'sub'",
        "ALTER TABLE cameras ADD COLUMN ai_stream VARCHAR(20) NOT NULL DEFAULT 'sub'",
    ]
    try:
        from sqlalchemy import text
        from database import engine
        with engine.connect() as conn:
            for sql in _v5_tables:
                conn.execute(text(sql)); conn.commit()
            for sql in _v5_columns:
                try:
                    conn.execute(text(sql)); conn.commit()
                except Exception as migration_error:
                    if "duplicate column" not in str(migration_error).lower() and "already exists" not in str(migration_error).lower():
                        print(f"Camera v5 migration warning: {migration_error}")
    except Exception as migration_error:
        print(f"Camera v5 migration block warning: {migration_error}")

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
