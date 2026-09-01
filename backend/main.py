import logging
import os

from config import settings
from database import create_tables, ensure_enterprise_schema
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from routers import alerts, auth, cameras, cctv, detection, faces, users, video
from routers.alarm import router as alarm_router
from routers.attendance import router as attendance_router
from routers.dashboard import router as dashboard_router
from routers.documents import router as documents_router
from routers.enterprise import router as enterprise_router
from routers.settings import baseline_router
from routers.settings import router as settings_router
from routers.tracks import router as tracks_router
from routers.workflow import router as workflow_router
from services.yolo_service import load_model

log = logging.getLogger("main")

app = FastAPI(
    title="Safety Monitor API",
    description="Real-time AI Safety Monitoring System",
    version="1.0.0",
)

# ── CORS configuration ──────────────────────────────────────────────────────
# In development (APP_ENV=development or ALLOWED_ORIGINS unset) all origins
# are permitted so Vite on localhost and mobile browsers on the LAN work
# without extra configuration.
#
# In production set: ALLOWED_ORIGINS=http://192.168.1.100:5173,http://192.168.1.100
# — only those origins will be accepted; requests from others are rejected.
_allowed_origins_raw = settings.ALLOWED_ORIGINS.strip()
if _allowed_origins_raw:
    _cors_origins = [o.strip() for o in _allowed_origins_raw.split(",") if o.strip()]
    _allow_credentials = True  # safe with an explicit origin list
else:
    _cors_origins = ["*"]
    _allow_credentials = False  # must be False when origin is "*"

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_allow_credentials,
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
app.include_router(settings_router)  # GET/PUT /settings/*
app.include_router(baseline_router)  # POST/DELETE /cameras/{id}/baseline
app.include_router(attendance_router)  # GET/POST /attendance/*
app.include_router(documents_router)  # GET/POST /documents/*
app.include_router(workflow_router)  # GET /workflow/*
app.include_router(alarm_router)  # POST /alarm/*
app.include_router(tracks_router)  # GET /tracks/*
app.include_router(dashboard_router)  # GET /dashboard/*

# ── /api prefix aggregate router (production single-origin compatibility) ──────
from fastapi import APIRouter as _APIRouter

_api_router = _APIRouter(prefix="/api")
_api_router.include_router(auth.router)
_api_router.include_router(users.router)
_api_router.include_router(alerts.router)
_api_router.include_router(detection.router)
_api_router.include_router(video.router)
_api_router.include_router(faces.router)
_api_router.include_router(cctv.router)
_api_router.include_router(cameras.router)
_api_router.include_router(enterprise_router)
_api_router.include_router(settings_router)
_api_router.include_router(baseline_router)
_api_router.include_router(attendance_router)
_api_router.include_router(documents_router)
_api_router.include_router(workflow_router)
_api_router.include_router(alarm_router)
_api_router.include_router(tracks_router)
_api_router.include_router(dashboard_router)
app.include_router(_api_router)

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")


@app.get("/")
def root():
    return {"message": "Safety Monitor API", "version": "1.0.0", "status": "running"}


@app.get("/health")
def health():
    """
    Real liveness check — verifies all critical runtime components.

    Returns HTTP 200 when all components are healthy.
    Returns HTTP 503 when any critical component is down so monitoring tools,
    uptime bots, systemd watchdogs, and Nginx can detect real failures.

    Components checked
    ------------------
    inference_pool : YOLO worker thread alive (detection pipeline operational)
    cameras        : number of server-managed camera readers running
    database       : SQLite connectivity (simple SELECT 1)
    """
    from database import engine
    from services import camera_manager
    from services.inference_pool import inference_pool
    from sqlalchemy import text as _text

    issues: list[str] = []

    # ── 1. Inference pool ────────────────────────────────────────────────────
    pool_healthy = inference_pool.is_healthy()
    if not pool_healthy:
        issues.append("inference_pool: worker thread not running")

    # ── 2. Camera manager ────────────────────────────────────────────────────
    cam_statuses = camera_manager.list_status()
    active_cams = sum(1 for c in cam_statuses if c.get("status") == "online")

    # ── 3. Database connectivity ─────────────────────────────────────────────
    db_ok = True
    try:
        with engine.connect() as conn:
            conn.execute(_text("SELECT 1"))
    except Exception as db_exc:
        db_ok = False
        issues.append(f"database: {db_exc}")

    # ── Response ─────────────────────────────────────────────────────────────
    payload = {
        "status": "ok" if not issues else "degraded",
        "inference_pool": {
            "healthy": pool_healthy,
            "restart_count": inference_pool.restart_count,
        },
        "cameras": {
            "total": len(cam_statuses),
            "online": active_cams,
        },
        "database": {
            "ok": db_ok,
        },
    }
    if issues:
        payload["issues"] = issues
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=503, content=payload)

    return payload


# ── Serve built frontend in production ───────────────────────────────────────
# After `npm run build` the Vite output lands in frontend/dist/.
# FastAPI serves it at / so there is no need for a separate Nginx process on
# the Windows server.  In development the Vite dev server handles this instead.
_FRONTEND_DIST = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.isdir(_FRONTEND_DIST):
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles as _SF

    # Serve JS/CSS/assets
    app.mount(
        "/assets",
        _SF(directory=os.path.join(_FRONTEND_DIST, "assets"), html=False),
        name="frontend-assets",
    )

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


_DEFAULT_SECRET = "safety-monitor-super-secret-key-2024-change-in-prod"


def _run_migrations() -> None:
    """Run all incremental SQLite schema migrations in version order."""
    from database import engine
    from sqlalchemy import text

    # v1: face_encodings thumbnail
    _migrate_columns(
        engine,
        [
            (
                "face_encodings",
                "thumbnail_b64",
                "ALTER TABLE face_encodings ADD COLUMN thumbnail_b64 TEXT",
            ),
        ],
    )

    # v2: alert factory columns
    _migrate_columns(
        engine,
        [
            (
                "alerts",
                "camera_id",
                "ALTER TABLE alerts ADD COLUMN camera_id   INTEGER REFERENCES cameras(id)",
            ),
            ("alerts", "floor", "ALTER TABLE alerts ADD COLUMN floor        TEXT"),
            (
                "alerts",
                "employee_id",
                "ALTER TABLE alerts ADD COLUMN employee_id  INTEGER REFERENCES employees(id)",
            ),
        ],
    )

    # v3: alert status + core tables
    _migrate_columns(
        engine,
        [
            (
                "alerts",
                "status",
                "ALTER TABLE alerts ADD COLUMN status TEXT NOT NULL DEFAULT 'confirmed'",
            ),
        ],
    )
    _migrate_tables(
        engine,
        [
            (
                "system_settings",
                "CREATE TABLE IF NOT EXISTS system_settings "
                "(id INTEGER PRIMARY KEY, key TEXT UNIQUE NOT NULL, "
                "value TEXT NOT NULL, description TEXT, updated_at DATETIME)",
            ),
            (
                "dirty_floor_baselines",
                "CREATE TABLE IF NOT EXISTS dirty_floor_baselines "
                "(id INTEGER PRIMARY KEY, camera_id INTEGER REFERENCES cameras(id), "
                "zone_name TEXT NOT NULL, image_path TEXT NOT NULL, uploaded_at DATETIME)",
            ),
        ],
    )

    # v4: attendance + lift + OCR log tables
    _migrate_tables(
        engine,
        [
            (
                "attendance_records",
                "CREATE TABLE IF NOT EXISTS attendance_records "
                "(id INTEGER PRIMARY KEY, employee_id INTEGER REFERENCES employees(id), "
                "user_id INTEGER REFERENCES users(id), camera_id INTEGER REFERENCES cameras(id), "
                "clock_in DATETIME NOT NULL, clock_out DATETIME, duration_seconds FLOAT, "
                "method VARCHAR(20) DEFAULT 'manual', notes VARCHAR(500))",
            ),
            (
                "lift_events",
                "CREATE TABLE IF NOT EXISTS lift_events "
                "(id INTEGER PRIMARY KEY, camera_id INTEGER NOT NULL REFERENCES cameras(id), "
                "track_id INTEGER NOT NULL, employee_id INTEGER REFERENCES employees(id), "
                "event_type VARCHAR(30) NOT NULL, floor_from VARCHAR(20), floor_to VARCHAR(20), "
                "duration_sec FLOAT, timestamp DATETIME)",
            ),
            (
                "invoice_logs",
                "CREATE TABLE IF NOT EXISTS invoice_logs "
                "(id INTEGER PRIMARY KEY, camera_id INTEGER REFERENCES cameras(id), "
                "employee_id INTEGER REFERENCES employees(id), direction VARCHAR(10) DEFAULT 'inward', "
                "raw_ocr_text TEXT, goods_count INTEGER, approved BOOLEAN NOT NULL DEFAULT 0, "
                "snapshot_b64 TEXT, ocr_available BOOLEAN NOT NULL DEFAULT 1, timestamp DATETIME NOT NULL)",
            ),
            (
                "order_form_logs",
                "CREATE TABLE IF NOT EXISTS order_form_logs "
                "(id INTEGER PRIMARY KEY, camera_id INTEGER REFERENCES cameras(id), "
                "employee_id INTEGER REFERENCES employees(id), direction VARCHAR(10) DEFAULT 'outward', "
                "raw_ocr_text TEXT, approved BOOLEAN NOT NULL DEFAULT 0, "
                "snapshot_b64 TEXT, person_snapshot_b64 TEXT, "
                "ocr_available BOOLEAN NOT NULL DEFAULT 1, timestamp DATETIME NOT NULL)",
            ),
        ],
    )
    # goods_count + person_snapshot on existing tables (safe on fresh DBs too)
    _migrate_columns(
        engine,
        [
            (
                "invoice_logs",
                "goods_count",
                "ALTER TABLE invoice_logs    ADD COLUMN goods_count         INTEGER",
            ),
            (
                "order_form_logs",
                "person_snapshot_b64",
                "ALTER TABLE order_form_logs ADD COLUMN person_snapshot_b64 TEXT",
            ),
        ],
    )

    # v5: camera credential / health tables + column extensions
    _migrate_tables(
        engine,
        [
            "CREATE TABLE IF NOT EXISTS camera_credentials (camera_id INTEGER PRIMARY KEY REFERENCES cameras(id), encrypted_username TEXT NOT NULL, encrypted_password TEXT NOT NULL, created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL)",
            "CREATE TABLE IF NOT EXISTS camera_streams (id INTEGER PRIMARY KEY, camera_id INTEGER NOT NULL REFERENCES cameras(id), profile_token VARCHAR(200), stream_type VARCHAR(20) NOT NULL, codec VARCHAR(40), width INTEGER, height INTEGER, fps FLOAT, encrypted_rtsp_uri TEXT NOT NULL, active BOOLEAN NOT NULL DEFAULT 1, created_at DATETIME NOT NULL)",
            "CREATE TABLE IF NOT EXISTS camera_health (id INTEGER PRIMARY KEY, camera_id INTEGER NOT NULL REFERENCES cameras(id), status VARCHAR(40) NOT NULL, fps FLOAT, bitrate_kbps FLOAT, latency_ms FLOAT, packet_loss FLOAT, last_frame_at DATETIME, reconnect_count INTEGER NOT NULL DEFAULT 0, last_error TEXT, created_at DATETIME NOT NULL)",
        ],
        name_from_sql=True,
    )
    _migrate_columns(
        engine,
        [
            (
                "cameras",
                "manufacturer",
                "ALTER TABLE cameras ADD COLUMN manufacturer VARCHAR(120)",
            ),
            ("cameras", "model", "ALTER TABLE cameras ADD COLUMN model VARCHAR(160)"),
            (
                "cameras",
                "ip_address",
                "ALTER TABLE cameras ADD COLUMN ip_address VARCHAR(64)",
            ),
            (
                "cameras",
                "onvif_endpoint",
                "ALTER TABLE cameras ADD COLUMN onvif_endpoint VARCHAR(500)",
            ),
            (
                "cameras",
                "discovery_id",
                "ALTER TABLE cameras ADD COLUMN discovery_id VARCHAR(100)",
            ),
            (
                "cameras",
                "preferred_stream",
                "ALTER TABLE cameras ADD COLUMN preferred_stream VARCHAR(20) NOT NULL DEFAULT 'sub'",
            ),
            (
                "cameras",
                "ai_stream",
                "ALTER TABLE cameras ADD COLUMN ai_stream VARCHAR(20) NOT NULL DEFAULT 'sub'",
            ),
        ],
    )


def _migrate_tables(engine, statements, *, name_from_sql: bool = False) -> None:
    """Execute CREATE TABLE IF NOT EXISTS statements, swallowing 'already exists'."""
    from sqlalchemy import text

    with engine.connect() as conn:
        for item in statements:
            sql = item if isinstance(item, str) else item[-1]
            label = item if isinstance(item, str) else item[0]
            try:
                conn.execute(text(sql))
                conn.commit()
            except Exception as err:
                if "already exists" not in str(err).lower():
                    log.warning("Migration table warning [%s]: %s", label, err)


def _migrate_columns(engine, specs) -> None:
    """Execute ALTER TABLE ADD COLUMN statements, swallowing 'duplicate column' / 'already exists'."""
    from sqlalchemy import text

    with engine.connect() as conn:
        for table, col, sql in specs:
            try:
                conn.execute(text(sql))
                conn.commit()
            except Exception as err:
                msg = str(err).lower()
                if "duplicate column" not in msg and "already exists" not in msg:
                    log.warning("Migration column warning [%s.%s]: %s", table, col, err)


@app.on_event("startup")
def startup():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Production guard ──────────────────────────────────────────────────────
    if settings.APP_ENV == "production" and settings.SECRET_KEY == _DEFAULT_SECRET:
        raise RuntimeError(
            "\n\n"
            "SECURITY ERROR: Running in production with the default SECRET_KEY.\n"
            "Set SECRET_KEY=<random 64-char string> in your .env file before starting.\n"
        )
    if settings.SECRET_KEY == _DEFAULT_SECRET:
        log.warning(
            "[Security] Using default SECRET_KEY — acceptable in development only. "
            "Set APP_ENV=production and a strong SECRET_KEY before going live."
        )
    if not _allowed_origins_raw:
        log.warning(
            "[Security] ALLOWED_ORIGINS is not set — all CORS origins permitted. "
            "Set ALLOWED_ORIGINS=http://<server-ip>:<port> in .env for production."
        )

    os.makedirs(os.path.join(settings.UPLOAD_DIR, "snapshots"), exist_ok=True)
    create_tables()
    ensure_enterprise_schema()
    _run_migrations()

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
        # Wire camera health events (offline / drift) → DB alerts + Telegram
        from services import alert_engine_v2  # noqa: F401
        from services import (camera_alert_handler, notification_engine,
                              rule_engine_v2)

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

    # ── Start server-managed camera streams ───────────────────────────────────

    from services import camera_manager

    _started = camera_manager.start_all()
    print(f"✅ Camera manager: {_started} camera(s) started")

    # ── Start shared YOLO inference pool ──────────────────────────────────
    # One daemon thread runs YOLO for all cameras — no duplicate model loads.
    try:
        from services.inference_pool import inference_pool

        inference_pool.start()
        print("✅ Shared inference pool started (single YOLO model for all cameras)")
    except Exception as _ie:
        print(f"⚠️  Inference pool startup warning: {_ie}")

    # ── Initialize WebSocket Broadcaster & Periodic Telemetry ─────────────
    try:
        import asyncio
        from services.ws_broadcaster import broadcaster

        try:
            loop = asyncio.get_event_loop()
            broadcaster.set_loop(loop)
        except Exception:
            pass

        def _telemetry_broadcast_loop():
            import time
            from services import camera_manager
            from services.tracking_layer import tracker as _tracker
            from database import SessionLocal, AlertCase

            while True:
                time.sleep(1.0)
                try:
                    statuses = camera_manager.list_status()
                    # Real-time person count across cameras
                    live_tracks = 0
                    for (cam_id, tid) in _tracker.all_track_keys():
                        obs = _tracker.get_track(cam_id, tid)
                        if obs and (time.time() - obs.last_seen_at.timestamp()) < 5:
                            live_tracks += 1

                    broadcaster.broadcast_sync({
                        "type": "state_update",
                        "timestamp": time.time(),
                        "cameras": statuses,
                        "live_persons": live_tracks,
                    })
                except Exception:
                    pass

        telemetry_thread = threading.Thread(
            target=_telemetry_broadcast_loop,
            name="telemetry-broadcaster",
            daemon=True,
        )
        telemetry_thread.start()
        print("✅ WebSocket real-time telemetry broadcaster started")
    except Exception as _we:
        print(f"⚠️  WS Broadcaster startup warning: {_we}")

    # ── Nightly auto clock-out scheduler (pure threading — no extra dependency) ──
    # Runs at 19:00 IST (UTC+05:30 = 13:30 UTC) every day.
    # Closes all open attendance sessions and sends a Telegram notification.
    import threading
    from zoneinfo import ZoneInfo

    _IST = ZoneInfo("Asia/Kolkata")

    def _auto_clockout_scheduler():
        """Background daemon: sleep until 19:00 IST each day, then bulk-close."""
        import datetime as _dt
        import logging as _logging

        from database import SessionLocal as _SessionLocal
        from services.attendance_service import \
            auto_clock_out_open_sessions as _aco

        _log = _logging.getLogger("auto_clockout")
        while True:
            now_ist = _dt.datetime.now(tz=_IST)
            target = now_ist.replace(hour=19, minute=0, second=0, microsecond=0)
            if now_ist >= target:
                # Already past 19:00 today — sleep until 19:00 tomorrow
                target += _dt.timedelta(days=1)
            sleep_secs = (target - now_ist).total_seconds()
            _log.info(
                f"[AutoClockOut] Next run in {sleep_secs/3600:.1f}h at {target.strftime('%Y-%m-%d %H:%M IST')}"
            )
            import time as _time

            _time.sleep(max(sleep_secs, 1))
            # Time to run
            _db = None
            try:
                _db = _SessionLocal()
                result = _aco(_db)
                _log.info(f"[AutoClockOut] {result}")
            except Exception as exc:
                _log.error(f"[AutoClockOut] Failed: {exc}")
            finally:
                if _db:
                    try:
                        _db.close()
                    except Exception:
                        pass

    _clockout_thread = threading.Thread(
        target=_auto_clockout_scheduler,
        name="auto-clockout-scheduler",
        daemon=True,
    )
    _clockout_thread.start()
    print("✅ Nightly auto clock-out scheduler started (19:00 IST)")

    # ── Weekly WAL checkpoint + VACUUM (P1 fix) ───────────────────────────────
    # SQLite in WAL mode never auto-checkpoints while writers are active.
    # On Windows, the .db-shm / .db-wal files can grow unbounded and slow reads.
    # This scheduler runs PRAGMA wal_checkpoint(TRUNCATE) + VACUUM every Sunday
    # at 02:00 IST (low-traffic window) to compact the WAL back to the main file.
    def _wal_checkpoint_scheduler():
        """Background daemon: weekly SQLite WAL checkpoint at 02:00 IST Sunday."""
        import datetime as _dt
        import time as _time

        from database import engine as _engine
        from sqlalchemy import text as _text

        _log = logging.getLogger("wal_checkpoint")
        _IST_wal = ZoneInfo("Asia/Kolkata")

        while True:
            now = _dt.datetime.now(tz=_IST_wal)
            # Next Sunday 02:00 IST
            days_until_sunday = (6 - now.weekday()) % 7  # Monday=0 … Sunday=6
            if days_until_sunday == 0 and now.hour >= 2:
                days_until_sunday = 7  # already past the window today
            next_run = (now + _dt.timedelta(days=days_until_sunday)).replace(
                hour=2, minute=0, second=0, microsecond=0
            )
            sleep_secs = (next_run - now).total_seconds()
            _log.info(
                "[WAL] Next checkpoint in %.1fh at %s IST",
                sleep_secs / 3600,
                next_run.strftime("%Y-%m-%d %H:%M"),
            )
            _time.sleep(max(sleep_secs, 1))
            try:
                with _engine.connect() as conn:
                    conn.execute(_text("PRAGMA wal_checkpoint(TRUNCATE)"))
                    conn.execute(_text("VACUUM"))
                    conn.commit()
                _log.info("[WAL] Checkpoint + VACUUM complete")
            except Exception as exc:
                _log.error("[WAL] Checkpoint failed: %s", exc)

    _wal_thread = threading.Thread(
        target=_wal_checkpoint_scheduler,
        name="wal-checkpoint-scheduler",
        daemon=True,
    )
    _wal_thread.start()
    print("✅ Weekly WAL checkpoint scheduler started (Sunday 02:00 IST)")

    print("✅ Safety Monitor API v4.0 started (factory monitoring schema)")


@app.on_event("shutdown")
def shutdown():
    """Gracefully stop all camera reader threads and shared inference pool on server shutdown."""
    from services import camera_manager
    from services.inference_pool import inference_pool

    camera_manager.stop_all()
    inference_pool.stop()
    print("🛑 Safety Monitor: all camera readers and inference pool stopped")
