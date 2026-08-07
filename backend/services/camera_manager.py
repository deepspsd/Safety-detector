"""
camera_manager.py — Server-owned persistent camera stream service
==================================================================
Runs one CameraReader per registered camera as a daemon thread.
Streams survive independently of browser WebSocket connections.

Design notes
────────────
• CameraReader is imported lazily from routers.cctv to avoid a
  services → routers circular import at module-load time.
  By the time start_all() is called (FastAPI startup event), all
  routers are already loaded, so the lazy import is just a dict
  lookup in sys.modules.

• Each ManagedCamera owns a heartbeat thread that writes
  Camera.status / Camera.last_seen_at to the DB every 10 s.
  The heartbeat opens and closes its own SQLAlchemy session so it
  never holds a connection open between ticks.

• The global registry is protected by a threading.Lock.
  All public functions are thread-safe.

• Alert ownership: server-managed alerts use user_id=0 (system) and
  stamp camera_id on the Alert row so admins can filter per-camera.
  This is separate from per-viewer alerts fired from WebSocket sessions.

Public API
──────────
  start_all(db_url)              → call from FastAPI startup
  stop_all()                     → call from FastAPI shutdown
  start_camera(camera_row)       → live-add camera (POST /cameras)
  stop_camera(camera_id)         → live-remove camera (DELETE /cameras)
  restart_camera(camera_id)      → stop + re-start reader thread
  get_latest_frame(camera_id)    → np.ndarray | None  (subscribers read this)
  get_reader_fps(camera_id)      → float
  get_reader_error(camera_id)    → str | None
  list_status()                  → list[dict]  (for GET /cameras response)
"""

from __future__ import annotations

import datetime
import logging
import threading
import time
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger("camera_manager")

# ── Module-level registry ────────────────────────────────────────────────────
_registry: Dict[int, "_ManagedCamera"] = {}
_lock = threading.Lock()

# ─────────────────────────────────────────────────────────────────────────────
# Internal: per-camera wrapper
# ─────────────────────────────────────────────────────────────────────────────

class _ManagedCamera:
    """
    Owns one CameraReader + one heartbeat thread for a single camera row.

    Lifecycle:
        mc = _ManagedCamera(camera_id, name, rtsp_url)
        mc.start()
        ...
        mc.stop()
    """

    # How often to tick the heartbeat DB write (seconds)
    _HB_INTERVAL = 10
    # After this many seconds with no fresh frame, declare offline
    _OFFLINE_TIMEOUT = 30

    def __init__(self, camera_id: int, name: str, url: str, floor: str = "ground"):
        self.camera_id   = camera_id
        self.name        = name
        self.url         = url
        self.floor       = floor   # ground | first | second | shop

        # Lazy-import CameraReader so this module can be imported before
        # routers.cctv without a circular import error.
        from routers.cctv import CameraReader
        self.reader = CameraReader(url)

        self._hb_thread: Optional[threading.Thread] = None
        self._hb_running = False
        self._det_thread: Optional[threading.Thread] = None
        self._det_running = False
        self._last_frame_ts: float = 0.0   # epoch; 0 = no frame yet

    # ── Public ────────────────────────────────────────────────────────────────

    def start(self):
        self.reader.start()
        self._hb_running = True
        self._hb_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"cam-hb-{self.camera_id}",
            daemon=True,
        )
        self._hb_thread.start()

        self._det_running = True
        self._det_thread = threading.Thread(
            target=self._detection_loop,
            name=f"cam-det-{self.camera_id}",
            daemon=True,
        )
        self._det_thread.start()

        log.info(f"[CamMgr] Started camera {self.camera_id} — '{self.name}' @ {self.url}")

    def stop(self):
        self._hb_running = False
        self._det_running = False
        self.reader.stop()

        # Clean up tracker & idle state
        try:
            from services import idle_service
            idle_service.close_all_for_camera(self.camera_id)
        except Exception as e:
            log.error(f"[CamMgr] idle_service cleanup error: {e}")

        try:
            from services import yolo_service
            yolo_service.reset_tracker(self.camera_id)
        except Exception as e:
            log.error(f"[CamMgr] yolo_service tracker reset error: {e}")

        try:
            from services.enterprise_runtime import runtime
            runtime.reset_camera(self.camera_id)
        except Exception as e:
            log.error(f"[CamMgr] enterprise runtime reset error: {e}")

        log.info(f"[CamMgr] Stopped camera {self.camera_id}")

    def latest_frame(self) -> Optional[np.ndarray]:
        frame = self.reader.latest_frame()
        if frame is not None:
            self._last_frame_ts = time.time()
        return frame

    def fps(self) -> float:
        return self.reader.fps()

    def last_error(self) -> Optional[str]:
        return self.reader.last_error()

    # ── Heartbeat ─────────────────────────────────────────────────────────────

    def _heartbeat_loop(self):
        """
        Ticks every _HB_INTERVAL seconds.
        Opens its own DB session, writes status + last_seen_at, closes it.
        Never holds a connection between ticks.
        """
        from database import SessionLocal
        from database import Camera as CameraModel  # avoid top-level circular import

        while self._hb_running:
            time.sleep(self._HB_INTERVAL)
            if not self._hb_running:
                break

            # Determine current status
            error  = self.reader.last_error()
            frame  = self.reader.latest_frame()

            if frame is not None:
                self._last_frame_ts = time.time()

            if error:
                status = "error"
            elif self._last_frame_ts == 0.0:
                status = "online"  # reader just started, no frame yet — give it time
            elif time.time() - self._last_frame_ts > self._OFFLINE_TIMEOUT:
                status = "offline"
            else:
                status = "online"

            # Write to DB — one session per tick
            db = None
            try:
                db = SessionLocal()
                cam = db.query(CameraModel).filter(
                    CameraModel.id == self.camera_id
                ).first()
                if cam:
                    cam.status = status
                    if status == "online":
                        cam.last_seen_at = datetime.datetime.utcnow()
                        cam.heartbeat_at = cam.last_seen_at
                    cam.health_status = status
                    db.commit()
                    try:
                        from services.health_monitor import collect_camera_health, record
                        from services.platform_events import emit
                        record("camera", str(self.camera_id), status,
                               collect_camera_health(self.camera_id, self.fps(), error))
                        if status in ("offline", "error"):
                            emit("CAMERA_OFFLINE", camera_id=self.camera_id, source="health-monitor",
                                 payload={"status": status, "error": error, "fps": self.fps()})
                    except Exception:
                        pass
                    log.debug(
                        f"[CamMgr] Heartbeat cam={self.camera_id} "
                        f"status={status} fps={self.fps():.1f}"
                    )
            except Exception as exc:
                log.error(
                    f"[CamMgr] Heartbeat DB write failed "
                    f"(cam={self.camera_id}): {exc}"
                )
                if db:
                    try:
                        db.rollback()
                    except Exception:
                        pass
            finally:
                if db:
                    try:
                        db.close()
                    except Exception:
                        pass

    # ── Detection & Idle Loop ──────────────────────────────────────────────────

    def _detection_loop(self):
        """
        Persistent detection + rule-engine loop (~5 fps).

        Each tick:
          1. Read latest frame from CameraReader.
          2. Run YOLO inference + ByteTrack via yolo_service.
          3. Feed persons to idle_service.
          4. Run rule-engine hooks:
               a. record_person_seen / check_shift_start
               b. process_cylinder_detections
               c. check_dirty_floor
               d. check_shop_absence  (shop floor only)
               e. gas_idle_update     (second floor only)
        All rule-engine calls share one SQLAlchemy session opened per tick and
        closed in a finally block — no session is held between ticks.
        """
        from services import idle_service
        from services import zone_service, rule_engine
        from services.enterprise_runtime import runtime as enterprise_runtime
        from database import SessionLocal, Camera as CameraModel

        zone_name = "default"
        db_init = None
        try:
            db_init = SessionLocal()
            cam = db_init.query(CameraModel).filter(CameraModel.id == self.camera_id).first()
            if cam and cam.zone_type:
                zone_name = cam.zone_type
            # Seed settings defaults once per camera-start (safe: upsert only)
            rule_engine.seed_defaults(db_init)
        except Exception as exc:
            log.warning(f"[CamMgr] init DB read error (cam={self.camera_id}): {exc}")
        finally:
            if db_init:
                try: db_init.close()
                except Exception: pass

        while self._det_running:
            time.sleep(0.2)
            if not self._det_running:
                break

            frame = self.latest_frame()
            if frame is None:
                continue

            db = None
            try:
                db = SessionLocal()

                # ── Zone config (cached) ────────────────────────────────────
                zones = zone_service.load_zones_without_db(self.camera_id)
                if zones is None:
                    zones = zone_service.load_zones_for_camera(self.camera_id, db)

                # ── YOLO inference + tracking ───────────────────────────────
                # Enterprise path: YOLO produces only raw detections; tracking,
                # semantic context, events, workflows and rules are independent.
                # The legacy idle/cylinder hooks below consume the neutral output
                # during the migration period, preserving existing behaviour.
                res = enterprise_runtime.process_frame(self.camera_id, frame, db)
                persons = [
                    {"bbox": track["bbox"], "confidence": track["confidence"], "track_id": int(track["track_id"])}
                    for track in res.get("tracks", [])
                ]
                raw_dets = res.get("detections", [])

                # ── Idle service ─────────────────────────────────────────────
                idle_service.process_frame(
                    camera_id=self.camera_id,
                    persons=persons,
                    zone_name=zone_name,
                )

                # ── Rule engine — person seen recording (shift-start) ────────
                if persons:
                    rule_engine.record_person_seen(self.floor)

                # ── Rule engine — shift-start check (once/day/floor) ─────────
                rule_engine.check_shift_start(self.camera_id, self.floor, db)

                # ── Rule engine — cylinder tracking ──────────────────────────
                rule_engine.process_cylinder_detections(
                    camera_id=self.camera_id,
                    floor=self.floor,
                    raw_detections=raw_dets,
                    db=db,
                )

                # ── Rule engine — dirty-floor heuristic ──────────────────────
                rule_engine.check_dirty_floor(
                    camera_id=self.camera_id,
                    floor=self.floor,
                    frame=frame,
                    db=db,
                )

                # ── Rule engine — shop absence (shop floor only) ──────────────
                rule_engine.check_shop_absence(
                    camera_id=self.camera_id,
                    floor=self.floor,
                    persons=persons,
                    db=db,
                )

                # ── Rule engine — gas/oven idle (second floor only) ───────────
                if self.floor == "second":
                    stove_polygon = (zones or {}).get("stove") or (zones or {}).get("oven")
                    rule_engine.gas_idle_update(
                        camera_id=self.camera_id,
                        floor=self.floor,
                        frame=frame,
                        person_present=bool(persons),
                        zone_polygon=stove_polygon,
                        db=db,
                    )

            except Exception as exc:
                log.error(f"[CamMgr] Detection loop error (cam={self.camera_id}): {exc}")
            finally:
                if db:
                    try: db.close()
                    except Exception: pass


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def start_all() -> int:
    """
    Load all Camera rows with status != 'offline' from the DB and start one
    ManagedCamera per row.  Called from FastAPI startup event in main.py.

    Returns the number of cameras started.
    """
    from database import SessionLocal
    from database import Camera as CameraModel

    db = SessionLocal()
    try:
        rows = (
            db.query(CameraModel)
            .filter(CameraModel.status != "offline")
            .filter(CameraModel.rtsp_url.isnot(None))
            .all()
        )
    finally:
        db.close()

    started = 0
    for row in rows:
        if not row.rtsp_url or not row.rtsp_url.strip():
            log.warning(f"[CamMgr] Skipping camera {row.id} '{row.name}' — no rtsp_url")
            continue
        try:
            _launch(row.id, row.name, row.rtsp_url, floor=row.floor or "ground")
            started += 1
        except Exception as exc:
            log.error(f"[CamMgr] Failed to start camera {row.id}: {exc}")

    log.info(f"[CamMgr] start_all(): {started}/{len(rows)} cameras launched")
    return started


def stop_all():
    """
    Stop all running readers gracefully.
    Called from FastAPI shutdown event in main.py.
    """
    with _lock:
        camera_ids = list(_registry.keys())

    for cid in camera_ids:
        try:
            stop_camera(cid)
        except Exception as exc:
            log.error(f"[CamMgr] stop_all error for cam {cid}: {exc}")

    log.info("[CamMgr] stop_all(): all cameras stopped")


def start_camera(camera_id: int, name: str, rtsp_url: str, floor: str = "ground"):
    """
    Start a reader for a single camera (live-add from POST /cameras).
    No-op if a reader is already running for camera_id.
    """
    with _lock:
        if camera_id in _registry:
            log.info(f"[CamMgr] camera {camera_id} already running — skipped")
            return
    _launch(camera_id, name, rtsp_url, floor=floor)


def stop_camera(camera_id: int):
    """
    Stop and remove the reader for camera_id (live-remove from DELETE /cameras).
    No-op if camera_id is not registered.
    """
    with _lock:
        mc = _registry.pop(camera_id, None)
    if mc:
        mc.stop()
        log.info(f"[CamMgr] camera {camera_id} removed from registry")
    else:
        log.warning(f"[CamMgr] stop_camera({camera_id}) — not in registry, skipped")


def restart_camera(camera_id: int, name: str, rtsp_url: str, floor: str = "ground"):
    """
    Stop the existing reader (if any) then start a fresh one.
    Used by POST /cameras/{id}/restart.
    """
    stop_camera(camera_id)
    time.sleep(0.5)   # brief pause to let the old thread exit cleanly
    _launch(camera_id, name, rtsp_url, floor=floor)
    log.info(f"[CamMgr] camera {camera_id} restarted")


def get_latest_frame(camera_id: int) -> Optional[np.ndarray]:
    """Return the most recent frame from the managed reader, or None."""
    with _lock:
        mc = _registry.get(camera_id)
    return mc.latest_frame() if mc else None


def get_reader_fps(camera_id: int) -> float:
    with _lock:
        mc = _registry.get(camera_id)
    return mc.fps() if mc else 0.0


def get_reader_error(camera_id: int) -> Optional[str]:
    with _lock:
        mc = _registry.get(camera_id)
    return mc.last_error() if mc else None


def is_running(camera_id: int) -> bool:
    with _lock:
        return camera_id in _registry


def list_status() -> List[dict]:
    """
    Return a snapshot of all registered cameras and their live state.
    Used by GET /cameras to enrich DB rows with real-time info.
    """
    with _lock:
        snapshot = list(_registry.items())

    result = []
    for cid, mc in snapshot:
        result.append({
            "camera_id":   cid,
            "name":        mc.name,
            "url":         mc.url,
            "fps":         mc.fps(),
            "error":       mc.last_error(),
            "has_frame":   mc.reader.latest_frame() is not None,
        })
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper
# ─────────────────────────────────────────────────────────────────────────────

def _launch(camera_id: int, name: str, url: str, floor: str = "ground"):
    """Create, register, and start a ManagedCamera.  Not lock-safe — callers manage."""
    mc = _ManagedCamera(camera_id, name, url, floor=floor)
    mc.start()
    with _lock:
        _registry[camera_id] = mc
