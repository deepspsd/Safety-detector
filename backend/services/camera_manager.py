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

    def __init__(self, camera_id: int, name: str, url: str):
        self.camera_id   = camera_id
        self.name        = name
        self.url         = url

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
                    db.commit()
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
        Runs continuously while self._det_running is True.
        Grabs frame -> runs process_frame_numpy_tracked() -> feeds idle_service.
        Runs at ~5 fps (sleeping ~0.2s between iterations).
        """
        from services import yolo_service, idle_service
        from database import SessionLocal, Camera as CameraModel

        zone_name = "default"
        # Fetch initial zone_type for camera from DB
        db = None
        try:
            db = SessionLocal()
            cam = db.query(CameraModel).filter(CameraModel.id == self.camera_id).first()
            if cam and cam.zone_type:
                zone_name = cam.zone_type
        except Exception:
            pass
        finally:
            if db:
                try: db.close()
                except Exception: pass

        while self._det_running:
            time.sleep(0.2)
            if not self._det_running:
                break

            frame = self.latest_frame()
            if frame is None:
                continue

            try:
                res = yolo_service.process_frame_numpy_tracked(
                    frame=frame,
                    role="Factory Worker",
                    camera_id=self.camera_id,
                )
                persons = res.get("persons", [])
                idle_service.process_frame(
                    camera_id=self.camera_id,
                    persons=persons,
                    zone_name=zone_name,
                )
            except Exception as exc:
                log.error(f"[CamMgr] Detection loop error (cam={self.camera_id}): {exc}")


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
            _launch(row.id, row.name, row.rtsp_url)
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


def start_camera(camera_id: int, name: str, rtsp_url: str):
    """
    Start a reader for a single camera (live-add from POST /cameras).
    No-op if a reader is already running for camera_id.
    """
    with _lock:
        if camera_id in _registry:
            log.info(f"[CamMgr] camera {camera_id} already running — skipped")
            return
    _launch(camera_id, name, rtsp_url)


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


def restart_camera(camera_id: int, name: str, rtsp_url: str):
    """
    Stop the existing reader (if any) then start a fresh one.
    Used by POST /cameras/{id}/restart.
    """
    stop_camera(camera_id)
    time.sleep(0.5)   # brief pause to let the old thread exit cleanly
    _launch(camera_id, name, rtsp_url)
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

def _launch(camera_id: int, name: str, url: str):
    """Create, register, and start a ManagedCamera.  Not lock-safe — callers manage."""
    mc = _ManagedCamera(camera_id, name, url)
    mc.start()
    with _lock:
        _registry[camera_id] = mc
