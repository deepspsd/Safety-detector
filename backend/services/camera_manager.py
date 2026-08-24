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
  start_all()                    → call from FastAPI startup
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
        self.camera_id = camera_id
        self.name = name
        self.url = url
        self.floor = floor  # ground | first | second | shop

        # Lazy-import CameraReader so this module can be imported before
        # routers.cctv without a circular import error.
        from routers.cctv import CameraReader

        self.reader = CameraReader(url)

        self._hb_thread: Optional[threading.Thread] = None
        self._hb_running = False
        self._det_thread: Optional[threading.Thread] = None
        self._det_running = False
        self._last_frame_ts: float = 0.0  # epoch; 0 = no frame yet
        self._last_face_rec_ts: float = 0.0  # rate-limit face recognition to 1 call/2s

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

        log.info(
            f"[CamMgr] Started camera {self.camera_id} — '{self.name}' @ {self.url}"
        )

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

    def metrics(self) -> dict:
        return self.reader.metrics()

    # ── Heartbeat ─────────────────────────────────────────────────────────────

    def _heartbeat_loop(self):
        """
        Ticks every _HB_INTERVAL seconds.
        Opens its own DB session, writes status + last_seen_at, closes it.
        Never holds a connection between ticks.
        """
        from database import Camera as CameraModel  # avoid top-level circular import
        from database import SessionLocal

        while self._hb_running:
            time.sleep(self._HB_INTERVAL)
            if not self._hb_running:
                break

            # Determine current status
            error = self.reader.last_error()
            frame = self.reader.latest_frame()

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
                cam = (
                    db.query(CameraModel)
                    .filter(CameraModel.id == self.camera_id)
                    .first()
                )
                if cam:
                    cam.status = status
                    if status == "online":
                        cam.last_seen_at = datetime.datetime.utcnow()
                        cam.heartbeat_at = cam.last_seen_at
                    cam.health_status = status
                    db.commit()
                    try:
                        from services.health_monitor import (
                            collect_camera_health,
                            record,
                        )
                        from services.platform_events import emit

                        record(
                            "camera",
                            str(self.camera_id),
                            status,
                            collect_camera_health(self.camera_id, self.fps(), error),
                        )
                        if status in ("offline", "error"):
                            emit(
                                "CAMERA_OFFLINE",
                                camera_id=self.camera_id,
                                source="health-monitor",
                                payload={
                                    "status": status,
                                    "error": error,
                                    "fps": self.fps(),
                                },
                            )
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
        Persistent detection + rule-engine loop (~5 fps per camera).

        Each tick:
          1. Read latest frame from CameraReader.
          2. Submit frame to the shared InferencePool (one YOLO model for all cameras).
          3. Read latest inference result (non-blocking — uses previous result if
             the pool hasn't produced a new one yet; rule-engine still runs).
          4. Feed persons to idle_service.
          5. Run rule-engine hooks:
               a. record_person_seen / check_shift_start
               b. process_cylinder_detections
               c. check_dirty_floor
               d. check_shop_absence  (shop floor only)
               e. gas_idle_update     (second floor only)
               f. process_packing_frame  (packing zone cameras only)
               g. process_lift_frame     (lift zone cameras only)
               h. check_cash_zone / check_stock_zone  (shop floor only)
        All rule-engine calls share one SQLAlchemy session opened per tick and
        closed in a finally block — no session is held between ticks.
        """
        from database import Camera as CameraModel
        from database import SessionLocal
        from services import idle_service, rule_engine, zone_service
        from services.enterprise_runtime import runtime as enterprise_runtime

        # zone_type is the camera-level tag (e.g. "packing", "lift").
        # We also merge polygon zone names each tick so that painting a zone
        # polygon is sufficient to activate the matching rule — no need to
        # also change zone_type.
        camera_zone_type = "default"
        db_init = None
        try:
            db_init = SessionLocal()
            cam = (
                db_init.query(CameraModel)
                .filter(CameraModel.id == self.camera_id)
                .first()
            )
            if cam and cam.zone_type:
                camera_zone_type = cam.zone_type
            # Seed settings defaults once per camera-start (safe: upsert only)
            rule_engine.seed_defaults(db_init)
        except Exception as exc:
            log.warning(f"[CamMgr] init DB read error (cam={self.camera_id}): {exc}")
        finally:
            if db_init:
                try:
                    db_init.close()
                except Exception:
                    pass

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

                # active_zones: union of the camera's zone_type tag and all
                # painted polygon names.  Every dispatch check below uses this
                # set — painting a zone polygon IS sufficient to activate the
                # matching rule (no need to also update zone_type).
                polygon_zone_names = set(zones.keys()) if zones else set()
                active_zones: set = polygon_zone_names | {camera_zone_type}

                # ── YOLO inference via shared pool ──────────────────────────
                # Submit the current frame for inference.  The pool runs one
                # YOLO model for ALL cameras — no duplicate model loads.
                # We read back the latest available result (may be from the
                # previous tick if the pool is busy) so the rule-engine loop
                # is never blocked waiting for inference.
                from services.inference_pool import inference_pool

                inference_pool.put_frame(self.camera_id, frame)
                pool_result = inference_pool.get_result(self.camera_id) or {}

                # Tracking is kept on the enterprise_runtime path so ByteTrack
                # state is preserved per-camera across ticks.
                res = enterprise_runtime.process_frame(self.camera_id, frame, db)
                persons = [
                    {
                        "bbox": track["bbox"],
                        "confidence": track["confidence"],
                        "track_id": int(track["track_id"]),
                    }
                    for track in res.get("tracks", [])
                ]
                # Prefer pool detections (fresher class labels) if available,
                # else fall back to enterprise_runtime detections.
                raw_dets = pool_result.get("detections") or res.get("detections", [])

                # ── Idle service ─────────────────────────────────────────────
                idle_service.process_frame(
                    camera_id=self.camera_id,
                    persons=persons,
                    zone_name=camera_zone_type,
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

                # ── Rule engine — gas/oven idle (second floor + any camera with oven/stove polygon) ─
                _has_oven = self.floor == "second" or bool(
                    active_zones & {"oven", "stove", "oven_stove"}
                )
                if _has_oven:
                    stove_polygon = (
                        (zones or {}).get("stove")
                        or (zones or {}).get("oven")
                        or (zones or {}).get("oven_stove")
                    )
                    rule_engine.gas_idle_update(
                        camera_id=self.camera_id,
                        floor=self.floor,
                        frame=frame,
                        person_present=bool(persons),
                        zone_polygon=stove_polygon,
                        db=db,
                    )

                # ── Camera-blocking check (runs on ALL cameras) ───────────────
                # Person standing in front of camera > 1 min → alert.
                rule_engine.check_camera_blocking(
                    self.camera_id, frame.shape, persons, db
                )

                # ── Stock zone check (ALL cameras — raw materials + items) ─────
                # Zone-polygon-gated internally by rule_engine.check_stock_zone.
                rule_engine.check_stock_zone(
                    self.camera_id, self.floor, raw_dets, zones, db
                )

                # ── Rule engine — OCR gate (entrance cameras) ─────────────────────
                # Triggers if zone_type contains "entrance" OR a polygon named
                # "entrance" / "entrance_outward" / "glass_door" is painted.
                _has_entrance = bool(
                    active_zones
                    & {"entrance", "entrance_outward", "glass_door", "loading"}
                )
                if _has_entrance:
                    try:
                        from services.yolo_service import run_ocr_gate_for_camera

                        # outward direction: explicit outward zone_type OR outward polygon present
                        direction = (
                            "outward"
                            if "outward" in camera_zone_type
                            or "entrance_outward" in active_zones
                            else "inward"
                        )
                        run_ocr_gate_for_camera(
                            frame, raw_dets, zones, db, self.camera_id, direction
                        )
                    except Exception as exc:
                        log.debug(
                            f"[CamMgr] ocr_gate error (cam={self.camera_id}): {exc}"
                        )

                # ── Packing monitor — hand-motion check ───────────────────────
                # Triggers if zone_type is packing OR a packing polygon is painted.
                if active_zones & {"packing"}:
                    try:
                        from services import packing_monitor

                        packing_monitor.process_packing_frame(
                            db=db,
                            camera_id=self.camera_id,
                            frame=frame,
                            persons=persons,
                            packing_polygon=(zones or {}).get("packing"),
                        )
                    except Exception as pm_exc:
                        log.debug(
                            f"[CamMgr] packing_monitor error (cam={self.camera_id}): {pm_exc}"
                        )

                # ── Lift monitor — person + item tracking across floors ────────
                # Triggers if zone_type is "lift" OR a "lift" polygon is painted
                # (glass_door cameras also track lift movement).
                if active_zones & {"lift", "glass_door"}:
                    try:
                        from services import lift_monitor

                        lift_monitor.process_lift_frame(
                            db=db,
                            camera_id=self.camera_id,
                            floor=self.floor,
                            persons=persons,
                            lift_polygon=(
                                (zones or {}).get("lift")
                                or (zones or {}).get("glass_door")
                            ),
                        )
                    except Exception as lm_exc:
                        log.debug(
                            f"[CamMgr] lift_monitor error (cam={self.camera_id}): {lm_exc}"
                        )

                # ── Cash + stock monitor ───────────────────────────────────────
                # Triggers on shop floor OR when a cashbox/stock polygon is painted
                # on ANY floor (e.g. a dedicated cash-counter camera on ground floor).
                _has_cash_zone = self.floor == "shop" or bool(
                    active_zones
                    & {
                        "cashbox",
                        "cash_counter",
                        "vendor_desk",
                        "payment_desk",
                        "shop_counter",
                    }
                )
                if _has_cash_zone:
                    try:
                        from services import cash_monitor

                        cash_monitor.check_cash_zone(
                            db=db,
                            camera_id=self.camera_id,
                            detections=raw_dets,
                            cashbox_polygon=(
                                (zones or {}).get("cashbox")
                                or (zones or {}).get("cash_counter")
                            ),
                        )
                        cash_monitor.check_stock_zone(
                            db=db,
                            camera_id=self.camera_id,
                            detections=raw_dets,
                            stock_polygon=(zones or {}).get("stock"),
                        )
                    except Exception as cm_exc:
                        log.debug(
                            f"[CamMgr] cash_monitor error (cam={self.camera_id}): {cm_exc}"
                        )

                # ── Cash-in-pocket heuristic (hand-to-pocket near cashbox) ───────
                if _has_cash_zone:
                    try:
                        rule_engine.check_cash_in_pocket(
                            self.camera_id,
                            self.floor,
                            persons,
                            frame,
                            zones or {},
                            db,
                        )
                    except Exception as cp_exc:
                        log.debug(
                            f"[CamMgr] cash-in-pocket error (cam={self.camera_id}): {cp_exc}"
                        )

                # ── Vendor payment snapshot (shop / vendor_desk cameras) ───────
                # Captures a photo of the payee when a vendor payment is detected.
                if self.floor == "shop" or active_zones & {
                    "vendor_desk",
                    "payment_desk",
                }:
                    rule_engine.trigger_vendor_snapshot(
                        self.camera_id, self.floor, persons, frame, zones or {}, db
                    )

                # ── Dough / cutting-machine post-job idle check ────────────────
                # After dough mixing or cutting finishes the worker must move.
                # Machinery zone check handles this: machine idle = no motion near machine.
                if active_zones & {"dough_table", "cutting_machine", "machine"}:
                    rule_engine.check_machinery_zone(
                        self.camera_id, self.floor, raw_dets, persons, zones or {}, db
                    )

                # ── Window-throw / theft detection (optical flow) ────────────────
                # Detects fast-moving objects near window zones (stealing / throwing).
                # Uses dense optical flow for velocity analysis, plus loiter detection.
                if active_zones & {"window", "window_throw"}:
                    rule_engine.check_window_throw(
                        self.camera_id,
                        self.floor,
                        raw_dets,
                        persons,
                        frame,
                        zones or {},
                        db,
                    )

                # ── Finished goods dispatch monitor ─────────────────────────────
                # Tracks items in finished_goods zone; alerts if no vehicle in
                # loading zone within timeout.
                if active_zones & {"finished_goods", "loading", "vehicle"}:
                    rule_engine.check_finished_goods_dispatch(
                        self.camera_id,
                        self.floor,
                        raw_dets,
                        persons,
                        zones or {},
                        db,
                    )

                # ── Workflow enforcement (dough/biscuit → packing) ──────────────
                # After machinery stops, worker must move to next zone within grace.
                if active_zones & {
                    "dough_table",
                    "dough_mixing",
                    "dough",
                    "cutting_machine",
                    "biscuit_cutting",
                    "machine",
                }:
                    rule_engine.check_workflow_enforcement(
                        self.camera_id,
                        self.floor,
                        raw_dets,
                        persons,
                        zones or {},
                        db,
                    )

                # ── Face recognition → Auto attendance (rate-limited 1× per 2s) ─
                # Runs at 0.5 fps to keep CPU load low while still catching
                # every employee who passes within a 2-second window.
                now_ts = time.time()
                if now_ts - self._last_face_rec_ts >= 2.0:
                    self._last_face_rec_ts = now_ts
                    try:
                        from services import attendance_service, face_service

                        face_result = face_service.process_face_numpy(
                            frame_bgr=frame,
                            user_id=1,  # system user; encodings are shared across users
                            db=db,
                        )
                        recognized = face_result.get("recognized_employees", {})
                        for emp_id, confidence in recognized.items():
                            # handle_face_match opens its own DB session internally
                            attendance_service.handle_face_match(
                                camera_id=self.camera_id,
                                employee_id=emp_id,
                                confidence=confidence,
                            )
                    except Exception as face_exc:
                        log.debug(
                            f"[CamMgr] Face recognition error (cam={self.camera_id}): {face_exc}"
                        )

                # ── Chewing + clean-shave detection (MediaPipe) ───────────────────
                # Runs at low frequency internally; skips if MEDIAPIPE_ENABLED=false.
                try:
                    from services import chew_monitor

                    chew_monitor.process_frame(
                        db=db,
                        camera_id=self.camera_id,
                        frame=frame,
                        persons=persons,
                    )
                except Exception as chew_exc:
                    log.debug(
                        f"[CamMgr] chew_monitor error (cam={self.camera_id}): {chew_exc}"
                    )

                # ── Eating from store detection (stock/storage zones) ────────────
                # Detects hand-to-mouth gestures near stock areas — employees
                # eating items from the store (REQ-SF-06).
                _has_stock_zone = bool(
                    active_zones
                    & {"stock", "stock_area", "raw_material", "store", "storage"}
                )
                if _has_stock_zone:
                    try:
                        from services import chew_monitor as _cm

                        _cm.check_eating_from_store(
                            db=db,
                            camera_id=self.camera_id,
                            frame=frame,
                            persons=persons,
                        )
                    except Exception as eat_exc:
                        log.debug(
                            f"[CamMgr] eating_from_store error (cam={self.camera_id}): {eat_exc}"
                        )

            except Exception as exc:
                log.error(
                    f"[CamMgr] Detection loop error (cam={self.camera_id}): {exc}"
                )
            finally:
                if db:
                    try:
                        db.close()
                    except Exception:
                        pass


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def start_all() -> int:
    """
    Load all Camera rows with status != 'offline' from the DB and start one
    ManagedCamera per row.  Called from FastAPI startup event in main.py.

    Returns the number of cameras started.
    """
    from database import Camera as CameraModel
    from database import CameraStreamProfile, SessionLocal
    from services.camera_credentials import decrypt

    db = SessionLocal()
    camera_configs = []
    try:
        rows = (
            db.query(CameraModel)
            .filter(CameraModel.status != "offline")
            .filter(CameraModel.status != "deleted")
            .all()
        )
        for row in rows:
            stream_url = (
                row.rtsp_url.strip() if row.rtsp_url and row.rtsp_url.strip() else None
            )
            if stream_url is None:
                desired = row.preferred_stream or "sub"
                profile = (
                    db.query(CameraStreamProfile)
                    .filter(
                        CameraStreamProfile.camera_id == row.id,
                        CameraStreamProfile.active.is_(True),
                        CameraStreamProfile.stream_type == desired,
                    )
                    .order_by(CameraStreamProfile.id)
                    .first()
                ) or (
                    db.query(CameraStreamProfile)
                    .filter(
                        CameraStreamProfile.camera_id == row.id,
                        CameraStreamProfile.active.is_(True),
                    )
                    .order_by(CameraStreamProfile.id)
                    .first()
                )
                if profile:
                    try:
                        stream_url = decrypt(profile.encrypted_rtsp_uri)
                    except Exception as exc:
                        log.error(
                            f"[CamMgr] Cannot decrypt stream for camera {row.id} "
                            f"'{row.name}': {exc}"
                        )

            if stream_url:
                camera_configs.append(
                    (row.id, row.name, stream_url, row.floor or "ground")
                )
            else:
                log.warning(
                    f"[CamMgr] Skipping camera {row.id} '{row.name}' — no configured stream"
                )
    finally:
        db.close()

    started = 0
    for camera_id, name, stream_url, floor in camera_configs:
        try:
            _launch(camera_id, name, stream_url, floor=floor)
            started += 1
        except Exception as exc:
            log.error(f"[CamMgr] Failed to start camera {camera_id}: {exc}")

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
        try:
            from services import chew_monitor

            chew_monitor.cleanup_camera(camera_id)
        except Exception:
            pass

        try:
            from services.rule_engine import cleanup_window_state

            cleanup_window_state(camera_id)
        except Exception:
            pass
        log.info(f"[CamMgr] camera {camera_id} removed from registry")
    else:
        log.warning(f"[CamMgr] stop_camera({camera_id}) — not in registry, skipped")


def restart_camera(camera_id: int, name: str, rtsp_url: str, floor: str = "ground"):
    """
    Stop the existing reader (if any) then start a fresh one.
    Used by POST /cameras/{id}/restart.
    """
    stop_camera(camera_id)
    time.sleep(0.5)  # brief pause to let the old thread exit cleanly
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


def get_metrics(camera_id: int) -> dict:
    """Browser-safe operational metrics. Never returns stream URLs."""
    with _lock:
        mc = _registry.get(camera_id)
    if not mc:
        return {
            "fps": 0.0,
            "last_frame_at": None,
            "reconnect_count": 0,
            "last_error": "Camera is not streaming",
        }
    return mc.metrics()


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
        result.append(
            {
                "camera_id": cid,
                "name": mc.name,
                "fps": mc.fps(),
                "error": mc.last_error(),
                "has_frame": mc.reader.latest_frame() is not None,
                **mc.metrics(),
            }
        )
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
