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
        from database import \
            Camera as CameraModel  # avoid top-level circular import
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
                            collect_camera_health, record)
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
                # NOW WITH MULTI-MODEL SUPPORT: Pass zone_type for model routing
                from services.inference_pool import inference_pool

                inference_pool.put_frame(self.camera_id, frame, zone_type=camera_zone_type)
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

                # ── Head-Cap monitor (crop-based inference + temporal smoothing) ─
                try:
                    from services.headcap_monitor import headcap_monitor
                    headcap_monitor.update_camera(
                        camera_id=self.camera_id,
                        persons=persons,
                        raw_dets=raw_dets,
                        db=db,
                        frame=frame,
                    )
                except Exception as hc_exc:
                    log.debug(f"[CamMgr] headcap_monitor error (cam={self.camera_id}): {hc_exc}")

                # ── Uniform monitor (crop-based inference + temporal smoothing) ──
                try:
                    from services.uniform_monitor import uniform_monitor
                    uniform_monitor.update_camera(
                        camera_id=self.camera_id,
                        persons=persons,
                        db=db,
                        frame=frame,
                    )
                except Exception as um_exc:
                    log.debug(f"[CamMgr] uniform_monitor error (cam={self.camera_id}): {um_exc}")

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
                        from services.yolo_service import \
                            run_ocr_gate_for_camera

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

                # ── Cash + stock monitor (CASHBOX & VENDOR PAYEE) ───────────
                _cashbox_polygon = (
                    (zones or {}).get("cashbox")
                    or (zones or {}).get("cash_counter")
                    or (zones or {}).get("cash_drawer")
                )
                _vendor_polygon = (
                    (zones or {}).get("vendor_desk")
                    or (zones or {}).get("payment_desk")
                    or (zones or {}).get("shop_counter")
                    or (zones or {}).get("vendor")
                )
                if _cashbox_polygon is not None or _vendor_polygon is not None or self.floor == "shop":
                    try:
                        from services import cash_monitor

                        cash_monitor.check_cash_zone(
                            db=db,
                            camera_id=self.camera_id,
                            detections=raw_dets,
                            persons=persons,
                            cashbox_polygon=_cashbox_polygon,
                            floor=self.floor,
                            frame=frame,
                            vendor_polygon=_vendor_polygon,
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


                # ── Cash-in-pocket: now handled inside CashEventTracker ──────────
                # check_cash_in_pocket() in rule_engine is a no-op shim kept
                # for API compatibility only. No separate call needed here.

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

                # ─── NEW PLATFORM SERVICES ────────────────────────────────────────
                # These calls are wrapped individually so a failure in one module
                # never breaks the rest of the detection loop tick.
                # ─────────────────────────────────────────────────────────────────

                # ── Zone transitions + line crossing ───────────────────────────
                try:
                    from services.tracking_layer import tracker as track_store
                    from services.zone_service import (
                        check_line_crossing,
                        compute_zone_for_track,
                        load_virtual_lines,
                        update_track_zone,
                    )
                    virtual_lines = load_virtual_lines(self.camera_id, db)
                    for person in persons:
                        tid = str(person.get("track_id", ""))
                        if not tid:
                            continue
                        bx = person["bbox"]
                        cx = (bx[0] + bx[2]) / 2
                        cy = (bx[1] + bx[3]) / 2
                        transition = update_track_zone(
                            self.camera_id, tid, cx, cy, zones or {}
                        )
                        obs = track_store.get_track(self.camera_id, tid)
                        if obs:
                            zone_now = compute_zone_for_track(
                                self.camera_id, tid, cx, cy, zones or {}
                            )
                            if transition:
                                obs.previous_zone = transition.get("previous_zone")
                            obs.current_zone = zone_now
                            if virtual_lines and len(obs.history) >= 2:
                                prev = obs.history[-2]
                                curr = obs.history[-1]
                                check_line_crossing(
                                    self.camera_id,
                                    tid,
                                    (prev["cx"], prev["cy"]),
                                    (curr["cx"], curr["cy"]),
                                    virtual_lines,
                                )
                except Exception as ze:
                    log.debug("[CamMgr] zone_transition error (cam=%s): %s", self.camera_id, ze)

                # ── Attribute classifiers (uniform, head_cap, bangle) ──────────
                try:
                    import json
                    from database import PersonAttributeResult
                    from services.classifier_adapter import (
                        ClassifierCache,
                        TemporalSmoother,
                        get_classifier,
                        get_classifier_configs,
                    )
                    from services.person_cropper import crop_person, save_debug_crop
                    from services.platform_events import emit
                    from services.tracking_layer import tracker as track_store

                    if not hasattr(self, "_clf_cache"):
                        self._clf_cache = ClassifierCache()
                        self._clf_smoothers = {}
                        self._classifier_configs = get_classifier_configs()
                        self._classifiers = {
                            name: get_classifier(name)
                            for name, cfg in self._classifier_configs.items()
                            if cfg.get("classes")
                        }
                        for name, cfg in self._classifier_configs.items():
                            self._clf_smoothers[name] = TemporalSmoother(
                                window=cfg.get("smoothing_window")
                            )

                    for person in persons:
                        tid = str(person.get("track_id", ""))
                        if not tid:
                            continue
                        for clf_name, clf in self._classifiers.items():
                            cfg = self._classifier_configs.get(clf_name, {})
                            if not self._clf_cache.should_run(
                                tid, clf_name, cfg.get("inference_interval_ms")
                            ):
                                continue
                            target_size = (
                                int(cfg.get("input_width", 224)),
                                int(cfg.get("input_height", 224)),
                            )
                            crop = crop_person(
                                frame,
                                person["bbox"],
                                crop_mode=cfg.get("crop_mode", "full_person"),
                                padding=float(cfg.get("padding", 0.05)),
                                upper_body_ratio=float(cfg.get("upper_body_ratio", 0.65)),
                                target_size=target_size,
                            )
                            if crop is None:
                                continue
                            result = clf.predict(crop)
                            self._clf_cache.record(tid, clf_name)
                            smoother = self._clf_smoothers[clf_name]
                            smoother.push(tid, clf_name, result)
                            smoothed = smoother.vote(tid, clf_name)
                            final_result = smoothed or result
                            save_debug_crop(
                                crop,
                                self.camera_id,
                                f"track_{tid}",
                                clf_name,
                                final_result.predicted_class,
                                final_result.confidence,
                            )

                            obs = track_store.get_track(self.camera_id, tid)
                            if obs:
                                obs.last_attribute_predictions[clf_name] = {
                                    "raw": result.to_dict(),
                                    "smoothed": smoothed.to_dict() if smoothed else None,
                                }

                            db.add(
                                PersonAttributeResult(
                                    camera_id=self.camera_id,
                                    track_id=tid,
                                    classifier_name=clf_name,
                                    predicted_class=result.predicted_class,
                                    confidence=result.confidence,
                                    smoothed_class=final_result.predicted_class,
                                    smoothed_confidence=final_result.confidence,
                                    model_loaded=result.model_loaded,
                                    raw_probabilities_json=json.dumps(
                                        result.all_class_probabilities
                                    ),
                                )
                            )
                            db.commit()

                            threshold = float(cfg.get("confidence_threshold", 0.80))
                            event_type = {
                                "uniform": "NO_UNIFORM",
                                "head_cap": "NO_HEAD_CAP",
                                "bangle": "BANGLE_DETECTED",
                            }.get(clf_name)
                            if (
                                event_type
                                and final_result.confidence >= threshold
                                and final_result.predicted_class
                                in {"NO_UNIFORM", "NO_HEAD_CAP", "BANGLE"}
                            ):
                                emit(
                                    event_type,
                                    camera_id=self.camera_id,
                                    track_id=tid,
                                    confidence=final_result.confidence,
                                    source="person-attribute-classifier",
                                    payload={
                                        "classifier": clf_name,
                                        "predicted_class": final_result.predicted_class,
                                        "model_loaded": result.model_loaded,
                                        "message": (
                                            f"{final_result.predicted_class} "
                                            f"for person #{tid}; review evidence"
                                        ),
                                    },
                                )

                        # Evict stale tracks every 30 s
                        active_ids = [str(p.get("track_id", "")) for p in persons]
                        self._clf_cache.evict_old_tracks(active_ids)
                        for smoother in self._clf_smoothers.values():
                            smoother.evict_old_tracks(active_ids)
                except Exception as clf_exc:
                    log.debug("[CamMgr] classifier error (cam=%s): %s", self.camera_id, clf_exc)

                # ── Idle / absence / camera-standing rule state machines ────────
                try:
                    from services.rule_engine_v2 import (
                        absence_rule, camera_standing_rule, idle_rule, shift_checker,
                    )
                    from services.tracking_layer import tracker as track_store

                    person_count_in_shop = 0
                    for person in persons:
                        tid = str(person.get("track_id", ""))
                        if not tid:
                            continue
                        obs = track_store.get_track(self.camera_id, tid)
                        mvstate = obs.movement_state if obs else "stationary"
                        cur_zone = obs.current_zone if obs else None

                        idle_rule.update(self.camera_id, tid, mvstate, zone=cur_zone)

                        in_standing_zone = cur_zone == "camera_standing" or bool(
                            active_zones & {"camera_standing", "camera_block"}
                        )
                        camera_standing_rule.update(self.camera_id, tid, in_standing_zone)

                        if cur_zone in ("shop_counter", "shop", None) and self.floor in ("shop", "bakery"):
                            person_count_in_shop += 1

                    absence_rule.update(self.camera_id, person_count_in_shop)
                    shift_checker.tick(self.camera_id, self.floor, len(persons))
                except Exception as re_exc:
                    log.debug("[CamMgr] rule-sm error (cam=%s): %s", self.camera_id, re_exc)

                # ─── END NEW PLATFORM SERVICES ────────────────────────────────

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
