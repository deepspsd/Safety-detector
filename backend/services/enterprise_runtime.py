"""Orchestrates independent layers for server-managed cameras.

This is the migration seam: legacy WebSocket payloads remain untouched, while
managed cameras use raw detection -> tracking -> context -> events -> workflow
-> configurable rules -> alerts/notifications.
"""
from __future__ import annotations

from datetime import datetime
import threading
import time
from typing import Dict, Tuple

from services.calibration_service import assess_drift
from services.context_engine import context_engine
from services.detection_layer import detector
from services.health_monitor import collect_camera_health, record
from services.platform_events import emit
from services.tracking_layer import tracker
from services.workflow_engine_v2 import workflow_engine


class EnterpriseRuntime:
    def __init__(self) -> None:
        self._idle_since: Dict[Tuple[int, str], float] = {}
        self._idle_emitted: set[Tuple[int, str]] = set()
        self._drift_check_at: Dict[int, float] = {}
        self._lock = threading.Lock()

    def process_frame(self, camera_id: int, frame, db) -> Dict:
        from database import Camera
        started = time.monotonic()
        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        if not camera or not camera.ai_enabled:
            return {"detections": [], "tracks": [], "contexts": []}
        detections = detector.detect(frame)
        tracks = tracker.update(camera_id, detections) if camera.supports_tracking else []
        contexts = context_engine.build(camera_id, tracks, detections, db)
        for context in contexts:
            workflow_engine.apply(context, camera.workflow_profile_id)
            self._emit_temporal_events(context)
        self._check_drift(camera, frame, db)
        elapsed_ms = (time.monotonic() - started) * 1000
        record("camera", str(camera_id), "online", {"inference_ms": round(elapsed_ms, 1),
               "detections": len(detections), "tracks": len(tracks)})
        return {"detections": [item.to_dict() for item in detections],
                "tracks": [item.to_dict() for item in tracks],
                "contexts": [item.to_dict() for item in contexts], "inference_ms": round(elapsed_ms, 1)}

    def _emit_temporal_events(self, context) -> None:
        key = (context.camera_id, context.track_id)
        now = time.monotonic()
        if context.movement_state == "moving":
            if key in self._idle_since:
                started = self._idle_since.pop(key)
                self._idle_emitted.discard(key)
                emit("IDLE_STOPPED", camera_id=context.camera_id, zone_id=context.zone_id, track_id=context.track_id,
                     calibration_version=context.calibration_version, source="temporal-context",
                     payload={"idle_seconds": round(now - started, 1), "context": context.to_dict()})
            return
        with self._lock:
            started = self._idle_since.setdefault(key, now)
        idle_threshold = 300.0
        # Zone-defined threshold is data/config, never a business action here.
        if context.zone_id:
            try:
                from database import SessionLocal, ZoneConfig
                local = SessionLocal()
                try:
                    zone = local.query(ZoneConfig).filter(ZoneConfig.id == context.zone_id).first()
                    idle_threshold = float(zone.idle_threshold or idle_threshold) if zone else idle_threshold
                finally:
                    local.close()
            except Exception:
                pass
        elapsed = now - started
        # Emit once per stationary period. Frame cadence is not stable enough
        # to rely on elapsed-time modulus boundaries.
        if elapsed >= idle_threshold and key not in self._idle_emitted:
            self._idle_emitted.add(key)
            emit("IDLE_STARTED", camera_id=context.camera_id, zone_id=context.zone_id, track_id=context.track_id,
                 calibration_version=context.calibration_version, source="temporal-context",
                 payload={"idle_seconds": round(elapsed, 1), "threshold_seconds": idle_threshold, "context": context.to_dict()})

    def _check_drift(self, camera, frame, db) -> None:
        now = time.monotonic()
        if now - self._drift_check_at.get(camera.id, 0) < 60:
            return
        self._drift_check_at[camera.id] = now
        drifted, score, method = assess_drift(camera, frame)
        camera.drift_score = score
        if drifted:
            camera.calibration_status = "required"
            emit("CAMERA_DRIFT_DETECTED", camera_id=camera.id, calibration_version=camera.calibration_version,
                 source="calibration-layer", payload={"score": score, "method": method, "rules_disabled": True})
        db.commit()

    def reset_camera(self, camera_id: int) -> None:
        """Release temporal state when a managed camera is stopped or restarted."""
        with self._lock:
            for key in [key for key in self._idle_since if key[0] == camera_id]:
                self._idle_since.pop(key, None)
                self._idle_emitted.discard(key)
        tracker.reset_camera(camera_id)


runtime = EnterpriseRuntime()
