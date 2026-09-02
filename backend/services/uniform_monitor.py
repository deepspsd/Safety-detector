"""
uniform_monitor.py -- Per-person Uniform compliance state machine & inference
=============================================================================
Uses Teachable Machine / ONNX image classifier on upper-body person crops.
Supports:
  • Multi-camera tracking & state machine with temporal smoothing
  • Single frame / person crop classification for live webcam & CCTV WS
  • Alert generation for sustained NO_UNIFORM violations

Pipeline per tracked person:
    1. Extract upper-body crop (top 65% of person bbox + 10% padding)
    2. Run uniform image classifier (via classifier_adapter)
    3. Push prediction to rolling window for temporal smoothing (majority vote)
    4. Transition state machine: COMPLIANT -> PENDING_MISSING -> ALERT_TRIGGERED
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import settings
from services.classifier_adapter import (
    ClassificationResult,
    ClassifierCache,
    TemporalSmoother,
    get_classifier,
    get_classifier_config,
)
from services.person_cropper import CropMode, crop_person

log = logging.getLogger("uniform_monitor")

# ── State constants ────────────────────────────────────────────────────────────
STATE_COMPLIANT = "COMPLIANT"
STATE_PENDING   = "PENDING_MISSING"
STATE_ALERT     = "ALERT_TRIGGERED"

# ── Prediction labels ──────────────────────────────────────────────────────────
PRED_UNIFORM    = "UNIFORM"
PRED_NO_UNIFORM = "NO_UNIFORM"
PRED_UNCERTAIN  = "UNCERTAIN"

RULE_MSG = "Not Wearing Proper Uniform"

# ── Tuning constants ──────────────────────────────────────────────────────────
_UPPER_BODY_RATIO = 0.65
_PADDING          = 0.10
_MIN_CROP_PX      = 32
_INFER_INTERVAL   = 1.0    # seconds between inferences per track
_MISSING_SECONDS  = 3.0    # seconds of sustained NO_UNIFORM before alert
_ALERT_COOLDOWN   = 30.0   # seconds between repeated alerts for same track


@dataclass
class PersonUniformStatus:
    camera_id: int
    track_id: str
    has_uniform: bool
    state: str
    missing_seconds: float
    alert_fired: bool
    confidence: float
    raw_prediction: str
    smoothed_prediction: str
    person_bbox: List[int]
    crop_bbox: Optional[List[int]] = None
    crop_width: int = 0
    crop_height: int = 0
    diagnostic_flag: str = "OK"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "has_uniform": self.has_uniform,
            "state": self.state,
            "missing_seconds": round(self.missing_seconds, 2),
            "alert_fired": self.alert_fired,
            "confidence": round(self.confidence, 4),
            "raw_prediction": self.raw_prediction,
            "smoothed_prediction": self.smoothed_prediction,
            "person_bbox": self.person_bbox,
            "crop_bbox": self.crop_bbox,
            "crop_width": self.crop_width,
            "crop_height": self.crop_height,
            "diagnostic_flag": self.diagnostic_flag,
        }


@dataclass
class _PersonState:
    track_id: str
    state: str = STATE_COMPLIANT
    has_uniform: bool = True
    raw_prediction: str = PRED_UNIFORM
    smoothed_prediction: str = PRED_UNIFORM
    confidence: float = 1.0
    missing_since: Optional[float] = None
    last_alert_at: Optional[float] = None
    last_infer_at: float = 0.0
    prediction_window: Deque[str] = field(default_factory=lambda: deque(maxlen=5))
    person_bbox: List[int] = field(default_factory=list)
    crop_bbox: List[int] = field(default_factory=list)
    crop_width: int = 0
    crop_height: int = 0
    diagnostic_flag: str = "OK"


class UniformMonitor:
    """Stateful uniform monitor managing compliance across all cameras and tracks."""

    def __init__(self):
        self._state: Dict[Tuple[int, str], _PersonState] = {}
        self._lock = threading.Lock()
        self._classifier = None

    def _get_clf(self):
        if self._classifier is None:
            self._classifier = get_classifier("uniform")
        return self._classifier

    def classify_crop(self, crop: np.ndarray) -> ClassificationResult:
        """Directly run inference on an image crop."""
        clf = self._get_clf()
        return clf.predict(crop)

    def classify_person_box(
        self,
        frame: np.ndarray,
        person_bbox: List[int],
    ) -> Tuple[ClassificationResult, Optional[np.ndarray]]:
        """Extract upper-body crop from frame and run uniform classifier."""
        if frame is None or not person_bbox or len(person_bbox) != 4:
            return (
                ClassificationResult(
                    predicted_class=PRED_UNCERTAIN,
                    confidence=0.0,
                    classifier_name="uniform",
                    model_loaded=False,
                ),
                None,
            )

        crop = crop_person(
            frame=frame,
            bbox=person_bbox,
            crop_mode=CropMode.UPPER_BODY,
            padding=_PADDING,
            upper_body_ratio=_UPPER_BODY_RATIO,
            target_size=(224, 224),
        )
        if crop is None or crop.shape[0] < _MIN_CROP_PX or crop.shape[1] < _MIN_CROP_PX:
            return (
                ClassificationResult(
                    predicted_class=PRED_UNCERTAIN,
                    confidence=0.0,
                    classifier_name="uniform",
                    model_loaded=False,
                ),
                crop,
            )

        res = self.classify_crop(crop)
        return res, crop

    def update_camera(
        self,
        camera_id: int,
        persons: List[dict],
        db: Any,
        frame: Optional[np.ndarray] = None,
    ) -> List[PersonUniformStatus]:
        """Update uniform compliance for all persons detected on a camera."""
        now = time.monotonic()
        wall_now = time.time()
        statuses: List[PersonUniformStatus] = []
        pending_alerts: List[PersonUniformStatus] = []
        active_keys: set = set()

        for person in persons:
            track_id = str(person.get("track_id", ""))
            if not track_id or track_id in ("-1", ""):
                # If no track_id, generate a temporary one based on bbox
                p_box = person.get("bbox", [])
                if not p_box or len(p_box) != 4:
                    continue
                track_id = f"temp_{p_box[0]}_{p_box[1]}"

            key = (camera_id, track_id)
            active_keys.add(key)
            p_box = person.get("bbox", [])
            if not p_box or len(p_box) != 4:
                continue

            with self._lock:
                ps = self._state.get(key)
                if ps is None:
                    ps = _PersonState(track_id=track_id)
                    self._state[key] = ps

            # Check if inference should run this tick
            do_infer = (
                frame is not None
                and (wall_now - ps.last_infer_at) >= _INFER_INTERVAL
            )

            if do_infer:
                clf_res, crop = self.classify_person_box(frame, p_box)
                pred_class = clf_res.predicted_class
                conf = clf_res.confidence

                # Normalize predicted class
                if pred_class in ("UNIFORM", "Uniform", "Uniform_1", "Uniform_2"):
                    raw_pred = PRED_UNIFORM
                elif pred_class in ("NO_UNIFORM", "No-Uniform", "No_Uniform", "NO-Uniform"):
                    raw_pred = PRED_NO_UNIFORM
                else:
                    raw_pred = PRED_UNCERTAIN

                diag = "OK" if clf_res.model_loaded else "MOCK_MODEL"
                if raw_pred == PRED_UNCERTAIN:
                    diag = "LOW_CONFIDENCE"

                with self._lock:
                    ps.last_infer_at = wall_now
                    ps.person_bbox = p_box
                    ps.confidence = conf
                    ps.raw_prediction = raw_pred
                    ps.diagnostic_flag = diag
                    if crop is not None:
                        ps.crop_width = crop.shape[1]
                        ps.crop_height = crop.shape[0]
                    if raw_pred != PRED_UNCERTAIN:
                        ps.prediction_window.append(raw_pred)
            else:
                with self._lock:
                    conf = ps.confidence
                    raw_pred = ps.raw_prediction
                    diag = ps.diagnostic_flag

            # Temporal smoothing (majority vote)
            with self._lock:
                window_list = list(ps.prediction_window)

            if window_list:
                uniform_votes = window_list.count(PRED_UNIFORM)
                no_uniform_votes = window_list.count(PRED_NO_UNIFORM)
                if uniform_votes > no_uniform_votes:
                    smoothed = PRED_UNIFORM
                elif no_uniform_votes > uniform_votes:
                    smoothed = PRED_NO_UNIFORM
                else:
                    smoothed = PRED_UNCERTAIN
            else:
                smoothed = PRED_UNIFORM  # default to compliant when uncertain

            has_uniform = (smoothed == PRED_UNIFORM)
            alert_fired = False

            with self._lock:
                ps.has_uniform = has_uniform
                ps.smoothed_prediction = smoothed
                ps.person_bbox = p_box

                if has_uniform or smoothed == PRED_UNCERTAIN:
                    ps.state = STATE_COMPLIANT
                    ps.missing_since = None
                else:
                    # smoothed == NO_UNIFORM
                    if ps.missing_since is None:
                        ps.missing_since = now
                        ps.state = STATE_PENDING
                        log.debug("[Uniform] cam=%d track=%s -> PENDING_MISSING", camera_id, track_id)

                    missing_secs = now - ps.missing_since
                    if missing_secs >= _MISSING_SECONDS:
                        ps.state = STATE_ALERT
                        since_last = (
                            (now - ps.last_alert_at)
                            if ps.last_alert_at is not None
                            else float("inf")
                        )
                        if since_last >= _ALERT_COOLDOWN:
                            ps.last_alert_at = now
                            alert_fired = True
                            log.warning(
                                "[Uniform] ALERT cam=%d track=%s missing=%.1fs",
                                camera_id, track_id, missing_secs,
                            )

                missing_seconds = (now - ps.missing_since) if ps.missing_since else 0.0
                state_snap = ps.state
                diag_snap = ps.diagnostic_flag
                cw_snap = ps.crop_width
                ch_snap = ps.crop_height

            st = PersonUniformStatus(
                camera_id=camera_id,
                track_id=track_id,
                has_uniform=has_uniform,
                state=state_snap,
                missing_seconds=missing_seconds,
                alert_fired=alert_fired,
                confidence=conf,
                raw_prediction=raw_pred,
                smoothed_prediction=smoothed,
                person_bbox=p_box,
                crop_width=cw_snap,
                crop_height=ch_snap,
                diagnostic_flag=diag_snap,
            )
            statuses.append(st)
            if alert_fired:
                pending_alerts.append(st)

        # Fire alerts outside the lock
        for st in pending_alerts:
            self._fire_alert(
                camera_id=camera_id,
                track_id=st.track_id,
                missing_seconds=st.missing_seconds,
                db=db,
                frame=frame,
            )

        self._evict_stale(camera_id, active_keys)
        return statuses

    def get_debug_info(self, camera_id: int) -> List[dict]:
        """Snapshot of per-person uniform state for debug overlays."""
        now = time.monotonic()
        with self._lock:
            return [
                {
                    "track_id": ps.track_id,
                    "state": ps.state,
                    "has_uniform": ps.has_uniform,
                    "confidence": round(ps.confidence, 3),
                    "raw_prediction": ps.raw_prediction,
                    "smoothed_prediction": ps.smoothed_prediction,
                    "missing_seconds": (
                        round(now - ps.missing_since, 2)
                        if ps.missing_since else 0.0
                    ),
                    "person_bbox": ps.person_bbox,
                    "crop_width": ps.crop_width,
                    "crop_height": ps.crop_height,
                    "diagnostic_flag": ps.diagnostic_flag,
                    "window": list(ps.prediction_window),
                }
                for (cam_id, _), ps in self._state.items()
                if cam_id == camera_id
            ]

    def reset_camera(self, camera_id: int) -> None:
        with self._lock:
            for key in [k for k in self._state if k[0] == camera_id]:
                del self._state[key]
        log.info("[Uniform] State reset cam=%d", camera_id)

    def _evict_stale(self, camera_id: int, active_keys: set) -> None:
        with self._lock:
            for k in [k for k in self._state if k[0] == camera_id and k not in active_keys]:
                del self._state[k]

    @staticmethod
    def _fire_alert(
        camera_id: int,
        track_id: str,
        missing_seconds: float,
        db: Any,
        frame: Optional[np.ndarray],
    ) -> None:
        if db is None:
            return
        try:
            import base64
            from services.alert_service import save_alert
            from services.rule_engine import _get_rule_engine_user_id

            uid = _get_rule_engine_user_id(db)

            snapshot_b64 = None
            if frame is not None:
                try:
                    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    snapshot_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")
                except Exception:
                    pass

            msg = f"⚠️ Person (Track #{track_id}) is not wearing required uniform ({missing_seconds:.1f}s)"
            save_alert(
                db=db,
                user_id=uid,
                message=msg,
                role="Factory Worker",
                severity="high",
                detected_issue="Not Wearing Uniform",
                confidence=0.85,
                snapshot_b64=snapshot_b64,
                camera_id=camera_id,
            )
            log.info("[Uniform] Saved alert to DB for cam=%d track=%s", camera_id, track_id)
        except Exception as exc:
            log.error("[Uniform] Failed to fire alert: %s", exc)


# Global singleton instance
uniform_monitor = UniformMonitor()
