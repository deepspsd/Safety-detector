"""Image Classifier Adapter
===========================
Generic adapter layer between person crops and image classification models
(Teachable Machine, ONNX, TFLite, or future runtimes).

Architecture
────────────
ClassificationResult  — output dataclass
AbstractClassifier    — interface (predict(image) → ClassificationResult)
MockClassifier        — deterministic placeholder (no model file required)
TeachableMachineAdapter — stub; loads a real TF SavedModel / ONNX when present
ClassifierCache       — per-(track_id, classifier_name) inference rate limiter
TemporalSmoother      — rolling window majority vote per track

Usage
─────
    from services.classifier_adapter import get_classifier, ClassifierCache, TemporalSmoother

    cache   = ClassifierCache()
    smoother = TemporalSmoother()
    clf = get_classifier("uniform")

    if cache.should_run(track_id="17", classifier_name="uniform"):
        crop = crop_person(frame, bbox, crop_mode="upper_body")
        if crop is not None:
            result = clf.predict(crop)
            cache.record(track_id="17", classifier_name="uniform")
            smoother.push(track_id="17", classifier_name="uniform", result=result)
            smoothed = smoother.vote(track_id="17", classifier_name="uniform")
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from config import settings

log = logging.getLogger("classifier_adapter")


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ClassificationResult:
    predicted_class: str
    confidence: float
    all_class_probabilities: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    classifier_name: str = ""
    model_loaded: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_class": self.predicted_class,
            "confidence": round(self.confidence, 4),
            "all_class_probabilities": {
                k: round(v, 4) for k, v in self.all_class_probabilities.items()
            },
            "timestamp": self.timestamp.isoformat(),
            "classifier_name": self.classifier_name,
            "model_loaded": self.model_loaded,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Abstract interface
# ─────────────────────────────────────────────────────────────────────────────


class AbstractClassifier(ABC):
    """All classifier implementations must satisfy this interface."""

    name: str = "abstract"
    classes: List[str] = []
    confidence_threshold: float = 0.80

    @abstractmethod
    def predict(self, image: np.ndarray) -> ClassificationResult:
        """Run inference on the pre-cropped *image* (BGR numpy array).

        Returns:
            ClassificationResult — never raises; returns a fallback result on error.
        """

    def is_loaded(self) -> bool:
        """Return True when the model is ready for inference."""
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Mock classifier — works without any model file
# ─────────────────────────────────────────────────────────────────────────────


class MockClassifier(AbstractClassifier):
    """Returns plausible but random predictions.  Used when no model file exists.

    The probability distribution is slightly biased toward the first class so
    that demo dashboards look realistic rather than always 50/50.
    """

    def __init__(self, name: str, classes: List[str], confidence_threshold: float = 0.80):
        self.name = name
        self.classes = classes if classes else ["UNKNOWN"]
        self.confidence_threshold = confidence_threshold
        log.info(
            "MockClassifier '%s' active (classes=%s) — load a real model to replace",
            name,
            classes,
        )

    def predict(self, image: np.ndarray) -> ClassificationResult:
        import random

        n = len(self.classes)
        # Bias toward class[0] (compliant) so the UI shows varied results
        weights = [0.65] + [0.35 / max(n - 1, 1)] * (n - 1)
        probs_raw = [w + random.uniform(-0.1, 0.1) for w in weights]
        total = sum(max(p, 0) for p in probs_raw) or 1.0
        probs = [max(p, 0) / total for p in probs_raw]

        best_idx = probs.index(max(probs))
        return ClassificationResult(
            predicted_class=self.classes[best_idx],
            confidence=round(probs[best_idx], 4),
            all_class_probabilities={cls: round(p, 4) for cls, p in zip(self.classes, probs)},
            classifier_name=self.name,
            model_loaded=False,  # signals this is a mock result
        )

    def is_loaded(self) -> bool:
        return False  # honest: no real model


# ─────────────────────────────────────────────────────────────────────────────
# Teachable Machine / ONNX adapter stub
# ─────────────────────────────────────────────────────────────────────────────


class TeachableMachineAdapter(AbstractClassifier):
    """Loads an exported TensorFlow SavedModel or ONNX image classifier.

    Place the exported model directory at the path configured in config/models.yaml.
    If the path does not exist, this falls back to MockClassifier behaviour and
    logs a clear warning — the rest of the system continues running.

    Supported export formats (detected automatically):
      • TensorFlow SavedModel directory (keras_metadata.pb present)
      • ONNX single-file model (.onnx extension)

    The interface is identical regardless of the backend runtime, so switching
    from TensorFlow to ONNX Runtime only requires a config change.
    """

    def __init__(
        self,
        name: str,
        model_path: str,
        classes: List[str],
        input_width: int = 224,
        input_height: int = 224,
        confidence_threshold: float = 0.80,
    ):
        self.name = name
        self.classes = classes if classes else ["UNKNOWN"]
        self.confidence_threshold = confidence_threshold
        self._input_size = (input_width, input_height)
        self._model = None
        self._backend: str = "none"
        self._load(model_path)

    def _load(self, model_path: str) -> None:
        import os

        if not model_path or not os.path.exists(model_path):
            log.warning(
                "TeachableMachineAdapter '%s': model not found at '%s'. "
                "Running as MockClassifier until model is placed at that path.",
                self.name,
                model_path,
            )
            return

        if model_path.endswith(".onnx"):
            self._load_onnx(model_path)
        else:
            self._load_tf(model_path)

    def _load_tf(self, path: str) -> None:
        try:
            import tensorflow as tf  # type: ignore

            self._model = tf.saved_model.load(path)
            self._backend = "tensorflow"
            log.info("TeachableMachineAdapter '%s': loaded TF SavedModel from %s", self.name, path)
        except Exception as exc:
            log.warning(
                "TeachableMachineAdapter '%s': TF load failed (%s) — using mock",
                self.name,
                exc,
            )

    def _load_onnx(self, path: str) -> None:
        try:
            import onnxruntime as ort  # type: ignore

            self._model = ort.InferenceSession(path)
            self._backend = "onnx"
            log.info("TeachableMachineAdapter '%s': loaded ONNX from %s", self.name, path)
        except Exception as exc:
            log.warning(
                "TeachableMachineAdapter '%s': ONNX load failed (%s) — using mock",
                self.name,
                exc,
            )

    def predict(self, image: np.ndarray) -> ClassificationResult:
        if self._model is None:
            # Fallback to mock output
            return MockClassifier(self.name, self.classes, self.confidence_threshold).predict(image)

        try:
            import cv2

            inp = cv2.resize(image, self._input_size).astype(np.float32) / 255.0
            inp = np.expand_dims(inp, 0)

            if self._backend == "tensorflow":
                probs = list(self._model(inp)[0].numpy())
            else:  # onnx
                session = self._model
                input_name = session.get_inputs()[0].name
                probs = list(session.run(None, {input_name: inp})[0][0])

            total = sum(probs) or 1.0
            probs = [p / total for p in probs]
            best_idx = int(np.argmax(probs))
            n = min(len(self.classes), len(probs))
            return ClassificationResult(
                predicted_class=self.classes[best_idx] if best_idx < len(self.classes) else "UNKNOWN",
                confidence=round(float(probs[best_idx]), 4),
                all_class_probabilities={
                    self.classes[i]: round(float(probs[i]), 4) for i in range(n)
                },
                classifier_name=self.name,
                model_loaded=True,
            )
        except Exception as exc:
            log.warning("TeachableMachineAdapter '%s' predict error: %s", self.name, exc)
            return ClassificationResult(
                predicted_class="UNKNOWN",
                confidence=0.0,
                classifier_name=self.name,
                model_loaded=True,
            )

    def is_loaded(self) -> bool:
        return self._model is not None


# ─────────────────────────────────────────────────────────────────────────────
# Per-track inference rate limiter
# ─────────────────────────────────────────────────────────────────────────────


class ClassifierCache:
    """Prevents running the classifier on every frame.

    Default interval: settings.CLASSIFIER_INTERVAL_MS milliseconds per
    (track_id, classifier_name) pair.
    """

    def __init__(self, interval_ms: Optional[int] = None):
        self._interval_s = (interval_ms or settings.CLASSIFIER_INTERVAL_MS) / 1000.0
        self._last_run: Dict[Tuple[str, str], float] = {}

    def should_run(self, track_id: str, classifier_name: str) -> bool:
        key = (track_id, classifier_name)
        last = self._last_run.get(key, 0.0)
        return (time.monotonic() - last) >= self._interval_s

    def record(self, track_id: str, classifier_name: str) -> None:
        self._last_run[(track_id, classifier_name)] = time.monotonic()

    def evict_old_tracks(self, active_track_ids: List[str]) -> None:
        """Remove cache entries for tracks that are no longer active."""
        active = set(active_track_ids)
        stale = [k for k in self._last_run if k[0] not in active]
        for k in stale:
            self._last_run.pop(k, None)


# ─────────────────────────────────────────────────────────────────────────────
# Temporal smoother — majority vote over a rolling window
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _SmoothEntry:
    predicted_class: str
    confidence: float
    ts: float = field(default_factory=time.monotonic)


class TemporalSmoother:
    """Maintains a rolling prediction history per (track_id, classifier_name).

    Smoothing prevents single-frame noise from triggering alerts.
    The vote method returns the majority class if it exceeds the quorum fraction.

    Args:
        window:         Number of recent predictions to keep (default from settings).
        quorum_fraction: Fraction of window votes needed for a decision (default 0.6).
    """

    def __init__(
        self,
        window: Optional[int] = None,
        quorum_fraction: float = 0.6,
    ):
        self._window = window or settings.CLASSIFIER_SMOOTHING_WINDOW
        self._quorum = quorum_fraction
        self._history: Dict[Tuple[str, str], Deque[_SmoothEntry]] = {}

    def push(self, track_id: str, classifier_name: str, result: ClassificationResult) -> None:
        key = (track_id, classifier_name)
        dq = self._history.setdefault(key, deque(maxlen=self._window))
        dq.append(_SmoothEntry(result.predicted_class, result.confidence))

    def vote(self, track_id: str, classifier_name: str) -> Optional[ClassificationResult]:
        """Return smoothed result via majority vote, or None if window is too small."""
        key = (track_id, classifier_name)
        dq = self._history.get(key)
        if not dq or len(dq) < max(1, self._window // 2):
            return None  # not enough history yet

        counts: Dict[str, int] = {}
        conf_sum: Dict[str, float] = {}
        for entry in dq:
            counts[entry.predicted_class] = counts.get(entry.predicted_class, 0) + 1
            conf_sum[entry.predicted_class] = conf_sum.get(entry.predicted_class, 0.0) + entry.confidence

        total = sum(counts.values())
        winner = max(counts, key=lambda c: counts[c])
        winner_fraction = counts[winner] / total

        if winner_fraction < self._quorum:
            return None  # no clear winner yet

        avg_conf = conf_sum[winner] / counts[winner]
        return ClassificationResult(
            predicted_class=winner,
            confidence=round(avg_conf, 4),
            all_class_probabilities={
                cls: round(cnt / total, 4) for cls, cnt in counts.items()
            },
            classifier_name=classifier_name,
            model_loaded=True,
        )

    def evict_old_tracks(self, active_track_ids: List[str]) -> None:
        """Remove history entries for tracks that are no longer active."""
        active = set(active_track_ids)
        stale = [k for k in self._history if k[0] not in active]
        for k in stale:
            self._history.pop(k, None)


# ─────────────────────────────────────────────────────────────────────────────
# Built-in classifier configs (no model files assumed yet)
# ─────────────────────────────────────────────────────────────────────────────

_BUILTIN_CONFIGS: Dict[str, Dict] = {
    "uniform": {
        "classes": ["UNIFORM", "NO_UNIFORM"],
        "crop_mode": "upper_body",
        "confidence_threshold": 0.80,
        "model_path": "",
    },
    "head_cap": {
        "classes": ["HEAD_CAP", "NO_HEAD_CAP"],
        "crop_mode": "head",
        "confidence_threshold": 0.80,
        "model_path": "",
    },
    "bangle": {
        "classes": ["BANGLE", "NO_BANGLE"],
        "crop_mode": "upper_body",
        "confidence_threshold": 0.75,
        "model_path": "",
    },
}


def get_classifier(classifier_name: str) -> AbstractClassifier:
    """Return an AbstractClassifier for the given name.

    If a real model file is configured and exists, a TeachableMachineAdapter is
    returned.  Otherwise a MockClassifier is returned so the rest of the
    pipeline continues operating without real model weights.
    """
    cfg = _BUILTIN_CONFIGS.get(classifier_name, {})
    classes = cfg.get("classes", ["POSITIVE", "NEGATIVE"])
    threshold = cfg.get("confidence_threshold", 0.80)
    model_path = cfg.get("model_path", "")

    import os

    if model_path and os.path.exists(model_path):
        return TeachableMachineAdapter(
            name=classifier_name,
            model_path=model_path,
            classes=classes,
            confidence_threshold=threshold,
        )

    return MockClassifier(
        name=classifier_name,
        classes=classes,
        confidence_threshold=threshold,
    )
