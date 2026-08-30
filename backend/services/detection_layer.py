"""Detection layer: converts model output to neutral object observations only.

No alert, workflow, zone, OCR or policy decision belongs here.  The module is
intentionally a small adapter over the already-loaded YOLO model so existing
weights and startup behaviour remain unchanged.

Architecture
────────────
AbstractDetector      → base class / interface (swap models without changing callers)
YoloDetectionAdapter  → wraps the live YOLO model
get_detector()        → factory that returns the configured implementation
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Enriched Detection dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Detection:
    """Normalised single-object detection result.

    bbox is [x1, y1, x2, y2] in pixel coordinates.
    Derived properties (center_x, center_y, width, height) are computed lazily.
    """

    label: str
    confidence: float
    bbox: List[int]          # [x1, y1, x2, y2]
    model_key: str = "yolo-ppe"
    class_id: int = -1       # raw class index from the model

    @property
    def x1(self) -> int:
        return self.bbox[0]

    @property
    def y1(self) -> int:
        return self.bbox[1]

    @property
    def x2(self) -> int:
        return self.bbox[2]

    @property
    def y2(self) -> int:
        return self.bbox[3]

    @property
    def center_x(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2.0

    @property
    def center_y(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2.0

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    def to_dict(self) -> Dict:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "model_key": self.model_key,
            "class_id": self.class_id,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "width": self.width,
            "height": self.height,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Abstract interface — swap implementations without touching business logic
# ─────────────────────────────────────────────────────────────────────────────


class AbstractDetector(ABC):
    """All detector implementations must satisfy this interface."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on *frame* and return Detection objects."""

    def is_healthy(self) -> bool:
        """Return True if the underlying model is loaded and ready."""
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Real adapter — wraps the already-loaded YOLO model
# ─────────────────────────────────────────────────────────────────────────────


class YoloDetectionAdapter(AbstractDetector):
    """Read raw classes from the current YOLO model without policy enrichment."""

    def detect(self, frame: np.ndarray) -> List[Detection]:
        from services import yolo_service

        model = getattr(yolo_service, "_model", None)
        if model is None:
            return []
        try:
            results = model(
                frame,
                verbose=False,
                conf=yolo_service.settings.DETECTION_CONF,
                iou=yolo_service.settings.NMS_IOU,
            )
        except Exception:
            return []

        output: List[Detection] = []
        for result in results:
            for box in result.boxes:
                index = int(box.cls[0])
                if getattr(yolo_service, "_model_is_ppe", False):
                    labels = getattr(yolo_service, "PPE_CLASS_NAMES", [])
                    label = labels[index] if index < len(labels) else f"class_{index}"
                else:
                    label = str(model.names[index])
                output.append(
                    Detection(
                        label=label,
                        confidence=round(float(box.conf[0]), 4),
                        bbox=[int(v) for v in box.xyxy[0]],
                        class_id=index,
                    )
                )
        return output

    def is_healthy(self) -> bool:
        from services import yolo_service
        return getattr(yolo_service, "_model", None) is not None


# ─────────────────────────────────────────────────────────────────────────────
# Factory — returns the singleton detector
# ─────────────────────────────────────────────────────────────────────────────

_detector_instance: Optional[AbstractDetector] = None


def get_detector() -> AbstractDetector:
    """Return the singleton detector instance.

    This factory exists so future adapters (ONNX, TFLite, Triton) can be
    plugged in here without touching calling code.
    """
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = YoloDetectionAdapter()
    return _detector_instance


# ── Module-level singleton for backward compatibility ─────────────────────────
detector = YoloDetectionAdapter()
