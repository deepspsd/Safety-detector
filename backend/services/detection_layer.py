"""Detection layer: converts model output to neutral object observations only.

No alert, workflow, zone, OCR or policy decision belongs here.  The module is
intentionally a small adapter over the already-loaded YOLO model so existing
weights and startup behaviour remain unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import numpy as np


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: List[int]
    model_key: str = "yolo-ppe"

    def to_dict(self) -> Dict:
        return asdict(self)


class YoloDetectionAdapter:
    """Read raw classes from the current YOLO model without policy enrichment."""

    def detect(self, frame: np.ndarray) -> List[Detection]:
        from services import yolo_service
        model = getattr(yolo_service, "_model", None)
        if model is None:
            return []
        results = model(
            frame,
            verbose=False,
            conf=yolo_service.settings.DETECTION_CONF,
            iou=yolo_service.settings.NMS_IOU,
        )
        output: List[Detection] = []
        for result in results:
            for box in result.boxes:
                index = int(box.cls[0])
                if getattr(yolo_service, "_model_is_ppe", False):
                    labels = getattr(yolo_service, "PPE_CLASS_NAMES", [])
                    label = labels[index] if index < len(labels) else f"class_{index}"
                else:
                    label = str(model.names[index])
                output.append(Detection(
                    label=label,
                    confidence=round(float(box.conf[0]), 4),
                    bbox=[int(v) for v in box.xyxy[0]],
                ))
        return output


detector = YoloDetectionAdapter()
