"""
Multi-Model Detection Layer
===========================

Orchestrates multiple AI models to process video frames based on zone spatial routing.

Key Features:
- Zone-aware model routing (dough_mixing → hairnet + fall; cashbox → banknote detection)
- Spatial ROI cropping (crops polygon bounding box, translates coords back)
- Point-in-polygon verification for zone-specific models
- Frame rate throttling per model to prevent compute saturation
- Normalized detection taxonomy (replaces hardcoded legacy classes)
- Detection fusion to merge multi-model outputs
- Zero crashes: isolated model errors
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import settings
from services.detection_layer import Detection
from services.model_manager import ModelManager
from services.zone_service import (
    bbox_center,
    load_zone_details_for_camera,
    load_zone_details_without_db,
    point_in_zone,
)

logger = logging.getLogger("multi_model_detector")


# ─────────────────────────────────────────────────────────────────────────────
# Class Label Normalization
# ─────────────────────────────────────────────────────────────────────────────

LABEL_NORMALIZATION: Dict[str, Tuple[Optional[str], str]] = {
    # (standard_label, det_type)
    # Negative classes to ignore completely
    "non-fall": (None, "ignore"),
    "non_fall": (None, "ignore"),
    # Fall detection
    "fall": ("Worker Fall", "violation"),
    "Fall": ("Worker Fall", "violation"),
    "Worker Fall": ("Worker Fall", "violation"),
    # Hairnet / Head cover
    "Hair_Cover": ("Bakery-Head-Cap", "compliant"),
    "hair_cover": ("Bakery-Head-Cap", "compliant"),
    "Hairnet": ("Bakery-Head-Cap", "compliant"),
    "hairnet": ("Bakery-Head-Cap", "compliant"),
    "Hair": ("NO-Bakery-Head-Cap", "violation"),
    "NO-Hairnet": ("NO-Bakery-Head-Cap", "violation"),
    "no_hairnet": ("NO-Bakery-Head-Cap", "violation"),
    "NO-Bakery-Head-Cap": ("NO-Bakery-Head-Cap", "violation"),
    "Bakery-Head-Cap": ("Bakery-Head-Cap", "compliant"),
    # Gloves (disabled per client requirements)
    "Hand_Gloves": (None, "ignore"),
    "hand_gloves": (None, "ignore"),
    "Gloves": (None, "ignore"),
    "gloves": (None, "ignore"),
    "Glove": (None, "ignore"),
    "Front_Palm": (None, "ignore"),
    "Back_Palm": (None, "ignore"),
    "NO-Gloves": (None, "ignore"),
    "no_gloves": (None, "ignore"),
    # Hardhat / Helmet
    "Hardhat": ("Hardhat", "compliant"),
    "hardhat": ("Hardhat", "compliant"),
    "helmet": ("Hardhat", "compliant"),
    "Helmet": ("Hardhat", "compliant"),
    "NO-Hardhat": ("NO-Hardhat", "violation"),
    "no_hardhat": ("NO-Hardhat", "violation"),
    "No Helmet": ("NO-Hardhat", "violation"),
    # Mask & Vest (masks disabled per client requirements)
    "Mask": (None, "ignore"),
    "mask": (None, "ignore"),
    "NO-Mask": (None, "ignore"),
    "no_mask": (None, "ignore"),
    "Safety Vest": ("Safety Vest", "compliant"),
    "NO-Safety Vest": ("NO-Safety Vest", "violation"),
    # Person
    "person": ("Person", "person"),
    "Person": ("Person", "person"),
    # Phone
    "cell phone": ("cell phone", "neutral"),
    "phone": ("cell phone", "neutral"),
    # Jewellery
    "Bangles": ("Bangles", "violation"),
    "bangles": ("Bangles", "violation"),
    # Anomaly
    "Machine_Sensor_Anomaly": ("Machine Anomaly", "violation"),
    "Machine_Visual_Anomaly": ("Machine Anomaly", "violation"),
    "Machine Anomaly": ("Machine Anomaly", "violation"),
    # Object Throwing
    "Object_Throwing": ("Object Throwing", "violation"),
    "Object Throwing": ("Object Throwing", "violation"),
    # Fight / Physical Altercation / Rage
    "fight": ("Physical Altercation", "violation"),
    "fighting": ("Physical Altercation", "violation"),
    "Fight": ("Physical Altercation", "violation"),
    "Fighting": ("Physical Altercation", "violation"),
    "Physical Altercation": ("Physical Altercation", "violation"),
    # Uniform compliance
    "uniform_1": ("Uniform", "compliant"),
    "Uniform_1": ("Uniform", "compliant"),
    "Uniform_2": ("Uniform", "compliant"),
    "uniform_2": ("Uniform", "compliant"),
    "Uniform": ("Uniform", "compliant"),
    "uniform": ("Uniform", "compliant"),
    "No_uniform": ("NO-Uniform", "violation"),
    "no_uniform": ("NO-Uniform", "violation"),
    "NO-Uniform": ("NO-Uniform", "violation"),
    "non_uniform": ("NO-Uniform", "violation"),
}


def normalize_class_label(raw_label: str) -> Tuple[Optional[str], str]:
    """Map raw model class name to standard label and detection type."""
    if not raw_label:
        return None, "ignore"

    if raw_label in LABEL_NORMALIZATION:
        return LABEL_NORMALIZATION[raw_label]

    lbl_lower = raw_label.lower().strip()

    # Fight / Altercation check
    if "fight" in lbl_lower or "altercation" in lbl_lower or "aggression" in lbl_lower:
        return "Physical Altercation", "violation"

    # Cash banknotes pattern (e.g. "10 BGN", "50 EUR", "100 INR", "₹500")
    if (
        "cash" in lbl_lower
        or "banknote" in lbl_lower
        or "rupee" in lbl_lower
        or "inr" in lbl_lower
        or "₹" in raw_label
        or lbl_lower.startswith("rs")
        or raw_label.endswith(" BGN")
        or raw_label.endswith(" EUR")
        or raw_label.endswith(" INR")
    ):
        return "Cash", "neutral"

    if "fall" in lbl_lower and "non" not in lbl_lower:
        return "Worker Fall", "violation"

    return raw_label, "neutral"


# ─────────────────────────────────────────────────────────────────────────────
# Detection Fusion
# ─────────────────────────────────────────────────────────────────────────────


def compute_iou(bbox1: List[int], bbox2: List[int]) -> float:
    """Compute Intersection over Union between two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i < x1_i or y2_i < y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def fuse_detections(detections: List[Detection]) -> List[Detection]:
    """Merge overlapping detections from multiple models."""
    if not getattr(settings, "ENABLE_DETECTION_FUSION", True) or len(detections) <= 1:
        return detections

    fused: List[Detection] = []
    used_indices = set()
    iou_thresh = getattr(settings, "FUSION_IOU_THRESHOLD", 0.50)

    for i, det1 in enumerate(detections):
        if i in used_indices:
            continue

        group = [det1]
        group_indices = [i]

        for j, det2 in enumerate(detections):
            if j <= i or j in used_indices:
                continue

            iou = compute_iou(det1.bbox, det2.bbox)
            is_head1 = det1.label in ("Bakery-Head-Cap", "NO-Bakery-Head-Cap", "Hairnet", "NO-Hairnet") or getattr(det1, "raw_label", "") in ("hairnet", "no_hairnet")
            is_head2 = det2.label in ("Bakery-Head-Cap", "NO-Bakery-Head-Cap", "Hairnet", "NO-Hairnet") or getattr(det2, "raw_label", "") in ("hairnet", "no_hairnet")

            if (is_head1 and is_head2 and iou > 0.30) or (iou > iou_thresh and (
                det1.label == det2.label
                or (det1.det_type == det2.det_type and det1.det_type == "violation")
            )):
                group.append(det2)
                group_indices.append(j)

        for idx in group_indices:
            used_indices.add(idx)

        best_det = max(group, key=lambda d: d.confidence)
        fused.append(best_det)

    return fused


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Model Detector Class
# ─────────────────────────────────────────────────────────────────────────────


class MultiModelDetector:
    """Orchestrates multi-model inference with spatial zone gating and throttling."""

    def __init__(self):
        self.manager = ModelManager()
        self.model_last_run: Dict[str, float] = {}
        logger.info("✅ MultiModelDetector initialized with ModelManager")

    def _should_run_model(self, model_key: str, target_fps: float) -> bool:
        """Check if model should run based on target FPS throttling."""
        if target_fps <= 0:
            return True
        last_run = self.model_last_run.get(model_key, 0.0)
        min_interval = 1.0 / target_fps
        return (time.time() - last_run) >= min_interval

    def detect(
        self,
        frame: np.ndarray,
        zone_type: Optional[str] = None,
        camera_id: Optional[int] = None,
        zones: Optional[Dict[str, Any]] = None,
        enabled_models: Optional[List[str]] = None,
        db=None,
    ) -> List[Detection]:
        """
        Run multi-model inference on frame.
        - Global models run on full frame.
        - Zone models run on cropped ROIs for precision and speed.
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        all_detections: List[Detection] = []

        # 1. Resolve camera zones if available
        cam_zones = zones
        if cam_zones is None and camera_id is not None:
            cam_zones = load_zone_details_without_db(camera_id)
            if cam_zones is None and db is not None:
                cam_zones = load_zone_details_for_camera(camera_id, db)

        # 2. Determine global vs zone models
        global_models: List[str] = []
        if enabled_models:
            global_models = list(enabled_models)
        else:
            # Primary person + object detector
            primary_key = getattr(settings, "PRIMARY_PERSON_MODEL", "yolov8x_coco")
            if self.manager.registry.is_model_enabled(primary_key):
                global_models.append(primary_key)
            elif self.manager.registry.is_model_enabled("yolov8x_coco"):
                global_models.append("yolov8x_coco")

            # Fall detection runs globally (full frame)
            if self.manager.registry.is_model_enabled("fall_detection"):
                global_models.append("fall_detection")

            # Hairnet detection runs globally on full frame at imgsz=960
            if self.manager.registry.is_model_enabled("hairnet_glove_detection"):
                global_models.append("hairnet_glove_detection")

        # 3. Run Global Models on Full Frame
        for m_key in global_models:
            m_info = self.manager.registry.get_model(m_key)
            target_fps = m_info.target_fps if m_info else 5.0
            if not self._should_run_model(m_key, target_fps):
                continue

            try:
                norm_dets = self.manager.infer(
                    m_key, frame, camera_id=camera_id
                )
                self.model_last_run[m_key] = time.time()

                for nd in norm_dets:
                    norm_lbl, d_type = normalize_class_label(nd.class_name)
                    if not norm_lbl or d_type == "ignore":
                        continue

                    det = Detection(
                        label=norm_lbl,
                        confidence=round(nd.confidence, 3),
                        bbox=nd.bbox,
                        class_id=nd.class_id,
                        model_key=m_key,
                        det_type=d_type,
                        raw_label=nd.class_name,
                        zone_id=nd.zone_id,
                        camera_id=camera_id,
                    )
                    all_detections.append(det)
            except Exception as exc:
                logger.error(f"Error running global model '{m_key}': {exc}")

        # 4. Run Zone-Specific Models (ROI Cropping + Spatial Gating)
        if cam_zones and isinstance(cam_zones, dict):
            for z_name, z_data in cam_zones.items():
                if not isinstance(z_data, dict):
                    continue

                z_id = z_data.get("id")
                z_poly = z_data.get("polygon")
                caps = z_data.get("capabilities") or []
                z_models = z_data.get("zone_models") or []

                # If no explicit capabilities, infer from zone_name / zone_type
                if not caps and not z_models:
                    eff_type = z_data.get("zone_type") or z_name
                    caps = settings.ZONE_CAPABILITY_MAP.get(eff_type, [])

                # Resolve models for this zone
                target_models = set(self.manager.resolve_capabilities(caps) + z_models)
                # Remove models already run globally
                target_models.difference_update(global_models)

                if not target_models:
                    continue

                # Compute ROI bounding box for polygon
                crop_bbox = None
                crop_frame = frame
                offset_x, offset_y = 0, 0

                if z_poly and len(z_poly) >= 3:
                    xs = [p[0] for p in z_poly]
                    ys = [p[1] for p in z_poly]
                    pad = 15
                    x1 = max(0, min(xs) - pad)
                    y1 = max(0, min(ys) - pad)
                    x2 = min(w, max(xs) + pad)
                    y2 = min(h, max(ys) + pad)

                    # Only crop if region is reasonable size
                    if (x2 - x1) >= 30 and (y2 - y1) >= 30:
                        crop_bbox = [x1, y1, x2, y2]
                        crop_frame = frame[y1:y2, x1:x2]
                        offset_x, offset_y = x1, y1

                # Execute zone models
                for zm_key in target_models:
                    m_info = self.manager.registry.get_model(zm_key)
                    target_fps = m_info.target_fps if m_info else 5.0
                    if not self._should_run_model(zm_key, target_fps):
                        continue

                    try:
                        zone_dets = self.manager.infer(
                            zm_key,
                            crop_frame,
                            zone_id=z_id,
                            camera_id=camera_id,
                        )
                        self.model_last_run[zm_key] = time.time()

                        for nd in zone_dets:
                            norm_lbl, d_type = normalize_class_label(nd.class_name)
                            if not norm_lbl or d_type == "ignore":
                                continue

                            # Translate crop coords to full frame coords
                            bx1 = nd.bbox[0] + offset_x
                            by1 = nd.bbox[1] + offset_y
                            bx2 = nd.bbox[2] + offset_x
                            by2 = nd.bbox[3] + offset_y
                            translated_bbox = [bx1, by1, bx2, by2]

                            # Spatial gating: ensure detection is within polygon
                            if z_poly and len(z_poly) >= 3:
                                cx, cy = bbox_center(translated_bbox)
                                if not point_in_zone(cx, cy, z_poly):
                                    continue

                            det = Detection(
                                label=norm_lbl,
                                confidence=round(nd.confidence, 3),
                                bbox=translated_bbox,
                                class_id=nd.class_id,
                                model_key=zm_key,
                                det_type=d_type,
                                raw_label=nd.class_name,
                                zone_id=z_id,
                                camera_id=camera_id,
                            )
                            all_detections.append(det)
                    except Exception as zm_exc:
                        logger.error(f"Error running zone model '{zm_key}' for zone '{z_name}': {zm_exc}")

        # 5. Fallback for zone_type string without polygon zones
        elif zone_type:
            eff_zone = zone_type if zone_type in settings.ZONE_CAPABILITY_MAP else "default"
            zone_caps = settings.ZONE_CAPABILITY_MAP.get(eff_zone, [])
            models_for_zone = self.manager.resolve_capabilities(zone_caps)
            for zm_key in models_for_zone:
                if zm_key in global_models:
                    continue
                m_info = self.manager.registry.get_model(zm_key)
                target_fps = m_info.target_fps if m_info else 5.0
                if not self._should_run_model(zm_key, target_fps):
                    continue
                try:
                    norm_dets = self.manager.infer(zm_key, frame, camera_id=camera_id)
                    self.model_last_run[zm_key] = time.time()
                    for nd in norm_dets:
                        norm_lbl, d_type = normalize_class_label(nd.class_name)
                        if not norm_lbl or d_type == "ignore":
                            continue
                        all_detections.append(
                            Detection(
                                label=norm_lbl,
                                confidence=round(nd.confidence, 3),
                                bbox=nd.bbox,
                                class_id=nd.class_id,
                                model_key=zm_key,
                                det_type=d_type,
                                raw_label=nd.class_name,
                                camera_id=camera_id,
                            )
                        )
                except Exception as exc:
                    logger.error(f"Error running zone_type model '{zm_key}': {exc}")

        # 6. Detection Fusion
        if len(all_detections) > 1:
            all_detections = fuse_detections(all_detections)

        return all_detections

    def get_stats(self) -> Dict[str, Any]:
        """Telemetry telemetry across all models."""
        return self.manager.get_health()


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton & Adapter
# ─────────────────────────────────────────────────────────────────────────────

_detector_instance: Optional[MultiModelDetector] = None


def get_multi_model_detector() -> MultiModelDetector:
    """Get the singleton MultiModelDetector instance."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = MultiModelDetector()
    return _detector_instance


class MultiModelDetectionAdapter:
    """Adapter matching AbstractDetector interface."""

    def __init__(self):
        self.detector = get_multi_model_detector()

    def detect(self, frame: np.ndarray, zone_type: Optional[str] = None) -> List[Detection]:
        return self.detector.detect(frame, zone_type=zone_type)

    def is_healthy(self) -> bool:
        return True


detector = MultiModelDetectionAdapter()
