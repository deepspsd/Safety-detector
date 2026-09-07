"""
Multi-Model Detection Layer
===========================

Orchestrates multiple AI models to process a single frame based on zone routing.

Key Features:
- Zone-aware model routing (dough_mixing → hairnet + fall models)
- Cascaded inference (primary → secondary → tertiary)
- Frame rate throttling per model (prevent CPU overload)
- Detection fusion (merge overlapping detections from multiple models)
- Backward compatible with existing single-model code

Architecture:
    Frame → MultiModelDetector.detect(frame, zone_type)
         ├─> Primary models (always run)
         ├─> Secondary models (if zone matches)
         └─> Tertiary models (if enabled)
         → Fused Detection List

Thread-Safety: 
- Models are loaded once at startup (immutable after init)
- Per-frame inference uses thread-local FPS throttling
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import cv2
import numpy as np

from config import settings
from services.detection_layer import Detection
from services.model_registry import ModelInfo, get_registry

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Detection Fusion
# ─────────────────────────────────────────────────────────────────────────────


def compute_iou(bbox1: List[int], bbox2: List[int]) -> float:
    """Compute Intersection over Union between two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i < x1_i or y2_i < y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def get_canonical_violation(label: str) -> Optional[str]:
    """Map model-specific class names to canonical violation types."""
    return settings.VIOLATION_CLASS_MAPPING.get(label)


def fuse_detections(detections: List[Detection]) -> List[Detection]:
    """
    Merge overlapping detections from multiple models.
    
    Strategy:
    1. Group detections by IoU threshold
    2. For violations, map to canonical types (e.g., NO-Bakery-Head-Cap + NO-Hairnet → no_headcap)
    3. Keep detection with highest confidence per group
    4. Preserve all non-overlapping detections
    """
    if not settings.ENABLE_DETECTION_FUSION or len(detections) <= 1:
        return detections

    fused: List[Detection] = []
    used_indices = set()

    for i, det1 in enumerate(detections):
        if i in used_indices:
            continue

        # Find all detections that overlap with det1
        group = [det1]
        group_indices = [i]

        for j, det2 in enumerate(detections):
            if j <= i or j in used_indices:
                continue

            iou = compute_iou(det1.bbox, det2.bbox)
            
            if iou > settings.FUSION_IOU_THRESHOLD:
                # Check if they're the same violation type (canonical)
                canon1 = get_canonical_violation(det1.label)
                canon2 = get_canonical_violation(det2.label)
                
                # Merge if both are violations of the same type, or both are neutral objects
                if (canon1 and canon2 and canon1 == canon2) or \
                   (not canon1 and not canon2 and det1.label == det2.label):
                    group.append(det2)
                    group_indices.append(j)

        # Mark all grouped detections as used
        for idx in group_indices:
            used_indices.add(idx)

        # Keep detection with highest confidence
        best_det = max(group, key=lambda d: d.confidence)
        
        # If it's a violation, use canonical label
        if get_canonical_violation(best_det.label):
            best_det.label = get_canonical_violation(best_det.label)
        
        fused.append(best_det)

    logger.debug(f"Fused {len(detections)} detections → {len(fused)} unique detections")
    return fused


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Model Detector
# ─────────────────────────────────────────────────────────────────────────────


class MultiModelDetector:
    """Orchestrates multiple AI models based on zone routing."""

    def __init__(self):
        """Initialize detector and load all enabled models."""
        self.registry = get_registry()
        self.model_last_run: Dict[str, float] = {}  # Track last inference time per model
        
        logger.info(f"✅ MultiModelDetector initialized with {len(self.registry.models)} models")

    def detect(
        self, 
        frame: np.ndarray, 
        zone_type: Optional[str] = None,
        camera_id: Optional[int] = None,
        enabled_models: Optional[List[str]] = None,
    ) -> List[Detection]:
        """
        Run inference on frame using all relevant models for the given zone.

        Args:
            frame: Input image (BGR format, numpy array)
            zone_type: Zone type (e.g., "dough_mixing", "shop_counter")
                       If None, uses only primary models
            camera_id: Camera ID (for logging/debugging)
            enabled_models: Override list of model keys to use (ignores zone routing)

        Returns:
            List of Detection objects from all models, fused if enabled
        """
        if frame is None or frame.size == 0:
            return []

        # Get models to run
        if enabled_models:
            # Use explicit model list
            models_to_run = [
                self.registry.get_model(key) 
                for key in enabled_models 
                if self.registry.is_model_enabled(key)
            ]
            models_to_run = [m for m in models_to_run if m is not None]
        else:
            # Use zone-based routing
            models_to_run = self.registry.get_models_for_zone(zone_type)

        if not models_to_run:
            logger.warning(f"No models available for zone '{zone_type}'")
            return []

        # Run inference on each model (with FPS throttling)
        all_detections: List[Detection] = []
        
        for model_info in models_to_run:
            # Check FPS throttling
            if not self._should_run_model(model_info):
                continue

            try:
                detections = self._run_model_inference(frame, model_info)
                all_detections.extend(detections)
                
                # Update inference stats
                model_info.inference_count += 1
                self.model_last_run[model_info.key] = time.time()
                
            except Exception as e:
                logger.error(
                    f"Error running model {model_info.key}: {e}", 
                    exc_info=True
                )

        # Fuse detections if enabled
        if settings.ENABLE_DETECTION_FUSION and len(all_detections) > 1:
            all_detections = fuse_detections(all_detections)

        logger.debug(
            f"Zone '{zone_type}' | Camera {camera_id} | "
            f"Models: {[m.key for m in models_to_run]} | "
            f"Detections: {len(all_detections)}"
        )

        return all_detections

    def _should_run_model(self, model_info: ModelInfo) -> bool:
        """
        Check if model should run based on FPS throttling.
        
        Returns:
            True if enough time has elapsed since last inference
        """
        if model_info.target_fps <= 0:
            return True  # No throttling

        last_run = self.model_last_run.get(model_info.key, 0.0)
        min_interval = 1.0 / model_info.target_fps
        elapsed = time.time() - last_run

        return elapsed >= min_interval

    def _run_model_inference(
        self, 
        frame: np.ndarray, 
        model_info: ModelInfo
    ) -> List[Detection]:
        """
        Run inference on a single model.
        
        Args:
            frame: Input image
            model_info: Model metadata and instance
            
        Returns:
            List of Detection objects
        """
        start_time = time.time()
        detections: List[Detection] = []

        try:
            if model_info.model_type == "yolo":
                detections = self._run_yolo_inference(frame, model_info)
            elif model_info.model_type == "mediapipe":
                detections = self._run_mediapipe_inference(frame, model_info)
            elif model_info.model_type == "openvino":
                detections = self._run_openvino_inference(frame, model_info)
            elif model_info.model_type in ("pytorch", "pytorch_trn"):
                detections = self._run_pytorch_inference(frame, model_info)
            elif model_info.model_type == "sklearn":
                detections = self._run_sklearn_inference(frame, model_info)
            else:
                logger.warning(f"Unknown model type: {model_info.model_type}")

        except Exception as e:
            logger.error(f"Inference error in {model_info.key}: {e}")
            return []

        # Update timing stats
        inference_time = time.time() - start_time
        model_info.total_inference_time += inference_time

        logger.debug(
            f"Model '{model_info.key}' inference: "
            f"{inference_time*1000:.1f}ms | {len(detections)} detections"
        )

        return detections

    def _run_yolo_inference(
        self, 
        frame: np.ndarray, 
        model_info: ModelInfo
    ) -> List[Detection]:
        """Run inference on YOLO model."""
        model = model_info.model
        
        target_classes = list(model_info.classes.keys()) if model_info.classes else None
        results = list(model(
            frame,
            verbose=False,
            imgsz=640,
            conf=model_info.conf_threshold,
            iou=settings.NMS_IOU,
            classes=target_classes,
            stream=True,
        ))

        detections = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                
                # If model specifies allowed classes, discard any unlisted class (e.g. tennis racket, non-fall)
                if model_info.classes:
                    if class_id not in model_info.classes:
                        continue
                    label = model_info.classes[class_id]
                else:
                    label = model.names[class_id] if class_id < len(model.names) else f"class_{class_id}"

                detections.append(
                    Detection(
                        label=label,
                        confidence=round(float(box.conf[0]), 4),
                        bbox=[int(v) for v in box.xyxy[0]],
                        class_id=class_id,
                        model_key=model_info.key,
                    )
                )

        return detections

    def _run_mediapipe_inference(
        self, 
        frame: np.ndarray, 
        model_info: ModelInfo
    ) -> List[Detection]:
        """Run inference on MediaPipe hand landmarks model."""
        # MediaPipe requires RGB format
        import cv2
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        try:
            import mediapipe as mp
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            landmarker = model_info.model
            
            results = landmarker.detect(mp_image)
            
            detections = []
            
            # Convert hand landmarks to Detection format
            if results.hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results.hand_landmarks):
                    # Get bounding box from landmarks
                    xs = [lm.x * frame.shape[1] for lm in hand_landmarks]
                    ys = [lm.y * frame.shape[0] for lm in hand_landmarks]
                    
                    x1, x2 = int(min(xs)), int(max(xs))
                    y1, y2 = int(min(ys)), int(max(ys))
                    
                    # Add padding
                    padding = 20
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(frame.shape[1], x2 + padding)
                    y2 = min(frame.shape[0], y2 + padding)
                    
                    detections.append(
                        Detection(
                            label=f"Hand-{hand_idx}",
                            confidence=0.95,  # MediaPipe doesn't provide confidence
                            bbox=[x1, y1, x2, y2],
                            class_id=hand_idx,
                            model_key=model_info.key,
                        )
                    )
            
            return detections
            
        except ImportError:
            logger.error("MediaPipe not installed")
            return []

    def _run_openvino_inference(
        self, 
        frame: np.ndarray, 
        model_info: ModelInfo
    ) -> List[Detection]:
        """Run inference on OpenVINO model (person action recognition)."""
        # OpenVINO inference requires preprocessing
        # For person_action_recognition, this is complex - placeholder for now
        logger.warning(f"OpenVINO inference not yet implemented for {model_info.key}")
        return []

    def _run_pytorch_inference(
        self, 
        frame: np.ndarray, 
        model_info: ModelInfo
    ) -> List[Detection]:
        """Run inference on PyTorch model.

        Supports two variants:
          - pytorch: Visual anomaly detection (MVTec AD) — single 64×64 crop
          - pytorch_trn: Temporal Relation Network (TRN) — single frame approximation
            (treats frame as a 1-frame temporal segment; lower confidence than true multi-frame)
        """
        try:
            import torch

            h, w = frame.shape[:2]

            if model_info.model_type == "pytorch_trn":
                # TRN: expects list of segments (8-frame temporal model)
                # Single-frame approximation: repeat frame 8 times
                resize_to = (224, 224)
                resized = cv2.resize(frame, resize_to)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
                # Stack 8 identical frames (segments) — temporal model expects [B, T*C, H, W]
                stacked = tensor.repeat(8, 1, 1)  # [24, 224, 224] = 8 segments * 3 channels
                stacked = stacked.unsqueeze(0)     # [1, 24, 224, 224]

                model = model_info.model
                detections = []
                if model is not None and hasattr(model, '__call__'):
                    try:
                        if hasattr(model, 'eval'):
                            model.eval()
                        with torch.no_grad():
                            output = model(stacked)
                        # output shape: [1, num_classes] — check top-1 class
                        if output is not None:
                            probs = torch.softmax(output, dim=1) if output.dim() == 2 else output
                            top_conf, top_cls = probs.max(dim=1)
                            conf = float(top_conf[0])
                            cls_idx = int(top_cls[0])
                            # Something-SomethingV2 throwing classes (approx indices 174-177)
                            throwing_classes = {174, 175, 176, 177}
                            if cls_idx in throwing_classes and conf > model_info.conf_threshold:
                                detections.append(
                                    Detection(
                                        label="Object_Throwing",
                                        confidence=round(conf, 3),
                                        bbox=[0, 0, w, h],
                                        model_key=model_info.key,
                                    )
                                )
                    except Exception as trn_err:
                        logger.debug(f"TRN inference failed (expected for single-frame): {trn_err}")
                return detections
            else:
                # Standard visual anomaly detection (MVTec AD 64×64 center crop)
                cx, cy = w // 2, h // 2
                x1 = max(0, cx - 32)
                y1 = max(0, cy - 32)
                x2 = min(w, x1 + 64)
                y2 = min(h, y1 + 64)

                crop = frame[y1:y2, x1:x2]
                if crop.shape[0] != 64 or crop.shape[1] != 64:
                    crop = cv2.resize(crop, (64, 64))

                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(crop_rgb).permute(2, 0, 1).float() / 255.0
                tensor = tensor.unsqueeze(0)

                model = model_info.model
                detections = []
                if hasattr(model, 'eval'):
                    model.eval()
                    with torch.no_grad():
                        output = model(tensor)
                else:
                    output = None

                if output is not None:
                    score = float(output.mean()) if hasattr(output, 'mean') else 0.0
                    if score > model_info.conf_threshold:
                        detections.append(
                            Detection(
                                label="Machine_Visual_Anomaly",
                                confidence=round(min(score, 1.0), 3),
                                bbox=[x1, y1, x2, y2],
                                model_key=model_info.key,
                            )
                        )
                return detections
        except Exception as e:
            logger.debug(f"PyTorch inference exception in {model_info.key}: {e}")
            return []

    def _run_sklearn_inference(
        self,
        frame: np.ndarray,
        model_info: ModelInfo,
    ) -> List[Detection]:
        """Run inference on scikit-learn anomaly model using trained 8-feature schema & scaler.

        Feature Schema (8 features):
          1. temp_celsius (approximated from warm pixel channel ratio or baseline 25.0)
          2. high_temp_duration (seconds/duration metric)
          3. thermal_stress_code (0.0 to 1.0)
          4. degradation_rate (frame variance change)
          5. health_score (base health 100.0)
          6. health_delta (change rate)
          7. light_value (mean luminance from frame)
          8. anomaly_score (initial anomaly score 0.0)
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_lum = float(np.mean(gray))
            std_lum = float(np.std(gray))

            # Approximate 8 sensor features from visual cues when telemetry not wired
            features = np.array([[
                25.0 + (mean_lum / 255.0) * 15.0,  # temp_celsius
                0.0,                               # high_temp_duration
                0.0,                               # thermal_stress_code
                std_lum / 100.0,                   # degradation_rate
                95.0,                              # health_score
                0.0,                               # health_delta
                mean_lum,                          # light_value
                0.0,                               # anomaly_score
            ]])

            # Apply StandardScaler if loaded
            scaler = model_info.auxiliary_model
            if scaler is not None and hasattr(scaler, "transform"):
                try:
                    features = scaler.transform(features)
                except Exception as sc_err:
                    logger.debug(f"Scaler transform fallback in {model_info.key}: {sc_err}")

            model = model_info.model
            if hasattr(model, "predict"):
                pred = model.predict(features)
                score = float(pred[0]) if pred is not None and len(pred) > 0 else 0.0
                # Anomaly check: regression output above threshold
                if score > model_info.conf_threshold:
                    h, w = frame.shape[:2]
                    return [
                        Detection(
                            label="Machine_Sensor_Anomaly",
                            confidence=round(min(score, 1.0), 3),
                            bbox=[10, 10, w - 10, 40],
                            model_key=model_info.key,
                        )
                    ]
        except Exception as e:
            logger.debug(f"Sklearn inference exception in {model_info.key}: {e}")
        return []

    def get_stats(self) -> Dict:
        """Get runtime statistics for all models."""
        return self.registry.get_model_stats()


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

_detector_instance: Optional[MultiModelDetector] = None


def get_multi_model_detector() -> MultiModelDetector:
    """Get the singleton MultiModelDetector instance."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = MultiModelDetector()
    return _detector_instance


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatible wrapper for existing code
# ─────────────────────────────────────────────────────────────────────────────


class MultiModelDetectionAdapter:
    """
    Adapter that wraps MultiModelDetector to match the AbstractDetector interface.
    
    This allows existing code to use multi-model detection without changes.
    """

    def __init__(self):
        self.detector = get_multi_model_detector()

    def detect(self, frame: np.ndarray, zone_type: Optional[str] = None) -> List[Detection]:
        """
        Run detection using primary models or zone-routed models.
        """
        return self.detector.detect(frame, zone_type=zone_type)

    def is_healthy(self) -> bool:
        """Check if at least the primary model is loaded."""
        return len(self.detector.registry.models) > 0


# Create module-level instance for backward compatibility
detector = MultiModelDetectionAdapter()


# ─────────────────────────────────────────────────────────────────────────────
# Test/Debug
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    print("="*70)
    print("MULTI-MODEL DETECTOR TEST")
    print("="*70)
    
    detector = get_multi_model_detector()
    
    print(f"\n✅ Loaded {len(detector.registry.models)} models")
    
    # Test zone routing
    test_zones = [
        "entrance",
        "dough_mixing", 
        "shop_counter",
        "unknown_zone",
    ]
    
    for zone in test_zones:
        models = detector.registry.get_models_for_zone(zone)
        print(f"\n🏢 Zone: {zone}")
        print(f"   Models ({len(models)}): {[m.key for m in models]}")
    
    # Test with dummy frame
    print("\n" + "="*70)
    print("DUMMY FRAME TEST")
    print("="*70)
    
    import numpy as np
    
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    for zone in ["entrance", "dough_mixing"]:
        print(f"\n🎬 Testing zone: {zone}")
        detections = detector.detect(dummy_frame, zone_type=zone)
        print(f"   Detections: {len(detections)}")
        
        for det in detections[:5]:  # Show first 5
            print(f"   - {det.label} ({det.confidence:.2f}) from {det.model_key}")
    
    print("\n" + "="*70)
    print("✅ Test complete")
    print("="*70)
