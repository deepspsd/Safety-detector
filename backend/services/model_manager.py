"""
Model Manager - Central AI Model Manager for OccuSafe
======================================================

Responsible for:
- Model discovery & registration
- Lazy / startup loading & unloading
- Memory caching (loaded ONCE and cached)
- Device selection (CUDA / CPU auto-detection)
- Single & multi-model inference dispatch
- Error isolation (failure in one model never crashes the application)
- Dynamic confidence configuration
- Real-time latency & health monitoring
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from config import settings
from services.detection_layer import NormalizedDetection
from services.model_registry import BaseModelAdapter, ModelInfo, get_registry

logger = logging.getLogger("model_manager")


class ModelManager:
    """Central manager for AI model lifecycle and inference."""

    _instance: Optional[ModelManager] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.registry = get_registry()

    def discover_models(self) -> List[str]:
        """List all available model keys configured in system."""
        return list(settings.MODEL_REGISTRY.keys())

    def load_model(self, model_name: str) -> Optional[ModelInfo]:
        """Load model into memory and cache it. Returns ModelInfo or None."""
        try:
            return self.registry.load_model(model_name)
        except Exception as e:
            logger.error(f"[ModelManager] Failed to load model '{model_name}': {e}")
            return None

    def unload_model(self, model_name: str) -> None:
        """Unload model from memory."""
        try:
            self.registry.unload_model(model_name)
            logger.info(f"[ModelManager] Unloaded model '{model_name}'")
        except Exception as e:
            logger.warning(f"[ModelManager] Error unloading model '{model_name}': {e}")

    def is_loaded(self, model_name: str) -> bool:
        """Check if model is currently loaded in memory."""
        return self.registry.is_model_enabled(model_name)

    def infer(
        self,
        model_name: str,
        frame: np.ndarray,
        conf: Optional[float] = None,
        zone_id: Optional[int] = None,
        camera_id: Optional[int] = None,
    ) -> List[NormalizedDetection]:
        """
        Run inference using a single specialized model.
        
        Args:
            model_name: Registered model key (e.g. 'cash_detection', 'fall_detection')
            frame: Numpy BGR image (full frame or zone ROI crop)
            conf: Optional confidence threshold override
            zone_id: Optional zone identifier to tag detections
            camera_id: Optional camera identifier to tag detections
        """
        if frame is None or frame.size == 0:
            return []

        model_info = self.registry.get_model(model_name)
        if model_info is None or model_info.adapter is None:
            logger.debug(f"[ModelManager] Model '{model_name}' is not available for inference")
            return []

        try:
            detections = model_info.adapter.predict(frame, conf=conf)
            for det in detections:
                if zone_id is not None:
                    det.zone_id = zone_id
                if camera_id is not None:
                    det.camera_id = camera_id
            return detections
        except Exception as e:
            logger.error(f"[ModelManager] Error running inference on '{model_name}': {e}")
            return []

    def infer_multiple(
        self,
        model_names: List[str],
        frame: np.ndarray,
        conf_overrides: Optional[Dict[str, float]] = None,
        zone_id: Optional[int] = None,
        camera_id: Optional[int] = None,
    ) -> List[NormalizedDetection]:
        """
        Run inference across multiple models on the same frame / crop.
        Merges detections into a single normalized list.
        """
        if frame is None or frame.size == 0 or not model_names:
            return []

        conf_overrides = conf_overrides or {}
        combined: List[NormalizedDetection] = []

        for m_name in model_names:
            c_override = conf_overrides.get(m_name)
            dets = self.infer(
                model_name=m_name,
                frame=frame,
                conf=c_override,
                zone_id=zone_id,
                camera_id=camera_id,
            )
            combined.extend(dets)

        return combined

    def resolve_capabilities(self, capabilities: List[str]) -> List[str]:
        """Resolve a list of required capabilities to corresponding model keys."""
        models: List[str] = []
        for cap in capabilities:
            m_list = self.registry.get_recommended_models(cap)
            for m in m_list:
                if m not in models:
                    models.append(m)
        return models

    def get_recommended_models(self, capability_or_use_case: str) -> List[str]:
        """Recommend models for a specific capability or use case."""
        return self.registry.get_recommended_models(capability_or_use_case)

    def get_health(self) -> Dict[str, Dict[str, Any]]:
        """Return comprehensive health telemetry for all models."""
        return self.registry.get_model_stats()


# ─────────────────────────────────────────────────────────────────────────────
# Module Singleton
# ─────────────────────────────────────────────────────────────────────────────

_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get the singleton ModelManager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


# ─────────────────────────────────────────────────────────────────────────────
# Database Model Registry Seeding & Administration (Preserved for backward compat)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MODELS = (
    {
        "model_key": "yolov8x_coco",
        "display_name": "YOLOv8x COCO Object Detection",
        "model_type": "detection",
        "version": "v8x",
        "capabilities": ["person_detection", "phone_detection", "vehicle_detection"],
    },
    {
        "model_key": "hairnet_glove_detection",
        "display_name": "Hairnet & Glove Compliance",
        "model_type": "detection",
        "version": "v1.0",
        "capabilities": ["head_cover_compliance", "hand_protection_compliance", "sanitation_check"],
    },
    {
        "model_key": "fall_detection",
        "display_name": "Worker Fall Detector",
        "model_type": "detection",
        "version": "v1.0",
        "capabilities": ["worker_fall_detection", "person_down_detection"],
    },
    {
        "model_key": "cash_detection",
        "display_name": "Banknote & Cash Monitor",
        "model_type": "detection",
        "version": "yolo11m",
        "capabilities": ["cash_monitoring", "banknote_detection"],
    },
    {
        "model_key": "helmet_model",
        "display_name": "Hardhat & Helmet Compliance",
        "model_type": "detection",
        "version": "v1.0",
        "capabilities": ["hardhat_compliance", "head_protection"],
    },
    {
        "model_key": "machine_sensor_anomaly",
        "display_name": "Machine Predictive Maintenance",
        "model_type": "sklearn",
        "version": "v1.0",
        "capabilities": ["machine_anomaly_prediction", "predictive_maintenance"],
    },
    {
        "model_key": "hand_landmarks",
        "display_name": "MediaPipe Hand Landmarks",
        "model_type": "pose",
        "version": "mediapipe",
        "capabilities": ["hand_motion_tracking", "packing_motion"],
    },
    {
        "model_key": "bytetrack",
        "display_name": "ByteTrack Multi-Object Tracker",
        "model_type": "tracking",
        "version": "v1.0",
        "capabilities": ["object_tracking", "persistent_id", "trajectory"],
    },
    {
        "model_key": "face-recognition",
        "display_name": "Face Recognition & Attendance",
        "model_type": "face",
        "version": "dlib",
        "capabilities": ["face_identity", "auto_attendance"],
    },
)


def seed_registry(db) -> None:
    from database import ModelRegistry

    for definition in DEFAULT_MODELS:
        row = (
            db.query(ModelRegistry)
            .filter(ModelRegistry.model_key == definition["model_key"])
            .first()
        )
        if not row:
            values = dict(definition)
            capabilities = values.pop("capabilities")
            db.add(
                ModelRegistry(
                    provider="local",
                    class_map_json="{}",
                    config_json="{}",
                    enabled=True,
                    health_status="unknown",
                    capabilities_json=json.dumps(capabilities),
                    **values,
                )
            )
    db.commit()


def list_models(db) -> List[Dict]:
    from database import ModelRegistry

    return [
        {
            "id": row.id,
            "model_key": row.model_key,
            "display_name": row.display_name,
            "model_type": row.model_type,
            "version": row.version,
            "enabled": row.enabled,
            "health_status": row.health_status,
            "capabilities": json.loads(row.capabilities_json or "[]"),
        }
        for row in db.query(ModelRegistry)
        .order_by(ModelRegistry.model_type, ModelRegistry.model_key)
        .all()
    ]


def register_model(db, definition: Dict) -> Dict:
    from database import ModelRegistry

    row = (
        db.query(ModelRegistry)
        .filter(ModelRegistry.model_key == definition["model_key"])
        .first()
    )
    if row:
        raise ValueError("Model key already exists")
    capabilities = definition.pop("capabilities", [])
    row = ModelRegistry(
        class_map_json=json.dumps(definition.pop("class_map", {})),
        capabilities_json=json.dumps(capabilities),
        config_json=json.dumps(definition.pop("config", {})),
        provider=definition.pop("provider", "local"),
        health_status="unknown",
        **definition,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return {"id": row.id, "model_key": row.model_key}
