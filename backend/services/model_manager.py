"""Model registry facade. Adding a model is a registry/configuration operation."""

from __future__ import annotations

import json
from typing import Dict, List

DEFAULT_MODELS = (
    {
        "model_key": "yolo-ppe",
        "display_name": "YOLO PPE / Bakery Objects",
        "model_type": "detection",
        "version": "v1.0",
        "capabilities": ["object_detection", "ppe", "person", "cylinder", "cash"],
    },
    {
        "model_key": "bytetrack",
        "display_name": "ByteTrack Tracker",
        "model_type": "tracking",
        "version": "v1.0",
        "capabilities": ["object_tracking", "persistent_id", "trajectory"],
    },
    {
        "model_key": "head_cap",
        "display_name": "Bakery Hairnet / Head Cap Classifier",
        "model_type": "image_classifier",
        "version": "tm_adapter",
        "capabilities": ["head_crop", "hairnet_detection", "sanitation"],
    },
    {
        "model_key": "uniform",
        "display_name": "Bakery Uniform Classifier",
        "model_type": "image_classifier",
        "version": "tm_adapter",
        "capabilities": ["upper_body_crop", "dress_code", "compliance"],
    },
    {
        "model_key": "bangle",
        "display_name": "Jewellery / Bangle Classifier",
        "model_type": "image_classifier",
        "version": "tm_adapter",
        "capabilities": ["wrist_crop", "food_safety"],
    },
    {
        "model_key": "activity",
        "display_name": "Worker Activity Classifier",
        "model_type": "image_classifier",
        "version": "tm_adapter",
        "capabilities": ["full_person_crop", "activity_state", "idle_aux"],
    },
    {
        "model_key": "tesseract-ocr",
        "display_name": "Tesseract OCR",
        "model_type": "ocr",
        "version": "legacy",
        "capabilities": ["document_text", "invoices", "orders"],
    },
    {
        "model_key": "pose-adapter",
        "display_name": "Pose / Motion Adapter",
        "model_type": "pose",
        "version": "adapter",
        "capabilities": ["keypoints", "hand_motion", "body_orientation"],
    },
    {
        "model_key": "face-recognition",
        "display_name": "Face Recognition & Attendance",
        "model_type": "face",
        "version": "legacy",
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
    """Register model metadata without wiring it into any business workflow."""
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
