"""Model registry facade. Adding a model is a registry/configuration operation."""
from __future__ import annotations

import json
from typing import Dict, List


DEFAULT_MODELS = (
    {"model_key": "yolo-ppe", "display_name": "YOLO PPE / Bakery Objects", "model_type": "detection", "version": "legacy", "capabilities": ["object_detection", "ppe"]},
    {"model_key": "bytetrack", "display_name": "ByteTrack", "model_type": "tracking", "version": "legacy", "capabilities": ["object_tracking"]},
    {"model_key": "tesseract-ocr", "display_name": "Tesseract OCR", "model_type": "ocr", "version": "legacy", "capabilities": ["document_text"]},
    {"model_key": "pose-adapter", "display_name": "Pose Adapter", "model_type": "pose", "version": "adapter", "capabilities": ["keypoints", "hand_motion", "body_orientation"]},
    {"model_key": "face-recognition", "display_name": "Face Recognition", "model_type": "face", "version": "legacy", "capabilities": ["face_identity"]},
)


def seed_registry(db) -> None:
    from database import ModelRegistry
    for definition in DEFAULT_MODELS:
        row = db.query(ModelRegistry).filter(ModelRegistry.model_key == definition["model_key"]).first()
        if not row:
            values = dict(definition)
            capabilities = values.pop("capabilities")
            db.add(ModelRegistry(provider="local", class_map_json="{}", config_json="{}", enabled=True,
                                 health_status="unknown", capabilities_json=json.dumps(capabilities), **values))
    db.commit()


def list_models(db) -> List[Dict]:
    from database import ModelRegistry
    return [{"id": row.id, "model_key": row.model_key, "display_name": row.display_name,
             "model_type": row.model_type, "version": row.version, "enabled": row.enabled,
             "health_status": row.health_status, "capabilities": json.loads(row.capabilities_json or "[]")}
            for row in db.query(ModelRegistry).order_by(ModelRegistry.model_type, ModelRegistry.model_key).all()]


def register_model(db, definition: Dict) -> Dict:
    """Register model metadata without wiring it into any business workflow."""
    from database import ModelRegistry
    row = db.query(ModelRegistry).filter(ModelRegistry.model_key == definition["model_key"]).first()
    if row:
        raise ValueError("Model key already exists")
    capabilities = definition.pop("capabilities", [])
    row = ModelRegistry(class_map_json=json.dumps(definition.pop("class_map", {})),
                        capabilities_json=json.dumps(capabilities), config_json=json.dumps(definition.pop("config", {})),
                        provider=definition.pop("provider", "local"), health_status="unknown", **definition)
    db.add(row); db.commit(); db.refresh(row)
    return {"id": row.id, "model_key": row.model_key}
