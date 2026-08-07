"""Health monitor records operational metrics independently of business alerts."""
from __future__ import annotations

from datetime import datetime
import os
from typing import Dict


def collect_camera_health(camera_id: int, fps: float, error: str | None, queue_depth: int = 0) -> Dict:
    return {"camera_id": camera_id, "reader_fps": round(fps, 2), "queue_depth": queue_depth,
            "error": error, "status": "error" if error else ("online" if fps > 0 else "degraded")}


def record(component_type: str, component_id: str, status: str, metrics: Dict, message: str | None = None) -> None:
    try:
        from database import SessionLocal, HealthLog
        import json
        db = SessionLocal()
        try:
            db.add(HealthLog(component_type=component_type, component_id=str(component_id), status=status,
                             metrics_json=json.dumps(metrics), message=message, measured_at=datetime.utcnow()))
            db.commit()
        finally:
            db.close()
    except Exception:
        pass
