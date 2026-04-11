"""
Alert persistence service — v2.0
=================================
Saves alerts with annotated snapshot images:
  1. Saves the ANNOTATED frame (with bounding boxes + labels) to disk.
  2. Stores relative path in Alert.snapshot_path for persistent access.
  3. Also stores base64 in Alert.snapshot_b64 for immediate WebSocket display.

Snapshots are saved to:
  uploads/snapshots/<YYYYMMDD_HHMMSS>_u<user_id>.jpg
Served via FastAPI static mount at:
  http://localhost:8000/uploads/snapshots/<filename>
"""
import os
import logging
import base64
import datetime
import numpy as np
import cv2
from sqlalchemy.orm import Session
from database import Alert

log = logging.getLogger("alert_service")

SNAPSHOT_DIR = os.path.join("uploads", "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)


def _save_snapshot_to_disk(
    annotated_b64: str,
    user_id: int,
) -> tuple[str | None, str | None]:
    """
    Decode base64 JPEG, save as file, return (relative_path, b64_string).
    relative_path is relative to the uploads/ static root.
    Returns (None, None) on failure.
    """
    try:
        # Strip data-URI prefix if present
        raw = annotated_b64
        if "," in raw:
            raw = raw.split(",")[1]

        img_bytes = base64.b64decode(raw)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None, annotated_b64

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{ts}_u{user_id}.jpg"
        abs_path = os.path.join(SNAPSHOT_DIR, filename)
        cv2.imwrite(abs_path, img, [cv2.IMWRITE_JPEG_QUALITY, 88])

        # relative path from uploads/ root → served as /uploads/snapshots/<filename>
        rel_path = os.path.join("snapshots", filename).replace("\\", "/")
        log.info(f"Snapshot saved: {abs_path}")
        return rel_path, annotated_b64

    except Exception as e:
        log.error(f"Failed to save snapshot: {e}")
        return None, annotated_b64


def save_alert(
    db: Session,
    user_id: int,
    message: str,
    role: str,
    severity: str,
    detected_issue: str,
    confidence: float = None,
    snapshot_b64: str = None,   # should be the ANNOTATED frame base64
) -> Alert:
    """
    Persist an alert to the database.
    - If snapshot_b64 is provided, saves it to disk and stores the path.
    - Keeps snapshot_b64 in DB for immediate modal display (no extra HTTP call).
    """
    snapshot_path = None

    if snapshot_b64:
        snapshot_path, snapshot_b64 = _save_snapshot_to_disk(snapshot_b64, user_id)

    alert = Alert(
        user_id=user_id,
        message=message,
        role=role,
        severity=severity,
        detected_issue=detected_issue,
        confidence=confidence,
        snapshot_b64=snapshot_b64,
        snapshot_path=snapshot_path,
        timestamp=datetime.datetime.utcnow(),
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    log.info(
        f"Alert saved: id={alert.id} user={user_id} severity={severity} "
        f"issue={detected_issue} snapshot={'✅' if snapshot_path else '—'}"
    )
    return alert
