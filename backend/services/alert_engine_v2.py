"""Alert engine v2 — consumes RULE_MATCHED events, creates AlertCase rows,
saves an evidence JPEG snapshot, and publishes ALERT_CREATED + NOTIFICATION_REQUESTED.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from uuid import uuid4

from config import settings
from services.platform_events import Event, bus, emit

log = logging.getLogger("alert.engine")


def _save_evidence_snapshot(frame, camera_id: int, alert_id: str) -> str | None:
    """
    Encode *frame* (numpy BGR array) as JPEG and save it to the evidence dir.
    Returns the relative path on success, None if frame is unavailable.
    """
    if frame is None:
        return None
    try:
        import cv2

        os.makedirs(settings.EVIDENCE_DIR, exist_ok=True)
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        subdir = os.path.join(settings.EVIDENCE_DIR, f"cam_{camera_id}", date_str)
        os.makedirs(subdir, exist_ok=True)
        filename = f"alert_{alert_id}.jpg"
        path = os.path.join(subdir, filename)
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return path
    except Exception as exc:
        log.debug("Evidence snapshot failed (cam=%s): %s", camera_id, exc)
        return None


def _get_latest_frame(camera_id: int):
    """Try to grab the latest frame from the camera manager (non-blocking)."""
    try:
        from services import camera_manager

        mc = camera_manager._cameras.get(camera_id)  # type: ignore[attr-defined]
        if mc:
            return mc.latest_frame()
    except Exception:
        pass
    return None


class AlertEngine:
    def __init__(self) -> None:
        bus.subscribe("RULE_MATCHED", self.create_alert)

    def create_alert(self, event: Event) -> None:
        from database import AlertCase, SessionLocal

        db = SessionLocal()
        try:
            payload = event.payload
            public_id = uuid4().hex
            title = payload.get("rule_name", "Configured surveillance rule")

            # ── Evidence snapshot ────────────────────────────────────────────
            frame = _get_latest_frame(event.camera_id)
            evidence_path = _save_evidence_snapshot(frame, event.camera_id or 0, public_id)

            case = AlertCase(
                public_id=public_id,
                rule_id=payload.get("rule_id"),
                camera_id=event.camera_id,
                event_id=payload.get("event", {}).get("event_id"),
                severity=payload.get("priority", "warning"),
                title=title,
                details_json=json.dumps(payload, default=str),
            )
            # Store evidence path if column exists
            if hasattr(case, "evidence_path"):
                case.evidence_path = evidence_path  # type: ignore[attr-defined]

            db.add(case)
            db.commit()
            db.refresh(case)

            log.info(
                "Alert created: %s | cam=%s | evidence=%s",
                title, event.camera_id, evidence_path,
            )

            emit(
                "ALERT_CREATED",
                camera_id=event.camera_id,
                zone_id=event.zone_id,
                track_id=event.track_id,
                correlation_id=case.public_id,
                source="alert-engine",
                payload={
                    "alert_case_id": case.id,
                    "public_id": case.public_id,
                    "title": title,
                    "severity": case.severity,
                    "evidence_path": evidence_path,
                    "targets": payload.get("notification_targets", []),
                    "event": payload.get("event", {}),
                },
            )
            emit(
                "NOTIFICATION_REQUESTED",
                camera_id=event.camera_id,
                correlation_id=case.public_id,
                source="alert-engine",
                payload={
                    "alert_case_id": case.id,
                    "targets": payload.get("notification_targets", []),
                    "title": title,
                    "severity": case.severity,
                    "evidence_path": evidence_path,
                },
            )
        except Exception as exc:
            log.exception("AlertEngine.create_alert failed: %s", exc)
            db.rollback()
        finally:
            db.close()


alert_engine_v2 = AlertEngine()
