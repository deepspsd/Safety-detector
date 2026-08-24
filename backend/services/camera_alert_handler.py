"""
camera_alert_handler.py — Camera health event → DB alert bridge
================================================================
Subscribes to platform events emitted by the camera heartbeat loop and
converts them into Alert rows so the admin dashboard and Telegram
notifications fire when a camera goes offline or drifts.

Events handled
──────────────
  CAMERA_OFFLINE        — camera status became "offline" or "error"
                          (emitted by _heartbeat_loop every 10 s while down)
  CAMERA_DRIFT_DETECTED — ORB feature-match score exceeded threshold
                          (emitted by enterprise_runtime._check_drift every 60 s)

Cooldown
────────
  To avoid spamming DB with an alert every 10 s while the camera stays down,
  each camera has an independent cooldown timer.  A new alert is only saved
  once per CAMERA_OFFLINE_COOLDOWN_SEC (default 300 = 5 min).

Integration
───────────
  Import this module in main.py startup so the bus subscriptions are active:
      from services import camera_alert_handler
      camera_alert_handler.register()
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Dict

log = logging.getLogger("camera_alert_handler")

# ── Per-camera alert cooldowns (monotonic epoch) ─────────────────────────────
# Structure: { camera_id: {"offline": last_ts, "drift": last_ts} }
_cooldowns: Dict[int, Dict[str, float]] = {}
_cooldown_lock = threading.Lock()

CAMERA_OFFLINE_COOLDOWN_SEC = 300  # 5 minutes between repeated "offline" alerts
CAMERA_DRIFT_COOLDOWN_SEC = 1800  # 30 minutes between repeated "drift" alerts

_CAMERA_ALERT_USER_ID: int = int(os.environ.get("RULE_ENGINE_USER_ID", "1"))


def _get_alert_user_id(db) -> int:
    """
    Return a valid user_id for system alerts.
    Tries RULE_ENGINE_USER_ID (default 1); falls back to the lowest-id user
    in the DB if user 1 doesn't exist, to prevent FK violations.
    """
    from database import User

    try:
        user = db.query(User).filter(User.id == _CAMERA_ALERT_USER_ID).first()
        if user:
            return user.id
        any_user = db.query(User).order_by(User.id).first()
        if any_user:
            log.warning(
                f"[camera_alert_handler] user_id={_CAMERA_ALERT_USER_ID} not found; "
                f"falling back to user_id={any_user.id}"
            )
            return any_user.id
    except Exception as exc:
        log.error(f"[camera_alert_handler] user lookup failed: {exc}")
    return _CAMERA_ALERT_USER_ID


def _within_cooldown(camera_id: int, alert_type: str, cooldown_sec: int) -> bool:
    """Return True if we are still within the cooldown window for this camera+type."""
    now = time.monotonic()
    with _cooldown_lock:
        cam_times = _cooldowns.setdefault(camera_id, {})
        last = cam_times.get(alert_type, 0.0)
        if now - last < cooldown_sec:
            return True
        cam_times[alert_type] = now
        return False


def _handle_camera_offline(event) -> None:
    """
    Fires when the heartbeat detects a camera went offline or errored.
    Saves a high-severity alert (rate-limited to once per CAMERA_OFFLINE_COOLDOWN_SEC).
    """
    camera_id = event.camera_id
    if camera_id is None:
        return

    if _within_cooldown(camera_id, "offline", CAMERA_OFFLINE_COOLDOWN_SEC):
        log.debug(
            f"[camera_alert_handler] CAMERA_OFFLINE cam={camera_id} suppressed (cooldown)"
        )
        return

    payload = event.payload or {}
    status = payload.get("status", "offline")
    error = payload.get("error") or ""
    fps = payload.get("fps", 0.0)
    status_label = "offline" if status == "offline" else "error"

    try:
        from database import Camera, SessionLocal
        from services.alert_service import save_alert

        db = SessionLocal()
        try:
            cam = db.query(Camera).filter(Camera.id == camera_id).first()
            cam_name = cam.name if cam else f"Camera {camera_id}"
            floor = cam.floor if cam else None

            full_msg = (
                f"📷 Camera '{cam_name}' went {status_label}. "
                f"Last FPS={fps:.1f}. "
                + (f"Error: {error[:200]}. " if error else "No frames received. ")
                + "Check network connection, power supply, and RTSP URL configuration."
            )

            user_id = _get_alert_user_id(db)
            save_alert(
                db=db,
                user_id=user_id,
                message=full_msg,
                role="System",
                severity="high",
                detected_issue="Camera offline / unreachable",
                camera_id=camera_id,
                floor=floor,
                confidence_tier="high",
            )
            log.warning(
                f"[camera_alert_handler] CAMERA_OFFLINE alert saved "
                f"cam={camera_id} name={cam_name!r} status={status_label}"
            )
        finally:
            db.close()
    except Exception as exc:
        log.error(f"[camera_alert_handler] CAMERA_OFFLINE alert save failed: {exc}")


def _handle_camera_drift(event) -> None:
    """
    Fires when ORB feature matching detects the camera has been moved.
    Saves a medium-severity alert (rate-limited to once per CAMERA_DRIFT_COOLDOWN_SEC).
    """
    camera_id = event.camera_id
    if camera_id is None:
        return

    if _within_cooldown(camera_id, "drift", CAMERA_DRIFT_COOLDOWN_SEC):
        return

    payload = event.payload or {}
    score = payload.get("score", 0.0)
    method = payload.get("method", "orb")

    try:
        from database import Camera, SessionLocal
        from services.alert_service import save_alert

        db = SessionLocal()
        try:
            cam = db.query(Camera).filter(Camera.id == camera_id).first()
            cam_name = cam.name if cam else f"Camera {camera_id}"
            floor = cam.floor if cam else None

            user_id = _get_alert_user_id(db)
            save_alert(
                db=db,
                user_id=user_id,
                message=(
                    f"📷 Camera '{cam_name}' may have been moved or tampered with "
                    f"(drift score={score:.1f}, method={method}). Zone calibration is now "
                    f"disabled for this camera until recalibrated. "
                    f"Please recalibrate via Settings → Cameras → Calibrate."
                ),
                role="System",
                severity="medium",
                detected_issue="Camera drift / tamper detected",
                camera_id=camera_id,
                floor=floor,
                confidence_tier="high",
            )
            log.warning(
                f"[camera_alert_handler] CAMERA_DRIFT alert saved "
                f"cam={camera_id} name={cam_name!r} score={score:.1f}"
            )
        finally:
            db.close()
    except Exception as exc:
        log.error(f"[camera_alert_handler] CAMERA_DRIFT alert save failed: {exc}")


def register() -> None:
    """
    Subscribe to camera health events on the platform event bus.
    Call once at startup (called from main.py startup event).
    Safe to call multiple times — bus.subscribe deduplicates handlers.
    """
    from services.platform_events import bus

    bus.subscribe("CAMERA_OFFLINE", _handle_camera_offline)
    bus.subscribe("CAMERA_DRIFT_DETECTED", _handle_camera_drift)
    log.info(
        "[camera_alert_handler] Subscribed to CAMERA_OFFLINE "
        "and CAMERA_DRIFT_DETECTED events"
    )
