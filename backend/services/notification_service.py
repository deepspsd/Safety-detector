"""
notification_service.py — Push alert dispatch (FCM)
====================================================
Central push notification wrapper for the safety monitoring system.

All alert code calls send_push_alert() — this file decides HOW to deliver it.
Currently: Firebase Cloud Messaging (FCM).

FCM works over the internet even when the app runs locally — the server
only needs outbound HTTPS to fcm.googleapis.com.

Setup (once):
  1. Firebase Console → Create project → Add Web App → copy firebaseConfig
  2. Project Settings → Cloud Messaging → copy Server Key
  3. Add to backend/.env:   FCM_SERVER_KEY=<your-server-key>
  4. Add to frontend/.env:  VITE_FIREBASE_* (see firebase.js)
  5. Restart server — done.

Throttling: 1 push per 2 minutes per (camera, issue_type) — prevents floods.
"""

from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger("notification_service")


def send_push_alert(
    message: str,
    severity: str = "medium",
    floor: Optional[str] = None,
    camera_name: Optional[str] = None,
    detected_issue: Optional[str] = None,
    snapshot_b64: Optional[str] = None,  # kept for API compat, not used by FCM
    camera_id: Optional[int] = None,
    db=None,
) -> bool:
    """
    Send a push notification via FCM.

    Drop-in replacement for the old ntfy / Telegram send_push_alert().
    All existing callers work unchanged — snapshot_b64 is accepted but
    ignored (FCM web push does not support inline images in the legacy API).

    Returns True if delivery succeeded, False otherwise.
    """
    try:
        from services.fcm_service import send_fcm_alert
        return send_fcm_alert(
            message=message,
            severity=severity,
            floor=floor,
            camera_name=camera_name,
            detected_issue=detected_issue,
            camera_id=camera_id,
            db=db,
        )
    except Exception as exc:
        log.error(f"[notification_service] send_push_alert error: {exc}")
        return False


# ── Legacy compatibility aliases ─────────────────────────────────────────────
# Old callers that still import send_telegram_alert() keep working without edits.

def send_telegram_alert(
    message: str,
    severity: str = "medium",
    floor: Optional[str] = None,
    camera_name: Optional[str] = None,
    detected_issue: Optional[str] = None,
    snapshot_b64: Optional[str] = None,
) -> bool:
    """Deprecated — now delegates to FCM via send_push_alert()."""
    return send_push_alert(
        message=message,
        severity=severity,
        floor=floor,
        camera_name=camera_name,
        detected_issue=detected_issue,
        snapshot_b64=snapshot_b64,
    )

