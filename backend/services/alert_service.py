"""
alert_service.py — v3.1
========================
Persists factory compliance alerts with:
  • Confidence-tier routing: high-confidence detectors → status="confirmed";
    low-confidence (Phase 4) detectors → status="pending_review".
  • Push notification: fires immediately on status="confirmed" alerts via
    send_push_alert() — tries ntfy.sh first, Telegram as fallback.
    "pending_review" alerts stay in-app only until an admin reviews them.
  • Full backward compatibility: all existing callers work with zero changes
    (confidence_tier defaults to "high" so old code paths save as "confirmed").

Confidence tiers
────────────────
  "high"  — confirmed detectors, always send push notification immediately:
              PPE violations, head-cap/no-cap, bangles, idle-time, shift-start,
              shop-absence, cylinder-swap, camera-down, shift-late.
  "low"   — Phase 4 / best-effort detectors, saved as "pending_review":
              chewing       ~70-80% accuracy
              eating        ~55-65% accuracy
              cash-in-pocket ~50-60% accuracy
              dirty-floor   ~75-85% accuracy  (frame-diff heuristic)
              gas-idle      ~40-60% steam recall (bakery_cv_plan.md §12)

  pending_review alerts are shown in the admin review queue (Block 9 UI).
  Admins promote them via PATCH /alerts/{id}/confirm which then fires push notification.
  Admins dismiss false positives via PATCH /alerts/{id}/dismiss.

Snapshot storage
────────────────
  Saves annotated JPEG to: uploads/snapshots/<YYYYMMDD_HHMMSS>_u<user_id>.jpg
  Served at: /uploads/snapshots/<filename>
  Also kept in DB (snapshot_b64) for immediate modal display.
"""

import os
import logging
import base64
import datetime

import numpy as np
import cv2
from sqlalchemy.orm import Session

from database import Alert, Camera

log = logging.getLogger("alert_service")

SNAPSHOT_DIR = os.path.join("uploads", "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# ── Confidence tier registry ──────────────────────────────────────────────────
# detected_issue strings that map to low-confidence (Phase 4) detectors.
# Everything NOT in this set is treated as high-confidence → "confirmed".
_LOW_CONFIDENCE_ISSUES: frozenset = frozenset({
    "Dirty floor detected",
    "Gas/oven idle — no supervision",
    "Chewing detected",
    "Eating at workstation",
    "Cash-in-pocket suspected",
    "Eating / chewing detected",  # combined label variant
})


def _resolve_status(detected_issue: str, confidence_tier: str) -> str:
    """
    Map a detected_issue + caller-provided tier to an Alert.status value.

    Rules (in priority order):
      1. If caller explicitly passes confidence_tier="low"  → "pending_review"
      2. If caller explicitly passes confidence_tier="high" → "confirmed"
      3. If detected_issue is in _LOW_CONFIDENCE_ISSUES     → "pending_review"
      4. Default                                            → "confirmed"
    """
    if confidence_tier == "low":
        return "pending_review"
    if confidence_tier == "high":
        return "confirmed"
    if detected_issue in _LOW_CONFIDENCE_ISSUES:
        return "pending_review"
    return "confirmed"


# ── Snapshot helper ───────────────────────────────────────────────────────────

def _save_snapshot_to_disk(
    annotated_b64: str,
    user_id: int,
) -> tuple[str | None, str | None]:
    """
    Decode base64 JPEG, write to disk, return (relative_path, b64_string).
    Returns (None, original_b64) on failure — never raises.
    """
    try:
        raw = annotated_b64
        if "," in raw:
            raw = raw.split(",", 1)[1]

        img_bytes = base64.b64decode(raw)
        nparr     = np.frombuffer(img_bytes, np.uint8)
        img       = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None, annotated_b64

        ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{ts}_u{user_id}.jpg"
        abs_path = os.path.join(SNAPSHOT_DIR, filename)
        cv2.imwrite(abs_path, img, [cv2.IMWRITE_JPEG_QUALITY, 88])

        rel_path = os.path.join("snapshots", filename).replace("\\", "/")
        log.info(f"Snapshot saved: {abs_path}")
        return rel_path, annotated_b64

    except Exception as exc:
        log.error(f"Failed to save snapshot: {exc}")
        return None, annotated_b64


# ── Camera name lookup ────────────────────────────────────────────────────────

def _get_camera_name(camera_id: int | None, db: Session) -> str | None:
    """Return camera display name for the Telegram caption, or None."""
    if camera_id is None:
        return None
    try:
        cam = db.query(Camera).filter(Camera.id == camera_id).first()
        return cam.name if cam else None
    except Exception:
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def save_alert(
    db: Session,
    user_id: int,
    message: str,
    role: str,
    severity: str,
    detected_issue: str,
    confidence: float = None,
    snapshot_b64: str = None,
    # v2 — factory FK fields (all optional, backward-compatible)
    camera_id: int = None,
    floor: str = None,
    employee_id: int = None,
    # v3 — confidence tier (optional, defaults to auto-detect via issue name)
    confidence_tier: str = "auto",   # "high" | "low" | "auto"
) -> Alert:
    """
    Persist an alert and (if confirmed) send a Telegram notification.

    Parameters
    ----------
    confidence_tier
        "high"  → status="confirmed", push notification sent immediately.
        "low"   → status="pending_review", notification skipped until admin confirms.
        "auto"  → auto-detect from detected_issue (default, backward-compatible).

    All other parameters are unchanged from v2 — existing callers need no edits.
    """
    # Resolve confidence tier → DB status
    status = _resolve_status(detected_issue or "", confidence_tier)

    # Save snapshot to disk
    snapshot_path = None
    if snapshot_b64:
        snapshot_path, snapshot_b64 = _save_snapshot_to_disk(snapshot_b64, user_id)

    # Persist to DB
    alert = Alert(
        user_id        = user_id,
        message        = message,
        role           = role,
        severity       = severity,
        detected_issue = detected_issue,
        confidence     = confidence,
        snapshot_b64   = snapshot_b64,
        snapshot_path  = snapshot_path,
        timestamp      = datetime.datetime.utcnow(),   # stored as naive UTC — serialised with Z suffix
        camera_id      = camera_id,
        floor          = floor,
        employee_id    = employee_id,
        status         = status,
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)

    log.info(
        f"Alert saved: id={alert.id} user={user_id} severity={severity} "
        f"status={status} issue={detected_issue} cam={camera_id} floor={floor} "
        f"snapshot={'✅' if snapshot_path else '—'}"
    )

    # Fire push notification for confirmed alerts only.
    # send_push_alert() auto-selects: ntfy.sh if configured, else Telegram.
    if status == "confirmed":
        camera_name = _get_camera_name(camera_id, db)
        _fire_push(
            message        = message,
            severity       = severity,
            floor          = floor,
            camera_name    = camera_name,
            detected_issue = detected_issue,
            snapshot_b64   = snapshot_b64,
        )

    return alert


def confirm_alert(alert_id: int, db: Session) -> Alert:
    """
    Promote a pending_review alert to confirmed and fire Telegram.
    Called by PATCH /alerts/{id}/confirm.
    Raises ValueError if alert not found or already confirmed/dismissed.
    """
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise ValueError(f"Alert {alert_id} not found")
    if alert.status != "pending_review":
        raise ValueError(
            f"Alert {alert_id} has status={alert.status!r} — only pending_review can be confirmed"
        )

    alert.status = "confirmed"
    db.commit()
    db.refresh(alert)

    log.info(f"Alert {alert_id} promoted to 'confirmed' by admin")

    # Now fire push notification (was deliberately skipped when first saved)
    camera_name = _get_camera_name(alert.camera_id, db)
    _fire_push(
        message        = alert.message,
        severity       = alert.severity,
        floor          = alert.floor,
        camera_name    = camera_name,
        detected_issue = alert.detected_issue,
        snapshot_b64   = alert.snapshot_b64,
    )
    return alert


def dismiss_alert(alert_id: int, db: Session) -> Alert:
    """
    Mark a pending_review alert as dismissed (false positive).
    Called by PATCH /alerts/{id}/dismiss.
    """
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise ValueError(f"Alert {alert_id} not found")
    if alert.status == "confirmed":
        raise ValueError(
            f"Alert {alert_id} is already confirmed — cannot dismiss a confirmed alert"
        )
    alert.status = "dismissed"
    db.commit()
    db.refresh(alert)
    log.info(f"Alert {alert_id} dismissed")
    return alert


# ── Internal: push notification dispatch ─────────────────────────────────────

def _fire_push(
    message: str,
    severity: str,
    floor: str | None,
    camera_name: str | None,
    detected_issue: str | None,
    snapshot_b64: str | None,
) -> None:
    """
    Background-safe push dispatch.  Tries ntfy first, Telegram as fallback.
    Failures are logged, never raised.
    """
    try:
        from services.notification_service import send_push_alert
        send_push_alert(
            message        = message,
            severity       = severity,
            floor          = floor,
            camera_name    = camera_name,
            detected_issue = detected_issue,
            snapshot_b64   = snapshot_b64,
        )
    except Exception as exc:
        log.error(f"[alert_service] Push dispatch error: {exc}")
