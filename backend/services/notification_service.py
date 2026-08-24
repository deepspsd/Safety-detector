"""
notification_service.py — Mobile push alert dispatch
=====================================================
Sends "confirmed" factory alerts to the client's phones via two channels:

  1. ntfy.sh (PRIMARY — recommended)
     ─────────────────────────────────
     Free, open-source, native push notifications for Android + iOS.
     No account needed.  No vendor lock-in.  Works self-hosted.

     Setup (60 seconds):
       a. Install ntfy app on phone:
            Android: https://play.google.com/store/apps/details?id=io.heckel.ntfy
            iOS:     https://apps.apple.com/app/ntfy/id1625396347
       b. Open app → tap "+" → type a unique topic name (e.g. "bakery-alerts-x7k2").
       c. Add to .env:
            NTFY_TOPIC=bakery-alerts-x7k2
       d. Restart server — done.  All confirmed alerts go to the phone immediately.

       Self-hosted (unlimited, no data leaves your server):
         Run: docker run -p 80:80 binwiederhier/ntfy serve
         Then: NTFY_SERVER=http://your-server-ip

  2. Telegram (FALLBACK — kept for compatibility)
     ─────────────────────────────────────────────
     Setup: @BotFather → /newbot → add to group → set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID.

Behaviour
 ─────────
  • send_push_alert() tries ntfy first, Telegram second.  Use this function everywhere.
  • Only "confirmed" alerts are sent.  "pending_review" stays in-app only.
  • If neither channel is configured, notifications are silently skipped.
  • No raw video is ever transmitted.  Only alert text + snapshot JPEG.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Optional

log = logging.getLogger("notification_service")

# ── Lazy httpx import ─────────────────────────────────────────────────────────
# httpx is listed in requirements.txt. If somehow not installed (edge case),
# we degrade gracefully — notifications are skipped, server keeps running.
try:
    import httpx

    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False
    log.warning("[notification_service] httpx not installed — Telegram disabled")


# ── Telegram API base URL ─────────────────────────────────────────────────────
_TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"

# ── Request timeout (seconds) ────────────────────────────────────────────────
_TIMEOUT = 10.0


def _is_telegram_configured() -> bool:
    """Return True only if Telegram token and chat_id are set in config."""
    from config import settings

    return bool(settings.TELEGRAM_BOT_TOKEN and settings.TELEGRAM_CHAT_ID)


# Keep the old name as an alias so existing callers still work.
_is_configured = _is_telegram_configured


def _is_ntfy_configured() -> bool:
    """Return True only if ntfy topic is set in config."""
    from config import settings

    return bool(settings.NTFY_TOPIC)


def _build_caption(
    message: str,
    severity: str,
    floor: Optional[str],
    camera_name: Optional[str],
    detected_issue: Optional[str],
) -> str:
    """
    Build a readable Telegram caption.
    Telegram captions max 1024 chars; messages max 4096.
    """
    severity_emoji = {
        "low": "🟡",
        "medium": "🟠",
        "high": "🔴",
        "critical": "🚨",
    }.get(severity, "⚠️")

    parts = [
        f"{severity_emoji} *{severity.upper()} ALERT*",
    ]
    if detected_issue:
        parts.append(f"Issue: {detected_issue}")
    if floor:
        parts.append(f"Floor: {floor.capitalize()}")
    if camera_name:
        parts.append(f"Camera: {camera_name}")
    parts.append(f"\n{message}")
    return "\n".join(parts)


def send_telegram_alert(
    message: str,
    severity: str = "medium",
    floor: Optional[str] = None,
    camera_name: Optional[str] = None,
    detected_issue: Optional[str] = None,
    snapshot_b64: Optional[str] = None,
) -> bool:
    """
    Send an alert to the configured Telegram chat.

    Parameters
    ----------
    message        : Main alert text.
    severity       : low / medium / high / critical
    floor          : ground / first / second / shop (optional, for context)
    camera_name    : Display name of the camera that fired the alert (optional)
    detected_issue : Short issue label e.g. "Late shift start" (optional)
    snapshot_b64   : Base64-encoded JPEG snapshot. If provided, sent as photo.
                     Accepts both raw base64 and data-URI ("data:image/...").

    Returns
    -------
    True on success, False on any failure (network error, bad token, etc.)
    Failures are logged but never raised — caller is never disrupted.
    """
    if not _HTTPX_AVAILABLE:
        log.debug("[notification_service] httpx missing — skipping Telegram send")
        return False

    if not _is_configured():
        log.debug("[notification_service] Telegram not configured — skipping send")
        return False

    from config import settings

    token = settings.TELEGRAM_BOT_TOKEN
    chat_id = settings.TELEGRAM_CHAT_ID

    caption = _build_caption(message, severity, floor, camera_name, detected_issue)

    try:
        if snapshot_b64:
            return _send_photo(token, chat_id, snapshot_b64, caption)
        else:
            return _send_message(token, chat_id, caption)
    except Exception as exc:
        log.error(f"[notification_service] Unexpected error: {exc}")
        return False


def _send_photo(token: str, chat_id: str, snapshot_b64: str, caption: str) -> bool:
    """
    POST to sendPhoto with the snapshot as a multipart file upload.
    Falls back to sendMessage if the photo upload fails.
    """
    # Strip data-URI prefix if present ("data:image/jpeg;base64,...")
    raw_b64 = snapshot_b64
    if "," in raw_b64:
        raw_b64 = raw_b64.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(raw_b64)
    except Exception as exc:
        log.warning(
            f"[notification_service] base64 decode failed: {exc} — sending text only"
        )
        return _send_message(token, chat_id, caption)

    url = _TELEGRAM_API.format(token=token, method="sendPhoto")

    # Caption limit is 1024 chars for photos
    safe_caption = caption[:1020] + "…" if len(caption) > 1024 else caption

    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.post(
                url,
                data={
                    "chat_id": chat_id,
                    "caption": safe_caption,
                    "parse_mode": "Markdown",
                },
                files={
                    "photo": ("alert.jpg", io.BytesIO(img_bytes), "image/jpeg"),
                },
            )
        if resp.status_code == 200 and resp.json().get("ok"):
            log.info(f"[notification_service] Telegram photo sent (chat={chat_id})")
            return True
        else:
            log.warning(
                f"[notification_service] sendPhoto failed "
                f"({resp.status_code}): {resp.text[:200]} — falling back to text"
            )
            return _send_message(token, chat_id, caption)
    except httpx.TimeoutException:
        log.warning("[notification_service] Telegram sendPhoto timed out")
        return False
    except httpx.RequestError as exc:
        log.warning(f"[notification_service] Telegram network error: {exc}")
        return False


def _send_message(token: str, chat_id: str, text: str) -> bool:
    """POST to sendMessage (plain text fallback, also used when no snapshot)."""
    url = _TELEGRAM_API.format(token=token, method="sendMessage")

    # Message limit is 4096 chars
    safe_text = text[:4090] + "…" if len(text) > 4096 else text

    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.post(
                url,
                json={
                    "chat_id": chat_id,
                    "text": safe_text,
                    "parse_mode": "Markdown",
                },
            )
        if resp.status_code == 200 and resp.json().get("ok"):
            log.info(f"[notification_service] Telegram message sent (chat={chat_id})")
            return True
        else:
            log.warning(
                f"[notification_service] sendMessage failed "
                f"({resp.status_code}): {resp.text[:200]}"
            )
            return False
    except httpx.TimeoutException:
        log.warning("[notification_service] Telegram sendMessage timed out")
        return False
    except httpx.RequestError as exc:
        log.warning(f"[notification_service] Telegram network error: {exc}")
        return False


def send_ntfy_alert(
    message: str,
    severity: str = "medium",
    floor: Optional[str] = None,
    camera_name: Optional[str] = None,
    detected_issue: Optional[str] = None,
) -> bool:
    """
    Send a push notification via ntfy.sh (or self-hosted ntfy server).

    ntfy delivers native Android/iOS push notifications with no account
    needed.  The client subscribes to a topic in the ntfy app.

    Parameters
    ----------
    message        : Main alert text.
    severity       : low / medium / high / critical (maps to ntfy priority)
    floor          : ground / first / second / shop (optional context)
    camera_name    : Display name of the camera that fired the alert.
    detected_issue : Short issue label e.g. "Packing zone idle".

    Returns True on success, False on failure.  Never raises.
    """
    if not _HTTPX_AVAILABLE:
        log.debug("[notification_service] httpx missing — skipping ntfy send")
        return False

    if not _is_ntfy_configured():
        log.debug("[notification_service] ntfy not configured — skipping send")
        return False

    from config import settings

    # ntfy priority: 1=min, 2=low, 3=default, 4=high, 5=max
    ntfy_priority = {
        "low": "2",
        "medium": "3",
        "high": "4",
        "critical": "5",
    }.get(severity, "3")

    severity_emoji = {
        "low": "🟡",
        "medium": "🟠",
        "high": "🔴",
        "critical": "🚨",
    }.get(severity, "⚠️")

    # Build title and body
    title_parts = [f"{severity_emoji} {severity.upper()} ALERT"]
    if detected_issue:
        title_parts.append(f"— {detected_issue}")
    title = " ".join(title_parts)

    body_parts = [message]
    if floor:
        body_parts.append(f"Floor: {floor.capitalize()}")
    if camera_name:
        body_parts.append(f"Camera: {camera_name}")
    body = "\n".join(body_parts)

    url = f"{settings.NTFY_SERVER.rstrip('/')}/{settings.NTFY_TOPIC}"

    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.post(
                url,
                data=body.encode("utf-8"),
                headers={
                    "Title": title,
                    "Priority": ntfy_priority,
                    "Tags": "warning,factory",
                },
            )
        if resp.status_code in (200, 201):
            log.info(
                f"[notification_service] ntfy push sent — topic={settings.NTFY_TOPIC} "
                f"priority={ntfy_priority}"
            )
            return True
        else:
            log.warning(
                f"[notification_service] ntfy send failed "
                f"({resp.status_code}): {resp.text[:200]}"
            )
            return False
    except httpx.TimeoutException:
        log.warning("[notification_service] ntfy send timed out")
        return False
    except httpx.RequestError as exc:
        log.warning(f"[notification_service] ntfy network error: {exc}")
        return False


def send_push_alert(
    message: str,
    severity: str = "medium",
    floor: Optional[str] = None,
    camera_name: Optional[str] = None,
    detected_issue: Optional[str] = None,
    snapshot_b64: Optional[str] = None,
) -> bool:
    """
    Convenience wrapper that dispatches to the best available push channel.

    Priority:
      1. ntfy.sh  — if NTFY_TOPIC is configured.
      2. Telegram — if TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID are configured.
      3. No-op    — logs a debug message and returns False.

    Use this function in all new code instead of calling send_telegram_alert()
    or send_ntfy_alert() directly.
    """
    if _is_ntfy_configured():
        return send_ntfy_alert(
            message=message,
            severity=severity,
            floor=floor,
            camera_name=camera_name,
            detected_issue=detected_issue,
        )

    if _is_telegram_configured():
        return send_telegram_alert(
            message=message,
            severity=severity,
            floor=floor,
            camera_name=camera_name,
            detected_issue=detected_issue,
            snapshot_b64=snapshot_b64,
        )

    log.debug(
        "[notification_service] No push channel configured — alert is in-app only"
    )
    return False
