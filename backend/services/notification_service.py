"""
notification_service.py — Telegram Bot alert dispatch
======================================================
Sends "confirmed" factory alerts to a Telegram group chat so the
client's staff are notified immediately on their phones.

Setup (5 minutes, completely free)
───────────────────────────────────
  1. Open Telegram → search @BotFather → send /newbot
  2. Follow the prompts → BotFather gives you an API token.
  3. Add the bot to your group chat (the one your workers/admins are in).
  4. Send any message in the group so the bot has a chat to post to.
  5. Fetch the chat_id:
       GET https://api.telegram.org/bot<token>/getUpdates
     Look for "chat":{"id": -100xxxxxxxxx} in the response.
  6. Add to your .env file:
       TELEGRAM_BOT_TOKEN=123456:ABCdefGHIjklMNOpqrsTUVwxyz
       TELEGRAM_CHAT_ID=-100123456789   # negative for group chats
  7. Restart the server.  Test with:
       POST /alerts/test-telegram    (admin only)

Behaviour
─────────
  • Only "confirmed" alerts are sent to Telegram.
  • "pending_review" alerts stay in-app only until an admin confirms them
    via PATCH /alerts/{id}/confirm — that endpoint calls
    send_telegram_alert() after promoting the status.
  • If TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is empty, notifications are
    silently disabled — no exception, no traceback, just a log.info().
    This means the server works out of the box without any Telegram setup.
  • Snapshot is sent as a photo (sendPhoto) if available; otherwise plain
    text (sendMessage).
  • httpx is used synchronously from a background thread (camera daemon).
    If called from an async context, wrap in asyncio.to_thread().

No paid API, no cloud service, no vendor lock-in.
All that leaves the server is the alert text + snapshot JPEG.
No raw video is ever transmitted to Telegram.
"""

from __future__ import annotations

import base64
import logging
import io
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


def _is_configured() -> bool:
    """Return True only if both token and chat_id are set in config."""
    from config import settings
    return bool(settings.TELEGRAM_BOT_TOKEN and settings.TELEGRAM_CHAT_ID)


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
        "low":      "🟡",
        "medium":   "🟠",
        "high":     "🔴",
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
    token   = settings.TELEGRAM_BOT_TOKEN
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
        log.warning(f"[notification_service] base64 decode failed: {exc} — sending text only")
        return _send_message(token, chat_id, caption)

    url = _TELEGRAM_API.format(token=token, method="sendPhoto")

    # Caption limit is 1024 chars for photos
    safe_caption = caption[:1020] + "…" if len(caption) > 1024 else caption

    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.post(
                url,
                data={
                    "chat_id":    chat_id,
                    "caption":    safe_caption,
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
                    "chat_id":    chat_id,
                    "text":       safe_text,
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
