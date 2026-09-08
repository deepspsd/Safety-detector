"""
fcm_service.py — Firebase Cloud Messaging push notifications (HTTP v1 API)
===========================================================================
Uses the modern FCM HTTP v1 API (OAuth2 service account), NOT the deprecated
Legacy API (which is disabled in this project).

Setup
─────
1. Firebase Console → Project Settings → Service Accounts
2. Click "Generate new private key" → download JSON → save as:
   backend/firebase_service_account.json
3. Set in backend/.env:
   FCM_SERVICE_ACCOUNT_PATH=firebase_service_account.json

Flow
────
  Browser/PWA  →  POST /users/me/fcm-token  →  backend stores token in DB
  Alert fires  →  fcm_service.send_fcm_alert()
               →  GET Google OAuth2 token  →  POST fcm.googleapis.com/v1
               →  Google relay  →  owner's phone (anywhere on internet)

Throttling
──────────
  Per (camera_id, issue_type) cooldown = FCM_NOTIFICATION_COOLDOWN_SEC (120 s).
  1 push per 2 minutes per (camera, issue_type) pair — prevents notification spam.
"""

from __future__ import annotations

import json
import logging
import os
import time
from threading import Lock
from typing import Dict, Optional, Tuple

log = logging.getLogger("fcm_service")

_throttle: Dict[Tuple[str, str], float] = {}
_throttle_lock = Lock()

# Cached OAuth token
_oauth_token: Optional[str] = None
_oauth_expiry: float = 0.0
_oauth_lock = Lock()

try:
    import httpx
    _HTTPX_OK = True
except ImportError:
    _HTTPX_OK = False
    log.warning("[FCM] httpx not installed — push notifications disabled")

_FCM_V1_ENDPOINT = "https://fcm.googleapis.com/v1/projects/{project_id}/messages:send"
_GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
_TIMEOUT = 10.0
_FCM_SCOPE = "https://www.googleapis.com/auth/firebase.messaging"


# ── Token management ──────────────────────────────────────────────────────────


def register_token(user_id: int, token: str, device_name: Optional[str], db) -> dict:
    """Upsert an FCM registration token for the given user."""
    import datetime
    from database import FcmToken

    existing = db.query(FcmToken).filter(FcmToken.token == token).first()
    if existing:
        existing.user_id = user_id
        existing.device_name = device_name or existing.device_name
        existing.updated_at = datetime.datetime.utcnow()
        db.commit()
        db.refresh(existing)
        log.info(f"[FCM] Token refreshed for user={user_id} id={existing.id}")
        return {"id": existing.id, "status": "updated"}

    rec = FcmToken(user_id=user_id, token=token, device_name=device_name)
    db.add(rec)
    db.commit()
    db.refresh(rec)
    log.info(f"[FCM] Token registered for user={user_id} id={rec.id}")
    return {"id": rec.id, "status": "created"}


def delete_token(token: str, user_id: int, db) -> bool:
    """Remove a specific FCM token (called on logout / device change)."""
    from database import FcmToken

    rec = (
        db.query(FcmToken)
        .filter(FcmToken.token == token, FcmToken.user_id == user_id)
        .first()
    )
    if not rec:
        return False
    db.delete(rec)
    db.commit()
    log.info(f"[FCM] Token deleted for user={user_id}")
    return True


def _load_all_tokens(db) -> list:
    """Return all distinct FCM tokens in the DB."""
    from database import FcmToken
    return [r.token for r in db.query(FcmToken).all()]


# ── OAuth2 Service Account ────────────────────────────────────────────────────


def _load_service_account() -> Optional[dict]:
    """Load service account JSON. Returns None if not configured."""
    from config import settings
    sa_path = getattr(settings, "FCM_SERVICE_ACCOUNT_PATH", "")
    if not sa_path:
        return None

    # __file__ = backend/services/fcm_service.py → dirname = backend/services/ → dirname = backend/
    services_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir  = os.path.dirname(services_dir)

    if os.path.isabs(sa_path):
        abs_path = sa_path
    else:
        # Try relative to backend/ first, then project root
        candidate_backend = os.path.join(backend_dir, sa_path)
        candidate_root    = os.path.join(os.path.dirname(backend_dir), sa_path)
        abs_path = candidate_backend if os.path.exists(candidate_backend) else candidate_root

    if not os.path.exists(abs_path):
        log.warning(f"[FCM] Service account file not found: {abs_path}")
        return None

    with open(abs_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _make_jwt(sa: dict) -> str:
    """Create a signed JWT for Google OAuth2."""
    import base64
    import json as _json
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding

    now = int(time.time())
    header = {"alg": "RS256", "typ": "JWT"}
    payload = {
        "iss": sa["client_email"],
        "scope": _FCM_SCOPE,
        "aud": _GOOGLE_TOKEN_URL,
        "iat": now,
        "exp": now + 3600,
    }

    def _b64(data: dict) -> str:
        return base64.urlsafe_b64encode(_json.dumps(data).encode()).rstrip(b"=").decode()

    header_b64 = _b64(header)
    payload_b64 = _b64(payload)
    signing_input = f"{header_b64}.{payload_b64}".encode()

    private_key = serialization.load_pem_private_key(sa["private_key"].encode(), password=None)
    signature = private_key.sign(signing_input, padding.PKCS1v15(), hashes.SHA256())
    sig_b64 = base64.urlsafe_b64encode(signature).rstrip(b"=").decode()

    return f"{header_b64}.{payload_b64}.{sig_b64}"


def _get_oauth_token() -> Optional[str]:
    """Return a cached or freshly obtained OAuth2 access token."""
    global _oauth_token, _oauth_expiry

    with _oauth_lock:
        if _oauth_token and time.monotonic() < _oauth_expiry - 60:
            return _oauth_token

        sa = _load_service_account()
        if not sa:
            log.debug("[FCM] No service account configured — push disabled")
            return None

        if not _HTTPX_OK:
            log.warning("[FCM] httpx unavailable — cannot fetch OAuth token")
            return None

        try:
            jwt = _make_jwt(sa)
            with httpx.Client(timeout=_TIMEOUT) as client:
                resp = client.post(
                    _GOOGLE_TOKEN_URL,
                    data={
                        "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                        "assertion": jwt,
                    },
                )
            if resp.status_code == 200:
                data = resp.json()
                _oauth_token = data["access_token"]
                _oauth_expiry = time.monotonic() + data.get("expires_in", 3600)
                log.debug("[FCM] OAuth token refreshed")
                return _oauth_token
            else:
                log.warning(f"[FCM] OAuth token fetch failed: {resp.status_code} {resp.text[:200]}")
                return None
        except Exception as exc:
            log.warning(f"[FCM] OAuth token error: {exc}")
            return None


# ── Throttle ──────────────────────────────────────────────────────────────────


def _is_throttled(camera_id: Optional[int], issue_type: str) -> bool:
    from config import settings
    cooldown = getattr(settings, "FCM_NOTIFICATION_COOLDOWN_SEC", 120)
    key = (str(camera_id), issue_type or "generic")
    now = time.monotonic()
    with _throttle_lock:
        last = _throttle.get(key, 0.0)
        if now - last < cooldown:
            return True
        _throttle[key] = now
        return False


# ── FCM v1 send ───────────────────────────────────────────────────────────────


def _send_one(token: str, title: str, body: str, data: dict, oauth_token: str, project_id: str) -> bool:
    """Send a single FCM v1 message to one device token."""
    url = _FCM_V1_ENDPOINT.format(project_id=project_id)
    payload = {
        "message": {
            "token": token,
            "notification": {
                "title": title,
                "body": body,
            },
            "data": {k: str(v) for k, v in data.items()},
            "android": {
                "priority": "high",
                "notification": {"sound": "default", "icon": "ic_notification"},
            },
            "apns": {
                "headers": {"apns-priority": "10"},
                "payload": {"aps": {"sound": "default"}},
            },
            "webpush": {
                "headers": {"Urgency": "high"},
                "notification": {
                    "icon": "/pwa-192x192.png",
                    "badge": "/pwa-192x192.png",
                    "requireInteraction": True,
                },
            },
        }
    }

    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {oauth_token}",
                    "Content-Type": "application/json",
                },
            )
        if resp.status_code == 200:
            return True
        log.warning(f"[FCM] v1 send failed ({resp.status_code}): {resp.text[:200]}")
        return False
    except Exception as exc:
        log.warning(f"[FCM] v1 send error: {exc}")
        return False


def send_fcm_alert(
    message: str,
    severity: str = "medium",
    floor: Optional[str] = None,
    camera_name: Optional[str] = None,
    detected_issue: Optional[str] = None,
    camera_id: Optional[int] = None,
    worker_name: Optional[str] = None,
    db=None,
) -> bool:
    """
    Send FCM push to all registered device tokens via HTTP v1 API.
    Returns True if at least one token succeeded.
    """
    from config import settings

    project_id = getattr(settings, "FCM_PROJECT_ID", "")
    if not project_id:
        log.debug("[FCM] FCM_PROJECT_ID not set — push skipped")
        return False

    if not _HTTPX_OK:
        log.warning("[FCM] httpx unavailable — push skipped")
        return False

    if _is_throttled(camera_id, detected_issue or "generic"):
        log.debug("[FCM] Throttled — cam=%s issue=%s", camera_id, detected_issue)
        return False

    oauth_token = _get_oauth_token()
    if not oauth_token:
        return False

    severity_emoji = {"low": "🟡", "medium": "🟠", "high": "🔴", "critical": "🚨"}.get(severity, "⚠️")
    issue_label = detected_issue or 'Safety Alert'
    # Include worker name in title if known
    if worker_name:
        title = f"{severity_emoji} {severity.upper()} — {worker_name}: {issue_label}"
    else:
        title = f"{severity_emoji} {severity.upper()} — {issue_label}"
    body_parts = []
    if worker_name:
        body_parts.append(f"👷 Worker: {worker_name}")
    body_parts.append(message)
    if camera_name:
        body_parts.append(f"📷 {camera_name}")
    if floor:
        body_parts.append(f"📍 Floor: {floor.capitalize()}")
    body = "\n".join(body_parts)

    data = {
        "severity": severity,
        "detected_issue": detected_issue or "",
        "camera_name": camera_name or "",
        "floor": floor or "",
        "worker_name": worker_name or "",
    }

    _own_db = None
    try:
        if db is None:
            from database import SessionLocal
            _own_db = SessionLocal()
            _db = _own_db
        else:
            _db = db
        tokens = _load_all_tokens(_db)
    finally:
        if _own_db is not None:
            try:
                _own_db.close()
            except Exception:
                pass

    if not tokens:
        log.debug("[FCM] No registered tokens — push skipped")
        return False

    ok_count = 0
    for token in tokens:
        if _send_one(token, title, body, data, oauth_token, project_id):
            ok_count += 1

    log.info(f"[FCM] Push sent: '{title}' total={len(tokens)} ok={ok_count}")
    return ok_count > 0
