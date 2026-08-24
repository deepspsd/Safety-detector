"""Credential encryption for server-owned camera access.

The browser never receives RTSP or ONVIF passwords.  The key is supplied by
the deployment environment, not generated silently at runtime.
"""

from __future__ import annotations

import base64
import hashlib
import os
from typing import Tuple

from cryptography.fernet import Fernet, InvalidToken


class CredentialConfigurationError(RuntimeError):
    pass


def _fernet() -> Fernet:
    secret = os.getenv("CAMERA_CREDENTIAL_KEY", "").strip()
    if not secret:
        raise CredentialConfigurationError(
            "CAMERA_CREDENTIAL_KEY is required before registering IP cameras"
        )
    # Accept Fernet keys directly.  A long deployment secret is deterministically
    # derived only for backwards-friendly setup; it is never written to disk.
    try:
        return Fernet(secret.encode("utf-8"))
    except (ValueError, TypeError):
        derived = base64.urlsafe_b64encode(
            hashlib.sha256(secret.encode("utf-8")).digest()
        )
        return Fernet(derived)


def encrypt(value: str) -> str:
    return _fernet().encrypt(value.encode("utf-8")).decode("ascii")


def decrypt(value: str) -> str:
    try:
        return _fernet().decrypt(value.encode("ascii")).decode("utf-8")
    except InvalidToken as exc:
        raise CredentialConfigurationError(
            "Stored camera credential cannot be decrypted with CAMERA_CREDENTIAL_KEY"
        ) from exc


def encrypt_credentials(username: str, password: str) -> Tuple[str, str]:
    return encrypt(username), encrypt(password)
