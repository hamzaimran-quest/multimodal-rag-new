"""Encrypt/decrypt user database connection strings at rest."""

from __future__ import annotations

import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken

from app.config import settings


def _fernet_key() -> bytes:
    raw = (settings.sql_credentials_key or settings.jwt_secret).encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    return base64.urlsafe_b64encode(digest)


def encrypt_connection_url(url: str) -> str:
    return Fernet(_fernet_key()).encrypt(url.encode("utf-8")).decode("ascii")


def decrypt_connection_url(token: str) -> str:
    try:
        return Fernet(_fernet_key()).decrypt(token.encode("ascii")).decode("utf-8")
    except InvalidToken as exc:
        raise ValueError("invalid_encrypted_connection") from exc
