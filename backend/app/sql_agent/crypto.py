"""Encrypt/decrypt user database connection strings at rest."""

from __future__ import annotations

import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from app.config import settings

_CREDENTIALS_SALT = b"multimodal-rag-sql-credentials-v1"
_PBKDF2_ITERATIONS = 390_000


def _secret_material() -> bytes:
    return (settings.sql_credentials_key or settings.jwt_secret).encode("utf-8")


def _legacy_fernet_key() -> bytes:
    """Previous key derivation (SHA-256 digest only) — kept for decrypt migration."""
    digest = hashlib.sha256(_secret_material()).digest()
    return base64.urlsafe_b64encode(digest)


def _fernet_key() -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=_CREDENTIALS_SALT,
        iterations=_PBKDF2_ITERATIONS,
    )
    return base64.urlsafe_b64encode(kdf.derive(_secret_material()))


def encrypt_connection_url(url: str) -> str:
    return Fernet(_fernet_key()).encrypt(url.encode("utf-8")).decode("ascii")


def decrypt_connection_url(token: str) -> str:
    for key_fn in (_fernet_key, _legacy_fernet_key):
        try:
            return Fernet(key_fn()).decrypt(token.encode("ascii")).decode("utf-8")
        except InvalidToken:
            continue
    raise ValueError("invalid_encrypted_connection")
