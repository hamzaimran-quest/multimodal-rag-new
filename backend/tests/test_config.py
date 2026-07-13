"""Configuration hardening tests."""

from __future__ import annotations

import pytest

from app.config import Settings


def test_cors_origins_parsed_from_comma_separated_string():
    cfg = Settings(CORS_ORIGINS="http://a.test, http://b.test")
    assert cfg.cors_origin_list == ["http://a.test", "http://b.test"]


def test_jwt_secret_insecure_defaults_detected():
    cfg = Settings(JWT_SECRET="dev-insecure-change-me")
    assert cfg.jwt_secret_is_secure is False


def test_jwt_secret_secure_when_random():
    cfg = Settings(JWT_SECRET="x" * 48)
    assert cfg.jwt_secret_is_secure is True


def test_validate_production_secrets_rejects_insecure_by_default():
    cfg = Settings(JWT_SECRET="dev-insecure-change-me", REQUIRE_SECURE_JWT_SECRET=True)
    with pytest.raises(RuntimeError, match="JWT_SECRET"):
        cfg.validate_production_secrets()


def test_validate_production_secrets_allows_insecure_when_disabled():
    cfg = Settings(JWT_SECRET="dev-insecure-change-me", REQUIRE_SECURE_JWT_SECRET=False)
    cfg.validate_production_secrets()
