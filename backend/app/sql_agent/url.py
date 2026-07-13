"""PostgreSQL connection URL validation and normalization."""

from __future__ import annotations

from urllib.parse import urlparse, urlunparse


_ALLOWED_SCHEMES = frozenset({"postgresql", "postgres", "postgresql+psycopg2"})


def validate_postgres_url(url: str) -> str:
    """Return normalized SQLAlchemy URL or raise ValueError."""
    cleaned = url.strip()
    if not cleaned:
        raise ValueError("connection_url_required")

    parsed = urlparse(cleaned)
    scheme = (parsed.scheme or "").lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise ValueError("postgresql_only")

    if not parsed.hostname:
        raise ValueError("connection_url_invalid")
    if not parsed.path or parsed.path in {"/", ""}:
        raise ValueError("connection_url_missing_database")

    # SQLAlchemy + psycopg2
    if scheme in {"postgresql", "postgres"}:
        netloc = parsed.netloc
        normalized = urlunparse(
            ("postgresql+psycopg2", netloc, parsed.path, parsed.params, parsed.query, parsed.fragment)
        )
        return normalized
    return cleaned


def redact_connection_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.password:
        netloc = parsed.netloc.replace(f":{parsed.password}@", ":***@")
        return urlunparse((parsed.scheme, netloc, parsed.path, parsed.params, parsed.query, parsed.fragment))
    return url
