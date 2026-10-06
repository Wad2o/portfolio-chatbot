from __future__ import annotations

import hashlib
import hmac

from fastapi import Request

from ..api.errors import ApiError


def secrets_match(provided: str, expected: str) -> bool:
    return hmac.compare_digest(provided.encode(), expected.encode())


def require_portfolio_secret(request: Request) -> None:
    """Server-to-server auth. The secret lives only on the portfolio server."""
    expected = request.app.state.container.settings.portfolio_api_secret
    if not expected:  # only possible outside production (settings enforce it)
        return
    provided = request.headers.get("x-portfolio-secret", "")
    if not secrets_match(provided, expected):
        raise ApiError(401, "unauthorized", "Unauthorized.")


def visitor_key(request: Request) -> str:
    """Stable anonymous visitor identifier for rate limiting.

    The portfolio proxy should forward a hash of the visitor's IP in
    X-Visitor-Id. We hash again so no raw identifier is ever kept.
    """
    raw = request.headers.get("x-visitor-id", "")[:128]
    if not raw:
        raw = request.client.host if request.client else "unknown"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]
