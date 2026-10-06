from __future__ import annotations

import json
import logging
import secrets
import time
import uuid
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from .ai.cache import TTLCache
from .ai.factory import build_provider
from .ai.prompt import build_static_prompt
from .ai.provider import AIProvider
from .api.deps import Container
from .api.errors import ApiError
from .api.v1 import chat, extras, health
from .config.settings import Settings, get_settings
from .i18n import t
from .knowledge.loader import KnowledgeBase, load_knowledge
from .observability.logging import setup_logging
from .observability.usage import UsageTracker
from .security.moderation import build_leak_index
from .security.rate_limit import DailyBudget, SlidingWindowLimiter
from .security.url_allowlist import UrlPolicy
from .sessions.manager import MemorySessionStore, SessionManager

log = logging.getLogger("chatbot")
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


class PayloadTooLarge(Exception):
    pass


class BodySizeLimitMiddleware:
    """Rejects oversized bodies, including chunked ones without Content-Length."""

    def __init__(self, app, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        length = dict(scope["headers"]).get(b"content-length")
        if length and length.isdigit() and int(length) > self.max_bytes:
            return await self._reject(scope, receive, send)

        received = 0
        responded = False

        async def guarded_send(message):
            if not responded:  # after our 413, drop whatever the app tries to send
                await send(message)

        async def limited_receive():
            nonlocal received, responded
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    # FastAPI turns body-read errors into a 400, so answer here.
                    if not responded:
                        responded = True
                        await self._send_413(send)
                    raise PayloadTooLarge()
            return message

        try:
            await self.app(scope, limited_receive, guarded_send)
        except PayloadTooLarge:
            pass

    @staticmethod
    async def _send_413(send):
        body = json.dumps(ApiError(413, "payload_too_large", t("en", "payload_too_large")).body()).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def _reject(self, scope, receive, send):
        await self._send_413(send)


def build_container(settings: Settings, kb: KnowledgeBase, provider: AIProvider) -> Container:
    canary = "CANARY-" + secrets.token_hex(8)
    static_prompt = build_static_prompt(kb.name, canary)
    return Container(
        settings=settings,
        kb=kb,
        provider=provider,
        urls=UrlPolicy(kb.known_urls),
        sessions=SessionManager(
            MemorySessionStore(settings.max_sessions), settings.session_ttl_minutes * 60
        ),
        limiter=SlidingWindowLimiter(
            [(60, settings.rate_limit_per_minute), (86400, settings.visitor_daily_limit)]
        ),
        aux_limiter=SlidingWindowLimiter([(60, 40)]),
        daily=DailyBudget(settings.daily_request_limit),
        usage=UsageTracker(
            settings.price_per_million_input_tokens, settings.price_per_million_output_tokens
        ),
        cache=TTLCache(settings.cache_ttl_seconds, settings.cache_max_entries),
        static_prompt=static_prompt,
        leak_index=build_leak_index(static_prompt),
        canary=canary,
        secret_values=[settings.ai_api_key, settings.portfolio_api_secret],
    )


def create_app(
    settings: Settings | None = None,
    *,
    provider: AIProvider | None = None,
    kb: KnowledgeBase | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_level)
    kb = kb or load_knowledge(settings.knowledge_dir, strict=settings.is_production)
    provider = provider or build_provider(settings)
    container = build_container(settings, kb, provider)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await provider.aclose()

    docs = not settings.is_production
    app = FastAPI(
        title="Portfolio chatbot API",
        docs_url="/docs" if docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if docs else None,
        lifespan=lifespan,
    )
    app.state.container = container

    async def request_context(request: Request, call_next):
        incoming = request.headers.get("x-request-id", "")
        request.state.request_id = incoming if _REQUEST_ID_RE.fullmatch(incoming) else uuid.uuid4().hex
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        if not (docs and request.url.path.startswith(("/docs", "/openapi"))):
            response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
        if settings.is_production:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        if request.url.path != "/health":
            log.info(
                "request",
                extra={
                    "request_id": request.state.request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": round((time.perf_counter() - started) * 1000),
                },
            )
        return response

    # Last added = outermost: CORS wraps everything so even errors carry CORS headers.
    app.add_middleware(BaseHTTPMiddleware, dispatch=request_context)
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_payload_bytes)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-Portfolio-Secret", "X-Visitor-Id", "X-Request-ID"],
        allow_credentials=False,
        max_age=600,
    )

    @app.exception_handler(ApiError)
    async def api_error_handler(_: Request, exc: ApiError):
        return JSONResponse(exc.body(), status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, __: RequestValidationError):
        # Never echo the submitted input back.
        return JSONResponse(ApiError(400, "invalid_request", t("en", "invalid_request")).body(), status_code=400)

    @app.exception_handler(StarletteHTTPException)
    async def http_handler(_: Request, exc: StarletteHTTPException):
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "error")
        return JSONResponse({"error": {"code": code, "message": "Not available."}}, status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception):
        log.error(
            "unhandled_error",
            extra={"request_id": getattr(request.state, "request_id", None), "error_type": type(exc).__name__},
        )
        return JSONResponse(ApiError(500, "internal_error", t("en", "generic_error")).body(), status_code=500)

    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(extras.router)
    return app


def _default_app() -> FastAPI:
    return create_app()


# `uvicorn src.main:app` -- lazy so importing this module in tests has no side effects.
def __getattr__(name: str):
    if name == "app":
        return _default_app()
    raise AttributeError(name)
