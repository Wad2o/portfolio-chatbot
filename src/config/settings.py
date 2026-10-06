from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Runtime
    app_env: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"

    # AI provider
    ai_provider: str = "mock"
    ai_model: str = "grok-4-fast-non-reasoning"
    ai_api_key: str = ""
    ai_base_url: str = ""
    ai_timeout_seconds: float = 20.0
    ai_temperature: float = 0.3
    ai_max_tokens_param: Literal["max_tokens", "max_completion_tokens"] = (
        "max_completion_tokens"
    )
    price_per_million_input_tokens: float = 0.0
    price_per_million_output_tokens: float = 0.0

    # Portfolio link / CORS
    portfolio_origin: str = "http://localhost:3000"
    portfolio_api_secret: str = ""

    # Limits
    max_message_length: int = 1500
    max_messages_per_session: int = 20
    max_output_tokens: int = 450
    max_history_messages: int = 10
    session_ttl_minutes: int = 45
    max_sessions: int = 5000
    rate_limit_per_minute: int = 8
    visitor_daily_limit: int = 100
    daily_request_limit: int = 1500
    max_payload_bytes: int = 8192
    max_prompt_chars: int = 12000
    max_context_chars: int = 6000
    max_response_chars: int = 1800

    # Cache
    cache_ttl_seconds: int = 3600
    cache_max_entries: int = 200

    # Optional infrastructure (Phase 8)
    redis_url: str = ""
    database_url: str = ""

    knowledge_dir: str = "knowledge"

    @property
    def allowed_origins(self) -> list[str]:
        origins = []
        for raw in self.portfolio_origin.split(","):
            origin = raw.strip().rstrip("/")
            if origin and origin != "*":
                origins.append(origin)
        return origins

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @model_validator(mode="after")
    def _production_checks(self) -> "Settings":
        if self.app_env != "production":
            return self
        problems = []
        if len(self.portfolio_api_secret) < 24:
            problems.append("PORTFOLIO_API_SECRET must be set (>= 24 chars)")
        if "*" in self.portfolio_origin:
            problems.append("PORTFOLIO_ORIGIN must not contain '*'")
        if self.ai_provider == "mock":
            problems.append("AI_PROVIDER=mock is not allowed in production")
        elif not self.ai_api_key:
            problems.append("AI_API_KEY must be set")
        if problems:
            raise ValueError("Invalid production configuration: " + "; ".join(problems))
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
