from __future__ import annotations

from ..config.settings import Settings
from .mock_provider import MockProvider
from .openai_compatible import DEFAULT_BASE_URLS, OpenAICompatibleProvider
from .provider import AIProvider


def build_provider(settings: Settings) -> AIProvider:
    name = settings.ai_provider.lower().strip()
    if name == "mock":
        return MockProvider()
    if name in DEFAULT_BASE_URLS or name == "openai_compatible":
        base_url = settings.ai_base_url or DEFAULT_BASE_URLS.get(name, "")
        if not base_url:
            raise ValueError("AI_BASE_URL is required for AI_PROVIDER=openai_compatible")
        if not settings.ai_api_key and name != "openai_compatible":
            raise ValueError(f"AI_API_KEY is required for AI_PROVIDER={name}")
        return OpenAICompatibleProvider(
            name=name,
            model=settings.ai_model,
            api_key=settings.ai_api_key,
            base_url=base_url,
            timeout=settings.ai_timeout_seconds,
            max_tokens_param=settings.ai_max_tokens_param,
        )
    raise ValueError(
        f"Unknown AI_PROVIDER '{settings.ai_provider}'. "
        "Use mock, xai, openai, openrouter or openai_compatible "
        "(add an adapter in src/ai/ for Anthropic or Gemini)."
    )
