from __future__ import annotations

import asyncio
import logging

import httpx

from .provider import AIProvider, GenerationContext, ProviderError, ProviderMessage, ProviderResult

log = logging.getLogger("chatbot.provider")

DEFAULT_BASE_URLS = {
    "xai": "https://api.x.ai/v1",
    "grok": "https://api.x.ai/v1",
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}


class OpenAICompatibleProvider(AIProvider):
    """Chat-completions adapter. One class covers xAI (Grok), OpenAI, OpenRouter
    and self-hosted servers (vLLM, Ollama, ...) that expose /chat/completions."""

    def __init__(
        self,
        *,
        name: str,
        model: str,
        api_key: str,
        base_url: str,
        timeout: float = 20.0,
        max_tokens_param: str = "max_completion_tokens",
        max_retries: int = 1,
    ) -> None:
        self.name = name
        self.model = model
        self._api_key = api_key
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._timeout = timeout
        self._max_tokens_param = max_tokens_param
        self._max_retries = max_retries
        self._client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self._timeout))
        return self._client

    async def generate_response(
        self, messages: list[ProviderMessage], context: GenerationContext
    ) -> ProviderResult:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": context.system_prompt}]
            + [{"role": m.role, "content": m.content} for m in messages],
            "temperature": context.temperature,
            self._max_tokens_param: context.max_output_tokens,
            "stream": False,
        }
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

        last_error = "provider_error"
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._http().post(self._url, json=payload, headers=headers)
            except httpx.TimeoutException:
                raise ProviderError("timeout") from None  # never retry: costs time and money
            except httpx.HTTPError:
                last_error = "network_error"
            else:
                status = response.status_code
                if status == 200:
                    return self._parse(response)
                if status in (401, 403):
                    log.error("provider rejected credentials (status=%s)", status)
                    raise ProviderError("provider_auth")
                if status == 429:
                    raise ProviderError("provider_rate_limited")
                last_error = f"provider_http_{status}"
                if status < 500:
                    raise ProviderError(last_error)
            if attempt < self._max_retries:
                await asyncio.sleep(0.4)
        raise ProviderError(last_error)

    def _parse(self, response: httpx.Response) -> ProviderResult:
        try:
            body = response.json()
            choice = body["choices"][0]
            text = (choice.get("message") or {}).get("content") or ""
            usage = body.get("usage") or {}
        except (ValueError, KeyError, IndexError, TypeError):
            raise ProviderError("malformed_provider_response") from None
        if not text.strip():
            # Typical with reasoning models when the token budget is spent thinking.
            raise ProviderError("empty_response")
        return ProviderResult(
            text=text,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            model=str(body.get("model") or self.model),
            finish_reason=choice.get("finish_reason"),
        )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
