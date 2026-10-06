from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


class ProviderError(Exception):
    """Raised for any provider failure. `code` is for logs only; the visitor
    only ever sees a generic message."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class ProviderMessage:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class GenerationContext:
    system_prompt: str
    max_output_tokens: int
    temperature: float
    request_id: str = ""


@dataclass
class ProviderResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    finish_reason: str | None = None


class AIProvider(ABC):
    """Everything outside src/ai/ talks to this interface only. To support a
    new vendor, add an adapter and register it in factory.py."""

    name: str
    model: str

    @abstractmethod
    async def generate_response(
        self, messages: list[ProviderMessage], context: GenerationContext
    ) -> ProviderResult: ...

    async def aclose(self) -> None:  # optional cleanup hook
        return None
