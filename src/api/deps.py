from __future__ import annotations

from dataclasses import dataclass, field

from ..ai.cache import TTLCache
from ..ai.provider import AIProvider
from ..config.settings import Settings
from ..knowledge.loader import KnowledgeBase
from ..observability.usage import UsageTracker
from ..security.rate_limit import DailyBudget, SlidingWindowLimiter
from ..security.url_allowlist import UrlPolicy
from ..sessions.manager import SessionManager


@dataclass
class Container:
    """Everything the request handlers need, built once in create_app()."""

    settings: Settings
    kb: KnowledgeBase
    provider: AIProvider
    urls: UrlPolicy
    sessions: SessionManager
    limiter: SlidingWindowLimiter
    aux_limiter: SlidingWindowLimiter  # feedback / events / summary (no AI cost)
    daily: DailyBudget
    usage: UsageTracker
    cache: TTLCache
    static_prompt: str
    leak_index: set
    canary: str
    secret_values: list[str] = field(default_factory=list)
