from __future__ import annotations

import time
from collections import deque
from datetime import datetime, timezone
from typing import Callable


class SlidingWindowLimiter:
    """In-memory sliding-window limiter with several (window_seconds, max) rules.

    Single-process only. For several workers, implement the same interface on
    Redis (Phase 8)."""

    def __init__(
        self,
        rules: list[tuple[int, int]],
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._rules = rules
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._calls = 0
        self._horizon = max(window for window, _ in rules)

    def check(self, key: str) -> bool:
        """Return True and record the hit if allowed; False if limited."""
        now = self._clock()
        self._calls += 1
        if self._calls % 500 == 0:
            self._sweep(now)
        hits = self._hits.setdefault(key, deque())
        while hits and now - hits[0] > self._horizon:
            hits.popleft()
        for window, maximum in self._rules:
            recent = sum(1 for ts in hits if now - ts <= window)
            if recent >= maximum:
                return False
        hits.append(now)
        return True

    def _sweep(self, now: float) -> None:
        stale = [k for k, h in self._hits.items() if not h or now - h[-1] > self._horizon]
        for key in stale:
            del self._hits[key]


class DailyBudget:
    """Global cap on AI calls per UTC day (hard cost ceiling)."""

    def __init__(self, limit: int, clock: Callable[[], float] = time.time) -> None:
        self._limit = limit
        self._clock = clock
        self._day = ""
        self._used = 0

    def _today(self) -> str:
        return datetime.fromtimestamp(self._clock(), tz=timezone.utc).strftime("%Y-%m-%d")

    def try_consume(self) -> bool:
        today = self._today()
        if today != self._day:
            self._day, self._used = today, 0
        if self._used >= self._limit:
            return False
        self._used += 1
        return True

    def refund(self) -> None:
        if self._used > 0:
            self._used -= 1
