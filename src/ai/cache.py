from __future__ import annotations

import time
from collections import OrderedDict
from typing import Callable


class TTLCache:
    """Small in-memory cache for first-turn answers (quick-action buttons and
    common questions are asked by many visitors, so this saves real money)."""

    def __init__(self, ttl_seconds: int, max_entries: int, clock: Callable[[], float] = time.time) -> None:
        self._ttl = ttl_seconds
        self._max = max_entries
        self._clock = clock
        self._data: OrderedDict[tuple, tuple[float, object]] = OrderedDict()

    def get(self, key: tuple):
        entry = self._data.get(key)
        if not entry:
            return None
        expires, value = entry
        if self._clock() > expires:
            del self._data[key]
            return None
        self._data.move_to_end(key)
        return value

    def set(self, key: tuple, value: object) -> None:
        if self._ttl <= 0 or self._max <= 0:
            return
        self._data[key] = (self._clock() + self._ttl, value)
        self._data.move_to_end(key)
        while len(self._data) > self._max:
            self._data.popitem(last=False)
