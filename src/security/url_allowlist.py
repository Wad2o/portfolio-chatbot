from __future__ import annotations

from urllib.parse import urlparse


def normalize_url(url: str) -> str:
    url = url.strip()
    if len(url) > 1:
        url = url.rstrip("/")
    return url


class UrlPolicy:
    """The model may only point at URLs that exist in the approved knowledge."""

    def __init__(self, known_urls: set[str]) -> None:
        self._known = {normalize_url(u) for u in known_urls}

    @property
    def known(self) -> set[str]:
        return set(self._known)

    def is_allowed(self, url: object) -> bool:
        if not isinstance(url, str) or not url or len(url) > 300:
            return False
        if "\\" in url or any(ch.isspace() or ord(ch) < 32 for ch in url):
            return False
        norm = normalize_url(url)
        if norm not in self._known:
            return False
        if norm.startswith("/"):
            return not norm.startswith("//")
        parsed = urlparse(norm)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
