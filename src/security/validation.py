from __future__ import annotations

import re
import unicodedata

_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TAG_RE = re.compile(r"<[^>]*>")
_WS_RE = re.compile(r"[ \t\f\v]+")
_NL_RE = re.compile(r"\n{3,}")
_URL_RE = re.compile(r"https?://[^\s)>\]]+", re.I)
_COMBINING_RE = re.compile(r"[̀-ͯ]")


class MessageInvalid(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def validate_user_message(raw: str, max_len: int) -> str:
    """Normalise and validate a visitor message. Content is kept as-is (it is
    only ever sent to the model as data), apart from control characters."""
    text = unicodedata.normalize("NFC", raw)
    text = _CTRL_RE.sub("", text).replace("\r\n", "\n").strip()
    if not text:
        raise MessageInvalid("empty_message")
    if len(text) > max_len:
        raise MessageInvalid("message_too_long")
    return text


def clean_text(value: object, max_len: int, *, keep_newlines: bool = False) -> str:
    """Sanitise model output: no control chars, no HTML tags or angle brackets."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFC", str(value))
    text = _CTRL_RE.sub("", text)
    text = _TAG_RE.sub("", text)
    text = text.replace("<", "").replace(">", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if keep_newlines:
        lines = [_WS_RE.sub(" ", line).strip() for line in text.split("\n")]
        text = _NL_RE.sub("\n\n", "\n".join(lines)).strip()
    else:
        text = " ".join(text.split())
    if len(text) > max_len:
        text = text[:max_len].rstrip()
    return text


def normalize(text: str) -> str:
    """Lowercase, strip accents, unify apostrophes. For matching only."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = _COMBINING_RE.sub("", text)
    text = text.replace("’", "'").replace("`", "'")
    return " ".join(text.split())


def strip_unknown_urls(text: str, is_allowed) -> str:
    """Remove URLs from assistant text unless the policy allows them."""

    def _replace(match: re.Match) -> str:
        url = match.group(0).rstrip(".,;:!?")
        tail = match.group(0)[len(url):]
        return (url if is_allowed(url) else "") + tail

    return _URL_RE.sub(_replace, text)
