from __future__ import annotations

import re

from .provider import ProviderMessage

_TAG_RE = re.compile(r"</?\s*visitor_message[^>]*>", re.I)
MAX_HISTORY_ITEM_CHARS = 800


def wrap_visitor(text: str) -> str:
    """Mark visitor text as data. Any lookalike tags inside it are neutralised
    so it cannot close the wrapper early."""
    safe = _TAG_RE.sub("[tag removed]", text)
    return f"<visitor_message>\n{safe}\n</visitor_message>"


def build_messages(
    history: list[dict],
    new_message: str,
    *,
    max_history: int,
    max_prompt_chars: int,
) -> list[ProviderMessage]:
    """Last N turns + the new visitor message, trimmed to a character budget
    (oldest first) so token usage stays predictable."""
    recent = history[-max_history:] if max_history > 0 else []
    messages: list[ProviderMessage] = []
    for item in recent:
        content = str(item["content"])
        if item["role"] == "user":
            content = wrap_visitor(content[:MAX_HISTORY_ITEM_CHARS])
        messages.append(ProviderMessage(role=item["role"], content=content))
    current = ProviderMessage(role="user", content=wrap_visitor(new_message))

    def total() -> int:
        return sum(len(m.content) for m in messages) + len(current.content)

    while messages and total() > max_prompt_chars:
        messages.pop(0)
    # The conversation must start with a user turn.
    while messages and messages[0].role != "user":
        messages.pop(0)
    return messages + [current]
