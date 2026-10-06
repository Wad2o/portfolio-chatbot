from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ..knowledge.loader import KnowledgeBase
from ..models.responses import Action, Suggestion
from ..security.url_allowlist import UrlPolicy
from ..security.validation import clean_text, strip_unknown_urls

INTENTS = {
    "greeting", "about_developer", "skills", "projects", "services", "offer_qualification",
    "offer_recommendation", "pricing", "contact", "out_of_scope", "security_refusal", "unknown",
}
STAGES = {"none", "discovery", "narrowing", "recommendation", "handoff"}
_VALUE_RE = re.compile(r"[^a-z0-9-]+")
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)


class OutputError(Exception):
    """The model output could not be turned into a safe response."""


@dataclass
class ParsedOutput:
    message: str
    suggestions: list[Suggestion] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)
    intent: str = "unknown"
    recommended_offer: str | None = None
    stage: str = "none"
    state_update: dict = field(default_factory=dict)


def _extract_json(raw: str) -> dict | None:
    text = _FENCE_RE.sub("", raw.strip())
    for candidate in (text, text[text.find("{") : text.rfind("}") + 1] if "{" in text else ""):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(data, dict):
            return data
    return None


def parse_model_output(
    raw: str, kb: KnowledgeBase, urls: UrlPolicy, *, max_message_chars: int
) -> ParsedOutput:
    """Validate everything the model produced. The model suggests; the server
    decides what is allowed. Anything unsafe is dropped, never repaired."""
    data = _extract_json(raw)

    if data is None:
        stripped = raw.strip()
        # Truncated or garbled JSON must not be shown to a visitor.
        if not stripped or stripped.startswith(("{", "[", "```")):
            raise OutputError("unparseable_output")
        message = clean_text(stripped, max_message_chars, keep_newlines=True)
        if not message:
            raise OutputError("empty_output")
        return ParsedOutput(message=strip_unknown_urls(message, urls.is_allowed).strip() or message)

    message = clean_text(data.get("message"), max_message_chars, keep_newlines=True)
    message = strip_unknown_urls(message, urls.is_allowed).strip()
    if not message:
        raise OutputError("missing_message")

    suggestions: list[Suggestion] = []
    for item in data.get("suggestions") or []:
        if not isinstance(item, dict):
            continue
        label = clean_text(item.get("label"), 40)
        value = _VALUE_RE.sub("-", clean_text(item.get("value") or label, 40).lower()).strip("-")
        if label and value and len(suggestions) < 4:
            suggestions.append(Suggestion(label=label, value=value))

    actions: list[Action] = []
    for item in data.get("actions") or []:
        if not isinstance(item, dict) or len(actions) >= 3:
            continue
        kind = item.get("type")
        label = clean_text(item.get("label"), 40)
        if not label:
            continue
        if kind == "navigate":
            url = item.get("url")
            if urls.is_allowed(url):
                actions.append(Action(type="navigate", label=label, url=url.strip()))
        elif kind == "configure":
            payload = item.get("data")
            if not isinstance(payload, dict) or payload.get("type") not in kb.project_type_ids:
                continue
            safe = {"type": payload["type"]}
            development = payload.get("development")
            if development in kb.development_type_ids:
                safe["development"] = development
            actions.append(Action(type="configure", label=label, data=safe))
        elif kind == "contact":
            actions.append(Action(type="contact", label=label))

    intent = data.get("intent") if data.get("intent") in INTENTS else "unknown"
    offer = data.get("recommendedOffer")
    offer = offer if isinstance(offer, str) and offer in kb.offer_ids else None
    stage = data.get("qualificationStage") if data.get("qualificationStage") in STAGES else "none"
    state_update = data.get("stateUpdate") if isinstance(data.get("stateUpdate"), dict) else {}

    return ParsedOutput(
        message=message,
        suggestions=suggestions,
        actions=actions,
        intent=intent,
        recommended_offer=offer,
        stage=stage,
        state_update=state_update,
    )
