from __future__ import annotations

import json
import re

from ..models.state import QualificationState
from ..security.validation import normalize
from .loader import KnowledgeBase, public

_STOPWORDS = set(
    "the and for with you your that this have has are was were from what how can does "
    "not but all any out our their them about into over also just like want need would "
    "une des les pour avec vous votre que qui est sont pas mais tout dans sur par aux "
    "ces ses mon mes notre nos veux voudrais besoin "
    "based where when which who whom whose there here some more much many very been being "
    "will shall should could tell give know get make made using used based "
    "ou quand dont etes etre avoir fait faire cette cet comme ici".split()
)

# Words that signal a collection is relevant even when no item matches directly.
_TRIGGERS: dict[str, set[str]] = {
    "projects": {"project", "projects", "work", "portfolio", "built", "examples", "projet",
                 "projets", "realisation", "realisations", "travaux", "exemples"},
    "services": {"service", "services", "offer", "offers", "hire", "hiring", "do", "build",
                 "offre", "offres", "embaucher", "proposez", "proposes", "faites"},
    "offers": {"offer", "offers", "price", "pricing", "cost", "quote", "estimate", "plan",
               "offre", "offres", "prix", "tarif", "configuration", "devis", "combien", "template",
               "custom", "difference", "fit", "choose", "pack"},
    "skills": {"skill", "skills", "technologies", "technology", "stack", "tech", "competence",
               "competences", "technologie", "technologies", "uses", "use", "utilisez"},
    "technologies": {"technologies", "technology", "stack", "tech", "technologie", "uses",
                     "use", "utilisez", "framework", "language", "languages"},
    "faq": {"how", "can", "do", "does", "what", "comment", "peut", "pouvez", "est"},
    "process": {"process", "steps", "timeline", "delivery", "workflow", "how", "processus",
                "etapes", "delai", "delais", "deroulement"},
}

_FIELDS_STRONG = ("keywords", "tags")
_FIELDS_NAME = ("id", "name", "title", "label", "question", "type", "category", "developmentType")
_FIELDS_TEXT = ("description", "summary", "answer", "bestFor", "features", "technologies")


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", normalize(text)) if len(w) >= 3 and w not in _STOPWORDS}


def _flatten(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return " ".join(_flatten(v) for v in value)
    if isinstance(value, dict):
        return " ".join(_flatten(v) for v in value.values())
    return "" if value is None else str(value)


def _score(item: dict, query: set[str]) -> int:
    score = 0
    for key in _FIELDS_STRONG:
        score += 3 * len(query & _tokens(_flatten(item.get(key))))
    for key in _FIELDS_NAME:
        score += 2 * len(query & _tokens(_flatten(item.get(key))))
    for key in _FIELDS_TEXT:
        score += len(query & _tokens(_flatten(item.get(key))))
    return score


def _select(items: list[dict], query: set[str], collection: str, limit: int, boost=None) -> list[dict]:
    scored = []
    for index, item in enumerate(items):
        score = _score(item, query) + (boost(item) if boost else 0)
        if score > 0:
            scored.append((-score, index, item))
    if scored:
        return [item for _, _, item in sorted(scored)[:limit]]
    if query & _TRIGGERS.get(collection, set()):
        return sorted(items, key=lambda i: i.get("priority", 100))[:limit]
    return []


def retrieve(
    kb: KnowledgeBase,
    query_text: str,
    qualification: QualificationState,
    *,
    max_chars: int = 6000,
    per_collection: int = 3,
) -> dict:
    """Pick the knowledge relevant to this turn. Keyword/category scoring is
    enough for a small knowledge base; replace this function with embedding
    search later without touching the rest of the pipeline."""
    query = _tokens(query_text)
    if qualification.projectType:
        query |= _tokens(qualification.projectType)

    def offer_boost(item: dict) -> int:
        return 6 if qualification.projectType and item.get("type") == qualification.projectType else 0

    for limit in range(per_collection, 0, -1):
        result = {
            "profile": public(kb.profile),
            "contact": public(kb.contact),
            "services": public(_select(kb.services, query, "services", limit)),
            "projects": public(_select(kb.projects, query, "projects", limit)),
            "offers": public(_select(kb.offers, query, "offers", limit, offer_boost)),
            "skills": public(_select(kb.skills, query, "skills", limit)),
            "technologies": public(_select(kb.technologies, query, "technologies", limit + 3)),
            "faq": public(_select(kb.faq, query, "faq", limit)),
            "process": public(_select(kb.process, query, "process", limit)),
        }
        result = {k: v for k, v in result.items() if v}
        if len(json.dumps(result, ensure_ascii=False)) <= max_chars:
            break
    return result


def project_type_guidance(kb: KnowledgeBase, qualification: QualificationState, query_text: str) -> list[dict]:
    """Qualification questions for the project type in play (or best guess)."""
    chosen = kb.project_type(qualification.projectType)
    if chosen:
        return [public(chosen)]
    query = _tokens(query_text)
    matches = [(_score(c, query), i, c) for i, c in enumerate(kb.project_types)]
    matches = [m for m in matches if m[0] > 0]
    return [public(c) for _, _, c in sorted(matches, key=lambda m: (-m[0], m[1]))[:2]]
