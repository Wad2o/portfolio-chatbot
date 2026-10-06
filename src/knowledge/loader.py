from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("chatbot.knowledge")

LIST_FILES = ["skills", "services", "projects", "offers", "faq", "process", "technologies"]
DICT_FILES = ["profile", "contact"]


class KnowledgeError(Exception):
    pass


@dataclass
class KnowledgeBase:
    profile: dict
    contact: dict
    categories: dict  # {"projectTypes": [...], "developmentTypes": [...]}
    skills: list[dict] = field(default_factory=list)
    services: list[dict] = field(default_factory=list)
    projects: list[dict] = field(default_factory=list)
    offers: list[dict] = field(default_factory=list)
    faq: list[dict] = field(default_factory=list)
    process: list[dict] = field(default_factory=list)
    technologies: list[dict] = field(default_factory=list)
    known_urls: set[str] = field(default_factory=set)

    @property
    def name(self) -> str:
        return str(self.profile.get("name") or "the developer")

    @property
    def project_types(self) -> list[dict]:
        return self.categories.get("projectTypes", [])

    @property
    def project_type_ids(self) -> set[str]:
        return {c["id"] for c in self.project_types}

    @property
    def development_type_ids(self) -> set[str]:
        return {c["id"] for c in self.categories.get("developmentTypes", [])}

    @property
    def offer_ids(self) -> set[str]:
        return {o["id"] for o in self.offers}

    def project_type(self, type_id: str | None) -> dict | None:
        return next((c for c in self.project_types if c["id"] == type_id), None)


def public(value: Any) -> Any:
    """Strip author-only keys (prefixed with '_') before sending to the model."""
    if isinstance(value, dict):
        return {k: public(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, list):
        return [public(v) for v in value]
    return value


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        raise KnowledgeError(f"{path.name}: invalid JSON ({exc.msg} at line {exc.lineno})") from exc


def _enabled_items(name: str, raw: Any) -> list[dict]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise KnowledgeError(f"{name}.json must be a list")
    items, seen = [], set()
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise KnowledgeError(f"{name}.json: every entry needs a string 'id'")
        if item["id"] in seen:
            raise KnowledgeError(f"{name}.json: duplicate id '{item['id']}'")
        seen.add(item["id"])
        if item.get("enabled", True):
            items.append(item)
    return items


def _collect_urls(value: Any, out: set[str]) -> None:
    if isinstance(value, dict):
        for key, val in value.items():
            if isinstance(val, str) and str(key).lower().endswith("url") and val:
                out.add(val)
            else:
                _collect_urls(val, out)
    elif isinstance(value, list):
        for val in value:
            _collect_urls(val, out)


def load_knowledge(directory: str | Path, *, strict: bool = False) -> KnowledgeBase:
    base = Path(directory)
    if not base.is_dir():
        raise KnowledgeError(f"Knowledge directory not found: {base}")

    profile = _read(base / "profile.json")
    if not isinstance(profile, dict) or not profile.get("name"):
        raise KnowledgeError("profile.json must be an object with a 'name'")
    contact = _read(base / "contact.json") or {}
    if not isinstance(contact, dict):
        raise KnowledgeError("contact.json must be an object")

    categories = _read(base / "categories.json") or {"projectTypes": [], "developmentTypes": []}
    if not isinstance(categories, dict):
        raise KnowledgeError("categories.json must be an object")
    categories["projectTypes"] = _enabled_items("categories.projectTypes", categories.get("projectTypes"))
    categories["developmentTypes"] = _enabled_items(
        "categories.developmentTypes", categories.get("developmentTypes")
    )

    lists = {name: _enabled_items(name, _read(base / f"{name}.json")) for name in LIST_FILES}
    kb = KnowledgeBase(profile=profile, contact=contact, categories=categories, **lists)

    for offer in kb.offers:
        ptype = offer.get("type")
        if ptype and ptype not in kb.project_type_ids:
            raise KnowledgeError(f"offers.json: '{offer['id']}' uses unknown type '{ptype}'")

    _collect_urls(
        [kb.profile, kb.contact, kb.services, kb.projects, kb.offers, kb.faq, kb.process], kb.known_urls
    )

    blob = json.dumps(public([profile, contact, kb.skills, kb.services, kb.projects, kb.offers, kb.faq]))
    if "TODO" in blob:
        message = "Knowledge still contains TODO placeholders"
        if strict:
            raise KnowledgeError(message)
        log.warning(message)

    return kb
