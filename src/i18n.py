"""Canned visitor-facing strings and language detection.

To add a language (e.g. "ar", "de"): add a table to MESSAGES, DEFAULT_SUGGESTIONS
and the word list in _MARKERS."""
from __future__ import annotations

import re

SUPPORTED = ("en", "fr")

MESSAGES: dict[str, dict[str, str]] = {
    "en": {
        "rate_limited": "You're sending messages a little fast. Please wait a moment and try again.",
        "session_limit": "You've reached the temporary chat limit. You can continue by contacting {name} directly.",
        "daily_limit": "The assistant has reached its limit for today. You can contact {name} directly instead.",
        "unavailable": "The assistant is temporarily unavailable.",
        "generic_error": "Something went wrong while contacting the assistant. Please try again.",
        "empty_message": "Please type a message.",
        "message_too_long": "Your message is too long. Please keep it under {max} characters.",
        "invalid_request": "Invalid request.",
        "payload_too_large": "Request too large.",
        "no_project_info": "There is no project information to summarise yet.",
        "summary_project": "Project",
        "summary_requirements": "Requirements",
        "summary_pages": "Approximate number of pages",
        "summary_integrations": "Integrations",
        "summary_notes": "Additional notes",
        "summary_dev_template": "template-based",
        "summary_dev_custom": "custom",
        "refusal": "I can only help with questions about {name}'s work, services and offers, and with finding the right solution for your project. What would you like to know?",
    },
    "fr": {
        "rate_limited": "Vous envoyez des messages un peu vite. Patientez un instant puis réessayez.",
        "session_limit": "Vous avez atteint la limite temporaire de la conversation. Vous pouvez continuer en contactant directement {name}.",
        "daily_limit": "L'assistant a atteint sa limite pour aujourd'hui. Vous pouvez contacter directement {name}.",
        "unavailable": "L'assistant est temporairement indisponible.",
        "generic_error": "Une erreur est survenue lors de la communication avec l'assistant. Veuillez réessayer.",
        "empty_message": "Veuillez saisir un message.",
        "message_too_long": "Votre message est trop long. Merci de rester sous {max} caractères.",
        "invalid_request": "Requête invalide.",
        "payload_too_large": "Requête trop volumineuse.",
        "no_project_info": "Aucune information de projet à résumer pour le moment.",
        "summary_project": "Projet",
        "summary_requirements": "Besoins",
        "summary_pages": "Nombre de pages approximatif",
        "summary_integrations": "Intégrations",
        "summary_notes": "Notes complémentaires",
        "summary_dev_template": "basé sur un modèle",
        "summary_dev_custom": "sur mesure",
        "refusal": "Je peux uniquement répondre aux questions sur le travail, les services et les offres de {name}, et vous aider à trouver la solution adaptée à votre projet. Que souhaitez-vous savoir ?",
    },
}

DEFAULT_SUGGESTIONS: dict[str, list[tuple[str, str]]] = {
    "en": [
        ("Find the right offer", "find-offer"),
        ("Explore projects", "projects"),
        ("Services", "services"),
    ],
    "fr": [
        ("Trouver la bonne offre", "find-offer"),
        ("Voir les projets", "projects"),
        ("Services", "services"),
    ],
}

_MARKERS: dict[str, set[str]] = {
    "fr": {
        "le", "la", "les", "un", "une", "des", "je", "vous", "tu", "est", "et", "pour",
        "avec", "quel", "quelle", "combien", "bonjour", "salut", "comment", "pouvez",
        "peux", "besoin", "veux", "voudrais", "mon", "ma", "mes", "que", "qui", "pas",
        "sur", "dans", "ou", "oui", "non", "merci", "site", "boutique", "magasin",
        "entreprise", "ai", "suis", "avez", "faites", "travaillez", "propose",
    },
    "en": {
        "the", "a", "an", "i", "you", "is", "are", "and", "for", "with", "what", "how",
        "need", "want", "can", "do", "does", "hello", "hi", "my", "your", "to", "of",
        "in", "on", "have", "who", "where", "which", "would", "like", "store", "shop",
    },
}


def t(locale: str, key: str, **kwargs: object) -> str:
    table = MESSAGES.get(locale) or MESSAGES["en"]
    return (table.get(key) or MESSAGES["en"][key]).format(**kwargs)


def normalize_locale(value: str | None) -> str | None:
    if not value:
        return None
    base = value.strip().lower().split("-")[0]
    return base if base in SUPPORTED else None


def detect_language(text: str) -> str | None:
    """Return 'en'/'fr' on a clear signal, else None (caller falls back)."""
    words = re.findall(r"[a-zà-ÿ']+", text.lower())
    scores = {lang: sum(1 for w in words if w in markers) for lang, markers in _MARKERS.items()}
    scores["fr"] += min(sum(text.count(c) for c in "éèêàçùâîôû"), 3)
    best = max(scores, key=scores.get)
    others = [v for k, v in scores.items() if k != best]
    if scores[best] == 0 or scores[best] == max(others, default=0):
        return None
    return best
