from __future__ import annotations

import re

from .validation import normalize

# Matched against normalised text (lowercase, no accents). Deliberately narrow:
# blatant extraction/override attempts are answered with a canned refusal and
# never reach the model. Subtle ones are handled by the system prompt.
_VERBS = (
    r"(?:show|reveal|print|display|repeat|give|tell|output|leak|dump|share|send|"
    r"what(?:'s| is| are)|donne|montre|affiche|revele|repete|partage|envoie|"
    r"quelle? (?:est|sont)|c'est quoi)"
)
_SECRETS = (
    r"(?:api[ _-]?keys?|secret keys?|access tokens?|environment variables?|env vars?|"
    r"\.env\b|credentials|passwords?|cles? (?:d')?api|variables? d'environnement|"
    r"mots? de passe)"
)

_INJECTION_PATTERNS = [
    r"(?:ignore|disregard|forget|override)\s+(?:all\s+|any\s+|the\s+|your\s+)*"
    r"(?:previous|prior|above|earlier|system|initial)?\s*(?:instructions?|prompts?|rules|guidelines)",
    r"ignore[zr]?\s+(?:toutes?\s+|les\s+|vos\s+|tes\s+)*(?:instructions?|consignes?|regles)",
    r"oublie[zr]?\s+(?:toutes?\s+|les\s+|vos\s+|tes\s+)*(?:instructions?|consignes?)",
    r"\b(?:system|developer|hidden|initial|internal)\s+(?:prompt|message|instructions?)\b",
    r"\bprompt\s+(?:systeme|initial|cache)\b",
    r"\bmessage\s+systeme\b",
    r"repeat\s+(?:everything|all|the\s+text|the\s+words)\s+(?:above|before)",
    r"repete[zr]?\s+(?:tout|le\s+texte)\s+(?:ci-dessus|au-dessus|precedent)",
    r"(?:you\s+are|you're|act\s+as|pretend\s+(?:to\s+be|you\s+are))\s+(?:now\s+)?"
    r"(?:chatgpt|gpt|dan|claude|gemini|grok|an?\s+(?:unrestricted|uncensored|different|new)\b)",
    r"(?:enter|enable|activate|switch\s+to|passe[zr]?\s+en)\s+(?:developer|dev|god|jailbreak|dan|debug|"
    r"developpeur)\s+mode",
    r"\b(?:developer|jailbreak|god|dan)\s+mode\b",
    r"\bmode\s+developpeur\b",
    rf"{_VERBS}[^.?!\n]{{0,40}}{_SECRETS}",
    r"(?:encode|base64|rot13|hex|reverse|translate|traduis|encode[zr]?)\s+(?:your|the|ton|votre|tes|vos)\s+"
    r"(?:prompt|instructions?|rules|consignes?)",
]
_INJECTION_RE = [re.compile(p) for p in _INJECTION_PATTERNS]

_BASE64ISH_RE = re.compile(r"[A-Za-z0-9+/_-]{60,}={0,2}")
_HEX_RE = re.compile(r"\b[0-9a-fA-F]{48,}\b")


def is_injection_attempt(text: str) -> bool:
    norm = normalize(text)
    return any(rx.search(norm) for rx in _INJECTION_RE)


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", normalize(text))


def build_leak_index(system_text: str, n: int = 8) -> set[tuple[str, ...]]:
    words = _words(system_text)
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def leaks_internal_data(
    output: str,
    *,
    leak_index: set[tuple[str, ...]],
    canary: str,
    secrets: list[str],
    n: int = 8,
) -> bool:
    """True if the output appears to expose the system prompt or a secret."""
    if canary and canary in output:
        return True
    for secret in secrets:
        if secret and len(secret) >= 8 and secret in output:
            return True
    if _BASE64ISH_RE.search(output) or _HEX_RE.search(output):
        return True
    words = _words(output)
    return any(tuple(words[i : i + n]) in leak_index for i in range(len(words) - n + 1))
