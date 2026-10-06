from __future__ import annotations

import json
import re

from ..security.validation import normalize
from .provider import AIProvider, GenerationContext, ProviderMessage, ProviderResult

_TAG_RE = re.compile(r"</?visitor_message>")


class MockProvider(AIProvider):
    """Deterministic, free provider that returns the real JSON schema.
    For local development and tests; it understands a handful of keywords only."""

    name = "mock"
    model = "mock-1"

    async def generate_response(
        self, messages: list[ProviderMessage], context: GenerationContext
    ) -> ProviderResult:
        last = next((m.content for m in reversed(messages) if m.role == "user"), "")
        text = normalize(_TAG_RE.sub("", last))
        reply = self._reply(text)
        raw = json.dumps(reply, ensure_ascii=False)
        return ProviderResult(
            text=raw, input_tokens=len(last) // 4, output_tokens=len(raw) // 4, model=self.model
        )

    @staticmethod
    def _reply(text: str) -> dict:
        shop_words = ("restaurant", "store", "shop", "clothing", "boutique", "magasin", "furniture", "meuble")
        if any(w in text for w in shop_words):
            return {
                "message": "Do you want customers to purchase products directly online, or mainly browse your catalogue?",
                "suggestions": [
                    {"label": "Purchase online", "value": "ecommerce"},
                    {"label": "Catalogue only", "value": "showcase"},
                ],
                "actions": [],
                "intent": "offer_qualification",
                "recommendedOffer": None,
                "qualificationStage": "discovery",
                "stateUpdate": {},
            }
        if "ecommerce" in text or "purchase online" in text or "vendre" in text:
            return {
                "message": "A template-based e-commerce solution may fit if you want a simpler and faster setup. A custom one may be more appropriate if you need custom functionality, integrations or more control over the design.",
                "suggestions": [],
                "actions": [
                    {"type": "configure", "label": "Configure my project", "data": {"type": "ecommerce", "development": "custom"}},
                    {"type": "navigate", "label": "View the pricing", "url": "/pricing/website/ecommerce"},
                ],
                "intent": "offer_recommendation",
                "recommendedOffer": "ecommerce-custom",
                "qualificationStage": "recommendation",
                "stateUpdate": {"projectType": "ecommerce"},
            }
        if any(w in text for w in ("contact", "hire", "embauch", "joindre")):
            return {
                "message": "You can get in touch through the contact page to discuss your project.",
                "suggestions": [],
                "actions": [{"type": "contact", "label": "Contact me"}],
                "intent": "contact",
                "recommendedOffer": None,
                "qualificationStage": "handoff",
                "stateUpdate": {},
            }
        if any(w in text for w in ("technolog", "stack", "next", "react", "python")):
            return {
                "message": "The stack includes Next.js, React, TypeScript, Tailwind CSS, PostgreSQL, Python and FastAPI.",
                "suggestions": [{"label": "Explore projects", "value": "projects"}],
                "actions": [],
                "intent": "skills",
                "recommendedOffer": None,
                "qualificationStage": "none",
                "stateUpdate": {},
            }
        return {
            "message": "I can help you explore the work, understand the services, or find the right solution for your project.",
            "suggestions": [
                {"label": "Find the right offer", "value": "find-offer"},
                {"label": "Explore projects", "value": "projects"},
            ],
            "actions": [{"type": "navigate", "label": "View projects", "url": "/projects"}],
            "intent": "greeting",
            "recommendedOffer": None,
            "qualificationStage": "none",
            "stateUpdate": {},
        }
