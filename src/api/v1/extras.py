"""Feedback, anonymous analytics events and the contact summary.

None of these call the AI provider. None store chat text."""
from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field

from ...i18n import detect_language, normalize_locale, t
from ...security.auth import require_portfolio_secret, visitor_key
from ..errors import ApiError

log = logging.getLogger("chatbot.events")
router = APIRouter(prefix="/api/v1")

EventName = Literal[
    "chat_opened",
    "chat_message_sent",
    "offer_flow_started",
    "offer_flow_completed",
    "project_selected",
    "pricing_clicked",
    "contact_clicked",
]


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    messageId: str = Field(pattern=r"^msg_[A-Za-z0-9_-]{8,32}$")
    feedback: Literal["up", "down"]


class EventRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event: EventName
    label: str | None = Field(default=None, pattern=r"^[a-z0-9-]{1,40}$")


class SummaryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sessionId: str = Field(max_length=128)
    locale: str | None = Field(default=None, max_length=10)


def _check_rate(request: Request) -> None:
    c = request.app.state.container
    if not c.aux_limiter.check(visitor_key(request)):
        raise ApiError(429, "rate_limited", t("en", "rate_limited"))


@router.post("/feedback")
async def feedback(payload: FeedbackRequest, request: Request, _auth: None = Depends(require_portfolio_secret)) -> dict:
    _check_rate(request)
    # Only the message id and the vote are recorded, never the conversation.
    log.info("feedback", extra={"request_id": request.state.request_id, "message_id": payload.messageId, "feedback": payload.feedback})
    return {"ok": True}


@router.post("/events")
async def events(payload: EventRequest, request: Request, _auth: None = Depends(require_portfolio_secret)) -> dict:
    _check_rate(request)
    log.info("analytics", extra={"request_id": request.state.request_id, "event": payload.event, "label": payload.label})
    return {"ok": True}


@router.post("/contact/summary")
async def contact_summary(payload: SummaryRequest, request: Request, _auth: None = Depends(require_portfolio_secret)) -> dict:
    """Readable project summary built from the structured state (not from chat text)."""
    _check_rate(request)
    c = request.app.state.container
    locale = normalize_locale(payload.locale) or "en"
    session = c.sessions.peek(payload.sessionId)
    if session is None:
        raise ApiError(404, "session_not_found", t(locale, "generic_error"))
    q = session.qualification
    if q.is_empty():
        raise ApiError(400, "no_project_info", t(locale, "no_project_info"))

    ptype = c.kb.project_type(q.projectType)
    title = (ptype or {}).get("label") or q.projectType or ""
    if q.developmentType:
        title = f"{title} ({t(locale, 'summary_dev_' + q.developmentType)})" if q.developmentType in c.kb.development_type_ids and q.developmentType in ("template", "custom") else title

    lines = [f"{t(locale, 'summary_project')}:", title or "-", ""]
    if q.features:
        lines += [f"{t(locale, 'summary_requirements')}:"] + [f"- {f}" for f in q.features] + [""]
    if q.numberOfPages:
        lines += [f"{t(locale, 'summary_pages')}: {q.numberOfPages}", ""]
    if q.integrations:
        lines += [f"{t(locale, 'summary_integrations')}:"] + [f"- {i}" for i in q.integrations] + [""]
    if q.additionalNotes:
        lines += [f"{t(locale, 'summary_notes')}:", q.additionalNotes, ""]

    return {
        "summary": "\n".join(lines).strip(),
        "qualification": q.model_dump(),
        "configure": {k: v for k, v in {"type": q.projectType, "development": q.developmentType}.items() if v},
        "contactUrl": c.kb.contact.get("contactUrl", "/contact"),
    }
