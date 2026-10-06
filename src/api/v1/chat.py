from __future__ import annotations

import json
import logging
import secrets
import time

from fastapi import APIRouter, Depends, Request

from ...ai.context import build_messages
from ...ai.output_parser import OutputError, parse_model_output
from ...ai.prompt import build_system_prompt
from ...ai.provider import GenerationContext, ProviderError
from ...i18n import DEFAULT_SUGGESTIONS, detect_language, normalize_locale, t
from ...knowledge.retrieval import project_type_guidance, retrieve
from ...models.requests import ChatRequest
from ...models.responses import ChatResponse, Metadata, Suggestion
from ...security.auth import require_portfolio_secret, visitor_key
from ...security.moderation import is_injection_attempt, leaks_internal_data
from ...security.validation import MessageInvalid, normalize, validate_user_message
from ..errors import ApiError

log = logging.getLogger("chatbot.chat")
router = APIRouter(prefix="/api/v1")


def _response(session, *, parsed_like: dict, locale: str) -> ChatResponse:
    return ChatResponse(
        sessionId=session.id,
        messageId="msg_" + secrets.token_urlsafe(12),
        message=parsed_like["message"],
        suggestions=parsed_like.get("suggestions", []),
        actions=parsed_like.get("actions", []),
        metadata=Metadata(
            intent=parsed_like.get("intent", "unknown"),
            recommendedOffer=parsed_like.get("recommendedOffer"),
            qualificationStage=parsed_like.get("stage", "none"),
            language=locale,
        ),
        qualification=session.qualification,
    )


def _refusal(c, session, locale: str) -> ChatResponse:
    labels = DEFAULT_SUGGESTIONS.get(locale) or DEFAULT_SUGGESTIONS["en"]
    return _response(
        session,
        locale=locale,
        parsed_like={
            "message": t(locale, "refusal", name=c.kb.name),
            "suggestions": [Suggestion(label=label, value=value) for label, value in labels],
            "intent": "security_refusal",
        },
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    request: Request,
    _auth: None = Depends(require_portfolio_secret),
) -> ChatResponse:
    c = request.app.state.container
    s = c.settings
    started = time.perf_counter()
    request_id = request.state.request_id
    fallback_locale = normalize_locale(payload.locale) or "en"

    try:
        message = validate_user_message(payload.message, s.max_message_length)
    except MessageInvalid as exc:
        raise ApiError(400, exc.code, t(fallback_locale, exc.code, max=s.max_message_length)) from None

    locale = detect_language(message) or fallback_locale

    if not c.limiter.check(visitor_key(request)):
        raise ApiError(429, "rate_limited", t(locale, "rate_limited"))

    session, _created = c.sessions.get_or_create(payload.sessionId)
    if session.message_count >= s.max_messages_per_session:
        raise ApiError(429, "session_limit", t(locale, "session_limit", name=c.kb.name))

    def finish(response: ChatResponse, **fields) -> ChatResponse:
        c.sessions.save(session)
        log.info(
            "chat",
            extra={
                "request_id": request_id,
                "status": 200,
                "duration_ms": round((time.perf_counter() - started) * 1000),
                "intent": response.metadata.intent,
                **fields,
            },
        )
        return response

    # Blatant extraction/override attempts never reach the model (and cost nothing).
    if is_injection_attempt(message):
        session.message_count += 1
        return finish(_refusal(c, session, locale), blocked="input_injection", model=None)

    cache_key = (locale, normalize(message))
    cacheable = not session.history and session.qualification.is_empty()
    parsed = c.cache.get(cache_key) if cacheable else None
    cached = parsed is not None
    input_tokens = output_tokens = 0
    cost = 0.0

    if parsed is None:
        if not c.daily.try_consume():
            raise ApiError(429, "daily_limit", t(locale, "daily_limit", name=c.kb.name))

        previous_user = [h["content"] for h in session.history if h["role"] == "user"][-1:]
        query_text = " ".join(previous_user + [message])
        retrieved = retrieve(c.kb, query_text, session.qualification, max_chars=s.max_context_chars)
        guidance = project_type_guidance(c.kb, session.qualification, query_text)
        system_prompt = build_system_prompt(c.static_prompt, c.kb, retrieved, session.qualification, guidance)
        messages = build_messages(
            session.history,
            message,
            max_history=s.max_history_messages,
            max_prompt_chars=s.max_prompt_chars,
        )
        context = GenerationContext(
            system_prompt=system_prompt,
            max_output_tokens=s.max_output_tokens,
            temperature=s.ai_temperature,
            request_id=request_id,
        )

        try:
            result = await c.provider.generate_response(messages, context)
        except ProviderError as exc:
            c.daily.refund()
            log.warning("provider_error", extra={"request_id": request_id, "error_type": exc.code, "model": c.provider.model})
            raise ApiError(503, "assistant_unavailable", t(locale, "unavailable")) from None
        except Exception as exc:  # never leak provider internals
            c.daily.refund()
            log.error("provider_unexpected", extra={"request_id": request_id, "error_type": type(exc).__name__})
            raise ApiError(503, "assistant_unavailable", t(locale, "unavailable")) from None

        input_tokens, output_tokens = result.input_tokens, result.output_tokens
        cost = c.usage.record(input_tokens, output_tokens)

        try:
            parsed = parse_model_output(result.text, c.kb, c.urls, max_message_chars=s.max_response_chars)
        except OutputError as exc:
            log.warning("output_rejected", extra={"request_id": request_id, "error_type": str(exc), "model": result.model})
            raise ApiError(503, "assistant_error", t(locale, "generic_error")) from None

        visible_text = " ".join(
            [parsed.message] + [x.label for x in parsed.suggestions] + [x.label for x in parsed.actions]
        )
        if leaks_internal_data(
            visible_text, leak_index=c.leak_index, canary=c.canary, secrets=c.secret_values
        ):
            session.message_count += 1
            return finish(
                _refusal(c, session, locale),
                blocked="output_leak",
                model=result.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost=cost,
            )

        if cacheable:
            c.cache.set(cache_key, parsed)

    session.qualification = session.qualification.merge(
        parsed.state_update, c.kb.project_type_ids, c.kb.development_type_ids
    )
    session.history.append({"role": "user", "content": message})
    session.history.append(
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "message": parsed.message[:600],
                    "intent": parsed.intent,
                    "recommendedOffer": parsed.recommended_offer,
                    "qualificationStage": parsed.stage,
                },
                ensure_ascii=False,
            ),
        }
    )
    session.message_count += 1

    response = _response(
        session,
        locale=locale,
        parsed_like={
            "message": parsed.message,
            "suggestions": parsed.suggestions,
            "actions": parsed.actions,
            "intent": parsed.intent,
            "recommendedOffer": parsed.recommended_offer,
            "stage": parsed.stage,
        },
    )
    return finish(
        response,
        model=c.provider.model,
        cached=cached,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost=cost,
    )
