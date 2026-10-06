# Portfolio chatbot API

Independent FastAPI service that powers the assistant on the portfolio site.
The portfolio only contains the chat UI and a server-side proxy route; all logic,
knowledge, security, sessions and the AI provider live here.

```
Browser -> Portfolio /api/chat (Next.js server) -> this API -> AI provider (Grok)
```

## Run locally

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env          # AI_PROVIDER=mock works with no API key
uvicorn src.main:app --reload --port 8000
pytest
```

Docs are at `/docs` outside production.

## Use Grok (xAI)

```env
AI_PROVIDER=xai
AI_MODEL=grok-4-fast-non-reasoning   # confirm the exact name in console.x.ai
AI_API_KEY=xai-...
```

Prefer a **non-reasoning** model: reasoning tokens eat the small output budget and can
return an empty answer (the API then reports a generic "unavailable"). Set
`PRICE_PER_MILLION_INPUT_TOKENS` / `PRICE_PER_MILLION_OUTPUT_TOKENS` from the price list
to get estimated costs in the logs. Other OpenAI-compatible providers work by changing
`AI_PROVIDER` (`openai`, `openrouter`, `openai_compatible` + `AI_BASE_URL`).

## API

`POST /api/v1/chat` with header `X-Portfolio-Secret` (server-to-server only) and
optionally `X-Visitor-Id` (hash of the visitor's IP, forwarded by the portfolio proxy so
rate limits are per visitor rather than per proxy).

```json
{ "message": "I own a clothing store", "sessionId": null, "locale": "en" }
```

Response: `message`, `suggestions[]`, `actions[]` (`navigate` / `configure` / `contact`),
`metadata` (`intent`, `recommendedOffer`, `qualificationStage`, `language`),
`qualification` (structured project state) and `sessionId` (store it, send it back).
Errors: `{ "error": { "code", "message" } }` with 400/401/413/429/503.
`GET /health` returns `{ "status": "ok" }`.

Other endpoints (same secret header, no AI cost):
- `POST /api/v1/feedback` `{ messageId, feedback: "up" | "down" }` logs only those two values.
- `POST /api/v1/events` `{ event, label? }` for the 7 anonymous analytics events; never message text.
- `POST /api/v1/contact/summary` `{ sessionId, locale }` returns a readable project summary built from the
  structured qualification state, plus the values to preselect in the configurator.

Feedback and events are written to the structured logs. Send the logs to your log service, or
replace the two `log.info` calls in `src/api/v1/extras.py` with a database insert (Phase 8).

## Knowledge (`knowledge/*.json`)

Source of truth for the assistant. Edit by hand; no code change needed.
- Any item can have `"enabled": false` (excluded) and `keywords`/`tags` (used by retrieval).
- Keys starting with `_` (e.g. `_todo`) are author notes and are never sent to the model.
- `categories.json` defines project types and their qualification questions. Add a
  service by adding a project type here and an offer in `offers.json`.
- Every `*Url` field is the allowlist of links the assistant may output.
- In production the app refuses to start while any value contains `TODO`.

**Before going live, fill in:** `profile.json` (availability, remote work, experience),
`contact.json`, `projects.json` (all disabled until you review them), `process.json`,
and check every offer in `offers.json` (features, trade-offs, which types really come in
template and custom versions). The assistant says "I don't have that information" for
anything missing; it is never allowed to invent prices, timelines or experience.

## Security model

- Server-to-server secret, constant-time comparison; strict CORS allowlist; no `*`.
- Body size limit (also for chunked uploads), message length, session and rate limits,
  global daily call budget, output token cap, request timeout, bounded history.
- Obvious injection / extraction attempts are answered locally and never reach the model.
- Visitor text is wrapped as data; the model output is parsed, then every field is
  validated: unknown action types, non-allowlisted URLs, HTML and invalid state updates
  are dropped.
- Outputs that echo the system prompt, a canary marker, a secret or an encoded blob are
  replaced with a refusal.
- Logs contain request id, status, duration, model, tokens and error types only.
  No message text, prompts, keys or headers.

## Deploy

Provider-agnostic: any host that runs `uvicorn src.main:app --host 0.0.0.0 --port $PORT`.
Set `APP_ENV=production` plus the variables from `.env.example`. Sessions, rate limits and
the cache are in memory, so run **one worker** until the Redis stores (Phase 8) exist.

## Status

Phases 1-7 are implemented (backend, knowledge, chat UI in `../portfolio-frontend`, offer flow,
qualification state, configurator/contact handoff, analytics, feedback).
Phase 8 (Redis, database, RAG, admin dashboard) is intentionally not built.
Offer and qualification quality depends on the real data you put in `knowledge/`.
