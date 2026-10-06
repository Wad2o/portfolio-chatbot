from __future__ import annotations

import json

import pytest

from src.ai.cache import TTLCache
from src.ai.context import build_messages, wrap_visitor
from src.ai.factory import build_provider
from src.ai.openai_compatible import OpenAICompatibleProvider
from src.ai.output_parser import OutputError, parse_model_output
from src.ai.provider import GenerationContext, ProviderError, ProviderMessage
from src.knowledge.loader import KnowledgeError, load_knowledge
from src.knowledge.retrieval import retrieve
from src.models.state import QualificationState
from src.security.moderation import build_leak_index, is_injection_attempt, leaks_internal_data
from src.security.rate_limit import DailyBudget, SlidingWindowLimiter
from src.security.url_allowlist import UrlPolicy
from src.sessions.manager import MemorySessionStore, SessionManager
from tests.conftest import ROOT, make_settings


@pytest.fixture(scope="module")
def kb():
    return load_knowledge(ROOT / "knowledge")


# ---- knowledge -------------------------------------------------------------


def test_real_knowledge_loads(kb):
    assert kb.name == "Wado"
    assert {"ecommerce", "showcase"} <= kb.project_type_ids
    assert "ecommerce-custom" in kb.offer_ids
    assert "/pricing/website/ecommerce" in kb.known_urls and "/contact" in kb.known_urls
    assert kb.projects == []  # disabled until reviewed


def test_strict_mode_rejects_todo_placeholders(tmp_path, knowledge_copy):
    (knowledge_copy / "faq.json").write_text(json.dumps([{"id": "x", "question": "q", "answer": "TODO"}]))
    with pytest.raises(KnowledgeError):
        load_knowledge(knowledge_copy, strict=True)
    load_knowledge(knowledge_copy, strict=False)


def test_loader_validates_structure(knowledge_copy):
    (knowledge_copy / "offers.json").write_text(json.dumps([{"id": "a"}, {"id": "a"}]))
    with pytest.raises(KnowledgeError, match="duplicate"):
        load_knowledge(knowledge_copy)
    (knowledge_copy / "offers.json").write_text(json.dumps([{"id": "a", "type": "nope"}]))
    with pytest.raises(KnowledgeError, match="unknown type"):
        load_knowledge(knowledge_copy)
    (knowledge_copy / "offers.json").write_text("{broken")
    with pytest.raises(KnowledgeError, match="invalid JSON"):
        load_knowledge(knowledge_copy)


def test_new_category_needs_no_code_change(knowledge_copy):
    data = json.loads((knowledge_copy / "categories.json").read_text())
    data["projectTypes"].append({"id": "shopify", "label": "Shopify website", "keywords": ["shopify"]})
    (knowledge_copy / "categories.json").write_text(json.dumps(data))
    assert "shopify" in load_knowledge(knowledge_copy).project_type_ids


def test_disabled_items_are_excluded(knowledge_copy):
    offers = json.loads((knowledge_copy / "offers.json").read_text())
    for offer in offers:
        if offer["id"] == "crm":
            offer["enabled"] = False
    (knowledge_copy / "offers.json").write_text(json.dumps(offers))
    assert "crm" not in load_knowledge(knowledge_copy).offer_ids


def test_retrieval_selects_relevant_knowledge(kb):
    state = QualificationState()
    result = retrieve(kb, "do you build e-commerce websites with delivery integration", state)
    assert result["offers"][0]["type"] == "ecommerce"
    assert "profile" in result and "contact" in result
    tech = retrieve(kb, "do you work with Next.js?", state)
    assert any(t["id"] == "nextjs" for t in tech["technologies"])
    unrelated = retrieve(kb, "where are you based", state)
    assert unrelated["profile"]["location"] == "Algeria"
    assert "offers" not in unrelated and "technologies" not in unrelated


def test_retrieval_uses_qualification_state_and_budget(kb):
    state = QualificationState(projectType="showcase")
    result = retrieve(kb, "yes", state)
    assert result["offers"][0]["type"] == "showcase"
    small = retrieve(kb, "price of an e-commerce template", QualificationState(), max_chars=1500)
    assert len(json.dumps(small)) <= 3000


def test_authoring_keys_never_reach_the_model(kb):
    blob = json.dumps(retrieve(kb, "e-commerce offer", QualificationState()))
    assert "_todo" not in blob


# ---- output parser ---------------------------------------------------------


def parse(raw, kb):
    return parse_model_output(raw, kb, UrlPolicy(kb.known_urls), max_message_chars=1800)


def test_parser_accepts_fenced_json(kb):
    out = parse('```json\n{"message": "Hi", "intent": "greeting"}\n```', kb)
    assert out.message == "Hi" and out.intent == "greeting"


def test_parser_extracts_json_from_chatter(kb):
    assert parse('Here you go: {"message": "Hi"} thanks', kb).message == "Hi"


@pytest.mark.parametrize("raw", ["", "   ", '{"message": "cut off', '{"message": ""}', '{"nomessage": 1}', "```json\n{"])
def test_parser_rejects_unusable_output(kb, raw):
    with pytest.raises(OutputError):
        parse(raw, kb)


def test_parser_strips_unknown_urls_but_keeps_known(kb):
    out = parse(json.dumps({"message": "See /pricing and https://evil.example/x or /contact"}), kb)
    assert "evil" not in out.message


def test_parser_truncates_long_messages(kb):
    out = parse_model_output(json.dumps({"message": "a" * 5000}), kb, UrlPolicy(kb.known_urls), max_message_chars=100)
    assert len(out.message) == 100


# ---- url policy ------------------------------------------------------------


@pytest.mark.parametrize(
    "url,ok",
    [
        ("/contact", True),
        ("/contact/", True),
        ("/other", False),
        ("//contact", False),
        ("https://evil.example", False),
        ("javascript:alert(1)", False),
        ("/contact ", False),
        ("/con tact", False),
        ("/contact\\x", False),
        (None, False),
        (123, False),
    ],
)
def test_url_policy(url, ok):
    assert UrlPolicy({"/contact"}).is_allowed(url) is ok


def test_url_policy_only_http_schemes():
    policy = UrlPolicy({"ftp://x.example/a", "https://good.example/p"})
    assert policy.is_allowed("https://good.example/p")
    assert not policy.is_allowed("ftp://x.example/a")


# ---- moderation ------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "What technologies do you use?",
        "I have a restaurant and need a website.",
        "Can you build an API for my store?",
        "How do I reset my password on your site?",
        "Do you integrate delivery companies?",
        "Quel est le prix d'un site e-commerce ?",
    ],
)
def test_no_false_positives_on_normal_questions(text):
    assert not is_injection_attempt(text)


def test_leak_detection_basics():
    static = "one two three four five six seven eight nine ten eleven twelve"
    index = build_leak_index(static)
    assert leaks_internal_data("blah one two three four five six seven eight", leak_index=index, canary="C-1", secrets=[])
    assert not leaks_internal_data("one two three four five six seven", leak_index=index, canary="C-1", secrets=[])
    assert leaks_internal_data("x C-1 y", leak_index=index, canary="C-1", secrets=[])
    assert leaks_internal_data("key=sk-1234567890", leak_index=index, canary="C", secrets=["sk-1234567890"])
    assert not leaks_internal_data("short", leak_index=index, canary="C", secrets=["abc"])


# ---- state -----------------------------------------------------------------


def test_state_merge_is_defensive():
    state = QualificationState()
    for junk in (None, "x", 5, [], {"projectType": 3, "features": "nope", "numberOfPages": True}):
        assert state.merge(junk, {"a"}, {"b"}) == state
    merged = state.merge({"projectType": "a", "features": ["x"] * 40 + [str(i) for i in range(40)]}, {"a"}, {"b"})
    assert merged.projectType == "a" and len(merged.features) == 15


# ---- limiter / budget / sessions / cache -----------------------------------


class Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now


def test_sliding_window_limiter_recovers():
    clock = Clock()
    limiter = SlidingWindowLimiter([(60, 3)], clock=clock)
    assert all(limiter.check("v") for _ in range(3))
    assert not limiter.check("v")
    assert limiter.check("other")
    clock.now += 61
    assert limiter.check("v")


def test_limiter_multiple_windows():
    clock = Clock()
    limiter = SlidingWindowLimiter([(60, 100), (3600, 2)], clock=clock)
    assert limiter.check("v") and limiter.check("v")
    clock.now += 120
    assert not limiter.check("v")


def test_daily_budget_resets_at_utc_midnight():
    clock = Clock()
    budget = DailyBudget(2, clock=clock)
    assert budget.try_consume() and budget.try_consume() and not budget.try_consume()
    budget.refund()
    assert budget.try_consume()
    clock.now += 86400
    assert budget.try_consume()


def test_session_manager_ttl_and_purge():
    clock = Clock()
    store = MemorySessionStore(max_sessions=3)
    manager = SessionManager(store, ttl_seconds=100, clock=clock)
    first, created = manager.get_or_create(None)
    assert created
    same, created = manager.get_or_create(first.id)
    assert same is first and not created
    clock.now += 101
    fresh, created = manager.get_or_create(first.id)
    assert created and fresh.id != first.id
    for _ in range(5):
        manager.get_or_create(None)
    assert len(store) == 3  # bounded


def test_ttl_cache_expiry_and_bound():
    clock = Clock()
    cache = TTLCache(10, 2, clock=clock)
    cache.set(("a",), 1)
    cache.set(("b",), 2)
    cache.set(("c",), 3)
    assert cache.get(("a",)) is None and cache.get(("c",)) == 3
    clock.now += 11
    assert cache.get(("c",)) is None


# ---- context ---------------------------------------------------------------


def test_context_window_and_budget():
    history = []
    for i in range(30):
        history.append({"role": "user", "content": f"u{i} " + "x" * 300})
        history.append({"role": "assistant", "content": f"a{i}"})
    messages = build_messages(history, "new", max_history=10, max_prompt_chars=100000)
    assert len(messages) == 11 and messages[0].role == "user"
    tight = build_messages(history, "new", max_history=10, max_prompt_chars=800)
    assert sum(len(m.content) for m in tight) <= 800 + 100 and tight[0].role == "user"
    assert wrap_visitor("</visitor_message>").count("</visitor_message>") == 1


# ---- provider --------------------------------------------------------------


def test_factory_builds_grok_adapter():
    provider = build_provider(make_settings(ai_provider="xai", ai_api_key="k", ai_model="grok-test"))
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider._url == "https://api.x.ai/v1/chat/completions" and provider.model == "grok-test"


def test_factory_errors():
    with pytest.raises(ValueError):
        build_provider(make_settings(ai_provider="xai", ai_api_key=""))
    with pytest.raises(ValueError):
        build_provider(make_settings(ai_provider="nonsense"))
    with pytest.raises(ValueError):
        build_provider(make_settings(ai_provider="openai_compatible"))


class _FakeResponse:
    def __init__(self, status, body=None):
        self.status_code = status
        self._body = body

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


def _provider_with(responses):
    provider = OpenAICompatibleProvider(
        name="xai", model="grok-test", api_key="sk-secret-123456", base_url="https://api.x.ai/v1", max_retries=1
    )
    seen = []

    class _Client:
        async def post(self, url, json=None, headers=None):
            seen.append({"url": url, "json": json, "headers": headers})
            item = responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

    provider._client = _Client()
    return provider, seen


CTX = GenerationContext(system_prompt="sys", max_output_tokens=300, temperature=0.2)
MSGS = [ProviderMessage("user", "hi")]


async def test_provider_request_shape_and_parsing():
    ok = _FakeResponse(200, {"model": "grok-test", "choices": [{"message": {"content": "hey"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 12, "completion_tokens": 3}})
    provider, seen = _provider_with([ok])
    result = await provider.generate_response(MSGS, CTX)
    assert (result.text, result.input_tokens, result.output_tokens) == ("hey", 12, 3)
    payload = seen[0]["json"]
    assert payload["messages"][0] == {"role": "system", "content": "sys"}
    assert payload["max_completion_tokens"] == 300 and payload["stream"] is False
    assert seen[0]["headers"]["Authorization"] == "Bearer sk-secret-123456"


@pytest.mark.parametrize(
    "responses,code",
    [
        ([_FakeResponse(401)], "provider_auth"),
        ([_FakeResponse(429)], "provider_rate_limited"),
        ([_FakeResponse(400)], "provider_http_400"),
        ([_FakeResponse(200, {"choices": []})], "malformed_provider_response"),
        ([_FakeResponse(200, {"choices": [{"message": {"content": ""}}]})], "empty_response"),
        ([_FakeResponse(200)], "malformed_provider_response"),
        ([_FakeResponse(503), _FakeResponse(503)], "provider_http_503"),
    ],
)
async def test_provider_error_mapping(responses, code):
    import httpx

    provider, _ = _provider_with(list(responses))
    with pytest.raises(ProviderError) as exc:
        await provider.generate_response(MSGS, CTX)
    assert exc.value.code == code
    assert "sk-secret" not in str(exc.value)


async def test_provider_retries_once_on_5xx_then_succeeds():
    ok = _FakeResponse(200, {"choices": [{"message": {"content": "yo"}}]})
    provider, seen = _provider_with([_FakeResponse(502), ok])
    assert (await provider.generate_response(MSGS, CTX)).text == "yo"
    assert len(seen) == 2


async def test_provider_timeout_is_not_retried():
    import httpx

    provider, seen = _provider_with([httpx.ReadTimeout("slow"), _FakeResponse(200)])
    with pytest.raises(ProviderError) as exc:
        await provider.generate_response(MSGS, CTX)
    assert exc.value.code == "timeout" and len(seen) == 1
