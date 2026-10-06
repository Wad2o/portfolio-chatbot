from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.ai.provider import AIProvider, ProviderError, ProviderResult
from src.config.settings import Settings
from src.main import create_app

ROOT = Path(__file__).resolve().parent.parent
SECRET = "test-secret-0123456789abcdef"
HEADERS = {"X-Portfolio-Secret": SECRET}


class ScriptedProvider(AIProvider):
    """Returns queued outputs and records what the pipeline sent."""

    name = "scripted"
    model = "scripted-1"

    def __init__(self) -> None:
        self.outputs: list[object] = []
        self.calls: list[dict] = []

    def queue(self, output: object) -> None:
        self.outputs.append(output)

    async def generate_response(self, messages, context):
        self.calls.append({"messages": messages, "system": context.system_prompt})
        if not self.outputs:
            raise AssertionError("ScriptedProvider has no queued output")
        item = self.outputs.pop(0)
        if isinstance(item, Exception):
            raise item
        text = item if isinstance(item, str) else json.dumps(item)
        return ProviderResult(text=text, input_tokens=100, output_tokens=40, model=self.model)


def reply(message="Hello!", **overrides) -> dict:
    base = {
        "message": message,
        "suggestions": [],
        "actions": [],
        "intent": "unknown",
        "recommendedOffer": None,
        "qualificationStage": "none",
        "stateUpdate": {},
    }
    base.update(overrides)
    return base


def make_settings(**overrides) -> Settings:
    values = dict(
        app_env="test",
        ai_provider="mock",
        portfolio_api_secret=SECRET,
        portfolio_origin="http://localhost:3000",
        knowledge_dir=str(ROOT / "knowledge"),
        rate_limit_per_minute=1000,
        visitor_daily_limit=100000,
    )
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.fixture
def provider() -> ScriptedProvider:
    return ScriptedProvider()


@pytest.fixture
def make_client(provider):
    clients = []

    def _make(use_scripted: bool = True, **settings_overrides) -> TestClient:
        app = create_app(make_settings(**settings_overrides), provider=provider if use_scripted else None)
        client = TestClient(app, headers=HEADERS)
        clients.append(client)
        return client

    yield _make
    for client in clients:
        client.close()


@pytest.fixture
def client(make_client) -> TestClient:
    return make_client()


@pytest.fixture
def knowledge_copy(tmp_path) -> Path:
    target = tmp_path / "knowledge"
    shutil.copytree(ROOT / "knowledge", target)
    return target
