from __future__ import annotations

import pytest

from tests.conftest import reply


def test_feedback_accepts_valid_and_rejects_junk(client):
    ok = client.post("/api/v1/feedback", json={"messageId": "msg_abcdefghij12", "feedback": "up"})
    assert ok.status_code == 200 and ok.json() == {"ok": True}
    for body in (
        {"messageId": "x", "feedback": "up"},
        {"messageId": "msg_abcdefghij12", "feedback": "meh"},
        {"messageId": "msg_abcdefghij12", "feedback": "up", "text": "full conversation"},
    ):
        assert client.post("/api/v1/feedback", json=body).status_code == 400


def test_feedback_and_events_require_secret(client):
    for path, body in (
        ("/api/v1/feedback", {"messageId": "msg_abcdefghij12", "feedback": "up"}),
        ("/api/v1/events", {"event": "chat_opened"}),
        ("/api/v1/contact/summary", {"sessionId": "A" * 43}),
    ):
        assert client.post(path, json=body, headers={"X-Portfolio-Secret": "nope"}).status_code == 401


@pytest.mark.parametrize("event", ["chat_opened", "chat_message_sent", "offer_flow_started", "offer_flow_completed", "project_selected", "pricing_clicked", "contact_clicked"])
def test_events_whitelisted(client, event):
    assert client.post("/api/v1/events", json={"event": event}).status_code == 200


def test_events_reject_unknown_names_and_free_text(client):
    assert client.post("/api/v1/events", json={"event": "hack"}).status_code == 400
    assert client.post("/api/v1/events", json={"event": "chat_opened", "label": "My private message here!"}).status_code == 400
    assert client.post("/api/v1/events", json={"event": "chat_opened", "message": "hi"}).status_code == 400


def test_summary_from_structured_state(client, provider):
    provider.queue(reply("q", stateUpdate={"projectType": "ecommerce", "developmentType": "custom", "numberOfPages": 8, "features": ["Product catalogue", "Cart"], "integrations": ["Delivery company"], "additionalNotes": "Clothing store"}))
    sid = client.post("/api/v1/chat", json={"message": "I own a clothing store"}).json()["sessionId"]
    res = client.post("/api/v1/contact/summary", json={"sessionId": sid, "locale": "en"})
    assert res.status_code == 200
    data = res.json()
    assert data["summary"].startswith("Project:\nE-commerce website (custom)")
    assert "- Product catalogue" in data["summary"] and "Approximate number of pages: 8" in data["summary"]
    assert data["configure"] == {"type": "ecommerce", "development": "custom"}
    assert data["contactUrl"] == "/contact"
    fr = client.post("/api/v1/contact/summary", json={"sessionId": sid, "locale": "fr"}).json()["summary"]
    assert fr.startswith("Projet:") and "Besoins:" in fr and "sur mesure" in fr


def test_summary_edge_cases(client, provider):
    assert client.post("/api/v1/contact/summary", json={"sessionId": "A" * 43}).status_code == 404
    assert client.post("/api/v1/contact/summary", json={"sessionId": "short"}).status_code == 404
    provider.queue(reply("nothing yet"))
    sid = client.post("/api/v1/chat", json={"message": "hello"}).json()["sessionId"]
    res = client.post("/api/v1/contact/summary", json={"sessionId": sid})
    assert res.status_code == 400 and res.json()["error"]["code"] == "no_project_info"


def test_summary_does_not_leak_other_sessions(client, provider):
    provider.queue(reply("a", stateUpdate={"projectType": "ecommerce", "additionalNotes": "secret of visitor A"}))
    a = client.post("/api/v1/chat", json={"message": "I have a shop"}).json()["sessionId"]
    provider.queue(reply("b", stateUpdate={"projectType": "portfolio"}))
    b = client.post("/api/v1/chat", json={"message": "I am a photographer"}).json()["sessionId"]
    summary_b = client.post("/api/v1/contact/summary", json={"sessionId": b}).json()["summary"]
    assert "secret of visitor A" not in summary_b and "Portfolio" in summary_b
    assert a != b
