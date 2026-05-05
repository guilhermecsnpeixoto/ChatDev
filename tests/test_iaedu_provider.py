from __future__ import annotations

import json

from entity.configs.node.agent import AgentConfig
from entity.messages import Message, MessageRole
from runtime.node.agent.providers.iaedu_provider import IAeduProvider


class DummyResponse:
    def __init__(self, payload, *, content_type="application/json"):
        self._payload = payload
        self.headers = {"Content-Type": content_type}
        self.text = payload if isinstance(payload, str) else json.dumps(payload)

    def raise_for_status(self):
        return None

    def json(self):
        if isinstance(self._payload, str):
            return json.loads(self._payload)
        return self._payload

    def iter_lines(self, decode_unicode=False):
        yield from ()


class DummySession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, url, headers=None, files=None, timeout=None, stream=None):
        self.calls.append(
            {
                "url": url,
                "headers": headers,
                "files": files,
                "timeout": timeout,
                "stream": stream,
            }
        )
        return self.response


def _provider(**params):
    config = AgentConfig.from_dict(
        {
            "provider": "iaedu",
            "base_url": "https://api.iaedu.pt/agent-chat//api/v1/agent/cmamvd3n40000c801qeacoad2/stream",
            "name": "iaedu-model",
            "api_key": "test-key",
            "params": params,
        },
        path="test.agent",
    )
    return IAeduProvider(config)


def test_iaedu_provider_builds_multipart_request():
    provider = _provider(channel_id="channel-1", thread_id="thread-1")
    response = DummyResponse({"message": "hello from iaedu"})
    session = DummySession(response)

    result = provider.call_model(
        session,
        [Message(role=MessageRole.USER, content="What is the value of X?")],
        timeline=[],
    )

    assert result.message.content == "hello from iaedu"
    assert session.calls[0]["url"].endswith("/stream")
    assert session.calls[0]["headers"] == {"x-api-key": "test-key"}
    assert session.calls[0]["timeout"] == 300
    assert session.calls[0]["stream"] is True

    files = session.calls[0]["files"]
    assert files["channel_id"] == (None, "channel-1")
    assert files["thread_id"] == (None, "thread-1")
    assert files["user_info"] == (None, "{}")
    assert files["message"] == (None, "What is the value of X?")


def test_iaedu_provider_uses_explicit_message_and_user_context():
    provider = _provider(
        channel_id="channel-1",
        thread_id="thread-1",
        message="Override message",
        user_info={"name": "Ada"},
        user_context={"course": "math"},
        user_id="user-9",
    )
    response = DummyResponse({"content": "ok"})
    session = DummySession(response)

    provider.call_model(
        session,
        [Message(role=MessageRole.USER, content="ignored")],
        timeline=[],
    )

    files = session.calls[0]["files"]
    assert files["message"] == (None, "Override message")
    assert files["user_info"] == (None, json.dumps({"name": "Ada"}))
    assert files["user_context"] == (None, json.dumps({"course": "math"}))
    assert files["user_id"] == (None, "user-9")