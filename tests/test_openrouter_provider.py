from __future__ import annotations

from entity.configs.node.agent import AgentConfig
from runtime.node.agent.providers.openrouter_provider import OpenRouterProvider


def _provider(*, base_url: str | None = None, api_key: str | None = None, params=None):
    data = {
        "provider": "openrouter",
        "name": "nvidia/nemotron-3-super-120b-a12b:free",
        "params": params or {},
    }
    if base_url is not None:
        data["base_url"] = base_url
    if api_key is not None:
        data["api_key"] = api_key

    cfg = AgentConfig.from_dict(data, path="test.openrouter")
    return OpenRouterProvider(cfg)


def test_openrouter_create_client_uses_env_defaults(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-env-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.local/v1")
    monkeypatch.setenv("OPENROUTER_HTTP_REFERER", "https://chatdev.local")
    monkeypatch.setenv("OPENROUTER_APP_NAME", "ChatDev")

    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "runtime.node.agent.providers.openrouter_provider.OpenAI",
        FakeOpenAI,
    )

    provider = _provider(base_url=None, api_key=None)
    provider.create_client()

    assert captured["api_key"] == "or-env-key"
    assert captured["base_url"] == "https://openrouter.local/v1"
    assert captured["default_headers"]["HTTP-Referer"] == "https://chatdev.local"
    assert captured["default_headers"]["X-Title"] == "ChatDev"


def test_openrouter_extracts_reasoning_tokens_from_usage():
    provider = _provider(base_url="https://openrouter.ai/api/v1", api_key="or-key")

    class DummyResponse:
        usage = {
            "prompt_tokens": 11,
            "completion_tokens": 22,
            "total_tokens": 33,
            "reasoning_tokens": 7,
        }

    usage = provider.extract_token_usage(DummyResponse())
    assert usage.input_tokens == 11
    assert usage.output_tokens == 22
    assert usage.total_tokens == 33
    assert usage.metadata["reasoning_tokens"] == 7