"""OpenRouter provider implementation."""

from __future__ import annotations

import os
from typing import Any, Dict

from openai import OpenAI

from runtime.node.agent.providers.openai_provider import OpenAIProvider


class OpenRouterProvider(OpenAIProvider):
    """OpenRouter provider using OpenAI-compatible Chat/Responses APIs."""

    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    def create_client(self):
        """Create and return the OpenRouter client (OpenAI-compatible)."""
        api_key = (self.api_key or os.getenv("OPENROUTER_API_KEY") or "").strip()
        if not api_key:
            raise ValueError(
                "OpenRouter provider requires api_key or OPENROUTER_API_KEY environment variable"
            )

        base_url = (
            self.base_url
            or os.getenv("OPENROUTER_BASE_URL")
            or self.DEFAULT_BASE_URL
        )
        base_url = str(base_url).strip() or self.DEFAULT_BASE_URL

        headers = self._build_default_headers()
        if headers:
            return OpenAI(api_key=api_key, base_url=base_url, default_headers=headers)
        return OpenAI(api_key=api_key, base_url=base_url)

    def extract_token_usage(self, response: Any):
        """Extract token usage and preserve OpenRouter reasoning token fields."""
        usage = super().extract_token_usage(response)
        raw_usage = getattr(response, "usage", None)
        reasoning_tokens = self._read_reasoning_tokens(raw_usage)
        if reasoning_tokens is not None:
            usage.metadata["reasoning_tokens"] = reasoning_tokens
        return usage

    def _track_token_usage(self, response: Any) -> None:
        """Record token usage using provider name openrouter."""
        token_tracker = getattr(self.config, "token_tracker", None)
        if not token_tracker:
            return

        usage = self.extract_token_usage(response)
        if usage.input_tokens == 0 and usage.output_tokens == 0 and not usage.metadata:
            return

        node_id = getattr(self.config, "node_id", "ALL")
        usage.node_id = node_id
        usage.model_name = self.model_name
        usage.workflow_id = token_tracker.workflow_id
        usage.provider = "openrouter"

        token_tracker.record_usage(node_id, self.model_name, usage, provider="openrouter")

    def _build_default_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        params = self.params or {}

        referer = params.get("http_referer") or os.getenv("OPENROUTER_HTTP_REFERER")
        app_name = (
            params.get("x_title")
            or params.get("app_name")
            or os.getenv("OPENROUTER_APP_NAME")
        )

        if referer:
            headers["HTTP-Referer"] = str(referer)
        if app_name:
            headers["X-Title"] = str(app_name)
        return headers

    def _read_reasoning_tokens(self, raw_usage: Any) -> int | None:
        if raw_usage is None:
            return None

        def _get(payload: Any, key: str) -> Any:
            if isinstance(payload, dict):
                return payload.get(key)
            if hasattr(payload, key):
                return getattr(payload, key)
            return None

        candidates = [
            _get(raw_usage, "reasoning_tokens"),
            _get(raw_usage, "reasoningTokens"),
        ]

        completion_details = _get(raw_usage, "completion_tokens_details")
        candidates.append(_get(completion_details, "reasoning_tokens"))
        candidates.append(_get(completion_details, "reasoningTokens"))

        for value in candidates:
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return None