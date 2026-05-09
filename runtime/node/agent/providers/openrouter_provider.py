"""OpenRouter provider implementation using direct HTTP requests."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests

from entity.messages import (
    FunctionCallOutputEvent,
    Message,
    MessageRole,
    ToolCallPayload,
)
from entity.tool_spec import ToolSpec
from runtime.node.agent import ModelProvider
from runtime.node.agent import ModelResponse
from utils.token_tracker import TokenUsage


class OpenRouterProvider(ModelProvider):
    """OpenRouter provider using direct HTTP requests for Chat Completions API."""

    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    def create_client(self):
        """Create and return a client config dict (no SDK client needed)."""
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

        return {
            "api_key": api_key,
            "base_url": base_url,
            "headers": self._build_headers(),
        }

    def call_model(
        self,
        client: Dict[str, Any],
        conversation: List[Message],
        timeline: List[Any],
        tool_specs: Optional[List[ToolSpec]] = None,
        **kwargs,
    ) -> ModelResponse:
        """Call OpenRouter Chat Completions API with direct HTTP request."""
        api_key = client["api_key"]
        base_url = client["base_url"]
        headers = client["headers"]

        # Build Chat Completions payload (no tool_choice auto-injection)
        payload = self._build_chat_payload(conversation, tool_specs, kwargs)

        # Make HTTP request
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers["Authorization"] = f"Bearer {api_key}"
        headers["Content-Type"] = "application/json"

        response = requests.post(url, json=payload, headers=headers, timeout=300)
        response.raise_for_status()

        response_data = response.json()
        self._track_token_usage(response_data)

        # Parse Chat Completions response
        message = self._deserialize_chat_response(response_data)
        
        # Append to timeline for conversation continuity
        self._append_chat_response_output(timeline, response_data)

        return ModelResponse(message=message, raw_response=response_data)

    def extract_token_usage(self, response: Any) -> TokenUsage:
        """Extract token usage from OpenRouter response."""
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        
        input_tokens = usage.get("prompt_tokens", 0) or 0
        output_tokens = usage.get("completion_tokens", 0) or 0
        total_tokens = usage.get("total_tokens", 0) or 0
        if total_tokens == 0:
            total_tokens = input_tokens + output_tokens

        metadata = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }

        # Preserve reasoning tokens if present
        reasoning_tokens = self._read_reasoning_tokens(usage)
        if reasoning_tokens is not None:
            metadata["reasoning_tokens"] = reasoning_tokens

        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            metadata=metadata,
        )

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

    def _build_headers(self) -> Dict[str, str]:
        """Build OpenRouter-specific headers."""
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

    def _build_chat_payload(
        self,
        conversation: List[Message],
        tool_specs: Optional[List[ToolSpec]],
        raw_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build OpenRouter Chat Completions payload (no auto tool_choice)."""
        params = dict(raw_params)
        max_tokens = params.pop("max_tokens", None)
        max_output_tokens = params.pop("max_output_tokens", None)
        if max_tokens is None and max_output_tokens is not None:
            max_tokens = max_output_tokens

        # Serialize messages
        messages: List[Dict[str, Any]] = []
        for item in conversation:
            serialized = self._serialize_message_for_chat(item)
            if serialized is not None:
                messages.append(serialized)

        if not messages:
            messages = [{"role": "user", "content": ""}]

        # Build payload
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": params.pop("temperature", 0.7),
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        elif self.params.get("max_tokens"):
            payload["max_tokens"] = self.params["max_tokens"]

        # Build tools (do NOT auto-inject tool_choice)
        user_tools = params.pop("tools", None)
        merged_tools: List[Any] = []
        if isinstance(user_tools, list):
            merged_tools.extend(user_tools)

        if tool_specs:
            for spec in tool_specs:
                merged_tools.append({
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": spec.parameters or {"type": "object", "properties": {}},
                    }
                })

        if merged_tools:
            payload["tools"] = merged_tools

        # Only set tool_choice if explicitly provided (no automatic injection)
        tool_choice = params.pop("tool_choice", None)
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        # Pass any remaining params
        payload.update(params)
        return payload

    def _serialize_message_for_chat(self, message: Message) -> Optional[Dict[str, Any]]:
        """Convert internal Message to Chat Completions schema."""
        role_value = message.role.value
        content = message.text_content()

        payload: Dict[str, Any] = {
            "role": role_value,
            "content": content,
        }
        if message.name:
            payload["name"] = message.name
        if message.tool_call_id:
            payload["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            payload["tool_calls"] = [tc.to_openai_dict() for tc in message.tool_calls]
        return payload

    def _deserialize_chat_response(self, response: Dict[str, Any]) -> Message:
        """Convert Chat Completions response to internal Message."""
        choices = response.get("choices", [])
        if not choices:
            return Message(role=MessageRole.ASSISTANT, content="")

        choice = choices[0]
        msg = choice.get("message", {})

        tool_calls: List[ToolCallPayload] = []
        tc_data = msg.get("tool_calls", [])
        if tc_data:
            for idx, tc in enumerate(tc_data):
                f_data = tc.get("function", {})
                function_name = f_data.get("name", "")
                arguments = f_data.get("arguments", "")
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments)
                call_id = tc.get("id")
                if not call_id:
                    call_id = f"tool_call_{idx}"
                tool_calls.append(ToolCallPayload(
                    id=call_id,
                    function_name=function_name,
                    arguments=arguments,
                    type="function"
                ))

        return Message(
            role=MessageRole.ASSISTANT,
            content=msg.get("content") or "",
            tool_calls=tool_calls
        )

    def _append_chat_response_output(self, timeline: List[Any], response: Dict[str, Any]) -> None:
        """Add chat response to timeline."""
        choices = response.get("choices", [])
        if not choices:
            return

        choice = choices[0]
        msg = choice.get("message", {})
        assistant_msg = {
            "role": "assistant",
            "content": msg.get("content") or ""
        }

        if msg.get("tool_calls"):
            assistant_msg["tool_calls"] = []
            for idx, tc in enumerate(msg["tool_calls"]):
                f_data = tc.get("function", {})
                function_name = f_data.get("name", "")
                arguments = f_data.get("arguments", "")
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments)
                call_id = tc.get("id") or f"tool_call_{idx}"
                assistant_msg["tool_calls"].append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": arguments,
                    },
                })

        timeline.append(assistant_msg)

    def _read_reasoning_tokens(self, usage: Dict[str, Any]) -> Optional[int]:
        """Extract reasoning tokens from usage metadata."""
        candidates = [
            usage.get("reasoning_tokens"),
            usage.get("reasoningTokens"),
        ]

        completion_details = usage.get("completion_tokens_details", {})
        if isinstance(completion_details, dict):
            candidates.append(completion_details.get("reasoning_tokens"))
            candidates.append(completion_details.get("reasoningTokens"))

        for value in candidates:
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return None