"""IAedu provider implementation."""

from __future__ import annotations

import io
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from entity.messages import Message, MessageRole
from entity.tool_spec import ToolSpec
from runtime.node.agent import ModelProvider, ModelResponse
from utils.token_tracker import TokenUsage


class IAeduProvider(ModelProvider):
    """IAedu provider implementation using multipart/form-data requests."""

    def create_client(self):
        """Create and return a reusable HTTP session."""
        session = requests.Session()
        session.headers.update({"Accept": "application/json, text/plain, text/event-stream"})
        return session

    def call_model(
        self,
        client,
        conversation: List[Message],
        timeline: List[Any],
        tool_specs: Optional[List[ToolSpec]] = None,
        **kwargs,
    ) -> ModelResponse:
        """Send the latest user message as a multipart form post."""
        request_payload = self._build_request_payload(conversation, kwargs)
        response = client.post(
            self.base_url,
            headers=self._build_headers(),
            files=request_payload["files"],
            timeout=request_payload["timeout"],
            stream=True,
        )
        response.raise_for_status()

        message_text = self._extract_response_text(response)
        message = Message(role=MessageRole.ASSISTANT, content=message_text)
        return ModelResponse(message=message, raw_response=response)

    def extract_token_usage(self, response: Any) -> TokenUsage:
        """IAedu does not expose token usage in the current request format."""
        return TokenUsage()

    def _build_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    def _build_request_payload(
        self,
        conversation: List[Message],
        raw_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        params = dict(self.params or {})
        params.update(raw_params)

        channel_id = self._resolve_required_value(params, "channel_id", "IAEDU_CHANNEL_ID")
        thread_id = self._resolve_required_value(params, "thread_id", "IAEDU_THREAD_ID")

        user_info = self._stringify_field(params.get("user_info", "{}"))
        message_text = self._resolve_message_text(conversation, params)

        fields: Dict[str, Any] = {
            "channel_id": channel_id,
            "thread_id": thread_id,
            "user_info": user_info,
            "message": message_text,
        }

        if params.get("user_id") is not None:
            fields["user_id"] = self._stringify_field(params.get("user_id"))
        if params.get("user_context") is not None:
            fields["user_context"] = self._stringify_field(params.get("user_context"))

        image_payload = self._build_image_payload(params)
        files = {name: (None, value) for name, value in fields.items()}
        if image_payload is not None:
            files["image"] = image_payload

        return {
            "files": files,
            "timeout": params.get("timeout", 300),
        }

    def _resolve_required_value(
        self,
        params: Dict[str, Any],
        key: str,
        env_name: str,
    ) -> str:
        value = params.get(key) or os.getenv(env_name)
        if value is None or str(value).strip() == "":
            raise ValueError(f"IAedu provider requires '{key}' (or env var {env_name})")
        return str(value)

    def _resolve_message_text(self, conversation: List[Message], params: Dict[str, Any]) -> str:
        explicit_message = params.get("message")
        if explicit_message is not None:
            return self._stringify_field(explicit_message)

        for item in reversed(conversation):
            if isinstance(item, Message) and item.role is MessageRole.USER:
                text = item.text_content().strip()
                if text:
                    return text

        for item in reversed(conversation):
            if isinstance(item, Message):
                text = item.text_content().strip()
                if text:
                    return text

        return ""

    def _stringify_field(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)

    def _build_image_payload(self, params: Dict[str, Any]) -> Optional[tuple[str, Any, str]]:
        image_value = params.get("image") or params.get("image_path")
        if image_value is None:
            return None

        if isinstance(image_value, tuple) and len(image_value) in {2, 3}:
            filename = str(image_value[0])
            file_obj = image_value[1]
            mime_type = image_value[2] if len(image_value) == 3 else mimetypes.guess_type(filename)[0] or "application/octet-stream"
            return (filename, file_obj, mime_type)

        if isinstance(image_value, (str, Path)):
            image_path = Path(image_value).expanduser()
            if not image_path.exists():
                raise ValueError(f"IAedu image path does not exist: {image_path}")
            mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
            return (image_path.name, io.BytesIO(image_path.read_bytes()), mime_type)

        raise ValueError("IAedu image must be a file path or a (filename, file, mime_type) tuple")

    def _extract_response_text(self, response: Any) -> str:
        content_type = str(getattr(response, "headers", {}).get("Content-Type", "")).lower()
        if "text/event-stream" in content_type:
            stream_chunks: List[str] = []
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                chunk = line.strip()
                if chunk.startswith("data:"):
                    chunk = chunk[5:].strip()
                if not chunk or chunk in {"[DONE]", "done"}:
                    continue
                extracted = self._extract_text_from_payload(chunk)
                if extracted:
                    stream_chunks.append(extracted)
            combined = "\n".join(stream_chunks).strip()
            if combined:
                return combined

        try:
            payload = response.json()
        except Exception:
            payload = None

        if payload is not None:
            extracted = self._extract_text_from_payload(payload)
            if extracted:
                return extracted

        return str(getattr(response, "text", "")).strip()

    def _extract_text_from_payload(self, payload: Any) -> str:
        if payload is None:
            return ""
        if isinstance(payload, str):
            stripped = payload.strip()
            if not stripped:
                return ""
            try:
                decoded = json.loads(stripped)
            except Exception:
                return stripped
            return self._extract_text_from_payload(decoded)
        if isinstance(payload, list):
            pieces = [self._extract_text_from_payload(item) for item in payload]
            return "\n".join(piece for piece in pieces if piece).strip()
        if isinstance(payload, dict):
            for key in ("message", "content", "text", "answer", "output", "response"):
                value = payload.get(key)
                if value:
                    extracted = self._extract_text_from_payload(value)
                    if extracted:
                        return extracted
            for key in ("data", "result", "delta", "choices"):
                value = payload.get(key)
                if value:
                    extracted = self._extract_text_from_payload(value)
                    if extracted:
                        return extracted
        return ""