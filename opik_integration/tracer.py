from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional
import opik 
from dotenv import load_dotenv


DEFAULT_OPIK_HOST = "https://www.comet.com/opik/api"
DEFAULT_PROJECT_NAME = "ChatDev"


@dataclass
class OpikTraceContext:
    trace_id: str
    thread_id: str
    run_id: str
    name: str


class OpikTracer:
    """Best-effort Opik tracer with safe no-op fallback."""

    def __init__(
        self,
        *,
        project_name: str,
        api_key: Optional[str],
        host: str = DEFAULT_OPIK_HOST,
        workspace: Optional[str] = None,
        thread_id: Optional[str] = None,
        run_id: Optional[str] = None,
        enabled: bool = True,
    ) -> None:
        self.project_name = project_name
        self.api_key = api_key
        self.host = host.rstrip("/")
        self.workspace = workspace
        self.thread_id = thread_id or ""
        self.run_id = run_id or ""
        self.enabled = bool(enabled and api_key and project_name and opik is not None)
        self._lock = threading.Lock()
        self._active_traces: Dict[str, Any] = {}
        self._active_spans: Dict[str, Any] = {}

        if self.enabled and opik is not None:
            try:
                opik.configure(project_name=self.project_name)
            except Exception:
                self.enabled = False

    def start_trace(
        self,
        *,
        name: str,
        thread_id: str,
        input_payload: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[OpikTraceContext]:
        if not self.enabled:
            return None

        trace_metadata = metadata or {}
        trace_cm = opik.start_as_current_trace(
            name,
            project_name=self.project_name,
            thread_id=thread_id,
        )
        trace_obj = trace_cm.__enter__()
        trace_obj.input = _safe_jsonable(input_payload)
        trace_obj.metadata = trace_metadata

        trace_id = getattr(trace_obj, "id", None) or _random_id()
        trace_id = str(trace_id)
        with self._lock:
            self._active_traces[trace_id] = (trace_cm, trace_obj)

        return OpikTraceContext(
            trace_id=trace_id,
            thread_id=thread_id,
            run_id=self.run_id or thread_id,
            name=name,
        )

    def end_trace(
        self,
        trace_id: str,
        *,
        output: Any = None,
        error_info: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return
        with self._lock:
            trace_entry = self._active_traces.pop(trace_id, None)

        if not trace_entry:
            return

        trace_cm, trace_obj = trace_entry
        trace_obj.output = _safe_jsonable(output)
        
        try:
            merged_metadata = dict(getattr(trace_obj, "metadata", {}) or {})
            
            # Ensure metadata is a dict before merging
            if metadata and isinstance(metadata, dict):
                merged_metadata.update(metadata)
            
            if error_info:
                merged_metadata["error_info"] = error_info
            trace_obj.metadata = merged_metadata
        except Exception as e:
            # Fallback: just set metadata to what we have
            trace_obj.metadata = {"error": f"metadata merge failed: {str(e)}"}
        
        try:
            trace_cm.__exit__(None, None, None)
        except Exception:
            pass

    def start_span(
        self,
        trace_id: str,
        *,
        name: str,
        span_type: str,
        input_payload: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not self.enabled:
            return _random_id()
        span_cm = opik.start_as_current_span(
            name,
            type=span_type,
            project_name=self.project_name,
        )
        span_obj = span_cm.__enter__()
        span_obj.input = _safe_jsonable(input_payload)
        span_obj.metadata = metadata or {}

        span_id = getattr(span_obj, "id", None) or _random_id()
        span_id = str(span_id)
        with self._lock:
            self._active_spans[span_id] = (span_cm, span_obj)
        return span_id

    def end_span(
        self,
        span_id: str,
        *,
        output: Any = None,
        error_info: Optional[str] = None,
        usage: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,        # ← new
        provider: Optional[str] = None,     # ← new
    ) -> None:
        if not self.enabled:
            return

        with self._lock:
            span_entry = self._active_spans.pop(span_id, None)
        if not span_entry:
            return

        span_cm, span_obj = span_entry
        span_obj.output = _safe_jsonable(output)

        # These three fields are what Opik uses for cost & aggregation
        if usage and isinstance(usage, dict):
            span_obj.usage = usage
        if model:
            span_obj.model = model
        if provider:
            span_obj.provider = provider

        # Metadata remains for extra info, but we no longer store usage/model there
        try:
            merged_metadata = dict(getattr(span_obj, "metadata", {}) or {})
            
            if metadata and isinstance(metadata, dict):
                merged_metadata.update(metadata)
            if error_info:
                merged_metadata["error_info"] = error_info
            span_obj.metadata = merged_metadata


            # If a cost was computed and stored in metadata, move it to the span's total_cost
            cost = merged_metadata.get("cost") or merged_metadata.get("total_cost") 
            if cost is not None:
                try:
                    span_obj.total_cost = float(cost)
                except (TypeError, ValueError):
                    pass  # non‑numeric value – ignore
        except Exception:
            span_obj.metadata = {"error": "metadata merge failed"}

        try:
            span_cm.__exit__(None, None, None)
        except Exception:
            pass


def build_opik_tracer(*, session_id: Optional[str], workflow_id: Optional[str]) -> OpikTracer:
    if load_dotenv is not None:
        try:
            load_dotenv()
        except Exception:
            pass
    api_key = os.getenv("OPIK_API_KEY")
    project_name = os.getenv("OPIK_PROJECT_NAME", DEFAULT_PROJECT_NAME)
    host = os.getenv("OPIK_URL_OVERRIDE", DEFAULT_OPIK_HOST)
    workspace = os.getenv("OPIK_WORKSPACE") or os.getenv("OPIK_WORKSPACE_NAME")

    thread_id = session_id or workflow_id or _random_id()
    run_id = session_id or thread_id

    return OpikTracer(
        project_name=project_name,
        api_key=api_key,
        host=host,
        workspace=workspace,
        thread_id=thread_id,
        run_id=run_id,
        enabled=bool(api_key and project_name),
    )


def _safe_jsonable(value: Any, *, max_chars: int = 20000) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return _clip_text(str(value), max_chars)
    if isinstance(value, (list, dict)):
        return _clip_mapping(value, max_chars)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _clip_mapping(model_dump(), max_chars)
        except Exception:
            pass
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _clip_mapping(to_dict(), max_chars)
        except Exception:
            pass
    try:
        return _clip_text(json.dumps(value, default=str), max_chars)
    except Exception:
        return _clip_text(str(value), max_chars)


def _clip_mapping(value: Any, max_chars: int) -> Any:
    try:
        raw = json.dumps(value, ensure_ascii=True, default=str)
    except Exception:
        raw = str(value)
    if len(raw) <= max_chars:
        return value
    return _clip_text(raw, max_chars)


def _clip_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20] + "...[truncated]"


def _random_id() -> str:
    return uuid.uuid4().hex
