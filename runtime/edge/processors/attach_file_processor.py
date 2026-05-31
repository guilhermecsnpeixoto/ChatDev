"""Edge payload processor that attaches file contents to the payload."""

from dataclasses import dataclass
from typing import Any

from entity.configs.edge.edge_processor import EdgeProcessorTypeConfig
from .base import EdgePayloadProcessor, ProcessorFactoryContext
from runtime.node.executor import ExecutionContext
from utils.log_manager import LogManager
from entity.messages import Message, MessageBlock, MessageBlockType
import os


@dataclass
class AttachFileEdgeProcessorConfig(EdgeProcessorTypeConfig):
    """Attach one or more files' contents to the outgoing payload.

    Fields:
    - path: single file path or list of file paths relative to workspace root
    - as_markdown: whether to wrap file content in a markdown code fence
    """

    path: str | list[str] = ""
    as_markdown: bool = True

    def display_label(self) -> str:
        return f"attach_file({self.path})"


class AttachFileEdgePayloadProcessor(EdgePayloadProcessor[AttachFileEdgeProcessorConfig]):
    """Read files and append their content as text blocks to the message."""

    def __init__(self, config: AttachFileEdgeProcessorConfig, ctx: ProcessorFactoryContext) -> None:
        super().__init__(config, ctx)
        self.label = f"attach_file:{config.path}"
        self.metadata = {"paths": config.path}

    def transform(
        self,
        payload: Message,
        *,
        source_result: Message,
        from_node,
        edge_link,
        log_manager: LogManager,
        context: ExecutionContext,
    ) -> Message | None:
        paths = self.config.path
        if isinstance(paths, str):
            paths = [paths]

        workspace_root = None
        try:
            workspace_root = getattr(context, "workspace_hook", None)
            # workspace_hook might be a Path-like or provide 'workspace_root' attr
            if workspace_root is not None and hasattr(workspace_root, "workspace_root"):
                workspace_root = workspace_root.workspace_root
        except Exception:
            workspace_root = None

        blocks = payload.blocks()

        for p in paths:
            rel = p
            # Normalize
            try:
                if workspace_root:
                    candidate = os.path.join(str(workspace_root), rel)
                else:
                    candidate = os.path.join(os.getcwd(), rel)
                if not os.path.exists(candidate):
                    log_manager.warning(f"AttachFile processor: file not found: {rel}")
                    continue
                with open(candidate, mode="r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as exc:  # pragma: no cover - defensive
                log_manager.error(f"AttachFile processor failed reading '{rel}': {exc}")
                continue

            if self.config.as_markdown:
                text = f"{rel}\n```python\n{content}\n```"
            else:
                text = f"FILE: {rel}\n\n{content}"

            blocks.append(MessageBlock.text_block(text))

        # return a new message with extended blocks
        new_msg = payload.clone()
        new_msg.content = [b.to_dict() for b in blocks]
        return new_msg
