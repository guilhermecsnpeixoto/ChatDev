"""Register built-in edge payload processors."""

from entity.configs.edge.edge_processor import (
    RegexEdgeProcessorConfig,
    FunctionEdgeProcessorConfig,
    ResetLoopCounterEdgeProcessorConfig,
)
from .registry import register_edge_processor
from .regex_processor import RegexEdgePayloadProcessor
from .function_processor import FunctionEdgePayloadProcessor
from .reset_loop_counter_processor import ResetLoopCounterEdgePayloadProcessor
from .attach_file_processor import AttachFileEdgePayloadProcessor
from entity.configs.edge.edge_processor import AttachFileEdgeProcessorConfig

register_edge_processor(
    "regex_extract",
    processor_cls=RegexEdgePayloadProcessor,
    summary="Extract payload fragments via Python regular expressions.",
    config_cls=RegexEdgeProcessorConfig,
)

register_edge_processor(
    "function",
    processor_cls=FunctionEdgePayloadProcessor,
    summary="Delegate message transformation to Python functions in functions/edge_processor.",
    config_cls=FunctionEdgeProcessorConfig,
)

register_edge_processor(
    "reset_loop_counter",
    processor_cls=ResetLoopCounterEdgePayloadProcessor,
    summary="Reset a loop_counter node runtime state before forwarding payload.",
    config_cls=ResetLoopCounterEdgeProcessorConfig,
)

register_edge_processor(
    "attach_file",
    processor_cls=AttachFileEdgePayloadProcessor,
    summary="Attach local workspace file contents to the message payload.",
    config_cls=AttachFileEdgeProcessorConfig,
)
