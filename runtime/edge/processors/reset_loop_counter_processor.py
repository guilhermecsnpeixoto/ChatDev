"""Edge payload processor that resets a loop counter runtime state."""

from entity.configs.edge.edge_processor import ResetLoopCounterEdgeProcessorConfig
from .base import EdgePayloadProcessor


class ResetLoopCounterEdgePayloadProcessor(EdgePayloadProcessor[ResetLoopCounterEdgeProcessorConfig]):
    """Reset a loop_counter node state when this edge is processed."""

    STATE_KEY = "loop_counter"

    def __init__(self, config: ResetLoopCounterEdgeProcessorConfig, ctx):
        super().__init__(config, ctx)
        self.label = f"reset_loop_counter:{config.counter_id}"
        self.metadata = {"counter_id": config.counter_id}

    def transform(
        self,
        payload,
        *,
        source_result,
        from_node,
        edge_link,
        log_manager,
        context,
    ):
        state = context.global_state.setdefault(self.STATE_KEY, {})
        counter = state.setdefault(self.config.counter_id, {"count": 0})
        previous = counter.get("count", 0)
        counter["count"] = 0

        log_manager.debug(
            f"Edge processor reset loop counter '{self.config.counter_id}' (previous={previous}, new=0)"
        )
        return payload
