from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core import QueryBundle
from llama_index.core.base.response.schema import RESPONSE_TYPE
from typing import List, Any, Optional, Sequence
from llama_index.core.schema import NodeWithScore, MetadataMode,QueryType
from llama_index.core.callbacks import CBEventType, EventPayload

##Custom bộ tổng hợp context từ db để tránh việc nếu không có context từ db hệ thống mặc định 
## sẽ ra output Empty Response
class CustomCompactAndRefine(CompactAndRefine):
    """Custom response synthesizer that calls LLM even when no nodes are retrieved."""

    QueryTextType = QueryType

    def synthesize(
        self,
        query: QueryTextType,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
        **response_kwargs: Any,
    ) -> RESPONSE_TYPE:
        if isinstance(query, str):
            query = QueryBundle(query_str=query)

        with self._callback_manager.event(
            CBEventType.SYNTHESIZE,
            payload={EventPayload.QUERY_STR: query.query_str},
        ) as event:
            response_str = self.get_response(
                query_str=query.query_str,
                text_chunks=[
                    n.node.get_content(metadata_mode=MetadataMode.LLM) for n in nodes
                ],
                **response_kwargs,
            )

            additional_source_nodes = additional_source_nodes or []
            source_nodes = list(nodes) + list(additional_source_nodes)

            response = self._prepare_response_output(response_str, source_nodes)

            event.on_end(payload={EventPayload.RESPONSE: response})

        return response