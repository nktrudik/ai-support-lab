import asyncio
from dataclasses import dataclass
from uuid import uuid4

from ai_support_lab.agent.graph import build_graph
from ai_support_lab.agent.nodes import AnalysisNodes
from ai_support_lab.agent.state import AgentAnalysisRequest, AgentAnalysisResponse
from ai_support_lab.agent.tools import build_tools
from ai_support_lab.llm.langchain_bridge import ProviderChatModel
from ai_support_lab.observability.protocols import ObservabilityProvider
from ai_support_lab.tickets.exceptions import InferenceTimeout
from ai_support_lab.tickets.service import TicketService


@dataclass
class AgentService:
    tickets: TicketService
    chat_model: ProviderChatModel
    observability: ObservabilityProvider
    backend: str
    timeout_seconds: float

    async def analyze(self, ticket_id: int, request: AgentAnalysisRequest) -> AgentAnalysisResponse:
        run_id = str(uuid4())
        tools = build_tools(self.tickets, ticket_id, run_id, self.observability)
        nodes = AnalysisNodes(self.tickets, tools, self.chat_model, self.backend)
        graph = build_graph(nodes)
        with self.observability.span(
            "ticket_analysis",
            inputs={"ticket_id": ticket_id, **request.model_dump()},
            metadata={"run_id": run_id, "ticket_id": ticket_id, "backend": self.backend},
        ) as span:
            try:
                async with asyncio.timeout(self.timeout_seconds):
                    result = await graph.ainvoke(
                        {
                            "ticket_id": ticket_id,
                            "run_id": run_id,
                            "include_context": request.include_context,
                        },
                        config={"configurable": {"thread_id": run_id}},
                    )
            except TimeoutError as exc:
                raise InferenceTimeout("Agent run exceeded its total deadline") from exc
            span.output = result["result"]
            return AgentAnalysisResponse.model_validate(result["result"])
