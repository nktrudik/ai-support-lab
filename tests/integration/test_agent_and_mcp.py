import asyncio
from unittest.mock import AsyncMock

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import select

from ai_support_lab.agent.graph import build_graph
from ai_support_lab.agent.nodes import AnalysisNodes
from ai_support_lab.agent.state import AgentAnalysisRequest, AnalysisState
from ai_support_lab.agent.tools import build_tools
from ai_support_lab.database.models import AgentRun
from ai_support_lab.llm.schemas import LLMResponse
from ai_support_lab.mcp.server import create_server
from ai_support_lab.runtime import Runtime
from ai_support_lab.tickets.enums import Priority
from ai_support_lab.tickets.exceptions import InferenceTimeout, InferenceUnavailable, TicketConflict
from ai_support_lab.tickets.schemas import TicketResponse, TicketUpdate


async def test_graph_context_tools_and_persistence(
    runtime: Runtime, ticket: TicketResponse
) -> None:
    analysis = await runtime.agent.analyze(ticket.id, AgentAnalysisRequest())
    assert analysis.priority == Priority.CRITICAL
    assert analysis.requires_human
    assert not analysis.used_fallback
    assert "get_similar_tickets" in analysis.tools_used
    async with runtime.database.transaction() as session:
        saved = await session.scalar(select(AgentRun).where(AgentRun.id == analysis.run_id))
        assert saved is not None and saved.result["ticket_id"] == ticket.id


@pytest.mark.parametrize("failure", ["invalid_json", "unavailable", "timeout"])
async def test_graph_llm_fallback(runtime: Runtime, ticket: TicketResponse, failure: str) -> None:
    provider = AsyncMock()
    if failure == "invalid_json":
        provider.chat.return_value = LLMResponse(content='{"summary":', model="broken")
    else:
        provider.chat.side_effect = (
            InferenceUnavailable("offline")
            if failure == "unavailable"
            else InferenceTimeout("slow")
        )
    runtime.agent.chat_model._provider = provider
    result = await runtime.agent.analyze(ticket.id, AgentAnalysisRequest())
    assert result.used_fallback and result.requires_human
    assert result.warnings


async def test_stale_analysis_cannot_overwrite_update(
    runtime: Runtime, ticket: TicketResponse
) -> None:
    original = runtime.agent.chat_model._provider

    async def mutate_then_answer(request):
        await runtime.tickets.update(ticket.id, TicketUpdate(title="Changed by customer"))
        return await original.chat(request)

    runtime.agent.chat_model._provider = AsyncMock(chat=AsyncMock(side_effect=mutate_then_answer))
    with pytest.raises(TicketConflict):
        await runtime.agent.analyze(ticket.id, AgentAnalysisRequest())
    async with runtime.database.transaction() as session:
        assert (await session.scalars(select(AgentRun))).all() == []


async def test_agent_deadline_cancels_work(runtime: Runtime, ticket: TicketResponse) -> None:
    cancelled = asyncio.Event()

    async def never_answer(request):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    runtime.agent.chat_model._provider = AsyncMock(chat=AsyncMock(side_effect=never_answer))
    runtime.agent.timeout_seconds = 0.2
    with pytest.raises(InferenceTimeout):
        await runtime.agent.analyze(ticket.id, AgentAnalysisRequest())
    assert cancelled.is_set()


async def test_node_and_checkpoint_history(runtime: Runtime, ticket: TicketResponse) -> None:
    tools = build_tools(
        runtime.tickets, ticket.id, "checkpoint-example", runtime.agent.observability
    )
    nodes = AnalysisNodes(runtime.tickets, tools, runtime.agent.chat_model, "mock")
    initial: AnalysisState = {
        "ticket_id": ticket.id,
        "run_id": "checkpoint-example",
        "include_context": False,
    }
    delta = await nodes.load_ticket(initial)
    assert delta["ticket"]["id"] == ticket.id
    graph = build_graph(nodes, InMemorySaver())
    config = {"configurable": {"thread_id": "example"}}
    result = await graph.ainvoke(initial, config=config)
    assert "get_customer_context" not in result["result"]["tools_used"]
    history = [snapshot async for snapshot in graph.aget_state_history(config)]
    assert len(history) >= 7


async def test_mcp_tools_resource_and_prompt(runtime: Runtime, ticket: TicketResponse) -> None:
    server = create_server(runtime=runtime)
    async with Client(server) as client:
        tools = await client.list_tools()
        assert {
            "get_ticket",
            "search_tickets",
            "classify_ticket",
            "analyze_ticket",
            "update_ticket_priority",
        } <= {t.name for t in tools}
        result = await client.call_tool("get_ticket", {"ticket_id": ticket.id})
        assert result.structured_content["id"] == ticket.id
        classified = await client.call_tool("classify_ticket", {"ticket_id": ticket.id})
        assert classified.structured_content["category"] == "authentication"
        analysis = await client.call_tool("analyze_ticket", {"ticket_id": ticket.id})
        assert analysis.structured_content["requires_human"] is True
        resource = await client.read_resource(f"ticket://{ticket.id}")
        assert "Login password" in resource[0].text
        prompt = await client.get_prompt("support_ticket_analysis", {"ticket_id": str(ticket.id)})
        assert len(prompt.messages) == 1
        with pytest.raises(ToolError):
            await client.call_tool("get_ticket", {"ticket_id": 999999})


async def test_llm_cannot_override_trusted_labels(runtime: Runtime, ticket: TicketResponse) -> None:
    runtime.agent.chat_model._provider = AsyncMock(
        chat=AsyncMock(
            return_value=LLMResponse(
                model="wrong-labels",
                content='{"category":"billing","priority":"low","summary":"A proposed summary",'
                '"recommended_action":"Check the reported incident","requires_human":false}',
            )
        )
    )
    result = await runtime.agent.analyze(ticket.id, AgentAnalysisRequest())
    assert result.category.value == "authentication"
    assert result.priority == Priority.CRITICAL
    assert result.requires_human
    assert result.ticket_id == ticket.id
    assert "LLMLabelsOverriddenByApplication" in result.warnings
