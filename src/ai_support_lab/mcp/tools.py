from typing import cast

from fastmcp import Context, FastMCP

from ai_support_lab.agent.state import AgentAnalysisRequest, AgentAnalysisResponse
from ai_support_lab.runtime import Runtime
from ai_support_lab.tickets.enums import Priority
from ai_support_lab.tickets.schemas import (
    ClassificationResponse,
    TicketFilters,
    TicketListResponse,
    TicketResponse,
    TicketUpdate,
)


def get_runtime(context: Context) -> Runtime:
    return cast(Runtime, context.lifespan_context["runtime"])


def register_capabilities(server: FastMCP) -> None:
    @server.tool
    async def get_ticket(ticket_id: int, ctx: Context) -> TicketResponse:
        """Прочитать обращение по существующему идентификатору, без изменения данных."""
        return await get_runtime(ctx).tickets.get(ticket_id)

    @server.tool
    async def search_tickets(
        ctx: Context, product: str | None = None, limit: int = 10
    ) -> TicketListResponse:
        """Найти обращения по продукту; limit ограничен схемой приложения до 100."""
        return await get_runtime(ctx).tickets.search(TicketFilters(product=product, limit=limit))

    @server.tool
    async def classify_ticket(ticket_id: int, ctx: Context) -> ClassificationResponse:
        """Вычислить категорию локальной моделью и записать результат классификации."""
        return await get_runtime(ctx).tickets.classify(ticket_id)

    @server.tool
    async def analyze_ticket(ticket_id: int, ctx: Context) -> AgentAnalysisResponse:
        """Запустить LangGraph, сохранить анализ и вычисленный приоритет обращения."""
        return await get_runtime(ctx).agent.analyze(ticket_id, AgentAnalysisRequest())

    @server.tool
    async def update_ticket_priority(
        ticket_id: int, priority: Priority, ctx: Context
    ) -> TicketResponse:
        """Явно изменить приоритет обращения через тот же service, что использует REST API."""
        return await get_runtime(ctx).tickets.update(ticket_id, TicketUpdate(priority=priority))

    @server.resource("ticket://{ticket_id}")
    async def ticket_resource(ticket_id: int, ctx: Context) -> str:
        """Предоставить JSON-представление обращения как адресуемый MCP resource."""
        return (await get_runtime(ctx).tickets.get(ticket_id)).model_dump_json()

    @server.prompt
    def support_ticket_analysis(ticket_id: int) -> str:
        """Сформировать инструкцию для клиента; prompt сам не вызывает LLM и tools."""
        return (
            f"Read ticket://{ticket_id}, then call analyze_ticket for ticket {ticket_id}. "
            "Use database facts and report whether a human must review the recommendation."
        )
