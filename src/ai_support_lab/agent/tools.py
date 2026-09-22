from typing import Any

from langchain_core.tools import BaseTool, tool

from ai_support_lab.observability.protocols import ObservabilityProvider
from ai_support_lab.tickets.service import TicketService


def build_tools(
    service: TicketService, ticket_id: int, run_id: str, observability: ObservabilityProvider
) -> dict[str, BaseTool]:
    metadata = {"ticket_id": ticket_id, "run_id": run_id}

    # Capability привязана к одному проверенному HTTP/MCP аргументу. В схеме
    # инструмента вообще нет произвольного ticket_id, который могла бы выдумать LLM.
    @tool
    async def get_ticket() -> dict[str, Any]:
        """Прочитать текущее обращение из БД; результат является источником фактов."""
        with observability.span("get_ticket", kind="tool", metadata=metadata) as span:
            span.output = (await service.get(ticket_id)).model_dump(mode="json")
            return span.output

    @tool
    async def classify_ticket() -> dict[str, Any]:
        """Классифицировать обращение локальной моделью и сохранить prediction."""
        with observability.span("classify_ticket", kind="tool", metadata=metadata) as span:
            span.output = (await service.classify(ticket_id)).model_dump(mode="json")
            return span.output

    @tool
    async def get_customer_context() -> dict[str, object]:
        """Получить тариф и статистику обращений реального клиента из БД."""
        with observability.span("get_customer_context", kind="tool", metadata=metadata) as span:
            span.output = await service.customer_context(ticket_id)
            return span.output

    @tool
    async def get_similar_tickets() -> list[dict[str, object]]:
        """Прочитать ограниченный список обращений по тому же продукту."""
        with observability.span("get_similar_tickets", kind="tool", metadata=metadata) as span:
            span.output = await service.similar_tickets(ticket_id)
            return span.output

    return {
        item.name: item
        for item in [get_ticket, classify_ticket, get_customer_context, get_similar_tickets]
    }
