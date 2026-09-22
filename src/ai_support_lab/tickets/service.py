import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ai_support_lab.classical_ml.inference import TicketClassifier
from ai_support_lab.database.models import AgentRun, Ticket, TicketPrediction
from ai_support_lab.database.repository import TicketRepository
from ai_support_lab.database.session import Database
from ai_support_lab.tickets.enums import Priority, TicketStatus
from ai_support_lab.tickets.exceptions import ModelUnavailable, TicketConflict
from ai_support_lab.tickets.schemas import (
    ClassificationResponse,
    ResolveRequest,
    TicketCreate,
    TicketFilters,
    TicketListResponse,
    TicketResponse,
    TicketUpdate,
)

if TYPE_CHECKING:
    from ai_support_lab.agent.state import AgentAnalysisResponse

logger = logging.getLogger(__name__)


@dataclass
class TicketService:
    database: Database
    classifiers: dict[str, TicketClassifier]
    default_classifier: str
    inference_slots: asyncio.Semaphore

    async def create(self, request: TicketCreate) -> TicketResponse:
        async with self.database.transaction() as session:
            ticket = Ticket(
                title=request.title,
                description=request.description,
                customer=request.customer.model_dump(),
                customer_name=request.customer.name,
                product=request.product,
                metadata_json=request.metadata,
            )
            await TicketRepository(session).add(ticket)
            result = TicketResponse.model_validate(ticket)
        logger.info("ticket_created ticket_id=%s", result.id)
        return result

    async def get(self, ticket_id: int) -> TicketResponse:
        async with self.database.transaction() as session:
            return TicketResponse.model_validate(await TicketRepository(session).get(ticket_id))

    async def search(self, filters: TicketFilters) -> TicketListResponse:
        async with self.database.transaction() as session:
            tickets, total = await TicketRepository(session).list(filters)
            return TicketListResponse(
                items=[TicketResponse.model_validate(t) for t in tickets],
                total=total,
                limit=filters.limit,
                offset=filters.offset,
            )

    async def update(self, ticket_id: int, request: TicketUpdate) -> TicketResponse:
        async with self.database.transaction() as session:
            ticket = await TicketRepository(session).get(ticket_id, for_update=True)
            for key, value in request.model_dump(exclude_unset=True).items():
                setattr(ticket, "metadata_json" if key == "metadata" else key, value)
            await session.flush()
            return TicketResponse.model_validate(ticket)

    async def classify(self, ticket_id: int, backend: str | None = None) -> ClassificationResponse:
        ticket = await self.get(ticket_id)
        name = backend or self.default_classifier
        classifier = self.classifiers.get(name)
        if classifier is None:
            raise ModelUnavailable(
                f"{name} artifact unavailable; run the corresponding training command"
            )
        # Inference синхронный и вычислительный. Thread не ускоряет произвольный
        # Python CPU-код, но освобождает event loop; NumPy/Torch выполняют ядра в native code.
        # Ограничиваем конкуренцию, чтобы запросы не создавали лавину вычислений.
        async with self.inference_slots:
            job = asyncio.create_task(
                asyncio.to_thread(classifier.predict, f"{ticket.title} {ticket.description}")
            )
            try:
                result = await asyncio.shield(job)
            except asyncio.CancelledError:
                # Python не останавливает выполняющийся native thread при отмене.
                # Держим semaphore до завершения расчёта, затем распространяем cancel.
                await job
                raise
        async with self.database.transaction() as session:
            current = await TicketRepository(session).get(ticket_id, for_update=True)
            if current.version != ticket.version:
                raise TicketConflict("Ticket changed while classification was running; retry")
            session.add(TicketPrediction(ticket_id=ticket_id, **result.model_dump()))
        return result

    async def customer_context(self, ticket_id: int) -> dict[str, object]:
        ticket = await self.get(ticket_id)
        history = await self.search(TicketFilters(customer=ticket.customer.name, limit=10))
        return {
            "customer": ticket.customer.model_dump(),
            "ticket_count": history.total,
            "support_policy": "human escalation for critical incidents",
        }

    async def similar_tickets(self, ticket_id: int) -> list[dict[str, object]]:
        ticket = await self.get(ticket_id)
        result = await self.search(TicketFilters(product=ticket.product, limit=6))
        # Это прозрачный baseline похожести по продукту, а не «семантический поиск».
        # Возвращаем только реальные id и ограниченный объём контекста.
        related: list[dict[str, object]] = [
            {"id": item.id, "title": item.title, "status": item.status.value}
            for item in result.items
            if item.id != ticket_id
        ]
        return related[:5]

    async def persist_analysis(
        self, ticket: TicketResponse, analysis: "AgentAnalysisResponse", backend: str
    ) -> None:
        async with self.database.transaction() as session:
            current = await TicketRepository(session).get(ticket.id, for_update=True)
            if current.version != ticket.version:
                raise TicketConflict("Ticket changed during analysis; run analysis again")
            # Сохраняем результат и вычисленный приложением приоритет атомарно.
            # Модель не получает метода «записать произвольную строку в БД».
            current.priority = analysis.priority
            session.add(
                AgentRun(
                    id=analysis.run_id,
                    ticket_id=ticket.id,
                    ticket_version=ticket.version,
                    backend=backend,
                    result=analysis.model_dump(mode="json"),
                )
            )

    async def resolve(self, ticket_id: int, request: ResolveRequest) -> TicketResponse:
        async with self.database.transaction() as session:
            ticket = await TicketRepository(session).get(ticket_id, for_update=True)
            if ticket.version != request.expected_version or ticket.status == TicketStatus.RESOLVED:
                raise TicketConflict("Ticket version changed or ticket is already resolved")
            ticket.resolution = request.resolution
            ticket.status = TicketStatus.RESOLVED
            await session.flush()
            return TicketResponse.model_validate(ticket)


def assess_priority(ticket: TicketResponse) -> Priority:
    text = f"{ticket.title} {ticket.description}".casefold()
    if any(term in text for term in ("outage", "all users blocked", "data loss")):
        return Priority.CRITICAL
    if ticket.priority in (Priority.HIGH, Priority.CRITICAL):
        return ticket.priority
    if ticket.customer.tier == "enterprise":
        return Priority.HIGH
    return ticket.priority
