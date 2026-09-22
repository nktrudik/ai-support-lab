from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ai_support_lab.database.models import Ticket
from ai_support_lab.tickets.exceptions import TicketNotFound
from ai_support_lab.tickets.schemas import TicketFilters


class TicketRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, ticket_id: int, *, for_update: bool = False) -> Ticket:
        statement = (
            select(Ticket).where(Ticket.id == ticket_id).options(selectinload(Ticket.predictions))
        )
        if for_update:
            statement = statement.with_for_update()
        ticket = await self.session.scalar(statement)
        if ticket is None:
            raise TicketNotFound(f"Ticket {ticket_id} does not exist")
        return ticket

    async def add(self, ticket: Ticket) -> Ticket:
        self.session.add(ticket)
        await self.session.flush()
        # flush отправляет INSERT, но не фиксирует транзакцию. refresh показывает,
        # как получить сгенерированный БД объект до формирования DTO.
        await self.session.refresh(
            ticket, attribute_names=["created_at", "updated_at", "predictions"]
        )
        return ticket

    async def list(self, filters: TicketFilters) -> tuple[list[Ticket], int]:
        statement = select(Ticket)
        if filters.product is not None:
            statement = statement.where(Ticket.product == filters.product.casefold())
        if filters.customer is not None:
            statement = statement.where(Ticket.customer_name == filters.customer)
        if filters.priority is not None:
            statement = statement.where(Ticket.priority == filters.priority)
        if filters.status is not None:
            statement = statement.where(Ticket.status == filters.status)
        total = await self.session.scalar(select(func.count()).select_from(statement.subquery()))
        result = await self.session.scalars(
            statement.options(selectinload(Ticket.predictions))
            .order_by(Ticket.created_at.desc(), Ticket.id.desc())
            .limit(filters.limit)
            .offset(filters.offset)
        )
        # Детерминированный tie-break по id нужен, когда created_at совпадает.
        # selectinload предотвращает N+1 и не вызывает неявный async I/O из DTO.
        return list(result.all()), int(total or 0)
