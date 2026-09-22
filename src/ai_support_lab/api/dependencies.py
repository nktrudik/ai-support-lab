from collections.abc import AsyncIterator
from typing import Annotated, cast

from fastapi import Depends, Request
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from ai_support_lab.agent.service import AgentService
from ai_support_lab.runtime import Runtime
from ai_support_lab.tickets.exceptions import DatabaseUnavailable
from ai_support_lab.tickets.service import TicketService


def get_runtime(request: Request) -> Runtime:
    return cast(Runtime, request.app.state.runtime)


def get_ticket_service(runtime: Annotated[Runtime, Depends(get_runtime)]) -> TicketService:
    return runtime.tickets


def get_agent_service(runtime: Annotated[Runtime, Depends(get_runtime)]) -> AgentService:
    return runtime.agent


async def get_read_session(
    runtime: Annotated[Runtime, Depends(get_runtime)],
) -> AsyncIterator[AsyncSession]:
    # Readiness делает короткий read-only запрос. yield отдаёт session endpoint,
    # а выход из async with гарантирует close, включая HTTP-ошибку или отмену.
    # Транзакции бизнес-операций находятся в service, не в этой dependency.
    async with runtime.database.sessions() as session:
        try:
            yield session
        except (SQLAlchemyError, OSError) as exc:
            raise DatabaseUnavailable("Database readiness check failed") from exc


TicketServiceDep = Annotated[TicketService, Depends(get_ticket_service)]
AgentServiceDep = Annotated[AgentService, Depends(get_agent_service)]
ReadSessionDep = Annotated[AsyncSession, Depends(get_read_session)]
