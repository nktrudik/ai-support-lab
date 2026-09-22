from typing import Annotated

from fastapi import APIRouter, Query, status

from ai_support_lab.api.dependencies import TicketServiceDep
from ai_support_lab.tickets.schemas import (
    ResolveRequest,
    TicketCreate,
    TicketFilters,
    TicketListResponse,
    TicketResponse,
    TicketUpdate,
)

router = APIRouter(prefix="/api/v1/tickets", tags=["tickets"])


@router.post("", response_model=TicketResponse, status_code=status.HTTP_201_CREATED)
async def create_ticket(body: TicketCreate, service: TicketServiceDep) -> TicketResponse:
    return await service.create(body)


@router.get("", response_model=TicketListResponse)
async def list_tickets(
    filters: Annotated[TicketFilters, Query()], service: TicketServiceDep
) -> TicketListResponse:
    return await service.search(filters)


@router.get("/{ticket_id}", response_model=TicketResponse)
async def get_ticket(ticket_id: int, service: TicketServiceDep) -> TicketResponse:
    return await service.get(ticket_id)


@router.patch("/{ticket_id}", response_model=TicketResponse)
async def update_ticket(
    ticket_id: int, body: TicketUpdate, service: TicketServiceDep
) -> TicketResponse:
    return await service.update(ticket_id, body)


@router.post("/{ticket_id}/resolve", response_model=TicketResponse)
async def resolve_ticket(
    ticket_id: int, body: ResolveRequest, service: TicketServiceDep
) -> TicketResponse:
    return await service.resolve(ticket_id, body)
