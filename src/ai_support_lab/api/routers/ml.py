from fastapi import APIRouter

from ai_support_lab.api.dependencies import TicketServiceDep
from ai_support_lab.tickets.schemas import ClassificationResponse

router = APIRouter(prefix="/api/v1/tickets", tags=["classification"])


@router.post("/{ticket_id}/classify/sklearn", response_model=ClassificationResponse)
async def classify_sklearn(ticket_id: int, service: TicketServiceDep) -> ClassificationResponse:
    return await service.classify(ticket_id, "sklearn")


@router.post("/{ticket_id}/classify/torch", response_model=ClassificationResponse)
async def classify_torch(ticket_id: int, service: TicketServiceDep) -> ClassificationResponse:
    return await service.classify(ticket_id, "torch")
