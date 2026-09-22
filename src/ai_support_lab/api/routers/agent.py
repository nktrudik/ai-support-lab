from fastapi import APIRouter

from ai_support_lab.agent.state import AgentAnalysisRequest, AgentAnalysisResponse
from ai_support_lab.api.dependencies import AgentServiceDep

router = APIRouter(prefix="/api/v1/tickets", tags=["agent"])


@router.post("/{ticket_id}/analyze", response_model=AgentAnalysisResponse)
async def analyze_ticket(
    ticket_id: int, body: AgentAnalysisRequest, service: AgentServiceDep
) -> AgentAnalysisResponse:
    return await service.analyze(ticket_id, body)
