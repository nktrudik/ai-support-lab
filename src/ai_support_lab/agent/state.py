from typing import Any, TypedDict

from pydantic import BaseModel, Field

from ai_support_lab.tickets.enums import Category, Priority


class AgentAnalysisRequest(BaseModel):
    include_context: bool = True


class LLMDraft(BaseModel):
    category: Category
    priority: Priority
    summary: str = Field(min_length=5, max_length=1000)
    recommended_action: str = Field(min_length=5, max_length=2000)
    requires_human: bool


class AgentAnalysisResponse(LLMDraft):
    ticket_id: int
    run_id: str
    classification_score: float = Field(ge=0, le=1)
    used_fallback: bool = False
    warnings: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)


class AnalysisState(TypedDict, total=False):
    ticket_id: int
    run_id: str
    include_context: bool
    ticket: dict[str, Any]
    classification: dict[str, Any]
    priority: str
    context: dict[str, Any]
    raw_response: str
    used_fallback: bool
    warnings: list[str]
    tools_used: list[str]
    result: dict[str, Any]
