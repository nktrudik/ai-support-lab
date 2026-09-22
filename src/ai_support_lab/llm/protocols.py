from typing import Protocol

from ai_support_lab.llm.schemas import LLMRequest, LLMResponse


class LLMProvider(Protocol):
    async def chat(self, request: LLMRequest) -> LLMResponse: ...
