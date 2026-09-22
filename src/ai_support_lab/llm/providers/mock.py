import json

from ai_support_lab.llm.schemas import LLMRequest, LLMResponse


class MockLLM:
    async def chat(self, request: LLMRequest) -> LLMResponse:
        # Mock не имитирует качество настоящей модели: он проверяет plumbing,
        # сериализацию, graph routing и запись результата без сети и GPU.
        context = json.loads(request.messages[-1].content)
        return LLMResponse(
            model="support-mock",
            content=json.dumps(
                {
                    "category": context["classification"]["category"],
                    "priority": context["priority"],
                    "summary": f"Support request: {context['ticket']['title']}",
                    "recommended_action": "Check service logs and verify the reported issue with the customer.",
                    "requires_human": context["priority"] in ("high", "critical"),
                }
            ),
        )
