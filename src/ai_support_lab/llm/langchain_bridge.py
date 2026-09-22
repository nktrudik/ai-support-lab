from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from ai_support_lab.llm.protocols import LLMProvider
from ai_support_lab.llm.schemas import LLMMessage, LLMRequest
from ai_support_lab.observability.protocols import ObservabilityProvider


class ProviderChatModel(BaseChatModel):
    _provider: LLMProvider = PrivateAttr()
    _observability: ObservabilityProvider = PrivateAttr()
    _backend: str = PrivateAttr()
    _model: str = PrivateAttr()

    def __init__(
        self, provider: LLMProvider, observability: ObservabilityProvider, backend: str, model: str
    ) -> None:
        super().__init__()
        self._provider, self._observability = provider, observability
        self._backend, self._model = backend, model

    @property
    def _llm_type(self) -> str:
        return "support-lab-provider"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Нельзя вызывать asyncio.run из уже работающего event loop. Контракт
        # адаптера намеренно async-only, а неподдерживаемый sync вызов явный.
        raise NotImplementedError("Use ainvoke() with this async-only provider")

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        roles = {"human": "user", "ai": "assistant", "system": "system"}
        request = LLMRequest(
            messages=[
                LLMMessage.model_validate({"role": roles[message.type], "content": message.content})
                for message in messages
            ]
        )
        with self._observability.span(
            "llm_generation",
            kind="llm",
            inputs=request.model_dump(),
            metadata={"backend": self._backend, "model": self._model},
        ) as span:
            response = await self._provider.chat(request)
            span.output = response.model_dump()
            span.usage = response.usage.model_dump() if response.usage else {}
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response.content))])
