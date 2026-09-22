import httpx
from pydantic import BaseModel, Field, ValidationError

from ai_support_lab.llm.schemas import LLMRequest, LLMResponse, TokenUsage
from ai_support_lab.tickets.exceptions import (
    InferenceTimeout,
    InferenceUnavailable,
    MalformedLLMOutput,
)


class ChatMessage(BaseModel):
    content: str


class ChatChoice(BaseModel):
    message: ChatMessage


class CompletionEnvelope(BaseModel):
    choices: list[ChatChoice] = Field(min_length=1)
    model: str
    usage: dict[str, int] | None = None


class OpenAICompatibleLLM:
    def __init__(
        self, client: httpx.AsyncClient, base_url: str, model: str, api_key: str = ""
    ) -> None:
        self.client, self.base_url, self.model = client, base_url.rstrip("/"), model
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    async def healthcheck(self) -> bool:
        try:
            response = await self.client.get(f"{self.base_url}/models", headers=self.headers)
            response.raise_for_status()
            return True
        except httpx.HTTPError:
            return False

    async def chat(self, request: LLMRequest) -> LLMResponse:
        try:
            # Клиент и пул соединений создаются один раз в lifespan. /v1 входит
            # в base_url; отсутствие конкретного SDK сохраняет одинаковый контракт
            # для vLLM, llama-server и trtllm-serve.
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json={"model": self.model, **request.model_dump()},
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise InferenceTimeout("Inference backend exceeded its timeout") from exc
        except httpx.HTTPError as exc:
            raise InferenceUnavailable("Inference backend is unavailable") from exc
        try:
            envelope = CompletionEnvelope.model_validate(response.json())
        except (ValidationError, ValueError) as exc:
            raise MalformedLLMOutput("Invalid chat completion envelope") from exc
        usage = envelope.usage
        return LLMResponse(
            content=envelope.choices[0].message.content,
            model=envelope.model,
            usage=TokenUsage(
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            )
            if usage
            else None,
        )
