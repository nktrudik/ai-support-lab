from unittest.mock import MagicMock

import httpx
import pytest

from ai_support_lab.config import Settings
from ai_support_lab.llm.providers.openai_compatible import OpenAICompatibleLLM
from ai_support_lab.llm.schemas import LLMMessage, LLMRequest
from ai_support_lab.observability.factory import create_observability
from ai_support_lab.observability.noop import NoopObservability
from ai_support_lab.tickets.exceptions import (
    InferenceTimeout,
    InferenceUnavailable,
    MalformedLLMOutput,
)


async def test_http_adapter_request_response_and_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer local-test-key"
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": []})
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            json={
                "model": "local",
                "choices": [{"message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleLLM(client, "http://localhost/v1", "local", "local-test-key")
        assert await provider.healthcheck()
        response = await provider.chat(
            LLMRequest(messages=[LLMMessage(role="user", content="hello")])
        )
        assert response.usage is not None and response.usage.total_tokens == 15


@pytest.mark.parametrize(
    "case,error",
    [("http", InferenceUnavailable), ("timeout", InferenceTimeout), ("json", MalformedLLMOutput)],
)
async def test_http_adapter_errors(case: str, error: type[Exception]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if case == "timeout":
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(503 if case == "http" else 200, json={"choices": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleLLM(client, "http://localhost/v1", "local")
        with pytest.raises(error):
            await provider.chat(LLMRequest(messages=[LLMMessage(role="user", content="hi")]))


def test_missing_credentials_fall_back_without_optional_imports() -> None:
    settings = Settings(_env_file=None, observability_provider="langfuse")
    assert isinstance(create_observability(settings), NoopObservability)


def test_langfuse_real_sdk_with_in_memory_exporter() -> None:
    Langfuse = pytest.importorskip("langfuse").Langfuse
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.sdk.trace.sampling import ALWAYS_ON

    from ai_support_lab.observability.langfuse import LangfuseObservability

    exporter = InMemorySpanExporter()
    tracer = TracerProvider(sampler=ALWAYS_ON)
    adapter = LangfuseObservability.__new__(LangfuseObservability)
    # Настоящий SDK, но exporter локальный: тест проверяет API и вложенность,
    # не выдавая отсутствие cloud credentials за успешную доставку в Langfuse.
    adapter.client = Langfuse(
        public_key="pk-test-local",
        secret_key="sk-test-local",
        tracer_provider=tracer,
        span_exporter=exporter,
    )
    with adapter.span("agent", metadata={"run_id": "one"}):
        with adapter.span("generation", kind="llm", metadata={"model": "mock"}) as span:
            span.output = {"summary": "test"}
            span.usage = {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7}
    adapter.close()
    spans = exporter.get_finished_spans()
    generation = next(span for span in spans if span.name == "generation")
    root = next(span for span in spans if span.name == "agent")
    assert generation.parent.span_id == root.context.span_id
    assert generation.attributes["langfuse.observation.type"] == "generation"
    tracer.shutdown()


def test_langsmith_real_sdk_with_mocked_delivery(monkeypatch: pytest.MonkeyPatch) -> None:
    from langsmith import Client

    from ai_support_lab.observability.langsmith import LangSmithObservability

    adapter = LangSmithObservability.__new__(LangSmithObservability)
    adapter.client = Client(api_key="local-test", auto_batch_tracing=False)
    create_run = MagicMock()
    monkeypatch.setattr(Client, "create_run", create_run)
    update_run = MagicMock()
    monkeypatch.setattr(Client, "update_run", update_run)
    adapter.project = "test"
    with adapter.span("agent", inputs={"ticket_id": 1}):
        with adapter.span("tool", kind="tool") as span:
            span.output = {"id": 1}
    assert create_run.call_count == 2
    assert update_run.call_count == 2


async def test_native_langchain_tool_roundtrip_with_http_contract(
    runtime, ticket, monkeypatch
) -> None:
    import json

    from langchain_openai import ChatOpenAI

    from ai_support_lab.llm import native_langchain

    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            assert payload["tools"][0]["function"]["name"] == "get_customer_context"
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_context",
                        "type": "function",
                        "function": {"name": "get_customer_context", "arguments": "{}"},
                    }
                ],
            }
            reason = "tool_calls"
        else:
            assert any(message["role"] == "tool" for message in payload["messages"])
            assert payload["response_format"]["type"] == "json_schema"
            message = {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "category": "authentication",
                        "priority": "critical",
                        "summary": "Users cannot log in",
                        "recommended_action": "Check authentication service health",
                        "requires_human": True,
                    }
                ),
            }
            reason = "stop"
        return httpx.Response(
            200,
            json={
                "id": "test-completion",
                "object": "chat.completion",
                "created": 1,
                "model": "local-test-model",
                "choices": [{"index": 0, "message": message, "finish_reason": reason}],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        monkeypatch.setattr(
            native_langchain,
            "ChatOpenAI",
            lambda **kwargs: ChatOpenAI(**kwargs, http_async_client=client),
        )
        settings = Settings(
            _env_file=None,
            llm_backend="openai",
            llm_base_url="http://test/v1",
            llm_model="local-test-model",
        )
        result = await native_langchain.native_tool_analysis(settings, runtime.tickets, ticket.id)
        assert result.requires_human
        assert len(requests) == 2
