import httpx

from ai_support_lab.config import Settings
from ai_support_lab.llm.protocols import LLMProvider
from ai_support_lab.llm.providers.mock import MockLLM
from ai_support_lab.llm.providers.openai_compatible import OpenAICompatibleLLM


def create_llm(settings: Settings, client: httpx.AsyncClient) -> LLMProvider:
    if settings.llm_backend == "mock":
        return MockLLM()
    # Именованные backends — конфигурационные профили одного HTTP-контракта.
    # Пустые subclasses VllmProvider/LlamaCppProvider не добавили бы поведения.
    return OpenAICompatibleLLM(
        client, settings.llm_base_url, settings.llm_model, settings.llm_api_key.get_secret_value()
    )
