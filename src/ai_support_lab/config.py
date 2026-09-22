from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "local"
    database_url: str = "postgresql+asyncpg://support:support@localhost:5432/support"
    classifier_backend: Literal["sklearn", "torch"] = "sklearn"
    sklearn_artifact: Path = Path("artifacts/sklearn.joblib")
    torch_artifact: Path = Path("artifacts/torch.pt")
    llm_backend: Literal["mock", "openai", "vllm", "llama_cpp", "tensorrt_llm"] = "mock"
    llm_base_url: str = ""
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = "support-mock"
    llm_timeout_seconds: float = Field(default=20, gt=0, le=120)
    agent_timeout_seconds: float = Field(default=45, gt=0, le=300)
    observability_provider: Literal["noop", "langfuse", "langsmith"] = "noop"
    langfuse_public_key: str = ""
    langfuse_secret_key: SecretStr = SecretStr("")
    langfuse_host: str = "https://cloud.langfuse.com"
    langsmith_api_key: SecretStr = SecretStr("")
    langsmith_project: str = "ai-support-lab"

    @model_validator(mode="after")
    def validate_remote_backend(self) -> "Settings":
        # Ошибку конфигурации обнаруживаем при старте, а не после первого дорогого
        # запроса. Mock не должен требовать ни URL, ни ключей внешнего провайдера.
        if self.llm_backend != "mock" and not self.llm_base_url.startswith(("http://", "https://")):
            raise ValueError("LLM_BASE_URL must be an HTTP(S) URL for remote inference")
        return self
