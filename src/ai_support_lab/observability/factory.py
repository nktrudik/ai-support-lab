import logging

from ai_support_lab.config import Settings
from ai_support_lab.observability.noop import NoopObservability
from ai_support_lab.observability.protocols import ObservabilityProvider

logger = logging.getLogger(__name__)


def create_observability(settings: Settings) -> ObservabilityProvider:
    if (
        settings.observability_provider == "langfuse"
        and settings.langfuse_public_key
        and settings.langfuse_secret_key.get_secret_value()
    ):
        from ai_support_lab.observability.langfuse import LangfuseObservability

        return LangfuseObservability(
            settings.langfuse_public_key,
            settings.langfuse_secret_key.get_secret_value(),
            settings.langfuse_host,
        )
    if (
        settings.observability_provider == "langsmith"
        and settings.langsmith_api_key.get_secret_value()
    ):
        from ai_support_lab.observability.langsmith import LangSmithObservability

        return LangSmithObservability(
            settings.langsmith_api_key.get_secret_value(), settings.langsmith_project
        )
    if settings.observability_provider != "noop":
        logger.warning("Observability credentials missing; using local logging")
    return NoopObservability()
