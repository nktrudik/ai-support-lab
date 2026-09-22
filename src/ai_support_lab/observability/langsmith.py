from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from langsmith import Client, trace, tracing_context

from ai_support_lab.observability.protocols import ObservationKind, Span


class LangSmithObservability:
    def __init__(self, api_key: str, project: str) -> None:
        self.client = Client(api_key=api_key)
        self.project = project

    @contextmanager
    def span(
        self,
        name: str,
        *,
        kind: ObservationKind = "chain",
        inputs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[Span]:
        span = Span()
        # tracing_context включает только выбранный provider и явно связывает
        # вложенные вызовы. Не требуется глобально включать LANGSMITH_TRACING.
        with tracing_context(enabled=True, client=self.client, project_name=self.project):
            with trace(
                name,
                run_type=kind,
                inputs=inputs or {},
                metadata=metadata or {},
                client=self.client,
                project_name=self.project,
            ) as run:
                yield span
                run.end(outputs={"result": span.output, "usage_metadata": span.usage})

    def close(self) -> None:
        self.client.flush()
