from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any

from langfuse import Langfuse

from ai_support_lab.observability.protocols import ObservationKind, Span


class LangfuseObservability:
    def __init__(self, public_key: str, secret_key: str, host: str) -> None:
        self.client = Langfuse(public_key=public_key, secret_key=secret_key, base_url=host)

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
        # Native context manager использует OpenTelemetry context propagation:
        # генерации и tools внутри agent span оказываются дочерними observations.
        context: AbstractContextManager[Any]
        if kind == "llm":
            context = self.client.start_as_current_observation(
                name=name,
                as_type="generation",
                input=inputs,
                metadata=metadata,
                model=(metadata or {}).get("model"),
            )
        elif kind == "tool":
            context = self.client.start_as_current_observation(
                name=name, as_type="tool", input=inputs, metadata=metadata
            )
        else:
            context = self.client.start_as_current_observation(
                name=name, as_type="span", input=inputs, metadata=metadata
            )
        with context as observation:
            try:
                yield span
                kwargs: dict[str, Any] = {"output": span.output}
                if kind == "llm" and span.usage:
                    kwargs["usage_details"] = {
                        "input": span.usage.get("input_tokens", 0),
                        "output": span.usage.get("output_tokens", 0),
                        "total": span.usage.get("total_tokens", 0),
                    }
                observation.update(**kwargs)
            except BaseException as exc:
                observation.update(level="ERROR", status_message=type(exc).__name__)
                raise

    def close(self) -> None:
        self.client.flush()
