from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

ObservationKind = Literal["chain", "tool", "llm"]


@dataclass
class Span:
    output: Any = None
    usage: dict[str, int] = field(default_factory=dict)


class ObservabilityProvider(Protocol):
    def span(
        self,
        name: str,
        *,
        kind: ObservationKind = "chain",
        inputs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AbstractContextManager[Span]: ...
    def close(self) -> None: ...
