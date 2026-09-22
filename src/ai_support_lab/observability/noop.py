import logging
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

from ai_support_lab.observability.protocols import ObservationKind, Span

logger = logging.getLogger(__name__)


class NoopObservability:
    @contextmanager
    def span(
        self,
        name: str,
        *,
        kind: ObservationKind = "chain",
        inputs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[Span]:
        start = perf_counter()
        try:
            yield Span()
        except BaseException:
            logger.exception(
                "operation_failed operation=%s kind=%s context=%s", name, kind, metadata
            )
            raise
        finally:
            # Default сохраняет latency и correlation metadata в обычном logging,
            # но не пишет содержание обращений в stdout и ничего не отправляет наружу.
            logger.info(
                "operation_finished operation=%s latency_ms=%.1f context=%s",
                name,
                (perf_counter() - start) * 1000,
                metadata,
            )

    def close(self) -> None:
        return None
