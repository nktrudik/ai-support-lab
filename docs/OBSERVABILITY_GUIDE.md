# Observability guide

Один run имеет root span `ticket_analysis`; дочерние spans создаются для tools и LLM generation. Точки instrumentation находятся в `agent/service.py`, `agent/tools.py` и `llm/langchain_bridge.py`. SDK provider выбирается в `observability/factory.py` один раз при startup.

| Данные | Где появляются |
|---|---|
| run_id, ticket_id | metadata root и tool spans |
| backend, model | generation metadata |
| input | request facts/messages |
| output | tool result, LLMResponse, final analysis |
| latency | duration native spans; perf_counter в noop logging |
| tokens | LLMResponse.usage, если backend их вернул |
| errors | SDK error context / explicit Langfuse ERROR |

## Noop: default

```text
OBSERVABILITY_PROVIDER=noop
```

Noop не отправляет данные наружу. Он оставляет operation name, correlation metadata и latency в standard logging, без полного текста tickets. Отсутствующие Langfuse/LangSmith credentials при выборе соответствующего provider дают warning и fallback к noop. Наличие ключей при отсутствии optional SDK — ошибка installation: используйте `uv sync --locked --extra observability`.

## Langfuse

Установите extras, создайте project в своём Langfuse и задайте `.env`:

```text
OBSERVABILITY_PROVIDER=langfuse
LANGFUSE_PUBLIC_KEY=<your-public-key>
LANGFUSE_SECRET_KEY=<your-secret-key>
LANGFUSE_HOST=https://cloud.langfuse.com
```

После restart API вызовите analyze и найдите root `ticket_analysis` по run_id. SDK использует `start_as_current_observation` с типами span/tool/generation, native parent context и `usage_details`. `flush` вызывается при завершении runtime через worker thread.

Внешний Langfuse server может быть hosted или self-hosted. Его полный ClickHouse/Redis/storage stack в Compose не добавлен: это отвлекло бы от Python patterns. Задайте URL уже работающего сервиса. [SDK instrumentation](https://langfuse.com/docs/observability/sdk/instrumentation).

## LangSmith

```text
OBSERVABILITY_PROVIDER=langsmith
LANGSMITH_API_KEY=<your-key>
LANGSMITH_PROJECT=ai-support-lab
```

Adapter использует `tracing_context` и `trace`, а не самодельную отправку HTTP events. Run types chain/tool/llm сохраняют hierarchy. Usage передаётся в outputs. [Custom instrumentation](https://docs.langchain.com/langsmith/annotate-code).

Не включайте отдельно глобальный `LANGSMITH_TRACING=true`, если хотите только Langfuse/noop: глобальные framework callbacks могут отправлять дополнительные traces независимо от выбранного application provider. Lab сам этот global switch не устанавливает.

## Что проверено

- Noop проходит настоящий graph flow.
- Langfuse SDK проверяется с локальным OpenTelemetry exporter: реальные spans, generation type и parent relationship.
- LangSmith SDK context проверяется с mocked create/update delivery.
- Доставка в облако и корректность project credentials этими тестами не подтверждаются.

При включении cloud tracing inputs/outputs могут включать customer data и description. Для реальных данных потребуется политика redaction/retention. Lab использует synthetic fixtures. Mock LLM не выдумывает token counts: usage отсутствует, если backend его не сообщил.

## Questions to answer after reading this code

1. Чем trace, span и generation отличаются друг от друга?
2. Как сохраняется parent context через async tasks?
3. Где вы найдёте backend/model и token usage?
4. Что произойдёт при исключении внутри span context?
5. Чем SDK contract test отличается от проверки доставки в cloud?
6. Почему native SDK context лучше ручного преобразования всех событий в общий JSON?
7. Какие поля требуют redaction перед использованием реальных tickets?
