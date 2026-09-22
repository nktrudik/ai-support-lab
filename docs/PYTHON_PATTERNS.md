# Python patterns

Пути относительно `src/ai_support_lab/`. Эта карта описывает реально используемый код; отдельные искусственные примеры patterns не добавлялись.

| Pattern | File | Short explanation |
|---|---|---|
| Structural typing / Protocol | `classical_ml/inference.py`, `llm/protocols.py` | Реализация совместима по сигнатуре без наследования |
| TypedDict | `agent/state.py` | Явные ключи промежуточного state, partial updates |
| Dataclass | `runtime.py`, `tickets/service.py`, `deep_learning/dataset.py` | Зависимости и данные конструктора видны непосредственно |
| Frozen dataclass | `runtime.py` | Набор ресурсов runtime не подменяется случайным присваиванием |
| StrEnum | `tickets/enums.py` | Ограниченный набор domain values + JSON-compatible значения |
| Dependency injection | `api/dependencies.py`, `runtime.py` | Composition root создаёт реализацию, потребители получают контракт |
| Decorator | `mcp/tools.py`, `agent/tools.py` | Регистрация typed callable как MCP/LangChain tool |
| Async generator dependency | `api/dependencies.py` | yield отдаёт session и сохраняет cleanup в одном месте |
| Async context manager | `database/session.py`, `runtime.py` | Ограничивает lifetime transaction/connection/client |
| Context manager | `observability/noop.py` | События duration/error в гарантированном finally |
| AsyncExitStack | `runtime.py` | Cleanup частично созданного runtime в обратном порядке |
| Repository | `database/repository.py` | SQL-запросы собраны отдельно от HTTP и business rules |
| Factory | `llm/factory.py`, `observability/factory.py` | Один выбор provider при старте |
| Strategy | `tickets/service.py` | sklearn/torch удовлетворяют TicketClassifier |
| Exception hierarchy | `tickets/exceptions.py` | Domain errors отделены от HTTP status |
| Exception chaining | `llm/providers/openai_compatible.py` | `raise ... from exc` сохраняет причину ошибки |
| Comprehensions | `classical_ml/dataset.py`, `tickets/service.py` | Компактная сборка records/DTO без скрытых side effects |
| pathlib | `classical_ml/train.py`, `deep_learning/training.py` | Путь и операции с артефактом без ручной склейки строк |
| Async concurrency | `agent/nodes.py` | gather независимых I/O tasks, cancel + await cleanup |
| Timeout | `agent/service.py` | Общий deadline run поверх timeout HTTP client |
| Native CPU offloading | `tickets/service.py` | to_thread освобождает event loop во время небольшого inference |

## Mutable state и lifetime

Не всё mutable плохо: SQLAlchemy session, model weights при training и graph state должны изменяться. Важно, кто ими владеет. Session принадлежит операции; обученная inference model — runtime; state — одному graph run. Нет process-global mutable registry. Module-level словари phrases используются только как read-only константы генерации.

`Protocol` проверяет совместимость статически. Он не выполняет network healthcheck и не гарантирует семантику implementation. `TypedDict` не заменяет runtime validation: входы HTTP, completion envelope и финальный draft проходят Pydantic.

`async` помогает при ожидании I/O. Он не делает матричное умножение бесплатным. Небольшой sklearn/Torch inference вынесен в worker thread; training остаётся отдельным синхронным процессом. Для тяжёлого pure-Python CPU-bound кода нужны процессы или другой serving design.

## Questions to answer after reading this code

1. Чем Protocol отличается от ABC и duck typing без аннотаций?
2. Почему TypedDict не валидирует JSON во время выполнения?
3. Чем async generator отличается от coroutine?
4. Что происходит при исключении до и после yield context manager?
5. Зачем re-raise после rollback и почему учитывается CancelledError?
6. Когда `asyncio.to_thread` освобождает loop, но не ускоряет расчёт?
