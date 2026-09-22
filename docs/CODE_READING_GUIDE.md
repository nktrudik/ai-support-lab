# Code Reading Guide

Читайте один пользовательский сценарий от входа до результата. После каждого уровня запустите соответствующую команду или тест и объясните flow своими словами. Все пути ниже относятся к `src/ai_support_lab/`, если не указан корень `tests/` или `scripts/`.

## Level 1: Python backend

1. `main.py`: app factory, lifespan, request ID, регистрация routers.
2. `api/routers/tickets.py`: вход запроса, path/query/body, HTTP status.
3. `tickets/schemas.py`: отдельные create/update/response DTO, nested customer, validators.
4. `api/dependencies.py`: где FastAPI находит service и где yield закрывает read session.
5. `tickets/service.py`: create/get/update/resolve и владелец транзакций.
6. `database/repository.py`: select/scalar/scalars, filters, eager loading.
7. `database/models.py`: Mapped, relationships, FK, timestamps, version.
8. `database/session.py`, затем `runtime.py`: session factory, rollback и cleanup ресурсов.

Запуск: `uv run pytest tests/integration/test_api_and_repository.py -q`.

Проверьте понимание: где request входит в приложение? Где происходит validation? Кто создаёт session? Кто имеет право commit? Какое ожидание async, а какое вычисление синхронно? Какие строки зависят от FastAPI, а какие выражают domain logic?

## Level 2: Classical ML

Порядок: `classical_ml/dataset.py` → `features.py` → `pipeline.py` → `train.py` → `inference.py` → вызов `TicketService.classify`.

Найдите: загрузку CSV, очистку, split, обучение vocabulary/IDF, estimator fit, predict, predict_proba, metrics и artifact loading. В `tests/unit/test_ml.py` сравните `fit_transform` и `transform`: почему число колонок одинаковое для train и test?

Запуск: `uv run python scripts/generate_dataset.py`, затем `uv run python scripts/train_sklearn.py`.

Вопросы: что произойдёт при неизвестном слове? Где появляется leakage, если обучить vectorizer до split? Почему высокий synthetic F1 не означает готовность к production?

## Level 3: PyTorch

Порядок: `deep_learning/dataset.py` → `model.py` → `training.py` → `inference.py`.

На листе запишите формы: token IDs `[B,L]`, embeddings `[B,L,D]`, pooled `[B,D]`, logits `[B,C]`, targets `[B]`. Найдите Dataset, DataLoader, dtype/device, train/eval, loss, zero_grad, backward, optimizer.step, checkpoint и inference_mode.

Запуск: `uv run python scripts/train_torch.py --epochs 8`.

Вопросы: зачем маска padding? Почему нельзя применить softmax перед CrossEntropyLoss? Что хранится в `.grad`? Что нужно сохранить кроме весов?

## Level 4: Lightning

Читайте `deep_learning/lightning_module.py`, держа рядом `training.py`. Модель и входные данные те же. Сопоставьте ручные операции с `training_step`, `validation_step`, `configure_optimizers`, `Trainer` и `ModelCheckpoint`.

Запуск: `uv run --extra lightning python scripts/train_lightning.py`.

Вопросы: где теперь backward? Кто переносит batch на device? Какая разница между Lightning checkpoint и экспортом `artifacts/lightning.pt`? Почему `self.log` должен знать batch size?

## Level 5: LangGraph

Порядок: `agent/state.py` → `agent/tools.py` → `agent/nodes.py` → `agent/graph.py` → `agent/service.py`.

Для critical ticket проследите START → load → classify → priority → context → generation → validation → persist → END. Каждый node получает state и возвращает только изменения. Router возвращает имя ветки, а не самостоятельно вызывает следующий node.

Запуск: `uv run pytest tests/integration/test_agent_and_mcp.py -q`.

Найдите tool execution, `asyncio.gather`, fallback, graph.compile, ainvoke и тест history с InMemorySaver. Сравните прямой `tickets.classify` с graph: какие дополнительные шаги и ответственность добавились?

## Level 6: MCP

Порядок: `mcp/server.py` → `mcp/tools.py` → `scripts/smoke_mcp.py`.

Найдите lifetime отдельного процесса, `@server.tool`, URI resource, prompt template и повторное использование service. Почему prompt ничего не анализирует сам? Почему stdio server нельзя проверять обычным curl?

Запуск после seed: `uv run python scripts/smoke_mcp.py 1`, заменив ID реальным.

## Level 7: Inference

Порядок: `llm/protocols.py` → `schemas.py` → `providers/mock.py` → `providers/openai_compatible.py` → `factory.py` → `langchain_bridge.py` → `native_langchain.py`.

Проследите agent → chat model bridge → LLMProvider → HTTP client → inference server. Выделите три независимые ошибки: HTTP transport, неправильный completion envelope и неправильный structured draft. Почему ID отсутствует в схеме draft? Какие capabilities нужны native tool-calling примеру сверх обычного chat endpoint?

## Level 8: Observability

Порядок: `observability/protocols.py` → `noop.py` → `factory.py` → `langfuse.py` → `langsmith.py`.

Затем вернитесь к `agent/service.py`, `agent/tools.py`, `llm/langchain_bridge.py`: это реальные точки создания run/tool/generation spans. Найдите input/output, backend/model, duration, usage и error. В `tests/unit/test_llm_and_observability.py` увидите отличие проверки SDK с локальным exporter от отправки в облачный сервис.

## Questions to answer after reading this code

1. Объясните create → classify → analyze → resolve без обращения к документации.
2. Какие части можно использовать без HTTP и без LangGraph?
3. Как добавить новый classifier, не меняя router?
4. Где хранится mutable state и кто отвечает за его lifetime?
5. Что проверено локально, а что требует PostgreSQL, GPU или API keys?
6. Какие изменения потребовались бы для публичного production deployment?
7. Где здесь абстракция уменьшает связность, а где достаточно обычной функции?
