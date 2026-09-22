# Verification

Дата проверки: 2026-09-22. Среда: Linux, Python 3.12.14, CPU PyTorch. Источник точных package versions — `uv.lock`.

## Verified core

- `uv sync --all-extras`: зависимости установлены, NVIDIA/CUDA packages для core не требуются.
- `uv run ruff check .` и `uv run ruff format --check .`.
- `uv run mypy`: весь `src`, включая adapters.
- `uv run pytest -q`: 37 tests проходят локально; PostgreSQL test отдельно ожидает TEST_DATABASE_URL.
- `uv run python scripts/generate_dataset.py`: 1400 deterministic records.
- `uv run python scripts/train_sklearn.py`: training, artifact save/load, accuracy/precision/recall/macro F1/confusion matrix.
- `uv run python scripts/train_torch.py`: 8 epochs CPU, validation, checkpoint, inference.
- `uv run python scripts/train_lightning.py`: 8 epochs CPU, best checkpoint и совместимый inference export.
- Alembic upgrade/check на disposable SQLite database: schema не расходится с ORM.
- Отдельный Uvicorn process + `scripts/smoke_api.py`: create → sklearn → torch → LangGraph, без fallback.
- Отдельный FastMCP stdio process + `scripts/smoke_mcp.py`: analyze tool, ticket resource и prompt работают с той же SQLite DB.

SQLite smoke запуск использовал `DATABASE_URL=sqlite+aiosqlite:////tmp/ai-support-migrations.db`. Это проверка service/ORM/API/MCP flow, а не PostgreSQL locking.

На синтетическом holdout sklearn accuracy/macro F1 = 1.0. Torch validation accuracy = 1.0 после 8 epochs. Эти результаты ожидаемы для простых повторяющихся templates и не являются оценкой на real-world dataset. Lightning export загружается через тот же TorchTicketClassifier.

## Контрактные SDK-проверки

- OpenAI-compatible HTTP adapter: request path/headers, completion envelope, usage, HTTP error, timeout, invalid response.
- LangChain native ChatOpenAI: реальный SDK выполняет bind_tools → ToolMessage → with_structured_output поверх MockTransport.
- Langfuse SDK: in-memory OpenTelemetry exporter, generation type и parent span relationship; sampler явно ALWAYS_ON в тесте.
- LangSmith SDK: native trace context с подменённой create/update delivery.

Это проверяет SDK API и application wiring, но не подтверждает доставку в облачные проекты или работу GPU engine.

## PostgreSQL и Docker

Локально Docker и PostgreSQL недоступны. GitHub Actions настроен на PostgreSQL 16 service, Alembic upgrade/check, отдельный PostgreSQL end-to-end test, полный CPU suite, Lightning smoke и сборку Docker image. Результат конкретного CI run нужно смотреть в Actions; наличие workflow само по себе не является успешной проверкой.

## Optional / hardware-dependent

Не запускались на реальном оборудовании/сервисе: vLLM server, llama.cpp + GGUF, TensorRT-LLM, CUDA training, cloud Langfuse и LangSmith. Команды и требования описаны в INFERENCE_GUIDE, OBSERVABILITY_GUIDE и TensorRT experiment. Python 3.13 и macOS/Windows не проверялись в данной среде.

## Questions to answer after reading this code

1. Что доказывает mocked HTTP contract, а что требует настоящего model server?
2. Чем PostgreSQL integration отличается от SQLite unit/integration tests?
3. Почему успешное обучение на templates не означает качество на production data?
4. Почему наличие CI configuration нельзя выдавать за зелёный CI run?
