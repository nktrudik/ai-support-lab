# AI Support Engineering Lab

Учебный production-like проект для опытного ML/NLP Engineer, который хочет восстановить навык чтения Python backend, ML и LLM/agent кода перед собеседованиями. Один домен — support tickets. Основной режим работает на CPU, без API-ключей и скачивания LLM.

Начните с [маршрута чтения](docs/CODE_READING_GUIDE.md). Архитектурные компромиссы записаны в [DECISIONS](docs/DECISIONS.md), фактически выполненные проверки — в [VERIFICATION](docs/VERIFICATION.md).

## What this repository teaches

Все пути в таблице относительно `src/ai_support_lab/`, если не указан другой каталог.

| Technology | Where to look | Что проследить |
|---|---|---|
| Modern Python | `runtime.py`, `tickets/service.py` | dataclass, Protocol, DI, context managers |
| FastAPI | `main.py`, `api/` | lifespan, routers, dependencies, HTTP errors |
| Pydantic v2 | `tickets/schemas.py`, `config.py` | validation, PATCH, aliases, ORM DTO |
| SQLAlchemy 2 | `database/` | AsyncSession, relationships, transaction, optimistic locking |
| PostgreSQL / Alembic | `migrations/`, `docker-compose.yml` в корне | schema evolution, healthcheck |
| pandas / NumPy | `classical_ml/dataset.py`, `features.py` | deterministic data, masks, broadcasting, aggregation |
| scikit-learn | `classical_ml/pipeline.py`, `train.py`, `inference.py` | split, Pipeline, TF-IDF, metrics, artifact |
| PyTorch | `deep_learning/` | Dataset, tensor shapes, pooling, backward, checkpoint |
| Lightning | `deep_learning/lightning_module.py` | та же модель, automatic optimization, Trainer |
| LangChain | `llm/langchain_bridge.py`, `agent/nodes.py`, `llm/native_langchain.py` | chat abstraction, messages, prompt, parser, native tools |
| LangGraph | `agent/state.py`, `nodes.py`, `graph.py` | state updates, conditional edge, invocation |
| MCP / FastMCP | `mcp/server.py`, `mcp/tools.py` | отдельный процесс, tools, resource, prompt |
| Mock / HTTP inference | `llm/providers/`, `llm/factory.py` | один Protocol, разные реализации |
| vLLM / llama.cpp | `experiments/*.env.example`, [Inference guide](docs/INFERENCE_GUIDE.md) | отдельные serving engines, общий HTTP adapter |
| TensorRT-LLM | `experiments/tensorrt_llm/` | advanced NVIDIA integration, build/runtime |
| Langfuse / LangSmith | `observability/` | trace, generation, tool span, usage, errors |
| Async Python | `database/session.py`, `agent/nodes.py` | await, gather, timeout, cancellation cleanup |
| pytest / mypy / Ruff | `tests/`, `pyproject.toml`, `.github/workflows/ci.yml` в корне | fixtures, mocks, overrides, CPU CI |

## Архитектура

FastAPI и отдельный FastMCP server используют общий composition root `runtime.py` и одни application services. PostgreSQL хранит tickets, predictions и agent runs. Модели загружаются при старте процесса; обучение выполняется отдельной CLI-командой. После обучения перезапустите API/MCP, чтобы перечитать артефакты.

LLM не закрывает обращения автоматически: `/analyze` сохраняет рекомендацию и вычисленный приложением приоритет. `/resolve` принимает явный текст решения и ожидаемую версию ticket. Подробнее: [диаграммы архитектуры](docs/ARCHITECTURE.md).

## Quickstart: PostgreSQL + API на CPU

Нужны Python 3.12, [uv](https://docs.astral.sh/uv/getting-started/installation/) и Docker Compose. Команды ниже выполняются из корня репозитория в Bash; на Windows подойдёт WSL. `.env.example` содержит только локальные демонстрационные значения.

```bash
cp .env.example .env
uv sync --locked --all-extras
docker compose up -d postgres
uv run alembic upgrade head
uv run python scripts/generate_dataset.py
uv run python scripts/train_sklearn.py
uv run python scripts/train_torch.py
uv run python scripts/seed_db.py
uv run uvicorn ai_support_lab.main:create_app --factory --reload
```

`uv sync --locked` достаточно для core; `--all-extras` добавляет Lightning и SDK observability. В Linux/Windows uv выбирает CPU wheel PyTorch. GPU engines не устанавливаются в этот environment. Lockfile фиксирует проверенное разрешение зависимостей; не удаляйте его для обычного запуска.

API: `http://localhost:8000/docs`. Readiness: `GET /health` проверяет соединение с БД. `seed_db.py` создаёт три новых обращения при каждом вызове и печатает их ID.

Минимальный путь без PyTorch-обучения: пропустите `train_torch.py`, используйте sklearn и `/analyze`. Torch endpoint до обучения вернёт `503 ModelUnavailable`; случайных predictions нет.

## End-to-end demonstration

В другом терминале:

```bash
uv run python scripts/smoke_api.py
```

Скрипт создаёт обращение, вызывает оба классификатора и LangGraph, затем печатает `ticket_id`. Тот же flow вручную:

```bash
curl -s http://localhost:8000/api/v1/tickets \
  -H 'Content-Type: application/json' \
  -d '{"title":"Login password rejected","description":"Production outage; all users blocked by authentication.","customer":{"name":"Example","tier":"enterprise"},"product":"cloud","metadata":{"source":"demo"}}'
```

Возьмите `id` из ответа и подставьте его вместо `1`:

```bash
curl -s -X POST http://localhost:8000/api/v1/tickets/1/classify/sklearn
curl -s -X POST http://localhost:8000/api/v1/tickets/1/classify/torch
curl -s -X POST http://localhost:8000/api/v1/tickets/1/analyze \
  -H 'Content-Type: application/json' -d '{"include_context":true}'
curl -s 'http://localhost:8000/api/v1/tickets?product=cloud&priority=critical&limit=10'
```

Для critical ticket graph вызывает `get_ticket`, `classify_ticket`, параллельно `get_customer_context` и `get_similar_tickets`. Список вызовов возвращается в `tools_used`. Ответ включает `category`, `priority`, `summary`, `recommended_action`, `requires_human`, `classification_score`, `run_id`, `used_fallback` и `warnings`.

С `OBSERVABILITY_PROVIDER=noop` терминал показывает operation/run ID и latency. Чтобы увидеть trace UI, настройте **один** provider в `.env`, перезапустите API и повторите `/analyze`; [инструкция](docs/OBSERVABILITY_GUIDE.md).

Закрытие обращения выполняется отдельно: получите актуальную `version` через GET, затем передайте её в `expected_version`:

```bash
curl -s -X POST http://localhost:8000/api/v1/tickets/1/resolve \
  -H 'Content-Type: application/json' \
  -d '{"resolution":"Verified token rotation with the customer.","expected_version":2}'
```

`2` — пример, фактическая версия зависит от выполненных изменений. При конфликте получите 409 и перечитайте ticket.

## PyTorch vs Lightning

```bash
uv run python scripts/train_torch.py --epochs 8
uv run --extra lightning python scripts/train_lightning.py --epochs 8
```

Одинаковые dataset, vocabulary, split, Embedding + masked pooling + MLP. Разные владельцы training loop. Артефакты: `artifacts/torch.pt` и `artifacts/lightning.pt`. Для обслуживания Lightning-обученной модели задайте `TORCH_ARTIFACT=artifacts/lightning.pt` и перезапустите API: inference class тот же. [Сравнение по операциям](docs/PYTORCH_GUIDE.md).

## MCP

```bash
uv run python -m ai_support_lab.mcp.server
uv run python -m ai_support_lab.mcp.server --transport http --port 8001
```

Первая команда — stdio server, ожидающий MCP-клиента; это не обычный interactive shell. Вторая — Streamable HTTP на `http://127.0.0.1:8001/mcp`. Для демонстрации отдельного stdio процесса используйте существующий ID:

```bash
uv run python scripts/smoke_mcp.py 1
```

Клиент вызовет `analyze_ticket`, прочитает `ticket://1` и получит prompt. БД и артефакты общие с API. [Tools vs resources vs prompts](docs/MCP_GUIDE.md).

## API в Docker

Сначала обучите артефакты на хосте командами quickstart:

```bash
docker compose build api
docker compose run --rm api uv run --no-sync alembic upgrade head
docker compose up -d api
```

Образ содержит core, `.env` и артефакты не запекаются в него; модели монтируются read-only. Compose по умолчанию использует mock и noop. Для экспериментов с remote inference удобнее API на хосте и отдельный сервер модели: адрес `localhost` внутри контейнера означает сам контейнер.

## Optional inference and tracing

- **Mock:** `LLM_BACKEND=mock`, ключи не нужны.
- **vLLM:** отдельное NVIDIA environment; профиль `experiments/vllm.env.example`.
- **llama.cpp:** отдельный `llama-server` с локальным GGUF; профиль `experiments/llama_cpp.env.example`.
- **TensorRT-LLM:** [hardware-dependent experiment](experiments/tensorrt_llm/README.md).
- **Langfuse/LangSmith:** [настройка и границы проверки](docs/OBSERVABILITY_GUIDE.md).
- **Native LangChain tool calling:** после настройки совместимой remote-модели `uv run python scripts/native_langchain.py 1`. Это read-only draft; backend должен поддерживать tools и JSON schema.

Команды запуска serving engines, healthchecks и пример HTTP-запроса находятся в [INFERENCE_GUIDE](docs/INFERENCE_GUIDE.md). Общий HTTP adapter проверен контрактными тестами; это не проверка GPU engines.

## Проверки

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -q
uv run alembic check
```

Основные tests используют SQLite. Отдельный PostgreSQL test выполняется в CI после миграций; локально задайте `TEST_DATABASE_URL` на подготовленную тестовую БД. Test добавляет synthetic records. Production DB для тестов не нужна.

Makefile даёт короткие команды `install`, `run`, `migrate`, `seed`, `test`, `lint`, `format`, `typecheck`, `generate-data`, `train-sklearn`, `train-torch`, `train-lightning`, `run-mcp`.

## Границы учебного проекта

Нет authentication, multi-tenant authorization, очереди jobs, semantic retrieval, автоматического исполнения LLM-рекомендаций и durable graph resume. HTTP порты привязаны к loopback. Это среда для локального изучения, не готовый публичный SaaS. Synthetic metrics не измеряют качество на реальных tickets. `score` — вероятность выбранного класса, без отдельной calibration.

Полный перечень решений и ограничений: [DECISIONS](docs/DECISIONS.md). Проверенные команды и неподтверждённые внешние интеграции: [VERIFICATION](docs/VERIFICATION.md).
