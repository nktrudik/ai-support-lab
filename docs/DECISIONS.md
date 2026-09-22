# Architecture decisions

## Исходное состояние

На момент начала в `main` был только README с заголовком `ai-support-lab`; Python version, package manager, tests, CI и Docker отсутствовали. Полезного существующего кода не удалено. Python 3.12 выбран как консервативная совместимая база. Диапазон проекта — 3.12/3.13, фактическая проверка выполнена на 3.12; точные версии dependencies в `uv.lock`.

| Решение | Почему | Компромисс |
|---|---|---|
| Modular monolith | Один домен, легко пройти весь call stack | Нет независимого масштабирования каждого module |
| PostgreSQL + SQLAlchemy 2 | Реальные transactions, constraints, locking и familiar ORM API | Нужен процесс БД; SQLite не доказывает PostgreSQL semantics |
| Короткие transaction contexts в service | Одно место управляет commit/rollback | Полный graph не атомарен |
| Customer как nested JSON | Нужны nested Pydantic и простая domain model | Нет отдельного CRUD customer; name считается ключом истории в lab |
| Relation tickets → predictions/runs | Естественный one-to-many пример | History растёт; archival/pagination history не реализованы |
| Общий OpenAI-compatible adapter | Не копировать одинаковый HTTP-код для engines | Engine-specific capabilities проверяются отдельно |
| GPU engines optional | Core должен запускаться без NVIDIA | Реальная serving latency не проверяется mock-путём |
| sklearn и PyTorch | Разные estimator и autograd workflows в одном домене | Не выбор победителя по игрушечному датасету |
| Pure PyTorch и Lightning | Видно, какую работу берёт Trainer | Небольшое дублирование orchestration намеренно |
| LangGraph | Явные state/nodes/conditional edge | Для одной классификации прямой service проще |
| MCP использует services | Единые rules, errors, artifacts и DB | Это local trusted tool surface, без multi-tenant auth |
| Observability provider | Выбирается один SDK, нет двойной отправки | Контракт покрывает только используемые события |
| Нет Redis/Celery/Kafka/Kubernetes | Не нужны для изучаемого короткого flow | Нет job queue и durable background runs |
| Agent-bound tools без аргумента ID | LLM не может выбрать другое обращение | Tools создаются на каждый run |
| Приоритет — domain policy | Явные outage/enterprise/manual escalation rules | Это не отдельная ML priority-модель |
| Похожие tickets по product | Честный минимальный context retrieval | Не semantic similarity и не RAG |
| InMemorySaver только в тесте | Показать checkpoints без бесконечной памяти в API | Нет durable restart/resume |
| Draft validation и safe fallback | Invalid JSON не выглядит успешным анализом | Fallback требует human review |
| Backend profiles вместо пустых subclasses | vLLM/llama/TRT используют одинаковый контракт | Файлы env и docs объясняют различия server setup |
| Обучение вне lifespan | Startup не скрывает дорогой training и не пишет артефакты | После training нужен restart API/MCP |

## Конкурентность и повторные запросы

`version_id_col` защищает ORM updates. Финальный persist сверяет исходную версию ticket; изменение во время LLM вызова приводит к 409. БД используется как source of truth; schema draft не содержит ID. Priority/category/score в результате берутся из application policy и classifier, даже если LLM предложила другие.

Повторный analyze создаёт новый run и prediction. Exactly-once, idempotency keys, deterministic replay и checkpoint-backed resume не реализованы. Добавлять их следует вместе с требованиями, а не имитировать надёжность in-memory checkpointer.

## Dependencies и версии

Все перечисленные пользователем P0 technologies используются в коде. GPU stacks живут вне core environment, Python packages serving engines не входят в lockfile. Lightning и Langfuse — optional extras; LangSmith также присутствует транзитивно через LangChain, но отправка traces включается только выбранным provider с credentials. Mypy проверяет весь `src`; для нетипизированных sklearn/joblib разрешены missing stubs. Dynamic framework state и JSON/SDK boundaries используют ограниченный `Any`.

## Questions to answer after reading this code

1. Какое решение вы изменили бы первым при росте нагрузки?
2. Что ломается при попытке сделать одну AsyncSession глобальным singleton?
3. Почему отсутствие пустых provider subclasses улучшает этот lab?
4. Как добавить idempotency и durable checkpoints без ложного exactly-once?
5. Какие компромиссы приемлемы только для локальной учебной среды?
