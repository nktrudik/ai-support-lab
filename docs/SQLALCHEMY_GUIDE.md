# SQLAlchemy guide

Читайте `database/models.py` → `database/session.py` → `database/repository.py` → `tickets/service.py`. Пути внутри `src/ai_support_lab/`.

## Mapping и запросы

`Base` наследует DeclarativeBase. `Mapped[T]` описывает Python-тип, `mapped_column` — колонку. Ticket имеет predictions и agent_runs; дочерние таблицы используют FK и индекс `ticket_id`. Composite index `(product, created_at)` соответствует типичному фильтру по продукту и сортировке.

Дата хранится как timezone-aware DateTime для PostgreSQL; defaults задаются Python-кодом. Raw SQL INSERT обязан сам передать значения, потому что это не server defaults. Enum хранится как строковое имя Python enum (`HIGH`), DTO сериализует value (`high`). `native_enum=False` уменьшает различия миграций SQLite/PostgreSQL.

| API | Где | Значение |
|---|---|---|
| `select(Ticket)` | repository.get/list | ORM SELECT нового стиля |
| `session.scalar` | get, count | Первый scalar или None |
| `session.scalars` | list | Последовательность ORM-объектов |
| `selectinload` | get/list | Отдельный запрос для relationships без N+1 |
| `lazy='raise'` | models | Запрет неявного I/O при доступе к relation |
| `with_for_update` | repository.get | PostgreSQL row lock внутри write transaction |
| `version_id_col` | Ticket | Optimistic concurrency для ORM updates |
| `flush` | repository.add, service.update | SQL выполнен, transaction ещё не committed |
| `refresh` | repository.add | Перечитать значения/relationship перед DTO |
| `commit / rollback` | Database.transaction | Финальная граница операции |

## Session не равна connection и не равна database

AsyncSession содержит identity map и unit of work. Connection берётся из pool, когда нужен I/O. Два чтения одной строки в одной session обычно возвращают один Python object; тест проверяет `first is second`. Одна AsyncSession не должна обслуживать параллельные tasks.

`expire_on_commit=False` позволяет не инициировать повторное чтение после commit. DTO всё равно формируется до закрытия session, relationships загружаются явно. Это не разрешение использовать detached ORM-объект как универсальный API response.

Service открывает `database.transaction()`, repository делает flush, context делает commit. Исключение после flush всё ещё приводит к rollback. Test изменяет title, flush-ит и выбрасывает исключение: следующий read видит исходное значение.

## Конкурентный анализ

Агент читает snapshot+version, освобождает transaction и вызывает LLM. Финальный persist открывает новую transaction, блокирует строку и сверяет версию. При изменении ticket результат не записывается. Prediction из предварительной классификации может остаться; это осознанная граница атомарности.

SQLite не воспроизводит PostgreSQL row locks, timestamp semantics и весь concurrency behavior. Поэтому `tests/integration/test_postgres.py` проверяет мигрированную PostgreSQL-схему отдельным CI service.

## Alembic

```bash
uv run alembic upgrade head
uv run alembic current
uv run alembic check
```

`migrations/env.py` запускает async engine и передаёт sync connection callback через `run_sync`. Начальная migration содержит фиксированные `op.create_table`, а не импорт будущего `Base.metadata.create_all`. Для изменения schema создавайте новую revision; уже применённую migration не редактируют. `downgrade base` удаляет таблицы, поэтому для чтения используйте disposable test DB.

## Questions to answer after reading this code

1. Что такое AsyncSession и identity map?
2. Чем flush отличается от commit?
3. Зачем refresh, если объект уже существует в Python?
4. Что происходит с изменениями и соединением при rollback?
5. Что такое lazy loading и почему он опасен в async serialization?
6. Чем pessimistic lock отличается от optimistic version check?
7. Почему нельзя использовать одну session в asyncio.gather?
8. Что именно SQLite-тест не доказывает о PostgreSQL?
