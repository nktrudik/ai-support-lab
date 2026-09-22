# FastAPI guide

Начните с `src/ai_support_lab/main.py`: `create_app` создаёт независимый app, а lifespan открывает runtime. Ничего не подключается к БД при простом импорте module. В тесте можно создать app с собственными Settings и отдельным SQLite-файлом.

## Request flow

`POST /api/v1/tickets` → `api/routers/tickets.py` → `TicketCreate` → `TicketService.create` → transaction → `TicketRepository.add` → `TicketResponse` → JSON.

FastAPI валидирует body до service. Router знает HTTP status и DTO, service знает бизнес-операцию, repository знает SQL. `Annotated[TicketService, Depends(...)]` скрывает повторение DI-сигнатуры, но не создаёт singleton самостоятельно.

| Endpoint | Ответ / назначение |
|---|---|
| GET `/health` | Readiness, SELECT 1 через yield dependency |
| POST `/api/v1/tickets` | 201, созданное обращение |
| GET `/api/v1/tickets/{id}` | TicketResponse, predictions загружены явно |
| GET `/api/v1/tickets` | Pagination + filters product/customer/priority/status |
| PATCH `/api/v1/tickets/{id}` | Только переданные non-null поля |
| POST `/api/v1/tickets/{id}/classify/sklearn` | Сохранённый prediction |
| POST `/api/v1/tickets/{id}/classify/torch` | Та же схема ответа от другой модели |
| POST `/api/v1/tickets/{id}/analyze` | LangGraph result; fallback явно отмечен |
| POST `/api/v1/tickets/{id}/resolve` | Явное решение + optimistic version check |

## Pydantic на границе

В `tickets/schemas.py` найдите `BaseModel`, `Field`, `field_validator`, `model_validator`, nested Customer и enums. `model_dump(exclude_unset=True)` различает переданное поле и default; явный null PATCH запрещён. `TicketResponse.model_validate(orm_object)` использует `from_attributes=True`. `validation_alias='metadata_json'` читает ORM-атрибут, публичный JSON при этом остаётся `metadata`.

`Settings` использует Pydantic Settings. SecretStr маскирует секрет при обычном repr; `get_secret_value()` вызывается только на границе создания внешнего клиента. Environment имеет приоритет над `.env`; тестовые Settings создаются с `_env_file=None`.

## Lifetime и errors

`get_read_session` показывает dependency с yield: context закрывает session после использования. Business services самостоятельно открывают короткие transactions. Это позволяет API и MCP переиспользовать одинаковые правила и не держать транзакцию во время HTTP-вызова LLM.

Централизованный handler переводит domain errors в HTTP. SQLAlchemy error не отдаёт SQL и connection string клиенту. Middleware создаёт request ID и log latency; unexpected exceptions логируются и продолжают распространяться.

Тест `test_fastapi_dependency_override` заменяет service через `app.dependency_overrides`. Остальные integration tests используют настоящие services и ORM, а не только проверяют, что mock был вызван.

## Questions to answer after reading this code

1. Почему endpoint async и где он действительно ждёт I/O?
2. Что делает dependency с yield после исключения?
3. Почему business logic не находится в router?
4. Как FastAPI отличает path, query и body parameters?
5. Зачем разные TicketCreate, TicketUpdate и TicketResponse?
6. Чем omission поля отличается от null в PATCH?
7. Что именно dependency override заменяет в тесте?
8. Зачем app factory вместо глобального подключения к БД при импорте?
