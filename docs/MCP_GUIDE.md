# MCP / FastMCP guide

`mcp/server.py` создаёт отдельный FastMCP process с lifespan; `mcp/tools.py` регистрирует capabilities. FastMCP установлен как Python framework, MCP — протокол взаимодействия клиента и сервера. Основной API может быть остановлен: MCP всё равно использует общую PostgreSQL БД и локальные trained artifacts.

| Capability | Пример | Кто инициирует / что происходит |
|---|---|---|
| Tool | analyze_ticket(ticket_id) | Клиент вызывает операцию, возможны side effects |
| Resource | ticket://123 | Клиент читает адресуемый контекст |
| Prompt | support_ticket_analysis | Клиент получает шаблон инструкции для дальнейшего использования |

Prompt не запускает модель, resource не классифицирует ticket. Tools `classify_ticket` и `analyze_ticket` записывают prediction/run, `update_ticket_priority` меняет данные. `get_ticket` и `search_tickets` читают. Все операции делегируют application layer; SQL и ML не копируются в MCP-файл.

## Lifetime

FastMCP lifespan открывает `open_runtime(Settings())`, помещает runtime в context и закрывает ресурсы при завершении сервера. `Context` — framework dependency, она не становится аргументом JSON tool schema. В in-process тесте runtime принадлежит pytest fixture; server его использует, но не закрывает самостоятельно.

## Transports

```bash
uv run python -m ai_support_lab.mcp.server
uv run python -m ai_support_lab.mcp.server --transport http --port 8001
```

stdio передаёт protocol messages через stdin/stdout дочернего процесса; logs должны идти в stderr. Streamable HTTP использует `http://127.0.0.1:8001/mcp`. Локальный HTTP server не имеет authentication: binding на loopback выбран намеренно. Для внешнего размещения потребуются authentication/authorization и transport security.

После создания tickets:

```bash
uv run python scripts/smoke_mcp.py 1
```

Скрипт использует `StdioTransport`, открывает новый сервер, вызывает tool, читает resource и получает prompt. В tests дополнительно используется `Client(server)` без сетевого процесса: это реальный MCP dispatch с protocol schemas, но другой transport.

## Typed boundaries

FastMCP строит tool schemas из signature и docstring. `TicketFilters` валидирует limit даже при прямом MCP-вызове. Priority — enum, а response — Pydantic DTO. `mask_error_details=True` не раскрывает детали SQL через неожиданные server errors.

В lab клиент считается доверенным пользователем всех tickets. Привязка tools основного LangGraph к одному ID защищает от произвольного выбора ID моделью, но не является полноценной tenant authorization для публичного MCP server.

Официальные ссылки: [lifespan](https://gofastmcp.com/servers/lifespan), [client](https://gofastmcp.com/clients/client), [tools](https://gofastmcp.com/servers/tools).

## Questions to answer after reading this code

1. Чем отличаются tool, resource и prompt?
2. Что такое transport и почему stdio не HTTP?
3. Как framework строит schemas из type hints?
4. Кто владеет DB/HTTP clients отдельного MCP процесса?
5. Почему application services общие с REST API?
6. Как отличить in-process MCP test от проверки отдельного процесса?
7. Какие tools имеют side effects и как клиент узнает об этом из описания?
