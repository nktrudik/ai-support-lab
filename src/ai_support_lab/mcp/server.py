import argparse
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastmcp import FastMCP

from ai_support_lab.config import Settings
from ai_support_lab.mcp.tools import register_capabilities
from ai_support_lab.runtime import Runtime, open_runtime


def create_server(settings: Settings | None = None, runtime: Runtime | None = None) -> FastMCP:
    @asynccontextmanager
    async def lifespan(server: FastMCP) -> AsyncIterator[dict[str, Any]]:
        if runtime is not None:
            # В in-process тесте ресурс принадлежит fixture, поэтому сервер
            # не закрывает чужой runtime. В отдельном процессе он владеет им сам.
            yield {"runtime": runtime}
        else:
            async with open_runtime(settings or Settings()) as owned_runtime:
                yield {"runtime": owned_runtime}

    server = FastMCP("AI Support Engineering Lab", lifespan=lifespan, mask_error_details=True)
    register_capabilities(server)
    return server


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    server = create_server()
    if args.transport == "stdio":
        server.run(transport="stdio")
    else:
        server.run(transport="http", host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
