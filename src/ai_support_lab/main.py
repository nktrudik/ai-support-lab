import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import RequestResponseEndpoint

from ai_support_lab.api.exception_handlers import register_handlers
from ai_support_lab.api.routers import agent, health, ml, tickets
from ai_support_lab.config import Settings
from ai_support_lab.runtime import open_runtime

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
        )
        async with open_runtime(config) as runtime:
            app.state.runtime = runtime
            yield

    app = FastAPI(title="AI Support Engineering Lab", version="0.1.0", lifespan=lifespan)
    register_handlers(app)
    for router in (health.router, tickets.router, ml.router, agent.router):
        app.include_router(router)

    @app.middleware("http")
    async def request_context(request: Request, call_next: RequestResponseEndpoint) -> Response:
        request.state.request_id = str(uuid4())
        start = perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # Последняя граница логирует неожиданный programming error, затем
            # отдаёт его стандартному ASGI error handler. Ошибка не скрывается.
            logger.exception("request_failed request_id=%s", request.state.request_id)
            raise
        response.headers["X-Request-ID"] = request.state.request_id
        logger.info(
            "request request_id=%s method=%s path=%s status=%s latency_ms=%.1f",
            request.state.request_id,
            request.method,
            request.url.path,
            response.status_code,
            (perf_counter() - start) * 1000,
        )
        return response

    return app
