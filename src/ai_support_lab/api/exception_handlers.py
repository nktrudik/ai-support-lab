import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm.exc import StaleDataError

from ai_support_lab.tickets.exceptions import (
    DatabaseUnavailable,
    DomainError,
    InferenceTimeout,
    InferenceUnavailable,
    MalformedLLMOutput,
    ModelUnavailable,
    TicketConflict,
    TicketNotFound,
)

logger = logging.getLogger(__name__)


def register_handlers(app: FastAPI) -> None:
    @app.exception_handler(DomainError)
    async def domain_error(request: Request, exc: DomainError) -> JSONResponse:
        codes = {
            DatabaseUnavailable: 503,
            TicketNotFound: 404,
            TicketConflict: 409,
            ModelUnavailable: 503,
            InferenceTimeout: 504,
            InferenceUnavailable: 502,
            MalformedLLMOutput: 502,
        }
        if isinstance(exc, DatabaseUnavailable):
            logger.error(
                "database_unavailable request_id=%s", request.state.request_id, exc_info=exc
            )
        return JSONResponse(
            status_code=codes.get(type(exc), 400),
            content={
                "error": type(exc).__name__,
                "detail": str(exc),
                "request_id": request.state.request_id,
            },
        )

    @app.exception_handler(StaleDataError)
    async def stale_data(request: Request, exc: StaleDataError) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={"error": "TicketConflict", "detail": "Concurrent update; retry"},
        )

    @app.exception_handler(SQLAlchemyError)
    async def database_error(request: Request, exc: SQLAlchemyError) -> JSONResponse:
        logger.error("database_error request_id=%s", request.state.request_id, exc_info=exc)
        # Не возвращаем клиенту SQL, connection string и внутренние детали driver.
        return JSONResponse(
            status_code=503,
            content={
                "error": "DatabaseUnavailable",
                "detail": "Database operation failed",
                "request_id": request.state.request_id,
            },
        )
