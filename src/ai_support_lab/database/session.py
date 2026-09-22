from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm.exc import StaleDataError

from ai_support_lab.tickets.exceptions import DatabaseUnavailable, TicketConflict


class Database:
    def __init__(self, url: str) -> None:
        self.engine = create_async_engine(url, pool_pre_ping=True)
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[AsyncSession]:
        # Один вызов service владеет транзакцией. Repository может делать flush,
        # но не commit: несколько его действий должны завершиться атомарно.
        async with self.sessions() as session:
            try:
                yield session
                await session.commit()
            except StaleDataError as exc:
                await session.rollback()
                raise TicketConflict("Concurrent update; retry") from exc
            except (SQLAlchemyError, OSError) as exc:
                await session.rollback()
                # asyncpg может вернуть сырой OSError при отказе соединения.
                # Обе формы ошибки driver получают один domain-контракт.
                raise DatabaseUnavailable("Database operation failed") from exc
            except BaseException:
                # CancelledError тоже требует отката; после cleanup отмена продолжает
                # распространяться. Контекст session закрывает соединение при выходе.
                await session.rollback()
                raise

    async def close(self) -> None:
        await self.engine.dispose()
