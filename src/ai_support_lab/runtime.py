import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass

import httpx

from ai_support_lab.agent.service import AgentService
from ai_support_lab.classical_ml.inference import SklearnTicketClassifier, TicketClassifier
from ai_support_lab.config import Settings
from ai_support_lab.database.session import Database
from ai_support_lab.deep_learning.inference import TorchTicketClassifier
from ai_support_lab.llm.factory import create_llm
from ai_support_lab.llm.langchain_bridge import ProviderChatModel
from ai_support_lab.observability.factory import create_observability
from ai_support_lab.tickets.exceptions import ModelUnavailable
from ai_support_lab.tickets.service import TicketService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Runtime:
    database: Database
    tickets: TicketService
    agent: AgentService


def load_classifiers(settings: Settings) -> dict[str, TicketClassifier]:
    factories: dict[str, Callable[[], TicketClassifier]] = {
        "sklearn": lambda: SklearnTicketClassifier(settings.sklearn_artifact),
        "torch": lambda: TorchTicketClassifier(settings.torch_artifact),
    }
    loaded: dict[str, TicketClassifier] = {}
    for name, factory in factories.items():
        try:
            loaded[name] = factory()
        except ModelUnavailable:
            # CRUD работает даже до обучения. ML endpoint отвечает явной 503,
            # а не подменяет отсутствие артефакта случайными предсказаниями.
            logger.warning(
                "model_unavailable backend=%s; train model and restart application", name
            )
    return loaded


@asynccontextmanager
async def open_runtime(settings: Settings) -> AsyncIterator[Runtime]:
    async with AsyncExitStack() as stack:
        database = Database(settings.database_url)
        stack.push_async_callback(database.close)
        http = await stack.enter_async_context(
            httpx.AsyncClient(timeout=settings.llm_timeout_seconds)
        )
        observability = create_observability(settings)
        stack.push_async_callback(asyncio.to_thread, observability.close)
        classifiers = await asyncio.to_thread(load_classifiers, settings)
        tickets = TicketService(
            database, classifiers, settings.classifier_backend, asyncio.Semaphore(2)
        )
        chat = ProviderChatModel(
            create_llm(settings, http), observability, settings.llm_backend, settings.llm_model
        )
        agent = AgentService(
            tickets, chat, observability, settings.llm_backend, settings.agent_timeout_seconds
        )
        # FastAPI и MCP используют один composition root. AsyncExitStack закрывает
        # уже созданные ресурсы даже при частичной ошибке startup следующего компонента.
        yield Runtime(database, tickets, agent)
