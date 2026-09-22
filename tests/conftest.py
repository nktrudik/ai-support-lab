from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from ai_support_lab.classical_ml.dataset import generate_dataset
from ai_support_lab.classical_ml.train import train as train_sklearn
from ai_support_lab.config import Settings
from ai_support_lab.database.models import Base
from ai_support_lab.deep_learning.training import train as train_torch
from ai_support_lab.main import create_app
from ai_support_lab.runtime import Runtime, open_runtime
from ai_support_lab.tickets.schemas import Customer, TicketCreate, TicketResponse


@pytest.fixture(scope="session")
def model_paths(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    directory = tmp_path_factory.mktemp("models")
    csv = directory / "tickets.csv"
    generate_dataset(350).to_csv(csv, index=False)
    sklearn, torch = directory / "sklearn.joblib", directory / "torch.pt"
    train_sklearn(csv, sklearn)
    train_torch(csv, torch, epochs=6)
    return sklearn, torch, csv


@pytest.fixture
def settings(tmp_path: Path, model_paths: tuple[Path, Path, Path]) -> Settings:
    return Settings(
        _env_file=None,
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'test.db'}",
        sklearn_artifact=model_paths[0],
        torch_artifact=model_paths[1],
    )


@pytest.fixture
async def runtime(settings: Settings) -> AsyncIterator[Runtime]:
    async with open_runtime(settings) as runtime:
        async with runtime.database.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        yield runtime


@pytest.fixture
async def client(settings: Settings) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with app.state.runtime.database.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client


@pytest.fixture
async def ticket(runtime: Runtime) -> TicketResponse:
    return await runtime.tickets.create(
        TicketCreate(
            title="Login password rejected",
            description="Production outage; all users blocked by authentication.",
            customer=Customer(name="Example", tier="enterprise"),
            product="cloud",
        )
    )
