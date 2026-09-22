import os
from pathlib import Path

import pytest
from sqlalchemy import select

from ai_support_lab.agent.state import AgentAnalysisRequest
from ai_support_lab.config import Settings
from ai_support_lab.database.models import AgentRun
from ai_support_lab.runtime import open_runtime
from ai_support_lab.tickets.schemas import Customer, TicketCreate


@pytest.mark.postgres
async def test_postgresql_migrated_schema_and_agent(model_paths: tuple[Path, Path, Path]) -> None:
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip(
            "Set TEST_DATABASE_URL and run Alembic against the isolated PostgreSQL database"
        )
    settings = Settings(
        _env_file=None,
        database_url=url,
        sklearn_artifact=model_paths[0],
        torch_artifact=model_paths[1],
    )
    async with open_runtime(settings) as runtime:
        ticket = await runtime.tickets.create(
            TicketCreate(
                title="Invoice charged twice",
                description="Refund payment missing after subscription renewal.",
                customer=Customer(name="CI synthetic customer"),
                product="cloud",
            )
        )
        prediction = await runtime.tickets.classify(ticket.id)
        assert prediction.category.value == "billing"
        analysis = await runtime.agent.analyze(ticket.id, AgentAnalysisRequest())
        assert not analysis.used_fallback
        async with runtime.database.transaction() as session:
            row = await session.scalar(select(AgentRun).where(AgentRun.id == analysis.run_id))
            assert row is not None and row.ticket_id == ticket.id
