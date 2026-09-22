from unittest.mock import AsyncMock

import httpx
import pytest
from sqlalchemy import func, select

from ai_support_lab.api.dependencies import get_ticket_service
from ai_support_lab.database.models import Ticket
from ai_support_lab.database.repository import TicketRepository
from ai_support_lab.runtime import Runtime
from ai_support_lab.tickets.schemas import TicketResponse

PAYLOAD = {
    "title": "Login password rejected",
    "description": "Authentication token expired for all users.",
    "customer": {"name": "Example", "tier": "pro"},
    "product": "cloud",
    "metadata": {"source": "demo"},
}


async def test_api_full_happy_path(client: httpx.AsyncClient) -> None:
    assert (await client.get("/health")).status_code == 200
    response = await client.post("/api/v1/tickets", json=PAYLOAD)
    assert response.status_code == 201
    assert response.headers["X-Request-ID"]
    ticket_id = response.json()["id"]
    assert response.json()["metadata"] == {"source": "demo"}
    for backend in ("sklearn", "torch"):
        prediction = await client.post(f"/api/v1/tickets/{ticket_id}/classify/{backend}")
        assert prediction.status_code == 200, prediction.text
        assert prediction.json()["category"] == "authentication"
    analyzed = await client.post(f"/api/v1/tickets/{ticket_id}/analyze", json={})
    assert analyzed.status_code == 200, analyzed.text
    assert analyzed.json()["used_fallback"] is False
    latest = (await client.get(f"/api/v1/tickets/{ticket_id}")).json()
    assert len(latest["predictions"]) == 3
    resolved = await client.post(
        f"/api/v1/tickets/{ticket_id}/resolve",
        json={
            "resolution": "Verified token rotation with the customer.",
            "expected_version": latest["version"],
        },
    )
    assert resolved.status_code == 200
    assert resolved.json()["status"] == "resolved"
    conflict = await client.post(
        f"/api/v1/tickets/{ticket_id}/resolve",
        json={"resolution": "Another resolution", "expected_version": latest["version"]},
    )
    assert conflict.status_code == 409
    filtered = await client.get(
        "/api/v1/tickets", params={"product": "cloud", "status": "resolved"}
    )
    assert filtered.json()["total"] == 1


async def test_validation_missing_resource_and_pagination(client: httpx.AsyncClient) -> None:
    assert (await client.post("/api/v1/tickets", json={})).status_code == 422
    assert (await client.get("/api/v1/tickets/999999")).status_code == 404
    assert (await client.get("/api/v1/tickets", params={"limit": 1000})).status_code == 422
    created = (await client.post("/api/v1/tickets", json=PAYLOAD)).json()
    assert (
        await client.patch(f"/api/v1/tickets/{created['id']}", json={"priority": "high"})
    ).status_code == 200
    assert (
        await client.patch(f"/api/v1/tickets/{created['id']}", json={"priority": None})
    ).status_code == 422


async def test_repository_identity_map_and_rollback(
    runtime: Runtime, ticket: TicketResponse
) -> None:
    with pytest.raises(RuntimeError, match="abort"):
        async with runtime.database.transaction() as session:
            repository = TicketRepository(session)
            first = await repository.get(ticket.id)
            second = await repository.get(ticket.id)
            assert first is second
            first.title = "Must be rolled back"
            await session.flush()
            raise RuntimeError("abort")
    assert (await runtime.tickets.get(ticket.id)).title == ticket.title
    async with runtime.database.transaction() as session:
        assert await session.scalar(select(func.count()).select_from(Ticket)) == 1


async def test_fastapi_dependency_override(client: httpx.AsyncClient) -> None:
    from ai_support_lab.tickets.exceptions import ModelUnavailable

    service = AsyncMock()
    service.classify.side_effect = ModelUnavailable("Train first")
    app = client._transport.app
    app.dependency_overrides[get_ticket_service] = lambda: service
    try:
        response = await client.post("/api/v1/tickets/1/classify/sklearn")
        assert response.status_code == 503
        assert response.json()["detail"] == "Train first"
    finally:
        app.dependency_overrides.clear()


async def test_database_connection_failure_has_explicit_domain_error() -> None:
    from ai_support_lab.database.session import Database
    from ai_support_lab.tickets.exceptions import DatabaseUnavailable

    database = Database("sqlite+aiosqlite:////missing-lab-directory/database.db")
    try:
        with pytest.raises(DatabaseUnavailable):
            async with database.transaction() as session:
                await session.scalar(select(1))
    finally:
        await database.close()
