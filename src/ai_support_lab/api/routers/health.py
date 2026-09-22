from fastapi import APIRouter
from sqlalchemy import text

from ai_support_lab.api.dependencies import ReadSessionDep

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(session: ReadSessionDep) -> dict[str, str]:
    await session.scalar(text("SELECT 1"))
    return {"status": "ok", "database": "ready"}
