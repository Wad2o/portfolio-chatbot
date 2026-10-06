from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
@router.get("/api/v1/health")
async def health() -> dict:
    # Deliberately minimal: no versions, hosts, provider or config details.
    return {"status": "ok"}
