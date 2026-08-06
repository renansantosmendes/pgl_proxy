import pytest
from fastapi import HTTPException

from app.main import require_active_matricula


@pytest.mark.parametrize("authorization", [None, "", "Token 123", "Bearer"])
async def test_missing_or_malformed_header_raises_401(authorization):
    with pytest.raises(HTTPException) as exc_info:
        await require_active_matricula(authorization=authorization)

    assert exc_info.value.status_code == 401


async def test_inactive_matricula_raises_403(mock_is_enrollment_active):
    with pytest.raises(HTTPException) as exc_info:
        await require_active_matricula(authorization="Bearer 00000000")

    assert exc_info.value.status_code == 403


async def test_active_matricula_is_returned(mock_is_enrollment_active):
    from conftest import ACTIVE_MATRICULA

    result = await require_active_matricula(authorization=f"Bearer {ACTIVE_MATRICULA}")

    assert result == ACTIVE_MATRICULA
