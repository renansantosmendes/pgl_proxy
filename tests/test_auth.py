import pytest
from fastapi import HTTPException

from app.main import require_active_matricula


@pytest.mark.parametrize(
    "authorization",
    [None, "", "Token 123", "Bearer", "Bearer nocolon", "Bearer :nomatricula", "Bearer matricula:"],
)
async def test_missing_or_malformed_header_raises_401(authorization):
    with pytest.raises(HTTPException) as exc_info:
        await require_active_matricula(authorization=authorization)

    assert exc_info.value.status_code == 401


async def test_invalid_credentials_raises_403(mock_verify_student_credentials):
    with pytest.raises(HTTPException) as exc_info:
        await require_active_matricula(authorization="Bearer 00000000:wrong-senha")

    assert exc_info.value.status_code == 403


async def test_valid_credentials_returns_matricula(mock_verify_student_credentials):
    from conftest import ACTIVE_MATRICULA, ACTIVE_SENHA

    result = await require_active_matricula(
        authorization=f"Bearer {ACTIVE_MATRICULA}:{ACTIVE_SENHA}"
    )

    assert result == ACTIVE_MATRICULA
