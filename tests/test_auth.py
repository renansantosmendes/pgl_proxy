import pytest
from fastapi import HTTPException

from app.main import require_active_matricula


@pytest.mark.parametrize("authorization", [None, "", "Token 123", "Bearer", "Bearer   "])
async def test_missing_or_malformed_header_raises_401(authorization):
    with pytest.raises(HTTPException) as exc_info:
        await require_active_matricula(authorization=authorization)

    assert exc_info.value.status_code == 401


async def test_garbage_token_raises_401():
    with pytest.raises(HTTPException) as exc_info:
        await require_active_matricula(authorization="Bearer not-a-real-jwt")

    assert exc_info.value.status_code == 401


async def test_expired_token_raises_401():
    from conftest import make_token

    expired = make_token(expires_in=-3600)

    with pytest.raises(HTTPException) as exc_info:
        await require_active_matricula(authorization=f"Bearer {expired}")

    assert exc_info.value.status_code == 401


async def test_token_signed_with_wrong_secret_raises_401():
    import jwt

    bad_token = jwt.encode(
        {"sub": "20230001", "exp": 9999999999}, "wrong-secret", algorithm="HS256"
    )

    with pytest.raises(HTTPException) as exc_info:
        await require_active_matricula(authorization=f"Bearer {bad_token}")

    assert exc_info.value.status_code == 401


async def test_inactive_matricula_raises_403(mock_is_enrollment_active):
    from conftest import make_token

    with pytest.raises(HTTPException) as exc_info:
        await require_active_matricula(authorization=f"Bearer {make_token('00000000')}")

    assert exc_info.value.status_code == 403


async def test_active_matricula_is_returned(mock_is_enrollment_active):
    from conftest import ACTIVE_MATRICULA, make_token

    result = await require_active_matricula(authorization=f"Bearer {make_token()}")

    assert result == ACTIVE_MATRICULA
