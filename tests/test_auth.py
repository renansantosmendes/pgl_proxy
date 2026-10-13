import pytest
from fastapi import HTTPException

from app.main import require_active_registration_number


@pytest.mark.parametrize("authorization", [None, "", "Token 123", "Bearer", "Bearer   "])
async def test_missing_or_malformed_header_raises_401(authorization):
    with pytest.raises(HTTPException) as exc_info:
        await require_active_registration_number(authorization=authorization)

    assert exc_info.value.status_code == 401


async def test_garbage_token_raises_401():
    with pytest.raises(HTTPException) as exc_info:
        await require_active_registration_number(authorization="Bearer not-a-real-jwt")

    assert exc_info.value.status_code == 401


async def test_expired_token_raises_401():
    from conftest import make_token

    expired = make_token(expires_in=-3600)

    with pytest.raises(HTTPException) as exc_info:
        await require_active_registration_number(authorization=f"Bearer {expired}")

    assert exc_info.value.status_code == 401


async def test_token_signed_with_wrong_secret_raises_401():
    import jwt

    bad_token = jwt.encode(
        {"sub": "20230001", "exp": 9999999999}, "wrong-secret", algorithm="HS256"
    )

    with pytest.raises(HTTPException) as exc_info:
        await require_active_registration_number(authorization=f"Bearer {bad_token}")

    assert exc_info.value.status_code == 401


async def test_inactive_registration_number_raises_403(mock_is_enrollment_active):
    from conftest import make_token

    with pytest.raises(HTTPException) as exc_info:
        await require_active_registration_number(authorization=f"Bearer {make_token('00000000')}")

    assert exc_info.value.status_code == 403


async def test_active_registration_number_is_returned(mock_is_enrollment_active):
    from conftest import ACTIVE_REGISTRATION_NUMBER, make_token

    result = await require_active_registration_number(authorization=f"Bearer {make_token()}")

    assert result == ACTIVE_REGISTRATION_NUMBER


async def test_x_api_key_header_is_accepted(mock_is_enrollment_active):
    from app.main import require_active_registration_number_from_api_key
    from conftest import ACTIVE_REGISTRATION_NUMBER, make_token

    result = await require_active_registration_number_from_api_key(
        authorization=None, x_api_key=make_token()
    )

    assert result == ACTIVE_REGISTRATION_NUMBER


async def test_x_api_key_falls_back_to_bearer(mock_is_enrollment_active):
    from app.main import require_active_registration_number_from_api_key
    from conftest import ACTIVE_REGISTRATION_NUMBER, make_token

    result = await require_active_registration_number_from_api_key(
        authorization=f"Bearer {make_token()}", x_api_key=None
    )

    assert result == ACTIVE_REGISTRATION_NUMBER


@pytest.mark.parametrize("x_api_key", ["not-a-real-jwt", None, "   "])
async def test_invalid_or_missing_x_api_key_raises_401(x_api_key):
    from app.main import require_active_registration_number_from_api_key

    with pytest.raises(HTTPException) as exc_info:
        await require_active_registration_number_from_api_key(
            authorization=None, x_api_key=x_api_key
        )

    assert exc_info.value.status_code == 401
