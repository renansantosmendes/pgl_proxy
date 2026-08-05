import pytest
from fastapi import HTTPException

from app.main import ALLOWED_MODELS, validate_model


@pytest.mark.parametrize("model_name", sorted(ALLOWED_MODELS))
def test_allowed_models_pass_validation(model_name):
    validate_model(model_name)  # should not raise


@pytest.mark.parametrize("model_name", [None, ""])
def test_missing_model_raises_400(model_name):
    with pytest.raises(HTTPException) as exc_info:
        validate_model(model_name)

    assert exc_info.value.status_code == 400


def test_disallowed_model_raises_403():
    with pytest.raises(HTTPException) as exc_info:
        validate_model("gpt-4")

    assert exc_info.value.status_code == 403
    assert "gpt-4" in exc_info.value.detail
