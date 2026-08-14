def test_register_success_returns_200(
    client, mock_register_password, mock_check_keyed_rate_limit
):
    response = client.post(
        "/v1/register",
        json={"matricula": "20230001", "senha": "a-strong-senha"},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    mock_register_password.assert_awaited_once_with("20230001", "a-strong-senha")


def test_register_unknown_matricula_returns_404(
    client, mock_register_password, mock_check_keyed_rate_limit
):
    mock_register_password.return_value = "not_found_or_inactive"

    response = client.post(
        "/v1/register",
        json={"matricula": "00000000", "senha": "a-strong-senha"},
    )

    assert response.status_code == 404


def test_register_already_registered_returns_409(
    client, mock_register_password, mock_check_keyed_rate_limit
):
    mock_register_password.return_value = "already_registered"

    response = client.post(
        "/v1/register",
        json={"matricula": "20230001", "senha": "a-strong-senha"},
    )

    assert response.status_code == 409


def test_register_short_senha_returns_422(
    client, mock_register_password, mock_check_keyed_rate_limit
):
    response = client.post(
        "/v1/register",
        json={"matricula": "20230001", "senha": "short"},
    )

    assert response.status_code == 422
    mock_register_password.assert_not_awaited()


def test_register_rate_limited_returns_429(client, mock_check_keyed_rate_limit):
    mock_check_keyed_rate_limit.return_value = False

    response = client.post(
        "/v1/register",
        json={"matricula": "20230001", "senha": "a-strong-senha"},
    )

    assert response.status_code == 429
