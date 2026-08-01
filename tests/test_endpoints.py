import pytest

from app import app, limiter


@pytest.fixture
def client():
    app.testing = True
    with app.test_client() as client:
        yield client


def test_predict_endpoint_returns_expected_structure(client):
    payload = {
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 0,
        "LoanAmount": 100,
        "Credit_History": 1,
        "Education": "Graduate",
    }

    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data, dict)
    # basic fields
    assert "prediction" in data
    assert "probability" in data or "probabilities" in data
    # Ensure the response contains input_data or model_info or ensemble summary
    assert any(k in data for k in ("input_data", "model_info", "ensemble", "results"))


def test_benchmark_endpoint_runs(client):
    resp = client.post("/models/benchmark", json={})
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data, dict)
    # should include results or consensus
    assert "results" in data or "consensus" in data


def test_health_endpoint_is_exempt_from_default_rate_limits(client):
    """Render's recurring health probes must never receive a 429 response."""
    limiter.reset()

    # The application-wide hourly limit is 200 requests. Without the exemption,
    # the final request is rejected with HTTP 429.
    for _ in range(201):
        response = client.get("/health")
        assert response.status_code == 200
