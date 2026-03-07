from typing import Any, Dict, List

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.data import api as api_module


@pytest.fixture(name="client")
def client_fixture():
    return TestClient(api_module.app)


def test_health_endpoint_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_market_endpoint_returns_symbol_and_data(client: TestClient) -> None:
    response = client.get("/api/v1/data/market/BTC/USDT")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"] == []
    assert body["symbol"] == "BTC/USDT"


def test_get_market_data_rejects_empty_symbol() -> None:
    with pytest.raises(HTTPException) as exc:
        api_module.get_market_data("", "USDT")
    assert exc.value.status_code == 400
    assert exc.value.detail == "invalid symbol"


def test_validate_data_uses_manager_when_available(client: TestClient, monkeypatch):
    class StubManager:
        def validate_data(self, data: List[Dict[str, Any]], data_type: str = "") -> Dict[str, Any]:
            return {
                "is_valid": False,
                "total_records": len(data),
                "valid_records": 0,
                "invalid_records": len(data),
                "errors": ["invalid"],
                "data_type": data_type,
            }

    monkeypatch.setattr(api_module, "DataManagerSingleton", StubManager)
    request_payload = {"data": [{"value": 1}], "data_type": "market"}
    response = client.post("/api/v1/data/validate", json=request_payload)
    assert response.status_code == 200
    result = response.json()["validation_result"]
    assert result["is_valid"] is False
    assert result["data_type"] == "market"
    assert result["invalid_records"] == 1


def test_validate_data_fallback_when_manager_missing_method(client: TestClient, monkeypatch):
    class ManagerWithoutValidate:
        pass

    monkeypatch.setattr(api_module, "DataManagerSingleton", ManagerWithoutValidate)
    response = client.post("/api/v1/data/validate", json={"data": [1, 2]})
    assert response.status_code == 200
    result = response.json()["validation_result"]
    assert result["is_valid"] is True
    assert result["total_records"] == 2
    assert result["errors"] == []

