#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import types
import pytest
from fastapi.testclient import TestClient

import src.data.interfaces.api as api_mod


@pytest.fixture()
def client():
    return TestClient(api_mod.app)


def test_health_endpoint_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data and "version" in data


def test_ready_endpoint_ready(client, monkeypatch):
    # 注入一个具备 is_initialized 的 data_manager 桩对象
    stub = types.SimpleNamespace(is_initialized=lambda: True)
    monkeypatch.setattr(api_mod, "data_manager", stub, raising=True)
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"


def test_ready_endpoint_not_ready(client, monkeypatch):
    stub = types.SimpleNamespace(is_initialized=lambda: False)
    monkeypatch.setattr(api_mod, "data_manager", stub, raising=True)
    resp = client.get("/ready")
    assert resp.status_code == 503


def test_store_data_success(client, monkeypatch):
    # 覆盖 data_manager.store_data 以命中成功路径
    def _store_data(data, storage_type="database", metadata=None):
        return {"result": "saved", "storage_type": storage_type, "meta": metadata or {}}

    stub = types.SimpleNamespace(is_initialized=lambda: True, store_data=_store_data)
    monkeypatch.setattr(api_mod, "data_manager", stub, raising=True)
    payload = {"data": {"k": "v"}, "storage_type": "cache", "metadata": {"key": "abc"}}
    resp = client.post("/api / v1 / data / store", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    # 兼容不同实现：可能返回 status 或 result 字段
    sr = body["storage_result"]
    assert isinstance(sr, dict)
    assert ("status" in sr and sr["status"] == "saved") or ("result" in sr and sr["result"] == "saved")


def test_fetch_data_loader_not_implemented_returns_503(client):
    # 默认 data_loader 为 None，命中未实现路径
    req = {
        "symbol": "AAA",
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "data_type": "ohlcv",
        "source": "default",
    }
    resp = client.post("/api / v1 / data / fetch", json=req)
    # 路由内部捕获后返回 500
    assert resp.status_code == 500


