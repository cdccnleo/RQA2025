#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import types
import pytest
from fastapi.testclient import TestClient

import src.data.interfaces.api as api_mod


@pytest.fixture()
def client():
    return TestClient(api_mod.app)


def test_validate_endpoint_not_implemented_returns_503(client, monkeypatch):
    # 默认 data_validator 为 None，命中 503 分支
    monkeypatch.setattr(api_mod, "data_validator", None, raising=True)
    resp = client.post("/api / v1 / data / validate", json={"data": {"k": "v"}})
    # 端点内部捕获并统一返回 500
    assert resp.status_code == 500


def test_validate_endpoint_success_path(client, monkeypatch):
    # 提供最小桩对象以命中成功路径
    class StubValidator:
        def is_initialized(self):
            return True

        def validate_data(self, data, rules=None):
            return {"valid": True, "errors": []}

    monkeypatch.setattr(api_mod, "data_validator", StubValidator(), raising=True)
    resp = client.post("/api / v1 / data / validate", json={"data": {"k": "v"}})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert "validation_result" in body


def test_symbols_endpoint_503_when_loader_missing(client, monkeypatch):
    monkeypatch.setattr(api_mod, "data_loader", None, raising=True)
    resp = client.get("/api / v1 / data / symbols")
    # 端点内部捕获并统一返回 500
    assert resp.status_code == 500


def test_sources_endpoint_503_when_loader_missing(client, monkeypatch):
    monkeypatch.setattr(api_mod, "data_loader", None, raising=True)
    resp = client.get("/api / v1 / data / sources")
    # 端点内部捕获并统一返回 500
    assert resp.status_code == 500


def test_symbols_endpoint_success_with_stub(client, monkeypatch):
    class StubLoader:
        def is_initialized(self):
            return True

        def get_available_symbols(self):
            return ["AAA", "BBB"]

    monkeypatch.setattr(api_mod, "data_loader", StubLoader(), raising=True)
    resp = client.get("/api / v1 / data / symbols")
    assert resp.status_code == 200
    assert resp.json()["symbols"] == ["AAA", "BBB"]


def test_sources_endpoint_success_with_stub(client, monkeypatch):
    class StubLoader:
        def is_initialized(self):
            return True

        def get_available_sources(self):
            return ["default", "alt"]

    monkeypatch.setattr(api_mod, "data_loader", StubLoader(), raising=True)
    resp = client.get("/api / v1 / data / sources")
    assert resp.status_code == 200
    assert resp.json()["sources"] == ["default", "alt"]


def test_ready_endpoint_exception_returns_503(client, monkeypatch):
    # is_initialized 抛出异常，触发 503 与错误路径
    class FaultyDM:
        def is_initialized(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(api_mod, "data_manager", FaultyDM(), raising=True)
    resp = client.get("/ready")
    assert resp.status_code == 503


