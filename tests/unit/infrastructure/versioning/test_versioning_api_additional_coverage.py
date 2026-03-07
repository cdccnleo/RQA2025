#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
补充 Versioning 模块的 API/策略/代理 覆盖率测试
"""

from datetime import datetime, timedelta
from pathlib import Path
import sys
import types

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_INFRA_PATH = _PROJECT_ROOT / "src" / "infrastructure"

_ORIGINAL_INFRA = sys.modules.get("src.infrastructure")
_ORIGINAL_MONITORING = sys.modules.get("src.infrastructure.monitoring")

if "src.infrastructure" not in sys.modules:
    infra_stub = types.ModuleType("src.infrastructure")
    infra_stub.__path__ = [str(_INFRA_PATH)]
    sys.modules["src.infrastructure"] = infra_stub
    sys.modules["src.infrastructure.optimization"] = types.ModuleType(
        "src.infrastructure.optimization"
    )

if "src.infrastructure.monitoring" not in sys.modules:
    monitoring_stub = types.ModuleType("src.infrastructure.monitoring")
    monitoring_stub.__path__ = []
    sys.modules["src.infrastructure.monitoring"] = monitoring_stub


from src.infrastructure.api.configs.base_config import BaseConfig
from src.infrastructure.versioning.core.version import Version, VersionComparator
from src.infrastructure.versioning.data.data_version_manager import DataVersionManager
from src.infrastructure.versioning.manager.manager import VersionManager
from src.infrastructure.versioning.manager.policy import VersionPolicy
from src.infrastructure.versioning.proxy.proxy import VersionProxy
from src.infrastructure.versioning.config.config_version_manager import ConfigVersionManager
from src.infrastructure.versioning.api.version_api_refactored import VersionAPIService

if _ORIGINAL_INFRA is None:
    sys.modules.pop("src.infrastructure", None)
else:
    sys.modules["src.infrastructure"] = _ORIGINAL_INFRA

if _ORIGINAL_MONITORING is None:
    sys.modules.pop("src.infrastructure.monitoring", None)
else:
    sys.modules["src.infrastructure.monitoring"] = _ORIGINAL_MONITORING

if not hasattr(BaseConfig, "set_validation_mode"):
    @classmethod
    def _set_validation_mode(cls, mode: str) -> None:  # pragma: no cover - 简易补丁
        cls._validation_mode = mode

    BaseConfig.set_validation_mode = _set_validation_mode  # type: ignore[attr-defined]


@pytest.fixture()
def version_api_client(tmp_path):
    """构建 VersionAPI 的测试客户端并隔离存储目录"""
    import importlib

    version_api_module = importlib.import_module("src.infrastructure.versioning.api.version_api")
    api = version_api_module.VersionAPI()
    api.config_version_manager = ConfigVersionManager(config_dir=tmp_path / "configs_api")
    api.data_version_manager = DataVersionManager(base_path=tmp_path / "data_api")
    return api, api.app.test_client(), version_api_module


def test_version_api_endpoints(version_api_client):
    """覆盖 VersionAPI 的主要路由"""
    api, client, _ = version_api_client

    # 创建版本
    resp = client.post("/api/v1/versions/service", json={"version": "1.0.0"})
    assert resp.status_code == 201

    # 获取版本
    resp = client.get("/api/v1/versions/service")
    assert resp.status_code == 200
    assert resp.json["version"] == "1.0.0"

    # 更新版本
    resp = client.put("/api/v1/versions/service", json={"version": "1.1.0"})
    assert resp.status_code == 200
    assert resp.json["status"] == "updated"

    # 列出版本
    resp = client.get("/api/v1/versions")
    assert resp.status_code == 200
    assert resp.json["count"] >= 1

    # 创建版本缺少字段
    resp = client.post("/api/v1/versions/service", json={})
    assert resp.status_code == 400

    # 删除不存在的版本
    resp = client.delete("/api/v1/versions/missing")
    assert resp.status_code == 404

    # 列出策略
    resp = client.get("/api/v1/policies")
    assert resp.status_code == 200
    assert "policies" in resp.json

    # 验证版本
    resp = client.post("/api/v1/validate", json={"version": "1.1.0", "policy": "stable"})
    assert resp.status_code == 200

    # 比较版本
    resp = client.post("/api/v1/compare", json={"version1": "1.0.0", "version2": "1.1.0"})
    assert resp.status_code == 200
    assert resp.json["comparison"] in {"<", ">", "="}

    # 准备数据/配置版本并列出
    api.data_version_manager.save_version("dataset", {"value": 1})
    api.config_version_manager.create_version("cfg", {"key": "value"})

    resp = client.get("/api/v1/data/versions")
    assert resp.status_code == 200

    resp = client.get("/api/v1/config/versions/cfg")
    assert resp.status_code == 200

    # 健康检查
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json["status"] == "healthy"


def test_version_policy_extended_behaviour():
    """验证 VersionPolicy 新增便捷方法"""
    policy = VersionPolicy()

    assert policy.get_next_version("1.0.0", increment_type="minor") == "1.1.0"
    assert policy.get_next_version(Version("1.0.0"), bump_type="major") == Version("2.0.0")
    assert policy.allows_upgrade("1.0.0", "1.0.1")
    assert policy.allows_downgrade("2.0.0", "1.0.0")
    assert policy.is_compatible("1.2.0", "1.3.5")

    with pytest.raises(ValueError):
        policy.get_next_version("1.0.0", increment_type="unsupported")


def test_data_version_manager_cleanup_and_history(tmp_path):
    """验证数据版本管理器的元数据和清理逻辑"""
    manager = DataVersionManager(base_path=tmp_path / "data_store")

    v1 = manager.save_version("dataset", {"value": 1}, metadata={"owner": "qa"}, tags=["stable"])
    v2 = manager.save_version("dataset", {"value": 2})

    history = manager.get_version_history("dataset")
    assert len(history) == 2
    assert history[0].metadata["owner"] == "qa"
    assert "stable" in history[0].tags

    info = manager.get_version_info("dataset", v1)
    assert info is not None and info.version == v1

    # 制造过期版本并触发清理
    for entry in history:
        entry.timestamp = datetime.now() - timedelta(days=60)

    removed = manager.cleanup_old_versions(days=30)
    assert removed >= 1
    assert manager.list_versions("dataset") == []


def test_version_proxy_execute_and_cache_behaviour():
    """验证 VersionProxy 的执行与缓存逻辑"""
    proxy = VersionProxy(VersionManager())

    proxy.register_version("service", "1.0.0")
    assert str(proxy.get_version("service")) == "1.0.0"

    proxy.update_version("service", "1.1.0")
    result = proxy.execute("get_version", name="service")
    assert str(result) == "1.1.0"

    proxy.set_version("0.9.0")
    current = proxy.get_version()
    assert current is None or isinstance(current, Version)

    with pytest.raises(ValueError):
        proxy.execute("unsupported_action")


def test_version_api_service_routes(tmp_path):
    """覆盖重构版 VersionAPIService 的路由"""
    service = VersionAPIService(
        version_manager=VersionManager(),
        policy_manager=VersionPolicy(),
        data_version_manager=DataVersionManager(base_path=tmp_path / "data_service"),
        config_version_manager=ConfigVersionManager(config_dir=tmp_path / "config_service"),
    )
    client = service.app.test_client()

    client.post("/api/v1/versions/app", json={"version": "1.0.0"})
    assert client.get("/api/v1/versions/app").status_code == 200
    assert client.put("/api/v1/versions/app", json={"version": "1.0.1"}).json["status"] == "updated"
    assert client.get("/api/v1/versions").json["count"] >= 1
    assert client.delete("/api/v1/versions/missing").status_code == 404

    assert client.get("/api/v1/policies").status_code == 200
    assert client.post("/api/v1/validate", json={"version": "1.0.1"}).status_code == 200
    assert client.post("/api/v1/compare", json={"version1": "1.0.0", "version2": "1.0.1"}).status_code == 200

    service.data_version_manager.save_version("dataset", {"payload": 1})
    service.config_version_manager.create_version("cfg", {"flag": True})
    assert client.get("/api/v1/data/versions").status_code == 200
    assert client.get("/api/v1/config/versions/cfg").status_code == 200
    assert client.get("/api/v1/health").json["status"] == "healthy"


def test_version_api_error_conditions(version_api_client, monkeypatch):
    """验证 VersionAPI 的错误处理路径"""
    api, client, version_api_module = version_api_client

    # 未注册的版本查询
    resp = client.get("/api/v1/versions/missing")
    assert resp.status_code == 404
    assert resp.json["error"] == "Version not found"

    # 创建版本 - 非法版本字符串
    resp = client.post("/api/v1/versions/service", json={"version": "not-a-version"})
    assert resp.status_code == 400
    assert "error" in resp.json

    # 更新版本 - 缺少 payload
    resp = client.put("/api/v1/versions/service", json={})
    assert resp.status_code == 400

    # 更新版本 - 后端更新失败
    monkeypatch.setattr(api.version_manager, "update_version", lambda name, version: False)
    resp = client.put("/api/v1/versions/service", json={"version": "1.2.0"})
    assert resp.status_code == 500

    # 验证版本 - 缺少字段
    resp = client.post("/api/v1/validate", json={"policy": "stable"})
    assert resp.status_code == 400

    # 验证版本 - 策略异常
    def _raise_policy(version, policy_name):
        raise ValueError("policy error")

    monkeypatch.setattr(api.policy_manager, "validate_version", _raise_policy)
    resp = client.post("/api/v1/validate", json={"version": "1.2.0", "policy": "stable"})
    assert resp.status_code == 400

    # 比较版本 - 参数不足
    resp = client.post("/api/v1/compare", json={"version1": "1.0.0"})
    assert resp.status_code == 400

    # 比较版本 - 非法版本字符串
    resp = client.post("/api/v1/compare", json={"version1": "invalid", "version2": "1.0.0"})
    assert resp.status_code == 400

    # 数据版本列表 - 后端异常
    def _fail_list_versions():
        raise RuntimeError("data store offline")

    monkeypatch.setattr(api.data_version_manager, "list_versions", _fail_list_versions)
    resp = client.get("/api/v1/data/versions")
    assert resp.status_code == 500
    assert "error" in resp.json

    # 配置版本列表 - 后端异常
    def _fail_list_config_versions(_name):
        raise RuntimeError("config store offline")

    monkeypatch.setattr(api.config_version_manager, "list_versions", _fail_list_config_versions)
    resp = client.get("/api/v1/config/versions/sample")
    assert resp.status_code == 500
    assert "error" in resp.json

    # 启动 API（模拟 run 路径）
    run_called = {"flag": False}

    def _fake_run(*args, **kwargs):
        run_called["flag"] = True

    monkeypatch.setattr(api.app, "run", _fake_run)
    api.run(host="127.0.0.1", port=0, debug=False)
    assert run_called["flag"]

    # CLI 路径：通过 runpy 执行 __main__ 分支
    import runpy
    from argparse import Namespace

    cli_called = {"flag": False}

    class _DummyAPI:
        def run(self, *args, **kwargs):
            cli_called["flag"] = True

    monkeypatch.setattr(
        version_api_module.argparse.ArgumentParser,
        "parse_args",
        lambda self: Namespace(host="0.0.0.0", port=4321, debug=False),
    )
    monkeypatch.setattr(version_api_module, "create_version_api", lambda app=None: _DummyAPI())
    runpy.run_module(version_api_module.__name__, run_name="__main__")
    assert cli_called["flag"]


def test_version_comparator_and_range(monkeypatch):
    """覆盖 Version 与 VersionComparator 的边界逻辑"""
    v = Version("1.2.3-alpha")
    assert v.is_prerelease()
    assert not v.is_stable()
    assert Version.is_valid_version_string("2.0.0")
    assert not Version.is_valid_version_string("bad.version")

    assert VersionComparator.is_less_or_equal("1.2.3", "1.2.3")
    assert VersionComparator.is_greater_than("1.3.0", "1.2.9")
    assert VersionComparator.is_version_in_range("1.5.0", "^1.0.0")
    assert VersionComparator.is_version_in_range("0.2.5", "~0.2.0")
    assert not VersionComparator.is_version_in_range("2.0.0", "<1.5.0")

    versions = ["1.0.0", "2.0.0", "1.5.0"]
    assert str(VersionComparator.find_latest_version(versions)) == "2.0.0"
    sorted_versions = VersionComparator.sort_versions(versions, reverse=False)
    assert [str(item) for item in sorted_versions][0] == "1.0.0"

    import src.infrastructure.versioning.core.version as version_module

    monkeypatch.setattr(version_module, "_should_preserve_original_increment", lambda: True)
    original = Version("3.2.1")
    new_version = original.increment_patch()
    assert str(original) == "3.2.1"
    assert str(new_version) == "3.2.2"


def test_version_policy_management():
    """覆盖 VersionPolicy 管理能力"""
    policy = VersionPolicy()
    policy.add_policy("beta-only", lambda v: v.is_prerelease())
    assert "beta-only" in policy.list_policies()

    versions = ["1.0.0", "1.1.0-beta", "2.0.0"]
    compliant = policy.find_compliant_versions(versions, "beta-only")
    assert any(v.is_prerelease() for v in compliant)

    results = policy.validate_version_range(versions, "stable")
    assert isinstance(results, list)

    stats = policy.get_policy_stats()
    assert stats["total_policies"] >= stats["predefined_policies"]

    desc = policy.get_policy_description("stable")
    assert desc is not None

    policy.clear_policies()
    assert "beta-only" not in policy.list_policies()

