import asyncio
import pandas as pd
from unittest.mock import Mock

# Mock数据管理器模块以绕过复杂的导入问题
mock_data_manager = Mock()
mock_data_manager.DataManager = Mock()
mock_data_manager.DataLoaderError = Exception

# 配置DataManager实例方法
mock_instance = Mock()
mock_instance.validate_all_configs.return_value = True
mock_instance.health_check.return_value = {"status": "healthy"}
mock_instance.store_data.return_value = True
mock_instance.has_data.return_value = True
mock_instance.get_metadata.return_value = {"data_type": "test", "symbol": "X"}
mock_instance.retrieve_data.return_value = pd.DataFrame({"col": [1, 2, 3]})
mock_instance.get_stats.return_value = {"total_items": 1}
mock_instance.validate_data.return_value = {"valid": True}
mock_instance.shutdown.return_value = None

mock_data_manager.DataManager.return_value = mock_instance

# Mock整个模块
import sys
sys.modules["src.data.data_manager"] = mock_data_manager


import types
import pytest
from datetime import datetime, timedelta

from src.data.ecosystem.data_ecosystem_manager import (
    DataEcosystemManager,
    EcosystemConfig,
)
from src.data.interfaces.standard_interfaces import DataSourceType


class _FakeHealthBridge:
    def __init__(self):
        self.registrations = []

    def register_data_health_check(self, name, func, data_type):
        self.registrations.append((name, func, data_type))


class _FakeIntegrationManager:
    def __init__(self):
        self._initialized = False
        self._integration_config = {
            "enable_data_catalog": True,
            "enable_data_marketplace": True,
            "catalog_update_interval": 5,
        }
        self._health_bridge = _FakeHealthBridge()

    def initialize(self):
        self._initialized = True

    def get_health_check_bridge(self):
        return self._health_bridge


def _install_fake_integration_manager(monkeypatch):
    fake_mgr = _FakeIntegrationManager()
    # 伪造模块与函数返回
    fake_module = types.SimpleNamespace(
        get_data_integration_manager=lambda: fake_mgr,
        log_data_operation=lambda *args, **kwargs: None,
        record_data_metric=lambda *args, **kwargs: None,
        publish_data_event=lambda *args, **kwargs: None,
    )
    monkeypatch.setitem(
        globals(),
        "__fake_infra_module__",  # 防止被优化掉
        fake_module,
    )
    monkeypatch.setattr(
        "src.data.ecosystem.data_ecosystem_manager.get_data_integration_manager",
        lambda: fake_mgr,
        raising=True,
    )
    monkeypatch.setattr(
        "src.data.ecosystem.data_ecosystem_manager.log_data_operation",
        lambda *args, **kwargs: None,
        raising=True,
    )
    monkeypatch.setattr(
        "src.data.ecosystem.data_ecosystem_manager.record_data_metric",
        lambda *args, **kwargs: None,
        raising=True,
    )
    monkeypatch.setattr(
        "src.data.ecosystem.data_ecosystem_manager.publish_data_event",
        lambda *args, **kwargs: None,
        raising=True,
    )
    return fake_mgr


def test_init_and_health_registration(monkeypatch):
    fake_mgr = _install_fake_integration_manager(monkeypatch)
    mgr = DataEcosystemManager(EcosystemConfig(enable_data_governance=False))
    assert fake_mgr._initialized is True
    # 健康检查注册在初始化过程中完成
    assert len(fake_mgr._health_bridge.registrations) >= 1
    names = [n for (n, _, _) in fake_mgr._health_bridge.registrations]
    assert "data_ecosystem" in names


def test_register_asset_and_update_quality(monkeypatch):
    _install_fake_integration_manager(monkeypatch)
    mgr = DataEcosystemManager(EcosystemConfig(enable_data_governance=False))
    asset_id = mgr.register_data_asset(
        name="test_asset",
        description="desc",
        data_type=DataSourceType.STOCK,
        owner="tester",
        tags=["t1", "t2"],
        metadata={"k": "v"},
    )
    assert asset_id in mgr.data_assets
    ok = mgr.update_data_quality(
        asset_id=asset_id,
        quality_score=0.9,
        quality_metrics={"p95_latency_ms": 12},
    )
    assert ok is True
    assert mgr.data_assets[asset_id].quality_score == 0.9


def test_track_lineage_and_get_lineage(monkeypatch):
    _install_fake_integration_manager(monkeypatch)
    mgr = DataEcosystemManager(EcosystemConfig(enable_data_governance=False))
    a = mgr.register_data_asset("A", "a", DataSourceType.STOCK, "owner")
    b = mgr.register_data_asset("B", "b", DataSourceType.STOCK, "owner")
    lineage_id = mgr.track_data_lineage(
        source_asset_id=a,
        target_asset_id=b,
        transformation_type="join",
        transformation_details={"on": "id"},
        execution_time=0.12,
        success=True,
    )
    assert lineage_id
    info = mgr.get_data_lineage(b, depth=2)
    assert info["asset_id"] == b
    assert info["total_lineages"] >= 0


def test_marketplace_filtering(monkeypatch):
    _install_fake_integration_manager(monkeypatch)
    mgr = DataEcosystemManager(EcosystemConfig(enable_data_governance=False))
    a = mgr.register_data_asset("A", "a", DataSourceType.STOCK, "owner", tags=["x"])
    # 未发布前列表应为空
    items = mgr.get_marketplace_items(category="financial_data")
    assert items == []
    # 发布商品但保持未发布状态（默认 False），仍为空
    _ = mgr.publish_to_marketplace(a, "t", "d", 10.0, "financial_data", ["x"])
    items = mgr.get_marketplace_items(category="financial_data")
    assert items == []


def test_cleanup_expired_lineage(monkeypatch):
    _install_fake_integration_manager(monkeypatch)
    cfg = EcosystemConfig(enable_data_governance=False, lineage_retention_days=1)
    mgr = DataEcosystemManager(cfg)
    a = mgr.register_data_asset("A", "a", DataSourceType.STOCK, "owner")
    b = mgr.register_data_asset("B", "b", DataSourceType.STOCK, "owner")
    _ = mgr.track_data_lineage(a, b, "copy", {}, 0.01, True)
    # 手动将历史时间回退，模拟过期
    for rec in mgr.data_lineage[b]:
        rec.created_at = datetime.now() - timedelta(days=10)
    # 执行清理（私有方法通过公共流程触发，这里直接调用确保路径覆盖）
    mgr._cleanup_expired_data()
    assert len(mgr.data_lineage[b]) == 0


