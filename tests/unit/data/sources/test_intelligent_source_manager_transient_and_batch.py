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


import asyncio
from unittest.mock import AsyncMock

import pytest

from src.data.sources.intelligent_source_manager import (
    IntelligentSourceManager,
    DataSourceConfig,
    DataSourceType,
)


@pytest.mark.asyncio
async def test_enable_disable_rapid_switch_maintains_ranking_consistency():
    """测试启用/禁用快速切换时排名的一致性"""
    mgr = IntelligentSourceManager()
    try:
        cfg1 = DataSourceConfig(name="s1", source_type=DataSourceType.STOCK, priority=1, enabled=True)
        cfg2 = DataSourceConfig(name="s2", source_type=DataSourceType.STOCK, priority=2, enabled=True)
        cfg3 = DataSourceConfig(name="s3", source_type=DataSourceType.STOCK, priority=3, enabled=True)

        loader = AsyncMock()
        loader.load_data = AsyncMock(return_value=None)

        mgr.register_source("s1", cfg1, loader)
        mgr.register_source("s2", cfg2, loader)
        mgr.register_source("s3", cfg3, loader)

        # Initial ranking: s1 (priority 1) > s2 (priority 2) > s3 (priority 3)
        assert mgr.get_best_source("stock") == "s1"

        # Disable s1, should switch to s2
        mgr.disable_source("s1")
        assert mgr.get_best_source("stock") == "s2"

        # Re-enable s1, should switch back
        mgr.enable_source("s1")
        assert mgr.get_best_source("stock") == "s1"

        # Rapid disable/enable cycle
        mgr.disable_source("s1")
        mgr.disable_source("s2")
        assert mgr.get_best_source("stock") == "s3"

        mgr.enable_source("s2")
        assert mgr.get_best_source("stock") == "s2"

        mgr.enable_source("s1")
        assert mgr.get_best_source("stock") == "s1"

        # Verify ranking list is updated correctly
        info = mgr.get_source_info()
        assert "s1" in info["ranking"]
        assert "s2" in info["ranking"]
        assert "s3" in info["ranking"]
    finally:
        mgr.cleanup()


@pytest.mark.asyncio
async def test_batch_register_without_health_records_uses_default_score():
    """测试批量注册后健康报告缺失时使用默认得分"""
    mgr = IntelligentSourceManager()
    try:
        # Register multiple sources without any health records
        sources = []
        loader = AsyncMock()
        loader.load_data = AsyncMock(return_value=None)

        for i in range(5):
            name = f"source_{i}"
            cfg = DataSourceConfig(
                name=name,
                source_type=DataSourceType.STOCK,
                priority=i + 1,  # Different priorities
                enabled=True
            )
            mgr.register_source(name, cfg, loader)
            sources.append(name)

        # All sources have no health records, so they get default score (100 + (10-priority)*10 + 30)
        # Priority 1: 100 + 90 + 30 = 220
        # Priority 2: 100 + 80 + 30 = 210
        # Priority 3: 100 + 70 + 30 = 200
        # Priority 4: 100 + 60 + 30 = 190
        # Priority 5: 100 + 50 + 30 = 180
        # So best should be source_0 (priority 1)
        best = mgr.get_best_source("stock")
        assert best == "source_0"

        # Verify health report is empty for new sources
        health_report = mgr.health_monitor.get_health_report()
        assert health_report.get("total_sources", 0) == 0  # No health records yet

        # Verify source info includes all registered sources
        info = mgr.get_source_info()
        assert len(info["sources"]) == 5
        for name in sources:
            assert name in info["sources"]
            assert info["sources"][name]["enabled"] is True

        # After recording some requests, health should be tracked
        mgr.health_monitor.record_request("source_0", response_time_ms=100, success=True)
        health_report2 = mgr.health_monitor.get_health_report()
        assert health_report2.get("total_sources", 0) >= 1
    finally:
        mgr.cleanup()


@pytest.mark.asyncio
async def test_batch_register_mixed_enabled_disabled_affects_ranking():
    """测试批量注册时混合启用/禁用状态对排名的影响"""
    mgr = IntelligentSourceManager()
    try:
        loader = AsyncMock()
        loader.load_data = AsyncMock(return_value=None)

        # Register with mixed enabled states
        cfg1 = DataSourceConfig(name="enabled1", source_type=DataSourceType.STOCK, priority=1, enabled=True)
        cfg2 = DataSourceConfig(name="disabled1", source_type=DataSourceType.STOCK, priority=2, enabled=False)
        cfg3 = DataSourceConfig(name="enabled2", source_type=DataSourceType.STOCK, priority=3, enabled=True)
        cfg4 = DataSourceConfig(name="disabled2", source_type=DataSourceType.STOCK, priority=4, enabled=False)

        mgr.register_source("enabled1", cfg1, loader)
        mgr.register_source("disabled1", cfg2, loader)
        mgr.register_source("enabled2", cfg3, loader)
        mgr.register_source("disabled2", cfg4, loader)

        # Only enabled sources should be available
        best = mgr.get_best_source("stock")
        assert best == "enabled1"  # Priority 1, enabled

        # Verify disabled sources are not in available list
        available = mgr._get_available_sources("stock")
        assert "enabled1" in available
        assert "enabled2" in available
        assert "disabled1" not in available
        assert "disabled2" not in available

        # Enable a disabled source
        mgr.enable_source("disabled1")
        best2 = mgr.get_best_source("stock")
        assert best2 == "enabled1"  # Still best (priority 1 vs 2)

        # Disable enabled1, should switch to disabled1 (now enabled, priority 2)
        mgr.disable_source("enabled1")
        best3 = mgr.get_best_source("stock")
        assert best3 == "disabled1"  # Priority 2, now enabled
    finally:
        mgr.cleanup()

