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
async def test_tie_breaker_registration_order_with_equal_scores():
    mgr = IntelligentSourceManager()
    try:
        # Two sources with identical priority and no health records -> equal scores
        cfg1 = DataSourceConfig(name="s1", source_type=DataSourceType.STOCK, priority=5, enabled=True)
        cfg2 = DataSourceConfig(name="s2", source_type=DataSourceType.STOCK, priority=5, enabled=True)

        loader = AsyncMock()
        loader.load_data = AsyncMock(return_value=None)

        # Register in order s1 then s2
        mgr.register_source("s1", cfg1, loader)
        mgr.register_source("s2", cfg2, loader)

        # With equal scores, Python sort is stable -> preserves insertion order
        best = mgr.get_best_source("stock")
        assert best == "s1"

        # Swap registration order to verify stability
        mgr.unregister_source("s1")
        mgr.unregister_source("s2")
        mgr.register_source("s2", cfg2, loader)
        mgr.register_source("s1", cfg1, loader)
        best2 = mgr.get_best_source("stock")
        assert best2 == "s2"
    finally:
        mgr.cleanup()


@pytest.mark.asyncio
async def test_health_fluctuation_switch_and_recover():
    mgr = IntelligentSourceManager()
    try:
        cfg_fast = DataSourceConfig(name="fast", source_type=DataSourceType.STOCK, priority=5, enabled=True)
        cfg_slow = DataSourceConfig(name="slow", source_type=DataSourceType.STOCK, priority=5, enabled=True)

        loader = AsyncMock()
        loader.load_data = AsyncMock(return_value=None)

        mgr.register_source("fast", cfg_fast, loader)
        mgr.register_source("slow", cfg_slow, loader)

        # Initially equal; stable order -> fast first
        assert mgr.get_best_source("stock") == "fast"

        # Degrade 'fast' via health monitor (simulate failures and high response time)
        for _ in range(5):
            mgr.health_monitor.record_request("fast", response_time_ms=4000, success=False)
        mgr._update_source_ranking()
        assert mgr.get_best_source("stock") == "slow"

        # Recover 'fast' with successful fast responses
        for _ in range(10):
            mgr.health_monitor.record_request("fast", response_time_ms=200, success=True)
        mgr._update_source_ranking()
        assert mgr.get_best_source("stock") == "fast"
    finally:
        mgr.cleanup()


