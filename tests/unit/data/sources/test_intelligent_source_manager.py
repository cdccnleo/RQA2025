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
from types import SimpleNamespace
import sys

import pandas as pd
import pytest


class _FakeDataModel:
    def __init__(self, data=None, frequency="1d", metadata=None):
        self.data = data or pd.DataFrame()
        self._frequency = frequency
        self._metadata = metadata or {}

    def get_frequency(self):
        return self._frequency

    def get_metadata(self):
        return self._metadata


class _FakeDataLoader:
    async def load_data(self, *args, **kwargs):
        raise NotImplementedError


try:
    import src.data.interfaces as data_interfaces
except ImportError:  # pragma: no cover
    data_interfaces = SimpleNamespace()
    sys.modules["src.data.interfaces"] = data_interfaces

if not hasattr(data_interfaces, "IDataModel"):
    data_interfaces.IDataModel = _FakeDataModel
if not hasattr(data_interfaces, "IDataLoader"):
    data_interfaces.IDataLoader = _FakeDataLoader

from src.data.sources.intelligent_source_manager import (  # noqa E402
    DataSourceConfig,
    DataSourceHealthMonitor,
    DataSourceStatus,
    DataSourceType,
    IntelligentSourceManager,
)


class DummyLoader(_FakeDataLoader):
    def __init__(self, succeed=True, delay=0):
        self.succeed = succeed
        self.delay = delay

    async def load_data(self, *args, **kwargs):
        await asyncio.sleep(self.delay)
        if not self.succeed:
            raise RuntimeError("loader failure")
        return pd.DataFrame({"value": [1]})


@pytest.mark.asyncio
async def test_load_data_returns_from_best_source(monkeypatch):
    manager = IntelligentSourceManager()
    manager.health_monitor.stop_monitoring()

    config = DataSourceConfig(name="alpha", source_type=DataSourceType.STOCK)
    manager.register_source("alpha", config, DummyLoader(succeed=True))

    result = await manager.load_data(
        data_type="stock", start_date="2025-01-01", end_date="2025-01-02"
    )

    assert isinstance(result, pd.DataFrame)
    assert list(result["value"]) == [1]
    manager.cleanup()


@pytest.mark.asyncio
async def test_load_data_fails_when_all_sources_fail(monkeypatch):
    manager = IntelligentSourceManager()
    manager.health_monitor.stop_monitoring()

    config = DataSourceConfig(name="fail_src", source_type=DataSourceType.STOCK)
    manager.register_source("fail_src", config, DummyLoader(succeed=False))

    with pytest.raises(Exception) as excinfo:
        await manager.load_data(
            data_type="stock", start_date="2025-01-01", end_date="2025-01-02"
        )
    assert "加载失败" in str(excinfo.value)
    manager.cleanup()


def test_health_monitor_records_and_updates_status():
    monitor = DataSourceHealthMonitor()
    monitor.record_request("src1", response_time_ms=500, success=True)
    monitor.record_request("src1", response_time_ms=1500, success=False)
    report = monitor.get_health_report()
    assert report["total_sources"] == 1
    assert "src1" in report["sources"]
    assert report["sources"]["src1"]["status"] in DataSourceStatus._value2member_map_


def test_update_source_config_changes_ranking():
    manager = IntelligentSourceManager()
    manager.health_monitor.stop_monitoring()
    cfg_fast = DataSourceConfig(name="fast", source_type=DataSourceType.STOCK, priority=1)
    cfg_slow = DataSourceConfig(name="slow", source_type=DataSourceType.STOCK, priority=5)
    manager.register_source("fast", cfg_fast, DummyLoader())
    manager.register_source("slow", cfg_slow, DummyLoader())

    ranking_before = manager.source_ranking.copy()
    manager.update_source_config("fast", priority=9)
    ranking_after = manager.source_ranking

    assert ranking_before != ranking_after
    manager.cleanup()

