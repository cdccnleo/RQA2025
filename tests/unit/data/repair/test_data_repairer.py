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


import pandas as pd
import pytest

from src.data.repair import DataRepairer, RepairConfig, RepairResult, RepairStrategy


class DummyDataModel:

    def __init__(self, data, frequency="1d", metadata=None):

        self.data = data
        self._frequency = frequency
        self._metadata = dict(metadata or {})

    def get_frequency(self):

        return self._frequency

    def get_metadata(self, user_only=False):

        if user_only:
            return dict(self._metadata)
        return dict(self._metadata)


def test_repair_data_applies_strategies_and_tracks_history():

    raw = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-02", "2025-01-02", "2025-01-03"],
            "value": [1.0, None, None, 120.0],
            "category": [1, 2, 2, 2],
        }
    )

    config = RepairConfig(
        null_strategy=RepairStrategy.FILL_FORWARD,
        null_threshold=0.6,
        outlier_strategy=RepairStrategy.REMOVE_OUTLIERS,
        duplicate_strategy=RepairStrategy.DROP,
        min_value=0.5,
        max_value=10.0,
        time_series_enabled=True,
        resample_freq="D",
    )

    repairer = DataRepairer(config)
    repaired, result = repairer.repair_data(raw)

    assert result.success is True
    assert repaired["value"].isnull().sum() == 0
    assert repaired["value"].max() <= 10.0
    assert result.repair_stats["null_repairs"] > 0
    assert result.repair_stats["duplicate_drops"] == 1
    assert result.repair_stats.get("ts_repairs", 0) > 0

    history = repairer.get_repair_history()
    assert len(history) == 1
    stats = repairer.get_repair_stats()
    assert stats["total_repairs"] == 1
    assert pytest.approx(stats["success_rate"]) == 1.0

    repairer.reset_history()
    assert repairer.get_repair_history() == []


def test_repair_data_handles_empty_dataframe():

    repairer = DataRepairer(RepairConfig())
    repaired, result = repairer.repair_data(pd.DataFrame())

    assert repaired.empty
    assert isinstance(result, RepairResult)
    assert result.success is False
    assert "数据为空" in result.errors[0]


def test_repair_data_model_adds_metadata_and_updates_shape():

    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "value": [1.0, None],
        }
    )
    model = DummyDataModel(data, metadata={"source": "unit-test"})

    repairer = DataRepairer(
        RepairConfig(
            null_strategy=RepairStrategy.FILL_FORWARD,
            duplicate_strategy=RepairStrategy.DROP,
            time_series_enabled=False,
        )
    )

    repaired_model, result = repairer.repair_data_model(model)

    assert result.success is True
    assert repaired_model.data["value"].isnull().sum() == 0
    metadata = repaired_model.get_metadata()
    assert "repair_info" in metadata
    assert metadata["repair_info"]["original_shape"] == result.original_shape

