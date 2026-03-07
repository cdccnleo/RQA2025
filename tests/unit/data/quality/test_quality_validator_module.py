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
import numpy as np
import pytest

from src.data.quality.validator import DataValidator


def test_validate_dataframe_collects_metrics_and_errors():
    validator = DataValidator()
    base = pd.DataFrame(
        {
            "price": [100.0, np.nan, 200000.0],
            "volume": ["1000", "invalid", "500"],
        }
    )
    df = pd.concat([base, base.iloc[[0]]], ignore_index=True)

    result = validator.validate(df, data_type="test_frame")

    assert result["is_valid"] is False
    errors = result["errors"]
    assert any("missing values detected" in err for err in errors)
    assert any("type mismatch detected" in err for err in errors)
    assert any("value out of expected range" in err for err in errors)
    assert any("duplicate rows detected" in err for err in errors)
    assert result["checks"]["row_count"] == df.shape[0]
    assert "completeness" in result["metrics"]


def test_validate_dict_payload_reports_missing_field():
    validator = DataValidator()
    payload = {"id": 1, "name": None}

    result = validator.validate(payload, data_type="dict_payload")

    assert result["is_valid"] is False
    assert any("missing value for field 'name'" in err for err in result["errors"])
    assert result["metrics"]["field_count"] == 2.0


def test_validate_with_unsupported_type():
    validator = DataValidator()
    result = validator.validate(["unexpected"], data_type="list_payload")
    assert result["is_valid"] is False
    assert any("unsupported data type" in err for err in result["errors"])


def test_custom_rules_and_history_tracking():
    validator = DataValidator()

    def custom_rule(data):
        return ["rule violation"] if isinstance(data, dict) else []

    validator.add_rule(custom_rule)
    result = validator.validate({"field": "value"}, data_type="custom")
    assert any("rule violation" in err for err in result["errors"])
    assert validator.validation_history
    assert validator.validation_history[-1].data_type == "custom"

    validator.clear_rules()
    assert validator.rules == []


def test_add_rule_requires_callable():
    validator = DataValidator()
    with pytest.raises(TypeError):
        validator.add_rule("not-callable")


def test_is_probably_numeric_column_sampling_logic():
    validator = DataValidator()
    series = pd.Series(["10", "20", "foo", "30"])
    assert validator._is_probably_numeric_column("custom", series) is True
    non_numeric = pd.Series(["foo", "bar", "baz"])
    assert validator._is_probably_numeric_column("custom", non_numeric) is False

