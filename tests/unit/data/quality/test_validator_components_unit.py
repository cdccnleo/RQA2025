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


import pytest
from src.data.quality.validator_components import (
    ValidatorComponent,
    ValidatorComponentFactory,
)


def test_quality_validator_component_info_status_process():
    comp = ValidatorComponent(validator_id=1, component_type="Validator")
    info = comp.get_info()
    assert info["validator_id"] == 1
    assert info["component_type"] == "Validator"
    status = comp.get_status()
    assert status["status"] == "active"
    res = comp.process({"k": 1})
    assert res["status"] == "success" and res["validator_id"] == 1


def test_quality_validator_factory_supported_and_invalid():
    ids = ValidatorComponentFactory.get_available_validators()
    assert all(isinstance(i, int) for i in ids)
    cmap = ValidatorComponentFactory.create_all_validators()
    assert all(i in cmap for i in ids)
    with pytest.raises(ValueError):
        ValidatorComponentFactory.create_component(99999)


