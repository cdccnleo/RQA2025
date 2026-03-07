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


from src.data.quality.assurance_components import (
    AssuranceComponent,
    AssuranceComponentFactory,
)
import pytest


def test_assurance_component_info_status_and_process():
    comp = AssuranceComponent(assurance_id=10, component_type="Assurance")
    info = comp.get_info()
    assert info["assurance_id"] == 10
    assert info["component_type"] == "Assurance"
    status = comp.get_status()
    assert status["status"] == "active"
    res = comp.process({"x": 1})
    assert res["status"] == "success" and res["assurance_id"] == 10


def test_assurance_factory_supported_ids_and_create_all():
    ids = AssuranceComponentFactory.get_available_assurances()
    assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)
    comp_map = AssuranceComponentFactory.create_all_assurances()
    for i in ids:
        assert i in comp_map
        assert comp_map[i].get_assurance_id() == i
    info = AssuranceComponentFactory.get_factory_info()
    assert info["factory_name"] == "AssuranceComponentFactory"


def test_assurance_factory_invalid_id_raises():
    with pytest.raises(ValueError):
        AssuranceComponentFactory.create_component(-1)


