"""
IndustryLoader 边界测试
测试 stock_loader.py 中的 IndustryLoader 类
"""
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
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from src.data.loader.stock_loader import IndustryLoader
from src.infrastructure.utils.core.exceptions import DataLoaderError


def test_industry_loader_init_invalid_save_path_type(tmp_path):
    """测试 IndustryLoader（__init__，save_path 类型无效，覆盖 801-802 行）"""
    with pytest.raises(ValueError, match="save_path必须是字符串"):
        IndustryLoader(save_path=123)  # 不是字符串


def test_industry_loader_init_invalid_max_retries(tmp_path):
    """测试 IndustryLoader（__init__，max_retries <= 0，覆盖 803-804 行）"""
    with pytest.raises(ValueError, match="max_retries必须大于0"):
        IndustryLoader(save_path=str(tmp_path), max_retries=0)


def test_industry_loader_init_invalid_frequency(tmp_path):
    """测试 IndustryLoader（__init__，frequency 无效，覆盖 805-806 行）"""
    with pytest.raises(ValueError, match="frequency必须是daily / weekly / monthly"):
        IndustryLoader(save_path=str(tmp_path), frequency="invalid")


def test_industry_loader_init_success(tmp_path):
    """测试 IndustryLoader（__init__，成功初始化）"""
    loader = IndustryLoader(save_path=str(tmp_path))
    assert loader.save_path == Path(tmp_path)
    assert loader.max_retries == 3
    assert loader.cache_days == 30
    assert loader.frequency == 'daily'


def test_industry_loader_init_custom_params(tmp_path):
    """测试 IndustryLoader（__init__，自定义参数）"""
    loader = IndustryLoader(
        save_path=str(tmp_path),
        max_retries=5,
        cache_days=60,
        frequency='weekly',
        adjust_type='pre'
    )
    assert loader.max_retries == 5
    assert loader.cache_days == 60
    assert loader.frequency == 'weekly'
    assert loader.adjust_type == 'pre'

