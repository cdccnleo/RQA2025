"""
StockListLoader 边界测试
测试 stock_loader.py 中的 StockListLoader 类
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
from src.data.loader.stock_loader import StockListLoader
from src.infrastructure.utils.core.exceptions import DataLoaderError


def test_stock_list_loader_init_default(tmp_path):
    """测试 StockListLoader（__init__，默认参数，覆盖 1146-1158 行）"""
    # 使用临时路径而不是默认路径，避免创建实际目录
    loader = StockListLoader(save_path=str(tmp_path))
    assert loader.save_path == Path(tmp_path)
    assert loader.cache_days == 7
    assert loader.max_retries == 3
    assert loader.list_path == loader.save_path / "stock_list.csv"


def test_stock_list_loader_init_custom(tmp_path):
    """测试 StockListLoader（__init__，自定义参数）"""
    loader = StockListLoader(
        save_path=str(tmp_path),
        max_retries=5,
        cache_days=14
    )
    assert loader.save_path == Path(tmp_path)
    assert loader.cache_days == 14
    assert loader.max_retries == 5


def test_stock_list_loader_setup(tmp_path):
    """测试 StockListLoader（_setup，覆盖 1160-1162 行）"""
    loader = StockListLoader(save_path=str(tmp_path))
    # 验证目录已创建
    assert loader.save_path.exists()
    assert loader.save_path.is_dir()

