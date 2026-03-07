"""
补充测试：stock_loader.py（第六批）
测试未覆盖的方法：_check_cache 异常处理、_validate_volume 等
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
from src.data.loader.stock_loader import StockDataLoader
from src.infrastructure.utils.core.exceptions import DataLoaderError


def test_stock_loader_check_cache_exception(tmp_path, monkeypatch):
    """测试 StockDataLoader（_check_cache，读取异常，覆盖 751-753 行）"""
    loader = StockDataLoader(save_path=str(tmp_path))
    
    file_path = tmp_path / "invalid.csv"
    file_path.write_text("invalid csv content that will cause error")
    
    # Mock pd.read_csv 抛出异常
    def mock_read_csv(*args, **kwargs):
        raise ValueError("Invalid CSV format")
    
    monkeypatch.setattr(pd, 'read_csv', mock_read_csv)
    
    exists, data = loader._check_cache(file_path)
    
    assert exists is False
    assert data is None


def test_stock_loader_check_cache_invalid_format(tmp_path, monkeypatch):
    """测试 StockDataLoader（_check_cache，文件格式无效，覆盖 739-744 行）"""
    loader = StockDataLoader(save_path=str(tmp_path))
    
    file_path = tmp_path / "invalid.csv"
    file_path.write_text("not a valid csv")
    
    # Mock _is_cache_valid 返回 True（先通过文件存在检查）
    monkeypatch.setattr(loader, '_is_cache_valid', lambda x: True)
    
    exists, data = loader._check_cache(file_path)
    
    # 由于文件格式无效，应该返回 False
    assert exists is False or data is None


def test_stock_loader_check_cache_expired(tmp_path, monkeypatch):
    """测试 StockDataLoader（_check_cache，缓存过期，覆盖 747-748 行）"""
    loader = StockDataLoader(save_path=str(tmp_path))
    
    file_path = tmp_path / "expired.csv"
    df = pd.DataFrame({
        'open': [100],
        'high': [110],
        'low': [90],
        'close': [105],
        'volume': [1000]
    }, index=pd.date_range('2023-01-01', periods=1))
    df.to_csv(file_path, encoding='utf-8')
    
    # Mock _is_cache_valid 返回 False（缓存过期）
    monkeypatch.setattr(loader, '_is_cache_valid', lambda x: False)
    
    exists, data = loader._check_cache(file_path)
    
    assert exists is False
    assert data is None


def test_stock_loader_validate_volume_none(tmp_path):
    """测试 StockDataLoader（_validate_volume，df 为 None，覆盖 757-758 行）"""
    loader = StockDataLoader(save_path=str(tmp_path))
    
    result = loader._validate_volume(None)
    assert result is False


def test_stock_loader_validate_volume_empty(tmp_path):
    """测试 StockDataLoader（_validate_volume，df 为空，覆盖 757-758 行）"""
    loader = StockDataLoader(save_path=str(tmp_path))
    
    result = loader._validate_volume(pd.DataFrame())
    assert result is False


def test_stock_loader_validate_volume_english(tmp_path):
    """测试 StockDataLoader（_validate_volume，使用英文列名，覆盖 761-762 行）"""
    loader = StockDataLoader(save_path=str(tmp_path))
    
    df = pd.DataFrame({
        'volume': [1000, 2000, 3000]
    })
    
    result = loader._validate_volume(df)
    assert result is True


def test_stock_loader_validate_volume_chinese(tmp_path):
    """测试 StockDataLoader（_validate_volume，使用中文列名，覆盖 763-764 行）"""
    loader = StockDataLoader(save_path=str(tmp_path))
    
    df = pd.DataFrame({
        '成交量': [1000, 2000, 3000]
    })
    
    result = loader._validate_volume(df)
    assert result is True


def test_stock_loader_validate_volume_no_volume_col(tmp_path):
    """测试 StockDataLoader（_validate_volume，没有 volume 列，覆盖 766-767 行）"""
    loader = StockDataLoader(save_path=str(tmp_path))
    
    df = pd.DataFrame({
        'open': [100, 101, 102]
    })
    
    result = loader._validate_volume(df)
    assert result is False


def test_stock_loader_validate_volume_has_nan(tmp_path):
    """测试 StockDataLoader（_validate_volume，包含 NaN，覆盖 769-770 行）"""
    loader = StockDataLoader(save_path=str(tmp_path))
    
    df = pd.DataFrame({
        'volume': [1000, None, 3000]  # 包含 NaN
    })
    
    result = loader._validate_volume(df)
    assert result is False


def test_stock_loader_validate_volume_negative(tmp_path):
    """测试 StockDataLoader（_validate_volume，包含负值，覆盖 772-773 行）"""
    loader = StockDataLoader(save_path=str(tmp_path))
    
    df = pd.DataFrame({
        'volume': [1000, -100, 3000]  # 包含负值
    })
    
    result = loader._validate_volume(df)
    assert result is False


def test_stock_loader_validate_volume_zero(tmp_path):
    """测试 StockDataLoader（_validate_volume，包含零值，覆盖 772-773 行）"""
    loader = StockDataLoader(save_path=str(tmp_path))
    
    df = pd.DataFrame({
        'volume': [1000, 0, 3000]  # 包含零值
    })
    
    result = loader._validate_volume(df)
    assert result is False


def test_stock_loader_validate_volume_valid(tmp_path):
    """测试 StockDataLoader（_validate_volume，有效数据，覆盖 755-775 行）"""
    loader = StockDataLoader(save_path=str(tmp_path))
    
    df = pd.DataFrame({
        'volume': [1000, 2000, 3000]  # 所有值都为正
    })
    
    result = loader._validate_volume(df)
    assert result is True


def test_stock_loader_validate_volume_string_values(tmp_path):
    """测试 StockDataLoader（_validate_volume，字符串值转换为数值，覆盖 762, 764 行）"""
    loader = StockDataLoader(save_path=str(tmp_path))
    
    df = pd.DataFrame({
        'volume': ['1000', '2000', '3000']  # 字符串值
    })
    
    result = loader._validate_volume(df)
    assert result is True


def test_stock_loader_validate_volume_chinese_string_values(tmp_path):
    """测试 StockDataLoader（_validate_volume，中文列名字符串值，覆盖 764 行）"""
    loader = StockDataLoader(save_path=str(tmp_path))
    
    df = pd.DataFrame({
        '成交量': ['1000', '2000', '3000']  # 中文列名，字符串值
    })
    
    result = loader._validate_volume(df)
    assert result is True


def test_stock_loader_validate_volume_invalid_string(tmp_path):
    """测试 StockDataLoader（_validate_volume，无效字符串值，覆盖 762 行）"""
    loader = StockDataLoader(save_path=str(tmp_path))
    
    df = pd.DataFrame({
        'volume': ['1000', 'invalid', '3000']  # 包含无效字符串
    })
    
    result = loader._validate_volume(df)
    # 无效字符串会被转换为 NaN，所以应该返回 False
    assert result is False

