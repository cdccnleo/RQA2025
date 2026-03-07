#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试指数数据加载器

测试目标：提升index_loader.py的覆盖率到80%+，确保100%测试通过率
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
import tempfile
import pickle
import time
import configparser
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from src.data.loader.index_loader import IndexDataLoader
try:
    from src.infrastructure.error import DataLoaderError
except ImportError:
    from src.infrastructure.utils.exceptions import DataLoaderError


class TestIndexDataLoader:
    """测试指数数据加载器"""

    @pytest.fixture
    def index_loader(self, tmp_path):
        """创建指数数据加载器实例"""
        loader = IndexDataLoader(
            save_path=str(tmp_path),
            max_retries=3,
            cache_days=30
        )
        return loader

    def test_index_loader_initialization(self, tmp_path):
        """测试指数加载器初始化"""
        loader = IndexDataLoader(save_path=str(tmp_path))
        assert loader.max_retries == 3
        assert loader.cache_days == 30
        assert loader.save_path == Path(tmp_path)
        assert loader.cache_dir.exists()
        assert loader.log_dir.exists()

    def test_index_loader_initialization_default_path(self):
        """测试使用默认路径初始化"""
        with patch('pathlib.Path.mkdir'):
            loader = IndexDataLoader()
            assert isinstance(loader.save_path, Path)

    def test_index_loader_get_required_config_fields(self, index_loader):
        """测试获取必需配置字段"""
        fields = index_loader.get_required_config_fields()
        assert isinstance(fields, list)
        assert 'save_path' in fields
        assert 'max_retries' in fields
        assert 'cache_days' in fields

    def test_index_loader_get_index_code(self, index_loader):
        """测试获取指数代码"""
        code = index_loader._get_index_code("HS300")
        assert code == "000300"
        
        # 不存在的指数代码返回原值
        code = index_loader._get_index_code("UNKNOWN")
        assert code == "UNKNOWN"

    def test_index_loader_get_cache_key(self, index_loader):
        """测试获取缓存键"""
        key = index_loader._get_cache_key("HS300")
        assert key == "HS300_index"

    def test_index_loader_get_cache_file_path(self, index_loader):
        """测试获取缓存文件路径"""
        path = index_loader._get_cache_file_path("HS300")
        assert isinstance(path, Path)
        assert "000300" in str(path)
        assert path.suffix == ".pkl"

    def test_index_loader_get_file_path(self, index_loader):
        """测试获取文件路径"""
        path = index_loader._get_file_path("HS300", "2024-01-01", "2024-01-31")
        assert isinstance(path, Path)
        assert "HS300" in str(path)  # 使用原始index_code，不是映射后的代码
        assert "20240101" in str(path)
        assert "20240131" in str(path)

    def test_index_loader_is_cache_valid_datetime(self, index_loader):
        """测试缓存有效性检查（datetime）"""
        # 有效的缓存时间（1天前）
        valid_time = datetime.now() - timedelta(days=1)
        result = index_loader._is_cache_valid(valid_time)
        assert result is True
        
        # 无效的缓存时间（31天前）
        invalid_time = datetime.now() - timedelta(days=31)
        index_loader.cache_days = 30
        result = index_loader._is_cache_valid(invalid_time)
        assert result is False

    def test_index_loader_is_cache_valid_file_not_exists(self, index_loader, tmp_path):
        """测试缓存有效性检查（文件不存在）"""
        fake_path = tmp_path / "nonexistent.csv"
        result = index_loader._is_cache_valid(fake_path)
        assert result is False

    def test_index_loader_is_cache_valid_file_expired(self, index_loader, tmp_path):
        """测试缓存有效性检查（文件过期）"""
        test_file = tmp_path / "test.csv"
        test_file.write_text("date,open,high,low,close,volume\n2024-01-01,100,110,90,105,1000")
        
        # 修改文件时间为31天前
        old_time = time.time() - (31 * 24 * 60 * 60)
        import os
        os.utime(test_file, (old_time, old_time))
        
        index_loader.cache_days = 30
        result = index_loader._is_cache_valid(test_file)
        assert result is False

    def test_index_loader_is_cache_valid_file_valid(self, index_loader, tmp_path):
        """测试缓存有效性检查（文件有效）"""
        test_file = tmp_path / "test.csv"
        test_file.write_text("date,open,high,low,close,volume\n2024-01-01,100,110,90,105,1000")
        
        index_loader.cache_days = 30
        result = index_loader._is_cache_valid(test_file)
        assert result is True

    def test_index_loader_is_cache_valid_file_missing_columns(self, index_loader, tmp_path):
        """测试缓存有效性检查（文件缺少必要列）"""
        test_file = tmp_path / "test.csv"
        test_file.write_text("date,open\n2024-01-01,100")  # 缺少必要列
        
        result = index_loader._is_cache_valid(test_file)
        assert result is False

    def test_index_loader_load_cache_payload_not_exists(self, index_loader, tmp_path):
        """测试加载缓存载荷（文件不存在）"""
        cache_file = tmp_path / "nonexistent.pkl"
        result = index_loader._load_cache_payload(cache_file)
        assert result is None

    def test_index_loader_load_cache_payload_valid(self, index_loader, tmp_path):
        """测试加载缓存载荷（有效）"""
        cache_file = tmp_path / "test.pkl"
        payload = {
            "data": pd.DataFrame({'open': [100], 'high': [110], 'low': [90], 'close': [105], 'volume': [1000]},
                               index=pd.date_range('2024-01-01', periods=1)),
            "metadata": {
                "cached_time": datetime.now() - timedelta(days=1)
            }
        }
        with cache_file.open("wb") as f:
            pickle.dump(payload, f)
        
        result = index_loader._load_cache_payload(cache_file)
        assert result is not None
        assert "data" in result
        assert "metadata" in result

    def test_index_loader_load_cache_payload_expired(self, index_loader, tmp_path):
        """测试加载缓存载荷（过期）"""
        cache_file = tmp_path / "test.pkl"
        payload = {
            "data": pd.DataFrame({'open': [100]}),
            "metadata": {
                "cached_time": datetime.now() - timedelta(days=31)
            }
        }
        with cache_file.open("wb") as f:
            pickle.dump(payload, f)
        
        index_loader.cache_days = 30
        result = index_loader._load_cache_payload(cache_file)
        assert result is None

    def test_index_loader_save_cache_payload(self, index_loader, tmp_path):
        """测试保存缓存载荷"""
        cache_file = tmp_path / "test.pkl"
        payload = {
            "data": pd.DataFrame({'open': [100]}),
            "metadata": {
                "cached_time": datetime.now()
            }
        }
        
        index_loader._save_cache_payload(cache_file, payload)
        assert cache_file.exists()
        
        # 验证可以加载
        with cache_file.open("rb") as f:
            loaded = pickle.load(f)
        assert "data" in loaded
        assert "metadata" in loaded

    def test_index_loader_validate_index_list(self, index_loader):
        """测试验证指数列表"""
        valid_list = ["HS300", "SZ50"]
        result = index_loader._validate_index_list(valid_list)
        assert result is True
        
        invalid_list = ["HS300", "UNKNOWN"]
        result = index_loader._validate_index_list(invalid_list)
        assert result is False

    def test_index_loader_validate_index_data_valid(self, index_loader):
        """测试验证指数数据（有效）"""
        df = pd.DataFrame({
            '日期': pd.date_range('2024-01-01', periods=2),
            '开盘': [100, 101],
            '收盘': [105, 106],
            '最高': [110, 111],
            '最低': [90, 91],
            '成交量': [1000, 1100]
        })
        
        is_valid, errors = index_loader._validate_index_data(df)
        assert is_valid is True
        assert len(errors) == 0

    def test_index_loader_validate_index_data_none(self, index_loader):
        """测试验证指数数据（None）"""
        is_valid, errors = index_loader._validate_index_data(None)
        assert is_valid is False
        assert "data is None" in errors

    def test_index_loader_validate_index_data_not_dataframe(self, index_loader):
        """测试验证指数数据（非DataFrame）"""
        is_valid, errors = index_loader._validate_index_data("not a dataframe")
        assert is_valid is False
        assert "data is not a DataFrame" in errors

    def test_index_loader_validate_index_data_missing_columns(self, index_loader):
        """测试验证指数数据（缺少列）"""
        df = pd.DataFrame({
            '日期': pd.date_range('2024-01-01', periods=2),
            '开盘': [100, 101]
        })
        
        is_valid, errors = index_loader._validate_index_data(df)
        assert is_valid is False
        assert any("missing columns" in err for err in errors)

    def test_index_loader_validate_index_data_negative_volume(self, index_loader):
        """测试验证指数数据（负交易量）"""
        df = pd.DataFrame({
            '日期': pd.date_range('2024-01-01', periods=2),
            '开盘': [100, 101],
            '收盘': [105, 106],
            '最高': [110, 111],
            '最低': [90, 91],
            '成交量': [-1000, 1100]  # 负值
        })
        
        is_valid, errors = index_loader._validate_index_data(df)
        assert is_valid is False
        assert any("negative values" in err for err in errors)

    def test_index_loader_validate(self, index_loader):
        """测试validate方法"""
        df = pd.DataFrame({
            'open': [100, 101],
            'high': [110, 111],
            'low': [90, 91],
            'close': [105, 106],
            'volume': [1000, 1100]
        })
        
        result = index_loader.validate(df)
        assert result is True
        
        # 无效数据
        invalid_df = pd.DataFrame()
        result = index_loader.validate(invalid_df)
        assert result is False

    def test_index_loader_get_metadata(self, index_loader):
        """测试获取元数据"""
        metadata = index_loader.get_metadata()
        assert isinstance(metadata, dict)
        assert metadata["loader_type"] == "IndexDataLoader"
        assert "cache_days" in metadata
        assert "max_retries" in metadata
        assert "supported_indices" in metadata

    def test_index_loader_load_data_invalid_index_code(self, index_loader):
        """测试加载数据（无效指数代码）"""
        with pytest.raises(ValueError, match="Invalid index code"):
            index_loader.load_data("INVALID", "2024-01-01", "2024-01-31")

    def test_index_loader_load_data_invalid_date_range(self, index_loader):
        """测试加载数据（无效日期范围）"""
        with pytest.raises(ValueError, match="开始日期不能大于结束日期"):
            index_loader.load_data("HS300", "2024-01-31", "2024-01-01")

    @patch('src.data.loader.index_loader.ak')
    def test_index_loader_load_data_from_cache(self, mock_ak, index_loader, tmp_path):
        """测试从缓存加载数据"""
        # 创建有效的缓存文件
        cache_file = index_loader._get_file_path("HS300", "2024-01-01", "2024-01-31")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=2),
            'open': [100, 101],
            'high': [110, 111],
            'low': [90, 91],
            'close': [105, 106],
            'volume': [1000, 1100]
        })
        df.to_csv(cache_file, encoding='utf-8', index=False)
        
        result = index_loader.load_data("HS300", "2024-01-01", "2024-01-31")
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        # 不应该调用API
        assert not hasattr(mock_ak, 'stock_zh_index_daily') or not mock_ak.stock_zh_index_daily.called

    @patch('src.data.loader.index_loader.ak')
    def test_index_loader_load_data_new_data(self, mock_ak, index_loader):
        """测试加载新数据"""
        # 模拟API返回数据
        mock_df = pd.DataFrame({
            '日期': pd.date_range('2024-01-01', periods=2),
            '开盘': [100, 101],
            '收盘': [105, 106],
            '最高': [110, 111],
            '最低': [90, 91],
            '成交量': [1000, 1100]
        })
        mock_ak.stock_zh_index_daily = Mock(return_value=mock_df)
        
        result = index_loader.load_data("HS300", "2024-01-01", "2024-01-31")
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert 'open' in result.columns
        assert 'close' in result.columns

    def test_index_loader_load_single_index_from_cache(self, index_loader, tmp_path):
        """测试从缓存加载单个指数"""
        cache_file = index_loader._get_cache_file_path("HS300")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        payload = {
            "data": pd.DataFrame({
                'open': [100, 101],
                'high': [110, 111],
                'low': [90, 91],
                'close': [105, 106],
                'volume': [1000, 1100]
            }, index=pd.date_range('2024-01-01', periods=2)),
            "metadata": {
                "cached_time": datetime.now() - timedelta(days=1)
            }
        }
        with cache_file.open("wb") as f:
            pickle.dump(payload, f)
        
        result = index_loader.load_single_index("HS300")
        assert result is not None
        assert "data" in result
        assert "metadata" in result
        assert result["cache_info"]["is_from_cache"] is True

    @patch('src.data.loader.index_loader.ak')
    def test_index_loader_load_single_index_new_data(self, mock_ak, index_loader):
        """测试加载新的单个指数数据"""
        mock_df = pd.DataFrame({
            '日期': pd.date_range('2024-01-01', periods=2),
            '开盘': [100, 101],
            '收盘': [105, 106],
            '最高': [110, 111],
            '最低': [90, 91],
            '成交量': [1000, 1100]
        })
        mock_ak.stock_zh_index_daily = Mock(return_value=mock_df)
        
        result = index_loader.load_single_index("HS300", force_refresh=True)
        assert result is not None
        assert "data" in result
        assert "metadata" in result
        assert result["cache_info"]["is_from_cache"] is False

    def test_index_loader_load_multiple_indexes(self, index_loader):
        """测试加载多个指数"""
        with patch.object(index_loader, '_load_single_index_with_cache', return_value={
            "data": pd.DataFrame({'open': [100]}),
            "metadata": {}
        }):
            result = index_loader.load_multiple_indexes(["HS300", "SZ50"])
            assert isinstance(result, dict)
            assert "HS300" in result
            assert "SZ50" in result

    def test_index_loader_load_multiple_indexes_empty(self, index_loader):
        """测试加载多个指数（空列表）"""
        result = index_loader.load_multiple_indexes([])
        assert result == {}

    def test_index_loader_cleanup(self, index_loader, tmp_path):
        """测试清理缓存"""
        cache_file = tmp_path / "cache" / "test.pkl"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_bytes(b"test data")
        
        index_loader.cleanup()
        
        # 文件应该被删除
        assert not cache_file.exists()

    def test_index_loader_create_from_config_dict(self, tmp_path):
        """测试从字典配置创建实例"""
        config = {
            'Index': {
                'save_path': str(tmp_path),
                'max_retries': '5',
                'cache_days': '7'
            }
        }
        
        with patch('configparser.ConfigParser.read'):
            loader = IndexDataLoader.create_from_config(config)
            assert loader.save_path == Path(tmp_path)
            assert loader.max_retries == 5
            assert loader.cache_days == 7

    def test_index_loader_create_from_config_invalid_type(self):
        """测试从无效配置类型创建实例"""
        with pytest.raises(ValueError, match="不支持的配置类型"):
            IndexDataLoader.create_from_config("invalid")

    def test_index_loader_create_from_config_invalid_int(self):
        """测试从配置创建实例（无效整数）"""
        config = {
            'Index': {
                'max_retries': 'invalid_int'
            }
        }
        
        with patch('configparser.ConfigParser.read'):
            with pytest.raises(DataLoaderError):  # 匹配任何DataLoaderError
                IndexDataLoader.create_from_config(config)

    def test_index_loader_merge_with_cache(self, index_loader, tmp_path):
        """测试合并缓存数据"""
        cache_file = tmp_path / "cache.csv"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        cached_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=2),
            'open': [100, 101],
            'high': [110, 111],
            'low': [90, 91],
            'close': [105, 106],
            'volume': [1000, 1100]
        })
        cached_df.set_index('date', inplace=True)
        cached_df.to_csv(cache_file, encoding='utf-8')
        
        new_df = pd.DataFrame({
            'open': [102],
            'high': [112],
            'low': [92],
            'close': [107],
            'volume': [1200]
        }, index=pd.date_range('2024-01-03', periods=1))
        
        result = index_loader._merge_with_cache(cache_file, new_df)
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 2

    def test_index_loader_merge_with_cache_no_file(self, index_loader):
        """测试合并缓存数据（文件不存在）"""
        fake_file = Path("/nonexistent/file.csv")
        new_df = pd.DataFrame({
            'open': [100],
            'high': [110],
            'low': [90],
            'close': [105],
            'volume': [1000]
        })
        
        result = index_loader._merge_with_cache(fake_file, new_df)
        assert result is not None
        assert result.equals(new_df)

    def test_index_loader_save_data(self, index_loader, tmp_path):
        """测试保存数据"""
        df = pd.DataFrame({
            'open': [100, 101],
            'high': [110, 111],
            'low': [90, 91],
            'close': [105, 106],
            'volume': [1000, 1100]
        }, index=pd.date_range('2024-01-01', periods=2))
        
        file_path = tmp_path / "test.csv"
        result = index_loader._save_data(df, file_path)
        assert result is True
        assert file_path.exists()
        
        # 验证可以读取
        loaded = pd.read_csv(file_path)
        assert 'date' in loaded.columns
        assert 'open' in loaded.columns

