"""
简单的stock_loader测试，用于验证覆盖率收集
"""
import pytest
import tempfile
from pathlib import Path
from src.data.loader.stock_loader import StockDataLoader


def test_stock_loader_simple():
    """简单的stock_loader测试"""
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = StockDataLoader(save_path=tmpdir)

        # 测试基本属性
        assert loader.save_path == Path(tmpdir)
        assert loader.max_retries == 3
        assert loader.cache_days == 30

        # 测试元数据
        metadata = loader.get_metadata()
        assert isinstance(metadata, dict)
        assert metadata['loader_type'] == 'StockDataLoader'

        # 测试配置验证
        valid_config = {'save_path': tmpdir, 'max_retries': 3, 'cache_days': 30}
        assert loader.validate_config(valid_config) == True

        invalid_config = {}
        assert loader.validate_config(invalid_config) == False

        # 测试缓存键生成（跳过需要end_date的文件路径测试）
        cache_key = loader._get_cache_key('000001', 'daily', 'none')
        assert isinstance(cache_key, str)
        assert '000001' in cache_key

        # 测试缓存键生成
        cache_key = loader._get_cache_key('000001', 'daily', 'none')
        assert isinstance(cache_key, str)
        assert '000001' in cache_key
