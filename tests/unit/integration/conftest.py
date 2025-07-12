import pytest
from src.data.data_loader import DataLoader
from src.data.cache.memory_cache import MemoryCache
from src.data.cache.disk_cache import DiskCache

@pytest.fixture(scope="module")
def test_loader():
    """提供配置好的测试数据加载器"""
    loader = DataLoader()

    # 使用测试专用的缓存实例
    loader.cache.memory_cache = MemoryCache()
    loader.cache.disk_cache = DiskCache()

    # 配置测试模式
    loader.parallel_loader.max_workers = 2  # 减少测试并发数

    yield loader

@pytest.fixture
def china_stock_config():
    """中国市场股票测试配置"""
    return {
        'market': 'china',
        'data_type': 'stock',
        'symbol': '600000',
        'parallel': False
    }

@pytest.fixture
def invalid_stock_config():
    """无效股票测试配置"""
    return {
        'market': 'china',
        'data_type': 'stock',
        'symbol': 'INVALID',
        'parallel': False
    }
