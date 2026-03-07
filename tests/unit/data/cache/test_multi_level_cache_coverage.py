"""
测试multi_level_cache的覆盖率提升
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
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.data.cache.multi_level_cache import (
    MultiLevelCache,
    CacheConfig
)


@pytest.fixture
def temp_cache_dir():
    """创建临时缓存目录"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_cache_config(temp_cache_dir):
    """创建示例缓存配置"""
    return CacheConfig(
        memory_max_size=100,
        disk_enabled=True,
        disk_cache_dir=str(temp_cache_dir),
        disk_ttl=3600,
        redis_enabled=False
    )


@pytest.fixture
def sample_cache(sample_cache_config):
    """创建示例缓存"""
    return MultiLevelCache(sample_cache_config)


def test_multi_level_cache_set_redis_exception(monkeypatch, sample_cache):
    """测试set方法的Redis异常处理（176-178行）"""
    # Mock redis_cache to raise exception
    mock_redis = Mock()
    mock_redis.set.side_effect = Exception("Redis failed")
    sample_cache.redis_cache = mock_redis
    
    # Set should still succeed for memory cache
    result = sample_cache.set("test_key", "test_value")
    
    # Should return True even if Redis fails
    assert result is True
    # Memory cache should still have the value
    assert sample_cache.get("test_key") == "test_value"


def test_multi_level_cache_delete_redis(sample_cache):
    """测试delete方法的Redis删除（206行）"""
    # Mock redis_cache
    mock_redis = Mock()
    sample_cache.redis_cache = mock_redis
    
    # Set a value first
    sample_cache.set("test_key", "test_value")
    
    # Delete
    result = sample_cache.delete("test_key")
    
    # Should call redis delete
    assert result is True
    mock_redis.delete.assert_called_once_with("test_key")


def test_multi_level_cache_delete_exception(monkeypatch, sample_cache):
    """测试delete方法的异常处理（209-211行）"""
    # Set a value first
    sample_cache.set("test_key", "test_value")
    
    # Mock disk cache deletion to raise exception
    def failing_unlink(path):
        raise Exception("Unlink failed")
    
    # Mock Path.unlink to raise exception
    with patch.object(Path, 'unlink', side_effect=Exception("Unlink failed")):
        # Delete should handle exception
        result = sample_cache.delete("test_key")
        
        # Should return False on exception
        assert result is False




def test_multi_level_cache_get_from_disk_expired(temp_cache_dir, sample_cache):
    """测试从磁盘获取过期缓存（293-294行）"""
    import pickle
    
    # Create an expired cache file
    cache_file = temp_cache_dir / "test_key.pkl"
    with open(cache_file, 'wb') as f:
        pickle.dump("old_value", f)
    
    # Make file old by setting mtime to past
    old_time = time.time() - 7200  # 2 hours ago
    import os
    os.utime(cache_file, (old_time, old_time))
    
    # Try to get - should return None because expired
    result = sample_cache._get_from_disk("test_key")
    
    assert result is None
    # File should be deleted
    assert not cache_file.exists()


def test_multi_level_cache_get_from_disk_exception(monkeypatch, sample_cache, temp_cache_dir):
    """测试从磁盘获取缓存异常处理（299-302行）"""
    import pickle
    
    # Create a cache file
    cache_file = temp_cache_dir / "test_key.pkl"
    with open(cache_file, 'wb') as f:
        pickle.dump("test_value", f)
    
    # Mock pickle.load to raise exception
    with patch('src.data.cache.multi_level_cache.pickle.load') as mock_load:
        mock_load.side_effect = Exception("Pickle failed")
        
        result = sample_cache._get_from_disk("test_key")
        
        # Should return None on exception
        assert result is None
        # File should be deleted
        assert not cache_file.exists()


def test_multi_level_cache_set_to_disk_disabled(sample_cache):
    """测试设置磁盘缓存当disk_enabled为False时（307行）"""
    # Disable disk cache
    sample_cache.config.disk_enabled = False
    
    # Set to disk should return early
    sample_cache._set_to_disk("test_key", "test_value", 3600)
    
    # Should not create file
    cache_file = sample_cache.disk_cache_dir / "test_key.pkl"
    assert not cache_file.exists()


def test_multi_level_cache_set_to_disk_ttl_zero(sample_cache, temp_cache_dir):
    """测试设置磁盘缓存当ttl <= 0时（312-314行）"""
    # Create an existing cache file
    cache_file = temp_cache_dir / "test_key.pkl"
    cache_file.touch()
    
    # Set with ttl = 0
    sample_cache._set_to_disk("test_key", "test_value", 0)
    
    # File should be deleted
    assert not cache_file.exists()


def test_multi_level_cache_set_to_disk_exception(monkeypatch, sample_cache):
    """测试设置磁盘缓存异常处理（317-318行）"""
    # Mock open to raise exception
    with patch('builtins.open', side_effect=Exception("Open failed")):
        sample_cache._set_to_disk("test_key", "test_value", 3600)
        
        # Should handle exception gracefully
        assert True


def test_multi_level_cache_is_memory_expired_not_in_timestamps(sample_cache):
    """测试内存缓存过期检查当key不在timestamps中时（323行）"""
    # Key not in timestamps
    result = sample_cache._is_memory_expired("nonexistent_key")
    
    # Should return True
    assert result is True


def test_multi_level_cache_evict_memory_lru_empty(sample_cache):
    """测试LRU淘汰当访问计数为空时（329行）"""
    # Clear access count
    sample_cache.memory_access_count.clear()
    
    # Should return early
    sample_cache._evict_memory_lru()
    
    # Should not raise exception
    assert True


def test_multi_level_cache_get_from_redis_exception(monkeypatch, sample_cache):
    """测试从Redis获取缓存异常处理（421-425行）"""
    # Mock redis_cache to raise exception
    mock_redis = Mock()
    mock_redis.get.side_effect = Exception("Redis get failed")
    sample_cache.redis_cache = mock_redis
    
    result = sample_cache._get_from_redis("test_key")
    
    # Should return None on exception
    assert result is None


def test_multi_level_cache_set_to_redis_none(sample_cache):
    """测试设置Redis缓存当redis_cache为None时（429-430行）"""
    # Ensure redis_cache is None
    sample_cache.redis_cache = None
    
    # Should return early
    sample_cache._set_to_redis("test_key", "test_value", 3600)
    
    # Should not raise exception
    assert True


def test_multi_level_cache_set_to_redis_ttl_zero(sample_cache):
    """测试设置Redis缓存当ttl <= 0时（433-435行）"""
    # Mock redis_cache
    mock_redis = Mock()
    sample_cache.redis_cache = mock_redis
    
    # Set with ttl = 0
    sample_cache._set_to_redis("test_key", "test_value", 0)
    
    # Should call delete
    mock_redis.delete.assert_called_once_with("test_key")
    mock_redis.set.assert_not_called()


def test_multi_level_cache_set_to_redis_exception(monkeypatch, sample_cache):
    """测试设置Redis缓存异常处理（437-438行）"""
    # Mock redis_cache to raise exception
    mock_redis = Mock()
    mock_redis.set.side_effect = Exception("Redis set failed")
    sample_cache.redis_cache = mock_redis
    
    # Should handle exception gracefully
    sample_cache._set_to_redis("test_key", "test_value", 3600)
    
    assert True


def test_multi_level_cache_get_redis_stats_none(sample_cache):
    """测试获取Redis统计当redis_cache为None时（447-448行）"""
    # Ensure redis_cache is None
    sample_cache.redis_cache = None
    
    result = sample_cache.get_redis_stats()
    
    # Should return None
    assert result is None


def test_multi_level_cache_get_redis_stats(sample_cache):
    """测试获取Redis统计（449-454行）"""
    # Mock redis_cache with get_stats method
    mock_redis = Mock()
    mock_redis.get_stats.return_value = {"hits": 10, "misses": 5}
    sample_cache.redis_cache = mock_redis
    
    result = sample_cache.get_redis_stats()
    
    # Should return stats
    assert result is not None
    assert "hits" in result


def test_multi_level_cache_cleanup_disk_exception(monkeypatch, sample_cache):
    """测试清理磁盘缓存异常处理（474-475行）"""
    # Mock shutil.rmtree to raise exception
    import shutil
    with patch('shutil.rmtree', side_effect=Exception("Rmtree failed")):
        # Should handle exception gracefully
        sample_cache.cleanup()
        
        assert True


def test_multi_level_cache_cleanup_redis_exception(monkeypatch, sample_cache):
    """测试清理Redis缓存异常处理（481-482行）"""
    # Mock redis_cache to raise exception
    mock_redis = Mock()
    mock_redis.clear.side_effect = Exception("Redis clear failed")
    sample_cache.redis_cache = mock_redis
    
    # Should handle exception gracefully
    sample_cache.cleanup()
    
    assert True



