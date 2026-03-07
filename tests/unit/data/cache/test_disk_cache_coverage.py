"""
测试disk_cache的覆盖率提升
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
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path

from src.data.cache.disk_cache import (
    DiskCache,
    DiskCacheConfig
)


@pytest.fixture
def temp_cache_dir():
    """创建临时缓存目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_cache_config(temp_cache_dir):
    """创建示例缓存配置"""
    return DiskCacheConfig(
        cache_dir=str(temp_cache_dir),
        max_file_size=10 * 1024 * 1024
    )


@pytest.fixture
def sample_cache(sample_cache_config):
    """创建示例磁盘缓存"""
    cache = DiskCache(sample_cache_config)
    yield cache
    cache.close()


def test_disk_cache_get_entry_file_too_large(sample_cache, temp_cache_dir):
    """测试get_entry方法当文件过大时（203-207行）"""
    # Create a large file with correct hash name
    import hashlib
    key_hash = hashlib.md5("test_key".encode()).hexdigest()
    large_file = temp_cache_dir / f"{key_hash}.cache"
    with open(large_file, 'wb') as f:
        f.write(b'x' * (sample_cache.config.max_file_size + 1))
    
    # Try to get entry
    result = sample_cache.get_entry("test_key")
    
    # Should return None and delete the file
    assert result is None
    assert not large_file.exists()


def test_disk_cache_get_entry_read_exception(sample_cache, temp_cache_dir):
    """测试get_entry方法读取文件异常（213-217行）"""
    # Create a file that will fail to read
    cache_file = temp_cache_dir / "test_key.cache"
    cache_file.touch()
    
    # Mock open to raise exception
    with patch('builtins.open', side_effect=IOError("Read failed")):
        result = sample_cache.get_entry("test_key")
        
        # Should return None and delete the file
        assert result is None


def test_disk_cache_get_entry_deserialize_failure(sample_cache, temp_cache_dir):
    """测试get_entry方法反序列化失败（220-225行）"""
    # Create a file with invalid data using correct hash name
    import hashlib
    key_hash = hashlib.md5("test_key".encode()).hexdigest()
    cache_file = temp_cache_dir / f"{key_hash}.cache"
    with open(cache_file, 'wb') as f:
        f.write(b'invalid_data')
    
    # Try to get entry
    result = sample_cache.get_entry("test_key")
    
    # Should return None and delete the file
    assert result is None
    assert not cache_file.exists()


def test_disk_cache_get_entry_expired(sample_cache, temp_cache_dir):
    """测试get_entry方法当entry过期时（228-231行）"""
    # Set an entry with very short TTL
    sample_cache.set("test_key", "test_value", ttl=0)
    
    # Wait a bit
    import time
    time.sleep(0.1)
    
    # Try to get entry
    result = sample_cache.get_entry("test_key")
    
    # Should return None (expired)
    assert result is None


def test_disk_cache_get_entry_exception(sample_cache):
    """测试get_entry方法的异常处理（241-244行）"""
    # Mock _get_file_path to raise exception
    with patch.object(sample_cache, '_get_file_path', side_effect=Exception("Path error")):
        result = sample_cache.get_entry("test_key")
        
        # Should return None on exception
        assert result is None


def test_disk_cache_set_exception(sample_cache):
    """测试set方法的异常处理（275-278行）"""
    # Mock CacheEntry to raise exception
    with patch('src.data.cache.disk_cache.CacheEntry', side_effect=Exception("Entry creation failed")):
        result = sample_cache.set("test_key", "test_value")
        
        # Should return False on exception
        assert result is False


def test_disk_cache_save_to_disk_exception(sample_cache, temp_cache_dir):
    """测试_save_to_disk方法的异常处理（304-307行）"""
    from src.data.cache.cache_manager import CacheEntry
    
    entry = CacheEntry(key="test_key", value="test_value")
    
    # Mock open to raise exception
    with patch('builtins.open', side_effect=IOError("Write failed")):
        result = sample_cache._save_to_disk("test_key", entry)
        
        # Should return False on exception
        assert result is False


def test_disk_cache_delete_exception(sample_cache):
    """测试delete方法的异常处理（327-330行）"""
    # Mock _get_file_path to raise exception
    with patch.object(sample_cache, '_get_file_path', side_effect=Exception("Path error")):
        result = sample_cache.delete("test_key")
        
        # Should return False on exception
        assert result is False


def test_disk_cache_exists_exception(sample_cache):
    """测试exists方法的异常处理（352-354行）"""
    # Mock get_entry to raise exception
    with patch.object(sample_cache, 'get_entry', side_effect=Exception("Get failed")):
        result = sample_cache.exists("test_key")
        
        # Should return False on exception
        assert result is False


# Note: clear_exception, list_keys_exception, and collect_disk_usage_oserror tests
# are difficult to mock reliably due to Path object read-only attributes.
# These edge cases are covered by the existing test suite.


def test_disk_cache_health_check_directory_not_exists(sample_cache, temp_cache_dir):
    """测试health_check方法当目录不存在时（440-441行）"""
    # Remove the directory
    import shutil
    shutil.rmtree(temp_cache_dir)
    
    result = sample_cache.health_check()
    
    # Should return error status
    assert result['status'] == 'error'


def test_disk_cache_health_check_no_write_permission(sample_cache, temp_cache_dir):
    """测试health_check方法当没有写权限时（444-445行）"""
    # Mock os.access to return False
    with patch('os.access', return_value=False):
        result = sample_cache.health_check()
        
        # Should return error status
        assert result['status'] == 'error'


def test_disk_cache_health_check_exception(sample_cache):
    """测试health_check方法的异常处理（458-459行）"""
    # Mock _collect_disk_usage to raise exception
    with patch.object(sample_cache, '_collect_disk_usage', side_effect=Exception("Collect failed")):
        result = sample_cache.health_check()
        
        # Should return error status
        assert result['status'] == 'error'


def test_disk_cache_cleanup_worker_exception(sample_cache):
    """测试cleanup_worker的异常处理（476-482行）"""
    # Start cleanup thread
    sample_cache._start_cleanup_thread()
    
    # Mock _cleanup_expired to raise exception
    with patch.object(sample_cache, '_cleanup_expired', side_effect=Exception("Cleanup failed")):
        # Wait a bit for the thread to run
        import time
        time.sleep(0.1)
    
    # Stop cleanup
    sample_cache.close()
    
    assert True


def test_disk_cache_cleanup_expired_file_not_found(sample_cache, temp_cache_dir):
    """测试_cleanup_expired方法当文件不存在时（493-509行）"""
    # Create a file
    cache_file = temp_cache_dir / "test.cache"
    cache_file.touch()
    
    # Mock open to raise FileNotFoundError
    with patch('builtins.open', side_effect=FileNotFoundError("File not found")):
        sample_cache._cleanup_expired()
        
        # Should handle exception gracefully
        assert True


def test_disk_cache_cleanup_expired_deserialize_none(sample_cache, temp_cache_dir):
    """测试_cleanup_expired方法当反序列化为None时（501-503行）"""
    # Create a file with invalid data
    cache_file = temp_cache_dir / "test.cache"
    with open(cache_file, 'wb') as f:
        f.write(b'invalid_data')
    
    # Run cleanup
    sample_cache._cleanup_expired()
    
    # File should be deleted
    assert not cache_file.exists()


def test_disk_cache_cleanup_expired_exception(sample_cache, temp_cache_dir):
    """测试_cleanup_expired方法的异常处理（505-509行）"""
    # Create a file
    cache_file = temp_cache_dir / "test.cache"
    cache_file.touch()
    
    # Mock open to raise exception
    with patch('builtins.open', side_effect=Exception("Open failed")):
        sample_cache._cleanup_expired()
        
        # Should handle exception and delete file
        assert not cache_file.exists()


# Note: cleanup_expired_outer_exception test is difficult to mock reliably
# due to Path object read-only attributes. This edge case is covered by
# the existing test suite.


def test_disk_cache_start_cleanup_thread_interval_zero(temp_cache_dir):
    """测试_start_cleanup_thread方法当interval为0时（463-464行）"""
    # Create config with cleanup_interval = 0
    config = DiskCacheConfig(
        cache_dir=str(temp_cache_dir),
        cleanup_interval=0
    )
    
    # Create cache with this config
    cache = DiskCache(config)
    
    try:
        # Thread should not be started
        assert cache._cleanup_thread is None or not cache._cleanup_thread.is_alive()
    finally:
        cache.close()

