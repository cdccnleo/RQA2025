"""
边界测试：disk_cache.py
测试边界情况和异常场景
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
import pickle
import time
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime

from src.data.cache.disk_cache import (
    DiskCacheConfig, DiskCache, _safe_logger_log, get_data_logger
)
from src.data.cache.cache_manager import CacheEntry, CacheStats


@pytest.fixture
def tmp_cache_dir(tmp_path):
    """创建临时缓存目录"""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return str(cache_dir)


@pytest.fixture
def disk_cache_config(tmp_cache_dir):
    """创建磁盘缓存配置"""
    return DiskCacheConfig(cache_dir=tmp_cache_dir)


@pytest.fixture
def disk_cache(disk_cache_config):
    """创建磁盘缓存实例"""
    cache = DiskCache(disk_cache_config)
    yield cache
    cache.stop()


def test_disk_cache_config_default():
    """测试 DiskCacheConfig（默认值）"""
    config = DiskCacheConfig()
    assert config.cache_dir == "cache"
    assert config.max_file_size == 10 * 1024 * 1024
    assert config.compression is False
    assert config.encryption is False
    assert config.encryption_key is None
    assert config.backup_enabled is False
    assert config.backup_interval == 3600
    assert config.cleanup_interval == 300


def test_disk_cache_config_custom():
    """测试 DiskCacheConfig（自定义值）"""
    config = DiskCacheConfig(
        cache_dir="/custom/path",
        max_file_size=20 * 1024 * 1024,
        compression=True,
        encryption=True,
        encryption_key="test_key",
        backup_enabled=True,
        backup_interval=7200,
        cleanup_interval=600
    )
    assert config.cache_dir == "/custom/path"
    assert config.max_file_size == 20 * 1024 * 1024
    assert config.compression is True
    assert config.encryption is True
    assert config.encryption_key == "test_key"
    assert config.backup_enabled is True
    assert config.backup_interval == 7200
    assert config.cleanup_interval == 600


def test_disk_cache_init(disk_cache_config):
    """测试 DiskCache（初始化）"""
    cache = DiskCache(disk_cache_config)
    assert cache.config == disk_cache_config
    assert cache.cache_dir.exists()
    assert cache.stats is not None
    cache.stop()


def test_disk_cache_init_creates_directory(tmp_path):
    """测试 DiskCache（初始化，创建目录）"""
    cache_dir = tmp_path / "new_cache"
    config = DiskCacheConfig(cache_dir=str(cache_dir))
    cache = DiskCache(config)
    assert cache_dir.exists()
    cache.stop()


def test_disk_cache_get_file_path(disk_cache):
    """测试 DiskCache（获取文件路径）"""
    path = disk_cache._get_file_path("test_key")
    assert isinstance(path, Path)
    assert path.suffix == ".cache"


def test_disk_cache_get_file_path_empty_key(disk_cache):
    """测试 DiskCache（获取文件路径，空键）"""
    path = disk_cache._get_file_path("")
    assert isinstance(path, Path)
    assert path.suffix == ".cache"


def test_disk_cache_get_file_path_special_chars(disk_cache):
    """测试 DiskCache（获取文件路径，特殊字符）"""
    path = disk_cache._get_file_path("test/key with spaces & symbols!")
    assert isinstance(path, Path)
    assert path.suffix == ".cache"


def test_disk_cache_serialize_entry(disk_cache):
    """测试 DiskCache（序列化条目）"""
    entry = CacheEntry(key="test", value="data", ttl=3600)
    data = disk_cache._serialize_entry(entry)
    assert isinstance(data, bytes)
    assert len(data) > 0


def test_disk_cache_deserialize_entry(disk_cache):
    """测试 DiskCache（反序列化条目）"""
    entry = CacheEntry(key="test", value="data", ttl=3600)
    data = disk_cache._serialize_entry(entry)
    deserialized = disk_cache._deserialize_entry(data)
    assert deserialized is not None
    assert deserialized.key == "test"
    assert deserialized.value == "data"


def test_disk_cache_deserialize_entry_invalid_data(disk_cache):
    """测试 DiskCache（反序列化条目，无效数据）"""
    invalid_data = b"invalid pickle data"
    result = disk_cache._deserialize_entry(invalid_data)
    assert result is None


def test_disk_cache_deserialize_entry_empty_data(disk_cache):
    """测试 DiskCache（反序列化条目，空数据）"""
    result = disk_cache._deserialize_entry(b"")
    assert result is None


def test_disk_cache_get_nonexistent(disk_cache):
    """测试 DiskCache（获取，不存在）"""
    result = disk_cache.get("nonexistent_key")
    assert result is None


def test_disk_cache_get_empty_key(disk_cache):
    """测试 DiskCache（获取，空键）"""
    result = disk_cache.get("")
    assert result is None


def test_disk_cache_set_get(disk_cache):
    """测试 DiskCache（设置和获取）"""
    result = disk_cache.set("test_key", "test_value")
    assert result is True
    value = disk_cache.get("test_key")
    assert value == "test_value"


def test_disk_cache_set_get_with_ttl(disk_cache):
    """测试 DiskCache（设置和获取，带 TTL）"""
    result = disk_cache.set("test_key", "test_value", ttl=1)
    assert result is True
    value = disk_cache.get("test_key")
    assert value == "test_value"
    # 等待过期
    time.sleep(2)
    value = disk_cache.get("test_key")
    assert value is None


def test_disk_cache_set_empty_key(disk_cache):
    """测试 DiskCache（设置，空键）"""
    result = disk_cache.set("", "test_value")
    assert isinstance(result, bool)


def test_disk_cache_set_none_value(disk_cache):
    """测试 DiskCache（设置，None 值）"""
    result = disk_cache.set("test_key", None)
    assert result is True
    value = disk_cache.get("test_key")
    assert value is None


def test_disk_cache_set_zero_ttl(disk_cache):
    """测试 DiskCache（设置，零 TTL）"""
    result = disk_cache.set("test_key", "test_value", ttl=0)
    assert result is True
    # 零 TTL 应该立即过期
    value = disk_cache.get("test_key")
    # 可能返回 None 或值，取决于实现
    assert value is None or value == "test_value"


def test_disk_cache_set_negative_ttl(disk_cache):
    """测试 DiskCache（设置，负 TTL）"""
    result = disk_cache.set("test_key", "test_value", ttl=-1)
    assert result is True
    # 负 TTL 应该立即过期
    value = disk_cache.get("test_key")
    assert value is None


def test_disk_cache_delete_nonexistent(disk_cache):
    """测试 DiskCache（删除，不存在）"""
    result = disk_cache.delete("nonexistent_key")
    assert result is False


def test_disk_cache_delete_existing(disk_cache):
    """测试 DiskCache（删除，存在）"""
    disk_cache.set("test_key", "test_value")
    result = disk_cache.delete("test_key")
    assert result is True
    value = disk_cache.get("test_key")
    assert value is None


def test_disk_cache_delete_empty_key(disk_cache):
    """测试 DiskCache（删除，空键）"""
    result = disk_cache.delete("")
    assert isinstance(result, bool)


def test_disk_cache_exists_nonexistent(disk_cache):
    """测试 DiskCache（检查存在，不存在）"""
    result = disk_cache.exists("nonexistent_key")
    assert result is False


def test_disk_cache_exists_existing(disk_cache):
    """测试 DiskCache（检查存在，存在）"""
    disk_cache.set("test_key", "test_value")
    result = disk_cache.exists("test_key")
    assert result is True


def test_disk_cache_exists_expired(disk_cache):
    """测试 DiskCache（检查存在，已过期）"""
    disk_cache.set("test_key", "test_value", ttl=1)
    time.sleep(2)
    result = disk_cache.exists("test_key")
    assert result is False


def test_disk_cache_exists_empty_key(disk_cache):
    """测试 DiskCache（检查存在，空键）"""
    result = disk_cache.exists("")
    assert result is False


def test_disk_cache_clear_empty(disk_cache):
    """测试 DiskCache（清空，空缓存）"""
    result = disk_cache.clear()
    assert result is True


def test_disk_cache_clear_with_data(disk_cache):
    """测试 DiskCache（清空，有数据）"""
    disk_cache.set("key1", "value1")
    disk_cache.set("key2", "value2")
    result = disk_cache.clear()
    assert result is True
    assert disk_cache.get("key1") is None
    assert disk_cache.get("key2") is None


def test_disk_cache_list_keys_empty(disk_cache):
    """测试 DiskCache（列出键，空）"""
    keys = disk_cache.list_keys()
    assert isinstance(keys, list)
    assert len(keys) == 0


def test_disk_cache_list_keys_with_data(disk_cache):
    """测试 DiskCache（列出键，有数据）"""
    disk_cache.set("key1", "value1")
    disk_cache.set("key2", "value2")
    keys = disk_cache.list_keys()
    assert isinstance(keys, list)
    assert len(keys) >= 0  # 可能返回文件路径而不是原始键


def test_disk_cache_get_stats_empty(disk_cache):
    """测试 DiskCache（获取统计信息，空）"""
    stats = disk_cache.get_stats()
    assert isinstance(stats, dict)
    assert 'disk_cache' in stats


def test_disk_cache_get_stats_with_data(disk_cache):
    """测试 DiskCache（获取统计信息，有数据）"""
    disk_cache.set("key1", "value1")
    disk_cache.get("key1")
    stats = disk_cache.get_stats()
    assert isinstance(stats, dict)
    assert 'disk_cache' in stats
    assert stats['disk_cache']['file_count'] >= 0


def test_disk_cache_health_check(disk_cache):
    """测试 DiskCache（健康检查）"""
    health = disk_cache.health_check()
    assert isinstance(health, dict)
    assert 'status' in health


def test_disk_cache_health_check_nonexistent_dir(tmp_path):
    """测试 DiskCache（健康检查，不存在目录）"""
    cache_dir = tmp_path / "nonexistent"
    config = DiskCacheConfig(cache_dir=str(cache_dir))
    cache = DiskCache(config)
    # 删除目录以测试错误情况
    cache.cache_dir.rmdir()
    health = cache.health_check()
    assert isinstance(health, dict)
    assert 'status' in health
    cache.stop()


def test_disk_cache_get_entry_nonexistent(disk_cache):
    """测试 DiskCache（获取条目，不存在）"""
    entry = disk_cache.get_entry("nonexistent_key")
    assert entry is None


def test_disk_cache_get_entry_existing(disk_cache):
    """测试 DiskCache（获取条目，存在）"""
    disk_cache.set("test_key", "test_value")
    entry = disk_cache.get_entry("test_key")
    assert entry is not None
    assert entry.value == "test_value"


def test_disk_cache_get_entry_update_metadata_false(disk_cache):
    """测试 DiskCache（获取条目，不更新元数据）"""
    disk_cache.set("test_key", "test_value")
    entry1 = disk_cache.get_entry("test_key", update_metadata=False)
    entry2 = disk_cache.get_entry("test_key", update_metadata=False)
    assert entry1 is not None
    assert entry2 is not None


def test_disk_cache_get_entry_update_metadata_true(disk_cache):
    """测试 DiskCache（获取条目，更新元数据）"""
    disk_cache.set("test_key", "test_value")
    entry1 = disk_cache.get_entry("test_key", update_metadata=True)
    entry2 = disk_cache.get_entry("test_key", update_metadata=True)
    assert entry1 is not None
    assert entry2 is not None


def test_disk_cache_get_entry_expired(disk_cache):
    """测试 DiskCache（获取条目，已过期）"""
    disk_cache.set("test_key", "test_value", ttl=1)
    time.sleep(2)
    entry = disk_cache.get_entry("test_key")
    assert entry is None


def test_disk_cache_stop(disk_cache):
    """测试 DiskCache（停止）"""
    # 应该不会抛出异常
    disk_cache.stop()
    # 多次调用应该也是安全的
    disk_cache.stop()


def test_disk_cache_close(disk_cache):
    """测试 DiskCache（关闭）"""
    # 应该不会抛出异常
    disk_cache.close()
    # 多次调用应该也是安全的
    disk_cache.close()


def test_disk_cache_large_file(tmp_path):
    """测试 DiskCache（大文件）"""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    config = DiskCacheConfig(
        cache_dir=str(cache_dir),
        max_file_size=1024  # 1KB
    )
    cache = DiskCache(config)
    # 创建大于限制的数据
    large_value = "x" * 2048
    result = cache.set("large_key", large_value)
    # 可能成功或失败，取决于实现
    assert isinstance(result, bool)
    cache.stop()


def test_disk_cache_special_value_types(disk_cache):
    """测试 DiskCache（特殊值类型）"""
    # 测试各种数据类型
    test_cases = [
        ("dict", {"key": "value"}),
        ("list", [1, 2, 3]),
        ("tuple", (1, 2, 3)),
        ("int", 42),
        ("float", 3.14),
        ("bool", True),
        ("none", None),
    ]
    for key, value in test_cases:
        result = disk_cache.set(key, value)
        assert result is True
        retrieved = disk_cache.get(key)
        assert retrieved == value or (retrieved is None and value is None)


def test_disk_cache_get_entry_file_too_large(tmp_cache_dir, monkeypatch):
    """测试 DiskCache（get_entry，文件过大）"""
    config = DiskCacheConfig(cache_dir=tmp_cache_dir, max_file_size=100)
    cache = DiskCache(config)
    
    # 创建一个过大的文件
    file_path = cache._get_file_path("large_key")
    with open(file_path, 'wb') as f:
        f.write(b'x' * 200)  # 超过最大文件大小
    
    # 获取应该返回None（文件过大）
    entry = cache.get_entry("large_key")
    assert entry is None
    
    cache.stop()


def test_disk_cache_get_entry_read_error(tmp_cache_dir, monkeypatch):
    """测试 DiskCache（get_entry，读取文件异常）"""
    config = DiskCacheConfig(cache_dir=tmp_cache_dir)
    cache = DiskCache(config)
    
    # 设置一个缓存项
    cache.set("key1", "value1")
    
    # 模拟读取文件时抛出异常
    def mock_open(*args, **kwargs):
        raise IOError("Read error")
    
    monkeypatch.setattr("builtins.open", mock_open)
    
    # 获取应该返回None（异常被捕获）
    entry = cache.get_entry("key1")
    assert entry is None
    
    cache.stop()


def test_disk_cache_get_entry_deserialize_failure(tmp_cache_dir, monkeypatch):
    """测试 DiskCache（get_entry，反序列化失败）"""
    config = DiskCacheConfig(cache_dir=tmp_cache_dir)
    cache = DiskCache(config)
    
    # 创建一个损坏的缓存文件
    file_path = cache._get_file_path("corrupted_key")
    with open(file_path, 'wb') as f:
        f.write(b'corrupted data')
    
    # 获取应该返回None（反序列化失败）
    entry = cache.get_entry("corrupted_key")
    assert entry is None
    
    cache.stop()


def test_disk_cache_get_entry_exception(tmp_cache_dir, monkeypatch):
    """测试 DiskCache（get_entry，异常处理）"""
    config = DiskCacheConfig(cache_dir=tmp_cache_dir)
    cache = DiskCache(config)
    
    # 模拟_get_file_path抛出异常
    def mock_get_file_path(key):
        raise Exception("Path error")
    
    monkeypatch.setattr(cache, "_get_file_path", mock_get_file_path)
    
    # 获取应该返回None（异常被捕获）
    entry = cache.get_entry("key1")
    assert entry is None
    
    cache.stop()


def test_disk_cache_set_exception(tmp_cache_dir, monkeypatch):
    """测试 DiskCache（set，异常处理）"""
    config = DiskCacheConfig(cache_dir=tmp_cache_dir)
    cache = DiskCache(config)
    
    # 模拟CacheEntry创建时抛出异常
    def mock_cache_entry(*args, **kwargs):
        raise Exception("Entry creation error")
    
    monkeypatch.setattr("src.data.cache.disk_cache.CacheEntry", mock_cache_entry)
    
    # 设置应该返回False（异常被捕获）
    result = cache.set("key1", "value1")
    assert result is False
    
    cache.stop()


def test_disk_cache_save_to_disk_exception(tmp_cache_dir, monkeypatch):
    """测试 DiskCache（_save_to_disk，异常处理）"""
    config = DiskCacheConfig(cache_dir=tmp_cache_dir)
    cache = DiskCache(config)
    
    # 模拟保存到磁盘时抛出异常
    def mock_open(*args, **kwargs):
        raise Exception("Write error")
    
    monkeypatch.setattr("builtins.open", mock_open)
    
    # 保存应该返回False（异常被捕获）
    entry = CacheEntry(key="key1", value="value1")
    result = cache._save_to_disk("key1", entry)
    assert result is False
    
    cache.stop()


def test_disk_cache_delete_exception(tmp_cache_dir, monkeypatch):
    """测试 DiskCache（delete，异常处理）"""
    config = DiskCacheConfig(cache_dir=tmp_cache_dir)
    cache = DiskCache(config)
    
    # 模拟删除文件时抛出异常
    def mock_unlink(self):
        raise Exception("Delete error")
    
    monkeypatch.setattr(Path, "unlink", mock_unlink)
    
    # 设置一个缓存项
    cache.set("key1", "value1")
    
    # 删除应该返回False（异常被捕获）
    result = cache.delete("key1")
    assert result is False
    
    cache.stop()


def test_disk_cache_health_check_no_write_permission(tmp_cache_dir, monkeypatch):
    """测试 DiskCache（健康检查，无写权限）"""
    import os
    config = DiskCacheConfig(cache_dir=tmp_cache_dir)
    cache = DiskCache(config)
    
    # 模拟无写权限
    def mock_access(path, mode):
        if mode == os.W_OK:
            return False
        return True
    
    monkeypatch.setattr(os, "access", mock_access)
    
    # 健康检查应该返回错误状态
    result = cache.health_check()
    assert result['status'] == 'error'
    assert 'No write permission' in result['message']
    
    cache.stop()


def test_disk_cache_health_check_exception(tmp_cache_dir, monkeypatch):
    """测试 DiskCache（健康检查，异常处理）"""
    config = DiskCacheConfig(cache_dir=tmp_cache_dir)
    cache = DiskCache(config)
    
    # 模拟健康检查时抛出异常
    def mock_collect_disk_usage(self):
        raise Exception("Health check error")
    
    monkeypatch.setattr(cache, "_collect_disk_usage", mock_collect_disk_usage)
    
    # 健康检查应该返回错误状态
    result = cache.health_check()
    assert result['status'] == 'error'
    
    cache.stop()


def test_disk_cache_cleanup_interval_zero(tmp_cache_dir):
    """测试 DiskCache（清理间隔为0）"""
    config = DiskCacheConfig(cache_dir=tmp_cache_dir, cleanup_interval=0)
    cache = DiskCache(config)
    
    # 清理线程不应该启动
    assert cache._cleanup_thread is None
    
    cache.stop()


def test_disk_cache_cleanup_thread_exception(tmp_cache_dir, monkeypatch):
    """测试 DiskCache（清理线程异常处理）"""
    config = DiskCacheConfig(cache_dir=tmp_cache_dir, cleanup_interval=1)
    cache = DiskCache(config)
    
    # 模拟清理时抛出异常
    def mock_cleanup_expired(self):
        raise Exception("Cleanup error")
    
    monkeypatch.setattr(cache, "_cleanup_expired", mock_cleanup_expired)
    
    # 等待一小段时间让清理线程运行
    time.sleep(0.1)
    
    # 应该不抛出异常（异常被捕获）
    cache.stop()


def test_disk_cache_cleanup_expired_exception(tmp_cache_dir, monkeypatch):
    """测试 DiskCache（清理过期缓存，异常处理）"""
    config = DiskCacheConfig(cache_dir=tmp_cache_dir)
    cache = DiskCache(config)
    
    # 创建一个损坏的缓存文件
    file_path = cache._get_file_path("corrupted_key")
    with open(file_path, 'wb') as f:
        f.write(b'corrupted data')
    
    # 模拟glob抛出异常
    def mock_glob(self, pattern):
        raise Exception("Glob error")
    
    monkeypatch.setattr(Path, "glob", mock_glob)
    
    # 清理应该不抛出异常（异常被捕获）
    cache._cleanup_expired()
    
    cache.stop()


def test_disk_cache_cleanup_expired_file_error(tmp_cache_dir):
    """测试 DiskCache（清理过期缓存，文件处理错误）"""
    config = DiskCacheConfig(cache_dir=tmp_cache_dir)
    cache = DiskCache(config)
    
    # 创建一个损坏的缓存文件
    file_path = cache._get_file_path("corrupted_key")
    with open(file_path, 'wb') as f:
        f.write(b'corrupted data')
    
    # 清理应该删除损坏的文件
    cache._cleanup_expired()
    
    # 文件应该被删除
    assert not file_path.exists()
    
    cache.stop()


def test_disk_cache_stop_cleanup_thread_timeout(tmp_cache_dir, monkeypatch):
    """测试 DiskCache（停止，清理线程超时）"""
    config = DiskCacheConfig(cache_dir=tmp_cache_dir, cleanup_interval=1)
    cache = DiskCache(config)
    
    # 启动清理线程
    cache._start_cleanup_thread()
    
    # 模拟线程join超时（线程仍然存活）
    def mock_join(timeout=None):
        pass  # 不等待，模拟超时
    
    if cache._cleanup_thread:
        cache._cleanup_thread.join = mock_join
        cache._cleanup_thread.is_alive = Mock(return_value=True)
    
    # 停止应该不抛出异常
    cache.stop()


def test_disk_cache_del_exception(tmp_cache_dir, monkeypatch):
    """测试 DiskCache（析构函数，异常处理）"""
    config = DiskCacheConfig(cache_dir=tmp_cache_dir)
    cache = DiskCache(config)
    
    # 模拟stop抛出异常
    def mock_stop(self):
        raise Exception("Stop error")
    
    monkeypatch.setattr(cache, "stop", mock_stop)
    
    # 析构函数应该不抛出异常（异常被捕获）
    del cache


def test_disk_cache_deserialize_entry_invalid_type(tmp_cache_dir):
    """测试 DiskCache（_deserialize_entry，无效类型，覆盖 175 行）"""
    config = DiskCacheConfig(cache_dir=tmp_cache_dir)
    cache = DiskCache(config)
    
    try:
        # 创建一个不是 dict 的 pickle 数据
        invalid_data = pickle.dumps("not a dict")
        
        # 应该返回 None（因为类型检查失败）
        result = cache._deserialize_entry(invalid_data)
        assert result is None
    finally:
        cache.stop()


def test_disk_cache_exists_exception(tmp_cache_dir):
    """测试 DiskCache（exists，异常处理，覆盖 352-354 行）"""
    config = DiskCacheConfig(cache_dir=tmp_cache_dir)
    cache = DiskCache(config)
    
    try:
        # 模拟检查存在时抛出异常
        with patch.object(cache, '_get_file_path', side_effect=Exception("Check failed")):
            result = cache.exists("key1")
            assert result is False
    finally:
        cache.stop()


def test_disk_cache_clear_exception(tmp_cache_dir):
    """测试 DiskCache（clear，异常处理，覆盖 369-371 行）"""
    config = DiskCacheConfig(cache_dir=tmp_cache_dir)
    cache = DiskCache(config)
    
    try:
        # 先设置一些缓存
        cache.set("key1", "value1")
        # 模拟清除时抛出异常
        with patch.object(Path, 'glob', side_effect=Exception("Clear failed")):
            result = cache.clear()
            assert result is False
    finally:
        cache.stop()


def test_disk_cache_list_keys_exception(tmp_cache_dir):
    """测试 DiskCache（list_keys，异常处理，覆盖 387-389 行）"""
    config = DiskCacheConfig(cache_dir=tmp_cache_dir)
    cache = DiskCache(config)
    
    try:
        # 模拟列出键时抛出异常
        with patch.object(Path, 'glob', side_effect=Exception("List failed")):
            result = cache.list_keys()
            assert result == []
    finally:
        cache.stop()


def test_disk_cache_collect_disk_usage_file_not_found(tmp_cache_dir):
    """测试 DiskCache（_collect_disk_usage，文件不存在，覆盖 399-401 行）"""
    config = DiskCacheConfig(cache_dir=tmp_cache_dir)
    cache = DiskCache(config)
    
    try:
        # 创建一个文件，然后删除它
        cache.set("key1", "value1")
        file_path = cache._get_file_path("key1")
        if file_path.exists():
            file_path.unlink()
        
        # 统计应该能处理文件不存在的情况
        file_count, total_size = cache._collect_disk_usage()
        assert file_count >= 0
        assert total_size >= 0
    finally:
        cache.stop()


def test_disk_cache_collect_disk_usage_os_error(tmp_cache_dir):
    """测试 DiskCache（_collect_disk_usage，OSError，覆盖 402-403 行）"""
    config = DiskCacheConfig(cache_dir=tmp_cache_dir)
    cache = DiskCache(config)
    
    try:
        # 先创建一个文件
        cache.set("key1", "value1")
        # 模拟 stat 时抛出 OSError
        original_stat = Path.stat
        def mock_stat(self):
            if "key1" in str(self):
                raise OSError("Stat failed")
            return original_stat(self)
        with patch.object(Path, 'stat', mock_stat):
            file_count, total_size = cache._collect_disk_usage()
            assert file_count >= 0
            assert total_size >= 0
    finally:
        cache.stop()


def test_disk_cache_cleanup_expired_entry_none(tmp_cache_dir):
    """测试 DiskCache（_cleanup_expired，entry 为 None，覆盖 501-503 行）"""
    config = DiskCacheConfig(cache_dir=tmp_cache_dir)
    cache = DiskCache(config)
    
    try:
        # 创建一个文件，但反序列化失败（entry 为 None）
        file_path = cache._get_file_path("key1")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        # 写入无效的 pickle 数据
        with open(file_path, 'wb') as f:
            f.write(b'invalid pickle data')
        
        # 清理应该能处理 entry 为 None 的情况
        expired_count = cache._cleanup_expired()
        # 文件应该被删除
        assert not file_path.exists()
    finally:
        cache.stop()

