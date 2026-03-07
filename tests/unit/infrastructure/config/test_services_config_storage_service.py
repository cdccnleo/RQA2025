from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Any, Dict, Optional, List

from src.infrastructure.config.services.config_storage_service import ConfigStorageService
from src.infrastructure.config.storage.types.iconfigstorage import IConfigStorage


class TestConfigStorageService:
    """测试配置存储服务"""

    @pytest.fixture
    def mock_storage_backend(self):
        """模拟存储后端"""
        return MagicMock()

    @pytest.fixture
    def storage_service(self, mock_storage_backend):
        """配置存储服务实例"""
        return ConfigStorageService(storage_backend=mock_storage_backend)

    @pytest.fixture
    def storage_service_no_cache(self, mock_storage_backend):
        """无缓存的配置存储服务"""
        return ConfigStorageService(storage_backend=mock_storage_backend, cache_enabled=False)

    def test_initialization(self, mock_storage_backend):
        """测试初始化"""
        service = ConfigStorageService(storage_backend=mock_storage_backend)

        assert service._storage_backend == mock_storage_backend
        assert service._cache_enabled is True
        assert service._cache_size == 1000
        assert service._cache == {}
        assert service._cache_timestamps == {}
        assert service._cache_access_times == {}
        assert service._stats == {
            'loads': 0,
            'saves': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0,
            'total_operations': 0,
            'cache_size': 0
        }

    def test_initialization_no_cache(self):
        """测试无缓存初始化"""
        service = ConfigStorageService(cache_enabled=False)

        assert service._cache_enabled is False
        assert service._storage_backend is None

    def test_set_storage_backend(self, storage_service, mock_storage_backend):
        """测试设置存储后端"""
        new_backend = Mock(spec=IConfigStorage)
        storage_service.set_storage_backend(new_backend)

        assert storage_service._storage_backend == new_backend

    def test_load_with_cache_hit(self, storage_service, mock_storage_backend):
        """测试缓存命中时的加载"""
        source = "test_config.json"
        cached_data = {"key": "cached_value"}
        cache_time = 1000000.0  # 过去的时间

        # 设置缓存
        storage_service._cache[source] = cached_data
        storage_service._cache_timestamps[source] = cache_time

        with patch('time.time', return_value=cache_time + 60):  # 1分钟后，仍在缓存有效期内
            result = storage_service.load(source)

        assert result == cached_data
        assert storage_service._stats['cache_hits'] == 1
        # 确保没有调用存储后端
        mock_storage_backend.load.assert_not_called()

    def test_load_with_cache_miss(self, storage_service, mock_storage_backend):
        """测试缓存未命中时的加载"""
        source = "test_config.json"
        loaded_data = {"key": "loaded_value"}

        mock_storage_backend.load.return_value = loaded_data

        result = storage_service.load(source)

        assert result == loaded_data
        assert storage_service._stats['loads'] == 1
        assert source in storage_service._cache
        assert storage_service._cache[source] == loaded_data
        mock_storage_backend.load.assert_called_once_with(source)

    def test_load_with_expired_cache(self, storage_service, mock_storage_backend):
        """测试缓存过期时的加载"""
        source = "test_config.json"
        cached_data = {"key": "cached_value"}
        loaded_data = {"key": "loaded_value"}
        cache_time = 1000000.0

        # 设置过期缓存
        storage_service._cache[source] = cached_data
        storage_service._cache_timestamps[source] = cache_time

        mock_storage_backend.load.return_value = loaded_data

        with patch('time.time', return_value=cache_time + 400):  # 过期时间
            result = storage_service.load(source)

        assert result == loaded_data
        assert storage_service._stats['loads'] == 1
        mock_storage_backend.load.assert_called_once_with(source)

    def test_load_without_storage_backend(self, storage_service):
        """测试没有存储后端时的加载"""
        storage_service._storage_backend = None

        with pytest.raises(ValueError, match="未设置存储后端"):
            storage_service.load("test.json")

    def test_save_success(self, storage_service, mock_storage_backend):
        """测试保存成功"""
        config = {"key": "value"}
        target = "test_config.json"

        mock_storage_backend.save.return_value = True

        result = storage_service.save(config, target)

        assert result is True
        assert storage_service._stats['saves'] == 1
        assert target in storage_service._cache
        assert storage_service._cache[target] == config
        mock_storage_backend.save.assert_called_once_with(config, target)

    def test_save_without_storage_backend(self, storage_service):
        """测试没有存储后端时的保存"""
        storage_service._storage_backend = None

        with pytest.raises(ValueError, match="未设置存储后端"):
            storage_service.save({"key": "value"}, "test.json")

    def test_reload_success(self, storage_service, mock_storage_backend):
        """测试重新加载成功"""
        # 设置一些缓存数据
        storage_service._cache["key1"] = "value1"
        storage_service._cache_timestamps["key1"] = 1000.0

        mock_storage_backend.reload.return_value = True

        result = storage_service.reload()

        assert result is True
        # 缓存应该被清空
        assert len(storage_service._cache) == 0
        assert len(storage_service._cache_timestamps) == 0
        assert len(storage_service._cache_access_times) == 0
        mock_storage_backend.reload.assert_called_once()

    def test_reload_without_reload_method(self, storage_service):
        """测试重新加载（存储后端没有reload方法）"""
        # 移除reload方法
        del storage_service._storage_backend.reload

        result = storage_service.reload()

        assert result is True

    def test_get_cache_stats_enabled(self, storage_service):
        """测试获取缓存统计（启用缓存）"""
        # 设置一些缓存数据
        storage_service._cache["key1"] = "value1"
        storage_service._cache["key2"] = "value2"
        storage_service._stats['cache_hits'] = 10
        storage_service._stats['cache_misses'] = 5

        stats = storage_service.get_cache_stats()

        assert stats["enabled"] is True
        assert stats["size"] == 2
        assert stats["max_size"] == 1000
        assert stats["hit_rate"] == 10 / 15  # 10 hits / 15 total
        assert "key1" in stats["entries"]
        assert "key2" in stats["entries"]

    def test_get_cache_stats_disabled(self, storage_service_no_cache):
        """测试获取缓存统计（禁用缓存）"""
        stats = storage_service_no_cache.get_cache_stats()

        assert stats == {"enabled": False}

    def test_get_storage_stats_enabled(self, storage_service):
        """测试获取存储统计（启用缓存）"""
        # 设置一些统计数据
        storage_service._cache["key1"] = "value1"
        storage_service._stats.update({
            'loads': 10,
            'saves': 5,
            'cache_hits': 8,
            'cache_misses': 2,
            'errors': 1,
            'total_operations': 20
        })

        stats = storage_service.get_storage_stats()

        assert stats["enabled"] is True
        assert stats["size"] == 1
        assert stats["loads"] == 10
        assert stats["saves"] == 5
        assert stats["cache_hits"] == 8
        assert stats["cache_misses"] == 2
        assert stats["errors"] == 1
        assert stats["total_operations"] == 20
        assert stats["cache_size"] == 1

    def test_get_storage_stats_disabled(self, storage_service_no_cache):
        """测试获取存储统计（禁用缓存）"""
        stats = storage_service_no_cache.get_storage_stats()

        assert stats == {"enabled": False}

    def test_update_cache_add(self, storage_service):
        """测试添加缓存"""
        key = "test_key"
        value = "test_value"

        storage_service._update_cache(key, value)

        assert storage_service._cache[key] == value
        assert key in storage_service._cache_timestamps
        assert key in storage_service._cache_access_times

    def test_update_cache_remove(self, storage_service):
        """测试删除缓存"""
        key = "test_key"
        storage_service._cache[key] = "value"
        storage_service._cache_timestamps[key] = 1000.0
        storage_service._cache_access_times[key] = 1000.0

        storage_service._update_cache(key, None)

        assert key not in storage_service._cache
        assert key not in storage_service._cache_timestamps
        assert key not in storage_service._cache_access_times

    def test_update_cache_size_limit(self, storage_service):
        """测试缓存大小限制"""
        storage_service._cache_size = 2

        # 添加超过限制的条目
        storage_service._update_cache("key1", "value1")
        storage_service._update_cache("key2", "value2")
        storage_service._update_cache("key3", "value3")  # 应该触发LRU淘汰

        assert len(storage_service._cache) == 2
        # 应该保留最新的两个
        assert "key2" in storage_service._cache
        assert "key3" in storage_service._cache

    def test_update_cache_access(self, storage_service):
        """测试更新缓存访问时间"""
        key = "test_key"
        storage_service._cache[key] = "value"

        with patch('time.time', return_value=123456.789):
            storage_service._update_cache_access(key)

        assert storage_service._cache_access_times[key] == 123456.789

    def test_evict_cache_entry(self, storage_service):
        """测试缓存条目淘汰（LRU）"""
        # 设置不同的访问时间
        storage_service._cache = {"key1": "value1", "key2": "value2", "key3": "value3"}
        storage_service._cache_timestamps = {"key1": 1000.0, "key2": 1000.0, "key3": 1000.0}
        storage_service._cache_access_times = {
            "key1": 1000.0,  # 最少使用
            "key2": 2000.0,
            "key3": 3000.0   # 最近使用
        }

        storage_service._evict_cache_entry()

        # key1 应该被淘汰
        assert "key1" not in storage_service._cache
        assert "key1" not in storage_service._cache_timestamps
        assert "key1" not in storage_service._cache_access_times
        assert len(storage_service._cache) == 2

    def test_calculate_hit_rate(self, storage_service):
        """测试计算命中率"""
        storage_service._stats['cache_hits'] = 8
        storage_service._stats['cache_misses'] = 2

        hit_rate = storage_service._calculate_hit_rate()
        assert hit_rate == 8 / 10  # 8 hits / 10 total

    def test_calculate_hit_rate_no_requests(self, storage_service):
        """测试计算命中率（没有请求）"""
        hit_rate = storage_service._calculate_hit_rate()
        assert hit_rate == 0.0

    def test_cleanup(self, storage_service):
        """测试清理"""
        # 设置一些数据
        storage_service._cache["key1"] = "value1"
        storage_service._cache_timestamps["key1"] = 1000.0
        storage_service._cache_access_times["key1"] = 1000.0

        storage_service.cleanup()

        assert len(storage_service._cache) == 0
        assert len(storage_service._cache_timestamps) == 0
        assert len(storage_service._cache_access_times) == 0

    def test_get_with_cache_hit(self, storage_service, mock_storage_backend):
        """测试获取配置值（缓存命中）"""
        key = "test_key"
        cached_value = "cached_value"
        cache_time = 1000000.0

        storage_service._cache[key] = cached_value
        storage_service._cache_timestamps[key] = cache_time

        with patch('time.time', return_value=cache_time + 60):
            result = storage_service.get(key)

        assert result == cached_value
        assert storage_service._stats['cache_hits'] == 1
        mock_storage_backend.get.assert_not_called()

    def test_get_with_cache_miss(self, storage_service, mock_storage_backend):
        """测试获取配置值（缓存未命中）"""
        key = "test_key"
        backend_value = "backend_value"

        mock_storage_backend.get.return_value = backend_value

        result = storage_service.get(key)

        assert result == backend_value
        assert storage_service._stats['cache_misses'] == 1
        assert storage_service._cache[key] == backend_value
        mock_storage_backend.get.assert_called_once_with(key, None)

    def test_get_without_storage_backend(self, storage_service):
        """测试获取配置值（无存储后端）"""
        storage_service._storage_backend = None

        result = storage_service.get("test_key", "default")

        assert result == "default"
        assert storage_service._stats['errors'] == 1

    def test_set_success(self, storage_service, mock_storage_backend):
        """测试设置配置值成功"""
        key = "test_key"
        value = "test_value"

        mock_storage_backend.set.return_value = True

        result = storage_service.set(key, value)

        assert result is True
        assert storage_service._stats['saves'] == 1
        assert storage_service._cache[key] == value
        mock_storage_backend.set.assert_called_once_with(key, value)

    def test_set_without_storage_backend(self, storage_service):
        """测试设置配置值（无存储后端）"""
        storage_service._storage_backend = None

        result = storage_service.set("test_key", "test_value")

        assert result is False
        assert storage_service._stats['errors'] == 1

    def test_delete_success(self, storage_service, mock_storage_backend):
        """测试删除配置成功"""
        key = "test_key"

        # 设置缓存
        storage_service._cache[key] = "value"
        storage_service._cache_timestamps[key] = 1000.0
        storage_service._cache_access_times[key] = 1000.0

        mock_storage_backend.delete.return_value = True

        result = storage_service.delete(key)

        assert result is True
        assert storage_service._stats['saves'] == 1
        assert key not in storage_service._cache
        assert key not in storage_service._cache_timestamps
        assert key not in storage_service._cache_access_times
        mock_storage_backend.delete.assert_called_once_with(key)

    def test_delete_without_storage_backend(self, storage_service):
        """测试删除配置（无存储后端）"""
        storage_service._storage_backend = None

        result = storage_service.delete("test_key")

        assert result is False
        assert storage_service._stats['errors'] == 1

    def test_exists_with_cache_hit(self, storage_service, mock_storage_backend):
        """测试检查存在性（缓存命中）"""
        key = "test_key"
        storage_service._cache[key] = "value"

        result = storage_service.exists(key)

        assert result is True
        mock_storage_backend.exists.assert_not_called()

    def test_exists_with_cache_miss(self, storage_service, mock_storage_backend):
        """测试检查存在性（缓存未命中）"""
        key = "test_key"
        mock_storage_backend.exists.return_value = True

        result = storage_service.exists(key)

        assert result is True
        mock_storage_backend.exists.assert_called_once_with(key)

    def test_exists_without_storage_backend(self, storage_service):
        """测试检查存在性（无存储后端）"""
        storage_service._storage_backend = None

        result = storage_service.exists("test_key")

        assert result is False
        assert storage_service._stats['errors'] == 1

    def test_keys_success(self, storage_service, mock_storage_backend):
        """测试获取键列表成功"""
        pattern = "test_*"
        expected_keys = ["test_key1", "test_key2"]

        mock_storage_backend.keys.return_value = expected_keys

        result = storage_service.keys(pattern)

        assert result == expected_keys
        mock_storage_backend.keys.assert_called_once_with(pattern)

    def test_keys_without_storage_backend(self, storage_service):
        """测试获取键列表（无存储后端）"""
        storage_service._storage_backend = None

        result = storage_service.keys("pattern")

        assert result == []
        assert storage_service._stats['errors'] == 1

    def test_clear_success(self, storage_service, mock_storage_backend):
        """测试清空配置成功"""
        # 设置缓存
        storage_service._cache["key1"] = "value1"
        storage_service._cache_timestamps["key1"] = 1000.0
        storage_service._cache_access_times["key1"] = 1000.0

        mock_storage_backend.clear.return_value = True

        result = storage_service.clear()

        assert result is True
        # 缓存应该被清空
        assert len(storage_service._cache) == 0
        assert len(storage_service._cache_timestamps) == 0
        assert len(storage_service._cache_access_times) == 0
        mock_storage_backend.clear.assert_called_once()

    def test_clear_without_storage_backend(self, storage_service):
        """测试清空配置（无存储后端）"""
        storage_service._storage_backend = None

        result = storage_service.clear()

        assert result is False
        assert storage_service._stats['errors'] == 1

    @pytest.mark.parametrize("operation,expected_stats_key", [
        ('load', 'loads'),
        ('save', 'saves'),
        ('get', None),  # get 操作不直接增加统计
        ('set', 'saves'),
        ('delete', 'saves'),
        ('exists', None),  # exists 操作不增加统计
        ('keys', None),  # keys 操作不增加统计
        ('clear', None),  # clear 操作不增加统计
    ])
    def test_operation_error_handling(self, storage_service, mock_storage_backend, operation, expected_stats_key):
        """测试操作错误处理"""
        mock_storage_backend.load.side_effect = Exception("Backend error")
        mock_storage_backend.save.side_effect = Exception("Backend error")
        mock_storage_backend.get.side_effect = Exception("Backend error")
        mock_storage_backend.set.side_effect = Exception("Backend error")
        mock_storage_backend.delete.side_effect = Exception("Backend error")
        mock_storage_backend.exists.side_effect = Exception("Backend error")
        mock_storage_backend.keys.side_effect = Exception("Backend error")
        mock_storage_backend.clear.side_effect = Exception("Backend error")

        with pytest.raises(Exception):
            if operation == 'load':
                storage_service.load("test.json")
            elif operation == 'save':
                storage_service.save({}, "test.json")
            elif operation == 'get':
                storage_service.get("test_key")
            elif operation == 'set':
                storage_service.set("test_key", "value")
            elif operation == 'delete':
                storage_service.delete("test_key")
            elif operation == 'exists':
                storage_service.exists("test_key")
            elif operation == 'keys':
                storage_service.keys()
            elif operation == 'clear':
                storage_service.clear()

        # 检查错误统计是否增加
        if expected_stats_key:
            assert storage_service._stats['errors'] >= 1
        else:
            # 对于不直接增加统计的操作，至少错误统计应该增加
            assert storage_service._stats['errors'] >= 1

    def test_load_storage_backend_exception(self, storage_service, mock_storage_backend):
        """测试加载时存储后端抛出异常"""
        mock_storage_backend.load.side_effect = ConnectionError("Connection failed")

        with pytest.raises(ConnectionError):
            storage_service.load("test_source")

        assert storage_service._stats["errors"] == 1

    def test_save_storage_backend_exception(self, storage_service, mock_storage_backend):
        """测试保存时存储后端抛出异常"""
        mock_storage_backend.save.side_effect = PermissionError("Permission denied")

        with pytest.raises(PermissionError):
            storage_service.save({"key": "value"}, "test_target")

        assert storage_service._stats["errors"] == 1

    def test_get_storage_backend_exception(self, storage_service, mock_storage_backend):
        """测试获取时存储后端抛出异常"""
        mock_storage_backend.get.side_effect = KeyError("Key not found")

        with pytest.raises(KeyError):
            storage_service.get("nonexistent_key")

        assert storage_service._stats["errors"] == 1

    def test_set_storage_backend_exception(self, storage_service, mock_storage_backend):
        """测试设置时存储后端抛出异常"""
        mock_storage_backend.set.side_effect = ValueError("Invalid value")

        with pytest.raises(ValueError):
            storage_service.set("test_key", "invalid_value")

        assert storage_service._stats["errors"] == 1

    def test_delete_storage_backend_exception(self, storage_service, mock_storage_backend):
        """测试删除时存储后端抛出异常"""
        mock_storage_backend.delete.side_effect = OSError("Delete failed")

        with pytest.raises(OSError):
            storage_service.delete("test_key")

        assert storage_service._stats["errors"] == 1

    def test_exists_storage_backend_exception(self, storage_service, mock_storage_backend):
        """测试存在检查时存储后端抛出异常"""
        mock_storage_backend.exists.side_effect = RuntimeError("Check failed")

        with pytest.raises(RuntimeError):
            storage_service.exists("test_key")

        assert storage_service._stats["errors"] == 1

    def test_keys_storage_backend_exception(self, storage_service, mock_storage_backend):
        """测试键列表时存储后端抛出异常"""
        mock_storage_backend.keys.side_effect = Exception("Keys retrieval failed")

        with pytest.raises(Exception):
            storage_service.keys()

        assert storage_service._stats["errors"] == 1

    def test_clear_storage_backend_exception(self, storage_service, mock_storage_backend):
        """测试清除时存储后端抛出异常"""
        mock_storage_backend.clear.side_effect = Exception("Clear failed")

        with pytest.raises(Exception):
            storage_service.clear()

        assert storage_service._stats["errors"] == 1

    def test_cache_eviction_under_pressure(self, storage_service):
        """测试缓存压力下的逐出"""
        # 填充缓存到极限
        for i in range(storage_service._cache_size + 10):
            storage_service._update_cache(f"key_{i}", f"value_{i}")

        # 验证缓存大小被控制
        assert len(storage_service._cache) <= storage_service._cache_size

        # 验证访问时间被更新
        assert len(storage_service._cache_access_times) <= storage_service._cache_size

    def test_concurrent_cache_access(self, storage_service):
        """测试并发缓存访问"""
        import threading
        import time

        results = []
        errors = []

        def cache_operation(operation_id):
            try:
                if operation_id % 2 == 0:
                    storage_service._update_cache(f"key_{operation_id}", f"value_{operation_id}")
                    results.append(f"updated_{operation_id}")
                else:
                    value = storage_service._cache.get(f"key_{operation_id}")
                    results.append(f"read_{operation_id}_{value}")
            except Exception as e:
                errors.append(str(e))

        # 创建多个线程并发访问
        threads = []
        for i in range(10):
            thread = threading.Thread(target=cache_operation, args=(i,))
            threads.append(thread)

        # 启动所有线程
        for thread in threads:
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证没有错误发生
        assert len(errors) == 0
        # 验证操作都执行了
        assert len(results) == 10

    def test_cache_timestamp_management(self, storage_service):
        """测试缓存时间戳管理"""
        import time

        # 添加缓存项
        storage_service._update_cache("test_key", "test_value")

        # 验证时间戳被设置
        assert "test_key" in storage_service._cache_timestamps
        assert storage_service._cache_timestamps["test_key"] > 0

        # 记录初始时间戳
        initial_timestamp = storage_service._cache_timestamps["test_key"]

        # 等待一小段时间
        time.sleep(0.01)

        # 更新缓存
        storage_service._update_cache("test_key", "updated_value")

        # 验证时间戳被更新
        assert storage_service._cache_timestamps["test_key"] > initial_timestamp

    def test_cache_access_pattern_tracking(self, storage_service):
        """测试缓存访问模式跟踪"""
        import time

        # 添加多个缓存项
        for i in range(5):
            storage_service._update_cache(f"key_{i}", f"value_{i}")

        # 访问其中一些项
        storage_service._update_cache_access("key_1")
        time.sleep(0.001)
        storage_service._update_cache_access("key_2")
        time.sleep(0.001)
        storage_service._update_cache_access("key_1")  # 再次访问key_1

        # 验证访问时间被记录
        assert "key_1" in storage_service._cache_access_times
        assert "key_2" in storage_service._cache_access_times

        # key_1应该有更新的访问时间（因为被访问了两次）
        assert storage_service._cache_access_times["key_1"] > storage_service._cache_access_times["key_2"]

    def test_memory_cache_creation(self, storage_service):
        """测试内存缓存创建"""
        cache = storage_service._create_memory_cache()

        assert isinstance(cache, dict)
        assert len(cache) == 0

        # 验证可以正常使用
        cache["test_key"] = "test_value"
        assert cache["test_key"] == "test_value"

    def test_service_initialization_states(self, storage_service):
        """测试服务初始化状态"""
        # 验证初始状态
        assert hasattr(storage_service, '_lock')
        assert hasattr(storage_service, '_stats')
        assert storage_service._stats['total_operations'] == 0

        # 执行一些操作
        storage_service._ensure_initialized()

        # 验证初始化后状态
        assert hasattr(storage_service, '_start_time')
        assert storage_service._start_time > 0