from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Any, Dict, Optional
import threading
import time

from src.infrastructure.config.services.cache_service import CacheService


class TestCacheService:
    """测试缓存服务"""

    @pytest.fixture
    def cache_service(self):
        """缓存服务实例"""
        return CacheService()

    @pytest.fixture
    def cache_service_with_config(self):
        """带配置的缓存服务实例"""
        config = {"cache_size": 500}
        return CacheService(config=config, maxsize=500)

    def test_initialization_default(self):
        """测试默认初始化"""
        service = CacheService()
        
        assert service.config == {}
        assert service.maxsize == 1000
        assert service.initialized is False
        assert service.cache == {}
        assert service.timestamps == {}
        assert service.access_times == {}
        assert service.hits == 0
        assert service.misses == 0
        assert hasattr(service.lock, 'acquire')  # 检查是否有RLock的方法

    def test_initialization_with_config(self):
        """测试带配置初始化"""
        config = {"test": "value"}
        service = CacheService(config=config, maxsize=500)
        
        assert service.config == config
        assert service.maxsize == 500

    def test_initialize_success(self, cache_service):
        """测试初始化成功"""
        result = cache_service.initialize()
        
        assert result is True
        assert cache_service.initialized is True

    def test_initialize_exception(self, cache_service):
        """测试初始化异常"""
        with patch('src.infrastructure.config.services.cache_service.logger') as mock_logger:
            # 模拟logger.info抛出异常
            mock_logger.info.side_effect = Exception("Test error")
            result = cache_service.initialize()
            
            # 由于异常被捕获，应该返回False
            assert result is False

    def test_shutdown_success(self, cache_service):
        """测试关闭成功"""
        cache_service.initialize()
        cache_service.cache['key1'] = 'value1'
        
        result = cache_service.shutdown()
        
        assert result is True
        assert cache_service.initialized is False
        assert len(cache_service.cache) == 0

    def test_shutdown_exception(self, cache_service):
        """测试关闭异常"""
        cache_service.initialize()
        
        with patch('src.infrastructure.config.services.cache_service.logger') as mock_logger:
            # 模拟logger.info抛出异常
            mock_logger.info.side_effect = Exception("Test error")
            result = cache_service.shutdown()
            
            assert result is False

    def test_get_not_initialized(self, cache_service):
        """测试未初始化时获取"""
        result = cache_service.get("key")
        assert result is None

    def test_get_hit(self, cache_service):
        """测试缓存命中"""
        cache_service.initialize()
        cache_service.cache['key1'] = 'value1'
        cache_service.access_times['key1'] = time.time()
        cache_service.timestamps['key1'] = time.time() + 3600  # 1小时后过期
        
        result = cache_service.get('key1')
        
        assert result == 'value1'
        assert cache_service.hits == 1
        assert cache_service.misses == 0

    def test_get_miss(self, cache_service):
        """测试缓存未命中"""
        cache_service.initialize()
        
        result = cache_service.get('nonexistent_key')
        
        assert result is None
        assert cache_service.hits == 0
        assert cache_service.misses == 1

    def test_get_expired(self, cache_service):
        """测试获取过期项"""
        cache_service.initialize()
        cache_service.cache['key1'] = 'value1'
        cache_service.access_times['key1'] = time.time() - 100
        cache_service.timestamps['key1'] = time.time() - 60  # 已过期
        
        result = cache_service.get('key1')
        
        assert result is None
        assert cache_service.misses == 1
        assert 'key1' not in cache_service.cache

    def test_set_not_initialized(self, cache_service):
        """测试未初始化时设置"""
        result = cache_service.set('key', 'value')
        assert result is False

    def test_set_success(self, cache_service):
        """测试设置成功"""
        cache_service.initialize()
        
        result = cache_service.set('key1', 'value1', ttl=1800)
        
        assert result is True
        assert cache_service.cache['key1'] == 'value1'
        assert 'key1' in cache_service.timestamps
        assert 'key1' in cache_service.access_times

    def test_set_capacity_exceeded(self, cache_service_with_config):
        """测试容量超限时的驱逐"""
        cache_service_with_config.initialize()
        
        # 填满缓存
        for i in range(500):
            cache_service_with_config.set(f'key{i}', f'value{i}')
        
        # 设置一个时间戳，确保LRU驱逐
        cache_service_with_config.access_times['key0'] = time.time() - 100  # 最旧的
        
        # 添加新项，应该驱逐最旧的
        result = cache_service_with_config.set('new_key', 'new_value')
        
        assert result is True
        assert len(cache_service_with_config.cache) == 500
        assert 'new_key' in cache_service_with_config.cache
        assert 'key0' not in cache_service_with_config.cache

    def test_delete_not_initialized(self, cache_service):
        """测试未初始化时删除"""
        result = cache_service.delete('key')
        assert result is False

    def test_delete_success(self, cache_service):
        """测试删除成功"""
        cache_service.initialize()
        cache_service.cache['key1'] = 'value1'
        cache_service.timestamps['key1'] = time.time()
        cache_service.access_times['key1'] = time.time()
        
        result = cache_service.delete('key1')
        
        assert result is True
        assert 'key1' not in cache_service.cache
        assert 'key1' not in cache_service.timestamps
        assert 'key1' not in cache_service.access_times

    def test_delete_nonexistent(self, cache_service):
        """测试删除不存在的项"""
        cache_service.initialize()
        
        result = cache_service.delete('nonexistent_key')
        
        assert result is False

    def test_clear_not_initialized(self, cache_service):
        """测试未初始化时清空"""
        with pytest.raises(RuntimeError, match="缓存服务未初始化"):
            cache_service.clear()

    def test_clear_success(self, cache_service):
        """测试清空成功"""
        cache_service.initialize()
        cache_service.cache['key1'] = 'value1'
        cache_service.cache['key2'] = 'value2'
        cache_service.hits = 10
        cache_service.misses = 5
        
        result = cache_service.clear()
        
        assert result is True
        assert len(cache_service.cache) == 0
        assert len(cache_service.timestamps) == 0
        assert len(cache_service.access_times) == 0
        assert cache_service.hits == 0
        assert cache_service.misses == 0

    def test_get_stats_not_initialized(self, cache_service):
        """测试未初始化时获取统计"""
        with pytest.raises(RuntimeError, match="缓存服务未初始化"):
            cache_service.get_stats()

    def test_get_stats_success(self, cache_service):
        """测试获取统计成功"""
        cache_service.initialize()
        cache_service.cache['key1'] = 'value1'
        cache_service.hits = 8
        cache_service.misses = 2
        
        stats = cache_service.get_stats()
        
        assert stats['size'] == 1
        assert stats['maxsize'] == 1000
        assert stats['hits'] == 8
        assert stats['misses'] == 2
        assert stats['hit_rate'] == 80.0  # 8/10 * 100
        assert stats['total_requests'] == 10
        assert stats['initialized'] is True

    def test_get_stats_no_requests(self, cache_service):
        """测试无请求时的统计"""
        cache_service.initialize()
        
        stats = cache_service.get_stats()
        
        assert stats['hit_rate'] == 0.0
        assert stats['total_requests'] == 0

    def test_is_expired_true(self, cache_service):
        """测试过期检查 - 已过期"""
        cache_service.timestamps['key1'] = time.time() - 100
        
        result = cache_service._is_expired('key1')
        
        assert result is True

    def test_is_expired_false(self, cache_service):
        """测试过期检查 - 未过期"""
        cache_service.timestamps['key1'] = time.time() + 3600
        
        result = cache_service._is_expired('key1')
        
        assert result is False

    def test_is_expired_no_timestamp(self, cache_service):
        """测试过期检查 - 无时间戳"""
        result = cache_service._is_expired('nonexistent_key')
        
        assert result is True

    def test_remove_item(self, cache_service):
        """测试移除项"""
        cache_service.cache['key1'] = 'value1'
        cache_service.timestamps['key1'] = time.time()
        cache_service.access_times['key1'] = time.time()
        
        cache_service._remove_item('key1')
        
        assert 'key1' not in cache_service.cache
        assert 'key1' not in cache_service.timestamps
        assert 'key1' not in cache_service.access_times

    def test_remove_item_partial(self, cache_service):
        """测试移除项 - 部分存在"""
        cache_service.cache['key1'] = 'value1'
        # 只有cache中有，timestamps和access_times中没有
        
        cache_service._remove_item('key1')
        
        assert 'key1' not in cache_service.cache

    def test_evict_items_empty(self, cache_service):
        """测试驱逐项 - 空缓存"""
        cache_service._evict_items()  # 不应该出错
        
        assert len(cache_service.cache) == 0

    def test_evict_items_lru(self, cache_service):
        """测试驱逐项 - LRU策略"""
        cache_service.cache = {'key1': 'value1', 'key2': 'value2'}
        cache_service.timestamps = {'key1': time.time(), 'key2': time.time()}
        cache_service.access_times = {
            'key1': time.time() - 100,  # 更早，应该被驱逐
            'key2': time.time()
        }
        
        cache_service._evict_items()
        
        assert len(cache_service.cache) == 1
        assert 'key1' not in cache_service.cache
        assert 'key2' in cache_service.cache

    def test_health_check_initialized(self, cache_service):
        """测试健康检查 - 已初始化"""
        cache_service.initialize()
        cache_service.cache['key1'] = 'value1'
        
        health = cache_service.health_check()
        
        assert health['service'] == 'config_cache_service'
        assert health['status'] == 'healthy'
        assert health['cache_size'] == 1

    def test_health_check_uninitialized(self, cache_service):
        """测试健康检查 - 未初始化"""
        health = cache_service.health_check()
        
        assert health['service'] == 'config_cache_service'
        assert health['status'] == 'uninitialized'
        assert health['cache_size'] == 0
        assert health['hit_rate'] == 0

    def test_thread_safety(self, cache_service):
        """测试线程安全性"""
        cache_service.initialize()
        
        def worker(worker_id):
            for i in range(10):
                key = f"worker_{worker_id}_key_{i}"
                value = f"value_{worker_id}_{i}"
                cache_service.set(key, value)
                time.sleep(0.001)  # 模拟一些工作
        
        # 启动多个线程
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证没有数据损坏
        assert len(cache_service.cache) <= cache_service.maxsize
        stats = cache_service.get_stats()
        assert stats['initialized'] is True
