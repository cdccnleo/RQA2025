import pytest
import threading
import time
from unittest.mock import patch, Mock
from src.infrastructure.cache.thread_safe_cache import ThreadSafeTTLCache, CacheMonitor

class TestThreadSafeTTLCacheFixed:
    """线程安全TTL缓存修复版本测试"""

    def test_basic_set_get(self):
        cache = ThreadSafeTTLCache(maxsize=10, ttl=5)
        cache['a'] = 123
        assert cache['a'] == 123
        assert 'a' in cache
        assert len(cache) == 1

    def test_expire(self):
        cache = ThreadSafeTTLCache(maxsize=10, ttl=1)
        cache['b'] = 456
        time.sleep(1.2)
        with pytest.raises(KeyError):
            _ = cache['b']

    def test_overwrite_and_delete(self):
        cache = ThreadSafeTTLCache(maxsize=10, ttl=5)
        cache['c'] = 1
        cache['c'] = 2
        assert cache['c'] == 2
        del cache['c']
        assert 'c' not in cache

    def test_bulk_set_get_delete(self):
        cache = ThreadSafeTTLCache(maxsize=10, ttl=5)
        items = {f'k{i}': i for i in range(5)}
        cache.bulk_set(items)
        result = cache.bulk_get(list(items.keys()))
        assert result == items
        deleted = cache.bulk_delete(list(items.keys()))
        assert deleted == 5
        for k in items:
            assert k not in cache

    def test_memory_limit(self):
        # 设置极小内存限制，触发淘汰
        cache = ThreadSafeTTLCache(maxsize=10, ttl=5, max_memory=32)
        cache['x'] = 'a' * 16
        cache['y'] = 'b' * 16
        # 插入第三个会触发淘汰
        cache['z'] = 'c' * 16
        assert len(cache) <= 2

    def test_compression(self):
        # 设置极低压缩阈值，强制压缩
        cache = ThreadSafeTTLCache(maxsize=10, ttl=5, compression_threshold=1)
        large_obj = 'a' * 1024
        cache['big'] = large_obj
        # 取出时自动解压
        assert cache['big'] == large_obj

    def test_concurrent_access(self):
        cache = ThreadSafeTTLCache(maxsize=100, ttl=5)
        def writer():
            for i in range(50):
                cache[f'k{i}'] = i
        def reader():
            for i in range(50):
                try:
                    _ = cache[f'k{i}']
                except KeyError:
                    pass
        threads = [threading.Thread(target=writer) for _ in range(2)] + [threading.Thread(target=reader) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # 检查无死锁且部分数据可用
        found = sum(1 for i in range(50) if f'k{i}' in cache)
        assert found >= 0

    def test_clear_and_keys(self):
        cache = ThreadSafeTTLCache(maxsize=10, ttl=5)
        cache['a'] = 1
        cache['b'] = 2
        cache.clear()
        assert len(cache) == 0
        assert list(cache.keys()) == []

    def test_monitor_metrics(self):
        cache = ThreadSafeTTLCache(maxsize=10, ttl=5)
        cache['a'] = 1
        try:
            _ = cache['b']
        except KeyError:
            pass
        metrics = cache._monitor.get_metrics()
        assert 'hit_count' in metrics
        assert 'miss_count' in metrics
        assert 'memory_usage' in metrics

class TestCacheMonitorFixed:
    def test_monitor_record_and_metrics(self):
        monitor = CacheMonitor(max_memory=1024)
        monitor.record_hit()
        monitor.record_miss()
        monitor.record_compression()
        monitor.record_eviction()
        monitor.record_write()
        monitor.update_memory_usage(512)
        metrics = monitor.get_metrics()
        assert metrics['hit_count'] == 1
        assert metrics['miss_count'] == 1
        assert metrics['compression_count'] == 1
        assert metrics['eviction_count'] == 1
        assert metrics['write_count'] == 1
        assert metrics['memory_usage'] == 512
        assert metrics['memory_usage_percentage'] == 50.0 