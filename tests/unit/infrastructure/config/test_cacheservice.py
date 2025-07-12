import pytest
import time
from unittest.mock import MagicMock
from src.infrastructure.config.services import CacheService

class TestCacheService:
    """缓存服务测试"""

    @pytest.fixture
    def cache_service(self):
        return CacheService(maxsize=100, ttl=60)  # 最大100项，60秒过期

    @pytest.mark.unit
    def test_basic_cache_operations(self, cache_service):
        """测试基本缓存操作"""
        # 测试设置和获取
        cache_service.set("key1", "value1")
        assert cache_service.get("key1") == "value1"

        # 测试覆盖写入
        cache_service.set("key1", "new_value")
        assert cache_service.get("key1") == "new_value"

        # 测试不存在的key
        assert cache_service.get("nonexistent") is None

    @pytest.mark.unit
    def test_cache_expiration(self, cache_service):
        """测试缓存过期"""
        cache_service.set("temp", "data", ttl=1)  # 1秒过期
        assert cache_service.get("temp") == "data"

        # 等待过期
        time.sleep(1.1)
        assert cache_service.get("temp") is None

    @pytest.mark.unit
    def test_cache_eviction(self, cache_service):
        """测试缓存淘汰"""
        # 填充缓存
        for i in range(100):
            cache_service.set(f"key{i}", f"value{i}")

        # 添加第101项，应该触发淘汰
        cache_service.set("new_key", "new_value")

        # 验证至少有一项被淘汰
        assert any(cache_service.get(f"key{i}") is None for i in range(100))

    @pytest.mark.performance
    def test_cache_concurrency(self, cache_service):
        """测试并发缓存访问"""
        import threading

        # 初始化测试数据
        cache_service.set("counter", 0)

        def increment():
            for _ in range(100):
                val = cache_service.get("counter")
                cache_service.set("counter", val + 1)

        threads = [threading.Thread(target=increment) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 验证计数器正确性
        assert cache_service.get("counter") == 1000

    @pytest.mark.integration
    def test_cache_metrics(self, cache_service):
        """测试缓存指标收集"""
        # 生成一些缓存活动
        cache_service.set("key1", "value1")
        cache_service.get("key1")  # 命中
        cache_service.get("key2")   # 未命中

        stats = cache_service.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['writes'] == 1
