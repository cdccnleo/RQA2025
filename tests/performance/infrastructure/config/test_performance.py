import pytest
import time
from src.infrastructure.config.services import (
    ConfigLoaderService,
    CacheService
)
from src.infrastructure.config.strategies import JSONLoader

class TestConfigPerformance:
    """配置管理模块性能测试"""

    @pytest.fixture
    def json_loader(self):
        return JSONLoader()

    @pytest.fixture
    def large_config(self):
        """生成大型测试配置"""
        return {
            f"key_{i}": f"value_{i}" for i in range(1000)  # 1000个键值对
        }

    @pytest.mark.performance
    def test_single_load_performance(self, json_loader, benchmark):
        """测试单配置文件加载性能"""
        import json
        test_data = {"db": {"host": "localhost"}}
        json_str = json.dumps(test_data)

        with patch("builtins.open", mock_open(read_data=json_str)):
            # 使用pytest-benchmark测试加载性能
            result = benchmark(json_loader.load, "test.json")
            assert result[0] == test_data

    @pytest.mark.performance
    def test_batch_load_throughput(self, json_loader):
        """测试批量加载吞吐量"""
        import json
        test_data = {"key": "value"}
        json_str = json.dumps(test_data)
        files = [f"config_{i}.json" for i in range(100)]  # 100个文件

        with patch("builtins.open", mock_open(read_data=json_str)):
            start_time = time.time()
            results = json_loader.batch_load(files)
            elapsed = time.time() - start_time

            assert len(results) == 100
            assert elapsed < 1.0  # 100个文件应在1秒内完成
            print(f"批量加载吞吐量: {len(files)/elapsed:.2f} 文件/秒")

    @pytest.mark.performance
    def test_cache_hit_rate(self, json_loader):
        """测试缓存命中率"""
        cache = CacheService(max_size=100)
        service = ConfigLoaderService(json_loader, cache)

        # 模拟加载10次，其中5个不同key
        keys = ["config1.json"] * 5 + [f"config_{i}.json" for i in range(5)]
        test_data = {"key": "value"}

        with patch("builtins.open", mock_open(read_data=json.dumps(test_data))):
            for key in keys:
                service.load(key)

        metrics = cache.get_metrics()
        assert metrics['hits'] == 4  # 前5次中4次命中(第一个key重复4次)
        assert metrics['misses'] == 6
        print(f"缓存命中率: {metrics['hit_rate']:.2%}")

    @pytest.mark.performance
    def test_high_concurrency(self, json_loader):
        """测试高并发配置访问性能"""
        import threading
        from concurrent.futures import ThreadPoolExecutor

        cache = CacheService(max_size=1000)
        service = ConfigLoaderService(json_loader, cache)
        test_data = {"key": "value"}

        def worker(key):
            with patch("builtins.open", mock_open(read_data=json.dumps(test_data))):
                return service.load(key)

        # 模拟100个并发请求
        with ThreadPoolExecutor(max_workers=20) as executor:
            start_time = time.time()
            keys = [f"config_{i}.json" for i in range(100)]
            results = list(executor.map(worker, keys))
            elapsed = time.time() - start_time

        assert len(results) == 100
        assert all(r == test_data for r in results)
        print(f"并发吞吐量: {len(keys)/elapsed:.2f} 请求/秒")
