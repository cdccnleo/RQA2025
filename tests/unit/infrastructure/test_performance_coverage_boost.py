#!/usr/bin/env python3
"""
基础设施层性能测试覆盖率提升

测试目标：通过轻量级性能测试快速提升覆盖率
测试范围：各模块的基本性能测试场景
测试策略：快速执行，覆盖关键性能路径
"""

import pytest
import time
from unittest.mock import Mock


class TestPerformanceCoverageBoost:
    """性能测试覆盖率提升"""

    def test_config_performance_basic(self):
        """配置模块基础性能测试"""
        from src.infrastructure.config.core.config_manager_complete import UnifiedConfigManager

        manager = UnifiedConfigManager()

        # 基础性能测试
        start_time = time.time()

        # 批量配置操作
        for i in range(100):
            key = f"perf_config_{i}"
            value = {"data": f"value_{i}", "timestamp": time.time()}
            manager.set(key, value)

        # 批量读取操作
        for i in range(100):
            key = f"perf_config_{i}"
            value = manager.get(key)
            assert value is not None

        end_time = time.time()
        duration = end_time - start_time

        # 性能断言
        assert duration < 2.0, f"Config operations too slow: {duration:.2f}s"

    def test_cache_performance_basic(self):
        """缓存模块基础性能测试"""
        from src.infrastructure.cache.core.cache_manager import UnifiedCacheManager

        manager = UnifiedCacheManager()

        # 基础性能测试
        start_time = time.time()

        # 批量缓存操作
        for i in range(200):
            key = f"perf_cache_{i}"
            value = f"cache_value_{i}_" + "x" * 50
            manager.set(key, value, ttl=300)

        # 批量读取操作
        for i in range(200):
            key = f"perf_cache_{i}"
            value = manager.get(key)
            assert value is not None

        end_time = time.time()
        duration = end_time - start_time

        # 性能断言
        assert duration < 3.0, f"Cache operations too slow: {duration:.2f}s"

    def test_logging_performance_basic(self):
        """日志模块基础性能测试"""
        from src.infrastructure.logging.core.unified_logger import UnifiedLogger

        logger = UnifiedLogger("perf_log_test")

        # 基础性能测试
        start_time = time.time()

        # 批量日志记录
        for i in range(500):
            if i % 50 == 0:
                logger.error(f"Performance error log {i}")
            elif i % 10 == 0:
                logger.warning(f"Performance warning log {i}")
            else:
                logger.info(f"Performance info log {i}")

        end_time = time.time()
        duration = end_time - start_time

        # 性能断言
        assert duration < 5.0, f"Logging operations too slow: {duration:.2f}s"

    def test_config_cache_interaction_performance(self):
        """配置缓存交互性能测试"""
        from src.infrastructure.config.core.config_manager_complete import UnifiedConfigManager
        from src.infrastructure.cache.core.cache_manager import UnifiedCacheManager

        config_manager = UnifiedConfigManager()
        cache_manager = UnifiedCacheManager()

        # 配置缓存交互测试
        start_time = time.time()

        for i in range(50):
            # 配置更新
            config_key = f"interactive_config_{i}"
            config_value = {"setting": f"value_{i}", "cache_enabled": True}
            config_manager.set(config_key, config_value)

            # 缓存同步
            cache_key = f"config_cache_{i}"
            cache_manager.set(cache_key, config_value, ttl=300)

            # 验证一致性
            cached_config = cache_manager.get(cache_key)
            assert cached_config is not None
            assert cached_config["setting"] == f"value_{i}"

        end_time = time.time()
        duration = end_time - start_time

        # 性能断言
        assert duration < 2.0, f"Config-cache interaction too slow: {duration:.2f}s"

    def test_multi_component_performance(self):
        """多组件协同性能测试"""
        from src.infrastructure.config.core.config_manager_complete import UnifiedConfigManager
        from src.infrastructure.cache.core.cache_manager import UnifiedCacheManager
        from src.infrastructure.logging.core.unified_logger import UnifiedLogger

        config_manager = UnifiedConfigManager()
        cache_manager = UnifiedCacheManager()
        logger = UnifiedLogger("multi_perf_test")

        # 多组件协同测试
        start_time = time.time()

        for i in range(100):
            # 1. 配置管理
            config_key = f"multi_config_{i}"
            config_data = {"component": "multi_test", "id": i}
            config_manager.set(config_key, config_data)

            # 2. 缓存操作
            cache_key = f"multi_cache_{i}"
            cache_data = {"config_ref": config_key, "data": f"test_{i}"}
            cache_manager.set(cache_key, cache_data, ttl=300)

            # 3. 日志记录
            logger.info(f"Multi-component operation {i}", extra={
                "config_key": config_key,
                "cache_key": cache_key,
                "operation_id": i
            })

            # 4. 验证数据一致性
            retrieved_config = config_manager.get(config_key)
            retrieved_cache = cache_manager.get(cache_key)

            assert retrieved_config is not None
            assert retrieved_cache is not None
            assert retrieved_cache["config_ref"] == config_key

        end_time = time.time()
        duration = end_time - start_time

        # 性能断言
        assert duration < 5.0, f"Multi-component operations too slow: {duration:.2f}s"

    def test_concurrent_operations_performance(self):
        """并发操作性能测试"""
        import threading
        from src.infrastructure.cache.core.cache_manager import UnifiedCacheManager

        manager = UnifiedCacheManager()

        results = {"operations": 0, "errors": 0}
        results_lock = threading.Lock()

        def concurrent_worker(worker_id):
            """并发工作线程"""
            try:
                for i in range(50):  # 每个线程50个操作
                    key = f"concurrent_key_{worker_id}_{i}"
                    value = f"concurrent_value_{worker_id}_{i}"

                    manager.set(key, value, ttl=300)

                    # 读取验证
                    retrieved = manager.get(key)
                    if retrieved == value:
                        with results_lock:
                            results["operations"] += 1
                    else:
                        with results_lock:
                            results["errors"] += 1

            except Exception as e:
                with results_lock:
                    results["errors"] += 1

        # 启动并发测试
        start_time = time.time()

        threads = []
        num_threads = 10

        for i in range(num_threads):
            t = threading.Thread(target=concurrent_worker, args=(i,))
            threads.append(t)
            t.start()

        # 等待所有线程完成
        for t in threads:
            t.join()

        end_time = time.time()
        duration = end_time - start_time

        # 验证并发操作结果
        assert results["operations"] > 0, "No successful concurrent operations"
        assert results["errors"] < results["operations"] * 0.1, f"Too many concurrent errors: {results['errors']}"
        assert duration < 10.0, f"Concurrent operations too slow: {duration:.2f}s"
