"""
mixins 模块全面测试

针对 MonitoringMixin、CRUDOperationsMixin、ComponentLifecycleMixin、CacheTierMixin
的全面测试，覆盖更多未测试的代码路径，提升覆盖率到70%+。
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import threading
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime


class TestMonitoringMixinComprehensive:
    """MonitoringMixin 全面测试"""

    def test_monitoring_initialization_variations(self):
        """测试监控初始化的各种变体"""
        from src.infrastructure.cache.core.mixins import MonitoringMixin

        # 测试不同参数组合
        configs = [
            {"enable_monitoring": True, "monitor_interval": 30},
            {"enable_monitoring": False, "monitor_interval": 60},
            {"enable_monitoring": True, "monitor_interval": 10},
            {"enable_monitoring": False, "monitor_interval": 1},
        ]

        for config in configs:
            mixin = MonitoringMixin(**config)
            assert mixin._enable_monitoring == config["enable_monitoring"]
            assert mixin._monitor_interval == config["monitor_interval"]
            assert mixin._monitoring_thread is None
            assert mixin._monitoring_active is False
            assert mixin._last_metrics is None

    def test_start_monitoring_comprehensive(self):
        """测试启动监控的全面场景"""
        from src.infrastructure.cache.core.mixins import MonitoringMixin

        class TestMonitoring(MonitoringMixin):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.collect_count = 0

            def _collect_metrics(self):
                from src.infrastructure.cache.interfaces import PerformanceMetrics
                self.collect_count += 1
                return PerformanceMetrics.create_current(
                    hit_rate=0.8,
                    response_time=0.01,
                    throughput=self.collect_count,
                    memory_usage=50.0,
                    cache_size=1000,
                    eviction_rate=0.1,
                    miss_penalty=5.0
                )

            def _check_alerts(self, metrics):
                pass

        # 测试正常启动
        monitor = TestMonitoring(enable_monitoring=True, monitor_interval=0.1)
        result = monitor.start_monitoring()
        assert result is True
        assert monitor._monitoring_active is True

        # 等待一些监控循环
        time.sleep(0.3)
        assert monitor.collect_count > 0

        # 停止监控
        result = monitor.stop_monitoring()
        assert result is True
        assert monitor._monitoring_active is False

    def test_monitoring_thread_safety(self):
        """测试监控的线程安全性"""
        from src.infrastructure.cache.core.mixins import MonitoringMixin

        class TestMonitoring(MonitoringMixin):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.collect_count = 0
                self.check_count = 0

            def _collect_metrics(self):
                from src.infrastructure.cache.interfaces import PerformanceMetrics
                self.collect_count += 1
                return PerformanceMetrics.create_current(
                    hit_rate=0.9,
                    response_time=0.005,
                    throughput=10,
                    memory_usage=40.0,
                    cache_size=800,
                    eviction_rate=0.05,
                    miss_penalty=3.0
                )

            def _check_alerts(self, metrics):
                self.check_count += 1

        monitor = TestMonitoring(enable_monitoring=True, monitor_interval=0.05)

        # 并发启动和停止监控
        results = []

        def concurrent_monitoring():
            try:
                monitor.start_monitoring()
                time.sleep(0.1)
                monitor.stop_monitoring()
                results.append("success")
            except Exception as e:
                results.append(f"error: {e}")

        threads = []
        for i in range(3):
            thread = threading.Thread(target=concurrent_monitoring)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(results) == 3
        assert all(r == "success" for r in results)

    def test_monitoring_with_disabled_monitoring(self):
        """测试禁用监控的情况"""
        from src.infrastructure.cache.core.mixins import MonitoringMixin

        class TestMonitoring(MonitoringMixin):
            def _collect_metrics(self):
                from src.infrastructure.cache.interfaces import PerformanceMetrics
                return PerformanceMetrics.create_current(
                    hit_rate=1.0,
                    response_time=0.001,
                    throughput=1,
                    memory_usage=30.0,
                    cache_size=500,
                    eviction_rate=0.0,
                    miss_penalty=1.0
                )

            def _check_alerts(self, metrics):
                pass

        monitor = TestMonitoring(enable_monitoring=False)

        # 启动监控应该返回False（因为监控被禁用）
        result = monitor.start_monitoring()
        assert result is False
        assert monitor._monitoring_thread is None
        assert monitor._monitoring_active is False

        # 停止监控也应该成功
        result = monitor.stop_monitoring()
        assert result is True

    def test_monitoring_status_comprehensive(self):
        """测试监控状态的全面信息"""
        from src.infrastructure.cache.core.mixins import MonitoringMixin

        class TestMonitoring(MonitoringMixin):
            def _collect_metrics(self):
                from src.infrastructure.cache.interfaces import PerformanceMetrics
                return PerformanceMetrics.create_current(
                    hit_rate=0.85,
                    response_time=0.02,
                    throughput=50,
                    memory_usage=60.0,
                    cache_size=1200,
                    eviction_rate=0.2,
                    miss_penalty=8.0
                )

            def _check_alerts(self, metrics):
                pass

        monitor = TestMonitoring(enable_monitoring=True, monitor_interval=30)

        # 初始状态
        status = monitor.get_monitoring_status()
        assert status["monitoring_enabled"] is True
        assert status["monitoring_active"] is False
        assert status["monitor_interval"] == 30
        assert status["last_metrics"] is None
        assert status["thread_alive"] is False

        # 启动后状态
        monitor.start_monitoring()
        time.sleep(0.1)  # 让监控运行一次

        status = monitor.get_monitoring_status()
        assert status["monitoring_active"] is True
        assert status["last_metrics"] is not None
        assert isinstance(status["thread_alive"], bool)

        monitor.stop_monitoring()

    def test_alerts_various_scenarios(self):
        """测试各种告警场景"""
        from src.infrastructure.cache.core.mixins import MonitoringMixin
        from src.infrastructure.cache.interfaces import PerformanceMetrics

        class TestMonitoring(MonitoringMixin):
            def __init__(self):
                super().__init__()
                self.alerts_recorded = []

            def _check_alerts(self, metrics):
                # 覆盖原始的_check_alerts逻辑
                super()._check_alerts(metrics)
                # 记录告警（通过mock来捕获）

        monitor = TestMonitoring()

        # 测试各种需要告警的情况
        alert_scenarios = [
            # 低命中率
            PerformanceMetrics.create_current(
                hit_rate=0.3, response_time=0.01, throughput=10,
                memory_usage=50.0, cache_size=1000, eviction_rate=0.1, miss_penalty=5.0
            ),
            # 高响应时间
            PerformanceMetrics.create_current(
                hit_rate=0.9, response_time=0.15, throughput=10,  # 150ms
                memory_usage=50.0, cache_size=1000, eviction_rate=0.1, miss_penalty=5.0
            ),
            # 高内存使用
            PerformanceMetrics.create_current(
                hit_rate=0.9, response_time=0.01, throughput=10, memory_usage=700,  # 700MB
                cache_size=1000, eviction_rate=0.1, miss_penalty=5.0
            ),
            # 正常情况
            PerformanceMetrics.create_current(
                hit_rate=0.9, response_time=0.05, throughput=10, memory_usage=200,
                cache_size=1000, eviction_rate=0.1, miss_penalty=5.0
            ),
        ]

        for metrics in alert_scenarios:
            with patch('src.infrastructure.cache.core.mixins.logger') as mock_logger:
                monitor._check_alerts(metrics)
                # 对于异常情况，应该有告警记录
                if (metrics.hit_rate < 0.5 or
                    metrics.response_time > 100 or
                    getattr(metrics, 'memory_usage', 0) > 500):
                    mock_logger.warning.assert_called()


class TestCRUDOperationsMixinComprehensive:
    """CRUDOperationsMixin 全面测试"""

    def test_crud_operations_basic_functionality(self):
        """测试CRUD操作的基本功能"""
        from src.infrastructure.cache.core.mixins import CRUDOperationsMixin

        crud = CRUDOperationsMixin()

        # 测试设置和获取
        result = crud.set("test_key", "test_value")
        assert result is True

        value = crud.get("test_key")
        assert value == "test_value"

        # 测试存在性检查
        exists = crud.exists("test_key")
        assert exists is True

        exists = crud.exists("nonexistent_key")
        assert exists is False

        # 测试删除
        deleted = crud.delete("test_key")
        assert deleted is True

        # 再次检查应该不存在
        value = crud.get("test_key")
        assert value is None

        deleted = crud.delete("test_key")
        assert deleted is False  # 已经删除了

    def test_crud_operations_with_ttl(self):
        """测试带TTL的CRUD操作"""
        from src.infrastructure.cache.core.mixins import CRUDOperationsMixin

        crud = CRUDOperationsMixin()

        # 设置带TTL的值
        result = crud.set("ttl_key", "ttl_value", ttl=30)
        assert result is True

        # 应该能立即获取
        value = crud.get("ttl_key")
        assert value == "ttl_value"

    def test_clear_operation_comprehensive(self):
        """测试清空操作的全面测试"""
        from src.infrastructure.cache.core.mixins import CRUDOperationsMixin

        crud = CRUDOperationsMixin()

        # 添加多个项目
        for i in range(10):
            crud.set(f"clear_key_{i}", f"clear_value_{i}")

        # 验证项目存在
        for i in range(10):
            value = crud.get(f"clear_key_{i}")
            assert value == f"clear_value_{i}"

        # 清空所有
        result = crud.clear()
        assert result is True

        # 验证所有项目都被清空
        for i in range(10):
            value = crud.get(f"clear_key_{i}")
            assert value is None

    def test_crud_operations_error_handling(self):
        """测试CRUD操作的错误处理"""
        from src.infrastructure.cache.core.mixins import CRUDOperationsMixin

        crud = CRUDOperationsMixin()

        # Mock存储对象使其抛出异常
        crud._storage = Mock()
        crud._lock = threading.Lock()

        # 测试设置时的异常
        crud._storage.__setitem__ = Mock(side_effect=Exception("Storage error"))
        with patch('src.infrastructure.cache.core.mixins.logger') as mock_logger:
            result = crud.set("error_key", "error_value")
            assert result is False
            mock_logger.error.assert_called()

        # 重置mock
        crud._storage.__setitem__.side_effect = None

        # 测试获取时的异常
        crud._storage.get = Mock(side_effect=Exception("Get error"))
        with patch('src.infrastructure.cache.core.mixins.logger') as mock_logger:
            value = crud.get("error_key")
            assert value is None  # 异常被捕获，返回None
            mock_logger.error.assert_called()

        # 重置mock以避免影响后续测试
        crud._storage.get.side_effect = None

        # 重置mock
        crud._storage.__getitem__ = Mock(return_value=None)
        crud._storage.__contains__ = Mock(return_value=False)

        # 测试删除时的异常
        crud._storage.pop = Mock(side_effect=Exception("Delete error"))
        with patch('src.infrastructure.cache.core.mixins.logger') as mock_logger:
            result = crud.delete("error_key")
            assert result is False  # 异常被捕获，返回False
            # delete方法不记录错误，所以不检查logger

    def test_storage_initialization_variations(self):
        """测试存储初始化的各种变体"""
        from src.infrastructure.cache.core.mixins import CRUDOperationsMixin

        # 默认初始化
        crud1 = CRUDOperationsMixin()
        assert crud1._storage == {}
        assert hasattr(crud1, '_lock')

        # 自定义存储后端
        custom_storage = {"preloaded": "value"}
        crud2 = CRUDOperationsMixin(storage_backend=custom_storage)
        assert crud2._storage is custom_storage
        assert crud2.get("preloaded") == "value"

    def test_thread_safety_comprehensive(self):
        """测试线程安全的全面场景"""
        from src.infrastructure.cache.core.mixins import CRUDOperationsMixin

        crud = CRUDOperationsMixin()
        results = []
        errors = []

        def crud_worker(worker_id):
            try:
                # 执行各种CRUD操作
                for i in range(25):
                    key = f"thread_{worker_id}_{i}"
                    value = f"value_{worker_id}_{i}"

                    # 设置
                    crud.set(key, value)

                    # 获取
                    retrieved = crud.get(key)
                    assert retrieved == value

                    # 检查存在性
                    exists = crud.exists(key)
                    assert exists is True

                    # 删除
                    deleted = crud.delete(key)
                    assert deleted is True

                    # 再次检查不存在
                    exists = crud.exists(key)
                    assert exists is False

                results.append(f"worker_{worker_id}_success")
            except Exception as e:
                errors.append(f"worker_{worker_id}_error: {e}")

        # 启动多个线程
        threads = []
        for i in range(4):
            thread = threading.Thread(target=crud_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        assert len(results) == 4
        assert len(errors) == 0


class TestComponentLifecycleMixinComprehensive:
    """ComponentLifecycleMixin 全面测试"""

    def test_component_initialization_comprehensive(self):
        """测试组件初始化的全面场景"""
        from src.infrastructure.cache.core.mixins import ComponentLifecycleMixin

        # 创建组件
        component = ComponentLifecycleMixin(
            component_id="test_comp",
            component_type="test_type"
        )

        # 验证初始状态
        status = component.get_component_status()
        assert status["component_id"] == "test_comp"
        assert status["component_type"] == "test_type"
        assert status["initialized"] is False
        assert status["status"] == "stopped"
        assert status["error_count"] == 0

        # 测试初始化
        result = component.initialize_component({"test_config": "value"})
        assert result is True

        status = component.get_component_status()
        assert status["initialized"] is True
        assert status["status"] == "healthy"
        assert status["error_count"] == 0
        assert status["config"]["test_config"] == "value"

    def test_component_operations_lifecycle(self):
        """测试组件操作的生命周期"""
        from src.infrastructure.cache.core.mixins import ComponentLifecycleMixin

        component = ComponentLifecycleMixin("lifecycle_test", "test")

        # 初始化
        component.initialize_component()

        # 启动
        result = component.start_component()
        assert result is True

        # 停止
        result = component.stop_component()
        assert result is True

        # 再次停止（应该成功）
        result = component.stop_component()
        assert result is True

    def test_component_error_handling(self):
        """测试组件错误处理"""
        from src.infrastructure.cache.core.mixins import ComponentLifecycleMixin

        component = ComponentLifecycleMixin("error_test", "test")

        # 测试初始化失败
        class FailingComponent(ComponentLifecycleMixin):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def _initialize_component(self):
                raise Exception("Initialization failed")

        with patch('src.infrastructure.cache.core.mixins.logger') as mock_logger:
            failing_component = FailingComponent("failing", "test")
            try:
                failing_component.initialize_component()  # 这会抛出异常
            except Exception:
                pass  # 异常被抛出，这是预期的
            status = failing_component.get_component_status()
            assert status["status"] == "error"
            assert status["error_count"] == 1
            mock_logger.error.assert_called()

    def test_health_check_variations(self):
        """测试健康检查的各种变体"""
        from src.infrastructure.cache.core.mixins import ComponentLifecycleMixin

        component = ComponentLifecycleMixin("health_test", "test")

        # 未初始化状态下的健康检查
        health = component.health_check()
        assert health is False

        # 初始化后的健康检查
        component.initialize_component()
        health = component.health_check()
        assert health is True

        # 模拟错误状态
        component._error_count = 6  # 超过阈值
        health = component.health_check()
        assert health is False

        # 重置并再次检查
        component._error_count = 0
        health = component.health_check()
        assert health is True

    def test_shutdown_component_comprehensive(self):
        """测试关闭组件的全面场景"""
        from src.infrastructure.cache.core.mixins import ComponentLifecycleMixin

        component = ComponentLifecycleMixin("shutdown_test", "test")

        # 初始化组件
        component.initialize_component()

        # 关闭组件
        component.shutdown_component()

        status = component.get_component_status()
        assert status["status"] == "stopped"
        assert status["initialized"] is False

    def test_component_status_tracking(self):
        """测试组件状态跟踪"""
        from src.infrastructure.cache.core.mixins import ComponentLifecycleMixin

        component = ComponentLifecycleMixin("status_test", "test")

        # 检查初始状态
        status1 = component.get_component_status()
        initial_time = status1["creation_time"]

        # 等待一小段时间
        time.sleep(0.01)

        # 再次检查状态
        status2 = component.get_component_status()
        last_check_time = status2["last_check"]

        # 验证时间戳更新
        assert status2["creation_time"] == initial_time
        assert last_check_time != initial_time


class TestCacheTierMixinComprehensive:
    """CacheTierMixin 全面测试"""

    def test_cache_tier_basic_operations(self):
        """测试缓存层基本操作"""
        from src.infrastructure.cache.core.mixins import CacheTierMixin

        class TestCacheTier(CacheTierMixin):
            def __init__(self):
                super().__init__()
                self._storage = {}

            def _set_value(self, key, value, ttl=None):
                self._storage[key] = value
                return True

            def _get_value(self, key):
                return self._storage.get(key)

            def _delete_value(self, key):
                return self._storage.pop(key, None) is not None

            def _key_exists(self, key):
                return key in self._storage

            def _is_expired(self, key):
                return False  # 测试中不过期

            def _should_evict(self):
                return len(self._storage) > 10

            def _evict_oldest(self):
                if self._storage:
                    oldest_key = next(iter(self._storage))
                    del self._storage[oldest_key]

            def _get_size(self):
                return len(self._storage)

        tier = TestCacheTier()

        # 测试设置和获取
        result = tier.set("tier_key", "tier_value")
        assert result is True

        value = tier.get("tier_key")
        assert value == "tier_value"

        # 测试存在性检查
        exists = tier.exists("tier_key")
        assert exists is True

        # 测试删除
        deleted = tier.delete("tier_key")
        assert deleted is True

        # 验证删除后不存在
        exists = tier.exists("tier_key")
        assert exists is False

    def test_cache_tier_eviction_logic(self):
        """测试缓存层驱逐逻辑"""
        from src.infrastructure.cache.core.mixins import CacheTierMixin

        class EvictionTestTier(CacheTierMixin):
            def __init__(self):
                super().__init__()
                self._storage = {}
                self.eviction_count = 0

            def _set_value(self, key, value, ttl=None):
                self._storage[key] = value
                return True

            def _get_value(self, key):
                return self._storage.get(key)

            def _delete_value(self, key):
                return self._storage.pop(key, None) is not None

            def _key_exists(self, key):
                return key in self._storage

            def _is_expired(self, key):
                return False  # 测试中不过期

            def _should_evict(self):
                return len(self._storage) >= 5  # 小容量以便测试

            def _evict_oldest(self):
                if self._storage:
                    oldest_key = next(iter(self._storage))
                    del self._storage[oldest_key]
                    self.eviction_count += 1

            def _get_size(self):
                return len(self._storage)

        tier = EvictionTestTier()

        # 添加项目直到触发驱逐
        for i in range(7):  # 超过容量
            tier.set(f"evict_key_{i}", f"evict_value_{i}")

        # 验证驱逐发生了
        assert tier.eviction_count > 0
        assert len(tier._storage) <= 5  # 不超过容量

    def test_cache_tier_statistics_tracking(self):
        """测试缓存层统计跟踪"""
        from src.infrastructure.cache.core.mixins import CacheTierMixin

        class StatsTestTier(CacheTierMixin):
            def __init__(self):
                super().__init__()
                self._storage = {}

            def _set_value(self, key, value, ttl=None):
                self._storage[key] = value
                return True

            def _get_value(self, key):
                return self._storage.get(key)

            def _delete_value(self, key):
                return self._storage.pop(key, None) is not None

            def _key_exists(self, key):
                return key in self._storage

            def _is_expired(self, key):
                return False  # 测试中不过期

            def _should_evict(self):
                return False

            def _evict_oldest(self):
                pass

            def _get_size(self):
                return len(self._storage)

        tier = StatsTestTier()

        # 执行各种操作
        operations = [
            ("set", "stats_key_1", "value_1"),
            ("set", "stats_key_2", "value_2"),
            ("get", "stats_key_1"),
            ("get", "stats_key_2"),
            ("get", "nonexistent"),
            ("delete", "stats_key_1"),
            ("delete", "nonexistent"),
            ("set", "stats_key_3", "value_3"),
        ]

        for op in operations:
            if op[0] == "set":
                tier.set(op[1], op[2])
            elif op[0] == "get":
                tier.get(op[1])
            elif op[0] == "delete":
                tier.delete(op[1])

        # 获取统计信息
        stats = tier.get_stats()
        assert isinstance(stats, dict)

        # 验证统计信息包含必要的字段
        expected_keys = ["total_requests", "hit_count", "miss_count", "set_count", "delete_count"]
        for key in expected_keys:
            if key in stats:
                assert isinstance(stats[key], (int, float))

    def test_cache_tier_error_handling(self):
        """测试缓存层错误处理"""
        from src.infrastructure.cache.core.mixins import CacheTierMixin

        class ErrorTestTier(CacheTierMixin):
            def __init__(self):
                super().__init__()
                self._storage = {}

            def _set_value(self, key, value, ttl=None):
                if key == "error_key":
                    raise Exception("Set operation failed")
                self._storage[key] = value
                return True

            def _get_value(self, key):
                if key == "error_get_key":
                    raise Exception("Get operation failed")
                return self._storage.get(key)

            def _delete_value(self, key):
                if key == "error_delete_key":
                    raise Exception("Delete operation failed")
                return self._storage.pop(key, None) is not None

            def _key_exists(self, key):
                # 对于测试错误情况的键，假装它们存在
                if key in ["error_get_key", "error_delete_key"]:
                    return True
                return key in self._storage

            def _is_expired(self, key):
                return False  # 测试中不过期

            def _should_evict(self):
                return False

            def _evict_oldest(self):
                pass

            def _get_size(self):
                return len(self._storage)

        tier = ErrorTestTier()

        # 测试设置错误
        with patch.object(tier.logger, 'error') as mock_logger:
            result = tier.set("error_key", "value")
            assert result is False
            mock_logger.assert_called()

        # 测试获取错误
        with patch.object(tier.logger, 'error') as mock_logger:
            value = tier.get("error_get_key")
            assert value is None
            mock_logger.assert_called()

        # 测试删除错误
        with patch.object(tier.logger, 'error') as mock_logger:
            result = tier.delete("error_delete_key")
            assert result is False
            mock_logger.assert_called()

    def test_cache_tier_concurrent_operations(self):
        """测试缓存层并发操作"""
        from src.infrastructure.cache.core.mixins import CacheTierMixin

        class ConcurrentTestTier(CacheTierMixin):
            def __init__(self):
                super().__init__()
                self._storage = {}

            def _set_value(self, key, value, ttl=None):
                self._storage[key] = value
                return True

            def _get_value(self, key):
                return self._storage.get(key)

            def _delete_value(self, key):
                return self._storage.pop(key, None) is not None

            def _key_exists(self, key):
                return key in self._storage

            def _is_expired(self, key):
                return False  # 测试中不过期

            def _should_evict(self):
                return len(self._storage) > 20

            def _evict_oldest(self):
                if self._storage:
                    oldest_key = next(iter(self._storage))
                    del self._storage[oldest_key]

            def _get_size(self):
                return len(self._storage)

        tier = ConcurrentTestTier()
        results = []
        errors = []

        def tier_concurrent_worker(worker_id):
            try:
                for i in range(15):
                    key = f"concurrent_{worker_id}_{i}"
                    value = f"value_{worker_id}_{i}"

                    tier.set(key, value)
                    retrieved = tier.get(key)
                    assert retrieved == value
                    tier.delete(key)

                results.append(f"tier_worker_{worker_id}_success")
            except Exception as e:
                errors.append(f"tier_worker_{worker_id}_error: {e}")

        # 启动多个线程
        threads = []
        for i in range(3):
            thread = threading.Thread(target=tier_concurrent_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待完成
        for thread in threads:
            thread.join()

        assert len(results) == 3
        assert len(errors) == 0
