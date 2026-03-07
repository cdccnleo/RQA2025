"""
中期行动阶段单元测试

测试内容：
- 事件总线集成
- 优先级队列
- 批量处理器
- 任务缓存
- Prometheus指标
- 加密模块
- 访问控制
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any


# ========== 事件总线集成测试 ==========

class TestEventBusIntegration:
    """测试事件总线集成"""

    @pytest.mark.asyncio
    async def test_event_bus_initialization(self):
        """测试事件总线初始化"""
        try:
            from src.core.orchestration.scheduler.integration.event_bus_integration import (
                EventBusIntegration, EventDrivenTaskTrigger, SchedulerEventType
            )
            # 验证事件类型定义
            assert hasattr(SchedulerEventType, 'TASK_CREATED')
            assert hasattr(SchedulerEventType, 'TASK_STARTED')
            assert hasattr(SchedulerEventType, 'TASK_COMPLETED')
            assert hasattr(SchedulerEventType, 'TASK_FAILED')
            assert hasattr(SchedulerEventType, 'TASK_CANCELLED')
            assert hasattr(SchedulerEventType, 'TASK_TIMEOUT')
            assert hasattr(SchedulerEventType, 'TASK_RETRIED')
        except ImportError:
            pytest.skip("事件总线集成模块不可用")

    @pytest.mark.asyncio
    async def test_scheduler_event_type_values(self):
        """测试调度器事件类型值"""
        try:
            from src.core.orchestration.scheduler.integration.event_bus_integration import SchedulerEventType

            assert SchedulerEventType.TASK_CREATED == "scheduler.task.created"
            assert SchedulerEventType.TASK_STARTED == "scheduler.task.started"
            assert SchedulerEventType.TASK_COMPLETED == "scheduler.task.completed"
            assert SchedulerEventType.TASK_FAILED == "scheduler.task.failed"
            assert SchedulerEventType.TASK_CANCELLED == "scheduler.task.cancelled"
            assert SchedulerEventType.TASK_TIMEOUT == "scheduler.task.timeout"
            assert SchedulerEventType.TASK_RETRIED == "scheduler.task.retried"
        except ImportError:
            pytest.skip("事件总线集成模块不可用")


# ========== 优先级队列测试 ==========

class TestPriorityQueue:
    """测试优先级队列"""

    def test_priority_queue_initialization(self):
        """测试优先级队列初始化"""
        try:
            from src.core.orchestration.scheduler.performance.priority_queue import (
                PriorityTaskQueue, TaskPriority
            )

            queue = PriorityTaskQueue()
            assert queue.is_empty()
            assert queue.size() == 0
        except ImportError:
            pytest.skip("优先级队列模块不可用")

    def test_priority_queue_enqueue_dequeue(self):
        """测试优先级队列入队出队"""
        try:
            from src.core.orchestration.scheduler.performance.priority_queue import (
                PriorityTaskQueue, TaskPriority
            )

            queue = PriorityTaskQueue()

            # 添加不同优先级的任务
            queue.enqueue("task1", "type1", TaskPriority.NORMAL, {"data": 1})
            queue.enqueue("task2", "type1", TaskPriority.HIGH, {"data": 2})
            queue.enqueue("task3", "type1", TaskPriority.LOW, {"data": 3})
            queue.enqueue("task4", "type1", TaskPriority.CRITICAL, {"data": 4})

            assert queue.size() == 4

            # 验证按优先级出队（CRITICAL > HIGH > NORMAL > LOW）
            task = queue.dequeue()
            assert task.task_id == "task4"  # CRITICAL

            task = queue.dequeue()
            assert task.task_id == "task2"  # HIGH

            task = queue.dequeue()
            assert task.task_id == "task1"  # NORMAL

            task = queue.dequeue()
            assert task.task_id == "task3"  # LOW

            assert queue.is_empty()
        except ImportError:
            pytest.skip("优先级队列模块不可用")

    def test_priority_queue_fifo_same_priority(self):
        """测试相同优先级FIFO"""
        try:
            from src.core.orchestration.scheduler.performance.priority_queue import (
                PriorityTaskQueue, TaskPriority
            )

            queue = PriorityTaskQueue()

            # 添加相同优先级的任务
            queue.enqueue("task1", "type1", TaskPriority.NORMAL, {})
            queue.enqueue("task2", "type1", TaskPriority.NORMAL, {})
            queue.enqueue("task3", "type1", TaskPriority.NORMAL, {})

            # 验证FIFO顺序
            assert queue.dequeue().task_id == "task1"
            assert queue.dequeue().task_id == "task2"
            assert queue.dequeue().task_id == "task3"
        except ImportError:
            pytest.skip("优先级队列模块不可用")

    def test_priority_queue_update_priority(self):
        """测试更新优先级"""
        try:
            from src.core.orchestration.scheduler.performance.priority_queue import (
                PriorityTaskQueue, TaskPriority
            )

            queue = PriorityTaskQueue()

            queue.enqueue("task1", "type1", TaskPriority.LOW, {})
            queue.enqueue("task2", "type1", TaskPriority.HIGH, {})

            # 更新task1优先级为CRITICAL
            result = queue.update_priority("task1", TaskPriority.CRITICAL)
            assert result is True

            # 验证task1现在先出队
            assert queue.dequeue().task_id == "task1"
            assert queue.dequeue().task_id == "task2"
        except ImportError:
            pytest.skip("优先级队列模块不可用")

    def test_priority_queue_statistics(self):
        """测试优先级队列统计"""
        try:
            from src.core.orchestration.scheduler.performance.priority_queue import (
                PriorityTaskQueue, TaskPriority
            )

            queue = PriorityTaskQueue()

            queue.enqueue("task1", "type1", TaskPriority.HIGH, {})
            queue.enqueue("task2", "type1", TaskPriority.NORMAL, {})
            queue.enqueue("task3", "type1", TaskPriority.HIGH, {})

            stats = queue.get_statistics()
            assert stats['size'] == 3
            assert stats['total_enqueued'] == 3
            assert stats['priority_distribution']['HIGH'] == 2
            assert stats['priority_distribution']['NORMAL'] == 1
        except ImportError:
            pytest.skip("优先级队列模块不可用")


# ========== 批量处理器测试 ==========

class TestBatchProcessor:
    """测试批量处理器"""

    @pytest.mark.asyncio
    async def test_batch_processor_initialization(self):
        """测试批量处理器初始化"""
        try:
            from src.core.orchestration.scheduler.performance.batch_processor import (
                BatchProcessor, BatchConfig, BatchStrategy
            )

            config = BatchConfig(
                strategy=BatchStrategy.HYBRID,
                max_batch_size=50,
                max_wait_time_ms=500
            )
            processor = BatchProcessor(config)

            assert processor._config.max_batch_size == 50
            assert processor._config.max_wait_time_ms == 500
        except ImportError:
            pytest.skip("批量处理器模块不可用")

    @pytest.mark.asyncio
    async def test_batch_processor_submit(self):
        """测试批量处理器提交"""
        try:
            from src.core.orchestration.scheduler.performance.batch_processor import (
                BatchProcessor, BatchConfig, BatchStrategy
            )

            config = BatchConfig(
                strategy=BatchStrategy.SIZE_BASED,
                max_batch_size=3
            )
            processor = BatchProcessor(config)

            await processor.start()

            # 提交任务
            result1 = await processor.submit("t1", "data_collection", {"symbol": "AAPL"})
            result2 = await processor.submit("t2", "data_collection", {"symbol": "GOOGL"})

            assert result1 is True
            assert result2 is True
            assert processor.get_pending_count("data_collection") == 2

            await processor.stop()
        except ImportError:
            pytest.skip("批量处理器模块不可用")

    @pytest.mark.asyncio
    async def test_batch_processor_statistics(self):
        """测试批量处理器统计"""
        try:
            from src.core.orchestration.scheduler.performance.batch_processor import (
                BatchProcessor, BatchConfig
            )

            processor = BatchProcessor(BatchConfig())
            await processor.start()

            # 提交任务
            await processor.submit("t1", "type1", {})
            await processor.submit("t2", "type1", {})
            await processor.submit("t3", "type2", {})

            stats = processor.get_statistics()
            assert stats['pending_tasks'] == 3
            assert stats['batch_groups'] == 2

            await processor.stop()
        except ImportError:
            pytest.skip("批量处理器模块不可用")


# ========== 任务缓存测试 ==========

class TestTaskCache:
    """测试任务缓存"""

    @pytest.mark.asyncio
    async def test_cache_initialization(self):
        """测试缓存初始化"""
        try:
            from src.core.orchestration.scheduler.performance.task_cache import (
                TaskCache, CacheConfig
            )

            config = CacheConfig(max_size=500, default_ttl_seconds=60)
            cache = TaskCache(config)

            assert cache._config.max_size == 500
            assert cache._config.default_ttl_seconds == 60
        except ImportError:
            pytest.skip("任务缓存模块不可用")

    @pytest.mark.asyncio
    async def test_cache_set_get(self):
        """测试缓存设置和获取"""
        try:
            from src.core.orchestration.scheduler.performance.task_cache import TaskCache

            cache = TaskCache()
            await cache.start()

            # 设置缓存
            await cache.set("task_type1", {"key": "value1"}, {"result": "data1"})

            # 获取缓存
            result = await cache.get("task_type1", {"key": "value1"})
            assert result == {"result": "data1"}

            # 获取不存在的缓存
            result = await cache.get("task_type1", {"key": "nonexistent"})
            assert result is None

            await cache.stop()
        except ImportError:
            pytest.skip("任务缓存模块不可用")

    @pytest.mark.asyncio
    async def test_cache_ttl_expiration(self):
        """测试缓存TTL过期"""
        try:
            from src.core.orchestration.scheduler.performance.task_cache import TaskCache

            cache = TaskCache()
            await cache.start()

            # 设置短TTL缓存
            await cache.set("type1", {"k": "v"}, "data", ttl_seconds=0.01)

            # 立即获取应该存在
            result = await cache.get("type1", {"k": "v"})
            assert result == "data"

            # 等待过期
            await asyncio.sleep(0.1)

            # 过期后获取应该为None
            result = await cache.get("type1", {"k": "v"})
            assert result is None

            await cache.stop()
        except ImportError:
            pytest.skip("任务缓存模块不可用")

    @pytest.mark.asyncio
    async def test_cache_statistics(self):
        """测试缓存统计"""
        try:
            from src.core.orchestration.scheduler.performance.task_cache import TaskCache

            cache = TaskCache()
            await cache.start()

            # 设置和获取缓存
            await cache.set("type1", {"k": "v1"}, "data1")
            await cache.set("type1", {"k": "v2"}, "data2")

            await cache.get("type1", {"k": "v1"})  # hit
            await cache.get("type1", {"k": "v1"})  # hit
            await cache.get("type1", {"k": "v3"})  # miss

            stats = cache.get_statistics()
            assert stats['hits'] == 2
            assert stats['misses'] == 1
            assert stats['size'] == 2
            assert stats['hit_rate'] == 2/3

            await cache.stop()
        except ImportError:
            pytest.skip("任务缓存模块不可用")


# ========== Prometheus指标测试 ==========

class TestPrometheusMetrics:
    """测试Prometheus指标"""

    def test_metrics_initialization(self):
        """测试指标初始化"""
        try:
            from src.core.orchestration.scheduler.metrics.prometheus_metrics import (
                PrometheusMetrics, get_prometheus_metrics
            )

            metrics = PrometheusMetrics()

            # 验证标准计数器已注册
            assert "scheduler_tasks_submitted_total" in metrics._counters
            assert "scheduler_tasks_completed_total" in metrics._counters
            assert "scheduler_tasks_failed_total" in metrics._counters

            # 验证标准仪表盘已注册
            assert "scheduler_tasks_running" in metrics._gauges
            assert "scheduler_workers_active" in metrics._gauges
        except ImportError:
            pytest.skip("Prometheus指标模块不可用")

    def test_counter_increment(self):
        """测试计数器递增"""
        try:
            from src.core.orchestration.scheduler.metrics.prometheus_metrics import PrometheusMetrics

            metrics = PrometheusMetrics()

            # 递增计数器
            metrics.increment_counter("scheduler_tasks_submitted_total", 1)
            metrics.increment_counter("scheduler_tasks_submitted_total", 2)

            value = metrics.get_metric_value("scheduler_tasks_submitted_total")
            assert value == 3
        except ImportError:
            pytest.skip("Prometheus指标模块不可用")

    def test_gauge_set(self):
        """测试仪表盘设置"""
        try:
            from src.core.orchestration.scheduler.metrics.prometheus_metrics import PrometheusMetrics

            metrics = PrometheusMetrics()

            # 设置仪表盘
            metrics.set_gauge("scheduler_tasks_running", 5)
            assert metrics.get_metric_value("scheduler_tasks_running") == 5

            metrics.set_gauge("scheduler_tasks_running", 10)
            assert metrics.get_metric_value("scheduler_tasks_running") == 10
        except ImportError:
            pytest.skip("Prometheus指标模块不可用")

    def test_histogram_observe(self):
        """测试直方图观测"""
        try:
            from src.core.orchestration.scheduler.metrics.prometheus_metrics import PrometheusMetrics

            metrics = PrometheusMetrics()

            # 记录观测值
            metrics.observe_histogram("scheduler_task_execution_duration_seconds", 0.05)
            metrics.observe_histogram("scheduler_task_execution_duration_seconds", 0.1)
            metrics.observe_histogram("scheduler_task_execution_duration_seconds", 0.2)

            # 验证直方图数据
            all_metrics = metrics.get_all_metrics()
            histograms = all_metrics['histograms']
            assert "scheduler_task_execution_duration_seconds" in histograms
        except ImportError:
            pytest.skip("Prometheus指标模块不可用")

    def test_metrics_format_generation(self):
        """测试指标格式生成"""
        try:
            from src.core.orchestration.scheduler.metrics.prometheus_metrics import PrometheusMetrics

            metrics = PrometheusMetrics()
            metrics.increment_counter("scheduler_tasks_submitted_total", 5)
            metrics.set_gauge("scheduler_tasks_running", 3)

            output = metrics.generate_metrics()

            # 验证输出格式
            assert "# HELP" in output
            assert "# TYPE" in output
            assert "scheduler_tasks_submitted_total" in output
            assert "scheduler_tasks_running" in output
        except ImportError:
            pytest.skip("Prometheus指标模块不可用")


# ========== 加密模块测试 ==========

class TestEncryption:
    """测试加密模块"""

    def test_encryption_initialization(self):
        """测试加密初始化"""
        try:
            from src.core.orchestration.scheduler.security.encryption import (
                TaskEncryption, EncryptionConfig, EncryptionLevel
            )

            config = EncryptionConfig(
                enabled=True,
                level=EncryptionLevel.PAYLOAD
            )
            encryption = TaskEncryption(config)

            # 如果cryptography不可用，加密会被禁用
            info = encryption.get_encryption_info()
            assert info['level'] == "payload"
        except ImportError:
            pytest.skip("加密模块不可用")

    def test_encryption_sensitive_fields(self):
        """测试敏感字段加密"""
        try:
            from src.core.orchestration.scheduler.security.encryption import (
                TaskEncryption, EncryptionConfig, EncryptionLevel
            )

            config = EncryptionConfig(
                enabled=True,
                level=EncryptionLevel.PAYLOAD,
                sensitive_fields={"api_key", "password"}
            )
            encryption = TaskEncryption(config)

            payload = {
                "symbol": "AAPL",
                "api_key": "secret_key_123",
                "password": "my_password",
                "public_data": "visible"
            }

            # 加密payload
            encrypted = encryption.encrypt_payload(payload)

            # 如果加密启用，敏感字段应该被加密
            if encryption.is_enabled():
                assert "__encrypted__" in str(encrypted.get("api_key", "")) or encrypted.get("api_key") != "secret_key_123"
                assert "__encrypted__" in str(encrypted.get("password", "")) or encrypted.get("password") != "my_password"
                # 非敏感字段应该保持原样
                assert encrypted.get("symbol") == "AAPL"
                assert encrypted.get("public_data") == "visible"
        except ImportError:
            pytest.skip("加密模块不可用")

    def test_encryption_roundtrip(self):
        """测试加密解密往返"""
        try:
            from src.core.orchestration.scheduler.security.encryption import (
                TaskEncryption, EncryptionConfig, EncryptionLevel
            )

            config = EncryptionConfig(enabled=True, level=EncryptionLevel.PAYLOAD)
            encryption = TaskEncryption(config)

            if not encryption.is_enabled():
                pytest.skip("加密未启用")

            original = {"api_key": "test_secret", "data": "value"}

            encrypted = encryption.encrypt_payload(original)
            decrypted = encryption.decrypt_payload(encrypted)

            assert decrypted["api_key"] == "test_secret"
            assert decrypted["data"] == "value"
        except ImportError:
            pytest.skip("加密模块不可用")


# ========== 访问控制测试 ==========

class TestAccessControl:
    """测试访问控制"""

    def test_access_control_initialization(self):
        """测试访问控制初始化"""
        try:
            from src.core.orchestration.scheduler.security.access_control import (
                AccessControl, Role, Permission
            )

            ac = AccessControl()

            # 验证默认管理员存在
            admin = ac.get_user("admin")
            assert admin is not None
            assert admin.role == Role.ADMIN
            assert admin.has_permission(Permission.ADMIN)
        except ImportError:
            pytest.skip("访问控制模块不可用")

    def test_user_creation(self):
        """测试用户创建"""
        try:
            from src.core.orchestration.scheduler.security.access_control import (
                AccessControl, Role, Permission
            )

            ac = AccessControl()

            user = ac.create_user("user1", "Test User", Role.OPERATOR)

            assert user.id == "user1"
            assert user.name == "Test User"
            assert user.role == Role.OPERATOR
            assert user.has_permission(Permission.TASK_SUBMIT)
            assert user.has_permission(Permission.TASK_VIEW)
        except ImportError:
            pytest.skip("访问控制模块不可用")

    def test_permission_check(self):
        """测试权限检查"""
        try:
            from src.core.orchestration.scheduler.security.access_control import (
                AccessControl, Role, Permission
            )

            ac = AccessControl()

            # 创建不同角色的用户
            admin = ac.create_user("admin2", "Admin", Role.ADMIN)
            operator = ac.create_user("op1", "Operator", Role.OPERATOR)
            viewer = ac.create_user("viewer1", "Viewer", Role.VIEWER)

            # 验证权限
            assert ac.check_permission(admin, Permission.ADMIN) is True
            assert ac.check_permission(admin, Permission.TASK_SUBMIT) is True

            assert ac.check_permission(operator, Permission.TASK_SUBMIT) is True
            assert ac.check_permission(operator, Permission.ADMIN) is False

            assert ac.check_permission(viewer, Permission.TASK_VIEW) is True
            assert ac.check_permission(viewer, Permission.TASK_SUBMIT) is False
        except ImportError:
            pytest.skip("访问控制模块不可用")

    def test_api_key_creation_and_validation(self):
        """测试API密钥创建和验证"""
        try:
            from src.core.orchestration.scheduler.security.access_control import (
                AccessControl, Role
            )

            ac = AccessControl()

            # 创建API密钥
            key_id, raw_key = ac.create_api_key("test_key", Role.SERVICE)

            assert key_id.startswith("ak_")
            assert len(raw_key) > 0

            # 验证API密钥
            api_key = ac.validate_api_key(raw_key)
            assert api_key is not None
            assert api_key.key_id == key_id
            assert api_key.name == "test_key"

            # 验证无效密钥
            invalid_key = ac.validate_api_key("invalid_key")
            assert invalid_key is None
        except ImportError:
            pytest.skip("访问控制模块不可用")

    def test_api_key_revocation(self):
        """测试API密钥撤销"""
        try:
            from src.core.orchestration.scheduler.security.access_control import (
                AccessControl, Role
            )

            ac = AccessControl()

            # 创建并撤销API密钥
            key_id, raw_key = ac.create_api_key("test_key", Role.SERVICE)

            # 验证密钥有效
            assert ac.validate_api_key(raw_key) is not None

            # 撤销密钥
            result = ac.revoke_api_key(key_id)
            assert result is True

            # 验证密钥无效
            assert ac.validate_api_key(raw_key) is None
        except ImportError:
            pytest.skip("访问控制模块不可用")

    def test_audit_log(self):
        """测试审计日志"""
        try:
            from src.core.orchestration.scheduler.security.access_control import (
                AccessControl, Role
            )

            ac = AccessControl()

            # 执行一些操作
            ac.create_user("user_audit", "Audit User", Role.VIEWER)
            ac.create_api_key("audit_key", Role.SERVICE)

            # 获取审计日志
            logs = ac.get_audit_log(limit=10)

            assert len(logs) > 0
            assert any(log["action"] == "create_user" for log in logs)
            assert any(log["action"] == "create_api_key" for log in logs)
        except ImportError:
            pytest.skip("访问控制模块不可用")

    def test_statistics(self):
        """测试统计信息"""
        try:
            from src.core.orchestration.scheduler.security.access_control import (
                AccessControl, Role
            )

            ac = AccessControl()

            # 创建一些用户和API密钥
            ac.create_user("user_stat", "Stat User", Role.OPERATOR)
            ac.create_api_key("stat_key1", Role.SERVICE)
            ac.create_api_key("stat_key2", Role.VIEWER)

            stats = ac.get_statistics()

            assert stats["users_count"] >= 2  # 包括默认admin
            assert stats["api_keys_count"] == 2
            assert stats["active_api_keys"] == 2
        except ImportError:
            pytest.skip("访问控制模块不可用")


# ========== 性能模块集成测试 ==========

class TestPerformanceIntegration:
    """测试性能模块集成"""

    @pytest.mark.asyncio
    async def test_all_performance_modules(self):
        """测试所有性能模块可以一起工作"""
        try:
            from src.core.orchestration.scheduler.performance import (
                PriorityTaskQueue, TaskPriority,
                BatchProcessor, BatchConfig,
                TaskCache, CacheConfig
            )

            # 创建所有组件
            queue = PriorityTaskQueue()
            batch_processor = BatchProcessor(BatchConfig())
            cache = TaskCache(CacheConfig())

            # 启动需要启动的组件
            await batch_processor.start()
            await cache.start()

            # 测试队列
            queue.enqueue("t1", "type1", TaskPriority.HIGH, {})
            assert queue.size() == 1

            # 测试批量处理器
            await batch_processor.submit("t2", "type2", {})
            assert batch_processor.get_pending_count() == 1

            # 测试缓存
            await cache.set("type3", {"k": "v"}, "result")
            assert await cache.get("type3", {"k": "v"}) == "result"

            # 停止组件
            await batch_processor.stop()
            await cache.stop()

        except ImportError:
            pytest.skip("性能模块不可用")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
