"""
RQA2025 基础设施服务接口单元测试

测试基础设施层的接口定义和协议实现。
"""

import unittest
from unittest.mock import Mock, MagicMock
from typing import Any, Dict, Optional
from datetime import datetime

from src.infrastructure.interfaces.infrastructure_services import (
    # 接口
    IConfigManager,
    ICacheService,
    IMultiLevelCache,
    ILogger,
    ILogManager,
    IMonitor,
    ISecurityManager,
    IHealthChecker,
    IResourceManager,
    IEventBus,
    IServiceContainer,
    IInfrastructureServiceProvider,
    EventHandler,

    # 数据结构
    CacheEntry,
    LogEntry,
    MetricData,
    UserCredentials,
    SecurityToken,
    HealthCheckResult,
    ResourceQuota,
    Event,
    InfrastructureServiceStatus,
    LogLevel,
)


class TestInfrastructureInterfaces(unittest.TestCase):
    """基础设施接口测试"""

    def setUp(self):
        """测试前准备"""
        self.mock_config = Mock(spec=IConfigManager)
        self.mock_cache = Mock(spec=ICacheService)
        self.mock_logger = Mock(spec=ILogger)
        self.mock_monitor = Mock(spec=IMonitor)
        self.mock_security = Mock(spec=ISecurityManager)
        self.mock_health = Mock(spec=IHealthChecker)
        self.mock_resource = Mock(spec=IResourceManager)
        self.mock_event_bus = Mock(spec=IEventBus)
        self.mock_container = Mock(spec=IServiceContainer)

    def test_config_manager_interface(self):
        """测试配置管理器接口"""
        # 设置mock行为
        self.mock_config.get.return_value = "test_value"
        self.mock_config.set.return_value = True
        self.mock_config.get_section.return_value = {"key": "value"}
        self.mock_config.reload.return_value = True
        self.mock_config.validate_config.return_value = []

        # 测试接口方法
        self.assertEqual(self.mock_config.get("test_key"), "test_value")
        self.assertTrue(self.mock_config.set("test_key", "test_value"))
        self.assertEqual(self.mock_config.get_section("test"), {"key": "value"})
        self.assertTrue(self.mock_config.reload())
        self.assertEqual(self.mock_config.validate_config(), [])

        # 验证方法调用
        self.mock_config.get.assert_called_with("test_key")
        self.mock_config.set.assert_called_with("test_key", "test_value")
        self.mock_config.get_section.assert_called_with("test")

    def test_cache_service_interface(self):
        """测试缓存服务接口"""
        # 设置mock行为
        self.mock_cache.get.return_value = "cached_value"
        self.mock_cache.set.return_value = True
        self.mock_cache.delete.return_value = True
        self.mock_cache.exists.return_value = True
        self.mock_cache.clear.return_value = True
        self.mock_cache.get_stats.return_value = {"hits": 10, "misses": 5}

        # 测试接口方法
        self.assertEqual(self.mock_cache.get("test_key"), "cached_value")
        self.assertTrue(self.mock_cache.set("test_key", "test_value"))
        self.assertTrue(self.mock_cache.delete("test_key"))
        self.assertTrue(self.mock_cache.exists("test_key"))
        self.assertTrue(self.mock_cache.clear())
        self.assertEqual(self.mock_cache.get_stats(), {"hits": 10, "misses": 5})

    def test_logger_interface(self):
        """测试日志器接口"""
        # 测试所有日志级别方法
        self.mock_logger.debug("Debug message")
        self.mock_logger.info("Info message")
        self.mock_logger.warning("Warning message")
        self.mock_logger.error("Error message")
        self.mock_logger.critical("Critical message")

        # 测试通用日志方法
        self.mock_logger.log(LogLevel.INFO, "Log message")
        self.mock_logger.is_enabled_for(LogLevel.DEBUG)

        # 验证方法调用
        self.mock_logger.debug.assert_called_with("Debug message")
        self.mock_logger.info.assert_called_with("Info message")
        self.mock_logger.log.assert_called_with(LogLevel.INFO, "Log message")

    def test_monitor_interface(self):
        """测试监控器接口"""
        # 测试指标记录
        self.mock_monitor.record_metric("test_metric", 42.0, {"tag": "value"})
        self.mock_monitor.increment_counter("test_counter", 5, {"env": "test"})
        self.mock_monitor.record_histogram("test_histogram", 1.5, {"service": "test"})

        # 测试计时器
        self.mock_monitor.start_timer.return_value = "timer_123"
        self.mock_monitor.stop_timer.return_value = 0.123

        timer_id = self.mock_monitor.start_timer("test_timer")
        duration = self.mock_monitor.stop_timer(timer_id)

        self.assertEqual(timer_id, "timer_123")
        self.assertEqual(duration, 0.123)

    def test_monitor_get_metrics(self):
        """测试监控器获取指标"""
        mock_metric = Mock(spec=MetricData)
        mock_metric.name = "test_metric"
        mock_metric.value = 42.0
        
        self.mock_monitor.get_metrics.return_value = [mock_metric]
        
        metrics = self.mock_monitor.get_metrics("test_*")
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0].name, "test_metric")
        
        # 测试无模式匹配
        all_metrics = self.mock_monitor.get_metrics()
        self.assertEqual(len(all_metrics), 1)

    def test_security_manager_interface(self):
        """测试安全管理器接口"""
        # Mock安全令牌
        mock_token = Mock(spec=SecurityToken)
        mock_token.token = "test_token"
        mock_token.user_id = "user123"

        # 设置mock行为
        self.mock_security.authenticate.return_value = mock_token
        self.mock_security.validate_token.return_value = mock_token
        self.mock_security.authorize.return_value = True
        self.mock_security.create_user.return_value = True
        self.mock_security.update_user.return_value = True
        self.mock_security.delete_user.return_value = True
        self.mock_security.get_user_permissions.return_value = ["read", "write"]

        # 测试认证
        token = self.mock_security.authenticate("user", "password")
        self.assertEqual(token.token, "test_token")

        # 测试授权
        self.assertTrue(self.mock_security.authorize("token", "resource", "action"))

        # 测试用户管理
        self.assertTrue(self.mock_security.create_user(Mock()))
        self.assertTrue(self.mock_security.update_user("user", {"email": "new@email.com"}))
        self.assertTrue(self.mock_security.delete_user("user"))
        self.assertEqual(self.mock_security.get_user_permissions("user"), ["read", "write"])

    def test_health_checker_interface(self):
        """测试健康检查器接口"""
        # Mock健康检查结果
        mock_result = Mock(spec=HealthCheckResult)
        mock_result.status = "healthy"
        mock_result.response_time = 0.001

        # 设置mock行为
        self.mock_health.check_health.return_value = mock_result
        self.mock_health.is_healthy.return_value = True
        self.mock_health.get_health_history.return_value = [mock_result]
        self.mock_health.get_detailed_status.return_value = {"status": "healthy", "details": {}}

        # 测试健康检查
        result = self.mock_health.check_health()
        self.assertEqual(result.status, "healthy")

        self.assertTrue(self.mock_health.is_healthy())
        self.assertEqual(len(self.mock_health.get_health_history()), 1)
        
        # 测试详细状态
        detailed_status = self.mock_health.get_detailed_status()
        self.assertIn("status", detailed_status)

    def test_resource_manager_interface(self):
        """测试资源管理器接口"""
        # Mock资源配额
        mock_quota = Mock(spec=ResourceQuota)
        mock_quota.resource_type = "cpu"
        mock_quota.limit = 100.0
        mock_quota.used = 75.0

        # 设置mock行为
        self.mock_resource.get_resource_usage.return_value = mock_quota
        self.mock_resource.set_resource_limit.return_value = True
        self.mock_resource.check_resource_available.return_value = True
        self.mock_resource.allocate_resource.return_value = True
        self.mock_resource.release_resource.return_value = True
        self.mock_resource.get_all_resource_quotas.return_value = {"cpu": mock_quota}

        # 测试资源管理
        quota = self.mock_resource.get_resource_usage("cpu")
        self.assertEqual(quota.resource_type, "cpu")

        self.assertTrue(self.mock_resource.set_resource_limit("cpu", 100.0, "cores"))
        self.assertTrue(self.mock_resource.check_resource_available("cpu", 25.0))
        self.assertTrue(self.mock_resource.allocate_resource("cpu", 10.0))
        self.assertTrue(self.mock_resource.release_resource("cpu", 5.0))
        
        # 测试获取所有配额
        all_quotas = self.mock_resource.get_all_resource_quotas()
        self.assertIn("cpu", all_quotas)

    def test_event_bus_interface(self):
        """测试事件总线接口"""
        # Mock事件
        mock_event = Mock(spec=Event)
        mock_event.event_id = "event_123"
        mock_event.event_type = "test_event"

        # Mock事件处理器
        mock_handler = Mock(spec=EventHandler)
        
        # 设置mock行为
        self.mock_event_bus.publish.return_value = True
        self.mock_event_bus.subscribe.return_value = "sub_123"
        self.mock_event_bus.unsubscribe.return_value = True
        self.mock_event_bus.publish_async.return_value = "task_123"
        self.mock_event_bus.get_event_history.return_value = [mock_event]

        # 测试事件发布
        self.assertTrue(self.mock_event_bus.publish(mock_event))

        # 测试订阅管理
        sub_id = self.mock_event_bus.subscribe("test_event", mock_handler)
        self.assertEqual(sub_id, "sub_123")
        self.assertTrue(self.mock_event_bus.unsubscribe(sub_id))

        # 测试异步发布
        task_id = self.mock_event_bus.publish_async(mock_event)
        self.assertEqual(task_id, "task_123")
        
        # 测试事件历史
        history = self.mock_event_bus.get_event_history("test_event", limit=10)
        self.assertEqual(len(history), 1)

    def test_service_container_interface(self):
        """测试服务容器接口"""
        # Mock服务实例
        mock_service = Mock()

        # 设置mock行为
        self.mock_container.register.return_value = None
        self.mock_container.register_instance.return_value = None
        self.mock_container.resolve.return_value = mock_service
        self.mock_container.has_service.return_value = True
        self.mock_container.unregister.return_value = True
        self.mock_container.get_registered_services.return_value = [str]

        # 测试服务注册
        self.mock_container.register(str, str)
        self.mock_container.register_instance(str, "instance")

        # 测试服务解析
        resolved = self.mock_container.resolve(str)
        self.assertEqual(resolved, mock_service)

        # 测试服务查询
        self.assertTrue(self.mock_container.has_service(str))
        self.assertTrue(self.mock_container.unregister(str))
        self.assertEqual(self.mock_container.get_registered_services(), [str])

    def test_multi_level_cache_interface(self):
        """测试多级缓存接口"""
        mock_multi_cache = Mock(spec=IMultiLevelCache)
        mock_multi_cache.get_from_level.return_value = "cached_value"
        mock_multi_cache.set_to_level.return_value = True
        mock_multi_cache.invalidate_level.return_value = True
        mock_multi_cache.get_cache_levels.return_value = ["L1", "L2", "L3"]

        # 测试多级缓存操作
        self.assertEqual(mock_multi_cache.get_from_level(1, "key"), "cached_value")
        self.assertTrue(mock_multi_cache.set_to_level(1, "key", "value", ttl=300))
        self.assertTrue(mock_multi_cache.invalidate_level(1, "key"))
        self.assertEqual(mock_multi_cache.get_cache_levels(), ["L1", "L2", "L3"])

    def test_log_manager_interface(self):
        """测试日志管理器接口"""
        mock_log_manager = Mock(spec=ILogManager)
        mock_logger = Mock(spec=ILogger)
        
        mock_log_manager.get_logger.return_value = mock_logger
        mock_log_manager.configure_logger.return_value = True
        mock_log_manager.get_all_loggers.return_value = {"logger1": mock_logger}

        # 测试日志管理器操作
        logger = mock_log_manager.get_logger("test_logger")
        self.assertEqual(logger, mock_logger)
        self.assertTrue(mock_log_manager.configure_logger("test_logger", {"level": "INFO"}))
        self.assertEqual(mock_log_manager.get_all_loggers(), {"logger1": mock_logger})


class TestInfrastructureServiceProvider(unittest.TestCase):
    """基础设施服务提供者测试"""

    def setUp(self):
        """测试前准备"""
        self.mock_provider = Mock(spec=IInfrastructureServiceProvider)

        # Mock各个服务
        self.mock_config = Mock(spec=IConfigManager)
        self.mock_cache = Mock(spec=ICacheService)
        self.mock_logger = Mock(spec=ILogger)
        self.mock_monitor = Mock(spec=IMonitor)

        # 设置服务属性
        self.mock_provider.config_manager = self.mock_config
        self.mock_provider.cache_service = self.mock_cache
        self.mock_provider.logger = self.mock_logger
        self.mock_provider.monitor = self.mock_monitor

    def test_service_provider_properties(self):
        """测试服务提供者属性访问"""
        self.assertEqual(self.mock_provider.config_manager, self.mock_config)
        self.assertEqual(self.mock_provider.cache_service, self.mock_cache)
        self.assertEqual(self.mock_provider.logger, self.mock_logger)
        self.assertEqual(self.mock_provider.monitor, self.mock_monitor)

    def test_service_provider_status(self):
        """测试服务提供者状态管理"""
        self.mock_provider.get_service_status.return_value = InfrastructureServiceStatus.RUNNING
        self.mock_provider.initialize_all_services.return_value = True
        self.mock_provider.shutdown_all_services.return_value = True

        # 测试状态获取
        status = self.mock_provider.get_service_status()
        self.assertEqual(status, InfrastructureServiceStatus.RUNNING)

        # 测试服务初始化
        self.assertTrue(self.mock_provider.initialize_all_services())
        self.assertTrue(self.mock_provider.shutdown_all_services())

    def test_service_provider_all_properties(self):
        """测试服务提供者所有属性"""
        # Mock所有服务
        mock_security = Mock(spec=ISecurityManager)
        mock_health = Mock(spec=IHealthChecker)
        mock_resource = Mock(spec=IResourceManager)
        mock_event_bus = Mock(spec=IEventBus)
        mock_container = Mock(spec=IServiceContainer)

        # 设置所有属性
        self.mock_provider.security_manager = mock_security
        self.mock_provider.health_checker = mock_health
        self.mock_provider.resource_manager = mock_resource
        self.mock_provider.event_bus = mock_event_bus
        self.mock_provider.service_container = mock_container

        # 验证所有属性
        self.assertEqual(self.mock_provider.security_manager, mock_security)
        self.assertEqual(self.mock_provider.health_checker, mock_health)
        self.assertEqual(self.mock_provider.resource_manager, mock_resource)
        self.assertEqual(self.mock_provider.event_bus, mock_event_bus)
        self.assertEqual(self.mock_provider.service_container, mock_container)

    def test_service_provider_health_report(self):
        """测试服务提供者健康报告"""
        mock_result = Mock(spec=HealthCheckResult)
        mock_result.service_name = "test_service"
        mock_result.status = "healthy"
        
        self.mock_provider.get_service_health_report.return_value = {
            "config": mock_result,
            "cache": mock_result
        }
        
        report = self.mock_provider.get_service_health_report()
        self.assertIn("config", report)
        self.assertIn("cache", report)


class TestDataStructures(unittest.TestCase):
    """数据结构测试"""

    def test_cache_entry_creation(self):
        """测试缓存条目创建"""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            ttl=300
        )

        self.assertEqual(entry.key, "test_key")
        self.assertEqual(entry.value, "test_value")
        self.assertEqual(entry.ttl, 300)
        self.assertIsInstance(entry.created_at, datetime)
        self.assertIsInstance(entry.accessed_at, datetime)
        self.assertEqual(entry.access_count, 0)

    def test_cache_entry_post_init(self):
        """测试缓存条目__post_init__方法"""
        # 测试自动设置时间戳
        entry1 = CacheEntry(key="key1", value="value1")
        self.assertIsNotNone(entry1.created_at)
        self.assertIsNotNone(entry1.accessed_at)
        
        # 测试手动设置时间戳
        custom_time = datetime(2024, 1, 1, 12, 0, 0)
        entry2 = CacheEntry(
            key="key2",
            value="value2",
            created_at=custom_time,
            accessed_at=custom_time
        )
        self.assertEqual(entry2.created_at, custom_time)
        self.assertEqual(entry2.accessed_at, custom_time)

    def test_log_entry_creation(self):
        """测试日志条目创建"""
        entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            logger_name="test_logger",
            message="Test message"
        )

        self.assertEqual(entry.level, LogLevel.INFO)
        self.assertEqual(entry.logger_name, "test_logger")
        self.assertEqual(entry.message, "Test message")

    def test_log_entry_with_optional_fields(self):
        """测试日志条目可选字段"""
        entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.ERROR,
            logger_name="test_logger",
            message="Error message",
            module="test_module",
            function="test_function",
            line=42,
            exception=ValueError("Test error"),
            extra_data={"key": "value"}
        )

        self.assertEqual(entry.module, "test_module")
        self.assertEqual(entry.function, "test_function")
        self.assertEqual(entry.line, 42)
        self.assertIsInstance(entry.exception, ValueError)
        self.assertEqual(entry.extra_data, {"key": "value"})

    def test_metric_data_creation(self):
        """测试监控指标数据创建"""
        metric = MetricData(
            name="test_metric",
            value=42.5,
            timestamp=datetime.now(),
            tags={"service": "test"}
        )

        self.assertEqual(metric.name, "test_metric")
        self.assertEqual(metric.value, 42.5)
        self.assertEqual(metric.tags, {"service": "test"})

    def test_metric_data_with_metadata(self):
        """测试监控指标数据元数据"""
        metric = MetricData(
            name="test_metric",
            value=100,
            timestamp=datetime.now(),
            tags={"env": "prod"},
            metadata={"unit": "ms", "threshold": 200}
        )

        self.assertEqual(metric.metadata, {"unit": "ms", "threshold": 200})

    def test_health_check_result_creation(self):
        """测试健康检查结果创建"""
        result = HealthCheckResult(
            service_name="test_service",
            status="healthy",
            response_time=0.001,
            message="Service is healthy"
        )

        self.assertEqual(result.service_name, "test_service")
        self.assertEqual(result.status, "healthy")
        self.assertEqual(result.response_time, 0.001)
        self.assertIsNotNone(result.timestamp)

    def test_health_check_result_post_init(self):
        """测试健康检查结果__post_init__方法"""
        # 测试自动设置时间戳
        result1 = HealthCheckResult(
            service_name="service1",
            status="healthy",
            response_time=0.001
        )
        self.assertIsNotNone(result1.timestamp)
        
        # 测试手动设置时间戳
        custom_time = datetime(2024, 1, 1, 12, 0, 0)
        result2 = HealthCheckResult(
            service_name="service2",
            status="unhealthy",
            response_time=0.5,
            timestamp=custom_time
        )
        self.assertEqual(result2.timestamp, custom_time)

    def test_resource_quota_creation(self):
        """测试资源配额创建"""
        quota = ResourceQuota(
            resource_type="cpu",
            limit=100.0,
            used=75.0,
            unit="cores"
        )

        self.assertEqual(quota.resource_type, "cpu")
        self.assertEqual(quota.limit, 100.0)
        self.assertEqual(quota.used, 75.0)
        self.assertEqual(quota.unit, "cores")
        self.assertIsNotNone(quota.last_updated)

    def test_resource_quota_post_init(self):
        """测试资源配额__post_init__方法"""
        # 测试自动设置时间戳
        quota1 = ResourceQuota(
            resource_type="memory",
            limit=1024.0,
            used=512.0,
            unit="MB"
        )
        self.assertIsNotNone(quota1.last_updated)
        
        # 测试手动设置时间戳
        custom_time = datetime(2024, 1, 1, 12, 0, 0)
        quota2 = ResourceQuota(
            resource_type="disk",
            limit=500.0,
            used=250.0,
            unit="GB",
            last_updated=custom_time
        )
        self.assertEqual(quota2.last_updated, custom_time)

    def test_event_creation(self):
        """测试事件创建"""
        event = Event(
            event_id="event_123",
            event_type="test_event",
            payload={"key": "value"},
            source="test_source"
        )

        self.assertEqual(event.event_id, "event_123")
        self.assertEqual(event.event_type, "test_event")
        self.assertEqual(event.payload, {"key": "value"})
        self.assertEqual(event.source, "test_source")
        self.assertIsNotNone(event.timestamp)

    def test_event_post_init(self):
        """测试事件__post_init__方法"""
        # 测试自动设置时间戳
        event1 = Event(
            event_id="event_1",
            event_type="test.event",
            payload={"data": "value"},
            source="test_source"
        )
        self.assertIsNotNone(event1.timestamp)
        
        # 测试手动设置时间戳
        custom_time = datetime(2024, 1, 1, 12, 0, 0)
        event2 = Event(
            event_id="event_2",
            event_type="test.event",
            payload={"data": "value"},
            source="test_source",
            timestamp=custom_time
        )
        self.assertEqual(event2.timestamp, custom_time)
        
        # 测试可选字段
        event3 = Event(
            event_id="event_3",
            event_type="test.event",
            payload={},
            source="test_source",
            correlation_id="corr_123",
            headers={"header1": "value1"}
        )
        self.assertEqual(event3.correlation_id, "corr_123")
        self.assertEqual(event3.headers, {"header1": "value1"})

    def test_user_credentials_creation(self):
        """测试用户凭据创建"""
        credentials = UserCredentials(
            username="test_user",
            password_hash="hash123",
            salt="salt123",
            roles=["admin", "user"],
            permissions=["read", "write"]
        )

        self.assertEqual(credentials.username, "test_user")
        self.assertEqual(credentials.password_hash, "hash123")
        self.assertEqual(credentials.salt, "salt123")
        self.assertEqual(credentials.roles, ["admin", "user"])
        self.assertEqual(credentials.permissions, ["read", "write"])
        self.assertTrue(credentials.is_active)
        self.assertIsNotNone(credentials.created_at)

    def test_user_credentials_post_init(self):
        """测试用户凭据__post_init__方法"""
        # 测试自动设置时间戳
        cred1 = UserCredentials(
            username="user1",
            password_hash="hash1",
            salt="salt1",
            roles=[],
            permissions=[]
        )
        self.assertIsNotNone(cred1.created_at)
        
        # 测试手动设置时间戳
        custom_time = datetime(2024, 1, 1, 12, 0, 0)
        cred2 = UserCredentials(
            username="user2",
            password_hash="hash2",
            salt="salt2",
            roles=["user"],
            permissions=["read"],
            created_at=custom_time,
            is_active=False
        )
        self.assertEqual(cred2.created_at, custom_time)
        self.assertFalse(cred2.is_active)

    def test_security_token_creation(self):
        """测试安全令牌创建"""
        issued_time = datetime(2024, 1, 1, 12, 0, 0)
        expires_time = datetime(2024, 1, 1, 13, 0, 0)
        
        token = SecurityToken(
            token="token123",
            user_id="user123",
            issued_at=issued_time,
            expires_at=expires_time,
            permissions=["read", "write"]
        )

        self.assertEqual(token.token, "token123")
        self.assertEqual(token.user_id, "user123")
        self.assertEqual(token.issued_at, issued_time)
        self.assertEqual(token.expires_at, expires_time)
        self.assertEqual(token.permissions, ["read", "write"])


class TestEnums(unittest.TestCase):
    """枚举测试"""

    def test_infrastructure_service_status_values(self):
        """测试基础设施服务状态枚举值"""
        self.assertEqual(InfrastructureServiceStatus.INITIALIZING.value, "initializing")
        self.assertEqual(InfrastructureServiceStatus.RUNNING.value, "running")
        self.assertEqual(InfrastructureServiceStatus.DEGRADED.value, "degraded")
        self.assertEqual(InfrastructureServiceStatus.STOPPED.value, "stopped")
        self.assertEqual(InfrastructureServiceStatus.ERROR.value, "error")

    def test_log_level_values(self):
        """测试日志级别枚举值"""
        self.assertEqual(LogLevel.DEBUG.value, "DEBUG")
        self.assertEqual(LogLevel.INFO.value, "INFO")
        self.assertEqual(LogLevel.WARNING.value, "WARNING")
        self.assertEqual(LogLevel.ERROR.value, "ERROR")
        self.assertEqual(LogLevel.CRITICAL.value, "CRITICAL")

    def test_infrastructure_service_status_comparison(self):
        """测试基础设施服务状态比较"""
        status1 = InfrastructureServiceStatus.RUNNING
        status2 = InfrastructureServiceStatus.RUNNING
        status3 = InfrastructureServiceStatus.STOPPED
        
        self.assertEqual(status1, status2)
        self.assertNotEqual(status1, status3)

    def test_log_level_comparison(self):
        """测试日志级别比较"""
        level1 = LogLevel.INFO
        level2 = LogLevel.INFO
        level3 = LogLevel.ERROR
        
        self.assertEqual(level1, level2)
        self.assertNotEqual(level1, level3)


if __name__ == '__main__':
    unittest.main()
