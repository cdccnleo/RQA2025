"""
测试核心常量定义

覆盖 core_constants.py 中的常量定义
"""

import pytest
from src.infrastructure.config.constants.core_constants import *


class TestCoreConstants:
    """核心常量定义测试"""

    def test_service_container_constants(self):
        """测试服务容器相关常量"""
        assert isinstance(DEFAULT_SERVICE_TTL, int)
        assert DEFAULT_SERVICE_TTL == 3600

        assert isinstance(SERVICE_DISCOVERY_TIMEOUT, int)
        assert SERVICE_DISCOVERY_TIMEOUT == 30

        assert isinstance(SERVICE_HEALTH_CHECK_INTERVAL, int)
        assert SERVICE_HEALTH_CHECK_INTERVAL == 60

    def test_event_bus_constants(self):
        """测试事件总线相关常量"""
        assert isinstance(EVENT_BUS_BUFFER_SIZE, int)
        assert EVENT_BUS_BUFFER_SIZE == 10000

        assert isinstance(EVENT_PROCESSING_TIMEOUT, int)
        assert EVENT_PROCESSING_TIMEOUT == 30

        assert isinstance(MAX_EVENT_SUBSCRIBERS, int)
        assert MAX_EVENT_SUBSCRIBERS == 100

    def test_workflow_constants(self):
        """测试工作流相关常量"""
        assert isinstance(WORKFLOW_EXECUTION_TIMEOUT, int)
        assert WORKFLOW_EXECUTION_TIMEOUT == 1800

        assert isinstance(MAX_WORKFLOW_DEPTH, int)
        assert MAX_WORKFLOW_DEPTH == 10

        assert isinstance(WORKFLOW_CHECK_INTERVAL, int)
        assert WORKFLOW_CHECK_INTERVAL == 10

    def test_integration_constants(self):
        """测试集成相关常量"""
        assert isinstance(INTEGRATION_TIMEOUT, int)
        assert INTEGRATION_TIMEOUT == 60

        assert isinstance(ADAPTER_CONNECTION_POOL_SIZE, int)
        assert ADAPTER_CONNECTION_POOL_SIZE == 20

        assert isinstance(BRIDGE_BUFFER_SIZE, int)
        assert BRIDGE_BUFFER_SIZE == 5000

    def test_security_constants(self):
        """测试安全相关常量"""
        assert isinstance(AUTH_TOKEN_EXPIRY, int)
        assert AUTH_TOKEN_EXPIRY == 3600

        assert isinstance(ENCRYPTION_KEY_SIZE, int)
        assert ENCRYPTION_KEY_SIZE == 256

        assert isinstance(SECURITY_AUDIT_RETENTION, int)
        assert SECURITY_AUDIT_RETENTION == 90

    def test_performance_constants(self):
        """测试性能相关常量"""
        assert isinstance(PERFORMANCE_MONITOR_INTERVAL, int)
        assert PERFORMANCE_MONITOR_INTERVAL == 30

        assert isinstance(RESPONSE_TIME_TARGET_MS, int)
        assert RESPONSE_TIME_TARGET_MS == 200

        assert isinstance(CONCURRENCY_TARGET, int)
        assert CONCURRENCY_TARGET == 1000

    def test_cache_constants(self):
        """测试缓存相关常量"""
        assert isinstance(CORE_CACHE_SIZE, int)
        assert CORE_CACHE_SIZE == 10000

        assert isinstance(CACHE_EVICTION_INTERVAL, int)
        assert CACHE_EVICTION_INTERVAL == 300

    def test_monitoring_constants(self):
        """测试监控相关常量"""
        assert isinstance(METRICS_COLLECTION_INTERVAL, int)
        assert METRICS_COLLECTION_INTERVAL == 15

        assert isinstance(ALERT_THRESHOLD_HIGH, float)
        assert ALERT_THRESHOLD_HIGH == 0.9

        assert isinstance(ALERT_THRESHOLD_MEDIUM, float)
        assert ALERT_THRESHOLD_MEDIUM == 0.7

    def test_async_constants(self):
        """测试异步处理相关常量"""
        assert isinstance(ASYNC_TASK_QUEUE_SIZE, int)
        assert ASYNC_TASK_QUEUE_SIZE == 1000

        assert isinstance(ASYNC_WORKER_COUNT, int)
        assert ASYNC_WORKER_COUNT == 4

        assert isinstance(ASYNC_TASK_TIMEOUT, int)
        assert ASYNC_TASK_TIMEOUT == 600

    def test_all_constants_positive(self):
        """测试所有常量都是正数"""
        constants = [
            DEFAULT_SERVICE_TTL,
            SERVICE_DISCOVERY_TIMEOUT,
            SERVICE_HEALTH_CHECK_INTERVAL,
            EVENT_BUS_BUFFER_SIZE,
            EVENT_PROCESSING_TIMEOUT,
            MAX_EVENT_SUBSCRIBERS,
            WORKFLOW_EXECUTION_TIMEOUT,
            MAX_WORKFLOW_DEPTH,
            WORKFLOW_CHECK_INTERVAL,
            INTEGRATION_TIMEOUT,
            ADAPTER_CONNECTION_POOL_SIZE,
            BRIDGE_BUFFER_SIZE,
            AUTH_TOKEN_EXPIRY,
            ENCRYPTION_KEY_SIZE,
            SECURITY_AUDIT_RETENTION,
            PERFORMANCE_MONITOR_INTERVAL,
            RESPONSE_TIME_TARGET_MS,
            CONCURRENCY_TARGET,
            CORE_CACHE_SIZE,
            CACHE_EVICTION_INTERVAL,
            METRICS_COLLECTION_INTERVAL,
            ALERT_THRESHOLD_HIGH,
            ALERT_THRESHOLD_MEDIUM,
            ASYNC_TASK_QUEUE_SIZE,
            ASYNC_WORKER_COUNT,
            ASYNC_TASK_TIMEOUT
        ]

        for const in constants:
            if isinstance(const, (int, float)):
                assert const > 0, f"Constant {const} should be positive"

    def test_alert_thresholds_valid(self):
        """测试告警阈值在有效范围内"""
        assert 0 < ALERT_THRESHOLD_HIGH <= 1
        assert 0 < ALERT_THRESHOLD_MEDIUM <= 1
        assert ALERT_THRESHOLD_HIGH > ALERT_THRESHOLD_MEDIUM

    def test_timeout_values_reasonable(self):
        """测试超时值合理性"""
        timeouts = [
            SERVICE_DISCOVERY_TIMEOUT,
            EVENT_PROCESSING_TIMEOUT,
            WORKFLOW_EXECUTION_TIMEOUT,
            INTEGRATION_TIMEOUT,
            ASYNC_TASK_TIMEOUT
        ]

        for timeout in timeouts:
            assert timeout >= 1, f"Timeout {timeout} should be at least 1 second"
            assert timeout <= 3600, f"Timeout {timeout} should not exceed 1 hour"

    def test_interval_values_reasonable(self):
        """测试间隔值合理性"""
        intervals = [
            SERVICE_HEALTH_CHECK_INTERVAL,
            WORKFLOW_CHECK_INTERVAL,
            PERFORMANCE_MONITOR_INTERVAL,
            CACHE_EVICTION_INTERVAL,
            METRICS_COLLECTION_INTERVAL
        ]

        for interval in intervals:
            assert interval >= 1, f"Interval {interval} should be at least 1 second"
            assert interval <= 3600, f"Interval {interval} should not exceed 1 hour"

    def test_pool_and_buffer_sizes_reasonable(self):
        """测试池和缓冲区大小合理性"""
        sizes = [
            EVENT_BUS_BUFFER_SIZE,
            ADAPTER_CONNECTION_POOL_SIZE,
            BRIDGE_BUFFER_SIZE,
            CORE_CACHE_SIZE,
            ASYNC_TASK_QUEUE_SIZE
        ]

        for size in sizes:
            assert size >= 1, f"Size {size} should be at least 1"
            assert size <= 100000, f"Size {size} should not exceed reasonable limit"

    def test_concurrency_values_reasonable(self):
        """测试并发值合理性"""
        assert CONCURRENCY_TARGET >= 1
        assert CONCURRENCY_TARGET <= 10000  # Reasonable upper limit

        assert ASYNC_WORKER_COUNT >= 1
        assert ASYNC_WORKER_COUNT <= 100  # Reasonable upper limit

    def test_constants_are_integers_or_floats(self):
        """测试常量是整数或浮点数类型"""
        all_constants = [
            DEFAULT_SERVICE_TTL, SERVICE_DISCOVERY_TIMEOUT, SERVICE_HEALTH_CHECK_INTERVAL,
            EVENT_BUS_BUFFER_SIZE, EVENT_PROCESSING_TIMEOUT, MAX_EVENT_SUBSCRIBERS,
            WORKFLOW_EXECUTION_TIMEOUT, MAX_WORKFLOW_DEPTH, WORKFLOW_CHECK_INTERVAL,
            INTEGRATION_TIMEOUT, ADAPTER_CONNECTION_POOL_SIZE, BRIDGE_BUFFER_SIZE,
            AUTH_TOKEN_EXPIRY, ENCRYPTION_KEY_SIZE, SECURITY_AUDIT_RETENTION,
            PERFORMANCE_MONITOR_INTERVAL, RESPONSE_TIME_TARGET_MS, CONCURRENCY_TARGET,
            CORE_CACHE_SIZE, CACHE_EVICTION_INTERVAL,
            METRICS_COLLECTION_INTERVAL, ALERT_THRESHOLD_HIGH, ALERT_THRESHOLD_MEDIUM,
            ASYNC_TASK_QUEUE_SIZE, ASYNC_WORKER_COUNT, ASYNC_TASK_TIMEOUT
        ]

        for const in all_constants:
            assert isinstance(const, (int, float)), f"Constant {const} should be int or float"
