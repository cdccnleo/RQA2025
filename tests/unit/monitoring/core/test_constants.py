#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控层常量测试
测试constants.py中定义的常量值
"""

import sys
import importlib
from pathlib import Path
import pytest

# 确保Python路径正确配置
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
project_root_str = str(project_root)
src_path_str = str(project_root / "src")

if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

# 动态导入模块
try:
    constants_module = importlib.import_module('src.monitoring.core.constants')
    DEFAULT_MONITORING_INTERVAL = getattr(constants_module, 'DEFAULT_MONITORING_INTERVAL', None)
    DEFAULT_RETENTION_HOURS = getattr(constants_module, 'DEFAULT_RETENTION_HOURS', None)
    ALERT_CHECK_INTERVAL = getattr(constants_module, 'ALERT_CHECK_INTERVAL', None)
    CPU_THRESHOLD_HIGH = getattr(constants_module, 'CPU_THRESHOLD_HIGH', None)
    CPU_THRESHOLD_CRITICAL = getattr(constants_module, 'CPU_THRESHOLD_CRITICAL', None)
    MEMORY_THRESHOLD_HIGH = getattr(constants_module, 'MEMORY_THRESHOLD_HIGH', None)
    MEMORY_THRESHOLD_CRITICAL = getattr(constants_module, 'MEMORY_THRESHOLD_CRITICAL', None)
    RESPONSE_TIME_HIGH = getattr(constants_module, 'RESPONSE_TIME_HIGH', None)
    RESPONSE_TIME_CRITICAL = getattr(constants_module, 'RESPONSE_TIME_CRITICAL', None)
    ERROR_RATE_HIGH = getattr(constants_module, 'ERROR_RATE_HIGH', None)
    ERROR_RATE_CRITICAL = getattr(constants_module, 'ERROR_RATE_CRITICAL', None)
    MAX_METRICS_BUFFER = getattr(constants_module, 'MAX_METRICS_BUFFER', None)
    MAX_ALERT_BUFFER = getattr(constants_module, 'MAX_ALERT_BUFFER', None)
    MAX_LOG_ENTRIES = getattr(constants_module, 'MAX_LOG_ENTRIES', None)
    MAX_RETRY_ATTEMPTS = getattr(constants_module, 'MAX_RETRY_ATTEMPTS', None)
    RETRY_DELAY_SECONDS = getattr(constants_module, 'RETRY_DELAY_SECONDS', None)
    OPERATION_TIMEOUT = getattr(constants_module, 'OPERATION_TIMEOUT', None)
    HEALTH_CHECK_TIMEOUT = getattr(constants_module, 'HEALTH_CHECK_TIMEOUT', None)
    HEALTH_SCORE_THRESHOLD = getattr(constants_module, 'HEALTH_SCORE_THRESHOLD', None)
    DEFAULT_MONITORING_PORT = getattr(constants_module, 'DEFAULT_MONITORING_PORT', None)
    DEFAULT_METRICS_PORT = getattr(constants_module, 'DEFAULT_METRICS_PORT', None)
    DEFAULT_BATCH_SIZE = getattr(constants_module, 'DEFAULT_BATCH_SIZE', None)
    MAX_BATCH_SIZE = getattr(constants_module, 'MAX_BATCH_SIZE', None)
    CACHE_TTL_DEFAULT = getattr(constants_module, 'CACHE_TTL_DEFAULT', None)
    CACHE_MAX_SIZE = getattr(constants_module, 'CACHE_MAX_SIZE', None)
    ALERT_COOLDOWN_MINUTES = getattr(constants_module, 'ALERT_COOLDOWN_MINUTES', None)
    MAX_CONSECUTIVE_ALERTS = getattr(constants_module, 'MAX_CONSECUTIVE_ALERTS', None)
    
    if DEFAULT_MONITORING_INTERVAL is None:
        pytest.skip("监控常量模块不可用", allow_module_level=True)
except ImportError:
    pytest.skip("监控常量模块导入失败", allow_module_level=True)


class TestTimeConstants:
    """测试时间相关常量"""

    def test_default_monitoring_interval(self):
        """测试默认监控间隔"""
        assert DEFAULT_MONITORING_INTERVAL == 60
        assert isinstance(DEFAULT_MONITORING_INTERVAL, int)

    def test_default_retention_hours(self):
        """测试默认数据保留时间"""
        assert DEFAULT_RETENTION_HOURS == 24
        assert isinstance(DEFAULT_RETENTION_HOURS, int)

    def test_alert_check_interval(self):
        """测试告警检查间隔"""
        assert ALERT_CHECK_INTERVAL == 30
        assert isinstance(ALERT_CHECK_INTERVAL, int)


class TestPerformanceThresholds:
    """测试性能阈值常量"""

    def test_cpu_thresholds(self):
        """测试CPU阈值"""
        assert CPU_THRESHOLD_HIGH == 80.0
        assert CPU_THRESHOLD_CRITICAL == 95.0
        assert isinstance(CPU_THRESHOLD_HIGH, float)
        assert isinstance(CPU_THRESHOLD_CRITICAL, float)
        assert CPU_THRESHOLD_HIGH < CPU_THRESHOLD_CRITICAL

    def test_memory_thresholds(self):
        """测试内存阈值"""
        assert MEMORY_THRESHOLD_HIGH == 85.0
        assert MEMORY_THRESHOLD_CRITICAL == 95.0
        assert isinstance(MEMORY_THRESHOLD_HIGH, float)
        assert isinstance(MEMORY_THRESHOLD_CRITICAL, float)
        assert MEMORY_THRESHOLD_HIGH < MEMORY_THRESHOLD_CRITICAL

    def test_response_time_thresholds(self):
        """测试响应时间阈值"""
        assert RESPONSE_TIME_HIGH == 1000
        assert RESPONSE_TIME_CRITICAL == 5000
        assert isinstance(RESPONSE_TIME_HIGH, int)
        assert isinstance(RESPONSE_TIME_CRITICAL, int)
        assert RESPONSE_TIME_HIGH < RESPONSE_TIME_CRITICAL

    def test_error_rate_thresholds(self):
        """测试错误率阈值"""
        assert ERROR_RATE_HIGH == 5.0
        assert ERROR_RATE_CRITICAL == 10.0
        assert isinstance(ERROR_RATE_HIGH, float)
        assert isinstance(ERROR_RATE_CRITICAL, float)
        assert ERROR_RATE_HIGH < ERROR_RATE_CRITICAL


class TestCapacityConstants:
    """测试容量常量"""

    def test_max_metrics_buffer(self):
        """测试最大指标缓冲区"""
        assert MAX_METRICS_BUFFER == 10000
        assert isinstance(MAX_METRICS_BUFFER, int)
        assert MAX_METRICS_BUFFER > 0

    def test_max_alert_buffer(self):
        """测试最大告警缓冲区"""
        assert MAX_ALERT_BUFFER == 1000
        assert isinstance(MAX_ALERT_BUFFER, int)
        assert MAX_ALERT_BUFFER > 0

    def test_max_log_entries(self):
        """测试最大日志条目数"""
        assert MAX_LOG_ENTRIES == 50000
        assert isinstance(MAX_LOG_ENTRIES, int)
        assert MAX_LOG_ENTRIES > 0


class TestRetryAndTimeoutConstants:
    """测试重试和超时常量"""

    def test_max_retry_attempts(self):
        """测试最大重试次数"""
        assert MAX_RETRY_ATTEMPTS == 3
        assert isinstance(MAX_RETRY_ATTEMPTS, int)
        assert MAX_RETRY_ATTEMPTS > 0

    def test_retry_delay_seconds(self):
        """测试重试延迟"""
        assert RETRY_DELAY_SECONDS == 5
        assert isinstance(RETRY_DELAY_SECONDS, int)
        assert RETRY_DELAY_SECONDS > 0

    def test_operation_timeout(self):
        """测试操作超时时间"""
        assert OPERATION_TIMEOUT == 30
        assert isinstance(OPERATION_TIMEOUT, int)
        assert OPERATION_TIMEOUT > 0


class TestHealthCheckConstants:
    """测试健康检查常量"""

    def test_health_check_timeout(self):
        """测试健康检查超时"""
        assert HEALTH_CHECK_TIMEOUT == 10
        assert isinstance(HEALTH_CHECK_TIMEOUT, int)
        assert HEALTH_CHECK_TIMEOUT > 0

    def test_health_score_threshold(self):
        """测试健康评分阈值"""
        assert HEALTH_SCORE_THRESHOLD == 70
        assert isinstance(HEALTH_SCORE_THRESHOLD, int)
        assert 0 <= HEALTH_SCORE_THRESHOLD <= 100


class TestPortConstants:
    """测试端口常量"""

    def test_default_monitoring_port(self):
        """测试默认监控端口"""
        assert DEFAULT_MONITORING_PORT == 8080
        assert isinstance(DEFAULT_MONITORING_PORT, int)
        assert 1 <= DEFAULT_MONITORING_PORT <= 65535

    def test_default_metrics_port(self):
        """测试默认指标端口"""
        assert DEFAULT_METRICS_PORT == 9090
        assert isinstance(DEFAULT_METRICS_PORT, int)
        assert 1 <= DEFAULT_METRICS_PORT <= 65535


class TestBatchConstants:
    """测试批处理常量"""

    def test_default_batch_size(self):
        """测试默认批处理大小"""
        assert DEFAULT_BATCH_SIZE == 100
        assert isinstance(DEFAULT_BATCH_SIZE, int)
        assert DEFAULT_BATCH_SIZE > 0

    def test_max_batch_size(self):
        """测试最大批处理大小"""
        assert MAX_BATCH_SIZE == 1000
        assert isinstance(MAX_BATCH_SIZE, int)
        assert MAX_BATCH_SIZE > DEFAULT_BATCH_SIZE


class TestCacheConstants:
    """测试缓存常量"""

    def test_cache_ttl_default(self):
        """测试默认缓存过期时间"""
        assert CACHE_TTL_DEFAULT == 300
        assert isinstance(CACHE_TTL_DEFAULT, int)
        assert CACHE_TTL_DEFAULT > 0

    def test_cache_max_size(self):
        """测试缓存最大大小"""
        assert CACHE_MAX_SIZE == 1000
        assert isinstance(CACHE_MAX_SIZE, int)
        assert CACHE_MAX_SIZE > 0


class TestAlertConstants:
    """测试告警常量"""

    def test_alert_cooldown_minutes(self):
        """测试告警冷却时间"""
        assert ALERT_COOLDOWN_MINUTES == 5
        assert isinstance(ALERT_COOLDOWN_MINUTES, int)
        assert ALERT_COOLDOWN_MINUTES > 0

    def test_max_consecutive_alerts(self):
        """测试最大连续告警次数"""
        assert MAX_CONSECUTIVE_ALERTS == 10
        assert isinstance(MAX_CONSECUTIVE_ALERTS, int)
        assert MAX_CONSECUTIVE_ALERTS > 0



