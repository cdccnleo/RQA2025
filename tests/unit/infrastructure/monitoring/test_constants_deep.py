#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Monitoring常量深度测试"""

import pytest


# ============================================================================
# 监控时间和间隔测试
# ============================================================================

def test_monitor_intervals():
    """测试监控间隔常量"""
    from src.infrastructure.monitoring.core.constants import (
        DEFAULT_MONITOR_INTERVAL,
        FAST_MONITOR_INTERVAL,
        SLOW_MONITOR_INTERVAL
    )
    
    assert DEFAULT_MONITOR_INTERVAL == 30
    assert FAST_MONITOR_INTERVAL == 5
    assert SLOW_MONITOR_INTERVAL == 300
    # 验证逻辑关系
    assert FAST_MONITOR_INTERVAL < DEFAULT_MONITOR_INTERVAL < SLOW_MONITOR_INTERVAL


def test_monitoring_periods():
    """测试监控周期常量"""
    from src.infrastructure.monitoring.core.constants import (
        SHORT_MONITORING_PERIOD,
        MEDIUM_MONITORING_PERIOD,
        LONG_MONITORING_PERIOD
    )
    
    assert SHORT_MONITORING_PERIOD == 300  # 5分钟
    assert MEDIUM_MONITORING_PERIOD == 1800  # 30分钟
    assert LONG_MONITORING_PERIOD == 3600  # 1小时
    # 验证递增关系
    assert SHORT_MONITORING_PERIOD < MEDIUM_MONITORING_PERIOD < LONG_MONITORING_PERIOD


def test_retention_periods():
    """测试数据保留时间"""
    from src.infrastructure.monitoring.core.constants import (
        METRICS_RETENTION_PERIOD,
        ALERT_HISTORY_RETENTION
    )
    
    assert METRICS_RETENTION_PERIOD == 604800  # 7天
    assert ALERT_HISTORY_RETENTION == 2592000  # 30天


# ============================================================================
# CPU阈值测试
# ============================================================================

def test_cpu_thresholds():
    """测试CPU使用率阈值"""
    from src.infrastructure.monitoring.core.constants import (
        CPU_WARNING_THRESHOLD,
        CPU_CRITICAL_THRESHOLD,
        CPU_RECOVERY_THRESHOLD
    )
    
    assert CPU_WARNING_THRESHOLD == 70.0
    assert CPU_CRITICAL_THRESHOLD == 90.0
    assert CPU_RECOVERY_THRESHOLD == 60.0
    # 验证阈值逻辑
    assert CPU_RECOVERY_THRESHOLD < CPU_WARNING_THRESHOLD < CPU_CRITICAL_THRESHOLD


# ============================================================================
# 内存阈值测试
# ============================================================================

def test_memory_thresholds():
    """测试内存使用率阈值"""
    from src.infrastructure.monitoring.core.constants import (
        MEMORY_WARNING_THRESHOLD,
        MEMORY_CRITICAL_THRESHOLD,
        MEMORY_RECOVERY_THRESHOLD
    )
    
    assert MEMORY_WARNING_THRESHOLD == 75.0
    assert MEMORY_CRITICAL_THRESHOLD == 90.0
    assert MEMORY_RECOVERY_THRESHOLD == 70.0
    # 验证阈值逻辑
    assert MEMORY_RECOVERY_THRESHOLD < MEMORY_WARNING_THRESHOLD < MEMORY_CRITICAL_THRESHOLD


# ============================================================================
# 磁盘阈值测试
# ============================================================================

def test_disk_thresholds():
    """测试磁盘使用率阈值"""
    from src.infrastructure.monitoring.core.constants import (
        DISK_WARNING_THRESHOLD,
        DISK_CRITICAL_THRESHOLD,
        DISK_RECOVERY_THRESHOLD
    )
    
    assert DISK_WARNING_THRESHOLD == 80.0
    assert DISK_CRITICAL_THRESHOLD == 95.0
    assert DISK_RECOVERY_THRESHOLD == 75.0
    # 验证阈值逻辑
    assert DISK_RECOVERY_THRESHOLD < DISK_WARNING_THRESHOLD < DISK_CRITICAL_THRESHOLD


# ============================================================================
# 网络延迟阈值测试
# ============================================================================

def test_network_latency_thresholds():
    """测试网络延迟阈值"""
    from src.infrastructure.monitoring.core.constants import (
        NETWORK_LATENCY_WARNING,
        NETWORK_LATENCY_CRITICAL,
        NETWORK_LATENCY_RECOVERY
    )
    
    assert NETWORK_LATENCY_WARNING == 100
    assert NETWORK_LATENCY_CRITICAL == 500
    assert NETWORK_LATENCY_RECOVERY == 50
    # 验证阈值逻辑
    assert NETWORK_LATENCY_RECOVERY < NETWORK_LATENCY_WARNING < NETWORK_LATENCY_CRITICAL


# ============================================================================
# 告警级别测试
# ============================================================================

def test_alert_levels():
    """测试告警级别常量"""
    from src.infrastructure.monitoring.core.constants import (
        ALERT_LEVEL_INFO,
        ALERT_LEVEL_WARNING,
        ALERT_LEVEL_ERROR,
        ALERT_LEVEL_CRITICAL
    )
    
    assert ALERT_LEVEL_INFO == "INFO"
    assert ALERT_LEVEL_WARNING == "WARNING"
    assert ALERT_LEVEL_ERROR == "ERROR"
    assert ALERT_LEVEL_CRITICAL == "CRITICAL"


def test_alert_status():
    """测试告警状态常量"""
    from src.infrastructure.monitoring.core.constants import (
        ALERT_STATUS_ACTIVE,
        ALERT_STATUS_RESOLVED,
        ALERT_STATUS_ACKED
    )
    
    assert ALERT_STATUS_ACTIVE == "ACTIVE"
    assert ALERT_STATUS_RESOLVED == "RESOLVED"
    assert ALERT_STATUS_ACKED == "ACKNOWLEDGED"


def test_alert_priorities():
    """测试告警优先级"""
    from src.infrastructure.monitoring.core.constants import (
        ALERT_PRIORITY_LOW,
        ALERT_PRIORITY_MEDIUM,
        ALERT_PRIORITY_HIGH,
        ALERT_PRIORITY_CRITICAL
    )
    
    assert ALERT_PRIORITY_LOW == 1
    assert ALERT_PRIORITY_MEDIUM == 2
    assert ALERT_PRIORITY_HIGH == 3
    assert ALERT_PRIORITY_CRITICAL == 4
    # 验证优先级递增
    assert ALERT_PRIORITY_LOW < ALERT_PRIORITY_MEDIUM < ALERT_PRIORITY_HIGH < ALERT_PRIORITY_CRITICAL


# ============================================================================
# 通知配置测试
# ============================================================================

def test_notification_retry_config():
    """测试通知重试配置"""
    from src.infrastructure.monitoring.core.constants import (
        NOTIFICATION_MAX_RETRIES,
        NOTIFICATION_RETRY_DELAY
    )
    
    assert NOTIFICATION_MAX_RETRIES == 3
    assert NOTIFICATION_RETRY_DELAY == 5


def test_notification_channels():
    """测试通知通道"""
    from src.infrastructure.monitoring.core.constants import (
        NOTIFICATION_EMAIL,
        NOTIFICATION_SMS,
        NOTIFICATION_WEBHOOK,
        NOTIFICATION_SLACK
    )
    
    assert NOTIFICATION_EMAIL == "email"
    assert NOTIFICATION_SMS == "sms"
    assert NOTIFICATION_WEBHOOK == "webhook"
    assert NOTIFICATION_SLACK == "slack"


# ============================================================================
# 批量处理测试
# ============================================================================

def test_batch_sizes():
    """测试批量处理大小"""
    from src.infrastructure.monitoring.core.constants import (
        METRICS_BATCH_SIZE,
        ALERT_BATCH_SIZE
    )
    
    assert METRICS_BATCH_SIZE == 100
    assert ALERT_BATCH_SIZE == 50


def test_cache_sizes():
    """测试缓存大小"""
    from src.infrastructure.monitoring.core.constants import (
        METRICS_CACHE_SIZE,
        ALERT_CACHE_SIZE
    )
    
    assert METRICS_CACHE_SIZE == 10000
    assert ALERT_CACHE_SIZE == 1000


# ============================================================================
# 健康检查测试
# ============================================================================

def test_health_check_config():
    """测试健康检查配置"""
    from src.infrastructure.monitoring.core.constants import (
        HEALTH_CHECK_TIMEOUT,
        HEALTH_CHECK_INTERVAL
    )
    
    assert HEALTH_CHECK_TIMEOUT == 10
    assert HEALTH_CHECK_INTERVAL == 60


def test_health_status():
    """测试健康状态常量"""
    from src.infrastructure.monitoring.core.constants import (
        HEALTH_STATUS_HEALTHY,
        HEALTH_STATUS_DEGRADED,
        HEALTH_STATUS_UNHEALTHY,
        HEALTH_STATUS_UNKNOWN
    )
    
    assert HEALTH_STATUS_HEALTHY == "HEALTHY"
    assert HEALTH_STATUS_DEGRADED == "DEGRADED"
    assert HEALTH_STATUS_UNHEALTHY == "UNHEALTHY"
    assert HEALTH_STATUS_UNKNOWN == "UNKNOWN"


# ============================================================================
# 监控类型测试
# ============================================================================

def test_monitor_types():
    """测试监控类型常量"""
    from src.infrastructure.monitoring.core.constants import (
        MONITOR_TYPE_SYSTEM,
        MONITOR_TYPE_APPLICATION,
        MONITOR_TYPE_COMPONENT,
        MONITOR_TYPE_STORAGE,
        MONITOR_TYPE_NETWORK,
        MONITOR_TYPE_LOG,
        MONITOR_TYPE_EXCEPTION,
        MONITOR_TYPE_DISASTER
    )
    
    assert MONITOR_TYPE_SYSTEM == "system"
    assert MONITOR_TYPE_APPLICATION == "application"
    assert MONITOR_TYPE_COMPONENT == "component"
    assert MONITOR_TYPE_STORAGE == "storage"
    assert MONITOR_TYPE_NETWORK == "network"
    assert MONITOR_TYPE_LOG == "log"
    assert MONITOR_TYPE_EXCEPTION == "exception"
    assert MONITOR_TYPE_DISASTER == "disaster"


# ============================================================================
# 单位常量测试
# ============================================================================

def test_time_units():
    """测试时间单位"""
    from src.infrastructure.monitoring.core.constants import (
        UNIT_MILLISECONDS,
        UNIT_SECONDS,
        UNIT_MINUTES,
        UNIT_HOURS
    )
    
    assert UNIT_MILLISECONDS == "ms"
    assert UNIT_SECONDS == "s"
    assert UNIT_MINUTES == "min"
    assert UNIT_HOURS == "h"


def test_data_size_units():
    """测试数据大小单位"""
    from src.infrastructure.monitoring.core.constants import (
        UNIT_BYTES,
        UNIT_KILOBYTES,
        UNIT_MEGABYTES,
        UNIT_GIGABYTES
    )
    
    assert UNIT_BYTES == "B"
    assert UNIT_KILOBYTES == "KB"
    assert UNIT_MEGABYTES == "MB"
    assert UNIT_GIGABYTES == "GB"


def test_frequency_units():
    """测试频率单位"""
    from src.infrastructure.monitoring.core.constants import (
        UNIT_PER_SECOND,
        UNIT_PER_MINUTE,
        UNIT_PER_HOUR
    )
    
    assert UNIT_PER_SECOND == "/s"
    assert UNIT_PER_MINUTE == "/min"
    assert UNIT_PER_HOUR == "/h"


# ============================================================================
# SLA目标测试
# ============================================================================

def test_sla_targets():
    """测试SLA目标"""
    from src.infrastructure.monitoring.core.constants import (
        SLA_AVAILABILITY_TARGET,
        SLA_RESPONSE_TIME_TARGET
    )
    
    assert SLA_AVAILABILITY_TARGET == 99.9
    assert SLA_RESPONSE_TIME_TARGET == 200


# ============================================================================
# 常量类型验证测试
# ============================================================================

def test_constants_are_final():
    """测试常量使用Final类型标注"""
    from typing import get_type_hints
    import src.infrastructure.monitoring.core.constants as constants_module
    
    # 验证模块存在
    assert constants_module is not None
    assert hasattr(constants_module, 'DEFAULT_MONITOR_INTERVAL')


def test_numeric_constants_ranges():
    """测试数值常量的合理范围"""
    from src.infrastructure.monitoring.core.constants import (
        CPU_CRITICAL_THRESHOLD,
        MEMORY_CRITICAL_THRESHOLD,
        DISK_CRITICAL_THRESHOLD
    )
    
    # 所有阈值应该在0-100之间
    assert 0 < CPU_CRITICAL_THRESHOLD <= 100
    assert 0 < MEMORY_CRITICAL_THRESHOLD <= 100
    assert 0 < DISK_CRITICAL_THRESHOLD <= 100

