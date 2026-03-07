#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Resource Alert枚举深度测试"""

import pytest
from enum import Enum


# ============================================================================
# AlertLevel测试
# ============================================================================

def test_alert_level_enum():
    """测试AlertLevel枚举"""
    from src.infrastructure.resource.models.alert_enums import AlertLevel
    
    assert AlertLevel.INFO.value == "info"
    assert AlertLevel.WARNING.value == "warning"
    assert AlertLevel.ERROR.value == "error"
    assert AlertLevel.CRITICAL.value == "critical"


def test_alert_level_is_enum():
    """测试AlertLevel是Enum子类"""
    from src.infrastructure.resource.models.alert_enums import AlertLevel
    
    assert issubclass(AlertLevel, Enum)
    assert isinstance(AlertLevel.INFO, Enum)


def test_alert_level_members():
    """测试AlertLevel成员"""
    from src.infrastructure.resource.models.alert_enums import AlertLevel
    
    members = list(AlertLevel)
    assert len(members) == 4
    assert AlertLevel.INFO in members
    assert AlertLevel.WARNING in members
    assert AlertLevel.ERROR in members
    assert AlertLevel.CRITICAL in members


def test_alert_level_by_name():
    """测试通过名称访问AlertLevel"""
    from src.infrastructure.resource.models.alert_enums import AlertLevel
    
    assert AlertLevel['INFO'] == AlertLevel.INFO
    assert AlertLevel['WARNING'] == AlertLevel.WARNING
    assert AlertLevel['ERROR'] == AlertLevel.ERROR
    assert AlertLevel['CRITICAL'] == AlertLevel.CRITICAL


def test_alert_level_by_value():
    """测试通过值访问AlertLevel"""
    from src.infrastructure.resource.models.alert_enums import AlertLevel
    
    assert AlertLevel("info") == AlertLevel.INFO
    assert AlertLevel("warning") == AlertLevel.WARNING
    assert AlertLevel("error") == AlertLevel.ERROR
    assert AlertLevel("critical") == AlertLevel.CRITICAL


def test_alert_level_comparison():
    """测试AlertLevel比较"""
    from src.infrastructure.resource.models.alert_enums import AlertLevel
    
    # 枚举成员相等性
    assert AlertLevel.INFO == AlertLevel.INFO
    assert AlertLevel.INFO != AlertLevel.WARNING


# ============================================================================
# AlertType测试
# ============================================================================

def test_alert_type_enum():
    """测试AlertType枚举"""
    from src.infrastructure.resource.models.alert_enums import AlertType
    
    assert AlertType.TEST_TIMEOUT.value == "test_timeout"
    assert AlertType.TEST_FAILURE.value == "test_failure"
    assert AlertType.PERFORMANCE_DEGRADATION.value == "performance_degradation"
    assert AlertType.SYSTEM_ERROR.value == "system_error"
    assert AlertType.RESOURCE_EXHAUSTION.value == "resource_exhaustion"
    assert AlertType.NETWORK_ISSUE.value == "network_issue"


def test_alert_type_is_enum():
    """测试AlertType是Enum子类"""
    from src.infrastructure.resource.models.alert_enums import AlertType
    
    assert issubclass(AlertType, Enum)
    assert isinstance(AlertType.TEST_TIMEOUT, Enum)


def test_alert_type_members():
    """测试AlertType成员"""
    from src.infrastructure.resource.models.alert_enums import AlertType
    
    members = list(AlertType)
    assert len(members) == 6
    assert AlertType.TEST_TIMEOUT in members
    assert AlertType.TEST_FAILURE in members
    assert AlertType.PERFORMANCE_DEGRADATION in members
    assert AlertType.SYSTEM_ERROR in members
    assert AlertType.RESOURCE_EXHAUSTION in members
    assert AlertType.NETWORK_ISSUE in members


def test_alert_type_by_name():
    """测试通过名称访问AlertType"""
    from src.infrastructure.resource.models.alert_enums import AlertType
    
    assert AlertType['TEST_TIMEOUT'] == AlertType.TEST_TIMEOUT
    assert AlertType['TEST_FAILURE'] == AlertType.TEST_FAILURE
    assert AlertType['PERFORMANCE_DEGRADATION'] == AlertType.PERFORMANCE_DEGRADATION


def test_alert_type_by_value():
    """测试通过值访问AlertType"""
    from src.infrastructure.resource.models.alert_enums import AlertType
    
    assert AlertType("test_timeout") == AlertType.TEST_TIMEOUT
    assert AlertType("test_failure") == AlertType.TEST_FAILURE
    assert AlertType("system_error") == AlertType.SYSTEM_ERROR


# ============================================================================
# MonitoringEvent测试
# ============================================================================

def test_monitoring_event_enum():
    """测试MonitoringEvent枚举"""
    from src.infrastructure.resource.models.alert_enums import MonitoringEvent
    
    assert MonitoringEvent.TEST_STARTED.value == "test_started"
    assert MonitoringEvent.TEST_COMPLETED.value == "test_completed"
    assert MonitoringEvent.TEST_FAILED.value == "test_failed"
    assert MonitoringEvent.TEST_TIMEOUT.value == "test_timeout"
    assert MonitoringEvent.PERFORMANCE_THRESHOLD_EXCEEDED.value == "performance_threshold_exceeded"
    assert MonitoringEvent.SYSTEM_RESOURCE_LOW.value == "system_resource_low"
    assert MonitoringEvent.NETWORK_LATENCY_HIGH.value == "network_latency_high"


def test_monitoring_event_is_enum():
    """测试MonitoringEvent是Enum子类"""
    from src.infrastructure.resource.models.alert_enums import MonitoringEvent
    
    assert issubclass(MonitoringEvent, Enum)
    assert isinstance(MonitoringEvent.TEST_STARTED, Enum)


def test_monitoring_event_members():
    """测试MonitoringEvent成员"""
    from src.infrastructure.resource.models.alert_enums import MonitoringEvent
    
    members = list(MonitoringEvent)
    assert len(members) == 7
    assert MonitoringEvent.TEST_STARTED in members
    assert MonitoringEvent.TEST_COMPLETED in members
    assert MonitoringEvent.TEST_FAILED in members


def test_monitoring_event_by_name():
    """测试通过名称访问MonitoringEvent"""
    from src.infrastructure.resource.models.alert_enums import MonitoringEvent
    
    assert MonitoringEvent['TEST_STARTED'] == MonitoringEvent.TEST_STARTED
    assert MonitoringEvent['TEST_COMPLETED'] == MonitoringEvent.TEST_COMPLETED
    assert MonitoringEvent['TEST_FAILED'] == MonitoringEvent.TEST_FAILED


def test_monitoring_event_by_value():
    """测试通过值访问MonitoringEvent"""
    from src.infrastructure.resource.models.alert_enums import MonitoringEvent
    
    assert MonitoringEvent("test_started") == MonitoringEvent.TEST_STARTED
    assert MonitoringEvent("test_completed") == MonitoringEvent.TEST_COMPLETED
    assert MonitoringEvent("test_failed") == MonitoringEvent.TEST_FAILED


# ============================================================================
# 跨枚举测试
# ============================================================================

def test_all_enums_are_independent():
    """测试所有枚举类互相独立"""
    from src.infrastructure.resource.models.alert_enums import (
        AlertLevel,
        AlertType,
        MonitoringEvent
    )
    
    # 枚举类不同
    assert AlertLevel is not AlertType
    assert AlertType is not MonitoringEvent
    assert AlertLevel is not MonitoringEvent


def test_enum_iteration():
    """测试枚举迭代"""
    from src.infrastructure.resource.models.alert_enums import AlertLevel
    
    levels = []
    for level in AlertLevel:
        levels.append(level.value)
    
    assert "info" in levels
    assert "warning" in levels
    assert "error" in levels
    assert "critical" in levels


def test_enum_in_dict():
    """测试枚举作为字典键"""
    from src.infrastructure.resource.models.alert_enums import AlertLevel
    
    mapping = {
        AlertLevel.INFO: 1,
        AlertLevel.WARNING: 2,
        AlertLevel.ERROR: 3,
        AlertLevel.CRITICAL: 4
    }
    
    assert mapping[AlertLevel.INFO] == 1
    assert mapping[AlertLevel.CRITICAL] == 4


def test_enum_in_set():
    """测试枚举在集合中"""
    from src.infrastructure.resource.models.alert_enums import AlertType
    
    alert_set = {AlertType.TEST_TIMEOUT, AlertType.TEST_FAILURE}
    
    assert AlertType.TEST_TIMEOUT in alert_set
    assert AlertType.TEST_FAILURE in alert_set
    assert AlertType.SYSTEM_ERROR not in alert_set

