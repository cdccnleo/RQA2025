#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
审计事件管理器综合测试
测试AuditEvent、AuditEventBuilder、AuditEventFilter、AuditEventManager等核心组件
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any
from unittest.mock import patch, MagicMock

from src.infrastructure.security.audit.audit_events import (
    AuditEvent,
    AuditEventType,
    AuditSeverity,
    AuditEventBuilder,
    AuditEventFilter,
    AuditEventManager
)


class TestAuditEventType:
    """测试审计事件类型枚举"""

    def test_audit_event_type_values(self):
        """测试事件类型枚举值"""
        assert AuditEventType.SECURITY.value == "security"
        assert AuditEventType.ACCESS.value == "access"
        assert AuditEventType.DATA_OPERATION.value == "data_operation"
        assert AuditEventType.CONFIG_CHANGE.value == "config_change"
        assert AuditEventType.USER_MANAGEMENT.value == "user_management"
        assert AuditEventType.SYSTEM_EVENT.value == "system_event"
        assert AuditEventType.COMPLIANCE.value == "compliance"

    def test_audit_event_type_membership(self):
        """测试事件类型枚举成员"""
        assert AuditEventType.SECURITY in AuditEventType
        assert AuditEventType.ACCESS in AuditEventType

        # 验证所有预期的成员都存在
        expected_values = [
            "security", "access", "data_operation", "config_change",
            "user_management", "system_event", "compliance"
        ]
        actual_values = [member.value for member in AuditEventType]
        assert set(actual_values) == set(expected_values)


class TestAuditSeverity:
    """测试审计事件严重程度枚举"""

    def test_audit_severity_values(self):
        """测试严重程度枚举值"""
        assert AuditSeverity.LOW.value == "low"
        assert AuditSeverity.MEDIUM.value == "medium"
        assert AuditSeverity.HIGH.value == "high"
        assert AuditSeverity.CRITICAL.value == "critical"

    def test_audit_severity_order(self):
        """测试严重程度排序"""
        # 按枚举定义顺序排序
        severities = [AuditSeverity.LOW, AuditSeverity.MEDIUM, AuditSeverity.HIGH, AuditSeverity.CRITICAL]
        for i in range(len(severities) - 1):
            assert severities[i].value != severities[i + 1].value  # 值不同
            assert severities[i] != severities[i + 1]  # 枚举实例不同


class TestAuditEvent:
    """测试审计事件类"""

    def test_audit_event_creation_minimal(self):
        """测试最小化审计事件创建"""
        timestamp = datetime.now()
        event = AuditEvent(
            event_id="test_event_123",
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.HIGH,
            timestamp=timestamp,
            action="login"
        )

        assert event.event_id == "test_event_123"
        assert event.event_type == AuditEventType.SECURITY
        assert event.severity == AuditSeverity.HIGH
        assert event.timestamp == timestamp
        assert event.action == "login"
        assert event.user_id is None
        assert event.result == ""  # 默认值
        assert event.details == {}  # 默认值
        assert event.risk_score == 0.0  # 默认值
        assert event.tags == set()  # 默认值

    def test_audit_event_creation_complete(self):
        """测试完整审计事件创建"""
        timestamp = datetime.now()
        details = {"ip": "192.168.1.1", "user_agent": "Test Browser"}
        tags = {"urgent", "security"}

        event = AuditEvent(
            event_id="complete_event_123",
            event_type=AuditEventType.ACCESS,
            severity=AuditSeverity.CRITICAL,
            timestamp=timestamp,
            user_id="user123",
            session_id="session456",
            resource="admin_panel",
            action="login_attempt",
            result="failure",
            details=details,
            ip_address="192.168.1.1",
            user_agent="Test Browser",
            location="New York",
            risk_score=8.5,
            tags=tags
        )

        assert event.event_id == "complete_event_123"
        assert event.event_type == AuditEventType.ACCESS
        assert event.severity == AuditSeverity.CRITICAL
        assert event.user_id == "user123"
        assert event.session_id == "session456"
        assert event.resource == "admin_panel"
        assert event.action == "login_attempt"
        assert event.result == "failure"
        assert event.details == details
        assert event.ip_address == "192.168.1.1"
        assert event.user_agent == "Test Browser"
        assert event.location == "New York"
        assert event.risk_score == 8.5
        assert event.tags == tags

    def test_audit_event_to_dict(self):
        """测试审计事件序列化"""
        timestamp = datetime(2025, 1, 1, 12, 0, 0)
        details = {"key": "value"}
        tags = {"tag1", "tag2"}

        event = AuditEvent(
            event_id="test_event_123",
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.HIGH,
            timestamp=timestamp,
            user_id="user123",
            action="login",
            details=details,
            tags=tags,
            risk_score=7.5
        )

        event_dict = event.to_dict()

        assert isinstance(event_dict, dict)
        assert event_dict["event_id"] == "test_event_123"
        assert event_dict["event_type"] == "security"
        assert event_dict["severity"] == "high"
        assert event_dict["timestamp"] == timestamp.isoformat()
        assert event_dict["user_id"] == "user123"
        assert event_dict["action"] == "login"
        assert event_dict["details"] == details
        assert event_dict["tags"] == list(tags)
        assert event_dict["risk_score"] == 7.5

    def test_audit_event_from_dict(self):
        """测试审计事件反序列化"""
        timestamp_str = "2025-01-01T12:00:00"
        event_dict = {
            "event_id": "test_event_123",
            "event_type": "security",
            "severity": "high",
            "timestamp": timestamp_str,
            "user_id": "user123",
            "session_id": "session456",
            "resource": "admin",
            "action": "login",
            "result": "success",
            "details": {"ip": "192.168.1.1"},
            "ip_address": "192.168.1.1",
            "user_agent": "Test Browser",
            "location": "NYC",
            "risk_score": 5.0,
            "tags": ["urgent", "security"]
        }

        event = AuditEvent.from_dict(event_dict)

        assert isinstance(event, AuditEvent)
        assert event.event_id == "test_event_123"
        assert event.event_type == AuditEventType.SECURITY
        assert event.severity == AuditSeverity.HIGH
        assert event.user_id == "user123"
        assert event.session_id == "session456"
        assert event.resource == "admin"
        assert event.action == "login"
        assert event.result == "success"
        assert event.details == {"ip": "192.168.1.1"}
        assert event.ip_address == "192.168.1.1"
        assert event.user_agent == "Test Browser"
        assert event.location == "NYC"
        assert event.risk_score == 5.0
        assert event.tags == {"urgent", "security"}

    def test_audit_event_equality(self):
        """测试审计事件相等性"""
        timestamp = datetime.now()

        event1 = AuditEvent(
            event_id="event_123",
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.HIGH,
            timestamp=timestamp,
            action="login"
        )

        event2 = AuditEvent(
            event_id="event_123",
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.HIGH,
            timestamp=timestamp,
            action="login"
        )

        event3 = AuditEvent(
            event_id="event_456",
            event_type=AuditEventType.ACCESS,
            severity=AuditSeverity.MEDIUM,
            timestamp=timestamp,
            action="logout"
        )

        assert event1 == event2
        assert event1 != event3
        assert event2 != event3


class TestAuditEventBuilder:
    """测试审计事件构建器"""

    def test_builder_initialization(self):
        """测试构建器初始化"""
        builder = AuditEventBuilder()
        assert builder._event is None

    def test_builder_new_event(self):
        """测试创建新事件"""
        builder = AuditEventBuilder()
        result = builder.new_event(AuditEventType.SECURITY, AuditSeverity.HIGH)

        assert result is builder  # 返回构建器自身
        assert builder._event is not None
        assert builder._event.event_type == AuditEventType.SECURITY
        assert builder._event.severity == AuditSeverity.HIGH

    def test_builder_with_user(self):
        """测试设置用户ID"""
        builder = AuditEventBuilder()
        # 先创建事件
        builder.new_event(AuditEventType.SECURITY, AuditSeverity.HIGH)
        result = builder.with_user("user123")

        assert result is builder
        assert builder._event.user_id == "user123"

    def test_builder_chaining(self):
        """测试构建器链式调用"""
        builder = AuditEventBuilder()
        event = (builder
                .new_event(AuditEventType.SECURITY, AuditSeverity.MEDIUM)
                .with_user("user123")
                .with_action("password_change")
                .with_result("success")
                .build())

        assert event.event_type == AuditEventType.SECURITY
        assert event.severity == AuditSeverity.MEDIUM
        assert event.user_id == "user123"
        assert event.action == "password_change"
        assert event.result == "success"

    def test_builder_build_minimal(self):
        """测试构建最小事件"""
        builder = AuditEventBuilder()
        event = (builder
                .new_event(AuditEventType.SECURITY, AuditSeverity.HIGH)
                .with_action("login")
                .build())

        assert isinstance(event, AuditEvent)
        assert event.event_type == AuditEventType.SECURITY
        assert event.severity == AuditSeverity.HIGH
        assert event.action == "login"
        assert isinstance(event.event_id, str)
        assert len(event.event_id) > 0
        assert isinstance(event.timestamp, datetime)

    def test_builder_build_complete(self):
        """测试构建完整事件"""
        builder = AuditEventBuilder()
        event = (builder
                .new_event(AuditEventType.ACCESS, AuditSeverity.CRITICAL)
                .with_user("user123")
                .with_session("session456")
                .with_resource("admin_panel")
                .with_action("login_attempt")
                .with_result("failure")
                .with_details({"attempts": 3})
                .with_ip_address("192.168.1.1")
                .with_user_agent("Test Browser")
                .with_location("NYC")
                .with_risk_score(9.0)
                .with_tags({"urgent", "failed_login"})
                .build())

        assert isinstance(event, AuditEvent)
        assert event.event_type == AuditEventType.ACCESS
        assert event.severity == AuditSeverity.CRITICAL
        assert event.user_id == "user123"
        assert event.session_id == "session456"
        assert event.resource == "admin_panel"
        assert event.action == "login_attempt"
        assert event.result == "failure"
        assert event.details == {"attempts": 3}
        assert event.ip_address == "192.168.1.1"
        assert event.user_agent == "Test Browser"
        assert event.location == "NYC"
        assert event.risk_score == 9.0
        assert event.tags == {"urgent", "failed_login"}

    def test_builder_chaining(self):
        """测试构建器链式调用"""
        builder = AuditEventBuilder()
        event = (builder
                .new_event(AuditEventType.SECURITY, AuditSeverity.MEDIUM)
                .with_user("user123")
                .with_action("password_change")
                .with_result("success")
                .build())

        assert event.event_type == AuditEventType.SECURITY
        assert event.severity == AuditSeverity.MEDIUM
        assert event.user_id == "user123"
        assert event.action == "password_change"
        assert event.result == "success"


class TestAuditEventFilter:
    """测试审计事件过滤器"""

    def test_filter_initialization(self):
        """测试过滤器初始化"""
        filter_obj = AuditEventFilter()
        assert filter_obj._filters == {}

    def test_filter_by_event_type(self):
        """测试按事件类型过滤"""
        filter_obj = AuditEventFilter()
        result = filter_obj.by_event_type([AuditEventType.SECURITY, AuditEventType.ACCESS])

        assert result is filter_obj
        assert filter_obj._filters['event_types'] == [AuditEventType.SECURITY, AuditEventType.ACCESS]

    def test_filter_by_severity(self):
        """测试按严重程度过滤"""
        filter_obj = AuditEventFilter()
        result = filter_obj.by_severity([AuditSeverity.HIGH, AuditSeverity.CRITICAL])

        assert result is filter_obj
        assert filter_obj._filters['severities'] == [AuditSeverity.HIGH, AuditSeverity.CRITICAL]

    def test_filter_by_user(self):
        """测试按用户过滤"""
        filter_obj = AuditEventFilter()
        result = filter_obj.by_user(["user1", "user2"])

        assert result is filter_obj
        assert filter_obj._filters['user_ids'] == ["user1", "user2"]

    def test_filter_by_time_range(self):
        """测试按时间范围过滤"""
        start_time = datetime(2025, 1, 1, 0, 0, 0)
        end_time = datetime(2025, 1, 2, 0, 0, 0)

        filter_obj = AuditEventFilter()
        result = filter_obj.by_time_range(start_time, end_time)

        assert result is filter_obj
        assert filter_obj._filters['start_time'] == start_time
        assert filter_obj._filters['end_time'] == end_time

    def test_filter_matches_no_criteria(self):
        """测试无过滤条件的情况"""
        filter_obj = AuditEventFilter()
        event = AuditEvent(
            event_id="test",
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.HIGH,
            timestamp=datetime.now(),
            action="login"
        )

        assert filter_obj.matches(event)

    def test_filter_matches_event_type(self):
        """测试事件类型匹配"""
        filter_obj = AuditEventFilter().by_event_type([AuditEventType.SECURITY])

        matching_event = AuditEvent(
            event_id="test1",
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.HIGH,
            timestamp=datetime.now(),
            action="login"
        )

        non_matching_event = AuditEvent(
            event_id="test2",
            event_type=AuditEventType.ACCESS,
            severity=AuditSeverity.HIGH,
            timestamp=datetime.now(),
            action="login"
        )

        assert filter_obj.matches(matching_event)
        assert not filter_obj.matches(non_matching_event)

    def test_filter_matches_severity(self):
        """测试严重程度匹配"""
        filter_obj = AuditEventFilter().by_severity([AuditSeverity.CRITICAL])

        matching_event = AuditEvent(
            event_id="test1",
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.CRITICAL,
            timestamp=datetime.now(),
            action="login"
        )

        non_matching_event = AuditEvent(
            event_id="test2",
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.HIGH,
            timestamp=datetime.now(),
            action="login"
        )

        assert filter_obj.matches(matching_event)
        assert not filter_obj.matches(non_matching_event)

    def test_filter_matches_user(self):
        """测试用户匹配"""
        filter_obj = AuditEventFilter().by_user(["user123"])

        matching_event = AuditEvent(
            event_id="test1",
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.HIGH,
            timestamp=datetime.now(),
            user_id="user123",
            action="login"
        )

        non_matching_event = AuditEvent(
            event_id="test2",
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.HIGH,
            timestamp=datetime.now(),
            user_id="user456",
            action="login"
        )

        assert filter_obj.matches(matching_event)
        assert not filter_obj.matches(non_matching_event)

    def test_filter_matches_time_range(self):
        """测试时间范围匹配"""
        start_time = datetime(2025, 1, 1, 0, 0, 0)
        end_time = datetime(2025, 1, 2, 0, 0, 0)
        filter_obj = AuditEventFilter().by_time_range(start_time, end_time)

        matching_event = AuditEvent(
            event_id="test1",
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.HIGH,
            timestamp=datetime(2025, 1, 1, 12, 0, 0),
            action="login"
        )

        non_matching_event = AuditEvent(
            event_id="test2",
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.HIGH,
            timestamp=datetime(2025, 2, 1, 12, 0, 0),
            action="login"
        )

        assert filter_obj.matches(matching_event)
        assert not filter_obj.matches(non_matching_event)

    def test_filter_matches_resource(self):
        """测试资源匹配"""
        filter_obj = AuditEventFilter().by_resource(["admin_panel"])

        matching_event = AuditEvent(
            event_id="test1",
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.HIGH,
            timestamp=datetime.now(),
            resource="admin_panel",
            action="login"
        )

        non_matching_event = AuditEvent(
            event_id="test2",
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.HIGH,
            timestamp=datetime.now(),
            resource="user_panel",
            action="login"
        )

        assert filter_obj.matches(matching_event)
        assert not filter_obj.matches(non_matching_event)

    def test_filter_matches_risk_score(self):
        """测试风险评分匹配"""
        filter_obj = AuditEventFilter().by_risk_score(5.0, 8.0)

        matching_event = AuditEvent(
            event_id="test1",
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.HIGH,
            timestamp=datetime.now(),
            risk_score=7.0,
            action="login"
        )

        non_matching_event = AuditEvent(
            event_id="test2",
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.HIGH,
            timestamp=datetime.now(),
            risk_score=9.0,
            action="login"
        )

        assert filter_obj.matches(matching_event)
        assert not filter_obj.matches(non_matching_event)

    def test_filter_complex_criteria(self):
        """测试复杂过滤条件"""
        filter_obj = (AuditEventFilter()
                     .by_event_type([AuditEventType.SECURITY])
                     .by_severity([AuditSeverity.HIGH, AuditSeverity.CRITICAL])
                     .by_user(["user123"]))

        matching_event = AuditEvent(
            event_id="test1",
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.HIGH,
            timestamp=datetime.now(),
            user_id="user123",
            action="login"
        )

        non_matching_event1 = AuditEvent(  # 错误的事件类型
            event_id="test2",
            event_type=AuditEventType.ACCESS,
            severity=AuditSeverity.HIGH,
            timestamp=datetime.now(),
            user_id="user123",
            action="login"
        )

        non_matching_event2 = AuditEvent(  # 错误的严重程度
            event_id="test3",
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.LOW,
            timestamp=datetime.now(),
            user_id="user123",
            action="login"
        )

        non_matching_event3 = AuditEvent(  # 错误的用户
            event_id="test4",
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.HIGH,
            timestamp=datetime.now(),
            user_id="user456",
            action="login"
        )

        assert filter_obj.matches(matching_event)
        assert not filter_obj.matches(non_matching_event1)
        assert not filter_obj.matches(non_matching_event2)
        assert not filter_obj.matches(non_matching_event3)


class TestAuditEventManager:
    """测试审计事件管理器"""

    def test_manager_initialization(self):
        """测试管理器初始化"""
        manager = AuditEventManager(max_events=500)
        assert manager._max_events == 500
        assert manager._events == []
        assert len(manager._events) == 0

    def test_manager_initialization_default(self):
        """测试管理器默认初始化"""
        manager = AuditEventManager()
        assert manager._max_events == 10000
        assert manager._events == []
        assert len(manager._events) == 0

    def test_create_event_minimal(self):
        """测试创建最小事件"""
        manager = AuditEventManager()
        event = manager.create_event(
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.HIGH,
            action="login"
        )

        assert isinstance(event, AuditEvent)
        assert event.event_type == AuditEventType.SECURITY
        assert event.severity == AuditSeverity.HIGH
        assert event.action == "login"
        assert isinstance(event.event_id, str)
        assert isinstance(event.timestamp, datetime)

    def test_create_event_complete(self):
        """测试创建完整事件"""
        manager = AuditEventManager()
        event = manager.create_event(
            event_type=AuditEventType.ACCESS,
            severity=AuditSeverity.CRITICAL,
            user_id="user123",
            session_id="session456",
            resource="admin_panel",
            action="login_attempt",
            result="failure",
            details={"attempts": 3},
            ip_address="192.168.1.1",
            user_agent="Test Browser",
            location="NYC",
            risk_score=9.0,
            tags={"urgent", "failed_login"}
        )

        assert isinstance(event, AuditEvent)
        assert event.event_type == AuditEventType.ACCESS
        assert event.severity == AuditSeverity.CRITICAL
        assert event.user_id == "user123"
        assert event.session_id == "session456"
        assert event.resource == "admin_panel"
        assert event.action == "login_attempt"
        assert event.result == "failure"
        assert event.details == {"attempts": 3}
        assert event.ip_address == "192.168.1.1"
        assert event.user_agent == "Test Browser"
        assert event.location == "NYC"
        assert event.risk_score == 9.0
        assert event.tags == {"urgent", "failed_login"}

    def test_add_event(self):
        """测试添加事件"""
        manager = AuditEventManager()
        event = AuditEvent(
            event_id="test_event",
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.HIGH,
            timestamp=datetime.now(),
            action="login"
        )

        manager.add_event(event)

        assert manager.get_event_count() == 1
        events = manager.get_events()
        assert len(events) == 1
        assert events[0] == event

    def test_add_event_overflow(self):
        """测试事件数量溢出"""
        manager = AuditEventManager(max_events=2)

        # 添加3个事件
        for i in range(3):
            event = AuditEvent(
                event_id=f"event_{i}",
                event_type=AuditEventType.SECURITY,
                severity=AuditSeverity.HIGH,
                timestamp=datetime.now(),
                action=f"action_{i}"
            )
            manager.add_event(event)

        # 应该只保留最新的2个事件
        assert manager.get_event_count() == 2
        events = manager.get_events()
        assert len(events) == 2
        assert events[0].event_id == "event_1"
        assert events[1].event_id == "event_2"

    def test_get_events_no_filter(self):
        """测试获取所有事件（无过滤）"""
        manager = AuditEventManager()

        # 添加一些事件
        events = []
        for i in range(3):
            event = AuditEvent(
                event_id=f"event_{i}",
                event_type=AuditEventType.SECURITY,
                severity=AuditSeverity.HIGH,
                timestamp=datetime.now(),
                action=f"action_{i}"
            )
            manager.add_event(event)
            events.append(event)

        result = manager.get_events()
        assert len(result) == 3
        assert result == events

    def test_get_events_with_filter(self):
        """测试获取过滤后的事件"""
        manager = AuditEventManager()

        # 添加不同类型的事件
        security_event = AuditEvent(
            event_id="security_1",
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.HIGH,
            timestamp=datetime.now(),
            action="login"
        )

        access_event = AuditEvent(
            event_id="access_1",
            event_type=AuditEventType.ACCESS,
            severity=AuditSeverity.MEDIUM,
            timestamp=datetime.now(),
            action="view"
        )

        manager.add_event(security_event)
        manager.add_event(access_event)

        # 过滤安全事件
        filter_obj = AuditEventFilter().by_event_type([AuditEventType.SECURITY])
        result = manager.get_events(filter_obj)

        assert len(result) == 1
        assert result[0] == security_event

    @pytest.mark.asyncio
    async def test_get_events_async(self):
        """测试异步获取事件"""
        manager = AuditEventManager()

        # 添加事件
        event = AuditEvent(
            event_id="async_test",
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.HIGH,
            timestamp=datetime.now(),
            action="login"
        )
        manager.add_event(event)

        result = await manager.get_events_async()
        assert len(result) == 1
        assert result[0] == event

    @pytest.mark.asyncio
    async def test_create_event_async(self):
        """测试异步创建事件"""
        manager = AuditEventManager()
        event = await manager.create_event_async(
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.HIGH,
            action="async_login"
        )

        assert isinstance(event, AuditEvent)
        assert event.event_type == AuditEventType.SECURITY
        assert event.severity == AuditSeverity.HIGH
        assert event.action == "async_login"

    def test_clear_events(self):
        """测试清除事件"""
        manager = AuditEventManager()

        # 添加事件
        for i in range(3):
            event = AuditEvent(
                event_id=f"event_{i}",
                event_type=AuditEventType.SECURITY,
                severity=AuditSeverity.HIGH,
                timestamp=datetime.now(),
                action=f"action_{i}"
            )
            manager.add_event(event)

        assert manager.get_event_count() == 3

        # 清除事件
        manager.clear_events()

        assert manager.get_event_count() == 0
        events = manager.get_events()
        assert len(events) == 0

    def test_get_event_count(self):
        """测试获取事件数量"""
        manager = AuditEventManager()

        assert manager.get_event_count() == 0

        # 添加事件
        for i in range(5):
            event = AuditEvent(
                event_id=f"event_{i}",
                event_type=AuditEventType.SECURITY,
                severity=AuditSeverity.HIGH,
                timestamp=datetime.now(),
                action=f"action_{i}"
            )
            manager.add_event(event)

        assert manager.get_event_count() == 5

    def test_get_events_by_type(self):
        """测试按类型获取事件"""
        manager = AuditEventManager()

        # 添加不同类型的事件
        security_event1 = AuditEvent(
            event_id="sec1",
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.HIGH,
            timestamp=datetime.now(),
            action="login1"
        )

        security_event2 = AuditEvent(
            event_id="sec2",
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.HIGH,
            timestamp=datetime.now(),
            action="login2"
        )

        access_event = AuditEvent(
            event_id="acc1",
            event_type=AuditEventType.ACCESS,
            severity=AuditSeverity.MEDIUM,
            timestamp=datetime.now(),
            action="view"
        )

        manager.add_event(security_event1)
        manager.add_event(security_event2)
        manager.add_event(access_event)

        security_events = manager.get_events_by_type(AuditEventType.SECURITY)
        access_events = manager.get_events_by_type(AuditEventType.ACCESS)

        assert len(security_events) == 2
        assert len(access_events) == 1
        assert security_events == [security_event1, security_event2]
        assert access_events == [access_event]

    def test_get_events_by_severity(self):
        """测试按严重程度获取事件"""
        manager = AuditEventManager()

        # 添加不同严重程度的事件
        high_event1 = AuditEvent(
            event_id="high1",
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.HIGH,
            timestamp=datetime.now(),
            action="login1"
        )

        high_event2 = AuditEvent(
            event_id="high2",
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.HIGH,
            timestamp=datetime.now(),
            action="login2"
        )

        low_event = AuditEvent(
            event_id="low1",
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.LOW,
            timestamp=datetime.now(),
            action="view"
        )

        manager.add_event(high_event1)
        manager.add_event(high_event2)
        manager.add_event(low_event)

        high_events = manager.get_events_by_severity(AuditSeverity.HIGH)
        low_events = manager.get_events_by_severity(AuditSeverity.LOW)

        assert len(high_events) == 2
        assert len(low_events) == 1
        assert high_events == [high_event1, high_event2]
        assert low_events == [low_event]

    def test_get_recent_events(self):
        """测试获取最近事件"""
        manager = AuditEventManager()

        # 添加不同时间的事件
        old_time = datetime.now() - timedelta(hours=25)
        recent_time = datetime.now() - timedelta(hours=1)

        old_event = AuditEvent(
            event_id="old",
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.HIGH,
            timestamp=old_time,
            action="old_action"
        )

        recent_event1 = AuditEvent(
            event_id="recent1",
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.HIGH,
            timestamp=recent_time,
            action="recent_action1"
        )

        recent_event2 = AuditEvent(
            event_id="recent2",
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.HIGH,
            timestamp=datetime.now(),
            action="recent_action2"
        )

        manager.add_event(old_event)
        manager.add_event(recent_event1)
        manager.add_event(recent_event2)

        recent_events = manager.get_recent_events(hours=24)

        assert len(recent_events) == 2
        assert recent_event1 in recent_events
        assert recent_event2 in recent_events
        assert old_event not in recent_events


class TestAuditEventIntegration:
    """测试审计事件集成场景"""

    def test_builder_manager_integration(self):
        """测试构建器和管理器的集成"""
        manager = AuditEventManager()

        # 使用构建器创建事件
        builder = AuditEventBuilder()
        event = (builder
                .new_event(AuditEventType.SECURITY, AuditSeverity.CRITICAL)
                .with_user("admin")
                .with_resource("system_config")
                .with_action("unauthorized_access")
                .with_result("blocked")
                .with_risk_score(10.0)
                .with_tags({"critical", "security"})
                .build())

        # 添加到管理器
        manager.add_event(event)

        # 验证事件存在
        events = manager.get_events()
        assert len(events) == 1
        assert events[0] == event

        # 验证事件属性
        retrieved_event = events[0]
        assert retrieved_event.event_type == AuditEventType.SECURITY
        assert retrieved_event.severity == AuditSeverity.CRITICAL
        assert retrieved_event.user_id == "admin"
        assert retrieved_event.resource == "system_config"
        assert retrieved_event.action == "unauthorized_access"
        assert retrieved_event.result == "blocked"
        assert retrieved_event.risk_score == 10.0
        assert retrieved_event.tags == {"critical", "security"}

    def test_filter_manager_integration(self):
        """测试过滤器和管理器的集成"""
        manager = AuditEventManager()

        # 创建多种事件
        events_data = [
            (AuditEventType.SECURITY, AuditSeverity.CRITICAL, "admin", "config_change"),
            (AuditEventType.ACCESS, AuditSeverity.HIGH, "user1", "file_access"),
            (AuditEventType.SECURITY, AuditSeverity.MEDIUM, "admin", "login"),
            (AuditEventType.ACCESS, AuditSeverity.LOW, "user2", "page_view"),
        ]

        for i, (event_type, severity, user_id, action) in enumerate(events_data):
            event = AuditEvent(
                event_id=f"event_{i}",
                event_type=event_type,
                severity=severity,
                timestamp=datetime.now(),
                user_id=user_id,
                action=action
            )
            manager.add_event(event)

        # 过滤admin用户的安全事件
        filter_obj = (AuditEventFilter()
                     .by_event_type([AuditEventType.SECURITY])
                     .by_user(["admin"]))

        filtered_events = manager.get_events(filter_obj)

        assert len(filtered_events) == 2
        for event in filtered_events:
            assert event.event_type == AuditEventType.SECURITY
            assert event.user_id == "admin"

    def test_serialization_integration(self):
        """测试序列化集成"""
        # 创建事件
        original_event = AuditEvent(
            event_id="test_serialization",
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.HIGH,
            timestamp=datetime(2025, 1, 1, 12, 0, 0),
            user_id="testuser",
            action="login",
            details={"ip": "192.168.1.1"},
            tags={"test", "serialization"},
            risk_score=5.0
        )

        # 序列化
        event_dict = original_event.to_dict()

        # 反序列化
        restored_event = AuditEvent.from_dict(event_dict)

        # 验证
        assert original_event == restored_event
        assert original_event.event_id == restored_event.event_id
        assert original_event.event_type == restored_event.event_type
        assert original_event.severity == restored_event.severity
        assert original_event.user_id == restored_event.user_id
        assert original_event.action == restored_event.action
        assert original_event.details == restored_event.details
        assert original_event.tags == restored_event.tags
        assert original_event.risk_score == restored_event.risk_score

    def test_performance_bulk_operations(self):
        """测试批量操作性能"""
        manager = AuditEventManager(max_events=1000)

        start_time = time.time()

        # 批量创建和添加事件
        num_events = 500
        for i in range(num_events):
            event = AuditEvent(
                event_id=f"perf_event_{i}",
                event_type=AuditEventType.SECURITY,
                severity=AuditSeverity.HIGH,
                timestamp=datetime.now(),
                user_id=f"user_{i % 10}",
                action="test_action"
            )
            manager.add_event(event)

        creation_time = time.time() - start_time

        # 验证事件数量
        assert manager.get_event_count() == num_events

        # 批量查询性能
        query_start = time.time()
        all_events = manager.get_events()
        query_time = time.time() - query_start

        assert len(all_events) == num_events

        # 性能断言（根据环境调整阈值）
        assert creation_time < 2.0  # 2秒内创建500个事件
        assert query_time < 0.5     # 0.5秒内查询500个事件

    def test_memory_management(self):
        """测试内存管理"""
        max_events = 100
        manager = AuditEventManager(max_events=max_events)

        # 填充到最大容量
        for i in range(max_events):
            event = AuditEvent(
                event_id=f"mem_event_{i}",
                event_type=AuditEventType.SECURITY,
                severity=AuditSeverity.HIGH,
                timestamp=datetime.now(),
                action="memory_test"
            )
            manager.add_event(event)

        assert manager.get_event_count() == max_events

        # 添加一个新事件，应该移除最旧的事件
        new_event = AuditEvent(
            event_id="new_mem_event",
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.HIGH,
            timestamp=datetime.now(),
            action="new_memory_test"
        )
        manager.add_event(new_event)

        # 验证数量仍然是最大值
        assert manager.get_event_count() == max_events

        # 验证最旧的事件被移除
        events = manager.get_events()
        event_ids = [e.event_id for e in events]
        assert "mem_event_0" not in event_ids  # 最旧的事件被移除
        assert "new_mem_event" in event_ids    # 新事件存在
