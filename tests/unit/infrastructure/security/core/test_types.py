#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 安全模块类型测试

测试安全模块中的所有类型定义，包括枚举和数据类
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, Set

from src.infrastructure.security.core.types import (
    UserRole, Permission, User, UserSession, AccessPolicy,
    EventType, EventSeverity, AuditEventParams, UserCreationParams,
    EncryptionParams
)


class TestEnums:
    """测试枚举类"""

    def test_user_role_enum(self):
        """测试UserRole枚举"""
        assert UserRole.ADMIN.value == "admin"
        assert UserRole.TRADER.value == "trader"
        assert UserRole.ANALYST.value == "analyst"
        assert UserRole.AUDITOR.value == "auditor"
        assert UserRole.GUEST.value == "guest"

        # 测试所有值都是唯一的
        values = [role.value for role in UserRole]
        assert len(values) == len(set(values))

    def test_permission_enum(self):
        """测试Permission枚举"""
        # 交易权限
        assert Permission.TRADE_EXECUTE.value == "trade:execute"
        assert Permission.TRADE_CANCEL.value == "trade:cancel"
        assert Permission.ORDER_PLACE.value == "order:place"
        assert Permission.ORDER_CANCEL.value == "order:cancel"

        # 数据权限
        assert Permission.DATA_READ.value == "data:read"
        assert Permission.DATA_WRITE.value == "data:write"
        assert Permission.DATA_EXPORT.value == "data:export"

        # 系统权限
        assert Permission.SYSTEM_CONFIG.value == "system:config"
        assert Permission.SYSTEM_MONITOR.value == "system:monitor"
        assert Permission.USER_MANAGE.value == "user:manage"

        # 审计权限
        assert Permission.AUDIT_READ.value == "audit:read"
        assert Permission.AUDIT_EXPORT.value == "audit:export"

    def test_event_type_enum(self):
        """测试EventType枚举"""
        assert EventType.SECURITY.value == "security"
        assert EventType.ACCESS.value == "access"
        assert EventType.DATA_OPERATION.value == "data_operation"
        assert EventType.CONFIG_CHANGE.value == "config_change"
        assert EventType.USER_MANAGEMENT.value == "user_management"
        assert EventType.SYSTEM_EVENT.value == "system_event"
        assert EventType.COMPLIANCE.value == "compliance"

    def test_event_severity_enum(self):
        """测试EventSeverity枚举"""
        assert EventSeverity.LOW.value == "low"
        assert EventSeverity.MEDIUM.value == "medium"
        assert EventSeverity.HIGH.value == "high"
        assert EventSeverity.CRITICAL.value == "critical"


class TestUser:
    """测试User数据类"""

    def test_user_creation_minimal(self):
        """测试User最小化创建"""
        user = User(
            user_id="user123",
            username="testuser",
            email="test@example.com",
            roles={UserRole.GUEST},
            is_active=True,
            created_at=datetime.now()
        )

        assert user.user_id == "user123"
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.roles == {UserRole.GUEST}
        assert user.is_active is True
        assert isinstance(user.created_at, datetime)
        assert user.last_login is None
        assert user.password_hash is None
        assert user.failed_login_attempts == 0
        assert user.locked_until is None

    def test_user_creation_full(self):
        """测试User完整创建"""
        now = datetime.now()
        future = now + timedelta(hours=1)

        user = User(
            user_id="user123",
            username="testuser",
            email="test@example.com",
            roles={UserRole.ADMIN, UserRole.TRADER},
            is_active=True,
            created_at=now,
            last_login=now,
            password_hash="hashed_password",
            failed_login_attempts=3,
            locked_until=future
        )

        assert user.user_id == "user123"
        assert user.roles == {UserRole.ADMIN, UserRole.TRADER}
        assert user.last_login == now
        assert user.password_hash == "hashed_password"
        assert user.failed_login_attempts == 3
        assert user.locked_until == future


class TestUserSession:
    """测试UserSession数据类"""

    def test_user_session_creation(self):
        """测试UserSession创建"""
        now = datetime.now()
        future = now + timedelta(hours=1)

        session = UserSession(
            session_id="session123",
            user_id="user123",
            created_at=now,
            expires_at=future,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            is_active=True
        )

        assert session.session_id == "session123"
        assert session.user_id == "user123"
        assert session.created_at == now
        assert session.expires_at == future
        assert session.ip_address == "192.168.1.1"
        assert session.user_agent == "Mozilla/5.0"
        assert session.is_active is True

    def test_user_session_not_expired(self):
        """测试未过期的会话"""
        now = datetime.now()
        future = now + timedelta(hours=1)

        session = UserSession(
            session_id="session123",
            user_id="user123",
            created_at=now,
            expires_at=future
        )

        assert not session.is_expired()

    def test_user_session_expired(self):
        """测试已过期的会话"""
        now = datetime.now()
        past = now - timedelta(hours=1)

        session = UserSession(
            session_id="session123",
            user_id="user123",
            created_at=past,
            expires_at=past
        )

        assert session.is_expired()


class TestAccessPolicy:
    """测试AccessPolicy数据类"""

    def test_access_policy_creation(self):
        """测试AccessPolicy创建"""
        policy = AccessPolicy(
            policy_id="policy123",
            name="Test Policy",
            description="A test access policy",
            resource_pattern="/api/*",
            permissions={Permission.DATA_READ, Permission.DATA_WRITE},
            roles={UserRole.ADMIN, UserRole.TRADER},
            conditions={"time_range": "business_hours"}
        )

        assert policy.policy_id == "policy123"
        assert policy.name == "Test Policy"
        assert policy.description == "A test access policy"
        assert policy.resource_pattern == "/api/*"
        assert policy.permissions == {Permission.DATA_READ, Permission.DATA_WRITE}
        assert policy.roles == {UserRole.ADMIN, UserRole.TRADER}
        assert policy.conditions == {"time_range": "business_hours"}


class TestAuditEventParams:
    """测试AuditEventParams数据类"""

    def test_audit_event_params_creation_minimal(self):
        """测试AuditEventParams最小化创建"""
        params = AuditEventParams(
            event_type=EventType.SECURITY,
            severity=EventSeverity.MEDIUM
        )

        assert params.event_type == EventType.SECURITY
        assert params.severity == EventSeverity.MEDIUM
        assert params.user_id is None
        assert params.session_id is None
        assert params.resource is None
        assert params.action is None
        assert params.result is None
        assert params.details == {}
        assert params.ip_address is None
        assert params.user_agent is None
        assert params.location is None
        assert params.risk_score == 0.0
        assert params.tags == set()
        assert isinstance(params.timestamp, datetime)

    def test_audit_event_params_creation_full(self):
        """测试AuditEventParams完整创建"""
        custom_time = datetime.now()

        params = AuditEventParams(
            event_type=EventType.ACCESS,
            severity=EventSeverity.HIGH,
            user_id="user123",
            session_id="session123",
            resource="/api/data",
            action="read",
            result="success",
            details={"query": "SELECT * FROM users"},
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            location="office",
            risk_score=0.7,
            tags={"suspicious", "admin_access"},
            timestamp=custom_time
        )

        assert params.event_type == EventType.ACCESS
        assert params.severity == EventSeverity.HIGH
        assert params.user_id == "user123"
        assert params.session_id == "session123"
        assert params.resource == "/api/data"
        assert params.action == "read"
        assert params.result == "success"
        assert params.details == {"query": "SELECT * FROM users"}
        assert params.ip_address == "192.168.1.1"
        assert params.user_agent == "Mozilla/5.0"
        assert params.location == "office"
        assert params.risk_score == 0.7
        assert params.tags == {"suspicious", "admin_access"}
        assert params.timestamp == custom_time

    def test_audit_event_params_timestamp_auto_set(self):
        """测试AuditEventParams时间戳自动设置"""
        before = datetime.now()
        params = AuditEventParams(
            event_type=EventType.SECURITY,
            severity=EventSeverity.LOW
        )
        after = datetime.now()

        assert before <= params.timestamp <= after


class TestUserCreationParams:
    """测试UserCreationParams数据类"""

    def test_user_creation_params_minimal(self):
        """测试UserCreationParams最小化创建"""
        params = UserCreationParams(
            username="testuser"
        )

        assert params.username == "testuser"
        assert params.email is None
        assert params.roles == set()
        assert params.permissions == set()
        assert params.is_active is True
        assert params.metadata == {}
        assert params.created_by is None
        assert params.password is None
        assert params.require_password_change is False
        assert params.expiry_date is None

    def test_user_creation_params_full(self):
        """测试UserCreationParams完整创建"""
        future = datetime.now() + timedelta(days=90)

        params = UserCreationParams(
            username="testuser",
            email="test@example.com",
            roles={"admin", "trader"},
            permissions={"read", "write"},
            is_active=False,
            metadata={"department": "IT"},
            created_by="admin_user",
            password="secret123",
            require_password_change=True,
            expiry_date=future
        )

        assert params.username == "testuser"
        assert params.email == "test@example.com"
        assert params.roles == {"admin", "trader"}
        assert params.permissions == {"read", "write"}
        assert params.is_active is False
        assert params.metadata == {"department": "IT"}
        assert params.created_by == "admin_user"
        assert params.password == "secret123"
        assert params.require_password_change is True
        assert params.expiry_date == future


class TestOtherTypes:
    """测试其他类型"""

    def test_encryption_params_creation(self):
        """测试EncryptionParams创建"""
        params = EncryptionParams(
            algorithm="AES-256-GCM",
            key_size=256,
            mode="GCM",
            enable_compression=True
        )

        assert params.algorithm == "AES-256-GCM"
        assert params.key_size == 256
        assert params.mode == "GCM"
        assert params.enable_compression is True
