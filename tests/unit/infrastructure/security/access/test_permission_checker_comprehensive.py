#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
权限检查器基础测试
测试PermissionChecker的核心功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from datetime import datetime
from typing import Dict, List, Any, Set

from src.infrastructure.security.access.permission_checker import (
    PermissionChecker,
    AccessRequest,
    AccessResult,
    AccessDecision
)


@pytest.fixture
def permission_checker():
    """创建权限检查器实例"""
    checker = PermissionChecker(cache_enabled=True, cache_ttl=300)
    return checker


@pytest.fixture
def permission_checker_no_cache():
    """创建无缓存的权限检查器实例"""
    checker = PermissionChecker(cache_enabled=False)
    return checker


class TestAccessDecision:
    """测试访问决定枚举"""

    def test_access_decisions_exist(self):
        """测试访问决定定义"""
        assert AccessDecision.ALLOW.value == "allow"
        assert AccessDecision.DENY.value == "deny"
        assert AccessDecision.ABSTAIN.value == "abstain"

    def test_access_decision_values_unique(self):
        """测试访问决定值唯一"""
        values = [decision.value for decision in AccessDecision]
        assert len(values) == len(set(values))


class TestAccessRequest:
    """测试访问请求类"""

    def test_access_request_creation_minimal(self):
        """测试最小化访问请求创建"""
        request = AccessRequest(
            user_id="user123",
            resource="resource1",
            action="read"
        )

        assert request.user_id == "user123"
        assert request.resource == "resource1"
        assert request.action == "read"
        assert request.context == {}
        assert isinstance(request.timestamp, datetime)

    def test_access_request_creation_complete(self):
        """测试完整访问请求创建"""
        timestamp = datetime(2025, 1, 1, 12, 0, 0)
        context = {"ip": "192.168.1.100", "user_agent": "test"}

        request = AccessRequest(
            user_id="user123",
            resource="/api/data",
            action="write",
            context=context,
            timestamp=timestamp
        )

        assert request.user_id == "user123"
        assert request.resource == "/api/data"
        assert request.action == "write"
        assert request.context == context
        assert request.timestamp == timestamp

    def test_access_request_to_dict(self):
        """测试访问请求转换为字典"""
        timestamp = datetime(2025, 1, 1, 12, 0, 0)
        request = AccessRequest(
            user_id="user123",
            resource="/api/data",
            action="read",
            context={"source": "web"},
            timestamp=timestamp
        )

        request_dict = request.to_dict()

        assert request_dict["user_id"] == "user123"
        assert request_dict["resource"] == "/api/data"
        assert request_dict["action"] == "read"
        assert request_dict["context"] == {"source": "web"}
        assert "timestamp" in request_dict


class TestAccessResult:
    """测试访问结果类"""

    def test_access_result_creation_minimal(self):
        """测试最小化访问结果创建"""
        result = AccessResult(
            decision=AccessDecision.ALLOW
        )

        assert result.decision == AccessDecision.ALLOW
        assert result.reason == ""
        assert result.risk_score == 0.0
        assert result.processing_time == 0.0

    def test_access_result_creation_complete(self):
        """测试完整访问结果创建"""
        result = AccessResult(
            decision=AccessDecision.DENY,
            reason="insufficient_permissions",
            risk_score=8.5,
            processing_time=0.123,
            metadata={"missing_permissions": ["admin"]}
        )

        assert result.decision == AccessDecision.DENY
        assert result.reason == "insufficient_permissions"
        assert result.risk_score == 8.5
        assert result.processing_time == 0.123
        assert result.metadata == {"missing_permissions": ["admin"]}

    def test_access_result_to_dict(self):
        """测试访问结果转换为字典"""
        result = AccessResult(
            decision=AccessDecision.ALLOW,
            reason="direct_permission",
            risk_score=2.1,
            metadata={"granted_by": "role_admin"}
        )

        result_dict = result.to_dict()

        assert result_dict["decision"] == "allow"
        assert result_dict["reason"] == "direct_permission"
        assert result_dict["risk_score"] == 2.1
        assert result_dict["metadata"] == {"granted_by": "role_admin"}


class TestPermissionCheckerInitialization:
    """测试权限检查器初始化"""

    def test_initialization_with_cache(self):
        """测试带缓存初始化"""
        checker = PermissionChecker(cache_enabled=True, cache_ttl=300)

        assert checker.cache_enabled is True
        assert checker.cache_ttl == 300
        assert hasattr(checker, '_permission_cache')

    def test_initialization_without_cache(self):
        """测试无缓存初始化"""
        checker = PermissionChecker(cache_enabled=False)

        assert checker.cache_enabled is False
        assert checker.cache_ttl == 300  # 默认值


class TestPermissionCheckerAccessCheck:
    """测试权限检查器访问检查功能"""

    def test_check_access_allow_direct_permission(self, permission_checker):
        """测试直接权限允许访问"""
        checker = permission_checker
        request = AccessRequest(user_id="user1", resource="file.txt", action="read")
        user_permissions = {"read", "write"}

        result = checker.check_access(request, user_permissions)

        assert isinstance(result, AccessResult)
        assert result.decision == AccessDecision.ALLOW
        assert result.reason == "用户拥有直接权限"

    def test_check_access_deny_no_permission(self, permission_checker):
        """测试无权限拒绝访问"""
        checker = permission_checker
        request = AccessRequest(user_id="user1", resource="file.txt", action="delete")
        user_permissions = {"read", "write"}

        result = checker.check_access(request, user_permissions)

        assert isinstance(result, AccessResult)
        assert result.decision == AccessDecision.DENY

    def test_check_access_with_empty_permissions(self, permission_checker):
        """测试空权限集合"""
        checker = permission_checker
        request = AccessRequest(user_id="user1", resource="file.txt", action="read")
        user_permissions = set()

        result = checker.check_access(request, user_permissions)

        assert isinstance(result, AccessResult)
        assert result.decision == AccessDecision.DENY



class TestPermissionCheckerBasicFunctionality:
    """测试权限检查器基本功能"""

    def test_context_based_permission(self, permission_checker):
        """测试基于上下文的权限"""
        checker = permission_checker
        request = AccessRequest(
            user_id="user1",
            resource="file.txt",
            action="special",
            context={"permissions": ["special"]}
        )
        user_permissions = set()  # 空权限集合

        result = checker.check_access(request, user_permissions)

        assert isinstance(result, AccessResult)
        assert result.decision == AccessDecision.ALLOW
        assert "资源权限检查通过" in result.reason

    def test_get_stats_basic(self, permission_checker):
        """测试基本统计功能"""
        checker = permission_checker

        # 执行一些检查
        request = AccessRequest(user_id="user1", resource="file.txt", action="read")
        checker.check_access(request, {"read"})
        checker.check_access(request, {"read"})  # 缓存命中

        stats = checker.get_stats()

        assert isinstance(stats, dict)
        assert "total_checks" in stats
        assert "cache_hits" in stats
        assert stats["total_checks"] >= 2
        assert stats["cache_hits"] >= 1
