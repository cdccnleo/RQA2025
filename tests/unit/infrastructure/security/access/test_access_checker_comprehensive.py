#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试 - 访问检查器深度测试
测试AccessChecker的核心访问控制功能、缓存管理和权限验证
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict, List, Set, Optional
from dataclasses import dataclass

from infrastructure.security.access.components.access_checker import (
    AccessChecker, AccessRequest, AccessDecision
)


@dataclass
class MockUser:
    """Mock User类，匹配实际的User结构"""
    user_id: str
    username: str
    roles: Set[str]
    permissions: Set[str]


class TestAccessRequest:
    """AccessRequest测试"""

    def test_access_request_initialization_basic(self):
        """测试基本初始化"""
        request = AccessRequest(
            user_id="user123",
            resource="document",
            permission="read"
        )

        assert request.user_id == "user123"
        assert request.resource == "document"
        assert request.permission == "read"
        assert request.context == {}
        assert isinstance(request.timestamp, datetime)

    def test_access_request_initialization_with_context(self):
        """测试带上下文初始化"""
        context = {"ip": "192.168.1.1", "user_agent": "test"}
        request = AccessRequest(
            user_id="user456",
            resource="file.txt",
            permission="write",
            context=context
        )

        assert request.user_id == "user456"
        assert request.resource == "file.txt"
        assert request.permission == "write"
        assert request.context == context
        assert isinstance(request.timestamp, datetime)

    def test_access_request_initialization_with_timestamp(self):
        """测试带时间戳初始化"""
        custom_time = datetime(2024, 1, 1, 12, 0, 0)
        request = AccessRequest(
            user_id="user789",
            resource="api",
            permission="execute",
            timestamp=custom_time
        )

        assert request.timestamp == custom_time


class TestAccessCheckerInitialization:
    """AccessChecker初始化测试"""

    def test_initialization_default(self):
        """测试默认初始化"""
        checker = AccessChecker()

        assert checker.user_manager is None
        assert checker.role_manager is None
        assert checker.policy_manager is None
        assert isinstance(checker._access_cache, dict)
        assert checker._max_cache_size == 10000
        assert checker._cache_ttl_seconds == 300
        assert isinstance(checker._cache_timestamps, dict)

    def test_initialization_with_managers(self):
        """测试带管理器初始化"""
        mock_user_manager = MagicMock()
        mock_role_manager = MagicMock()
        mock_policy_manager = MagicMock()

        checker = AccessChecker(
            user_manager=mock_user_manager,
            role_manager=mock_role_manager,
            policy_manager=mock_policy_manager
        )

        assert checker.user_manager == mock_user_manager
        assert checker.role_manager == mock_role_manager
        assert checker.policy_manager == mock_policy_manager

    def test_initialization_custom_cache_settings(self):
        """测试自定义缓存设置初始化"""
        checker = AccessChecker(max_cache_size=5000, cache_ttl_seconds=600)

        assert checker._max_cache_size == 5000
        assert checker._cache_ttl_seconds == 600


class TestAccessCheckerBasicAccessCheck:
    """AccessChecker基本访问检查测试"""

    @pytest.fixture
    def checker(self):
        """AccessChecker fixture"""
        return AccessChecker()

    @pytest.fixture
    def mock_user_manager(self):
        """Mock UserManager fixture"""
        manager = MagicMock()
        mock_user = MockUser(
            user_id="test_user",
            username="testuser",
            roles={"user", "editor"},
            permissions={"read", "write:document"}
        )
        manager.get_user.return_value = mock_user
        return manager

    @pytest.fixture
    def checker_with_managers(self, mock_user_manager):
        """带管理器的AccessChecker fixture"""
        mock_role_manager = MagicMock()
        mock_role_manager.get_role_permissions.return_value = {"read", "write"}

        return AccessChecker(
            user_manager=mock_user_manager,
            role_manager=mock_role_manager
        )

    def test_check_access_user_not_found(self, checker):
        """测试用户未找到的情况"""
        checker.user_manager = MagicMock()
        checker.user_manager.get_user.return_value = None

        decision = checker.check_access("nonexistent", "resource", "read")

        assert decision == AccessDecision.DENY

    def test_check_access_successful(self, checker_with_managers):
        """测试成功的访问检查"""
        decision = checker_with_managers.check_access("test_user", "document", "read")

        assert decision == AccessDecision.ALLOW

    def test_check_access_denied(self, checker_with_managers):
        """测试拒绝的访问检查"""
        decision = checker_with_managers.check_access("test_user", "admin", "delete")

        assert decision == AccessDecision.DENY

    def test_check_access_request_object(self, checker_with_managers):
        """测试使用AccessRequest对象的访问检查"""
        request = AccessRequest(
            user_id="test_user",
            resource="document",
            permission="read"
        )

        decision = checker_with_managers.check_access_request(request)

        assert decision == AccessDecision.ALLOW

    @pytest.mark.asyncio
    async def test_check_access_async(self, checker_with_managers):
        """测试异步访问检查"""
        decision = await checker_with_managers.check_access_async("test_user", "document", "read")

        assert decision == AccessDecision.ALLOW

    def test_batch_check_access(self, checker_with_managers):
        """测试批量访问检查"""
        requests = [
            AccessRequest(user_id="test_user", resource="document", permission="read"),
            AccessRequest(user_id="test_user", resource="admin", permission="delete"),
            AccessRequest(user_id="test_user", resource="document", permission="write")
        ]

        decisions = checker_with_managers.batch_check_access(requests)

        assert len(decisions) == 3
        assert decisions[0] == AccessDecision.ALLOW  # read allowed
        assert decisions[1] == AccessDecision.DENY   # delete denied
        assert decisions[2] == AccessDecision.ALLOW  # write allowed

    @pytest.mark.asyncio
    async def test_batch_check_access_async(self, checker_with_managers):
        """测试异步批量访问检查"""
        requests = [
            AccessRequest(user_id="test_user", resource="document", permission="read"),
            AccessRequest(user_id="test_user", resource="document", permission="write")
        ]

        decisions = await checker_with_managers.batch_check_access_async(requests)

        assert len(decisions) == 2
        assert all(d == AccessDecision.ALLOW for d in decisions)


class TestAccessCheckerCacheManagement:
    """AccessChecker缓存管理测试"""

    @pytest.fixture
    def checker(self):
        """AccessChecker fixture"""
        return AccessChecker(max_cache_size=100, cache_ttl_seconds=60)

    def test_cache_initially_empty(self, checker):
        """测试缓存初始为空"""
        assert len(checker._access_cache) == 0
        assert len(checker._cache_timestamps) == 0

    def test_cache_hit(self, checker):
        """测试缓存命中"""
        request = AccessRequest(user_id="user1", resource="res1", permission="read")
        cache_key = f"{request.user_id}:{request.resource}:{request.permission}"

        # 手动设置缓存
        checker._access_cache[cache_key] = AccessDecision.ALLOW
        checker._cache_timestamps[cache_key] = datetime.now()

        decision = checker._check_cache(request)

        assert decision == AccessDecision.ALLOW

    def test_cache_miss(self, checker):
        """测试缓存未命中"""
        request = AccessRequest(user_id="user1", resource="res1", permission="read")

        decision = checker._check_cache(request)

        assert decision is None

    def test_cache_expired(self, checker):
        """测试缓存过期"""
        request = AccessRequest(user_id="user1", resource="res1", permission="read")
        cache_key = f"{request.user_id}:{request.resource}:{request.permission}"

        # 设置过期的缓存
        expired_time = datetime.now() - timedelta(seconds=120)  # 超过TTL
        checker._access_cache[cache_key] = AccessDecision.ALLOW
        checker._cache_timestamps[cache_key] = expired_time

        decision = checker._check_cache(request)

        assert decision is None  # 应该返回None，因为缓存过期

    def test_update_cache(self, checker):
        """测试更新缓存"""
        request = AccessRequest(user_id="user1", resource="res1", permission="read")

        checker._update_cache(request, AccessDecision.ALLOW)

        cache_key = f"{request.user_id}:{request.resource}:{request.permission}"
        assert cache_key in checker._access_cache
        assert checker._access_cache[cache_key] == AccessDecision.ALLOW
        assert cache_key in checker._cache_timestamps

    def test_cache_size_limit_enforcement(self, checker):
        """测试缓存大小限制执行"""
        # 填充缓存到接近限制
        for i in range(101):  # 超过max_cache_size=100
            request = AccessRequest(user_id=f"user{i}", resource="res", permission="read")
            checker._update_cache(request, AccessDecision.ALLOW)

        # 缓存大小应该被限制
        assert len(checker._access_cache) <= checker._max_cache_size

    def test_clear_cache(self, checker):
        """测试清空缓存"""
        # 添加一些缓存项
        for i in range(5):
            request = AccessRequest(user_id=f"user{i}", resource="res", permission="read")
            checker._update_cache(request, AccessDecision.ALLOW)

        assert len(checker._access_cache) > 0

        # 清空缓存
        checker.clear_cache()

        assert len(checker._access_cache) == 0
        assert len(checker._cache_timestamps) == 0

    def test_invalidate_user_cache(self, checker):
        """测试使特定用户的缓存无效"""
        # 添加多个用户的缓存
        users = ["user1", "user2", "user3"]
        for user in users:
            request = AccessRequest(user_id=user, resource="res", permission="read")
            checker._update_cache(request, AccessDecision.ALLOW)

        assert len(checker._access_cache) == 3

        # 使user2的缓存无效
        checker.invalidate_user_cache("user2")

        # user2的缓存应该被清除，其他用户保持
        assert len(checker._access_cache) == 2
        remaining_keys = list(checker._access_cache.keys())
        assert not any("user2:" in key for key in remaining_keys)

    def test_get_cache_stats(self, checker):
        """测试获取缓存统计"""
        # 添加一些缓存项
        for i in range(3):
            request = AccessRequest(user_id=f"user{i}", resource="res", permission="read")
            checker._update_cache(request, AccessDecision.ALLOW)

        stats = checker.get_cache_stats()

        assert isinstance(stats, dict)
        assert "total_entries" in stats
        assert "hit_rate" in stats
        assert "oldest_entry_age_seconds" in stats
        assert stats["total_entries"] == 3


class TestAccessCheckerPermissionEvaluation:
    """AccessChecker权限评估测试"""

    @pytest.fixture
    def mock_user(self):
        """Mock User fixture"""
        return MockUser(
            user_id="test_user",
            username="testuser",
            roles={"user", "editor"},
            permissions={"read", "write:document", "edit"}
        )

    @pytest.fixture
    def checker(self):
        """AccessChecker fixture"""
        return AccessChecker()

    def test_get_user_permissions_basic(self, checker, mock_user):
        """测试获取用户基本权限"""
        permissions = checker._get_user_permissions(mock_user)

        expected_permissions = {"read", "write:document", "edit"}
        assert permissions == expected_permissions

    def test_get_user_permissions_with_roles(self, checker, mock_user):
        """测试获取包含角色权限的用户权限"""
        # Mock role manager
        mock_role_manager = MagicMock()
        checker.role_manager = mock_role_manager

        # Mock role permissions
        role_permissions = {"view:reports", "create:documents"}
        mock_role_manager.get_role_permissions.return_value = role_permissions

        permissions = checker._get_user_permissions(mock_user)

        # 应该包含用户直接权限和角色权限
        expected = {"read", "write:document", "edit", "view:reports", "create:documents"}
        assert permissions == expected

    def test_check_user_permissions_exact_match(self, checker, mock_user):
        """测试精确权限匹配"""
        request = AccessRequest(
            user_id="test_user",
            resource="document",
            permission="write"
        )

        # 用户有 "write:document" 权限
        decision = checker._check_user_permissions(mock_user, request)

        assert decision == AccessDecision.ALLOW

    def test_check_user_permissions_full_resource_permission(self, checker, mock_user):
        """测试完整资源权限匹配"""
        request = AccessRequest(
            user_id="test_user",
            resource="document",
            permission="write"
        )

        decision = checker._check_user_permissions(mock_user, request)

        # 用户有 "write:document" 权限，应该允许
        assert decision == AccessDecision.ALLOW

    def test_check_user_permissions_permission_only(self, checker, mock_user):
        """测试仅权限匹配"""
        request = AccessRequest(
            user_id="test_user",
            resource="file",
            permission="read"
        )

        decision = checker._check_user_permissions(mock_user, request)

        # 用户有 "read" 权限，应该允许
        assert decision == AccessDecision.ALLOW

    def test_check_user_permissions_no_match(self, checker, mock_user):
        """测试无权限匹配"""
        request = AccessRequest(
            user_id="test_user",
            resource="admin",
            permission="delete"
        )

        decision = checker._check_user_permissions(mock_user, request)

        assert decision == AccessDecision.DENY

    def test_check_user_permissions_wildcard(self, checker, mock_user):
        """测试通配符权限"""
        # 添加通配符权限到用户
        mock_user.permissions.add("admin:*")

        request = AccessRequest(
            user_id="test_user",
            resource="admin",
            permission="delete"
        )

        decision = checker._check_user_permissions(mock_user, request)

        assert decision == AccessDecision.ALLOW


class TestAccessCheckerPolicyEvaluation:
    """AccessChecker策略评估测试"""

    @pytest.fixture
    def checker(self):
        """AccessChecker fixture"""
        return AccessChecker()

    @pytest.fixture
    def mock_user(self):
        """Mock User fixture"""
        return MockUser(
            user_id="test_user",
            username="testuser",
            roles={"user"},
            permissions={"read"}
        )

    def test_evaluate_access_policies_no_policies(self, checker, mock_user):
        """测试无策略时的策略评估"""
        request = AccessRequest(
            user_id="test_user",
            resource="document",
            permission="read"
        )

        decision = checker._evaluate_access_policies(mock_user, request)

        # 无策略时应该弃权
        assert decision == AccessDecision.ABSTAIN

    def test_evaluate_access_policies_with_policy_manager(self, checker, mock_user):
        """测试带策略管理器的策略评估"""
        # Mock policy manager
        mock_policy_manager = MagicMock()
        mock_policy_manager.evaluate_policies.return_value = AccessDecision.ALLOW
        checker.policy_manager = mock_policy_manager

        request = AccessRequest(
            user_id="test_user",
            resource="document",
            permission="read"
        )

        decision = checker._evaluate_access_policies(mock_user, request)

        assert decision == AccessDecision.ALLOW
        mock_policy_manager.evaluate_policies.assert_called_once_with(mock_user, request)


class TestAccessCheckerEdgeCases:
    """AccessChecker边界情况测试"""

    @pytest.fixture
    def checker(self):
        """AccessChecker fixture"""
        return AccessChecker()

    def test_check_access_with_none_managers(self, checker):
        """测试管理器为None时的访问检查"""
        decision = checker.check_access("user", "resource", "permission")

        assert decision == AccessDecision.DENY

    def test_check_access_request_with_special_characters(self, checker):
        """测试特殊字符的访问请求检查"""
        special_user_id = "user@domain.com:123"
        special_resource = "api/v1/users/123?filter=active"
        special_permission = "read:admin"

        request = AccessRequest(
            user_id=special_user_id,
            resource=special_resource,
            permission=special_permission
        )

        # 即使没有管理器，也应该能处理特殊字符
        decision = checker.check_access_request(request)

        assert decision == AccessDecision.DENY

    def test_batch_check_access_empty_list(self, checker):
        """测试空列表的批量访问检查"""
        decisions = checker.batch_check_access([])

        assert decisions == []

    def test_batch_check_access_large_batch(self, checker):
        """测试大批量访问检查"""
        # 创建100个请求
        requests = [
            AccessRequest(
                user_id=f"user{i}",
                resource=f"resource{i}",
                permission="read"
            )
            for i in range(100)
        ]

        decisions = checker.batch_check_access(requests)

        assert len(decisions) == 100
        assert all(d == AccessDecision.DENY for d in decisions)

    def test_cache_key_generation(self, checker):
        """测试缓存键生成"""
        request1 = AccessRequest(user_id="user1", resource="res1", permission="read")
        request2 = AccessRequest(user_id="user1", resource="res1", permission="write")
        request3 = AccessRequest(user_id="user2", resource="res1", permission="read")

        key1 = f"{request1.user_id}:{request1.resource}:{request1.permission}"
        key2 = f"{request2.user_id}:{request2.resource}:{request2.permission}"
        key3 = f"{request3.user_id}:{request3.resource}:{request3.permission}"

        # 键应该不同
        assert key1 != key2
        assert key1 != key3
        assert key2 != key3

        # 手动验证键格式
        assert key1 == "user1:res1:read"
        assert key2 == "user1:res1:write"
        assert key3 == "user2:res1:read"

    def test_concurrent_cache_access(self, checker):
        """测试并发缓存访问"""
        import threading
        import time

        exceptions = []
        results = []

        def cache_worker(worker_id: int):
            """缓存工作线程"""
            try:
                for i in range(50):
                    request = AccessRequest(
                        user_id=f"user{worker_id}",
                        resource=f"res{i}",
                        permission="read"
                    )

                    # 随机读写缓存
                    if i % 2 == 0:
                        checker._update_cache(request, AccessDecision.ALLOW)
                    else:
                        decision = checker._check_cache(request)
                        results.append((worker_id, i, decision))
            except Exception as e:
                exceptions.append(f"Worker {worker_id}: {e}")

        # 启动5个并发线程
        threads = []
        for i in range(5):
            t = threading.Thread(target=cache_worker, args=(i,))
            threads.append(t)
            t.start()

        # 等待完成
        for t in threads:
            t.join()

        # 不应该有异常
        assert len(exceptions) == 0, f"Concurrent exceptions: {exceptions}"

        # 应该有一些缓存操作成功
        assert len(results) > 0


class TestAccessCheckerLogging:
    """AccessChecker日志测试"""

    @pytest.fixture
    def checker(self):
        """AccessChecker fixture"""
        return AccessChecker()

    @patch('src.infrastructure.security.access.components.access_checker.logger')
    def test_log_access_check(self, mock_logger, checker):
        """测试访问检查日志"""
        request = AccessRequest(
            user_id="test_user",
            resource="document",
            permission="read",
            context={"ip": "192.168.1.1"}
        )

        checker._log_access_check(request, AccessDecision.ALLOW)

        # 验证日志调用
        mock_logger.info.assert_called_once()
        log_call = mock_logger.info.call_args[0][0]

        # 验证日志内容包含关键信息
        assert "test_user" in log_call
        assert "document" in log_call
        assert "read" in log_call
        assert "ALLOW" in log_call


class TestAccessCheckerIntegration:
    """AccessChecker集成测试"""

    def test_complete_access_check_workflow(self):
        """测试完整的访问检查工作流"""
        # 创建完整的组件
        mock_user_manager = MagicMock()
        mock_role_manager = MagicMock()
        mock_policy_manager = MagicMock()

        # 设置用户管理器
        mock_user = MockUser(
            user_id="workflow_user",
            username="workflowuser",
            roles={"user", "editor"},
            permissions={"read", "write:document"}
        )
        mock_user_manager.get_user.return_value = mock_user

        # 设置角色管理器
        mock_role_manager.get_role_permissions.return_value = {"view:reports"}
        mock_role_manager.get_role_by_name.return_value = MagicMock(
            role_id="editor",
            permissions={"write:document"}
        )

        # 设置策略管理器
        mock_policy_manager.evaluate_policies.return_value = AccessDecision.ALLOW

        checker = AccessChecker(
            user_manager=mock_user_manager,
            role_manager=mock_role_manager,
            policy_manager=mock_policy_manager
        )

        # 执行访问检查
        request = AccessRequest(
            user_id="workflow_user",
            resource="document",
            permission="write"
        )

        # 第一次检查（缓存未命中）
        decision1 = checker.check_access_request(request)
        assert decision1 == AccessDecision.ALLOW

        # 验证管理器调用
        mock_user_manager.get_user.assert_called_with("workflow_user")
        mock_role_manager.get_role_permissions.assert_called()
        mock_policy_manager.evaluate_policies.assert_called_with(mock_user, request)

        # 第二次检查（应该使用缓存）
        decision2 = checker.check_access_request(request)
        assert decision2 == AccessDecision.ALLOW

        # 验证缓存统计
        stats = checker.get_cache_stats()
        assert stats["total_entries"] >= 1

    def test_access_checker_with_realistic_scenario(self):
        """测试现实场景的访问检查器"""
        # 模拟一个文档管理系统场景

        # 创建用户管理器
        user_manager = MagicMock()
        role_manager = MagicMock()

        # 定义用户和角色
        admin_user = MockUser(
            user_id="admin",
            username="administrator",
            roles={"admin"},
            permissions={"*"}  # 超级权限
        )

        editor_user = MockUser(
            user_id="editor1",
            username="editor",
            roles={"editor"},
            permissions={"read", "write:document", "edit:document"}
        )

        viewer_user = MockUser(
            user_id="viewer1",
            username="viewer",
            roles={"viewer"},
            permissions={"read"}
        )

        def mock_get_user(user_id):
            users = {
                "admin": admin_user,
                "editor1": editor_user,
                "viewer1": viewer_user
            }
            return users.get(user_id)

        user_manager.get_user = mock_get_user

        # 定义角色权限
        def mock_get_role_permissions(role_id, include_inherited=False):
            role_permissions = {
                "admin": {"*", "manage:users", "delete:any"},
                "editor": {"read", "write:document", "edit:document", "create:document"},
                "viewer": {"read"}
            }
            return role_permissions.get(role_id, set())

        role_manager.get_role_permissions = mock_get_role_permissions

        # 创建访问检查器
        checker = AccessChecker(
            user_manager=user_manager,
            role_manager=role_manager
        )

        # 测试各种访问场景
        test_cases = [
            # (user_id, resource, permission, expected_decision)
            ("admin", "users", "manage", AccessDecision.ALLOW),        # 管理员可以管理用户
            ("admin", "documents", "delete", AccessDecision.ALLOW),     # 管理员可以删除任何文档
            ("editor1", "document.txt", "write", AccessDecision.ALLOW), # 编辑可以写文档
            ("editor1", "users", "manage", AccessDecision.DENY),        # 编辑不能管理用户
            ("viewer1", "document.txt", "read", AccessDecision.ALLOW),  # 查看者可以读文档
            ("viewer1", "document.txt", "write", AccessDecision.DENY),  # 查看者不能写文档
            ("nonexistent", "anything", "any", AccessDecision.DENY),    # 不存在的用户被拒绝
        ]

        for user_id, resource, permission, expected in test_cases:
            decision = checker.check_access(user_id, resource, permission)
            assert decision == expected, f"Failed for {user_id}:{resource}:{permission}, got {decision}, expected {expected}"

    def test_performance_under_load(self):
        """测试负载下的性能"""
        checker = AccessChecker(max_cache_size=1000)

        # 创建大量请求
        requests = []
        for i in range(100):
            request = AccessRequest(
                user_id=f"user{i % 10}",  # 10个用户循环
                resource=f"resource{i % 20}",  # 20个资源循环
                permission=["read", "write", "delete"][i % 3]  # 3种权限循环
            )
            requests.append(request)

        import time
        start_time = time.time()

        # 执行批量检查
        decisions = checker.batch_check_access(requests)

        end_time = time.time()
        duration = end_time - start_time

        # 应该在合理时间内完成
        assert duration < 5.0, f"Batch check took too long: {duration}s"
        assert len(decisions) == 100

        # 大部分应该是DENY（因为没有真实的权限管理器）
        deny_count = sum(1 for d in decisions if d == AccessDecision.DENY)
        assert deny_count >= 90  # 至少90%被拒绝