#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
访问控制管理器综合测试
测试重构版AccessControlManager的核心功能和组件协同
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import asyncio
import time
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import tempfile
import os
from pathlib import Path

from src.infrastructure.security.access.access_control import AccessControlManager
from src.infrastructure.security.core.types import UserRole


@pytest.fixture
def temp_config_dir():
    """创建临时配置目录"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # 清理
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def access_control_manager(temp_config_dir):
    """创建访问控制管理器实例"""
    manager = AccessControlManager(
        config_path=temp_config_dir,
        enable_audit=True,
        cache_enabled=True,
        max_cache_size=100
    )
    return manager


class TestAccessControlManagerInitialization:
    """测试访问控制管理器初始化"""

    def test_initialization_with_default_params(self):
        """测试默认参数初始化"""
        manager = AccessControlManager()

        assert manager.config_path is not None
        assert manager.enable_audit is True
        assert manager.cache_enabled is True

        # 验证组件初始化
        assert hasattr(manager, 'user_manager')
        assert hasattr(manager, 'role_manager')
        assert hasattr(manager, 'access_checker')
        assert hasattr(manager, 'policy_manager')
        assert hasattr(manager, 'audit_logger')
        assert hasattr(manager, 'config_manager')
        assert hasattr(manager, 'cache_manager')

    def test_initialization_with_custom_params(self, temp_config_dir):
        """测试自定义参数初始化"""
        manager = AccessControlManager(
            config_path=temp_config_dir,
            enable_audit=False,
            cache_enabled=False,
            max_cache_size=500
        )

        assert str(manager.config_path) == temp_config_dir
        assert manager.enable_audit is False
        assert manager.cache_enabled is False

    def test_component_initialization(self, access_control_manager):
        """测试组件正确初始化"""
        manager = access_control_manager

        # 验证所有组件都被正确初始化
        assert manager.user_manager is not None
        assert manager.role_manager is not None
        assert manager.access_checker is not None
        assert manager.policy_manager is not None
        assert manager.audit_logger is not None
        assert manager.config_manager is not None
        assert manager.cache_manager is not None


class TestUserManagement:
    """测试用户管理功能"""

    def test_create_user(self, access_control_manager):
        """测试创建用户"""
        manager = access_control_manager

        # 创建用户
        user_id = manager.create_user(
            username="testuser",
            email="test@example.com"
        )

        assert user_id is not None

        # 验证用户存在
        user_info = manager.user_manager.get_user(user_id)
        assert user_info is not None
        assert user_info.username == "testuser"
        assert user_info.email == "test@example.com"

    def test_create_user_with_roles(self, access_control_manager):
        """测试创建带角色的用户"""
        manager = access_control_manager

        # 创建用户并分配角色
        user_id = manager.create_user(
            username="roleuser",
            email="role@example.com",
            roles=["trader", "analyst"]
        )

        assert user_id is not None

        # 验证角色分配
        user_info = manager.user_manager.get_user(user_id)
        assert user_info is not None
        assert "trader" in user_info.roles
        assert "analyst" in user_info.roles

    def test_assign_role_to_user(self, access_control_manager):
        """测试为用户分配角色"""
        manager = access_control_manager

        # 创建用户
        user_id = manager.create_user("assignuser", "assign@example.com")

        # 先创建角色
        role_id = manager.create_role("Admin Role", permissions=["admin:manage"])

        # 分配角色
        success = manager.assign_role_to_user(user_id, role_id)
        assert success

        # 验证角色分配
        user_info = manager.user_manager.get_user(user_id)
        assert user_info is not None
        assert "Admin Role" in user_info.roles

    def test_revoke_role_from_user(self, access_control_manager):
        """测试从用户撤销角色"""
        manager = access_control_manager

        # 创建用户（不指定角色）
        user_id = manager.create_user("revokeuser", "revoke@example.com")

        # 创建角色
        role_id = manager.create_role("Trader", permissions=["trade:execute"])

        # 为用户分配角色
        manager.assign_role_to_user(user_id, role_id)

        # 验证角色已分配
        user_info = manager.user_manager.get_user(user_id)
        assert user_info is not None
        assert "Trader" in user_info.roles

        # 撤销角色
        success = manager.revoke_role_from_user(user_id, role_id)
        assert success

        # 验证角色已被撤销
        user_info = manager.user_manager.get_user(user_id)
        assert user_info is not None
        assert "Trader" not in user_info.roles


class TestRoleManagement:
    """测试角色管理功能"""

    def test_create_role(self, access_control_manager):
        """测试创建角色"""
        manager = access_control_manager

        # 创建角色
        role_id = manager.create_role(
            name="Test Role",
            description="A test role",
            permissions=["read:data", "write:reports"]
        )

        assert role_id is not None

        # 验证角色存在
        role_info = manager.role_manager.get_role(role_id)
        assert role_info is not None
        assert role_info.name == "Test Role"
        assert "read:data" in role_info.permissions

    def test_create_role_with_parent(self, access_control_manager):
        """测试创建带父角色的角色"""
        manager = access_control_manager

        # 创建父角色
        parent_role_id = manager.create_role("Parent Role", permissions=["read:basic"])

        # 创建子角色
        child_role_id = manager.create_role(
            name="Child Role",
            description="Child role with inheritance",
            permissions=["write:advanced"]
        )

        # 手动设置父角色关系
        child_role = manager.role_manager.get_role(child_role_id)
        child_role.add_parent_role(parent_role_id)

        assert child_role_id is not None

        # 验证继承关系
        child_role = manager.role_manager.get_role(child_role_id)
        assert parent_role_id in child_role.parent_roles

    def test_get_role_permissions_with_inheritance(self, access_control_manager):
        """测试获取角色权限（包含继承）"""
        manager = access_control_manager

        # 创建父角色
        parent_id = manager.create_role("Parent", permissions=["read:basic"])

        # 创建子角色
        child_id = manager.create_role(
            name="Child",
            permissions=["write:advanced"]
        )

        # 设置父角色关系
        child_role = manager.role_manager.get_role(child_id)
        child_role.add_parent_role(parent_id)

        # 获取子角色的所有权限
        permissions = manager.role_manager.get_role_permissions(child_id, include_inherited=True)

        assert "read:basic" in permissions  # 从父角色继承
        assert "write:advanced" in permissions  # 自身权限


class TestAccessControl:
    """测试访问控制功能"""

    def test_check_access_allowed(self, access_control_manager):
        """测试允许访问"""
        manager = access_control_manager

        # 创建用户和角色
        user_id = manager.create_user("accessuser", "access@example.com", roles=["trader"])
        trader_role_id = manager.create_role("Trader Role", permissions=["trade:execute", "data:read"])

        # 为用户分配角色
        manager.assign_role_to_user(user_id, trader_role_id)

        # 检查访问权限
        decision = manager.check_access(user_id, "trade", "execute")
        assert decision.value == "allow"

    def test_check_access_denied(self, access_control_manager):
        """测试拒绝访问"""
        manager = access_control_manager

        # 创建用户（无权限）
        user_id = manager.create_user("denieduser", "denied@example.com", roles=["guest"])

        # 检查访问权限
        decision = manager.check_access(user_id, "admin", "manage")
        assert decision.value == "deny"

    @pytest.mark.asyncio
    async     def test_check_access_async(self, access_control_manager):
        """测试异步访问检查"""
        manager = access_control_manager

        # 创建用户
        user_id = manager.create_user("asyncuser", "async@example.com", roles=["trader"])

        # 异步检查访问权限
        decision = await manager.check_access_async(user_id, "trade", "execute")

        # 注意：这里可能需要根据实际实现调整断言
        assert decision is not None

    def test_check_access_with_policy(self, access_control_manager):
        """测试基于策略的访问控制"""
        manager = access_control_manager

        # 创建访问策略
        policy_id = manager.create_access_policy(
            name="Test Policy",
            resource_pattern="admin/*",
            permissions=["admin:manage"],
            roles=["admin"],
            conditions={"time_range": "09:00-17:00"}
        )

        assert policy_id is not None

        # 创建管理员用户
        admin_user_id = manager.create_user("adminuser", "admin@example.com", roles=["admin"])

        # 检查访问权限
        decision = manager.check_access(admin_user_id, "admin", "manage")
        assert decision.value == "allow"


class TestPolicyManagement:
    """测试策略管理功能"""

    def test_create_access_policy(self, access_control_manager):
        """测试创建访问策略"""
        manager = access_control_manager

        policy_id = manager.create_access_policy(
            name="Office Hours Policy",
            resource_pattern="system/*",
            permissions=["system:config"],
            roles=["admin"],
            conditions={"time_range": "09:00-17:00", "ip_whitelist": ["192.168.1.0/24"]}
        )

        assert policy_id is not None

        # 验证策略存在
        policy = manager.policy_manager.get_policy(policy_id)
        assert policy is not None
        assert policy.name == "Office Hours Policy"
        assert policy.resource_pattern == "system/*"

    def test_policy_evaluation(self, access_control_manager):
        """测试策略评估"""
        manager = access_control_manager

        # 创建时间限制策略
        policy_id = manager.create_access_policy(
            name="Time Restricted Policy",
            resource_pattern="sensitive/*",
            permissions=["sensitive:access"],
            roles=["analyst"],
            conditions={"time_range": "09:00-17:00"}
        )

        # 创建用户
        user_id = manager.create_user("timeuser", "time@example.com", roles=["analyst"])

        # 检查在允许时间内的访问
        with patch('src.infrastructure.security.access.access_control.datetime') as mock_datetime:
            # 模拟上午10点
            mock_datetime.now.return_value = datetime(2025, 1, 1, 10, 0, 0)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw) if args else datetime.now()

            decision = manager.check_access(user_id, "sensitive", "access")
            assert decision.value == "allow"


class TestAuditAndLogging:
    """测试审计和日志功能"""

    def test_audit_log_generation(self, access_control_manager):
        """测试审计日志生成"""
        manager = access_control_manager

        # 创建用户并执行操作
        user_id = manager.create_user("audituser", "audit@example.com", "password")
        manager.check_access(user_id, "resource", "read")

        # 获取审计日志
        logs = manager.get_audit_logs(user_id=user_id, limit=10)

        assert isinstance(logs, list)
        assert len(logs) > 0

        # 验证日志内容
        latest_log = logs[0]
        assert "timestamp" in latest_log
        assert "user_id" in latest_log
        assert "action" in latest_log

    def test_audit_log_filtering(self, access_control_manager):
        """测试审计日志过滤"""
        manager = access_control_manager

        # 创建多个用户并执行操作
        user1_id = manager.create_user("audituser1", "audit1@example.com", "password")
        user2_id = manager.create_user("audituser2", "audit2@example.com", "password")

        manager.check_access(user1_id, "resource1", "read")
        manager.check_access(user2_id, "resource2", "write")

        # 获取特定用户的日志
        user1_logs = manager.get_audit_logs(user_id=user1_id)
        user2_logs = manager.get_audit_logs(user_id=user2_id)

        assert len(user1_logs) >= 1
        assert len(user2_logs) >= 1

        # 验证日志隔离
        for log in user1_logs:
            assert log["user_id"] == user1_id
        for log in user2_logs:
            assert log["user_id"] == user2_id


class TestCaching:
    """测试缓存功能"""

    def test_cache_enabled(self, access_control_manager):
        """测试缓存启用"""
        manager = access_control_manager

        # 执行相同的访问检查多次
        user_id = manager.create_user("cacheuser", "cache@example.com", roles=["trader"])

        # 第一次检查
        decision1 = manager.check_access(user_id, "trade", "execute")

        # 第二次检查（应该从缓存获取）
        decision2 = manager.check_access(user_id, "trade", "execute")

        assert decision1.value == decision2.value

    def test_clear_cache(self, access_control_manager):
        """测试清除缓存"""
        manager = access_control_manager

        # 执行一些操作填充缓存
        user_id = manager.create_user("clearuser", "clear@example.com", roles=["trader"])
        manager.check_access(user_id, "trade", "execute")

        # 清除缓存
        manager.clear_cache()

        # 验证缓存已清除
        cache_stats = manager.get_cache_stats()
        assert cache_stats["entries"] == 0

    def test_cache_statistics(self, access_control_manager):
        """测试缓存统计"""
        manager = access_control_manager

        # 执行一些操作
        user_id = manager.create_user("statsuser", "stats@example.com", roles=["trader"])
        manager.check_access(user_id, "trade", "execute")

        # 获取缓存统计
        stats = manager.get_cache_stats()

        assert isinstance(stats, dict)
        assert "entries" in stats
        assert "hits" in stats
        assert "misses" in stats


class TestStatisticsAndReporting:
    """测试统计和报告功能"""

    def test_access_statistics(self, access_control_manager):
        """测试访问统计"""
        manager = access_control_manager

        # 执行各种操作
        user_id = manager.create_user("statsuser", "stats@example.com", roles=["trader"])
        manager.check_access(user_id, "trade", "execute")
        manager.check_access(user_id, "data", "read")
        manager.check_access(user_id, "admin", "manage")  # 应该被拒绝

        # 获取统计信息
        stats = manager.get_access_statistics()

        assert isinstance(stats, dict)
        assert "total_requests" in stats
        assert "allowed_requests" in stats
        assert "denied_requests" in stats
        assert stats["total_requests"] >= 3


class TestLifecycleManagement:
    """测试生命周期管理"""

    def test_shutdown(self, access_control_manager):
        """测试关闭管理器"""
        manager = access_control_manager

        # 执行一些操作
        user_id = manager.create_user("shutdownuser", "shutdown@example.com", "password")
        manager.check_access(user_id, "resource", "read")

        # 关闭管理器
        manager.shutdown()

        # 验证组件已被正确关闭
        # 这里可能需要根据实际实现验证清理逻辑

    def test_context_manager(self, temp_config_dir):
        """测试上下文管理器"""
        with AccessControlManager(config_path=temp_config_dir) as manager:
            # 在上下文中使用管理器
            user_id = manager.create_user("contextuser", "context@example.com", "password")
            assert user_id is not None

        # 上下文退出后，管理器应该被正确关闭


class TestErrorHandling:
    """测试错误处理"""

    def test_invalid_user_operations(self, access_control_manager):
        """测试无效用户操作"""
        manager = access_control_manager

        # 尝试为不存在的用户分配角色
        success = manager.assign_role_to_user("nonexistent_user", "admin")
        assert not success

        # 尝试撤销不存在的角色
        success = manager.revoke_role_from_user("nonexistent_user", "admin")
        assert not success

    def test_invalid_role_operations(self, access_control_manager):
        """测试无效角色操作"""
        manager = access_control_manager

        # 尝试创建重复角色
        role1_id = manager.create_role("Test Role", permissions=["test:perm"])
        role2_id = manager.create_role("Test Role", permissions=["test:perm"])

        # 根据实现，这可能返回None或不同的ID

    def test_access_check_with_invalid_user(self, access_control_manager):
        """测试使用无效用户检查访问"""
        manager = access_control_manager

        # 使用不存在的用户检查访问
        decision = manager.check_access("nonexistent_user", "resource", "permission")
        assert decision.value == "deny"


class TestPerformance:
    """测试性能"""

    def test_bulk_user_creation(self, access_control_manager):
        """测试批量用户创建"""
        manager = access_control_manager

        start_time = time.time()
        num_users = 50

        # 批量创建用户
        user_ids = []
        for i in range(num_users):
            user_id = manager.create_user(
                f"bulkuser{i}",
                f"bulk{i}@example.com",
                roles=["trader"] if i % 2 == 0 else ["analyst"]
            )
            user_ids.append(user_id)

        end_time = time.time()
        creation_time = end_time - start_time

        assert len(user_ids) == num_users
        # 批量创建应该在合理时间内完成
        assert creation_time < 10.0  # 10秒内完成

    def test_bulk_access_checks(self, access_control_manager):
        """测试批量访问检查"""
        manager = access_control_manager

        # 创建多个用户
        user_ids = []
        for i in range(20):
            user_id = manager.create_user(
                f"perfuser{i}",
                f"perf{i}@example.com",
                roles=["trader"]
            )
            user_ids.append(user_id)

        start_time = time.time()
        num_checks = 100

        # 批量执行访问检查
        for _ in range(num_checks):
            user_id = user_ids[_ % len(user_ids)]
            manager.check_access(user_id, "trade", "execute")

        end_time = time.time()
        check_time = end_time - start_time

        # 批量检查应该在合理时间内完成
        assert check_time < 5.0  # 5秒内完成


class TestConcurrency:
    """测试并发性"""

    @pytest.mark.asyncio
    async def test_concurrent_access_checks(self, access_control_manager):
        """测试并发访问检查"""
        manager = access_control_manager

        # 创建用户
        user_id = manager.create_user("concurrentuser", "concurrent@example.com", roles=["trader"])

        # 并发执行访问检查
        tasks = []
        num_tasks = 10

        for _ in range(num_tasks):
            task = asyncio.create_task(
                manager.check_access_async(user_id, "trade", "execute")
            )
            tasks.append(task)

        # 等待所有任务完成
        results = await asyncio.gather(*tasks)

        assert len(results) == num_tasks
        # 所有结果应该是一致的
        first_decision = results[0].decision
        for result in results:
            assert result.decision == first_decision

    def test_thread_safe_operations(self, access_control_manager):
        """测试线程安全操作"""
        import threading

        manager = access_control_manager
        results = []
        errors = []

        def worker(worker_id):
            try:
                # 创建用户
                user_id = manager.create_user(
                    f"threaduser{worker_id}",
                    f"thread{worker_id}@example.com",
                    "password123"
                )
                results.append(user_id)

                # 执行访问检查
                decision = manager.check_access(user_id, "trade", "execute")
                results.append(decision.decision)

            except Exception as e:
                errors.append(str(e))

        # 启动多个线程
        threads = []
        num_threads = 5

        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # 等待所有线程完成
        for t in threads:
            t.join()

        # 验证结果
        assert len(results) >= num_threads * 2  # 每个线程创建用户和检查访问
        assert len(errors) == 0  # 不应该有错误
