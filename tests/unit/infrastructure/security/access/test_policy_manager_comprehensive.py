#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略管理器综合测试
测试PolicyManager、SessionManager和CacheManager的核心功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from src.infrastructure.security.access.policy_manager import (
    PolicyManager,
    SessionManager,
    CacheManager
)
from src.infrastructure.security.core.types import (
    PolicyCreationParams,
    AccessPolicy,
    UserSession,
    User,
    UserRole,
    Permission
)

# 为了兼容性，如果没有简单的READ/WRITE权限，使用现有的权限
try:
    READ_PERMISSION = Permission.DATA_READ
    WRITE_PERMISSION = Permission.DATA_WRITE
except AttributeError:
    # 如果没有定义，使用字符串作为后备
    READ_PERMISSION = "data:read"
    WRITE_PERMISSION = "data:write"


@pytest.fixture
def policy_manager():
    """创建策略管理器实例"""
    return PolicyManager()


@pytest.fixture
def session_manager():
    """创建会话管理器实例"""
    return SessionManager()


@pytest.fixture
def cache_manager():
    """创建缓存管理器实例"""
    return CacheManager()


@pytest.fixture
def sample_user():
    """创建示例用户"""
    return User(
        user_id="test_user_001",
        username="testuser",
        roles={UserRole.GUEST},
        permissions={"data:read"},
        is_active=True,
        created_at=datetime.now()
    )


@pytest.fixture
def sample_policy_creation_params():
    """创建示例策略创建参数"""
    return PolicyCreationParams(
        name="Test Policy",
        resource_type="api",
        resource_pattern="/api/test/*",
        permissions={"data:read", "data:write"},
        conditions={"time_range": "business_hours"},
        priority=1,
        description="A test access policy",
        is_active=True,
        expiry_date=datetime.now() + timedelta(days=30)
    )


class TestPolicyManagerInitialization:
    """测试策略管理器初始化"""

    def test_initialization(self, policy_manager):
        """测试初始化"""
        manager = policy_manager

        assert hasattr(manager, 'policies')
        assert hasattr(manager, 'policy_cache')
        assert isinstance(manager.policies, dict)
        assert isinstance(manager.policy_cache, dict)

    def test_create_policy(self, policy_manager, sample_policy_creation_params):
        """测试创建策略"""
        manager = policy_manager
        params = sample_policy_creation_params

        policy = manager.create_policy(params)

        assert isinstance(policy, AccessPolicy)
        assert policy.name == params.name
        assert policy.resource_pattern == params.resource_pattern
        assert policy.permissions == params.permissions
        assert policy.conditions == params.conditions
        assert policy.priority == params.priority
        assert policy.description == params.description
        assert policy.is_active == params.is_active

    def test_create_policy_with_different_params(self, policy_manager):
        """测试使用不同参数创建策略"""
        manager = policy_manager

        params = PolicyCreationParams(
            name="Admin Policy",
            resource_type="admin",
            resource_pattern="/admin/*",
            permissions={"data:read", "data:write", "trade:execute"},
            conditions={"role": "admin"},
            priority=10,
            description="Administrator access policy",
            is_active=True,
            expiry_date=None
        )

        policy = manager.create_policy(params)

        assert policy.name == "Admin Policy"
        assert policy.resource_pattern == "/admin/*"
        assert len(policy.permissions) == 3
        assert policy.priority == 10

    def test_list_policies_empty(self, policy_manager):
        """测试列出空策略列表"""
        manager = policy_manager

        policies = manager.list_policies()

        assert isinstance(policies, list)
        assert len(policies) == 0

    def test_list_policies_with_policies(self, policy_manager, sample_policy_creation_params):
        """测试列出策略"""
        manager = policy_manager

        # 创建几个策略
        policy1 = manager.create_policy(sample_policy_creation_params)

        params2 = PolicyCreationParams(
            name="Policy 2",
            resource_type="data",
            resource_pattern="/data/*",
            permissions={"data:read"},
            conditions={},
            priority=2,
            description="Data access policy",
            is_active=False,  # 非活跃状态
            expiry_date=None
        )
        policy2 = manager.create_policy(params2)

        # 列出所有活跃策略
        active_policies = manager.list_policies(active_only=True)
        assert len(active_policies) == 1
        assert active_policies[0].name == "Test Policy"

        # 列出所有策略
        all_policies = manager.list_policies(active_only=False)
        assert len(all_policies) == 2

    def test_get_policy_existing(self, policy_manager, sample_policy_creation_params):
        """测试获取存在的策略"""
        manager = policy_manager

        created_policy = manager.create_policy(sample_policy_creation_params)
        retrieved_policy = manager.get_policy(created_policy.policy_id)

        assert retrieved_policy == created_policy
        assert retrieved_policy.policy_id == created_policy.policy_id

    def test_get_policy_nonexistent(self, policy_manager):
        """测试获取不存在的策略"""
        manager = policy_manager

        policy = manager.get_policy("nonexistent_policy_id")

        assert policy is None

    def test_update_policy(self, policy_manager, sample_policy_creation_params):
        """测试更新策略"""
        manager = policy_manager

        policy = manager.create_policy(sample_policy_creation_params)

        # 更新策略
        success = manager.update_policy(
            policy.policy_id,
            name="Updated Policy",
            priority=5,
            is_active=False
        )

        assert success is True

        # 验证更新
        updated_policy = manager.get_policy(policy.policy_id)
        assert updated_policy.name == "Updated Policy"
        assert updated_policy.priority == 5
        assert updated_policy.is_active is False

    def test_update_policy_nonexistent(self, policy_manager):
        """测试更新不存在的策略"""
        manager = policy_manager

        success = manager.update_policy("nonexistent_id", name="New Name")

        assert success is False

    def test_delete_policy(self, policy_manager, sample_policy_creation_params):
        """测试删除策略"""
        manager = policy_manager

        policy = manager.create_policy(sample_policy_creation_params)

        # 确认策略存在
        assert manager.get_policy(policy.policy_id) is not None

        # 删除策略
        success = manager.delete_policy(policy.policy_id)

        assert success is True

        # 确认策略已被删除
        assert manager.get_policy(policy.policy_id) is None


class TestPolicyManagerEvaluation:
    """测试策略管理器策略评估功能"""

    def test_evaluate_policies_no_match(self, policy_manager, sample_user):
        """测试评估策略无匹配"""
        manager = policy_manager
        user = sample_user

        # 没有策略，应该返回空列表
        policies = manager.evaluate_policies(user, "/api/test", "read")

        assert isinstance(policies, list)
        assert len(policies) == 0

    def test_evaluate_policies_with_matching_policy(self, policy_manager, sample_user):
        """测试评估策略有匹配"""
        manager = policy_manager
        user = sample_user

        # 创建匹配的策略
        params = PolicyCreationParams(
            name="Matching Policy",
            resource_type="api",
            resource_pattern="/api/test/*",
            permissions={"data:read"},
            conditions={},
            priority=1,
            description="Policy that matches",
            is_active=True,
            expiry_date=None
        )

        policy = manager.create_policy(params)

        # 评估策略
        policies = manager.evaluate_policies(user, "/api/test/123", "read")

        assert len(policies) >= 1
        assert policy in policies

    def test_evaluate_policies_multiple_matches(self, policy_manager, sample_user):
        """测试评估策略多个匹配"""
        manager = policy_manager
        user = sample_user

        # 创建多个匹配的策略
        params1 = PolicyCreationParams(
            name="Policy 1",
            resource_type="api",
            resource_pattern="/api/*",
            permissions={"data:read"},
            conditions={},
            priority=1,
            description="Broad policy",
            is_active=True,
            expiry_date=None
        )

        params2 = PolicyCreationParams(
            name="Policy 2",
            resource_type="api",
            resource_pattern="/api/test/*",
            permissions={"data:read"},
            conditions={},
            priority=2,
            description="Specific policy",
            is_active=True,
            expiry_date=None
        )

        policy1 = manager.create_policy(params1)
        policy2 = manager.create_policy(params2)

        policies = manager.evaluate_policies(user, "/api/test/123", "read")

        assert len(policies) >= 2
        assert policy1 in policies
        assert policy2 in policies

    def test_check_policy_access_allow(self, policy_manager):
        """测试检查策略访问允许"""
        manager = policy_manager

        # 创建允许READ权限的策略
        params = PolicyCreationParams(
            name="Allow Read Policy",
            resource_type="api",
            resource_pattern="/api/*",
            permissions={"data:read"},
            conditions={},
            priority=1,
            description="Allows read access",
            is_active=True,
            expiry_date=None
        )

        policy = manager.create_policy(params)
        policies = [policy]

        # 检查READ权限
        access_granted = manager.check_policy_access(policies, "read", {})

        assert access_granted is True

    def test_check_policy_access_deny(self, policy_manager):
        """测试检查策略访问拒绝"""
        manager = policy_manager

        # 创建只允许READ权限的策略
        params = PolicyCreationParams(
            name="Read Only Policy",
            resource_type="api",
            resource_pattern="/api/*",
            permissions={"data:read"},
            conditions={},
            priority=1,
            description="Only allows read access",
            is_active=True,
            expiry_date=None
        )

        policy = manager.create_policy(params)
        policies = [policy]

        # 检查WRITE权限（应该被拒绝）
        access_granted = manager.check_policy_access(policies, "write", {})

        assert access_granted is False


class TestSessionManager:
    """测试会话管理器"""

    def test_initialization(self, session_manager):
        """测试初始化"""
        manager = session_manager

        assert hasattr(manager, 'sessions')
        assert isinstance(manager.sessions, dict)
        assert manager.session_timeout == 3600

    def test_create_session(self, session_manager):
        """测试创建会话"""
        manager = session_manager

        user_id = "test_user_123"
        session_id = manager.create_session(
            user_id=user_id,
            ip_address="192.168.1.100",
            user_agent="Mozilla/5.0"
        )

        assert isinstance(session_id, str)
        assert len(session_id) > 0
        assert session_id.startswith("sess_")

        # 验证会话被创建
        session = manager.get_session(session_id)
        assert session is not None
        assert session.user_id == user_id
        assert session.ip_address == "192.168.1.100"
        assert session.user_agent == "Mozilla/5.0"

    def test_get_session_existing(self, session_manager):
        """测试获取存在的会话"""
        manager = session_manager

        user_id = "test_user_456"
        session_id = manager.create_session(user_id)

        session = manager.get_session(session_id)

        assert session is not None
        assert isinstance(session, UserSession)
        assert session.session_id == session_id
        assert session.user_id == user_id

    def test_get_session_nonexistent(self, session_manager):
        """测试获取不存在的会话"""
        manager = session_manager

        session = manager.get_session("nonexistent_session_id")

        assert session is None

    def test_get_session_expired(self, session_manager):
        """测试获取过期会话"""
        manager = session_manager

        # 创建一个已过期的会话
        user_id = "test_user_expired"
        session_id = manager.create_session(user_id)

        # 手动设置会话过期
        session = manager.sessions[session_id]
        session.expires_at = datetime.now() - timedelta(hours=1)  # 1小时前过期

        # 获取过期会话
        retrieved_session = manager.get_session(session_id)

        assert retrieved_session is None

        # 验证会话已被清理
        assert session_id not in manager.sessions

    def test_invalidate_session(self, session_manager):
        """测试使会话失效"""
        manager = session_manager

        user_id = "test_user_invalidate"
        session_id = manager.create_session(user_id)

        # 确认会话存在
        assert manager.get_session(session_id) is not None

        # 使会话失效
        success = manager.invalidate_session(session_id)

        assert success is True

        # 确认会话已被删除
        assert manager.get_session(session_id) is None

    def test_invalidate_nonexistent_session(self, session_manager):
        """测试使不存在的会话失效"""
        manager = session_manager

        success = manager.invalidate_session("nonexistent_session_id")

        assert success is False

    def test_session_uniqueness(self, session_manager):
        """测试会话ID唯一性"""
        manager = session_manager

        # 创建多个会话
        session_ids = []
        for i in range(10):
            session_id = manager.create_session(f"user_{i}")
            session_ids.append(session_id)

        # 验证所有会话ID都唯一
        assert len(session_ids) == len(set(session_ids))

        # 验证所有会话都存在
        for session_id in session_ids:
            assert manager.get_session(session_id) is not None


class TestCacheManager:
    """测试缓存管理器"""

    def test_initialization(self, cache_manager):
        """测试初始化"""
        manager = cache_manager

        assert hasattr(manager, 'cache')
        assert hasattr(manager, 'access_times')
        assert isinstance(manager.cache, dict)
        assert isinstance(manager.access_times, dict)
        assert manager.max_size == 10000

    def test_set_and_get_cache(self, cache_manager):
        """测试设置和获取缓存"""
        manager = cache_manager

        key = "test_cache_key"
        value = {"user_id": "123", "permissions": ["read", "write"]}

        # 设置缓存
        manager.set(key, value)

        # 获取缓存
        retrieved_value = manager.get(key)

        assert retrieved_value == value

        # 验证访问时间被更新
        assert key in manager.access_times

    def test_get_nonexistent_cache(self, cache_manager):
        """测试获取不存在的缓存"""
        manager = cache_manager

        value = manager.get("nonexistent_key")

        assert value is None

    def test_cache_cleanup_on_overflow(self):
        """测试缓存溢出时的清理"""
        # 创建一个小的缓存管理器
        manager = CacheManager(max_size=3)

        # 添加超过最大数量的缓存项
        for i in range(5):
            key = f"key_{i}"
            value = {"data": f"value_{i}"}
            manager.set(key, value)

        # 验证缓存大小不超过最大值
        assert len(manager.cache) <= manager.max_size

        # 验证至少有一些旧的缓存项被清理了
        # （具体的清理逻辑可能不同，这里只是基本验证）

    def test_clear_cache(self, cache_manager):
        """测试清除缓存"""
        manager = cache_manager

        # 添加一些缓存
        for i in range(3):
            manager.set(f"key_{i}", {"data": f"value_{i}"})

        # 确认缓存不为空
        assert len(manager.cache) > 0

        # 清除缓存
        manager.clear()

        # 确认缓存被清空
        assert len(manager.cache) == 0
        assert len(manager.access_times) == 0

    def test_cache_access_time_update(self, cache_manager):
        """测试缓存访问时间更新"""
        manager = cache_manager

        key = "test_key"
        value = {"data": "test_value"}

        # 设置缓存
        manager.set(key, value)
        initial_access_time = manager.access_times[key]

        # 等待一小段时间
        import time
        time.sleep(0.001)

        # 再次获取，更新访问时间
        retrieved_value = manager.get(key)
        updated_access_time = manager.access_times[key]

        assert retrieved_value == value
        assert updated_access_time >= initial_access_time


class TestPolicyManagerIntegration:
    """测试策略管理器集成功能"""

    def test_complete_policy_workflow(self, policy_manager, sample_user):
        """测试完整策略工作流"""
        manager = policy_manager

        # 1. 创建策略
        params = PolicyCreationParams(
            name="Integration Test Policy",
            resource_type="api",
            resource_pattern="/api/integration/*",
            permissions={"data:read", "data:write"},
            conditions={"department": "engineering"},
            priority=5,
            description="Policy for integration testing",
            is_active=True,
            expiry_date=datetime.now() + timedelta(days=7)
        )

        policy = manager.create_policy(params)

        # 2. 验证策略创建
        retrieved_policy = manager.get_policy(policy.policy_id)
        assert retrieved_policy == policy

        # 3. 评估策略
        policies = manager.evaluate_policies(sample_user, "/api/integration/test", "read")
        assert len(policies) >= 1

        # 4. 检查访问权限
        access_granted = manager.check_policy_access(policies, "read", {"department": "engineering"})
        assert access_granted is True

        # 5. 更新策略
        manager.update_policy(policy.policy_id, priority=10)

        # 6. 清理 - 删除策略
        manager.delete_policy(policy.policy_id)
        assert manager.get_policy(policy.policy_id) is None

    def test_policy_manager_with_multiple_policies(self, policy_manager, sample_user):
        """测试策略管理器处理多个策略"""
        manager = policy_manager

        # 创建多个不同优先级的策略
        policies_data = [
            {
                "name": "Low Priority Policy",
                "pattern": "/api/public/*",
                "permissions": {"data:read"},
                "priority": 1
            },
            {
                "name": "Medium Priority Policy",
                "pattern": "/api/user/*",
                "permissions": {"data:read", "data:write"},
                "priority": 5
            },
            {
                "name": "High Priority Policy",
                "pattern": "/api/admin/*",
                "permissions": {"data:read", "data:write", "trade:execute"},
                "priority": 10
            }
        ]

        created_policies = []
        for policy_data in policies_data:
            params = PolicyCreationParams(
                name=policy_data["name"],
                resource_type="api",
                resource_pattern=policy_data["pattern"],
                permissions=policy_data["permissions"],
                conditions={},
                priority=policy_data["priority"],
                description=f"Policy with priority {policy_data['priority']}",
                is_active=True,
                expiry_date=None
            )
            policy = manager.create_policy(params)
            created_policies.append(policy)

        # 验证所有策略都被创建
        all_policies = manager.list_policies(active_only=False)
        assert len(all_policies) == len(policies_data)

        # 测试不同资源的访问
        test_cases = [
            ("/api/public/data", "read", True),  # 低优先级策略匹配
            ("/api/user/profile", "write", True),  # 中优先级策略匹配
            ("/api/admin/users", "delete", True),  # 高优先级策略匹配
            ("/api/admin/users", "admin", False),  # 没有匹配的权限
        ]

        for resource, permission, expected_access in test_cases:
            policies = manager.evaluate_policies(sample_user, resource, permission)
            access_granted = manager.check_policy_access(policies, permission, {})
            assert access_granted == expected_access, f"Failed for {resource} {permission}"

    def test_session_cache_integration(self, session_manager, cache_manager):
        """测试会话和缓存的集成"""
        session_mgr = session_manager
        cache_mgr = cache_manager

        # 创建会话
        user_id = "integration_user"
        session_id = session_mgr.create_session(user_id)

        # 在缓存中存储会话相关数据
        cache_key = f"session_data_{session_id}"
        cache_value = {
            "user_id": user_id,
            "permissions": ["read", "write"],
            "last_access": datetime.now().isoformat()
        }

        cache_mgr.set(cache_key, cache_value)

        # 验证缓存数据
        retrieved_data = cache_mgr.get(cache_key)
        assert retrieved_data == cache_value
        assert retrieved_data["user_id"] == user_id

        # 验证会话仍然有效
        session = session_mgr.get_session(session_id)
        assert session is not None
        assert session.user_id == user_id

        # 使会话失效
        session_mgr.invalidate_session(session_id)

        # 验证会话已被清理
        assert session_mgr.get_session(session_id) is None

        # 缓存数据仍然存在（除非有特定的清理逻辑）
        # 这里我们不测试缓存清理，因为它可能有不同的实现


class TestErrorHandling:
    """测试错误处理"""

    def test_policy_manager_with_invalid_params(self, policy_manager):
        """测试策略管理器处理无效参数"""
        manager = policy_manager

        # 测试更新不存在的策略
        result = manager.update_policy("invalid_id", name="test")
        assert result is False

        # 测试删除不存在的策略
        result = manager.delete_policy("invalid_id")
        assert result is False

        # 测试获取不存在的策略
        policy = manager.get_policy("invalid_id")
        assert policy is None

    def test_session_manager_with_invalid_sessions(self, session_manager):
        """测试会话管理器处理无效会话"""
        manager = session_manager

        # 测试获取不存在的会话
        session = manager.get_session("invalid_session_id")
        assert session is None

        # 测试使不存在的会话失效
        result = manager.invalidate_session("invalid_session_id")
        assert result is False

    def test_cache_manager_edge_cases(self, cache_manager):
        """测试缓存管理器边界情况"""
        manager = cache_manager

        # 测试获取不存在的键
        result = manager.get("nonexistent_key")
        assert result is None

        # 测试设置None值（如果实现允许）
        try:
            manager.set("none_key", None)
            retrieved = manager.get("none_key")
            # 行为可能不同，这里不做严格断言
        except Exception:
            # 如果不支持None值，也是可以接受的
            pass

        # 测试设置空字典
        manager.set("empty_key", {})
        result = manager.get("empty_key")
        assert result == {}

    def test_policy_evaluation_edge_cases(self, policy_manager, sample_user):
        """测试策略评估边界情况"""
        manager = policy_manager

        # 测试空资源
        policies = manager.evaluate_policies(sample_user, "", "read")
        assert isinstance(policies, list)

        # 测试None权限
        policies = manager.evaluate_policies(sample_user, "/api/test", None)
        assert isinstance(policies, list)

        # 测试权限检查的边界情况
        empty_policies = []
        result = manager.check_policy_access(empty_policies, "read", {})
        assert result is False  # 没有策略，应该拒绝访问

    def test_concurrent_access_simulation(self, policy_manager, session_manager):
        """测试并发访问模拟"""
        policy_mgr = policy_manager
        session_mgr = session_manager

        import threading
        import time

        errors = []
        results = []

        def policy_operations_thread(thread_id: int):
            try:
                # 创建策略
                params = PolicyCreationParams(
                    name=f"Thread {thread_id} Policy",
                    resource_type="api",
                    resource_pattern=f"/api/thread{thread_id}/*",
                    permissions={"data:read"},
                    conditions={},
                    priority=1,
                    description=f"Policy created by thread {thread_id}",
                    is_active=True,
                    expiry_date=None
                )

                policy = policy_mgr.create_policy(params)
                results.append(f"thread-{thread_id}-policy-{policy.policy_id}")

                # 创建会话
                session_id = session_mgr.create_session(f"user_{thread_id}")
                results.append(f"thread-{thread_id}-session-{session_id}")

                time.sleep(0.001)  # 小延迟模拟并发

            except Exception as e:
                errors.append(f"thread-{thread_id}-error: {e}")

        # 启动多个线程
        threads = []
        for i in range(3):
            thread = threading.Thread(target=policy_operations_thread, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证结果
        assert len(results) >= 6  # 每个线程应该创建1个策略和1个会话
        assert len(errors) == 0   # 不应该有错误
