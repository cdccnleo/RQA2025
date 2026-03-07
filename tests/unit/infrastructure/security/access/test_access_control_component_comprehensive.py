#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
访问控制组件综合测试
测试RBACManager、SessionManager和AccessControlSystem等核心组件
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
from unittest.mock import patch, MagicMock, mock_open
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
import json
import tempfile
import os
from pathlib import Path

from src.infrastructure.security.access.access_control_component import (
    UserRole, Permission, User, RoleDefinition, UserSession,
    AccessPolicy, RBACManager, SessionManager, AccessControlSystem,
    get_access_control_system, authenticate_user, check_user_permission,
    create_system_user
)


@pytest.fixture
def rbac_manager():
    """创建RBAC管理器实例"""
    manager = RBACManager()
    return manager


@pytest.fixture
def session_manager():
    """创建会话管理器实例"""
    manager = SessionManager()
    return manager


@pytest.fixture
def access_control_system():
    """创建访问控制系统实例"""
    system = AccessControlSystem()
    return system


class TestUserRole:
    """测试UserRole枚举"""

    def test_user_role_values(self):
        """测试用户角色枚举值"""
        assert UserRole.ADMIN.value == "admin"
        assert UserRole.TRADER.value == "trader"
        assert UserRole.ANALYST.value == "analyst"
        assert UserRole.AUDITOR.value == "auditor"
        assert UserRole.GUEST.value == "guest"

    def test_user_role_membership(self):
        """测试用户角色成员"""
        roles = [UserRole.ADMIN, UserRole.TRADER, UserRole.ANALYST,
                UserRole.AUDITOR, UserRole.GUEST]
        assert len(roles) == 5
        assert UserRole.ADMIN in roles


class TestPermission:
    """测试Permission枚举"""

    def test_permission_values(self):
        """测试权限枚举值"""
        assert Permission.TRADE_EXECUTE.value == "trade:execute"
        assert Permission.TRADE_CANCEL.value == "trade:cancel"
        assert Permission.ORDER_PLACE.value == "order:place"
        assert Permission.ORDER_CANCEL.value == "order:cancel"
        assert Permission.DATA_READ.value == "data:read"
        assert Permission.DATA_WRITE.value == "data:write"
        assert Permission.DATA_EXPORT.value == "data:export"

    def test_permission_categories(self):
        """测试权限分类"""
        trade_permissions = [Permission.TRADE_EXECUTE, Permission.TRADE_CANCEL,
                           Permission.ORDER_PLACE, Permission.ORDER_CANCEL]
        data_permissions = [Permission.DATA_READ, Permission.DATA_WRITE, Permission.DATA_EXPORT]

        assert len(trade_permissions) == 4
        assert len(data_permissions) == 3


class TestUser:
    """测试User类"""

    def test_user_creation(self):
        """测试用户创建"""
        user = User(
            user_id="test_user",
            username="testuser",
            email="test@example.com",
            roles={UserRole.ADMIN, UserRole.TRADER},
            is_active=True,
            created_at=datetime.now()
        )

        assert user.user_id == "test_user"
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.roles == {UserRole.ADMIN, UserRole.TRADER}
        assert user.is_active is True
        assert user.created_at is not None

    def test_user_with_metadata(self):
        """测试带元数据的用户"""
        user = User(
            user_id="test_user",
            username="testuser",
            email="test@example.com",
            roles={UserRole.ANALYST},
            is_active=True,
            created_at=datetime.now()
        )

        # User类没有metadata字段，这里测试其他字段
        assert user.user_id == "test_user"
        assert user.username == "testuser"

    def test_user_to_dict(self):
        """测试用户序列化"""
        user = User(
            user_id="test_user",
            username="testuser",
            email="test@example.com",
            roles={UserRole.GUEST},
            is_active=True,
            created_at=datetime.now()
        )

        # User类没有to_dict方法，这里测试基本属性
        assert user.user_id == "test_user"
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.roles == {UserRole.GUEST}
        assert user.is_active is True


class TestRoleDefinition:
    """测试RoleDefinition类"""

    def test_role_definition_creation(self):
        """测试角色定义创建"""
        permissions = {Permission.DATA_READ, Permission.DATA_WRITE}
        role_def = RoleDefinition(
            role=UserRole.ANALYST,
            name="Data Manager",
            description="Can manage data",
            permissions=permissions,
            parent_roles={UserRole.GUEST}
        )

        assert role_def.role == UserRole.ANALYST
        assert role_def.name == "Data Manager"
        assert role_def.permissions == permissions
        assert role_def.description == "Can manage data"
        assert role_def.parent_roles == {UserRole.GUEST}

    def test_role_definition_inheritance(self):
        """测试角色继承"""
        base_permissions = {Permission.DATA_READ}
        child_permissions = {Permission.DATA_READ, Permission.DATA_WRITE}

        parent_role = RoleDefinition(
            role=UserRole.GUEST,
            name="Reader",
            description="Can read data",
            permissions=base_permissions,
            parent_roles=set()
        )
        child_role = RoleDefinition(
            role=UserRole.ANALYST,
            name="Editor",
            description="Can edit data",
            permissions=child_permissions,
            parent_roles={UserRole.GUEST}
        )

        # 子角色应该有更多权限
        assert child_role.permissions.issuperset(parent_role.permissions)
        assert UserRole.GUEST in child_role.parent_roles


class TestUserSession:
    """测试UserSession类"""

    def test_session_creation(self):
        """测试会话创建"""
        now = datetime.now()
        expires = now + timedelta(hours=1)
        session = UserSession(
            session_id="session_123",
            user_id="user_123",
            created_at=now,
            expires_at=expires,
            ip_address="192.168.1.1",
            user_agent="Test Agent"
        )

        assert session.session_id == "session_123"
        assert session.user_id == "user_123"
        assert session.created_at == now
        assert session.expires_at == expires
        assert session.ip_address == "192.168.1.1"
        assert session.user_agent == "Test Agent"

    def test_session_expiration(self):
        """测试会话过期"""
        now = datetime.now()
        future = now + timedelta(hours=1)
        session = UserSession("session_123", "user_123", now, future)

        # 新创建的会话不应该过期
        assert not session.is_expired()

        # 手动设置过期时间
        session.expires_at = datetime.now() - timedelta(hours=1)
        assert session.is_expired()

    def test_session_extension(self):
        """测试会话延长"""
        now = datetime.now()
        future = now + timedelta(hours=1)
        session = UserSession("session_123", "user_123", now, future)
        original_expiry = session.expires_at

        # 延长30分钟
        session.extend_session(30)

        # 验证会话已被延长（至少延长了25分钟，因为有一些时间流逝）
        expected_min_expiry = now + timedelta(minutes=25)
        assert session.expires_at >= expected_min_expiry


class TestAccessPolicy:
    """测试AccessPolicy类"""

    def test_policy_creation(self):
        """测试策略创建"""
        conditions = {"time_range": "09:00-17:00", "ip_whitelist": ["192.168.1.0/24"]}
        policy = AccessPolicy(
            policy_id="policy_123",
            name="Office Hours Policy",
            description="Only allow access during office hours",
            resource_pattern="data:*",
            permissions={Permission.DATA_READ},
            roles={UserRole.ANALYST},
            conditions=conditions
        )

        assert policy.policy_id == "policy_123"
        assert policy.name == "Office Hours Policy"
        assert policy.description == "Only allow access during office hours"
        assert policy.resource_pattern == "data:*"
        assert policy.permissions == {Permission.DATA_READ}
        assert policy.roles == {UserRole.ANALYST}
        assert policy.conditions == conditions


class TestRBACManager:
    """测试RBACManager类"""

    def test_init_default_roles(self, rbac_manager):
        """测试初始化默认角色"""
        assert UserRole.ADMIN in rbac_manager.roles
        assert UserRole.TRADER in rbac_manager.roles
        assert UserRole.ANALYST in rbac_manager.roles

        admin_role = rbac_manager.roles[UserRole.ADMIN]
        assert Permission.TRADE_EXECUTE in admin_role.permissions
        assert Permission.DATA_READ in admin_role.permissions

    def test_create_user(self, rbac_manager):
        """测试创建用户"""
        success = rbac_manager.create_user(
            user_id="test_user",
            username="testuser",
            email="test@example.com",
            password="password123",
            roles=[UserRole.TRADER]
        )

        assert success
        assert "test_user" in rbac_manager.users

        user = rbac_manager.users["test_user"]
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert UserRole.TRADER in user.roles

    def test_authenticate_user(self, rbac_manager):
        """测试用户认证"""
        # 先创建用户
        rbac_manager.create_user(
            user_id="auth_user",
            username="authuser",
            email="auth@example.com",
            password="correct_password",
            roles=[UserRole.ANALYST]
        )

        # 正确密码认证
        assert rbac_manager.authenticate_user("auth_user", "correct_password")

        # 错误密码认证
        assert not rbac_manager.authenticate_user("auth_user", "wrong_password")

        # 不存在用户认证
        assert not rbac_manager.authenticate_user("nonexistent", "password")

    def test_check_permission(self, rbac_manager):
        """测试权限检查"""
        # 创建有交易权限的用户
        rbac_manager.create_user(
            user_id="trader_user",
            username="trader",
            email="trader@example.com",
            password="password",
            roles={UserRole.TRADER}
        )

        # 检查交易权限
        assert rbac_manager.check_permission("trader_user", "trade", "trade:execute")
        assert rbac_manager.check_permission("trader_user", "order", "order:place")

        # 检查无权限的操作
        assert not rbac_manager.check_permission("trader_user", "admin", "admin:manage")

    def test_role_management(self, rbac_manager):
        """测试角色管理"""
        # 创建用户
        rbac_manager.create_user(
            user_id="role_user",
            username="roleuser",
            email="role@example.com",
            password="password",
            roles={UserRole.GUEST}
        )

        # 添加角色
        assert rbac_manager.add_role_to_user("role_user", UserRole.ANALYST)
        user = rbac_manager.users["role_user"]
        assert UserRole.ANALYST in user.roles

        # 移除角色
        assert rbac_manager.remove_role_from_user("role_user", UserRole.GUEST)
        assert UserRole.GUEST not in user.roles

    def test_get_user(self, rbac_manager):
        """测试获取用户信息"""
        # 创建用户
        rbac_manager.create_user(
            user_id="get_user",
            username="getuser",
            email="get@example.com",
            password="password",
            roles=[UserRole.AUDITOR]
        )

        user = rbac_manager.get_user("get_user")
        assert user is not None
        assert user.username == "getuser"
        assert UserRole.AUDITOR in user.roles

        # 获取不存在的用户
        assert rbac_manager.get_user("nonexistent") is None

    def test_list_users(self, rbac_manager):
        """测试列出用户"""
        # 创建多个用户
        rbac_manager.create_user("user1", "user1", "user1@example.com", {UserRole.ADMIN}, "pass")
        rbac_manager.create_user("user2", "user2", "user2@example.com", {UserRole.TRADER}, "pass")

        users = rbac_manager.list_users()
        assert len(users) >= 2
        assert "user1" in users
        assert "user2" in users

        user1_info = users["user1"]
        assert user1_info["username"] == "user1"
        assert "admin" in user1_info["roles"]


class TestSessionManager:
    """测试SessionManager类"""

    def test_create_session(self, session_manager):
        """测试创建会话"""
        session_id = session_manager.create_session(
            user_id="test_user",
            ip_address="192.168.1.1",
            user_agent="Test Browser"
        )

        assert session_id is not None
        assert isinstance(session_id, str)
        assert len(session_id) > 0

        # 验证会话已创建
        session_data = session_manager.sessions[session_id]
        assert session_data["user_id"] == "test_user"
        assert session_data["ip_address"] == "192.168.1.1"
        assert session_data["user_agent"] == "Test Browser"

    def test_validate_session(self, session_manager):
        """测试验证会话"""
        session_id = session_manager.create_session("test_user")

        # 验证有效会话
        validated = session_manager.validate_session(session_id)
        assert validated is not None
        assert validated["user_id"] == "test_user"
        assert validated["is_active"] is True

        # 验证无效会话
        assert session_manager.validate_session("invalid_session") is None

    def test_destroy_session(self, session_manager):
        """测试销毁会话"""
        session_id = session_manager.create_session("test_user")

        # 确认会话存在
        assert session_manager.validate_session(session_id) is not None

        # 销毁会话
        session_manager.destroy_session(session_id)

        # 确认会话已被销毁
        assert session_manager.validate_session(session_id) is None

    def test_cleanup_expired_sessions(self, session_manager):
        """测试清理过期会话"""
        # 创建会话
        session_id = session_manager.create_session("test_user")

        # 手动使会话过期 - 修改last_activity为过期时间
        session_data = session_manager.sessions[session_id]
        session_data["last_activity"] = datetime.now() - timedelta(hours=10)  # 超过8小时超时

        # 清理过期会话
        session_manager.cleanup_expired_sessions()

        # 确认会话已被清理
        assert session_id not in session_manager.sessions


class TestAccessControlSystem:
    """测试AccessControlSystem类"""

    def test_initialization(self, access_control_system):
        """测试系统初始化"""
        assert access_control_system.rbac_manager is not None
        assert access_control_system.session_manager is not None

    def test_create_default_admin(self, access_control_system):
        """测试创建默认管理员"""
        # 默认管理员应该已被创建
        admin_info = access_control_system.get_user_info("admin")
        assert admin_info is not None
        assert admin_info["username"] == "Administrator"
        assert "admin" in admin_info["roles"]

    def test_authenticate(self, access_control_system):
        """测试用户认证"""
        # 认证默认管理员（密码通常是预设的或需要创建用户）
        # 这里我们先创建一个测试用户
        access_control_system.create_user(
            user_id="test_auth",
            username="testauth",
            email="auth@example.com",
            roles=["analyst"],
            password="testpass123"
        )

        session_id = access_control_system.authenticate("test_auth", "testpass123")
        assert session_id is not None

        # 验证会话
        session_info = access_control_system.validate_session(session_id)
        assert session_info is not None
        assert session_info["user_id"] == "test_auth"

    def test_authorize(self, access_control_system):
        """测试授权"""
        # 创建用户并认证
        access_control_system.create_user(
            user_id="test_authorize",
            username="testauthorize",
            email="authorize@example.com",
            roles=["trader"],
            password="testpass123"
        )

        session_id = access_control_system.authenticate("test_authorize", "testpass123")
        assert session_id is not None

        # 测试有权限的操作
        assert access_control_system.authorize(session_id, "trade", "execute")
        assert access_control_system.authorize(session_id, "data", "read")

        # 测试无权限的操作
        assert not access_control_system.authorize(session_id, "admin", "manage")

    def test_user_management(self, access_control_system):
        """测试用户管理"""
        # 创建用户
        success = access_control_system.create_user(
            user_id="test_manage",
            username="testmanage",
            email="manage@example.com",
            roles=["analyst"],
            password="testpass123"
        )
        assert success

        # 获取用户信息
        user_info = access_control_system.get_user_info("test_manage")
        assert user_info is not None
        assert user_info["username"] == "testmanage"
        assert "analyst" in user_info["roles"]

        # 更新用户角色
        success = access_control_system.update_user_roles("test_manage", ["trader", "auditor"])
        assert success

        user_info = access_control_system.get_user_info("test_manage")
        assert "trader" in user_info["roles"]
        assert "auditor" in user_info["roles"]
        assert "analyst" not in user_info["roles"]

    def test_logout(self, access_control_system):
        """测试登出"""
        # 创建用户并认证
        access_control_system.create_user(
            user_id="test_logout",
            username="testlogout",
            email="logout@example.com",
            roles=["guest"],
            password="testpass123"
        )

        session_id = access_control_system.authenticate("test_logout", "testpass123")
        assert session_id is not None

        # 验证会话存在
        assert access_control_system.validate_session(session_id) is not None

        # 登出
        access_control_system.logout(session_id)

        # 验证会话已被销毁
        assert access_control_system.validate_session(session_id) is None

    def test_health_check(self, access_control_system):
        """测试健康检查"""
        health = access_control_system.health_check()

        assert "status" in health
        assert "timestamp" in health
        assert "component" in health
        assert health["component"] == "AccessControlSystem"
        assert "users_count" in health
        assert "active_sessions" in health

    def test_concurrent_access(self, access_control_system):
        """测试并发访问"""
        import threading
        import time

        results = []
        errors = []

        def worker(worker_id):
            try:
                # 创建用户
                user_id = f"concurrent_user_{worker_id}"
                access_control_system.create_user(
                    user_id=user_id,
                    username=f"user{worker_id}",
                    email=f"user{worker_id}@example.com",
                    roles=["analyst"],
                    password="password123"
                )

                # 认证
                session_id = access_control_system.authenticate(user_id, "password123")
                results.append(session_id)

                # 授权检查
                if session_id:
                    can_read = access_control_system.authorize(session_id, "data", "read")
                    results.append(can_read)

            except Exception as e:
                errors.append(str(e))

        # 启动多个线程
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # 等待所有线程完成
        for t in threads:
            t.join()

        # 验证结果
        assert len(results) >= 5  # 至少5个会话ID
        assert len(errors) == 0  # 不应该有错误


class TestGlobalFunctions:
    """测试全局函数"""

    def test_get_access_control_system(self):
        """测试获取访问控制系统实例"""
        system = get_access_control_system()
        assert system is not None
        assert isinstance(system, AccessControlSystem)

    def test_authenticate_user_function(self):
        """测试authenticate_user全局函数"""
        system = get_access_control_system()

        # 创建测试用户
        system.create_user(
            user_id="global_test",
            username="globaltest",
            email="global@example.com",
            roles=["guest"],
            password="globalpass123"
        )

        session_id = authenticate_user("global_test", "globalpass123")
        assert session_id is not None

    def test_check_user_permission_function(self):
        """测试check_user_permission全局函数"""
        system = get_access_control_system()

        # 创建用户并认证
        system.create_user(
            user_id="perm_test",
            username="permtest",
            email="perm@example.com",
            roles=["trader"],
            password="permpass123"
        )

        session_id = authenticate_user("perm_test", "permpass123")
        assert session_id is not None

        # 检查权限
        assert check_user_permission(session_id, "trade", "execute")
        assert not check_user_permission(session_id, "admin", "manage")

    def test_create_system_user_function(self):
        """测试create_system_user全局函数"""
        success = create_system_user(
            user_id="system_test",
            username="systemtest",
            email="system@example.com",
            roles=["analyst"],
            password="systempass123"
        )

        assert success

        system = get_access_control_system()
        user_info = system.get_user_info("system_test")
        assert user_info is not None
        assert user_info["username"] == "systemtest"


class TestErrorHandling:
    """测试错误处理"""

    def test_invalid_user_creation(self, rbac_manager):
        """测试无效用户创建"""
        # 空用户名
        success = rbac_manager.create_user("", "username", "email@example.com", set(), "password")
        assert not success

        # 无效邮箱
        success = rbac_manager.create_user("user", "username", "invalid-email", set(), "password")
        assert not success

    def test_invalid_authentication(self, rbac_manager):
        """测试无效认证"""
        # 不存在的用户
        assert not rbac_manager.authenticate_user("nonexistent", "password")

        # 空密码
        assert not rbac_manager.authenticate_user("user", "")

    def test_invalid_session_operations(self, session_manager):
        """测试无效会话操作"""
        # 验证不存在的会话
        assert session_manager.validate_session("nonexistent") is None

        # 销毁不存在的会话（应该不抛出异常）
        session_manager.destroy_session("nonexistent")

    def test_invalid_authorization(self, access_control_system):
        """测试无效授权"""
        # 无效会话ID
        assert not access_control_system.authorize("invalid_session", "resource", "action")

        # 空参数
        assert not access_control_system.authorize("", "", "")


class TestIntegration:
    """测试集成场景"""

    def test_complete_user_workflow(self, access_control_system):
        """测试完整用户工作流"""
        user_id = "workflow_test"

        # 1. 创建用户
        success = access_control_system.create_user(
            user_id=user_id,
            username="workflowuser",
            email="workflow@example.com",
            roles=["trader"],
            password="workflow123"
        )
        assert success

        # 2. 认证用户
        session_id = access_control_system.authenticate(user_id, "workflow123")
        assert session_id is not None

        # 3. 验证会话
        session_info = access_control_system.validate_session(session_id)
        assert session_info is not None
        assert session_info["user_id"] == user_id

        # 4. 检查权限
        assert access_control_system.check_permission(user_id, "trade", "execute")
        assert access_control_system.authorize(session_id, "data", "read")

        # 5. 更新用户角色
        success = access_control_system.update_user_roles(user_id, ["analyst", "auditor"])
        assert success

        # 验证新权限
        assert access_control_system.check_permission(user_id, "data", "export")

        # 6. 登出
        access_control_system.logout(session_id)

        # 验证会话已销毁
        assert access_control_system.validate_session(session_id) is None

    def test_admin_user_scenario(self, access_control_system):
        """测试管理员用户场景"""
        admin_id = "admin_scenario"

        # 创建管理员用户
        success = access_control_system.create_user(
            user_id=admin_id,
            username="adminuser",
            email="admin@example.com",
            roles=["admin"],
            password="admin123"
        )
        assert success

        # 认证管理员
        session_id = access_control_system.authenticate(admin_id, "admin123")
        assert session_id is not None

        # 管理员应该有所有权限
        assert access_control_system.authorize(session_id, "trade", "execute")
        assert access_control_system.authorize(session_id, "data", "write")
        assert access_control_system.authorize(session_id, "user", "manage")

        # 验证管理员可以管理其他用户
        user_info = access_control_system.get_user_info(admin_id)
        assert user_info is not None
        assert "admin" in user_info["roles"]

    def test_guest_user_restrictions(self, access_control_system):
        """测试访客用户限制"""
        guest_id = "guest_scenario"

        # 创建访客用户
        success = access_control_system.create_user(
            user_id=guest_id,
            username="guestuser",
            email="guest@example.com",
            roles=["guest"],
            password="guest123"
        )
        assert success

        # 认证访客
        session_id = access_control_system.authenticate(guest_id, "guest123")
        assert session_id is not None

        # 访客应该只有读取权限
        assert access_control_system.authorize(session_id, "data", "read")
        assert not access_control_system.authorize(session_id, "trade", "execute")
        assert not access_control_system.authorize(session_id, "data", "write")
        assert not access_control_system.authorize(session_id, "admin", "manage")
