#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 UserManager综合测试

测试用户管理器的所有功能，包括：
- 用户CRUD操作
- 角色管理
- 权限检查
- 权限继承
- 缓存机制
- 审计日志
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import uuid
from datetime import datetime
from unittest.mock import patch, MagicMock
from src.infrastructure.security.core.types import UserCreationParams, AccessCheckParams
from src.infrastructure.security.auth.user_manager import UserManager


@pytest.fixture
def user_manager():
    """创建UserManager实例"""
    return UserManager()


class TestUserManagerComprehensive:
    """UserManager综合测试"""

    def test_initialization(self, user_manager):
        """测试初始化"""
        assert user_manager.users == {}
        assert user_manager.roles == {}

    def test_create_user_basic(self, user_manager):
        """测试基本用户创建"""
        params = UserCreationParams(
            username="testuser",
            email="test@example.com",
            is_active=True,
            roles={"user"}
        )

        user = user_manager.create_user(params)

        assert user.user_id is not None
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.is_active == True
        assert user.roles == {"user"}
        assert user.created_at is not None
        assert user.user_id in user_manager.users

    def test_create_user_with_inactive(self, user_manager):
        """测试创建非活跃用户"""
        params = UserCreationParams(
            username="inactive_user",
            email="inactive@example.com",
            is_active=False,
            roles={"user"}
        )

        user = user_manager.create_user(params)

        assert user.is_active == False

    def test_get_user_existing(self, user_manager):
        """测试获取现有用户"""
        params = UserCreationParams(
            username="getuser",
            email="get@example.com",
            is_active=True,
            roles={"user"}
        )
        created_user = user_manager.create_user(params)

        retrieved_user = user_manager.get_user(created_user.user_id)

        assert retrieved_user is not None
        assert retrieved_user.user_id == created_user.user_id
        assert retrieved_user.username == created_user.username

    def test_get_user_nonexistent(self, user_manager):
        """测试获取不存在的用户"""
        user = user_manager.get_user("nonexistent")
        assert user is None

    def test_update_user_basic(self, user_manager):
        """测试基本用户更新"""
        # 创建用户
        params = UserCreationParams(
            username="updateuser",
            email="update@example.com",
            is_active=True,
            roles={"user"}
        )
        user = user_manager.create_user(params)

        # 更新用户
        success = user_manager.update_user(user.user_id, email="newemail@example.com")

        assert success == True
        updated_user = user_manager.get_user(user.user_id)
        assert updated_user.email == "newemail@example.com"

    def test_update_user_nonexistent(self, user_manager):
        """测试更新不存在的用户"""
        success = user_manager.update_user("nonexistent", email="test@example.com")
        assert success == False

    def test_update_user_invalid_field(self, user_manager):
        """测试更新无效字段"""
        params = UserCreationParams(
            username="invalidupdate",
            email="invalid@example.com",
            is_active=True,
            roles={"user"}
        )
        user = user_manager.create_user(params)

        # 尝试更新不存在的字段
        success = user_manager.update_user(user.user_id, nonexistent_field="value")

        # 应该成功但不影响其他字段
        assert success == True
        updated_user = user_manager.get_user(user.user_id)
        assert updated_user.username == "invalidupdate"

    def test_delete_user_existing(self, user_manager):
        """测试删除现有用户"""
        params = UserCreationParams(
            username="deleteuser",
            email="delete@example.com",
            is_active=True,
            roles={"user"}
        )
        user = user_manager.create_user(params)

        success = user_manager.delete_user(user.user_id)

        assert success == True
        assert user_manager.get_user(user.user_id) is None
        assert user.user_id not in user_manager.users

    def test_delete_user_nonexistent(self, user_manager):
        """测试删除不存在的用户"""
        success = user_manager.delete_user("nonexistent")
        assert success == False

    def test_list_users_all(self, user_manager):
        """测试列出所有用户"""
        # 创建多个用户
        users_data = [
            ("user1", "user1@example.com", True),
            ("user2", "user2@example.com", False),
            ("user3", "user3@example.com", True),
        ]

        created_users = []
        for username, email, is_active in users_data:
            params = UserCreationParams(
                username=username,
                email=email,
                is_active=is_active,
                roles={"user"}
            )
            user = user_manager.create_user(params)
            created_users.append(user)

        all_users = user_manager.list_users(active_only=False)

        assert len(all_users) == 3
        assert all(user in all_users for user in created_users)

    def test_list_users_active_only(self, user_manager):
        """测试只列出活跃用户"""
        # 创建活跃和非活跃用户
        active_params = UserCreationParams(
            username="active",
            email="active@example.com",
            is_active=True,
            roles={"user"}
        )
        inactive_params = UserCreationParams(
            username="inactive",
            email="inactive@example.com",
            is_active=False,
            roles={"user"}
        )

        active_user = user_manager.create_user(active_params)
        inactive_user = user_manager.create_user(inactive_params)

        active_users = user_manager.list_users(active_only=True)

        assert len(active_users) == 1
        assert active_user in active_users
        assert inactive_user not in active_users

    def test_create_role_basic(self, user_manager):
        """测试基本角色创建"""
        role = user_manager.create_role("admin", {"read", "write"}, "管理员角色")

        assert role.role_id is not None
        assert role.name == "admin"
        assert role.permissions == {"read", "write"}
        assert role.description == "管理员角色"
        assert role.role_id in user_manager.roles

    def test_create_role_no_description(self, user_manager):
        """测试创建没有描述的角色"""
        role = user_manager.create_role("basic", {"read"})

        assert role.description == ""

    def test_assign_role_to_user(self, user_manager):
        """测试为用户分配角色"""
        # 创建用户
        user_params = UserCreationParams(
            username="assignuser",
            email="assign@example.com",
            is_active=True,
            roles={"user"}
        )
        user = user_manager.create_user(user_params)

        # 创建角色
        role = user_manager.create_role("manager", {"manage"})

        # 分配角色
        success = user_manager.assign_role_to_user(user.user_id, role.role_id)

        assert success == True
        updated_user = user_manager.get_user(user.user_id)
        assert role.name in updated_user.roles

    def test_assign_role_to_nonexistent_user(self, user_manager):
        """测试为不存在的用户分配角色"""
        success = user_manager.assign_role_to_user("nonexistent", "some_role")
        assert success == False

    def test_assign_nonexistent_role_to_user(self, user_manager):
        """测试为用户分配不存在的角色"""
        user_params = UserCreationParams(
            username="badassign",
            email="badassign@example.com",
            is_active=True,
            roles={"user"}
        )
        user = user_manager.create_user(user_params)

        success = user_manager.assign_role_to_user(user.user_id, "nonexistent_role")
        assert success == False

    def test_revoke_role_from_user(self, user_manager):
        """测试从用户撤销角色"""
        # 创建用户
        user_params = UserCreationParams(
            username="revokeuser",
            email="revoke@example.com",
            is_active=True,
            roles={"user"}
        )
        user = user_manager.create_user(user_params)

        # 创建角色并分配给用户
        role = user_manager.create_role("manager", {"manage"})
        user_manager.assign_role_to_user(user.user_id, role.role_id)

        # 撤销角色
        success = user_manager.revoke_role_from_user(user.user_id, role.role_id)

        assert success == True
        updated_user = user_manager.get_user(user.user_id)
        assert "manager" not in updated_user.roles
        assert "user" in updated_user.roles

    def test_revoke_nonexistent_role_from_user(self, user_manager):
        """测试从用户撤销不存在的角色"""
        user_params = UserCreationParams(
            username="badrevoke",
            email="badrevoke@example.com",
            is_active=True,
            roles={"user"}
        )
        user = user_manager.create_user(user_params)

        success = user_manager.revoke_role_from_user(user.user_id, "nonexistent")
        assert success == False

    def test_revoke_role_from_nonexistent_user(self, user_manager):
        """测试从不存在的用户撤销角色"""
        success = user_manager.revoke_role_from_user("nonexistent", "some_role")
        assert success == False

    def test_user_id_generation_uniqueness(self, user_manager):
        """测试用户ID生成的唯一性"""
        params1 = UserCreationParams(
            username="unique1",
            email="unique1@example.com",
            is_active=True,
            roles={"user"}
        )
        params2 = UserCreationParams(
            username="unique2",
            email="unique2@example.com",
            is_active=True,
            roles={"user"}
        )

        user1 = user_manager.create_user(params1)
        user2 = user_manager.create_user(params2)

        assert user1.user_id != user2.user_id
        assert user1.user_id.startswith("user_")
        assert user2.user_id.startswith("user_")

    def test_role_id_generation_uniqueness(self, user_manager):
        """测试角色ID生成的唯一性"""
        role1 = user_manager.create_role("role1", {"perm1"})
        role2 = user_manager.create_role("role2", {"perm2"})

        assert role1.role_id != role2.role_id
        assert role1.role_id.startswith("role_")
        assert role2.role_id.startswith("role_")

    def test_audit_logging_on_user_creation(self, user_manager):
        """测试用户创建时的审计日志"""
        params = UserCreationParams(
            username="audituser",
            email="audit@example.com",
            is_active=True,
            roles={"user"}
        )

        with patch('src.infrastructure.security.auth.user_manager.logging') as mock_logging:
            user = user_manager.create_user(params)

            # 验证审计日志调用
            mock_logging.info.assert_called()

    def test_bulk_operations_performance(self, user_manager):
        """测试批量操作性能"""
        import time

        # 创建大量用户
        start_time = time.time()
        user_count = 100

        for i in range(user_count):
            params = UserCreationParams(
                username=f"bulkuser{i}",
                email=f"bulk{i}@example.com",
                is_active=True,
                roles={"user"}
            )
            user_manager.create_user(params)

        creation_time = time.time() - start_time

        # 验证创建了正确数量的用户
        assert len(user_manager.users) == user_count

        # 验证列表操作
        start_time = time.time()
        users = user_manager.list_users(active_only=True)
        list_time = time.time() - start_time

        assert len(users) == user_count

        # 性能断言（允许合理的时间）
        assert creation_time < 5.0  # 创建100个用户应该在5秒内完成
        assert list_time < 1.0   # 列出100个用户应该在1秒内完成

    def test_memory_usage_with_large_user_base(self, user_manager):
        """测试大量用户基础的内存使用"""
        # 创建大量用户和角色
        for i in range(50):
            # 创建用户
            user_params = UserCreationParams(
                username=f"memuser{i}",
                email=f"mem{i}@example.com",
                is_active=True,
                roles={"user"}
            )
            user_manager.create_user(user_params)

            # 创建角色
            user_manager.create_role(f"memrole{i}", {f"perm{i}"})

        # 验证数据完整性
        assert len(user_manager.users) == 50
        assert len(user_manager.roles) == 50

        # 验证可以访问所有用户和角色
        all_users = user_manager.list_users(active_only=True)
        assert len(all_users) == 50

        # 验证用户名正确性
        usernames = {user.username for user in all_users}
        expected_usernames = {f"memuser{i}" for i in range(50)}
        assert usernames == expected_usernames

    def test_concurrent_access_simulation(self, user_manager):
        """测试并发访问模拟"""
        import threading
        import queue

        # 创建初始用户和角色
        user_params = UserCreationParams(
            username="concurrent",
            email="concurrent@example.com",
            is_active=True,
            roles={"user"}
        )
        user = user_manager.create_user(user_params)
        role = user_manager.create_role("concurrent_role", {"concurrent_perm"})

        results = queue.Queue()
        errors = queue.Queue()

        def worker(worker_id):
            try:
                # 模拟各种操作
                if worker_id % 4 == 0:
                    # 读取操作
                    retrieved = user_manager.get_user(user.user_id)
                    results.put(f"read_{worker_id}")
                elif worker_id % 4 == 1:
                    # 更新操作
                    success = user_manager.update_user(user.user_id, email=f"new{worker_id}@example.com")
                    results.put(f"update_{worker_id}_{success}")
                elif worker_id % 4 == 2:
                    # 角色分配
                    success = user_manager.assign_role_to_user(user.user_id, role.role_id)
                    results.put(f"assign_{worker_id}_{success}")
                else:
                    # 列表操作
                    users = user_manager.list_users()
                    results.put(f"list_{worker_id}_{len(users)}")

            except Exception as e:
                errors.put(f"worker_{worker_id}: {e}")

        # 启动多个线程
        threads = []
        for i in range(20):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # 等待所有线程完成
        for t in threads:
            t.join(timeout=5)

        # 验证至少有一些操作成功完成
        result_count = 0
        while not results.empty():
            results.get()
            result_count += 1

        # 至少应该有一些操作成功
        assert result_count > 0

    def test_error_handling_and_recovery(self, user_manager):
        """测试错误处理和恢复"""
        # 测试边界情况：空用户名
        try:
            invalid_params = UserCreationParams(
                username="",  # 空用户名
                email="test@example.com",
                is_active=True,
                roles={"user"}
            )
            user = user_manager.create_user(invalid_params)
            # 即使用户名为空，系统也应该能处理（可能生成默认ID）
            assert user is not None
        except Exception:
            # 如果抛出异常也是可以接受的
            pass

        # 验证系统仍然正常工作
        valid_params = UserCreationParams(
            username="recovery",
            email="recovery@example.com",
            is_active=True,
            roles={"user"}
        )
        user = user_manager.create_user(valid_params)
        assert user is not None
        assert user.username == "recovery"

    def test_data_consistency_after_operations(self, user_manager):
        """测试操作后的数据一致性"""
        # 执行一系列操作
        params = UserCreationParams(
            username="consistency",
            email="consistency@example.com",
            is_active=True,
            roles={"user"}
        )
        user = user_manager.create_user(params)

        role = user_manager.create_role("consistency_role", {"consistency_perm"})

        # 分配角色
        user_manager.assign_role_to_user(user.user_id, role.role_id)

        # 更新用户
        user_manager.update_user(user.user_id, email="newconsistency@example.com")

        # 撤销角色
        user_manager.revoke_role_from_user(user.user_id, role.role_id)

        # 最终验证
        final_user = user_manager.get_user(user.user_id)
        assert final_user.username == "consistency"
        assert final_user.email == "newconsistency@example.com"
        assert role.role_id not in final_user.roles
        assert "user" in final_user.roles  # 原始角色应该保留

    def test_update_user(self, user_manager):
        """测试更新用户信息"""
        # 创建用户
        params = UserCreationParams(
            username="update_test",
            email="update@example.com",
            is_active=True,
            roles={"user"}
        )
        user = user_manager.create_user(params)

        # 更新用户信息
        success = user_manager.update_user(user.user_id, email="newemail@example.com", is_active=False)

        assert success == True

        # 验证更新结果
        updated_user = user_manager.get_user(user.user_id)
        assert updated_user.email == "newemail@example.com"
        assert updated_user.is_active == False

    def test_update_user_nonexistent(self, user_manager):
        """测试更新不存在的用户"""
        success = user_manager.update_user("nonexistent", email="test@example.com")
        assert success == False

    def test_list_users_active_only(self, user_manager):
        """测试列出活跃用户"""
        # 创建多个用户
        active_params = UserCreationParams(
            username="active_user",
            email="active@example.com",
            is_active=True,
            roles={"user"}
        )
        inactive_params = UserCreationParams(
            username="inactive_user",
            email="inactive@example.com",
            is_active=False,
            roles={"user"}
        )

        user_manager.create_user(active_params)
        user_manager.create_user(inactive_params)

        # 测试只列出活跃用户
        active_users = user_manager.list_users(active_only=True)
        all_users = user_manager.list_users(active_only=False)

        assert len(active_users) == 1
        assert len(all_users) == 2
        assert active_users[0].username == "active_user"

    def test_generate_user_id_uniqueness(self, user_manager):
        """测试用户ID生成的唯一性"""
        ids = set()
        for _ in range(100):
            user_id = user_manager._generate_user_id()
            assert user_id not in ids
            ids.add(user_id)
            assert user_id.startswith("user_")
            assert len(user_id) == 13  # "user_" + 8 chars

    def test_generate_role_id_uniqueness(self, user_manager):
        """测试角色ID生成的唯一性"""
        ids = set()
        for _ in range(100):
            role_id = user_manager._generate_role_id()
            assert role_id not in ids
            ids.add(role_id)
            assert role_id.startswith("role_")
            assert len(role_id) == 13  # "role_" + 8 chars

    @patch('src.infrastructure.security.auth.user_manager.logging')
    def test_audit_user_creation(self, mock_logging, user_manager):
        """测试用户创建审计日志"""
        # 创建用户
        params = UserCreationParams(
            username="audit_test",
            email="audit@example.com",
            is_active=True,
            roles={"user"}
        )
        user = user_manager.create_user(params)

        # 验证审计日志被调用
        mock_logging.info.assert_called()
        call_args = mock_logging.info.call_args[0][0]
        assert "audit_test" in call_args
        assert user.user_id in call_args

    @patch('src.infrastructure.security.auth.user_manager.logging')
    def test_audit_user_creation_with_creator(self, mock_logging, user_manager):
        """测试带创建者信息的用户创建审计日志"""
        # 创建用户
        params = UserCreationParams(
            username="audit_creator_test",
            email="audit@example.com",
            is_active=True,
            roles={"user"},
            created_by="admin"
        )
        user = user_manager.create_user(params)

        # 验证审计日志包含创建者信息
        mock_logging.info.assert_called()
        call_args = mock_logging.info.call_args[0][0]
        assert "admin" in call_args

    def test_create_user_with_inactive_status(self, user_manager):
        """测试创建非活跃用户"""
        params = UserCreationParams(
            username="inactive_user",
            email="inactive@example.com",
            is_active=False,
            roles={"user"}
        )

        user = user_manager.create_user(params)

        assert user.is_active == False

        # 测试list_users时不包含非活跃用户
        active_users = user_manager.list_users(active_only=True)
        assert len([u for u in active_users if u.username == "inactive_user"]) == 0

    def test_role_operations_with_invalid_ids(self, user_manager):
        """测试角色操作时的无效ID处理"""
        # 创建用户和角色
        user_params = UserCreationParams(
            username="test_user",
            email="test@example.com",
            is_active=True,
            roles={"user"}
        )
        user = user_manager.create_user(user_params)
        role = user_manager.create_role("test_role", {"read"})

        # 测试分配不存在的角色
        success = user_manager.assign_role_to_user(user.user_id, "nonexistent_role")
        assert success == False

        # 测试分配角色给不存在的用户
        success = user_manager.assign_role_to_user("nonexistent_user", role.role_id)
        assert success == False

        # 测试撤销不存在的角色
        success = user_manager.revoke_role_from_user(user.user_id, "nonexistent_role")
        assert success == False

        # 测试撤销角色从不存在的用户
        success = user_manager.revoke_role_from_user("nonexistent_user", role.role_id)
        assert success == False

    def test_create_role_with_description(self, user_manager):
        """测试创建带描述的角色"""
        role = user_manager.create_role(
            name="manager_role",
            permissions={"read", "write", "manage"},
            description="管理人员角色"
        )

        assert role.name == "manager_role"
        assert role.description == "管理人员角色"
        assert role.permissions == {"read", "write", "manage"}

    def test_user_role_assignment_workflow(self, user_manager):
        """测试用户角色分配完整工作流"""
        # 1. 创建用户
        user_params = UserCreationParams(
            username="workflow_user",
            email="workflow@example.com",
            is_active=True,
            roles={"basic"}  # 初始角色
        )
        user = user_manager.create_user(user_params)

        # 2. 创建多个角色
        admin_role = user_manager.create_role("admin", {"admin:access", "manage:users"})
        moderator_role = user_manager.create_role("moderator", {"moderate:content", "delete:posts"})

        # 3. 分配角色
        success1 = user_manager.assign_role_to_user(user.user_id, admin_role.role_id)
        success2 = user_manager.assign_role_to_user(user.user_id, moderator_role.role_id)

        assert success1 == True
        assert success2 == True

        # 4. 验证角色分配
        updated_user = user_manager.get_user(user.user_id)
        assert "admin" in updated_user.roles
        assert "moderator" in updated_user.roles

        # 5. 撤销一个角色
        success3 = user_manager.revoke_role_from_user(user.user_id, admin_role.role_id)
        assert success3 == True

        # 6. 验证角色撤销
        final_user = user_manager.get_user(user.user_id)
        assert "admin" not in final_user.roles
        assert "moderator" in final_user.roles

    def test_edge_case_empty_permissions(self, user_manager):
        """测试空权限集合的边界情况"""
        role = user_manager.create_role("empty_role", set())
        assert role.permissions == set()

        user_params = UserCreationParams(
            username="empty_perm_user",
            email="empty@example.com",
            is_active=True,
            roles=set()
        )
        user = user_manager.create_user(user_params)
        assert user.roles == set()

    def test_user_manager_state_consistency(self, user_manager):
        """测试用户管理器状态一致性"""
        # 执行一系列操作，验证状态保持一致
        initial_user_count = len(user_manager.users)
        initial_role_count = len(user_manager.roles)

        # 创建用户和角色
        user_params = UserCreationParams(
            username="consistency_test",
            email="consistency@example.com",
            is_active=True,
            roles={"user"}
        )
        user = user_manager.create_user(user_params)
        role = user_manager.create_role("consistency_role", {"test"})

        assert len(user_manager.users) == initial_user_count + 1
        assert len(user_manager.roles) == initial_role_count + 1

        # 删除用户和角色
        user_manager.delete_user(user.user_id)
        # 注意：这里没有删除角色的方法，需要手动删除
        del user_manager.roles[role.role_id]

        assert len(user_manager.users) == initial_user_count
        assert len(user_manager.roles) == initial_role_count

    def test_error_handling_in_user_operations(self, user_manager):
        """测试用户操作中的错误处理"""
        # 测试删除不存在的用户
        success = user_manager.delete_user("nonexistent")
        assert success == False

        # 测试获取不存在的用户
        user = user_manager.get_user("nonexistent")
        assert user is None

        # 测试更新不存在的用户
        success = user_manager.update_user("nonexistent", email="test@example.com")
        assert success == False


@pytest.fixture
def permission_manager(user_manager):
    """创建PermissionManager实例"""
    from src.infrastructure.security.auth.user_manager import PermissionManager
    return PermissionManager(user_manager)


@pytest.fixture
def user_manager_with_data(user_manager):
    """创建带有测试数据的UserManager"""
    # 创建基本角色
    employee_role = user_manager.create_role("employee", {"basic:read"})
    manager_role = user_manager.create_role("manager", {"manager:approve", "team:manage"})
    user_manager.roles[manager_role.role_id].parent_roles = {"employee"}  # 设置父角色

    # 创建用户
    user_params = UserCreationParams(
        username="test_user",
        email="test@example.com",
        is_active=True,
        roles={"employee"}  # 使用角色名而不是ID
    )
    user = user_manager.create_user(user_params)

    return user_manager, user


@pytest.fixture
def permission_manager_with_data(user_manager_with_data):
    """创建带有数据的PermissionManager实例"""
    from src.infrastructure.security.auth.user_manager import PermissionManager
    user_manager, _ = user_manager_with_data
    return PermissionManager(user_manager)


class TestPermissionManagerComprehensive:
    """PermissionManager综合测试"""

    def test_initialization(self, permission_manager_with_data):
        """测试初始化"""
        assert permission_manager_with_data.user_manager is not None
        assert permission_manager_with_data.permission_cache == {}

    def test_check_user_permission_role_based(self, permission_manager_with_data, user_manager_with_data):
        """测试基于角色的权限检查"""
        user_manager, user = user_manager_with_data

        params = AccessCheckParams(
            user_id=user.user_id,
            resource="/api/test",
            permission="basic:read",
            check_cache=False
        )

        result = permission_manager_with_data.check_permission(params)
        assert result == True

    def test_check_user_permission_inherited(self, permission_manager_with_data, user_manager_with_data):
        """测试继承权限检查"""
        user_manager, user = user_manager_with_data

        # 将用户角色改为manager（继承自employee）
        user.roles = {"manager"}

        params = AccessCheckParams(
            user_id=user.user_id,
            resource="/api/test",
            permission="basic:read",  # 继承自employee角色的权限
            check_cache=False,
            include_inherited=True
        )

        result = permission_manager_with_data.check_permission(params)
        assert result == True

    def test_check_user_permission_no_permission(self, permission_manager_with_data, user_manager_with_data):
        """测试无权限的情况"""
        user_manager, user = user_manager_with_data

        params = AccessCheckParams(
            user_id=user.user_id,
            resource="/api/admin",
            permission="admin:access",  # 用户没有的权限
            check_cache=False
        )

        result = permission_manager_with_data.check_permission(params)
        assert result == False

    def test_check_user_permission_nonexistent_user(self, permission_manager_with_data):
        """测试不存在的用户"""
        params = AccessCheckParams(
            user_id="nonexistent_user",
            resource="/api/test",
            permission="any:permission",
            check_cache=False
        )

        result = permission_manager_with_data.check_permission(params)
        assert result == False

    def test_permission_caching(self, permission_manager_with_data, user_manager_with_data):
        """测试权限缓存功能"""
        user_manager, user = user_manager_with_data

        params = AccessCheckParams(
            user_id=user.user_id,
            resource="/api/test",
            permission="basic:read",
            check_cache=True
        )

        # 第一次检查，应该缓存结果
        result1 = permission_manager_with_data.check_permission(params)
        assert result1 == True

        # 验证缓存
        cache_key = permission_manager_with_data._get_cache_key(params)
        assert cache_key in permission_manager_with_data.permission_cache
        assert permission_manager_with_data.permission_cache[cache_key] == True

        # 第二次检查，应该使用缓存
        result2 = permission_manager_with_data.check_permission(params)
        assert result2 == True

    def test_permission_caching_disabled(self, permission_manager_with_data, user_manager_with_data):
        """测试禁用缓存的情况"""
        user_manager, user = user_manager_with_data

        params = AccessCheckParams(
            user_id=user.user_id,
            resource="/api/test",
            permission="basic:read",
            check_cache=False
        )

        # 检查缓存为空
        cache_key = permission_manager_with_data._get_cache_key(params)
        assert cache_key not in permission_manager_with_data.permission_cache

        # 执行检查
        result = permission_manager_with_data.check_permission(params)
        assert result == True

        # 验证没有缓存结果
        assert cache_key not in permission_manager_with_data.permission_cache

    def test_get_cache_key_generation(self, permission_manager_with_data):
        """测试缓存键生成"""
        params = AccessCheckParams(
            user_id="user123",
            resource="/api/data",
            permission="read",
            context={"extra": "info"}
        )

        cache_key = permission_manager_with_data._get_cache_key(params)
        # 缓存键应该包含用户ID、资源、权限
        assert "user123" in cache_key
        assert "/api/data" in cache_key
        assert "read" in cache_key
        # 确保缓存键是字符串
        assert isinstance(cache_key, str)

    def test_check_inherited_permissions_no_inheritance(self, permission_manager_with_data, user_manager_with_data):
        """测试不包含继承权限的情况"""
        user_manager, user = user_manager_with_data

        # 设置用户只有employee角色，但检查manager权限

        params = AccessCheckParams(
            user_id=user.user_id,
            resource="/api/test",
            permission="manager:approve",  # manager角色的权限，但用户没有继承
            check_cache=False,
            include_inherited=False
        )

        result = permission_manager_with_data.check_permission(params)
        assert result == False

    def test_check_inherited_permissions_with_inheritance(self, permission_manager_with_data, user_manager_with_data):
        """测试包含继承权限的情况"""
        user_manager, user = user_manager_with_data

        # 将用户角色改为manager（继承自employee）
        user.roles = {"manager"}

        params = AccessCheckParams(
            user_id=user.user_id,
            resource="/api/test",
            permission="basic:read",  # employee角色的权限，通过继承获得
            check_cache=False,
            include_inherited=True
        )

        result = permission_manager_with_data.check_permission(params)
        assert result == True

    def test_permission_manager_with_multiple_roles(self, permission_manager_with_data, user_manager):
        """测试多角色权限检查"""
        # 创建多个角色
        admin_role = user_manager.create_role("admin", {"admin:full", "system:config"})
        user_role = user_manager.create_role("user", {"user:profile", "data:read"})

        # 创建用户并分配多个角色
        user_params = UserCreationParams(
            username="multi_role_user",
            email="multi@example.com",
            is_active=True,
            roles={"admin", "user"}
        )
        user = user_manager.create_user(user_params)

        # 测试admin权限
        admin_params = AccessCheckParams(
            user_id=user.user_id,
            resource="/api/admin",
            permission="admin:full",
            check_cache=False
        )
        assert permission_manager_with_data.check_permission(admin_params) == True

        # 测试user权限
        user_params_check = AccessCheckParams(
            user_id=user.user_id,
            resource="/api/profile",
            permission="user:profile",
            check_cache=False
        )
        assert permission_manager_with_data.check_permission(user_params_check) == True

        # 测试不存在的权限
        invalid_params = AccessCheckParams(
            user_id=user.user_id,
            resource="/api/unknown",
            permission="unknown:permission",
            check_cache=False
        )
        assert permission_manager_with_data.check_permission(invalid_params) == False

    @patch('src.infrastructure.security.auth.user_manager.logging')
    def test_log_access_check(self, mock_logging, permission_manager_with_data, user_manager_with_data):
        """测试访问检查日志记录"""
        user_manager, user = user_manager_with_data

        params = AccessCheckParams(
            user_id=user.user_id,
            resource="/api/test",
            permission="basic:read",
            check_cache=False
        )

        # 执行权限检查，这会触发日志记录
        result = permission_manager_with_data.check_permission(params)

        # 验证日志被调用
        mock_logging.info.assert_called()
        call_args = mock_logging.info.call_args[0][0]
        assert user.user_id in call_args
        assert "/api/test" in call_args
        assert "basic:read" in call_args
        assert str(result) in call_args

    def test_permission_cache_invalidation(self, permission_manager_with_data, user_manager_with_data):
        """测试权限缓存失效"""
        user_manager, user = user_manager_with_data

        params = AccessCheckParams(
            user_id=user.user_id,
            resource="/api/test",
            permission="basic:read",
            check_cache=True
        )

        # 第一次检查，缓存结果
        result1 = permission_manager_with_data.check_permission(params)
        assert result1 == True

        # 验证缓存存在
        cache_key = permission_manager_with_data._get_cache_key(params)
        assert cache_key in permission_manager_with_data.permission_cache

        # 清除缓存
        permission_manager_with_data.permission_cache.clear()

        # 再次检查，重新计算
        result2 = permission_manager_with_data.check_permission(params)
        assert result2 == True

        # 缓存再次存在
        assert cache_key in permission_manager_with_data.permission_cache

    def test_complex_permission_hierarchy(self, permission_manager_with_data, user_manager):
        """测试复杂的权限层次结构"""
        # 创建多层继承结构
        # basic -> intermediate -> advanced
        basic_role = user_manager.create_role("basic", {"read"})
        intermediate_role = user_manager.create_role("intermediate", {"write"})
        user_manager.roles[intermediate_role.role_id].parent_roles = {"basic"}

        advanced_role = user_manager.create_role("advanced", {"admin"})
        user_manager.roles[advanced_role.role_id].parent_roles = {"intermediate"}

        # 创建用户并分配高级角色
        user_params = UserCreationParams(
            username="advanced_user",
            email="advanced@example.com",
            is_active=True,
            roles={"advanced"}
        )
        user = user_manager.create_user(user_params)

        # 测试所有层级的权限
        test_cases = [
            ("read", True),      # 基本权限（通过3层继承）
            ("write", True),     # 中间权限（通过2层继承）
            ("admin", True),     # 高级权限（直接权限）
            ("delete", False),   # 不存在的权限
        ]

        for permission, expected in test_cases:
            params = AccessCheckParams(
                user_id=user.user_id,
                resource="/api/test",
                permission=permission,
                check_cache=False,
                include_inherited=True
            )
            result = permission_manager_with_data.check_permission(params)
            assert result == expected, f"Permission {permission} should be {expected}"

    def test_permission_context_sensitivity(self, permission_manager_with_data):
        """测试权限上下文敏感性"""
        # 不同的上下文应该产生不同的缓存键
        params1 = AccessCheckParams(
            user_id="user1",
            resource="/api/data",
            permission="read",
            context={"source": "web"}
        )

        params2 = AccessCheckParams(
            user_id="user1",
            resource="/api/data",
            permission="read",
            context={"source": "mobile"}
        )

        key1 = permission_manager_with_data._get_cache_key(params1)
        key2 = permission_manager_with_data._get_cache_key(params2)

        assert key1 != key2  # 不同的上下文应该有不同的缓存键

    def test_permission_manager_edge_cases(self, permission_manager_with_data):
        """测试权限管理器的边界情况"""
        # 测试空权限
        params = AccessCheckParams(
            user_id="user1",
            resource="/api/test",
            permission="",  # 空权限
            check_cache=False
        )

        # 这应该不会崩溃，但结果取决于具体实现
        result = permission_manager_with_data.check_permission(params)
        assert isinstance(result, bool)

        # 测试特殊字符权限
        params = AccessCheckParams(
            user_id="user1",
            resource="/api/test",
            permission="special:chars!@#$%",
            check_cache=False
        )

        result = permission_manager_with_data.check_permission(params)
        assert isinstance(result, bool)
