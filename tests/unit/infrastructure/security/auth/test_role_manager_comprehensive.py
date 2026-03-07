#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 角色管理器综合测试

全面测试RoleManager类的所有功能，包括：
- 角色创建、更新和删除
- 权限管理和继承
- 角色层次结构和验证
- 模板角色创建
- 用户角色分配
- 统计信息获取
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
from typing import Dict, List, Set, Any

from src.infrastructure.security.auth.role_manager import (
    RoleManager, Role, RoleDefinition, UserRole, Permission
)


class TestRoleManagerComprehensive:
    """角色管理器综合测试"""

    @pytest.fixture
    def role_manager(self):
        """角色管理器fixture"""
        return RoleManager()

    def test_initialization(self, role_manager):
        """测试初始化"""
        assert isinstance(role_manager.roles, dict)
        assert isinstance(role_manager.role_definitions, dict)
        assert len(role_manager.role_definitions) == 5  # 默认5个角色

        # 验证默认角色定义
        assert UserRole.ADMIN in role_manager.role_definitions
        assert UserRole.TRADER in role_manager.role_definitions
        assert UserRole.ANALYST in role_manager.role_definitions
        assert UserRole.AUDITOR in role_manager.role_definitions
        assert UserRole.GUEST in role_manager.role_definitions

    def test_create_role_basic(self, role_manager):
        """测试基本角色创建"""
        role = role_manager.create_role(
            role_id="test_role",
            name="测试角色",
            description="用于测试的角色"
        )

        assert role.role_id == "test_role"
        assert role.name == "测试角色"
        assert role.description == "用于测试的角色"
        assert role.permissions == set()
        assert role.parent_roles == set()
        assert role.is_active == True
        assert isinstance(role.created_at, datetime)

        # 验证角色已添加到管理器
        assert "test_role" in role_manager.roles
        assert role_manager.get_role("test_role") == role

    def test_create_role_with_permissions(self, role_manager):
        """测试创建带权限的角色"""
        permissions = {"read:data", "write:data"}
        parent_roles = {"parent1", "parent2"}

        role = role_manager.create_role(
            role_id="advanced_role",
            name="高级角色",
            description="具有高级权限的角色",
            permissions=permissions,
            parent_roles=parent_roles
        )

        assert role.permissions == permissions
        assert role.parent_roles == parent_roles

    def test_create_role_duplicate_id(self, role_manager):
        """测试创建重复ID的角色"""
        # 先创建角色
        role_manager.create_role("duplicate", "原始角色")

        # 再次创建相同ID的角色应该失败
        with pytest.raises(ValueError, match="角色已存在"):
            role_manager.create_role("duplicate", "重复角色")

    def test_update_role(self, role_manager):
        """测试角色更新"""
        # 创建角色
        role = role_manager.create_role("update_test", "更新测试")

        # 更新角色
        success = role_manager.update_role(
            "update_test",
            name="已更新角色",
            description="更新后的描述",
            is_active=False
        )

        assert success == True

        # 验证更新结果
        updated_role = role_manager.get_role("update_test")
        assert updated_role.name == "已更新角色"
        assert updated_role.description == "更新后的描述"
        assert updated_role.is_active == False

    def test_update_role_nonexistent(self, role_manager):
        """测试更新不存在的角色"""
        success = role_manager.update_role("nonexistent", name="新名称")
        assert success == False

    def test_update_role_invalid_attribute(self, role_manager):
        """测试更新无效属性"""
        role = role_manager.create_role("invalid_attr", "测试角色")

        # 更新不存在的属性
        success = role_manager.update_role("invalid_attr", invalid_attr="value")
        assert success == True

        # 属性不会被设置（因为hasattr检查会失败）
        role = role_manager.get_role("invalid_attr")
        assert not hasattr(role, 'invalid_attr')

    def test_delete_role(self, role_manager):
        """测试角色删除"""
        # 创建角色
        role_manager.create_role("delete_test", "待删除角色")

        # 验证角色存在
        assert role_manager.get_role("delete_test") is not None

        # 删除角色
        success = role_manager.delete_role("delete_test")
        assert success == True

        # 验证角色已删除
        assert role_manager.get_role("delete_test") is None

    def test_delete_role_nonexistent(self, role_manager):
        """测试删除不存在的角色"""
        success = role_manager.delete_role("nonexistent")
        assert success == False

    def test_get_role(self, role_manager):
        """测试获取角色"""
        created_role = role_manager.create_role("get_test", "获取测试")

        retrieved_role = role_manager.get_role("get_test")
        assert retrieved_role == created_role

    def test_get_role_nonexistent(self, role_manager):
        """测试获取不存在的角色"""
        role = role_manager.get_role("nonexistent")
        assert role is None

    def test_list_roles_all(self, role_manager):
        """测试列出所有角色"""
        # 创建一些角色
        role1 = role_manager.create_role("list1", "角色1")
        role2 = role_manager.create_role("list2", "角色2")

        roles = role_manager.list_roles(active_only=False)

        # 应该包含我们创建的角色和默认角色
        role_ids = {r.role_id for r in roles}
        assert "list1" in role_ids
        assert "list2" in role_ids
        # 默认角色也会被包含

    def test_list_roles_active_only(self, role_manager):
        """测试只列出活跃角色"""
        # 创建活跃和非活跃角色
        active_role = role_manager.create_role("active", "活跃角色")
        inactive_role = role_manager.create_role("inactive", "非活跃角色")
        role_manager.update_role("inactive", is_active=False)

        roles = role_manager.list_roles(active_only=True)

        # 应该只包含活跃角色
        role_ids = {r.role_id for r in roles}
        assert "active" in role_ids
        assert "inactive" not in role_ids

    def test_get_role_permissions_simple(self, role_manager):
        """测试获取简单角色的权限"""
        # 创建有权限的角色
        permissions = {"read:data", "write:data"}
        role = role_manager.create_role("perm_test", "权限测试", permissions=permissions)

        retrieved_permissions = role_manager.get_role_permissions("perm_test")
        assert retrieved_permissions == permissions

    def test_get_role_permissions_with_inheritance(self, role_manager):
        """测试获取带继承关系的角色权限"""
        # 创建父角色
        parent_permissions = {"read:data"}
        parent_role = role_manager.create_role("parent", "父角色", permissions=parent_permissions)

        # 创建子角色
        child_permissions = {"write:data"}
        child_role = role_manager.create_role(
            "child",
            "子角色",
            permissions=child_permissions,
            parent_roles={"parent"}
        )

        # 获取子角色的所有权限
        all_permissions = role_manager.get_role_permissions("child")
        expected_permissions = parent_permissions | child_permissions
        assert all_permissions == expected_permissions

    def test_get_role_permissions_nonexistent(self, role_manager):
        """测试获取不存在角色的权限"""
        permissions = role_manager.get_role_permissions("nonexistent")
        assert permissions == set()

    def test_check_role_permission_direct(self, role_manager):
        """测试检查角色直接权限"""
        role = role_manager.create_role("check_perm", "权限检查", permissions={"read:data"})

        assert role_manager.check_role_permission("check_perm", "read:data") == True
        assert role_manager.check_role_permission("check_perm", "write:data") == False

    def test_check_role_permission_inherited(self, role_manager):
        """测试检查角色继承权限"""
        # 创建父角色
        parent_role = role_manager.create_role("parent_perm", "父权限", permissions={"read:data"})

        # 创建子角色
        child_role = role_manager.create_role(
            "child_perm",
            "子权限",
            permissions={"write:data"},
            parent_roles={"parent_perm"}
        )

        # 检查继承权限
        assert role_manager.check_role_permission("child_perm", "read:data") == True
        assert role_manager.check_role_permission("child_perm", "write:data") == True
        assert role_manager.check_role_permission("child_perm", "admin") == False

    def test_check_role_permission_nonexistent_role(self, role_manager):
        """测试检查不存在角色的权限"""
        result = role_manager.check_role_permission("nonexistent", "any:perm")
        assert result == False

    def test_get_roles_with_permission(self, role_manager):
        """测试获取拥有指定权限的所有角色"""
        # 创建具有不同权限的角色
        role1 = role_manager.create_role("role1", "角色1", permissions={"read:data", "write:data"})
        role2 = role_manager.create_role("role2", "角色2", permissions={"read:data", "admin"})
        role3 = role_manager.create_role("role3", "角色3", permissions={"write:data"})

        # 查找拥有read:data权限的角色
        roles_with_read = role_manager.get_roles_with_permission("read:data")
        assert "role1" in roles_with_read
        assert "role2" in roles_with_read
        assert "role3" not in roles_with_read

        # 查找拥有admin权限的角色
        roles_with_admin = role_manager.get_roles_with_permission("admin")
        assert "role2" in roles_with_admin
        assert len(roles_with_admin) == 1

    def test_get_roles_with_permission_inherited(self, role_manager):
        """测试获取拥有继承权限的所有角色"""
        # 创建父角色
        parent = role_manager.create_role("parent_inherit", "父角色", permissions={"inherited:perm"})

        # 创建继承父角色的子角色
        child = role_manager.create_role(
            "child_inherit",
            "子角色",
            permissions={"own:perm"},
            parent_roles={"parent_inherit"}
        )

        # 子角色应该拥有继承的权限
        roles_with_inherited = role_manager.get_roles_with_permission("inherited:perm")
        assert "child_inherit" in roles_with_inherited

    def test_create_role_from_template_admin(self, role_manager):
        """测试从管理员模板创建角色"""
        # 管理员角色在初始化时已创建，直接获取
        role = role_manager.get_role("admin")
        assert role is not None
        assert role.role_id == "admin"
        assert role.name == "管理员"
        assert "system:admin" in role.permissions  # 管理员应该有所有权限

    def test_create_role_from_template_trader(self, role_manager):
        """测试从交易员模板创建角色"""
        # 交易员角色在初始化时已创建，直接获取
        role = role_manager.get_role("trader")
        assert role is not None
        assert role.role_id == "trader"
        assert role.name == "交易员"
        assert "trade:execute" in role.permissions
        assert "data:read" in role.permissions

    def test_create_role_from_template_guest(self, role_manager):
        """测试从访客模板创建角色"""
        # 访客角色在初始化时已创建，直接获取
        role = role_manager.get_role("guest")
        assert role is not None
        assert role.role_id == "guest"
        assert role.name == "访客"
        assert role.permissions == {"data:read"}

    def test_create_role_from_template_invalid(self, role_manager):
        """测试从无效模板创建角色"""
        # 先删除一个角色定义来模拟
        if UserRole.GUEST in role_manager.role_definitions:
            del role_manager.role_definitions[UserRole.GUEST]

        role = role_manager.create_role_from_template(UserRole.GUEST)
        assert role is None

    def test_get_role_hierarchy(self, role_manager):
        """测试获取角色层次结构"""
        # 创建带父子关系的角色
        parent = role_manager.create_role("hierarchy_parent", "父角色")
        child = role_manager.create_role(
            "hierarchy_child",
            "子角色",
            parent_roles={"hierarchy_parent"}
        )

        hierarchy = role_manager.get_role_hierarchy()

        assert "hierarchy_parent" in hierarchy
        assert "hierarchy_child" in hierarchy
        assert hierarchy["hierarchy_child"] == ["hierarchy_parent"]
        assert hierarchy["hierarchy_parent"] == []

    def test_validate_role_hierarchy_valid(self, role_manager):
        """测试验证有效的角色层次结构"""
        # 创建有效的层次结构
        role_manager.create_role("valid_parent", "有效父角色")
        role_manager.create_role("valid_child", "有效子角色", parent_roles={"valid_parent"})

        issues = role_manager.validate_role_hierarchy()
        assert len(issues) == 0

    def test_validate_role_hierarchy_cycle(self, role_manager):
        """测试验证循环依赖的角色层次结构"""
        # 创建循环依赖
        role_a = role_manager.create_role("cycle_a", "循环A", parent_roles={"cycle_b"})
        role_b = role_manager.create_role("cycle_b", "循环B", parent_roles={"cycle_a"})

        issues = role_manager.validate_role_hierarchy()

        # 应该检测到循环依赖
        cycle_issues = [issue for issue in issues if "循环依赖" in issue]
        assert len(cycle_issues) > 0

    def test_validate_role_hierarchy_missing_parent(self, role_manager):
        """测试验证引用不存在父角色的层次结构"""
        # 创建引用不存在父角色的角色
        role_manager.create_role("orphan", "孤儿角色", parent_roles={"nonexistent_parent"})

        issues = role_manager.validate_role_hierarchy()

        # 应该检测到不存在的父角色
        missing_parent_issues = [issue for issue in issues if "不存在的父角色" in issue]
        assert len(missing_parent_issues) > 0

    def test_get_role_stats(self, role_manager):
        """测试获取角色统计信息"""
        # 创建一些测试角色
        active_role = role_manager.create_role("stats_active", "活跃角色", permissions={"perm1"})
        inactive_role = role_manager.create_role("stats_inactive", "非活跃角色", permissions={"perm2"})
        role_manager.update_role("stats_inactive", is_active=False)

        stats = role_manager.get_role_stats()

        assert isinstance(stats, dict)
        assert "total_roles" in stats
        assert "active_roles" in stats
        assert "inactive_roles" in stats
        assert "permissions_distribution" in stats
        assert "hierarchy_issues" in stats

        # 验证统计数据
        assert stats["total_roles"] >= 2  # 至少包含我们创建的角色
        assert stats["active_roles"] >= 1
        assert stats["inactive_roles"] >= 1

        # 验证权限分布
        perm_dist = stats["permissions_distribution"]
        assert "perm1" in perm_dist
        assert "perm2" in perm_dist

    def test_assign_role_to_user(self, role_manager):
        """测试为用户分配角色"""
        with patch('src.infrastructure.security.auth.role_manager.logging') as mock_logging:
            success = role_manager.assign_role_to_user("user123", "admin")

            assert success == True
            mock_logging.info.assert_called_once_with("为用户 user123 分配角色 admin")

    def test_revoke_role_from_user(self, role_manager):
        """测试撤销用户的角色"""
        with patch('src.infrastructure.security.auth.role_manager.logging') as mock_logging:
            success = role_manager.revoke_role_from_user("user123", "admin")

            assert success == True
            mock_logging.info.assert_called_once_with("撤销用户 user123 的角色 admin")

    def test_get_user_roles(self, role_manager):
        """测试获取用户的角色"""
        # 这个方法目前只是返回空列表（需要在UserManager中实现）
        roles = role_manager.get_user_roles("user123")
        assert roles == []

    def test_role_add_permission(self):
        """测试角色添加权限"""
        role = Role(role_id="test", name="测试角色")

        role.add_permission("new:permission")
        assert "new:permission" in role.permissions

        # 重复添加应该不会出错
        initial_count = len(role.permissions)
        role.add_permission("new:permission")
        assert len(role.permissions) == initial_count

    def test_role_remove_permission(self):
        """测试角色移除权限"""
        role = Role(
            role_id="test",
            name="测试角色",
            permissions={"perm1", "perm2"}
        )

        role.remove_permission("perm1")
        assert "perm1" not in role.permissions
        assert "perm2" in role.permissions

        # 移除不存在的权限应该不会出错
        role.remove_permission("nonexistent")
        assert len(role.permissions) == 1

    def test_role_add_parent_role(self):
        """测试角色添加父角色"""
        role = Role(role_id="test", name="测试角色")

        role.add_parent_role("parent1")
        assert "parent1" in role.parent_roles

        # 重复添加应该不会出错
        initial_count = len(role.parent_roles)
        role.add_parent_role("parent1")
        assert len(role.parent_roles) == initial_count

    def test_role_remove_parent_role(self):
        """测试角色移除父角色"""
        role = Role(
            role_id="test",
            name="测试角色",
            parent_roles={"parent1", "parent2"}
        )

        role.remove_parent_role("parent1")
        assert "parent1" not in role.parent_roles
        assert "parent2" in role.parent_roles

    def test_role_get_all_permissions_no_inheritance(self):
        """测试获取角色所有权限（无继承）"""
        role = Role(
            role_id="test",
            name="测试角色",
            permissions={"perm1", "perm2"}
        )

        all_roles = {"test": role}
        permissions = role.get_all_permissions(all_roles)

        assert permissions == {"perm1", "perm2"}

    def test_role_get_all_permissions_with_inheritance(self):
        """测试获取角色所有权限（带继承）"""
        parent_role = Role(
            role_id="parent",
            name="父角色",
            permissions={"parent_perm"}
        )

        child_role = Role(
            role_id="child",
            name="子角色",
            permissions={"child_perm"},
            parent_roles={"parent"}
        )

        all_roles = {"parent": parent_role, "child": child_role}

        permissions = child_role.get_all_permissions(all_roles)
        assert permissions == {"parent_perm", "child_perm"}

    def test_role_get_all_permissions_circular_inheritance(self):
        """测试循环继承的权限获取"""
        role_a = Role(
            role_id="a",
            name="角色A",
            permissions={"perm_a"},
            parent_roles={"b"}
        )

        role_b = Role(
            role_id="b",
            name="角色B",
            permissions={"perm_b"},
            parent_roles={"a"}
        )

        all_roles = {"a": role_a, "b": role_b}

        # 循环继承应该被正确处理（避免无限递归）
        permissions_a = role_a.get_all_permissions(all_roles)
        permissions_b = role_b.get_all_permissions(all_roles)

        # 应该至少包含自身的权限
        assert "perm_a" in permissions_a
        assert "perm_b" in permissions_b

    def test_role_initialization_defaults(self):
        """测试角色初始化默认值"""
        role = Role(role_id="test", name="测试")

        assert role.role_id == "test"
        assert role.name == "测试"
        assert role.description == ""
        assert role.permissions == set()
        assert role.parent_roles == set()
        assert role.is_active == True
        assert isinstance(role.created_at, datetime)
        assert role.metadata == {}

    def test_concurrent_role_operations(self, role_manager):
        """测试并发角色操作"""
        import threading
        import time

        results = []
        errors = []

        def role_operations(thread_id):
            try:
                # 创建角色
                role_id = f"concurrent_role_{thread_id}"
                role = role_manager.create_role(role_id, f"并发角色{thread_id}")
                results.append(("create", thread_id, role.role_id))

                # 更新角色
                success = role_manager.update_role(role_id, description=f"更新描述{thread_id}")
                results.append(("update", thread_id, success))

                # 获取角色
                retrieved = role_manager.get_role(role_id)
                results.append(("get", thread_id, retrieved is not None))

                # 检查权限
                permissions = role_manager.get_role_permissions(role_id)
                results.append(("permissions", thread_id, len(permissions)))

            except Exception as e:
                errors.append((thread_id, str(e)))

        # 启动多个线程
        threads = []
        for i in range(5):
            thread = threading.Thread(target=role_operations, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证结果
        assert len(results) == 20  # 5线程 * 4操作
        assert len(errors) == 0

        # 验证创建操作都成功
        create_results = [r for r in results if r[0] == "create"]
        assert len(create_results) == 5

    def test_role_manager_logging(self, role_manager):
        """测试角色管理器的日志记录"""
        with patch('src.infrastructure.security.auth.role_manager.logging') as mock_logging:
            # 创建角色
            role_manager.create_role("logging_test", "日志测试")

            # 更新角色
            role_manager.update_role("logging_test", name="更新名称")

            # 删除角色
            role_manager.delete_role("logging_test")

            # 验证日志调用
            assert mock_logging.info.call_count >= 3

    def test_default_role_definitions(self, role_manager):
        """测试默认角色定义"""
        definitions = role_manager.role_definitions

        # 验证管理员角色
        admin_def = definitions[UserRole.ADMIN]
        assert admin_def.name == "管理员"
        assert admin_def.description == "系统管理员，拥有所有权限"
        assert len(admin_def.permissions) > 0  # 应该有很多权限

        # 验证交易员角色
        trader_def = definitions[UserRole.TRADER]
        assert trader_def.name == "交易员"
        assert "trade:execute" in {p.value for p in trader_def.permissions}

        # 验证访客角色
        guest_def = definitions[UserRole.GUEST]
        assert guest_def.name == "访客"
        assert guest_def.permissions == {Permission.DATA_READ}

    def test_role_creation_with_metadata(self, role_manager):
        """测试创建带元数据的角色"""
        metadata = {"created_by": "admin", "department": "IT", "priority": "high"}

        role = role_manager.create_role(
            "metadata_test",
            "元数据测试",
            metadata=metadata
        )

        assert role.metadata == metadata

    def test_bulk_role_operations_performance(self, role_manager):
        """测试批量角色操作性能"""
        import time

        # 创建大量角色
        start_time = time.time()
        for i in range(100):
            role_manager.create_role(f"bulk_role_{i}", f"批量角色{i}")
        creation_time = time.time() - start_time

        # 查询所有角色
        start_time = time.time()
        roles = role_manager.list_roles(active_only=False)
        query_time = time.time() - start_time

        # 验证结果
        assert len(roles) >= 100  # 至少包含我们创建的角色

        # 性能应该在合理范围内
        assert creation_time < 5.0  # 创建100个角色应该在5秒内完成
        assert query_time < 1.0    # 查询应该在1秒内完成

    def test_role_hierarchy_complex_scenario(self, role_manager):
        """测试复杂的角色层次结构场景"""
        # 创建多层继承结构
        # grandparent -> parent -> child -> grandchild

        grandparent = role_manager.create_role(
            "grandparent",
            "祖父角色",
            permissions={"ancient:perm"}
        )

        parent = role_manager.create_role(
            "parent",
            "父角色",
            permissions={"parent:perm"},
            parent_roles={"grandparent"}
        )

        child = role_manager.create_role(
            "child",
            "子角色",
            permissions={"child:perm"},
            parent_roles={"parent"}
        )

        grandchild = role_manager.create_role(
            "grandchild",
            "孙子角色",
            permissions={"grandchild:perm"},
            parent_roles={"child"}
        )

        # 验证权限继承
        grandchild_perms = role_manager.get_role_permissions("grandchild")
        expected_perms = {
            "ancient:perm",    # 从祖父继承
            "parent:perm",     # 从父继承
            "child:perm",      # 从子继承
            "grandchild:perm"  # 自己的权限
        }
        assert grandchild_perms == expected_perms

        # 验证层次结构
        hierarchy = role_manager.get_role_hierarchy()
        assert hierarchy["grandchild"] == ["child"]
        assert hierarchy["child"] == ["parent"]
        assert hierarchy["parent"] == ["grandparent"]
        assert hierarchy["grandparent"] == []

    def test_role_validation_edge_cases(self, role_manager):
        """测试角色验证的边界情况"""
        # 测试空权限集合
        role1 = role_manager.create_role("empty_perms", "空权限", permissions=set())
        assert role_manager.get_role_permissions("empty_perms") == set()

        # 测试只有父角色的权限
        parent = role_manager.create_role("parent_only", "只有父角色", permissions={"parent:only"})
        child = role_manager.create_role("child_no_perms", "无权限子角色", parent_roles={"parent_only"})
        child_perms = role_manager.get_role_permissions("child_no_perms")
        assert child_perms == {"parent:only"}

        # 测试角色名和描述的边界情况
        long_name = "A" * 200  # 很长的名字
        role3 = role_manager.create_role("long_name", long_name)
        assert role3.name == long_name

        # 测试特殊字符
        special_name = "测试角色@#$%^&*()"
        role4 = role_manager.create_role("special_chars", special_name)
        assert role4.name == special_name
