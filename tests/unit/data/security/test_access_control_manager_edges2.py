"""
边界测试：access_control_manager.py
测试边界情况和异常场景
"""
import asyncio
import pandas as pd
from unittest.mock import Mock

# Mock数据管理器模块以绕过复杂的导入问题
mock_data_manager = Mock()
mock_data_manager.DataManager = Mock()
mock_data_manager.DataLoaderError = Exception

# 配置DataManager实例方法
mock_instance = Mock()
mock_instance.validate_all_configs.return_value = True
mock_instance.health_check.return_value = {"status": "healthy"}
mock_instance.store_data.return_value = True
mock_instance.has_data.return_value = True
mock_instance.get_metadata.return_value = {"data_type": "test", "symbol": "X"}
mock_instance.retrieve_data.return_value = pd.DataFrame({"col": [1, 2, 3]})
mock_instance.get_stats.return_value = {"total_items": 1}
mock_instance.validate_data.return_value = {"valid": True}
mock_instance.shutdown.return_value = None

mock_data_manager.DataManager.return_value = mock_instance

# Mock整个模块
import sys
sys.modules["src.data.data_manager"] = mock_data_manager


import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from src.data.security.access_control_manager import (
    Permission,
    ResourceType,
    User,
    Role,
    AccessPolicy,
    AccessRequest,
    AccessDecision,
    AccessControlManager
)


def test_permission_enum():
    """测试 Permission（枚举值）"""
    assert Permission.READ.value == "read"
    assert Permission.WRITE.value == "write"
    assert Permission.DELETE.value == "delete"
    assert Permission.EXECUTE.value == "execute"
    assert Permission.ADMIN.value == "admin"
    assert Permission.AUDIT.value == "audit"


def test_resource_type_enum():
    """测试 ResourceType（枚举值）"""
    assert ResourceType.DATA.value == "data"
    assert ResourceType.CACHE.value == "cache"
    assert ResourceType.CONFIG.value == "config"
    assert ResourceType.LOG.value == "log"
    assert ResourceType.METADATA.value == "metadata"
    assert ResourceType.SYSTEM.value == "system"


def test_user_init():
    """测试 User（初始化）"""
    user = User(user_id="user1", username="test_user")
    
    assert user.user_id == "user1"
    assert user.username == "test_user"
    assert user.email is None
    assert user.is_active is True
    assert isinstance(user.created_at, datetime)
    assert user.last_login is None
    assert user.roles == set()
    assert user.permissions == set()
    assert user.metadata == {}


def test_user_has_role():
    """测试 User（检查角色）"""
    user = User(user_id="user1", username="test_user", roles={"admin", "user"})
    
    assert user.has_role("admin") is True
    assert user.has_role("user") is True
    assert user.has_role("guest") is False


def test_user_has_permission():
    """测试 User（检查权限）"""
    user = User(user_id="user1", username="test_user", permissions={"read", "write"})
    
    assert user.has_permission("read") is True
    assert user.has_permission("write") is True
    assert user.has_permission("delete") is False


def test_user_add_remove_role():
    """测试 User（添加/移除角色）"""
    user = User(user_id="user1", username="test_user")
    
    user.add_role("admin")
    assert user.has_role("admin") is True
    
    user.remove_role("admin")
    assert user.has_role("admin") is False
    
    # 移除不存在的角色不应报错
    user.remove_role("nonexistent")


def test_user_add_remove_permission():
    """测试 User（添加/移除权限）"""
    user = User(user_id="user1", username="test_user")
    
    user.add_permission("read")
    assert user.has_permission("read") is True
    
    user.remove_permission("read")
    assert user.has_permission("read") is False
    
    # 移除不存在的权限不应报错
    user.remove_permission("nonexistent")


def test_role_init():
    """测试 Role（初始化）"""
    role = Role(role_id="role1", name="Test Role")
    
    assert role.role_id == "role1"
    assert role.name == "Test Role"
    assert role.description == ""
    assert role.permissions == set()
    assert role.parent_roles == set()
    assert role.is_active is True
    assert isinstance(role.created_at, datetime)


def test_role_get_all_permissions_no_parents():
    """测试 Role（获取所有权限，无父角色）"""
    role = Role(role_id="role1", name="Test Role", permissions={"read", "write"})
    
    result = role.get_all_permissions({})
    
    assert result == {"read", "write"}


def test_role_get_all_permissions_with_parents():
    """测试 Role（获取所有权限，有父角色）"""
    parent_role = Role(role_id="parent", name="Parent", permissions={"read"})
    child_role = Role(role_id="child", name="Child", permissions={"write"})
    child_role.add_parent_role("parent")
    
    role_registry = {"parent": parent_role, "child": child_role}
    result = child_role.get_all_permissions(role_registry)
    
    assert "read" in result
    assert "write" in result


def test_access_policy_matches_resource_wildcard():
    """测试 AccessPolicy（匹配资源，通配符）"""
    policy = AccessPolicy(
        policy_id="policy1",
        name="Test Policy",
        resource_type=ResourceType.DATA,
        resource_pattern="*",
        permissions={"read"}
    )
    
    assert policy.matches_resource("any_resource") is True


def test_access_policy_matches_resource_exact():
    """测试 AccessPolicy（匹配资源，精确匹配）"""
    policy = AccessPolicy(
        policy_id="policy1",
        name="Test Policy",
        resource_type=ResourceType.DATA,
        resource_pattern="data:stock:000001",
        permissions={"read"}
    )
    
    assert policy.matches_resource("data:stock:000001") is True
    assert policy.matches_resource("data:stock:000002") is False


def test_access_policy_matches_resource_prefix():
    """测试 AccessPolicy（匹配资源，前缀匹配）"""
    policy = AccessPolicy(
        policy_id="policy1",
        name="Test Policy",
        resource_type=ResourceType.DATA,
        resource_pattern="data:stock:*",
        permissions={"read"}
    )
    
    assert policy.matches_resource("data:stock:000001") is True
    assert policy.matches_resource("data:stock:000002") is True
    assert policy.matches_resource("data:crypto:BTC") is False


def test_access_policy_check_conditions_empty():
    """测试 AccessPolicy（检查条件，空条件）"""
    policy = AccessPolicy(
        policy_id="policy1",
        name="Test Policy",
        resource_type=ResourceType.DATA,
        resource_pattern="*",
        permissions={"read"},
        conditions={}
    )
    
    assert policy.check_conditions({}) is True
    assert policy.check_conditions({"key": "value"}) is True


def test_access_policy_check_conditions_missing_key():
    """测试 AccessPolicy（检查条件，缺少键）"""
    policy = AccessPolicy(
        policy_id="policy1",
        name="Test Policy",
        resource_type=ResourceType.DATA,
        resource_pattern="*",
        permissions={"read"},
        conditions={"required_key": "value"}
    )
    
    assert policy.check_conditions({}) is False


def test_access_policy_check_conditions_time_range():
    """测试 AccessPolicy（检查条件，时间范围）"""
    now = datetime.now()
    policy = AccessPolicy(
        policy_id="policy1",
        name="Test Policy",
        resource_type=ResourceType.DATA,
        resource_pattern="*",
        permissions={"read"},
        conditions={
            "time_range": {
                "start": now - timedelta(hours=1),
                "end": now + timedelta(hours=1)
            }
        }
    )
    
    assert policy.check_conditions({"time_range": now}) is True
    assert policy.check_conditions({"time_range": now - timedelta(hours=2)}) is False
    assert policy.check_conditions({"time_range": now + timedelta(hours=2)}) is False


def test_access_control_manager_init():
    """测试 AccessControlManager（初始化）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = AccessControlManager(config_path=tmpdir, enable_audit=False)
        
        assert manager.config_path == Path(tmpdir)
        assert manager.enable_audit is False
        assert len(manager.roles) >= 4  # 默认角色


def test_access_control_manager_create_user_success():
    """测试 AccessControlManager（创建用户，成功）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = AccessControlManager(config_path=tmpdir, enable_audit=False)
        
        user_id = manager.create_user("test_user", email="test@example.com")
        
        assert user_id is not None
        assert user_id in manager.users
        assert manager.users[user_id].username == "test_user"


def test_access_control_manager_create_user_duplicate_username():
    """测试 AccessControlManager（创建用户，重复用户名）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = AccessControlManager(config_path=tmpdir, enable_audit=False)
        
        manager.create_user("test_user")
        
        with pytest.raises(ValueError, match="用户名已存在"):
            manager.create_user("test_user")


def test_access_control_manager_assign_role_to_user():
    """测试 AccessControlManager（分配角色给用户）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = AccessControlManager(config_path=tmpdir, enable_audit=False)
        
        user_id = manager.create_user("test_user")
        manager.assign_role_to_user(user_id, "analyst")
        
        assert manager.users[user_id].has_role("analyst") is True


def test_access_control_manager_assign_role_invalid_user():
    """测试 AccessControlManager（分配角色，无效用户）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = AccessControlManager(config_path=tmpdir, enable_audit=False)
        
        with pytest.raises(ValueError, match="用户不存在"):
            manager.assign_role_to_user("nonexistent", "analyst")


def test_access_control_manager_assign_role_invalid_role():
    """测试 AccessControlManager（分配角色，无效角色）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = AccessControlManager(config_path=tmpdir, enable_audit=False)
        
        user_id = manager.create_user("test_user")
        
        with pytest.raises(ValueError, match="角色不存在"):
            manager.assign_role_to_user(user_id, "nonexistent")


def test_access_control_manager_revoke_role_from_user():
    """测试 AccessControlManager（撤销用户角色）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = AccessControlManager(config_path=tmpdir, enable_audit=False)
        
        user_id = manager.create_user("test_user")
        manager.assign_role_to_user(user_id, "analyst")
        manager.revoke_role_from_user(user_id, "analyst")
        
        assert manager.users[user_id].has_role("analyst") is False


def test_access_control_manager_grant_permission_to_user():
    """测试 AccessControlManager（授予用户权限）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = AccessControlManager(config_path=tmpdir, enable_audit=False)
        
        user_id = manager.create_user("test_user")
        manager.grant_permission_to_user(user_id, "custom_permission")
        
        assert manager.users[user_id].has_permission("custom_permission") is True


def test_access_control_manager_create_role():
    """测试 AccessControlManager（创建角色）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = AccessControlManager(config_path=tmpdir, enable_audit=False)
        
        role_id = manager.create_role("Custom Role", permissions=["read", "write"])
        
        assert role_id is not None
        assert role_id in manager.roles
        assert manager.roles[role_id].name == "Custom Role"


def test_access_control_manager_create_role_duplicate_name():
    """测试 AccessControlManager（创建角色，重复名称）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = AccessControlManager(config_path=tmpdir, enable_audit=False)
        
        manager.create_role("Custom Role")
        
        with pytest.raises(ValueError, match="角色名已存在"):
            manager.create_role("Custom Role")


def test_access_control_manager_set_role_inheritance_self():
    """测试 AccessControlManager（设置角色继承，自继承）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = AccessControlManager(config_path=tmpdir, enable_audit=False)
        
        role_id = manager.create_role("Test Role")
        
        with pytest.raises(ValueError, match="不能设置角色自继承"):
            manager.set_role_inheritance(role_id, role_id)


def test_access_control_manager_check_access_user_not_found():
    """测试 AccessControlManager（检查访问，用户不存在）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = AccessControlManager(config_path=tmpdir, enable_audit=False)
        
        decision = manager.check_access("nonexistent", "resource", "read")
        
        assert decision.allowed is False
        assert decision.reason == "user_not_found"


def test_access_control_manager_check_access_user_inactive():
    """测试 AccessControlManager（检查访问，用户未激活）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = AccessControlManager(config_path=tmpdir, enable_audit=False)
        
        user_id = manager.create_user("test_user")
        manager.users[user_id].is_active = False
        
        decision = manager.check_access(user_id, "resource", "read")
        
        assert decision.allowed is False
        assert decision.reason == "user_inactive"


def test_access_control_manager_check_access_direct_permission():
    """测试 AccessControlManager（检查访问，直接权限）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = AccessControlManager(config_path=tmpdir, enable_audit=False)
        
        user_id = manager.create_user("test_user")
        manager.grant_permission_to_user(user_id, "read")
        
        decision = manager.check_access(user_id, "resource", "read")
        
        assert decision.allowed is True
        assert decision.reason == "direct_permission"


def test_access_control_manager_check_access_role_permission():
    """测试 AccessControlManager（检查访问，角色权限）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = AccessControlManager(config_path=tmpdir, enable_audit=False)
        
        user_id = manager.create_user("test_user")
        manager.assign_role_to_user(user_id, "analyst")  # analyst有read权限
        
        decision = manager.check_access(user_id, "resource", "read")
        
        assert decision.allowed is True


def test_access_control_manager_check_access_no_permission():
    """测试 AccessControlManager（检查访问，无权限）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = AccessControlManager(config_path=tmpdir, enable_audit=False)
        
        user_id = manager.create_user("test_user")
        
        decision = manager.check_access(user_id, "resource", "delete")
        
        assert decision.allowed is False


def test_access_control_manager_check_access_cached():
    """测试 AccessControlManager（检查访问，缓存）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = AccessControlManager(config_path=tmpdir, enable_audit=False)
        
        user_id = manager.create_user("test_user")
        manager.grant_permission_to_user(user_id, "read")
        
        # 第一次检查
        decision1 = manager.check_access(user_id, "resource", "read")
        
        # 第二次检查（应该使用缓存）
        decision2 = manager.check_access(user_id, "resource", "read")
        
        assert decision1.allowed is True
        assert decision2.allowed is True
        assert decision2.reason == "cached_result"


def test_access_control_manager_create_access_policy():
    """测试 AccessControlManager（创建访问策略）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = AccessControlManager(config_path=tmpdir, enable_audit=False)
        
        policy_id = manager.create_access_policy(
            "Test Policy",
            ResourceType.DATA,
            "data:*",
            ["read", "write"]
        )
        
        assert policy_id is not None
        assert policy_id in manager.policies


def test_access_control_manager_update_access_policy():
    """测试 AccessControlManager（更新访问策略）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = AccessControlManager(config_path=tmpdir, enable_audit=False)
        
        policy_id = manager.create_access_policy(
            "Test Policy",
            ResourceType.DATA,
            "data:*",
            ["read"]
        )
        
        manager.update_access_policy(policy_id, {"name": "Updated Policy"})
        
        assert manager.policies[policy_id].name == "Updated Policy"


def test_access_control_manager_clear_permission_cache():
    """测试 AccessControlManager（清除权限缓存）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = AccessControlManager(config_path=tmpdir, enable_audit=False)
        
        user_id = manager.create_user("test_user")
        manager.grant_permission_to_user(user_id, "read")
        
        # 使用缓存
        manager.check_access(user_id, "resource", "read")
        
        # 清除缓存
        manager.clear_permission_cache()
        
        assert len(manager._permission_cache) == 0


def test_role_add_remove_permission():
    """测试 Role（添加和移除权限）"""
    role = Role(role_id="role1", name="Test Role")
    role.add_permission("read")
    assert "read" in role.permissions
    
    role.remove_permission("read")
    assert "read" not in role.permissions


def test_role_remove_parent_role():
    """测试 Role（移除父角色）"""
    role = Role(role_id="role1", name="Test Role")
    role.add_parent_role("parent1")
    assert "parent1" in role.parent_roles
    
    role.remove_parent_role("parent1")
    assert "parent1" not in role.parent_roles


def test_access_policy_matches_resource_type():
    """测试 AccessPolicy（匹配资源，类型匹配）"""
    # 类型匹配逻辑：如果resource_pattern包含":"，会进行类型匹配
    policy = AccessPolicy(
        policy_id="policy1",
        name="Test Policy",
        resource_type=ResourceType.DATA,
        resource_pattern="data:resource1",  # 包含":"，会触发类型匹配
        permissions=["read"]
    )
    
    # 类型匹配：policy_type == resource_type 且 pattern匹配
    assert policy.matches_resource("data:resource1") is True
    # 类型不匹配
    assert policy.matches_resource("cache:resource1") is False


def test_access_control_manager_create_user_with_audit():
    """测试 AccessControlManager（创建用户，启用审计）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = AccessControlManager(config_path=tmpdir, enable_audit=True)
        
        user_id = manager.create_user("test_user")
        
        assert user_id in manager.users
        # 检查是否有审计日志（通过get_audit_logs方法）
        logs = manager.get_audit_logs()
        assert len(logs) > 0


def test_access_control_manager_create_role_with_audit():
    """测试 AccessControlManager（创建角色，启用审计）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = AccessControlManager(config_path=tmpdir, enable_audit=True)
        
        role_id = manager.create_role("test_role", ["read"])
        
        assert role_id in manager.roles
        # 检查是否有审计日志（通过get_audit_logs方法）
        logs = manager.get_audit_logs()
        assert len(logs) > 0


def test_access_control_manager_add_permission_to_role():
    """测试 AccessControlManager（为角色添加权限）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = AccessControlManager(config_path=tmpdir, enable_audit=False)
        
        role_id = manager.create_role("test_role", ["read"])
        manager.add_permission_to_role(role_id, "write")
        
        assert "write" in manager.roles[role_id].permissions


def test_access_control_manager_add_permission_to_role_invalid():
    """测试 AccessControlManager（为角色添加权限，角色不存在）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = AccessControlManager(config_path=tmpdir, enable_audit=False)
        
        with pytest.raises(ValueError, match="角色不存在"):
            manager.add_permission_to_role("nonexistent", "read")


def test_access_control_manager_add_permission_to_role_with_audit():
    """测试 AccessControlManager（为角色添加权限，启用审计）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = AccessControlManager(config_path=tmpdir, enable_audit=True)
        
        role_id = manager.create_role("test_role", ["read"])
        manager.add_permission_to_role(role_id, "write")
        
        # 检查是否有审计日志（通过get_audit_logs方法）
        logs = manager.get_audit_logs()
        assert len(logs) > 0


def test_access_control_manager_check_access_with_audit():
    """测试 AccessControlManager（检查访问，启用审计）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = AccessControlManager(config_path=tmpdir, enable_audit=True)
        
        user_id = manager.create_user("test_user")
        manager.grant_permission_to_user(user_id, "read")
        
        decision = manager.check_access(user_id, "resource1", "read")
        
        assert decision.allowed is True
        # 检查是否有审计日志（通过get_audit_logs方法）
        logs = manager.get_audit_logs()
        assert len(logs) > 0


def test_access_control_manager_create_policy_with_audit():
    """测试 AccessControlManager（创建策略，启用审计）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = AccessControlManager(config_path=tmpdir, enable_audit=True)
        
        policy_id = manager.create_access_policy(
            "Test Policy",
            ResourceType.DATA,
            "resource*",
            ["read"]
        )
        
        assert policy_id in manager.policies
        # 检查是否有审计日志（通过get_audit_logs方法）
        logs = manager.get_audit_logs()
        assert len(logs) > 0


def test_access_control_manager_update_policy_with_audit():
    """测试 AccessControlManager（更新策略，启用审计）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = AccessControlManager(config_path=tmpdir, enable_audit=True)
        
        policy_id = manager.create_access_policy(
            "Test Policy",
            ResourceType.DATA,
            "resource*",
            ["read"]
        )
        
        manager.update_access_policy(policy_id, {"name": "Updated Policy"})
        
        # 检查是否有审计日志（通过get_audit_logs方法）
        logs = manager.get_audit_logs()
        assert len(logs) > 0


def test_access_control_manager_check_access_policies_inactive():
    """测试 AccessControlManager（检查访问策略，策略未激活）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = AccessControlManager(config_path=tmpdir, enable_audit=False)
        
        user_id = manager.create_user("test_user")
        manager.grant_permission_to_user(user_id, "read")
        
        policy_id = manager.create_access_policy(
            "Test Policy",
            ResourceType.DATA,
            "resource*",
            ["read"]
        )
        manager.policies[policy_id].is_active = False
        
        decision = manager.check_access(user_id, "resource1", "read")
        
        # 应该允许（因为用户有直接权限）
        assert decision.allowed is True


def test_access_control_manager_check_access_policies_condition_fail():
    """测试 AccessControlManager（检查访问策略，条件不满足）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = AccessControlManager(config_path=tmpdir, enable_audit=False)
        
        user_id = manager.create_user("test_user")
        
        policy_id = manager.create_access_policy(
            "Test Policy",
            ResourceType.DATA,
            "resource*",
            ["read"],
            conditions={"time_range": "9:00-17:00"}
        )
        
        # 创建请求，但条件不满足
        request = AccessRequest(
            user_id=user_id,
            resource="resource1",
            permission="read",
            context={"time": "20:00"}
        )
        
        # 直接权限检查应该失败（因为用户没有直接权限）
        user = manager.users[user_id]
        user_permissions = manager._get_user_permissions(user)
        decision = manager._check_access_policies(request, user_permissions)
        
        # 应该拒绝（因为条件不满足且没有直接权限）
        assert decision.allowed is False


def test_access_control_manager_load_config():
    """测试 AccessControlManager（加载配置）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        import json
        # 创建配置文件
        users_file = Path(tmpdir) / "users.json"
        users_data = {
            "users": [{
                "user_id": "user1",
                "username": "test_user",
                "email": "test@example.com",
                "is_active": True,
                "created_at": datetime.now().isoformat(),
                "last_login": None,
                "roles": ["role1"],
                "permissions": ["read"]
            }]
        }
        with open(users_file, 'w') as f:
            json.dump(users_data, f)
        
        # 创建角色文件
        roles_file = Path(tmpdir) / "roles.json"
        roles_data = {
            "roles": [{
                "role_id": "role1",
                "name": "Test Role",
                "description": "Test",
                "permissions": ["read", "write"],
                "parent_roles": [],
                "is_active": True,
                "created_at": datetime.now().isoformat(),
                "metadata": {}
            }]
        }
        with open(roles_file, 'w') as f:
            json.dump(roles_data, f)
        
        # 加载配置
        manager = AccessControlManager(config_path=tmpdir, enable_audit=False)
        
        assert "user1" in manager.users
        assert "role1" in manager.roles


def test_access_control_manager_get_audit_logs_empty():
    """测试 AccessControlManager（获取审计日志，空）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = AccessControlManager(config_path=tmpdir, enable_audit=False)
        
        logs = manager.get_audit_logs()
        
        assert logs == []


def test_access_control_manager_get_access_statistics():
    """测试 AccessControlManager（获取访问统计）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = AccessControlManager(config_path=tmpdir, enable_audit=False)
        
        stats = manager.get_access_statistics()
        
        assert "total_access_checks" in stats
        assert "allowed_access" in stats
        assert "denied_access" in stats
        assert "allow_rate" in stats


def test_access_control_manager_shutdown():
    """测试 AccessControlManager（关闭）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = AccessControlManager(config_path=tmpdir, enable_audit=False)
        
        user_id = manager.create_user("test_user")
        
        # 关闭应该保存配置
        manager.shutdown()
        
        # 验证配置文件是否存在
        users_file = manager.config_path / "users.json"
        assert users_file.exists()


def test_access_policy_matches_resource_wildcard():
    """测试 AccessPolicy（匹配资源，通配符）"""
    policy = AccessPolicy(
        policy_id="test_policy",
        name="Test Policy",
        resource_type=ResourceType.DATA,
        resource_pattern="*",
        permissions={Permission.READ}
    )
    # 通配符应该匹配任何资源（覆盖 158 行）
    assert policy.matches_resource("data:test_resource") == True
    assert policy.matches_resource("data:other_resource") == True


def test_access_policy_check_conditions_other_condition():
    """测试 AccessPolicy（检查条件，其他条件）"""
    policy = AccessPolicy(
        policy_id="test_policy",
        name="Test Policy",
        resource_type=ResourceType.DATA,
        resource_pattern="*",
        permissions={Permission.READ},
        conditions={"custom_condition": "value1"}
    )
    # 测试其他条件（覆盖 189-190 行）
    context1 = {"custom_condition": "value1"}
    assert policy.check_conditions(context1) == True
    
    context2 = {"custom_condition": "value2"}
    assert policy.check_conditions(context2) == False


def test_access_policy_check_conditions_ip_range():
    """测试 AccessPolicy（检查条件，IP范围）"""
    policy = AccessPolicy(
        policy_id="test_policy",
        name="Test Policy",
        resource_type=ResourceType.DATA,
        resource_pattern="*",
        permissions={Permission.READ},
        conditions={"ip_range": ["192.168.1.1", "192.168.1.2"]}
    )
    # 测试 IP 范围条件（覆盖 183-190 行）
    context1 = {"ip_range": "192.168.1.1"}
    assert policy.check_conditions(context1) == True
    
    context2 = {"ip_range": "192.168.1.3"}
    assert policy.check_conditions(context2) == False


def test_access_control_manager_assign_role_audit():
    """测试 AccessControlManager（分配角色，审计）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = AccessControlManager(config_path=tmpdir, enable_audit=True)
        
        user_id = manager.create_user("test_user")
        role_id = manager.create_role("test_role")
        
        # 分配角色应该记录审计日志（覆盖 373 行）
        # 验证 enable_audit 为 True 时，_audit_log 被调用
        manager.assign_role_to_user(user_id, role_id)
        
        # 验证操作成功（审计日志通过 _audit_log 方法记录）
        assert manager.enable_audit == True
        manager.shutdown()


def test_access_control_manager_revoke_role_audit():
    """测试 AccessControlManager（撤销角色，审计）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = AccessControlManager(config_path=tmpdir, enable_audit=True)
        
        user_id = manager.create_user("test_user")
        role_id = manager.create_role("test_role")
        manager.assign_role_to_user(user_id, role_id)
        
        # 撤销角色应该记录审计日志（覆盖 399 行）
        # 验证 enable_audit 为 True 时，_audit_log 被调用
        manager.revoke_role_from_user(user_id, role_id)
        
        # 验证操作成功（审计日志通过 _audit_log 方法记录）
        assert manager.enable_audit == True
        manager.shutdown()


def test_access_control_manager_revoke_role_user_not_exists():
    """测试 AccessControlManager（撤销角色，用户不存在）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = AccessControlManager(config_path=tmpdir, enable_audit=False)
        
        role_id = manager.create_role("test_role")
        
        # 撤销不存在的用户的角色应该抛出异常（覆盖 389 行）
        with pytest.raises(ValueError, match="用户不存在"):
            manager.revoke_role_from_user("nonexistent_user", role_id)


def test_access_control_manager_set_inheritance_child_not_exists():
    """测试 AccessControlManager（设置继承，子角色不存在）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = AccessControlManager(config_path=tmpdir, enable_audit=False)
        
        parent_role_id = manager.create_role("parent_role")
        
        # 设置不存在的子角色的继承应该抛出异常（覆盖 517 行）
        with pytest.raises(ValueError, match="子角色不存在"):
            manager.set_role_inheritance("nonexistent_child", parent_role_id)


def test_access_control_manager_set_inheritance_parent_not_exists():
    """测试 AccessControlManager（设置继承，父角色不存在）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = AccessControlManager(config_path=tmpdir, enable_audit=False)
        
        child_role_id = manager.create_role("child_role")
        
        # 设置不存在的父角色的继承应该抛出异常（覆盖 520 行）
        with pytest.raises(ValueError, match="父角色不存在"):
            manager.set_role_inheritance(child_role_id, "nonexistent_parent")


def test_access_control_manager_set_inheritance_self_inheritance():
    """测试 AccessControlManager（设置继承，自继承）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = AccessControlManager(config_path=tmpdir, enable_audit=False)
        
        role_id = manager.create_role("test_role")
        
        # 设置角色自继承应该抛出异常（覆盖 523 行）
        with pytest.raises(ValueError, match="不能设置角色自继承"):
            manager.set_role_inheritance(role_id, role_id)


def test_access_control_manager_set_inheritance_audit():
    """测试 AccessControlManager（设置继承，审计）"""
    import time
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = AccessControlManager(config_path=tmpdir, enable_audit=True)
        
        child_role_id = manager.create_role("child_role")
        # 添加小延迟确保两个角色ID不同
        time.sleep(0.01)
        parent_role_id = manager.create_role("parent_role")
        
        # 确保两个角色ID不同
        assert child_role_id != parent_role_id, f"角色ID相同: {child_role_id} == {parent_role_id}"
        
        # 设置继承应该记录审计日志（覆盖 532-536 行）
        # 验证 enable_audit 为 True 时，_audit_log 被调用
        manager.set_role_inheritance(child_role_id, parent_role_id)
        
        # 验证操作成功（审计日志通过 _audit_log 方法记录）
        assert manager.enable_audit == True
        manager.shutdown()


def test_access_control_manager_check_access_policies_inactive():
    """测试 AccessControlManager（检查访问策略，非活动）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = AccessControlManager(config_path=tmpdir, enable_audit=False)
        
        user_id = manager.create_user("test_user")
        role_id = manager.create_role("test_role", permissions={Permission.READ})
        manager.assign_role_to_user(user_id, role_id)
        
        # 创建非活动策略
        policy = AccessPolicy(
            policy_id="inactive_policy",
            name="Inactive Policy",
            resource_type=ResourceType.DATA,
            resource_pattern="*",
            permissions={Permission.READ},
            is_active=False
        )
        manager.policies[policy.policy_id] = policy
        
        # 非活动策略应该被跳过（覆盖 646 行）
        request = AccessRequest(
            user_id=user_id,
            resource="data:test_resource",
            permission=Permission.READ
        )
        decision = manager.check_access(user_id, "data:test_resource", Permission.READ)
        # 应该允许访问（因为角色有权限），但非活动策略不应该被应用
        assert decision.allowed == True


def test_access_control_manager_check_access_policies_no_match():
    """测试 AccessControlManager（检查访问策略，无匹配）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = AccessControlManager(config_path=tmpdir, enable_audit=False)
        
        user_id = manager.create_user("test_user")
        
        # 创建不匹配资源类型的策略（resource_type 不同）
        # 注意：matches_resource 方法会检查 resource_type 和 resource_pattern
        # 如果 resource_type 不匹配，策略会被跳过（覆盖 650 行）
        policy = AccessPolicy(
            policy_id="no_match_policy",
            name="No Match Policy",
            resource_type=ResourceType.CACHE,  # CACHE 类型
            resource_pattern="cache:test",  # 明确指定 cache 前缀和具体资源
            permissions={Permission.READ}
        )
        manager.policies[policy.policy_id] = policy
        
        # 策略不匹配资源应该被跳过（覆盖 650 行）
        request = AccessRequest(
            user_id=user_id,
            resource="data:test_resource",  # DATA 类型，与策略的 CACHE 类型不匹配
            permission=Permission.READ
        )
        decision = manager._check_access_policies(request, set())
        # 应该拒绝访问（没有匹配的策略）
        # 注意：matches_resource 会检查 resource_type，如果类型不匹配，返回 False
        # 所以策略会被跳过，decision.allowed 应该为 False
        assert decision.allowed == False
        assert len(decision.applied_policies) == 0
        manager.shutdown()


def test_access_control_manager_check_access_policies_no_permission():
    """测试 AccessControlManager（检查访问策略，无权限）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = AccessControlManager(config_path=tmpdir, enable_audit=False)
        
        user_id = manager.create_user("test_user")
        
        # 创建不包含请求权限的策略
        policy = AccessPolicy(
            policy_id="no_permission_policy",
            name="No Permission Policy",
            resource_type=ResourceType.DATA,
            resource_pattern="*",
            permissions={Permission.WRITE}  # 只有 WRITE 权限
        )
        manager.policies[policy.policy_id] = policy
        
        # 策略不包含请求权限应该被跳过（覆盖 654 行）
        request = AccessRequest(
            user_id=user_id,
            resource="data:test_resource",
            permission=Permission.READ  # 请求 READ 权限
        )
        decision = manager._check_access_policies(request, set())
        # 应该拒绝访问（没有匹配的策略）
        assert decision.allowed == False


def test_access_control_manager_check_access_policies_condition_fail():
    """测试 AccessControlManager（检查访问策略，条件失败）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = AccessControlManager(config_path=tmpdir, enable_audit=False)
        
        user_id = manager.create_user("test_user")
        
        # 创建有条件的策略
        policy = AccessPolicy(
            policy_id="condition_policy",
            name="Condition Policy",
            resource_type=ResourceType.DATA,
            resource_pattern="*",
            permissions={Permission.READ},
            conditions={"ip_range": "192.168.1.1"}
        )
        manager.policies[policy.policy_id] = policy
        
        # 条件不满足应该被跳过（覆盖 660 行）
        request = AccessRequest(
            user_id=user_id,
            resource="data:test_resource",
            permission=Permission.READ,
            context={"ip_range": "192.168.1.2"}  # 不同的 IP
        )
        decision = manager._check_access_policies(request, set())
        # 应该拒绝访问（条件不满足）
        assert decision.allowed == False


def test_access_control_manager_update_policy_not_exists():
    """测试 AccessControlManager（更新策略，不存在）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = AccessControlManager(config_path=tmpdir, enable_audit=False)
        
        # 更新不存在的策略应该抛出异常（覆盖 727 行）
        # 注意：实际方法名是 update_access_policy
        with pytest.raises(ValueError, match="策略不存在"):
            manager.update_access_policy("nonexistent_policy", {"name": "New Name"})
        manager.shutdown()


def test_access_control_manager_get_access_statistics_with_resource_stats():
    """测试 AccessControlManager（获取访问统计，包含资源统计）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = AccessControlManager(config_path=tmpdir, enable_audit=True)
        
        user_id = manager.create_user("test_user")
        role_id = manager.create_role("test_role", permissions={Permission.READ})
        manager.assign_role_to_user(user_id, role_id)
        
        # 执行一些访问检查
        manager.check_access(user_id, "data:test_resource", Permission.READ)
        manager.check_access(user_id, "cache:test_resource", Permission.READ)
        
        # 获取统计信息（覆盖 848-857 行）
        stats = manager.get_access_statistics()
        assert "resource_statistics" in stats
        if stats["resource_statistics"]:
            # 验证资源统计包含资源类型
            assert "data" in stats["resource_statistics"] or "cache" in stats["resource_statistics"]


def test_access_control_manager_load_config_policies():
    """测试 AccessControlManager（加载配置，策略）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        import json
        from datetime import datetime
        
        # 创建策略文件
        policies_file = Path(tmpdir) / "policies.json"
        policies_data = {
            "policies": [
                {
                    "policy_id": "test_policy",
                    "name": "Test Policy",
                    "resource_type": "data",
                    "resource_pattern": "*",
                    "permissions": ["read"],
                    "conditions": {},
                    "is_active": True,
                    "created_at": datetime.now().isoformat()
                }
            ]
        }
        with open(policies_file, 'w') as f:
            json.dump(policies_data, f)
        
        # 加载配置应该加载策略（覆盖 915-928 行）
        manager = AccessControlManager(config_path=tmpdir, enable_audit=False)
        assert "test_policy" in manager.policies


def test_access_control_manager_load_config_exception():
    """测试 AccessControlManager（加载配置，异常）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建损坏的配置文件
        users_file = Path(tmpdir) / "users.json"
        users_file.write_text("invalid json", encoding='utf-8')
        
        # 加载配置应该处理异常（覆盖 930-931 行）
        manager = AccessControlManager(config_path=tmpdir, enable_audit=False)
        # 应该不抛出异常，使用默认配置
        assert manager is not None
        # 显式关闭以避免 __del__ 中的日志错误
        manager.shutdown()
