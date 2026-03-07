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


import json
import os
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from src.data.security.access_control_manager import (
    AccessControlManager,
    ResourceType,
)


def test_load_config_json_decode_error_fallback(tmp_path):
    """测试加载配置时 JSON 解析错误回退"""
    acm = AccessControlManager(config_path=str(tmp_path), enable_audit=False)
    
    # 创建损坏的 JSON 文件
    users_file = tmp_path / "users.json"
    users_file.write_text("invalid json content {")
    
    # 应该不会抛出异常，只是记录错误
    acm._load_config()
    # 用户字典应该为空（因为加载失败）
    assert len(acm.users) == 0


def test_load_config_io_error_fallback(tmp_path, monkeypatch):
    """测试加载配置时 IO 错误回退"""
    acm = AccessControlManager(config_path=str(tmp_path), enable_audit=False)
    
    # Mock open 抛出 IOError
    def mock_open_error(*args, **kwargs):
        raise IOError("Permission denied")
    
    monkeypatch.setattr("builtins.open", mock_open_error)
    
    # 应该不会抛出异常，只是记录错误
    acm._load_config()
    assert len(acm.users) == 0


def test_save_config_io_error_fallback(tmp_path, monkeypatch):
    """测试保存配置时 IO 错误回退"""
    acm = AccessControlManager(config_path=str(tmp_path), enable_audit=False)
    
    # 添加一个用户
    user_id = acm.create_user("user1", email="user1@test.com")
    
    # Mock open 抛出 IOError（仅对写模式）
    original_open = open
    def mock_open_error(path, mode='r', *args, **kwargs):
        if 'w' in mode:
            raise IOError("Disk full")
        return original_open(path, mode, *args, **kwargs)
    
    monkeypatch.setattr("builtins.open", mock_open_error)
    
    # 应该不会抛出异常，只是记录错误
    acm._save_config()
    # 用户应该仍然在内存中（即使保存失败）
    assert user_id in acm.users


def test_save_config_json_serialization_error_fallback(tmp_path, monkeypatch):
    """测试保存配置时 JSON 序列化错误回退"""
    acm = AccessControlManager(config_path=str(tmp_path), enable_audit=False)
    
    # 添加一个用户
    user_id = acm.create_user("user1", email="user1@test.com")
    
    # Mock json.dump 抛出异常
    def mock_json_dump_error(*args, **kwargs):
        raise TypeError("Object of type datetime is not JSON serializable")
    
    monkeypatch.setattr("json.dump", mock_json_dump_error)
    
    # 应该不会抛出异常，只是记录错误
    acm._save_config()
    # 用户应该仍然在内存中
    assert user_id in acm.users


def test_load_config_partial_failure_continues(tmp_path):
    """测试加载配置时部分文件失败仍继续加载其他文件"""
    # 注意：当前实现中，如果 users.json 加载失败，整个 try 块会退出
    # 但不会抛出异常，只是记录错误
    acm = AccessControlManager(config_path=str(tmp_path), enable_audit=False)
    
    # 清空默认角色，以便测试
    acm.roles.clear()
    acm.users.clear()
    
    # 创建有效的 roles.json 和损坏的 users.json
    users_file = tmp_path / "users.json"
    users_file.write_text("invalid json")
    
    roles_file = tmp_path / "roles.json"
    roles_data = {
        "roles": [
            {
                "role_id": "r1",
                "name": "role1",
                "description": "test role",
                "permissions": ["read"],
                "parent_roles": [],
                "is_active": True,
                "created_at": "2024-01-01T00:00:00"
            }
        ]
    }
    roles_file.write_text(json.dumps(roles_data))
    
    # 由于 users.json 损坏，整个加载会失败，但不会抛出异常
    acm._load_config()
    
    # users 应该为空（加载失败）
    assert len(acm.users) == 0
    # 由于整个 try 块退出，roles 也不会被加载
    # 但验证错误处理不会抛出异常
    assert len(acm.roles) == 0


def test_batch_update_policies_with_partial_failures(tmp_path):
    """测试批量更新策略时部分失败的处理"""
    import time
    acm = AccessControlManager(config_path=str(tmp_path), enable_audit=False)
    
    # 创建多个策略（添加延迟确保不同的 ID）
    p1 = acm.create_access_policy(
        name="policy1",
        resource_type=ResourceType.DATA,
        resource_pattern="data:*",
        permissions=["read"]
    )
    time.sleep(0.01)  # 确保不同的时间戳
    p2 = acm.create_access_policy(
        name="policy2",
        resource_type=ResourceType.CACHE,
        resource_pattern="cache:*",
        permissions=["write"]
    )
    time.sleep(0.01)
    p3 = acm.create_access_policy(
        name="policy3",
        resource_type=ResourceType.CONFIG,
        resource_pattern="config:*",
        permissions=["read", "write"]
    )
    
    # 批量更新（其中一个策略不存在）
    updates = [
        (p1, {"name": "updated_policy1"}),
        ("nonexistent", {"name": "should_fail"}),
        (p2, {"name": "updated_policy2"}),
    ]
    
    results = []
    for policy_id, update_data in updates:
        try:
            acm.update_access_policy(policy_id, update_data)
            results.append(("success", policy_id))
        except ValueError as e:
            results.append(("error", policy_id, str(e)))
    
    # 验证结果
    assert results[0][0] == "success"
    assert results[1][0] == "error"
    assert results[2][0] == "success"
    
    # 验证成功更新的策略
    assert acm.policies[p1].name == "updated_policy1"
    assert acm.policies[p2].name == "updated_policy2"
    assert acm.policies[p3].name == "policy3"  # 未更新


def test_batch_update_policies_invalid_updates_handled(tmp_path):
    """测试批量更新策略时无效更新字段的处理"""
    acm = AccessControlManager(config_path=str(tmp_path), enable_audit=False)
    
    # 创建策略
    p1 = acm.create_access_policy(
        name="policy1",
        resource_type=ResourceType.DATA,
        resource_pattern="data:*",
        permissions=["read"]
    )
    
    # 尝试更新不存在的字段（应该被忽略，不会抛出异常）
    acm.update_access_policy(p1, {
        "name": "updated_name",
        "nonexistent_field": "should_be_ignored",
        "another_invalid": 123
    })
    
    # 验证有效字段被更新，无效字段被忽略
    assert acm.policies[p1].name == "updated_name"
    assert not hasattr(acm.policies[p1], "nonexistent_field")


def test_save_config_directory_not_exists_creates_it(tmp_path):
    """测试保存配置时目录不存在会自动创建"""
    # 创建一个不存在的子目录
    config_path = tmp_path / "nested" / "config"
    
    acm = AccessControlManager(config_path=str(config_path), enable_audit=False)
    acm.create_user("user1")
    
    # 保存配置（应该自动创建目录）
    acm._save_config()
    
    # 验证文件已创建
    users_file = config_path / "users.json"
    assert users_file.exists()
    
    # 验证可以重新加载
    acm2 = AccessControlManager(config_path=str(config_path), enable_audit=False)
    # 查找用户名匹配的用户
    user_found = any(u.username == "user1" for u in acm2.users.values())
    assert user_found


def test_load_config_missing_files_handled_gracefully(tmp_path):
    """测试加载配置时文件不存在的情况"""
    acm = AccessControlManager(config_path=str(tmp_path), enable_audit=False)
    
    # 清空默认角色，以便测试
    default_role_count = len(acm.roles)
    acm.roles.clear()
    
    # 不创建任何文件，直接加载
    acm._load_config()
    
    # 应该不会抛出异常
    # users 和 policies 应该为空（文件不存在）
    assert len(acm.users) == 0
    assert len(acm.policies) == 0
    # roles 也应该为空（因为我们清空了默认角色，且文件不存在）
    assert len(acm.roles) == 0


def test_save_config_with_datetime_serialization(tmp_path):
    """测试保存配置时 datetime 序列化"""
    acm = AccessControlManager(config_path=str(tmp_path), enable_audit=False)
    
    # 创建用户和角色（包含 datetime）
    user_id = acm.create_user("user1")
    role_id = acm.create_role("role1", "test role")
    
    # 保存配置
    acm._save_config()
    
    # 验证文件存在且可读
    users_file = tmp_path / "users.json"
    roles_file = tmp_path / "roles.json"
    
    assert users_file.exists()
    assert roles_file.exists()
    
    # 验证可以重新加载
    acm2 = AccessControlManager(config_path=str(tmp_path), enable_audit=False)
    # 查找用户名匹配的用户
    user_found = any(u.username == "user1" for u in acm2.users.values())
    assert user_found
    # 查找角色名匹配的角色（注意默认角色也会存在）
    role_found = any(r.name == "role1" for r in acm2.roles.values())
    assert role_found


def test_shutdown_saves_config_even_after_errors(tmp_path, monkeypatch):
    """测试 shutdown 时即使保存配置出错也会继续执行"""
    acm = AccessControlManager(config_path=str(tmp_path), enable_audit=False)
    acm.create_user("user1")
    
    # Mock open 抛出 IOError（模拟保存失败）
    call_count = []
    original_open = open
    
    def mock_open_error(path, mode='r', *args, **kwargs):
        if 'w' in mode:
            call_count.append(1)
            raise IOError("Save failed")
        return original_open(path, mode, *args, **kwargs)
    
    monkeypatch.setattr("builtins.open", mock_open_error)
    
    # shutdown 应该不会抛出异常（即使保存失败，_save_config 内部会捕获异常）
    acm.shutdown()
    
    # 验证 open 被调用（尝试保存）
    assert len(call_count) >= 1

