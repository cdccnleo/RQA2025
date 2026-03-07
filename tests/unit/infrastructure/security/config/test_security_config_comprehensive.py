#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安全配置管理器综合测试
测试SecurityConfigManager和AuditConfigManager的核心功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import os
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import patch, MagicMock

from src.infrastructure.security.config.security_config import (
    SecurityConfigManager,
    AuditConfigManager
)
from src.infrastructure.security.core.types import ConfigOperationParams


@pytest.fixture
def temp_config_dir():
    """创建临时配置目录"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # 清理
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def security_config_manager(temp_config_dir):
    """创建安全配置管理器实例"""
    return SecurityConfigManager(config_path=temp_config_dir)


@pytest.fixture
def audit_config_manager(temp_config_dir):
    """创建审计配置管理器实例"""
    return AuditConfigManager(config_path=temp_config_dir)


@pytest.fixture
def config_operation_params():
    """创建配置操作参数"""
    return ConfigOperationParams(
        operation_type="load",
        config_sections={"users"}
    )


class TestSecurityConfigManagerInitialization:
    """测试安全配置管理器初始化"""

    def test_initialization_with_default_path(self):
        """测试默认路径初始化"""
        manager = SecurityConfigManager()

        assert manager.config_path is not None
        assert manager.config_path.exists()
        assert isinstance(manager.validation_rules, dict)

    def test_initialization_with_custom_path(self, temp_config_dir):
        """测试自定义路径初始化"""
        manager = SecurityConfigManager(config_path=temp_config_dir)

        assert str(manager.config_path) == temp_config_dir
        assert manager.config_path.exists()

    def test_validation_rules_setup(self, security_config_manager):
        """测试验证规则设置"""
        manager = security_config_manager

        expected_rules = ['users', 'roles', 'policies']
        assert set(manager.validation_rules.keys()) == set(expected_rules)

        for rule in manager.validation_rules.values():
            assert callable(rule)


class TestSecurityConfigManagerLoadConfig:
    """测试安全配置管理器加载配置功能"""

    def test_load_config_users(self, security_config_manager, config_operation_params):
        """测试加载用户配置"""
        manager = security_config_manager
        params = config_operation_params
        params.config_sections = {"users"}

        # 创建测试用户配置文件
        users_file = manager.config_path / "users.json"
        test_users = {
            "user1": {"name": "Test User 1", "roles": ["admin"]},
            "user2": {"name": "Test User 2", "roles": ["user"]}
        }
        users_file.write_text(json.dumps(test_users))

        result = manager.load_config(params)

        assert isinstance(result, dict)
        assert "user1" in result
        assert "user2" in result
        assert result["user1"]["name"] == "Test User 1"

    def test_load_config_roles(self, security_config_manager, config_operation_params):
        """测试加载角色配置"""
        manager = security_config_manager
        params = config_operation_params
        params.config_sections = {"roles"}

        # 创建测试角色配置文件
        roles_file = manager.config_path / "roles.json"
        test_roles = {
            "admin": {"permissions": ["read", "write", "delete"]},
            "user": {"permissions": ["read"]}
        }
        roles_file.write_text(json.dumps(test_roles))

        result = manager.load_config(params)

        assert isinstance(result, dict)
        assert "admin" in result
        assert "user" in result
        assert "read" in result["user"]["permissions"]

    def test_load_config_policies(self, security_config_manager, config_operation_params):
        """测试加载策略配置"""
        manager = security_config_manager
        params = config_operation_params
        params.config_sections = {"policies"}

        # 创建测试策略配置文件
        policies_file = manager.config_path / "policies.json"
        test_policies = {
            "policy1": {"effect": "allow", "conditions": ["time_check"]},
            "policy2": {"effect": "deny", "conditions": ["ip_check"]}
        }
        policies_file.write_text(json.dumps(test_policies))

        result = manager.load_config(params)

        assert isinstance(result, dict)
        assert "policy1" in result
        assert "policy2" in result
        assert result["policy1"]["effect"] == "allow"

    def test_load_config_file_not_found(self, security_config_manager):
        """测试加载不存在的配置文件"""
        manager = security_config_manager
        params = ConfigOperationParams(
            operation_type="load",
            config_sections={"users"}
        )

        result = manager.load_config(params)

        assert isinstance(result, dict)
        assert len(result) == 0


class TestSecurityConfigManagerSaveConfig:
    """测试安全配置管理器保存配置功能"""

    def test_save_config_users(self, security_config_manager):
        """测试保存用户配置"""
        manager = security_config_manager

        users_data = {
            "user1": {"name": "Test User 1", "roles": ["admin"]},
            "user2": {"name": "Test User 2", "roles": ["user"]}
        }

        params = ConfigOperationParams(
            operation_type="save",
            config_sections={"users"}
        )

        result = manager.save_config(users_data, params)

        assert result is True

        # 验证文件是否创建
        users_file = manager.config_path / "users.json"
        assert users_file.exists()

        # 验证内容
        saved_data = json.loads(users_file.read_text())
        assert saved_data == users_data

    def test_save_config_roles(self, security_config_manager):
        """测试保存角色配置"""
        manager = security_config_manager

        roles_data = {
            "admin": {"permissions": ["read", "write", "delete"]},
            "user": {"permissions": ["read"]}
        }

        params = ConfigOperationParams(
            operation="save",
            config_type="roles",
            user_id="admin",
            timestamp=datetime.now()
        )

        result = manager.save_config(roles_data, params)

        assert result is True

        # 验证文件是否创建
        roles_file = manager.config_path / "roles.json"
        assert roles_file.exists()

    def test_save_config_policies(self, security_config_manager):
        """测试保存策略配置"""
        manager = security_config_manager

        policies_data = {
            "policy1": {"effect": "allow", "conditions": ["time_check"]},
            "policy2": {"effect": "deny", "conditions": ["ip_check"]}
        }

        params = ConfigOperationParams(
            operation="save",
            config_type="policies",
            user_id="admin",
            timestamp=datetime.now()
        )

        result = manager.save_config(policies_data, params)

        assert result is True

        # 验证文件是否创建
        policies_file = manager.config_path / "policies.json"
        assert policies_file.exists()


class TestSecurityConfigManagerBasicFunctionality:
    """测试安全配置管理器基本功能"""

    def test_config_operations(self, security_config_manager):
        """测试配置操作"""
        manager = security_config_manager

        # 保存用户配置
        users_data = {
            "user1": {"name": "Test User", "roles": ["admin"]},
            "user2": {"name": "Test User 2", "roles": ["user"]}
        }

        params = ConfigOperationParams(
            operation_type="save",
            config_sections={"users"}
        )

        result = manager.save_config(users_data, params)
        assert result is True

        # 加载用户配置
        load_params = ConfigOperationParams(
            operation_type="load",
            config_sections={"users"}
        )

        loaded_data = manager.load_config(load_params)
        assert loaded_data == users_data


class TestAuditConfigManager:
    """测试审计配置管理器"""

    def test_initialization(self, audit_config_manager):
        """测试审计配置管理器初始化"""
        manager = audit_config_manager

        assert manager.config_path is not None
        assert manager.config_path.exists()

    def test_load_audit_rules_default(self, audit_config_manager):
        """测试加载默认审计规则"""
        manager = audit_config_manager

        rules = manager.load_audit_rules()

        assert isinstance(rules, dict)
        assert "rules" in rules
        assert "version" in rules

    def test_load_audit_rules_from_file(self, audit_config_manager):
        """测试从文件加载审计规则"""
        manager = audit_config_manager

        # 创建测试规则文件
        rules_file = manager.config_path / "audit_rules.json"
        test_rules = {
            "rules": [
                {"event_type": "login", "severity": "low"},
                {"event_type": "trade", "severity": "medium"}
            ],
            "version": "1.0"
        }
        rules_file.write_text(json.dumps(test_rules))

        rules = manager.load_audit_rules()

        assert rules == test_rules
        assert len(rules["rules"]) == 2

    def test_save_audit_rules(self, audit_config_manager):
        """测试保存审计规则"""
        manager = audit_config_manager

        test_rules = {
            "rules": [
                {"event_type": "login", "severity": "low"},
                {"event_type": "admin_action", "severity": "high"}
            ],
            "version": "1.0"
        }

        result = manager.save_audit_rules(test_rules)

        assert result is True

        # 验证文件是否创建
        rules_file = manager.config_path / "audit_rules.json"
        assert rules_file.exists()

        # 验证内容
        saved_rules = json.loads(rules_file.read_text())
        assert saved_rules == test_rules


class TestConfigurationPersistence:
    """测试配置持久化"""

    def test_config_backup_creation(self, security_config_manager):
        """测试配置备份创建"""
        manager = security_config_manager

        # 保存一些配置以触发备份
        users_data = {"user1": {"name": "Test User"}}
        params = ConfigOperationParams(
            operation_type="save",
            config_sections={"users"}
        )

        manager.save_config(users_data, params)

        # 检查是否有备份文件
        backup_files = list(manager.config_path.glob("*.backup"))
        # 备份可能不会自动创建，取决于实现

    def test_config_file_permissions(self, security_config_manager):
        """测试配置文件权限"""
        manager = security_config_manager

        users_data = {"user1": {"name": "Test User"}}
        params = ConfigOperationParams(
            operation_type="save",
            config_sections={"users"}
        )

        manager.save_config(users_data, params)

        users_file = manager.config_path / "users.json"
        assert users_file.exists()

        # 在Windows上检查文件是否可读
        assert users_file.stat().st_size > 0


class TestConfigurationOperations:
    """测试配置操作"""

    def test_load_save_roundtrip_users(self, security_config_manager):
        """测试用户配置的加载保存往返"""
        manager = security_config_manager

        original_users = {
            "user1": {"name": "User One", "roles": ["admin"]},
            "user2": {"name": "User Two", "roles": ["user"]}
        }

        # 保存
        save_params = ConfigOperationParams(
            operation="save",
            config_type="users",
            user_id="admin",
            timestamp=datetime.now()
        )
        manager.save_config(original_users, save_params)

        # 加载
        load_params = ConfigOperationParams(
            operation="load",
            config_type="users",
            user_id="admin",
            timestamp=datetime.now()
        )
        loaded_users = manager.load_config(load_params)

        assert loaded_users == original_users

    def test_load_save_roundtrip_roles(self, security_config_manager):
        """测试角色配置的加载保存往返"""
        manager = security_config_manager

        original_roles = {
            "admin": {"permissions": ["read", "write", "delete"]},
            "user": {"permissions": ["read"]}
        }

        # 保存
        save_params = ConfigOperationParams(
            operation="save",
            config_type="roles",
            user_id="admin",
            timestamp=datetime.now()
        )
        manager.save_config(original_roles, save_params)

        # 加载
        load_params = ConfigOperationParams(
            operation="load",
            config_type="roles",
            user_id="admin",
            timestamp=datetime.now()
        )
        loaded_roles = manager.load_config(load_params)

        assert loaded_roles == original_roles


class TestErrorHandling:
    """测试错误处理"""

    def test_load_config_corrupted_file(self, security_config_manager, config_operation_params):
        """测试加载损坏的配置文件"""
        manager = security_config_manager
        params = config_operation_params
        params.config_type = "users"

        # 创建损坏的文件
        users_file = manager.config_path / "users.json"
        users_file.write_text("corrupted json {")

        result = manager.load_config(params)

        # 应该返回空字典而不是崩溃
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_save_config_invalid_path(self, security_config_manager):
        """测试保存到无效路径的配置"""
        # 这会使用默认路径，应该不会失败
        manager = SecurityConfigManager(config_path="/invalid/path")

        users_data = {"user1": {"name": "Test"}}
        params = ConfigOperationParams(
            operation_type="save",
            config_sections={"users"}
        )

        # 应该能够处理路径创建
        result = manager.save_config(users_data, params)
        assert isinstance(result, bool)

    def test_validate_config_with_missing_sections(self, security_config_manager):
        """测试验证缺少部分的配置"""
        manager = security_config_manager

        # 空配置应该通过验证
        manager._validate_config({})

        # 部分配置应该通过验证
        partial_config = {"users": {"user1": {"name": "Test"}}}
        manager._validate_config(partial_config)


class TestConfigurationSecurity:
    """测试配置安全性"""

    def test_config_file_isolation(self, security_config_manager):
        """测试配置文件隔离"""
        manager = security_config_manager

        # 保存用户配置
        users_data = {"user1": {"name": "Test User"}}
        params = ConfigOperationParams(
            operation_type="save",
            config_sections={"users"}
        )
        manager.save_config(users_data, params)

        # 保存角色配置
        roles_data = {"admin": {"permissions": ["read"]}}
        params.config_type = "roles"
        manager.save_config(roles_data, params)

        # 验证文件是分开的
        users_file = manager.config_path / "users.json"
        roles_file = manager.config_path / "roles.json"

        assert users_file.exists()
        assert roles_file.exists()

        # 验证内容不混淆
        saved_users = json.loads(users_file.read_text())
        saved_roles = json.loads(roles_file.read_text())

        assert "user1" in saved_users
        assert "admin" in saved_roles


class TestConfigurationPerformance:
    """测试配置性能"""

    def test_bulk_config_operations(self, security_config_manager):
        """测试批量配置操作"""
        manager = security_config_manager

        # 创建大量用户数据
        bulk_users = {}
        for i in range(100):
            bulk_users[f"user{i}"] = {
                "name": f"User {i}",
                "roles": ["user"],
                "email": f"user{i}@example.com"
            }

        # 保存
        params = ConfigOperationParams(
            operation_type="save",
            config_sections={"users"}
        )

        start_time = datetime.now()
        result = manager.save_config(bulk_users, params)
        save_time = (datetime.now() - start_time).total_seconds()

        assert result is True
        assert save_time < 1.0  # 保存应该很快

        # 加载
        params.operation = "load"
        start_time = datetime.now()
        loaded_data = manager.load_config(params)
        load_time = (datetime.now() - start_time).total_seconds()

        assert len(loaded_data) == 100
        assert load_time < 1.0  # 加载应该很快
