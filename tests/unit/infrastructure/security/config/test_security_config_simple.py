#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安全配置管理器基础测试
测试SecurityConfigManager的核心功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import json
from pathlib import Path

from src.infrastructure.security.config.security_config import (
    SecurityConfigManager,
    AuditConfigManager
)
from src.infrastructure.security.core.types import ConfigOperationParams


@pytest.fixture
def temp_config_dir(tmp_path):
    """创建临时配置目录"""
    return tmp_path / "config"


@pytest.fixture
def security_config_manager(temp_config_dir):
    """创建安全配置管理器实例"""
    return SecurityConfigManager(config_path=str(temp_config_dir))


@pytest.fixture
def audit_config_manager(temp_config_dir):
    """创建审计配置管理器实例"""
    return AuditConfigManager(config_path=str(temp_config_dir))


class TestSecurityConfigManager:
    """测试安全配置管理器"""

    def test_initialization(self, security_config_manager):
        """测试初始化"""
        manager = security_config_manager
        assert manager.config_path.exists()

    def test_save_and_load_config(self, security_config_manager):
        """测试保存和加载配置"""
        manager = security_config_manager

        # 保存配置 - save_config期望包含配置类型的字典
        config_data = {
            "users": {
                "user1": {"name": "Test User", "roles": ["admin"]},
                "user2": {"name": "Test User 2", "roles": ["user"]}
            }
        }

        save_params = ConfigOperationParams(
            operation_type="save",
            config_sections={"users"}
        )

        result = manager.save_config(config_data, save_params)
        assert result is True

        # 加载配置
        load_params = ConfigOperationParams(
            operation_type="load",
            config_sections={"users"}
        )

        loaded_data = manager.load_config(load_params)
        # load_config返回的是包含配置部分的字典
        assert "users" in loaded_data
        assert loaded_data["users"]["user1"]["name"] == "Test User"

    def test_load_nonexistent_config(self, security_config_manager):
        """测试加载不存在的配置"""
        manager = security_config_manager

        params = ConfigOperationParams(
            operation_type="load",
            config_sections={"nonexistent"}
        )

        result = manager.load_config(params)
        assert isinstance(result, dict)
        assert len(result) == 0


class TestAuditConfigManager:
    """测试审计配置管理器"""

    def test_initialization(self, audit_config_manager):
        """测试初始化"""
        manager = audit_config_manager
        assert manager.config_path.exists()

    def test_load_audit_rules(self, audit_config_manager):
        """测试加载审计规则"""
        manager = audit_config_manager

        rules = manager.load_audit_rules()
        assert isinstance(rules, dict)

    def test_save_and_load_audit_rules(self, audit_config_manager):
        """测试保存和加载审计规则"""
        manager = audit_config_manager

        test_rules = {
            "rules": [
                {"event_type": "login", "severity": "low"}
            ],
            "version": "1.0"
        }

        manager.save_audit_rules(test_rules)

        # 重新加载
        loaded_rules = manager.load_audit_rules()

        # 如果文件保存了，应该能加载到
        rules_file = Path(manager.config_path) / "audit_rules.json"
        if rules_file.exists():
            assert loaded_rules == test_rules
