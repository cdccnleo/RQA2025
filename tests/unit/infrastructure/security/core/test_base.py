#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 安全组件基类测试

测试BaseSecurityComponent类的核心功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, Any

from src.infrastructure.security.core.base import BaseSecurityComponent


class TestBaseSecurityComponent:
    """基础安全组件测试"""

    @pytest.fixture
    def base_component(self):
        """基础组件fixture"""
        return BaseSecurityComponent()

    def test_initialization(self, base_component):
        """测试初始化"""
        assert base_component._initialized is False
        assert base_component._status == "stopped"
        assert isinstance(base_component.config, dict)
        assert len(base_component.config) == 0

    def test_initialize_success(self, base_component):
        """测试成功初始化"""
        config = {"key": "value", "enabled": True}

        result = base_component.initialize(config)

        assert result is True
        assert base_component._initialized is True
        assert base_component._status == "running"
        assert base_component.config == config

    def test_initialize_with_exception(self, base_component):
        """测试初始化时发生异常"""
        config = {"key": "value"}

        # 模拟配置更新时发生异常
        with patch.object(base_component.config, 'update', side_effect=Exception("Config error")):
            result = base_component.initialize(config)

            assert result is False
            assert base_component._initialized is False
            assert base_component._status == "error"

    def test_get_status_initial(self, base_component):
        """测试获取初始状态"""
        status = base_component.get_status()

        expected_status = {
            "component": "security",
            "status": "stopped",
            "initialized": False,
            "config": {}
        }

        assert status == expected_status

    def test_get_status_after_initialize(self, base_component):
        """测试初始化后获取状态"""
        config = {"enabled": True, "level": "high"}
        base_component.initialize(config)

        status = base_component.get_status()

        expected_status = {
            "component": "security",
            "status": "running",
            "initialized": True,
            "config": config
        }

        assert status == expected_status

    def test_shutdown(self, base_component):
        """测试关闭组件"""
        # 先初始化组件
        base_component.initialize({"enabled": True})
        assert base_component._initialized is True
        assert base_component._status == "running"

        # 关闭组件
        base_component.shutdown()

        assert base_component._initialized is False
        assert base_component._status == "stopped"

    def test_shutdown_uninitialized(self, base_component):
        """测试关闭未初始化的组件"""
        # 直接关闭未初始化的组件
        base_component.shutdown()

        assert base_component._initialized is False
        assert base_component._status == "stopped"

    def test_initialize_partial_update(self, base_component):
        """测试部分配置更新"""
        # 初始配置
        base_component.initialize({"key1": "value1", "key2": "value2"})

        # 部分更新
        base_component.initialize({"key2": "new_value", "key3": "value3"})

        expected_config = {
            "key1": "value1",
            "key2": "new_value",
            "key3": "value3"
        }

        assert base_component.config == expected_config
        assert base_component._initialized is True
        assert base_component._status == "running"

    def test_multiple_initialize_calls(self, base_component):
        """测试多次初始化调用"""
        configs = [
            {"step": 1, "enabled": True},
            {"step": 2, "level": "high"},
            {"step": 3, "timeout": 30}
        ]

        for config in configs:
            result = base_component.initialize(config)
            assert result is True

        # 最终配置应该包含所有设置
        assert base_component.config["step"] == 3
        assert base_component.config["enabled"] is True
        assert base_component.config["level"] == "high"
        assert base_component.config["timeout"] == 30

        assert base_component._initialized is True
        assert base_component._status == "running"

    def test_get_status_returns_copy(self, base_component):
        """测试get_status返回配置的副本"""
        config = {"mutable": ["item"]}
        base_component.initialize(config)

        status = base_component.get_status()

        # 修改返回的状态不应该影响原始配置
        status["config"]["mutable"].append("new_item")

        assert base_component.config["mutable"] == ["item"]
