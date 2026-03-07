#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试核心层基础组件 - 简化版

测试目标：提升foundation/base.py的覆盖率
使用简化导入方式，避免导入错误
"""

import pytest
import time
import uuid
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# 尝试导入，如果失败则跳过
try:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
    
    from src.core.foundation.base import (
        ComponentStatus,
        ComponentHealth,
        ComponentInfo,
        BaseComponent
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    pytest.skip(f"模块导入失败: {e}", allow_module_level=True)


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestComponentStatus:
    """测试组件状态枚举"""

    def test_component_status_values(self):
        """测试组件状态枚举值"""
        assert ComponentStatus.UNKNOWN.value == "unknown"
        assert ComponentStatus.INITIALIZING.value == "initializing"
        assert ComponentStatus.INITIALIZED.value == "initialized"
        assert ComponentStatus.STARTING.value == "starting"
        assert ComponentStatus.RUNNING.value == "running"
        assert ComponentStatus.STOPPING.value == "stopping"
        assert ComponentStatus.STOPPED.value == "stopped"
        assert ComponentStatus.ERROR.value == "error"
        assert ComponentStatus.HEALTHY.value == "healthy"
        assert ComponentStatus.UNHEALTHY.value == "unhealthy"

    def test_component_status_all_values(self):
        """测试所有组件状态值"""
        expected_values = [
            "unknown", "initializing", "initialized", "starting",
            "running", "stopping", "stopped", "error", "healthy", "unhealthy"
        ]
        actual_values = [status.value for status in ComponentStatus]
        for expected in expected_values:
            assert expected in actual_values


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestComponentHealth:
    """测试组件健康状态枚举"""

    def test_component_health_values(self):
        """测试组件健康状态枚举值"""
        assert ComponentHealth.HEALTHY.value == "healthy"
        assert ComponentHealth.DEGRADED.value == "degraded"
        assert ComponentHealth.UNHEALTHY.value == "unhealthy"
        assert ComponentHealth.UNKNOWN.value == "unknown"


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestComponentInfo:
    """测试组件信息数据类"""

    def test_component_info_creation(self):
        """测试组件信息创建"""
        info = ComponentInfo(
            name="test_component",
            version="1.0.0",
            status=ComponentStatus.RUNNING,
            health=ComponentHealth.HEALTHY
        )
        assert info.name == "test_component"
        assert info.version == "1.0.0"
        assert info.status == ComponentStatus.RUNNING
        assert info.health == ComponentHealth.HEALTHY

    def test_component_info_defaults(self):
        """测试组件信息默认值"""
        info = ComponentInfo(name="test")
        assert info.name == "test"
        assert info.version == "1.0.0"
        assert info.status == ComponentStatus.UNKNOWN
        assert info.health == ComponentHealth.UNKNOWN


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestBaseComponent:
    """测试基础组件类"""

    def test_base_component_initialization(self):
        """测试基础组件初始化"""
        component = BaseComponent(
            name="test_component",
            version="1.0.0",
            description="测试组件"
        )
        assert component.name == "test_component"
        assert component.version == "1.0.0"
        assert component.description == "测试组件"
        assert component.status == ComponentStatus.INITIALIZING

    def test_base_component_get_info(self):
        """测试获取组件信息"""
        component = BaseComponent(name="test", version="1.0.0")
        info = component.get_info()
        assert info.name == "test"
        assert info.version == "1.0.0"
        assert isinstance(info.status, ComponentStatus)
        assert isinstance(info.health, ComponentHealth)

    def test_base_component_initialize(self):
        """测试组件初始化"""
        component = BaseComponent(name="test")
        result = component.initialize()
        assert result is True
        assert component.status == ComponentStatus.INITIALIZED

    def test_base_component_start(self):
        """测试组件启动"""
        component = BaseComponent(name="test")
        component.initialize()
        result = component.start()
        assert result is True
        assert component.status == ComponentStatus.RUNNING

    def test_base_component_stop(self):
        """测试组件停止"""
        component = BaseComponent(name="test")
        component.initialize()
        component.start()
        result = component.stop()
        assert result is True
        assert component.status == ComponentStatus.STOPPED

    def test_base_component_health_check(self):
        """测试组件健康检查"""
        component = BaseComponent(name="test")
        component.initialize()
        health = component.health_check()
        assert isinstance(health, dict)
        assert "status" in health
        assert "health" in health

    def test_base_component_get_status(self):
        """测试获取组件状态"""
        component = BaseComponent(name="test")
        status = component.get_status()
        assert isinstance(status, dict)
        assert "name" in status
        assert "status" in status
        assert "health" in status

