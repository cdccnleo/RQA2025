#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控组件质量测试
测试覆盖 MonitorComponent 和 MonitorComponentFactory 的核心功能
"""

import sys
import importlib
from pathlib import Path
import pytest
from unittest.mock import Mock, patch
from datetime import datetime

# 确保Python路径正确配置
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
project_root_str = str(project_root)
src_path_str = str(project_root / "src")

if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

# 动态导入模块
try:
    engine_monitor_components_module = importlib.import_module('src.monitoring.engine.monitor_components')
    MonitorComponent = getattr(engine_monitor_components_module, 'MonitorComponent', None)
    MonitorComponentFactory = getattr(engine_monitor_components_module, 'MonitorComponentFactory', None)
    IMonitorComponent = getattr(engine_monitor_components_module, 'IMonitorComponent', None)
    
    if MonitorComponent is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


@pytest.fixture
def monitor_component():
    """创建监控组件实例"""
    return MonitorComponent(monitor_id=1, component_type="Monitor")


# MonitorComponentFactory使用静态方法，不需要实例化


class TestMonitorComponent:
    """MonitorComponent测试类"""

    def test_initialization(self, monitor_component):
        """测试初始化"""
        assert monitor_component.monitor_id == 1
        assert monitor_component.component_type == "Monitor"
        assert "Monitor_Component_1" in monitor_component.component_name
        assert isinstance(monitor_component.creation_time, datetime)

    def test_get_monitor_id(self, monitor_component):
        """测试获取monitor ID"""
        assert monitor_component.get_monitor_id() == 1

    def test_get_info(self, monitor_component):
        """测试获取组件信息"""
        info = monitor_component.get_info()
        assert isinstance(info, dict)
        assert info['monitor_id'] == 1
        assert info['component_type'] == "Monitor"

    def test_process(self, monitor_component):
        """测试处理数据"""
        test_data = {'key': 'value'}
        result = monitor_component.process(test_data)
        assert isinstance(result, dict)
        assert result['monitor_id'] == 1
        assert result['input_data'] == test_data

    def test_get_status(self, monitor_component):
        """测试获取组件状态"""
        status = monitor_component.get_status()
        assert isinstance(status, dict)
        assert status['monitor_id'] == 1
        assert status['status'] == 'active'
        assert status['health'] == 'good'


class TestMonitorComponentFactory:
    """MonitorComponentFactory测试类"""

    def test_get_available_monitors(self):
        """测试获取所有可用的monitor ID"""
        available_ids = MonitorComponentFactory.get_available_monitors()
        assert isinstance(available_ids, list)
        assert len(available_ids) > 0

    def test_create_component_valid_id(self):
        """测试创建组件（有效ID）"""
        component = MonitorComponentFactory.create_component(2)
        assert component is not None
        assert component.get_monitor_id() == 2

    def test_create_component_invalid_id(self):
        """测试创建组件（无效ID）"""
        with pytest.raises(ValueError):
            MonitorComponentFactory.create_component(999)

    def test_create_all_monitors(self):
        """测试创建所有monitor组件"""
        components = MonitorComponentFactory.create_all_monitors()
        assert isinstance(components, dict)
        assert len(components) > 0

    def test_get_factory_info(self):
        """测试获取工厂信息"""
        info = MonitorComponentFactory.get_factory_info()
        assert isinstance(info, dict)
        assert 'factory_name' in info

    def test_create_monitor_component_functions(self):
        """测试创建monitor组件的函数"""
        # 测试所有可用的monitor ID
        available_ids = MonitorComponentFactory.get_available_monitors()
        for monitor_id in available_ids:
            component = MonitorComponentFactory.create_component(monitor_id)
            assert component is not None
            assert component.get_monitor_id() == monitor_id
            # 通过组件获取信息
            info = component.get_info()
            assert isinstance(info, dict)
            assert info['monitor_id'] == monitor_id

