#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控组件质量测试
测试覆盖 MonitoringComponent 和 MonitoringComponentFactory 的核心功能
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
    engine_monitoring_components_module = importlib.import_module('src.monitoring.engine.monitoring_components')
    MonitoringComponent = getattr(engine_monitoring_components_module, 'MonitoringComponent', None)
    MonitoringComponentFactory = getattr(engine_monitoring_components_module, 'MonitoringComponentFactory', None)
    IMonitoringComponent = getattr(engine_monitoring_components_module, 'IMonitoringComponent', None)
    
    if MonitoringComponent is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


@pytest.fixture
def monitoring_component():
    """创建监控组件实例"""
    return MonitoringComponent(monitoring_id=1, component_type="Monitoring")


class TestMonitoringComponent:
    """MonitoringComponent测试类"""

    def test_initialization(self, monitoring_component):
        """测试初始化"""
        assert monitoring_component.monitoring_id == 1
        assert monitoring_component.component_type == "Monitoring"
        assert "Monitoring_Component_1" in monitoring_component.component_name
        assert isinstance(monitoring_component.creation_time, datetime)

    def test_get_monitoring_id(self, monitoring_component):
        """测试获取monitoring ID"""
        assert monitoring_component.get_monitoring_id() == 1

    def test_get_info(self, monitoring_component):
        """测试获取组件信息"""
        info = monitoring_component.get_info()
        assert isinstance(info, dict)
        assert info['monitoring_id'] == 1
        assert info['component_type'] == "Monitoring"

    def test_process(self, monitoring_component):
        """测试处理数据"""
        test_data = {'key': 'value'}
        result = monitoring_component.process(test_data)
        assert isinstance(result, dict)
        assert result['monitoring_id'] == 1
        assert result['input_data'] == test_data

    def test_get_status(self, monitoring_component):
        """测试获取组件状态"""
        status = monitoring_component.get_status()
        assert isinstance(status, dict)
        assert status['monitoring_id'] == 1
        assert status['status'] == 'active'
        assert status['health'] == 'good'


class TestMonitoringComponentFactory:
    """MonitoringComponentFactory测试类"""

    def test_get_available_monitorings(self):
        """测试获取所有可用的monitoring ID"""
        available_ids = MonitoringComponentFactory.get_available_monitorings()
        assert isinstance(available_ids, list)
        assert len(available_ids) > 0

    def test_create_component_valid_id(self):
        """测试创建组件（有效ID）"""
        available_ids = MonitoringComponentFactory.get_available_monitorings()
        if available_ids:
            component = MonitoringComponentFactory.create_component(available_ids[0])
            assert component is not None
            assert component.get_monitoring_id() == available_ids[0]

    def test_create_component_invalid_id(self):
        """测试创建组件（无效ID）"""
        with pytest.raises(ValueError):
            MonitoringComponentFactory.create_component(999)

    def test_create_all_monitorings(self):
        """测试创建所有monitoring组件"""
        components = MonitoringComponentFactory.create_all_monitorings()
        assert isinstance(components, dict)
        assert len(components) > 0

    def test_get_factory_info(self):
        """测试获取工厂信息"""
        info = MonitoringComponentFactory.get_factory_info()
        assert isinstance(info, dict)
        assert 'factory_name' in info

    def test_create_monitoring_component_functions(self):
        """测试创建monitoring组件的函数"""
        available_ids = MonitoringComponentFactory.get_available_monitorings()
        for monitoring_id in available_ids:
            component = MonitoringComponentFactory.create_component(monitoring_id)
            assert component is not None
            assert component.get_monitoring_id() == monitoring_id
            info = component.get_info()
            assert isinstance(info, dict)
            assert info['monitoring_id'] == monitoring_id


