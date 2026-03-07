#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monitoring Components覆盖率测试
专注提升monitoring_components.py的测试覆盖率
"""

import pytest
from datetime import datetime
from typing import Dict, Any

import sys
import importlib
from pathlib import Path
import pytest

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
    IMonitoringComponent = getattr(engine_monitoring_components_module, 'IMonitoringComponent', None)
    MonitoringComponent = getattr(engine_monitoring_components_module, 'MonitoringComponent', None)
    MonitoringComponentFactory = getattr(engine_monitoring_components_module, 'MonitoringComponentFactory', None)
    create_monitoring_monitoring_component_1 = getattr(engine_monitoring_components_module, 'create_monitoring_monitoring_component_1', None)
    create_monitoring_monitoring_component_6 = getattr(engine_monitoring_components_module, 'create_monitoring_monitoring_component_6', None)
    
    if MonitoringComponent is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


class TestMonitoringComponent:
    """测试MonitoringComponent类"""

    def test_init(self):
        """测试初始化"""
        component = MonitoringComponent(monitoring_id=1)
        assert component.monitoring_id == 1
        assert component.component_type == "Monitoring"
        assert "Component_1" in component.component_name
        assert isinstance(component.creation_time, datetime)

    def test_init_with_custom_type(self):
        """测试自定义类型初始化"""
        component = MonitoringComponent(monitoring_id=6, component_type="CustomMonitoring")
        assert component.monitoring_id == 6
        assert component.component_type == "CustomMonitoring"

    def test_get_monitoring_id(self):
        """测试获取monitoring ID"""
        component = MonitoringComponent(monitoring_id=5)
        assert component.get_monitoring_id() == 5

    def test_get_info(self):
        """测试获取组件信息"""
        component = MonitoringComponent(monitoring_id=1)
        info = component.get_info()
        
        assert isinstance(info, dict)
        assert info['monitoring_id'] == 1
        assert info['component_type'] == "Monitoring"
        assert 'component_name' in info
        assert 'creation_time' in info
        assert 'version' in info

    def test_process_success(self):
        """测试处理数据成功"""
        component = MonitoringComponent(monitoring_id=1)
        data = {"key": "value", "number": 123}
        
        result = component.process(data)
        
        assert isinstance(result, dict)
        assert result['monitoring_id'] == 1
        assert result['status'] == "success"
        assert result['input_data'] == data
        assert 'processed_at' in result

    def test_get_status(self):
        """测试获取组件状态"""
        component = MonitoringComponent(monitoring_id=6)
        status = component.get_status()
        
        assert isinstance(status, dict)
        assert status['monitoring_id'] == 6
        assert status['status'] == "active"
        assert status['health'] == "good"


class TestMonitoringComponentFactory:
    """测试MonitoringComponentFactory类"""

    def test_create_component_supported_id(self):
        """测试创建支持的组件ID"""
        component = MonitoringComponentFactory.create_component(1)
        assert isinstance(component, MonitoringComponent)
        assert component.get_monitoring_id() == 1

    def test_create_component_supported_id_6(self):
        """测试创建支持的组件ID 6"""
        component = MonitoringComponentFactory.create_component(6)
        assert isinstance(component, MonitoringComponent)
        assert component.get_monitoring_id() == 6

    def test_create_component_unsupported_id(self):
        """测试创建不支持的组件ID"""
        with pytest.raises(ValueError):
            MonitoringComponentFactory.create_component(999)

    def test_get_available_monitorings(self):
        """测试获取所有可用的monitoring ID"""
        ids = MonitoringComponentFactory.get_available_monitorings()
        assert isinstance(ids, list)
        assert 1 in ids
        assert 6 in ids
        assert len(ids) == 2

    def test_create_all_monitorings(self):
        """测试创建所有可用monitorings"""
        all_components = MonitoringComponentFactory.create_all_monitorings()
        
        assert isinstance(all_components, dict)
        assert 1 in all_components
        assert 6 in all_components
        assert len(all_components) == 2

    def test_get_factory_info(self):
        """测试获取工厂信息"""
        info = MonitoringComponentFactory.get_factory_info()
        
        assert isinstance(info, dict)
        assert info['factory_name'] == "MonitoringComponentFactory"
        assert info['version'] == "2.0.0"
        assert info['total_monitorings'] == 2


class TestBackwardCompatibility:
    """测试向后兼容函数"""

    def test_create_monitoring_monitoring_component_1(self):
        """测试向后兼容函数创建组件1"""
        component = create_monitoring_monitoring_component_1()
        assert isinstance(component, MonitoringComponent)
        assert component.get_monitoring_id() == 1

    def test_create_monitoring_monitoring_component_6(self):
        """测试向后兼容函数创建组件6"""
        component = create_monitoring_monitoring_component_6()
        assert isinstance(component, MonitoringComponent)
        assert component.get_monitoring_id() == 6


class TestComponentIntegration:
    """测试组件集成功能"""

    def test_component_workflow(self):
        """测试完整组件工作流程"""
        component = MonitoringComponentFactory.create_component(1)
        
        info = component.get_info()
        assert info['monitoring_id'] == 1
        
        data = {"test": "data"}
        result = component.process(data)
        assert result['status'] == "success"
        
        status = component.get_status()
        assert status['status'] == "active"

    def test_multiple_components(self):
        """测试多个组件同时使用"""
        comp1 = MonitoringComponentFactory.create_component(1)
        comp6 = MonitoringComponentFactory.create_component(6)
        
        assert comp1.get_monitoring_id() == 1
        assert comp6.get_monitoring_id() == 6

