#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monitor Components覆盖率测试
专注提升monitor_components.py的测试覆盖率
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
    engine_monitor_components_module = importlib.import_module('src.monitoring.engine.monitor_components')
    IMonitorComponent = getattr(engine_monitor_components_module, 'IMonitorComponent', None)
    MonitorComponent = getattr(engine_monitor_components_module, 'MonitorComponent', None)
    MonitorComponentFactory = getattr(engine_monitor_components_module, 'MonitorComponentFactory', None)
    create_monitor_monitor_component_2 = getattr(engine_monitor_components_module, 'create_monitor_monitor_component_2', None)
    create_monitor_monitor_component_7 = getattr(engine_monitor_components_module, 'create_monitor_monitor_component_7', None)
    
    if MonitorComponent is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


class TestMonitorComponent:
    """测试MonitorComponent类"""

    def test_init(self):
        """测试初始化"""
        component = MonitorComponent(monitor_id=2)
        assert component.monitor_id == 2
        assert component.component_type == "Monitor"
        assert "Component_2" in component.component_name
        assert isinstance(component.creation_time, datetime)

    def test_init_with_custom_type(self):
        """测试自定义类型初始化"""
        component = MonitorComponent(monitor_id=7, component_type="CustomMonitor")
        assert component.monitor_id == 7
        assert component.component_type == "CustomMonitor"

    def test_get_monitor_id(self):
        """测试获取monitor ID"""
        component = MonitorComponent(monitor_id=5)
        assert component.get_monitor_id() == 5

    def test_get_info(self):
        """测试获取组件信息"""
        component = MonitorComponent(monitor_id=2)
        info = component.get_info()
        
        assert isinstance(info, dict)
        assert info['monitor_id'] == 2
        assert info['component_type'] == "Monitor"
        assert 'component_name' in info
        assert 'creation_time' in info
        assert 'version' in info

    def test_process_success(self):
        """测试处理数据成功"""
        component = MonitorComponent(monitor_id=2)
        data = {"key": "value", "number": 123}
        
        result = component.process(data)
        
        assert isinstance(result, dict)
        assert result['monitor_id'] == 2
        assert result['status'] == "success"
        assert result['input_data'] == data
        assert 'processed_at' in result

    def test_get_status(self):
        """测试获取组件状态"""
        component = MonitorComponent(monitor_id=7)
        status = component.get_status()
        
        assert isinstance(status, dict)
        assert status['monitor_id'] == 7
        assert status['status'] == "active"
        assert status['health'] == "good"


class TestMonitorComponentFactory:
    """测试MonitorComponentFactory类"""

    def test_create_component_supported_id(self):
        """测试创建支持的组件ID"""
        component = MonitorComponentFactory.create_component(2)
        assert isinstance(component, MonitorComponent)
        assert component.get_monitor_id() == 2

    def test_create_component_supported_id_7(self):
        """测试创建支持的组件ID 7"""
        component = MonitorComponentFactory.create_component(7)
        assert isinstance(component, MonitorComponent)
        assert component.get_monitor_id() == 7

    def test_create_component_unsupported_id(self):
        """测试创建不支持的组件ID"""
        with pytest.raises(ValueError):
            MonitorComponentFactory.create_component(999)

    def test_get_available_monitors(self):
        """测试获取所有可用的monitor ID"""
        ids = MonitorComponentFactory.get_available_monitors()
        assert isinstance(ids, list)
        assert 2 in ids
        assert 7 in ids
        assert len(ids) == 2

    def test_create_all_monitors(self):
        """测试创建所有可用monitors"""
        all_components = MonitorComponentFactory.create_all_monitors()
        
        assert isinstance(all_components, dict)
        assert 2 in all_components
        assert 7 in all_components
        assert len(all_components) == 2

    def test_get_factory_info(self):
        """测试获取工厂信息"""
        info = MonitorComponentFactory.get_factory_info()
        
        assert isinstance(info, dict)
        assert info['factory_name'] == "MonitorComponentFactory"
        assert info['version'] == "2.0.0"
        assert info['total_monitors'] == 2


class TestBackwardCompatibility:
    """测试向后兼容函数"""

    def test_create_monitor_monitor_component_2(self):
        """测试向后兼容函数创建组件2"""
        component = create_monitor_monitor_component_2()
        assert isinstance(component, MonitorComponent)
        assert component.get_monitor_id() == 2

    def test_create_monitor_monitor_component_7(self):
        """测试向后兼容函数创建组件7"""
        component = create_monitor_monitor_component_7()
        assert isinstance(component, MonitorComponent)
        assert component.get_monitor_id() == 7


class TestComponentIntegration:
    """测试组件集成功能"""

    def test_component_workflow(self):
        """测试完整组件工作流程"""
        component = MonitorComponentFactory.create_component(2)
        
        info = component.get_info()
        assert info['monitor_id'] == 2
        
        data = {"test": "data"}
        result = component.process(data)
        assert result['status'] == "success"
        
        status = component.get_status()
        assert status['status'] == "active"

    def test_multiple_components(self):
        """测试多个组件同时使用"""
        comp2 = MonitorComponentFactory.create_component(2)
        comp7 = MonitorComponentFactory.create_component(7)
        
        assert comp2.get_monitor_id() == 2
        assert comp7.get_monitor_id() == 7

