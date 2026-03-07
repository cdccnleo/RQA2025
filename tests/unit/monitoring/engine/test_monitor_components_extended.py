#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MonitorComponent扩展测试
补充monitor_components.py中未覆盖的边界情况和错误处理
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

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
    ComponentFactory = getattr(engine_monitor_components_module, 'ComponentFactory', None)
    IMonitorComponent = getattr(engine_monitor_components_module, 'IMonitorComponent', None)
    MonitorComponent = getattr(engine_monitor_components_module, 'MonitorComponent', None)
    MonitorComponentFactory = getattr(engine_monitor_components_module, 'MonitorComponentFactory', None)
    create_monitor_monitor_component_2 = getattr(engine_monitor_components_module, 'create_monitor_monitor_component_2', None)
    create_monitor_monitor_component_7 = getattr(engine_monitor_components_module, 'create_monitor_monitor_component_7', None)
    
    if MonitorComponent is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


class TestMonitorComponentFactoryEdgeCases:
    """测试MonitorComponentFactory边界情况"""

    def test_create_component_invalid_id(self):
        """测试创建无效ID的组件"""
        with pytest.raises(ValueError) as exc_info:
            MonitorComponentFactory.create_component(999)
        
        assert "不支持的monitor ID" in str(exc_info.value)
        assert "999" in str(exc_info.value)

    def test_create_component_supported_ids(self):
        """测试创建支持的ID"""
        for monitor_id in MonitorComponentFactory.SUPPORTED_MONITOR_IDS:
            component = MonitorComponentFactory.create_component(monitor_id)
            assert isinstance(component, MonitorComponent)
            assert component.get_monitor_id() == monitor_id

    def test_get_available_monitors(self):
        """测试获取可用monitor ID"""
        available = MonitorComponentFactory.get_available_monitors()
        
        assert isinstance(available, list)
        assert len(available) > 0
        assert all(isinstance(id, int) for id in available)
        assert available == sorted(available)

    def test_get_available_monitors_content(self):
        """测试获取可用monitor ID的内容"""
        available = MonitorComponentFactory.get_available_monitors()
        expected = sorted(MonitorComponentFactory.SUPPORTED_MONITOR_IDS)
        
        assert available == expected

    def test_create_all_monitors(self):
        """测试创建所有monitors"""
        all_monitors = MonitorComponentFactory.create_all_monitors()
        
        assert isinstance(all_monitors, dict)
        assert len(all_monitors) == len(MonitorComponentFactory.SUPPORTED_MONITOR_IDS)
        
        for monitor_id, component in all_monitors.items():
            assert isinstance(component, MonitorComponent)
            assert component.get_monitor_id() == monitor_id

    def test_get_factory_info_structure(self):
        """测试工厂信息结构"""
        info = MonitorComponentFactory.get_factory_info()
        
        assert 'factory_name' in info
        assert 'version' in info
        assert 'total_monitors' in info
        assert 'supported_ids' in info
        assert 'created_at' in info
        assert 'description' in info

    def test_get_factory_info_values(self):
        """测试工厂信息值"""
        info = MonitorComponentFactory.get_factory_info()
        
        assert info['factory_name'] == "MonitorComponentFactory"
        assert info['version'] == "2.0.0"
        assert info['total_monitors'] == len(MonitorComponentFactory.SUPPORTED_MONITOR_IDS)
        assert info['supported_ids'] == sorted(MonitorComponentFactory.SUPPORTED_MONITOR_IDS)
        assert isinstance(info['created_at'], str)


class TestMonitorComponentErrorHandling:
    """测试MonitorComponent错误处理"""

    def test_process_error_response_structure(self):
        """测试process方法错误响应的结构"""
        component = MonitorComponent(2, "Monitor")
        data = {'test': 'data'}
        result = component.process(data)
        
        # 验证响应结构
        assert 'monitor_id' in result
        assert 'component_name' in result
        assert 'component_type' in result
        assert 'input_data' in result
        assert 'processed_at' in result
        assert 'status' in result


class TestMonitorComponentInitialization:
    """测试MonitorComponent初始化"""

    def test_init_default_component_type(self):
        """测试使用默认component_type初始化"""
        component = MonitorComponent(2)
        
        assert component.monitor_id == 2
        assert component.component_type == "Monitor"
        assert component.component_name == "Monitor_Component_2"

    def test_init_custom_component_type(self):
        """测试使用自定义component_type初始化"""
        component = MonitorComponent(7, "CustomType")
        
        assert component.monitor_id == 7
        assert component.component_type == "CustomType"
        assert component.component_name == "CustomType_Component_7"

    def test_init_creation_time(self):
        """测试creation_time的设置"""
        before = datetime.now()
        component = MonitorComponent(2)
        after = datetime.now()
        
        assert before <= component.creation_time <= after


class TestMonitorComponentMethods:
    """测试MonitorComponent方法"""

    def test_get_info_all_fields(self):
        """测试get_info返回所有字段"""
        component = MonitorComponent(7, "TestType")
        info = component.get_info()
        
        assert 'monitor_id' in info
        assert 'component_name' in info
        assert 'component_type' in info
        assert 'creation_time' in info
        assert 'description' in info
        assert 'version' in info
        assert 'type' in info

    def test_process_success_structure(self):
        """测试process成功响应的结构"""
        component = MonitorComponent(2)
        data = {'test': 'value'}
        result = component.process(data)
        
        assert result['status'] == 'success'
        assert result['input_data'] == data
        assert 'processed_at' in result
        assert 'result' in result

    def test_get_status_all_fields(self):
        """测试get_status返回所有字段"""
        component = MonitorComponent(7)
        status = component.get_status()
        
        assert 'monitor_id' in status
        assert 'component_name' in status
        assert 'component_type' in status
        assert 'status' in status
        assert 'creation_time' in status
        assert 'health' in status

    def test_get_status_values(self):
        """测试get_status返回值"""
        component = MonitorComponent(2, "TestType")
        status = component.get_status()
        
        assert status['monitor_id'] == 2
        assert status['component_name'] == "TestType_Component_2"
        assert status['component_type'] == "TestType"
        assert status['status'] == "active"
        assert status['health'] == "good"


class TestBackwardCompatibilityMonitor:
    """测试向后兼容函数"""

    def test_create_monitor_monitor_component_2(self):
        """测试create_monitor_monitor_component_2"""
        component = create_monitor_monitor_component_2()
        
        assert isinstance(component, MonitorComponent)
        assert component.get_monitor_id() == 2

    def test_create_monitor_monitor_component_7(self):
        """测试create_monitor_monitor_component_7"""
        component = create_monitor_monitor_component_7()
        
        assert isinstance(component, MonitorComponent)
        assert component.get_monitor_id() == 7



