#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Status Components覆盖率测试
专注提升status_components.py的测试覆盖率
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
    engine_status_components_module = importlib.import_module('src.monitoring.engine.status_components')
    IStatusComponent = getattr(engine_status_components_module, 'IStatusComponent', None)
    StatusComponent = getattr(engine_status_components_module, 'StatusComponent', None)
    StatusComponentFactory = getattr(engine_status_components_module, 'StatusComponentFactory', None)
    create_status_status_component_5 = getattr(engine_status_components_module, 'create_status_status_component_5', None)

    if StatusComponent is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


class TestStatusComponent:
    """测试StatusComponent类"""

    def test_init(self):
        """测试初始化"""
        component = StatusComponent(status_id=5)
        assert component.status_id == 5
        assert component.component_type == "Status"
        assert "Component_5" in component.component_name
        assert isinstance(component.creation_time, datetime)

    def test_init_with_custom_type(self):
        """测试自定义类型初始化"""
        component = StatusComponent(status_id=5, component_type="CustomStatus")
        assert component.status_id == 5
        assert component.component_type == "CustomStatus"

    def test_get_status_id(self):
        """测试获取status ID"""
        component = StatusComponent(status_id=5)
        assert component.get_status_id() == 5

    def test_get_info(self):
        """测试获取组件信息"""
        component = StatusComponent(status_id=5)
        info = component.get_info()
        
        assert isinstance(info, dict)
        assert info['status_id'] == 5
        assert info['component_type'] == "Status"
        assert 'component_name' in info
        assert 'creation_time' in info
        assert 'version' in info

    def test_process_success(self):
        """测试处理数据成功"""
        component = StatusComponent(status_id=5)
        data = {"key": "value", "number": 123}
        
        result = component.process(data)
        
        assert isinstance(result, dict)
        assert result['status_id'] == 5
        assert result['status'] == "success"
        assert result['input_data'] == data
        assert 'processed_at' in result

    def test_get_status(self):
        """测试获取组件状态"""
        component = StatusComponent(status_id=5)
        status = component.get_status()
        
        assert isinstance(status, dict)
        assert status['status_id'] == 5
        assert status['status'] == "active"
        assert status['health'] == "good"


class TestStatusComponentFactory:
    """测试StatusComponentFactory类"""

    def test_create_component_supported_id(self):
        """测试创建支持的组件ID"""
        component = StatusComponentFactory.create_component(5)
        assert isinstance(component, StatusComponent)
        assert component.get_status_id() == 5

    def test_create_component_unsupported_id(self):
        """测试创建不支持的组件ID"""
        with pytest.raises(ValueError):
            StatusComponentFactory.create_component(999)

    def test_get_available_statuss(self):
        """测试获取所有可用的status ID"""
        ids = StatusComponentFactory.get_available_statuss()
        assert isinstance(ids, list)
        assert 5 in ids
        assert len(ids) == 1

    def test_create_all_statuss(self):
        """测试创建所有可用status"""
        all_components = StatusComponentFactory.create_all_statuss()
        
        assert isinstance(all_components, dict)
        assert 5 in all_components
        assert len(all_components) == 1

    def test_get_factory_info(self):
        """测试获取工厂信息"""
        info = StatusComponentFactory.get_factory_info()
        
        assert isinstance(info, dict)
        assert info['factory_name'] == "StatusComponentFactory"
        assert info['version'] == "2.0.0"
        assert info['total_statuss'] == 1


class TestBackwardCompatibility:
    """测试向后兼容函数"""

    def test_create_status_status_component_5(self):
        """测试向后兼容函数创建组件5"""
        component = create_status_status_component_5()
        assert isinstance(component, StatusComponent)
        assert component.get_status_id() == 5


class TestComponentIntegration:
    """测试组件集成功能"""

    def test_component_workflow(self):
        """测试完整组件工作流程"""
        component = StatusComponentFactory.create_component(5)
        
        info = component.get_info()
        assert info['status_id'] == 5
        
        data = {"test": "data"}
        result = component.process(data)
        assert result['status'] == "success"
        
        status = component.get_status()
        assert status['status'] == "active"

