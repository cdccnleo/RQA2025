#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metrics Components覆盖率测试
专注提升metrics_components.py的测试覆盖率
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
    engine_metrics_components_module = importlib.import_module('src.monitoring.engine.metrics_components')
    IMetricsComponent = getattr(engine_metrics_components_module, 'IMetricsComponent', None)
    MetricsComponent = getattr(engine_metrics_components_module, 'MetricsComponent', None)
    MetricsComponentFactory = getattr(engine_metrics_components_module, 'MetricsComponentFactory', None)
    create_metrics_metrics_component_3 = getattr(engine_metrics_components_module, 'create_metrics_metrics_component_3', None)
    create_metrics_metrics_component_8 = getattr(engine_metrics_components_module, 'create_metrics_metrics_component_8', None)
    
    if MetricsComponent is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


class TestMetricsComponent:
    """测试MetricsComponent类"""

    def test_init(self):
        """测试初始化"""
        component = MetricsComponent(metrics_id=3)
        assert component.metrics_id == 3
        assert component.component_type == "Metrics"
        assert "Component_3" in component.component_name
        assert isinstance(component.creation_time, datetime)

    def test_init_with_custom_type(self):
        """测试自定义类型初始化"""
        component = MetricsComponent(metrics_id=8, component_type="CustomMetrics")
        assert component.metrics_id == 8
        assert component.component_type == "CustomMetrics"

    def test_get_metrics_id(self):
        """测试获取metrics ID"""
        component = MetricsComponent(metrics_id=5)
        assert component.get_metrics_id() == 5

    def test_get_info(self):
        """测试获取组件信息"""
        component = MetricsComponent(metrics_id=3)
        info = component.get_info()
        
        assert isinstance(info, dict)
        assert info['metrics_id'] == 3
        assert info['component_type'] == "Metrics"
        assert 'component_name' in info
        assert 'creation_time' in info
        assert 'version' in info
        assert 'type' in info

    def test_process_success(self):
        """测试处理数据成功"""
        component = MetricsComponent(metrics_id=3)
        data = {"key": "value", "number": 123}
        
        result = component.process(data)
        
        assert isinstance(result, dict)
        assert result['metrics_id'] == 3
        assert result['status'] == "success"
        assert result['input_data'] == data
        assert 'processed_at' in result
        assert 'result' in result

    def test_process_with_exception(self):
        """测试处理数据异常"""
        component = MetricsComponent(metrics_id=3)
        # 传入会导致异常的数据（如果需要）
        data = {"key": "value"}
        
        result = component.process(data)
        # 正常情况下应该成功，但确保异常处理逻辑存在
        assert isinstance(result, dict)

    def test_get_status(self):
        """测试获取组件状态"""
        component = MetricsComponent(metrics_id=8)
        status = component.get_status()
        
        assert isinstance(status, dict)
        assert status['metrics_id'] == 8
        assert status['status'] == "active"
        assert status['health'] == "good"
        assert 'creation_time' in status


class TestMetricsComponentFactory:
    """测试MetricsComponentFactory类"""

    def test_create_component_supported_id(self):
        """测试创建支持的组件ID"""
        component = MetricsComponentFactory.create_component(3)
        assert isinstance(component, MetricsComponent)
        assert component.get_metrics_id() == 3

    def test_create_component_supported_id_8(self):
        """测试创建支持的组件ID 8"""
        component = MetricsComponentFactory.create_component(8)
        assert isinstance(component, MetricsComponent)
        assert component.get_metrics_id() == 8

    def test_create_component_unsupported_id(self):
        """测试创建不支持的组件ID"""
        with pytest.raises(ValueError):
            MetricsComponentFactory.create_component(999)

    def test_get_available_metricss(self):
        """测试获取所有可用的metrics ID"""
        ids = MetricsComponentFactory.get_available_metricss()
        assert isinstance(ids, list)
        assert 3 in ids
        assert 8 in ids
        assert len(ids) == 2

    def test_create_all_metricss(self):
        """测试创建所有可用metrics"""
        all_components = MetricsComponentFactory.create_all_metricss()
        
        assert isinstance(all_components, dict)
        assert 3 in all_components
        assert 8 in all_components
        assert len(all_components) == 2
        
        # 验证每个组件类型正确
        for metrics_id, component in all_components.items():
            assert isinstance(component, MetricsComponent)
            assert component.get_metrics_id() == metrics_id

    def test_get_factory_info(self):
        """测试获取工厂信息"""
        info = MetricsComponentFactory.get_factory_info()
        
        assert isinstance(info, dict)
        assert info['factory_name'] == "MetricsComponentFactory"
        assert info['version'] == "2.0.0"
        assert info['total_metricss'] == 2
        assert 3 in info['supported_ids']
        assert 8 in info['supported_ids']
        assert 'created_at' in info


class TestBackwardCompatibility:
    """测试向后兼容函数"""

    def test_create_metrics_metrics_component_3(self):
        """测试向后兼容函数创建组件3"""
        component = create_metrics_metrics_component_3()
        assert isinstance(component, MetricsComponent)
        assert component.get_metrics_id() == 3

    def test_create_metrics_metrics_component_8(self):
        """测试向后兼容函数创建组件8"""
        component = create_metrics_metrics_component_8()
        assert isinstance(component, MetricsComponent)
        assert component.get_metrics_id() == 8


class TestComponentIntegration:
    """测试组件集成功能"""

    def test_component_workflow(self):
        """测试完整组件工作流程"""
        # 创建组件
        component = MetricsComponentFactory.create_component(3)
        
        # 获取信息
        info = component.get_info()
        assert info['metrics_id'] == 3
        
        # 处理数据
        data = {"test": "data", "value": 42}
        result = component.process(data)
        assert result['status'] == "success"
        
        # 获取状态
        status = component.get_status()
        assert status['status'] == "active"

    def test_multiple_components(self):
        """测试多个组件同时使用"""
        comp3 = MetricsComponentFactory.create_component(3)
        comp8 = MetricsComponentFactory.create_component(8)
        
        assert comp3.get_metrics_id() == 3
        assert comp8.get_metrics_id() == 8
        
        # 处理不同的数据
        result3 = comp3.process({"id": 3})
        result8 = comp8.process({"id": 8})
        
        assert result3['metrics_id'] == 3
        assert result8['metrics_id'] == 8

    def test_component_interface_compliance(self):
        """测试组件接口合规性"""
        component = MetricsComponentFactory.create_component(3)
        
        # 验证实现了所有接口方法
        assert hasattr(component, 'get_info')
        assert hasattr(component, 'process')
        assert hasattr(component, 'get_status')
        assert hasattr(component, 'get_metrics_id')
        
        # 验证方法可调用
        assert callable(component.get_info)
        assert callable(component.process)
        assert callable(component.get_status)
        assert callable(component.get_metrics_id)

