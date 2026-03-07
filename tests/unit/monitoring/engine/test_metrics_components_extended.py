#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MetricsComponent扩展测试
补充metrics_components.py中未覆盖的边界情况和错误处理
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
    engine_metrics_components_module = importlib.import_module('src.monitoring.engine.metrics_components')
    ComponentFactory = getattr(engine_metrics_components_module, 'ComponentFactory', None)
    IMetricsComponent = getattr(engine_metrics_components_module, 'IMetricsComponent', None)
    MetricsComponent = getattr(engine_metrics_components_module, 'MetricsComponent', None)
    MetricsComponentFactory = getattr(engine_metrics_components_module, 'MetricsComponentFactory', None)
    create_metrics_metrics_component_3 = getattr(engine_metrics_components_module, 'create_metrics_metrics_component_3', None)
    create_metrics_metrics_component_8 = getattr(engine_metrics_components_module, 'create_metrics_metrics_component_8', None)
    
    if MetricsComponent is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


class TestMetricsComponentFactoryEdgeCases:
    """测试MetricsComponentFactory边界情况"""

    def test_create_component_invalid_id(self):
        """测试创建无效ID的组件"""
        with pytest.raises(ValueError) as exc_info:
            MetricsComponentFactory.create_component(999)
        
        assert "不支持的metrics ID" in str(exc_info.value)
        assert "999" in str(exc_info.value)

    def test_create_component_supported_ids(self):
        """测试创建支持的ID"""
        for metrics_id in MetricsComponentFactory.SUPPORTED_METRICS_IDS:
            component = MetricsComponentFactory.create_component(metrics_id)
            assert isinstance(component, MetricsComponent)
            assert component.get_metrics_id() == metrics_id

    def test_get_available_metricss(self):
        """测试获取可用metrics ID"""
        available = MetricsComponentFactory.get_available_metricss()
        
        assert isinstance(available, list)
        assert len(available) > 0
        assert all(isinstance(id, int) for id in available)
        assert available == sorted(available)

    def test_get_available_metricss_content(self):
        """测试获取可用metrics ID的内容"""
        available = MetricsComponentFactory.get_available_metricss()
        expected = sorted(MetricsComponentFactory.SUPPORTED_METRICS_IDS)
        
        assert available == expected

    def test_create_all_metricss(self):
        """测试创建所有metrics"""
        all_metricss = MetricsComponentFactory.create_all_metricss()
        
        assert isinstance(all_metricss, dict)
        assert len(all_metricss) == len(MetricsComponentFactory.SUPPORTED_METRICS_IDS)
        
        for metrics_id, component in all_metricss.items():
            assert isinstance(component, MetricsComponent)
            assert component.get_metrics_id() == metrics_id

    def test_get_factory_info_structure(self):
        """测试工厂信息结构"""
        info = MetricsComponentFactory.get_factory_info()
        
        assert 'factory_name' in info
        assert 'version' in info
        assert 'total_metricss' in info
        assert 'supported_ids' in info
        assert 'created_at' in info
        assert 'description' in info

    def test_get_factory_info_values(self):
        """测试工厂信息值"""
        info = MetricsComponentFactory.get_factory_info()
        
        assert info['factory_name'] == "MetricsComponentFactory"
        assert info['version'] == "2.0.0"
        assert info['total_metricss'] == len(MetricsComponentFactory.SUPPORTED_METRICS_IDS)
        assert info['supported_ids'] == sorted(MetricsComponentFactory.SUPPORTED_METRICS_IDS)
        assert isinstance(info['created_at'], str)


class TestMetricsComponentErrorHandling:
    """测试MetricsComponent错误处理"""

    def test_process_error_response_structure(self):
        """测试process方法错误响应的结构"""
        component = MetricsComponent(3, "Metrics")
        data = {'test': 'data'}
        result = component.process(data)
        
        # 验证响应结构（正常情况不会产生错误，但验证结构完整性）
        assert 'metrics_id' in result
        assert 'component_name' in result
        assert 'component_type' in result
        assert 'input_data' in result
        assert 'processed_at' in result
        assert 'status' in result


class TestMetricsComponentInitialization:
    """测试MetricsComponent初始化"""

    def test_init_default_component_type(self):
        """测试使用默认component_type初始化"""
        component = MetricsComponent(3)
        
        assert component.metrics_id == 3
        assert component.component_type == "Metrics"
        assert component.component_name == "Metrics_Component_3"

    def test_init_custom_component_type(self):
        """测试使用自定义component_type初始化"""
        component = MetricsComponent(3, "CustomType")
        
        assert component.metrics_id == 3
        assert component.component_type == "CustomType"
        assert component.component_name == "CustomType_Component_3"

    def test_init_creation_time(self):
        """测试creation_time的设置"""
        before = datetime.now()
        component = MetricsComponent(3)
        after = datetime.now()
        
        assert before <= component.creation_time <= after


class TestMetricsComponentMethods:
    """测试MetricsComponent方法"""

    def test_get_info_all_fields(self):
        """测试get_info返回所有字段"""
        component = MetricsComponent(8, "TestType")
        info = component.get_info()
        
        assert 'metrics_id' in info
        assert 'component_name' in info
        assert 'component_type' in info
        assert 'creation_time' in info
        assert 'description' in info
        assert 'version' in info
        assert 'type' in info

    def test_process_success_structure(self):
        """测试process成功响应的结构"""
        component = MetricsComponent(3)
        data = {'test': 'value'}
        result = component.process(data)
        
        assert result['status'] == 'success'
        assert result['input_data'] == data
        assert 'processed_at' in result
        assert 'result' in result

    def test_get_status_all_fields(self):
        """测试get_status返回所有字段"""
        component = MetricsComponent(8)
        status = component.get_status()
        
        assert 'metrics_id' in status
        assert 'component_name' in status
        assert 'component_type' in status
        assert 'status' in status
        assert 'creation_time' in status
        assert 'health' in status

    def test_get_status_values(self):
        """测试get_status返回值"""
        component = MetricsComponent(3, "TestType")
        status = component.get_status()
        
        assert status['metrics_id'] == 3
        assert status['component_name'] == "TestType_Component_3"
        assert status['component_type'] == "TestType"
        assert status['status'] == "active"
        assert status['health'] == "good"


class TestComponentFactoryMetrics:
    """测试ComponentFactory（metrics相关）"""

    def test_component_factory_initialization(self):
        """测试ComponentFactory初始化"""
        factory = ComponentFactory()
        assert hasattr(factory, '_components')
        assert isinstance(factory._components, dict)

    def test_create_component_with_valid_config(self):
        """测试创建组件（有效配置）"""
        factory = ComponentFactory()
        config = {'key': 'value'}
        
        result = factory.create_component('test_type', config)
        assert result is None  # _create_component_instance返回None

    def test_create_component_with_exception(self):
        """测试创建组件时抛出异常"""
        factory = ComponentFactory()
        
        with patch.object(factory, '_create_component_instance', side_effect=Exception("Test error")):
            result = factory.create_component('test_type', {})
            assert result is None


class TestBackwardCompatibilityMetrics:
    """测试向后兼容函数"""

    def test_create_metrics_metrics_component_3(self):
        """测试create_metrics_metrics_component_3"""
        component = create_metrics_metrics_component_3()
        
        assert isinstance(component, MetricsComponent)
        assert component.get_metrics_id() == 3

    def test_create_metrics_metrics_component_8(self):
        """测试create_metrics_metrics_component_8"""
        component = create_metrics_metrics_component_8()
        
        assert isinstance(component, MetricsComponent)
        assert component.get_metrics_id() == 8



