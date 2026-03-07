#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
指标组件质量测试
测试覆盖 MetricsComponent 和 MetricsComponentFactory 的核心功能
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
    engine_metrics_components_module = importlib.import_module('src.monitoring.engine.metrics_components')
    MetricsComponent = getattr(engine_metrics_components_module, 'MetricsComponent', None)
    MetricsComponentFactory = getattr(engine_metrics_components_module, 'MetricsComponentFactory', None)
    IMetricsComponent = getattr(engine_metrics_components_module, 'IMetricsComponent', None)
    
    if MetricsComponent is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


@pytest.fixture
def metrics_component():
    """创建指标组件实例"""
    return MetricsComponent(metrics_id=1, component_type="Metrics")


# MetricsComponentFactory使用静态方法，不需要实例化


class TestMetricsComponent:
    """MetricsComponent测试类"""

    def test_initialization(self, metrics_component):
        """测试初始化"""
        assert metrics_component.metrics_id == 1
        assert metrics_component.component_type == "Metrics"
        assert "Metrics_Component_1" in metrics_component.component_name
        assert isinstance(metrics_component.creation_time, datetime)

    def test_get_metrics_id(self, metrics_component):
        """测试获取metrics ID"""
        assert metrics_component.get_metrics_id() == 1

    def test_get_info(self, metrics_component):
        """测试获取组件信息"""
        info = metrics_component.get_info()
        assert isinstance(info, dict)
        assert info['metrics_id'] == 1
        assert info['component_type'] == "Metrics"

    def test_process(self, metrics_component):
        """测试处理数据"""
        test_data = {'key': 'value'}
        result = metrics_component.process(test_data)
        assert isinstance(result, dict)
        assert result['metrics_id'] == 1
        assert result['input_data'] == test_data

    def test_get_status(self, metrics_component):
        """测试获取组件状态"""
        status = metrics_component.get_status()
        assert isinstance(status, dict)
        assert status['metrics_id'] == 1
        assert status['status'] == 'active'
        assert status['health'] == 'good'


class TestMetricsComponentFactory:
    """MetricsComponentFactory测试类"""

    def test_get_available_metrics(self):
        """测试获取所有可用的metrics ID"""
        available_ids = MetricsComponentFactory.get_available_metricss()
        assert isinstance(available_ids, list)
        assert len(available_ids) > 0

    def test_create_component_valid_id(self):
        """测试创建组件（有效ID）"""
        component = MetricsComponentFactory.create_component(3)
        assert component is not None
        assert component.get_metrics_id() == 3

    def test_create_component_invalid_id(self):
        """测试创建组件（无效ID）"""
        with pytest.raises(ValueError):
            MetricsComponentFactory.create_component(999)

    def test_create_all_metrics(self):
        """测试创建所有metrics组件"""
        components = MetricsComponentFactory.create_all_metricss()
        assert isinstance(components, dict)
        assert len(components) > 0

    def test_get_factory_info(self):
        """测试获取工厂信息"""
        info = MetricsComponentFactory.get_factory_info()
        assert isinstance(info, dict)
        assert 'factory_name' in info

    def test_create_metrics_component_functions(self):
        """测试创建metrics组件的函数"""
        # 测试所有可用的metrics ID
        available_ids = MetricsComponentFactory.get_available_metricss()
        for metrics_id in available_ids:
            component = MetricsComponentFactory.create_component(metrics_id)
            assert component is not None
            assert component.get_metrics_id() == metrics_id
            # 通过组件获取信息
            info = component.get_info()
            assert isinstance(info, dict)
            assert info['metrics_id'] == metrics_id

