#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metrics组件测试覆盖
测试monitoring/metrics_components.py
"""

import pytest
from datetime import datetime
from typing import Dict, Any
from unittest.mock import patch, MagicMock

from src.features.monitoring.metrics_components import (
    IMetricsComponent,
    MetricsComponent,
    MetricsComponentFactory,
    create_metrics_metrics_component_5,
    create_metrics_metrics_component_10,
    create_metrics_metrics_component_15,
    create_metrics_metrics_component_20,
)


class TestMetricsComponent:
    """Metrics组件测试"""

    def test_metrics_component_initialization(self):
        """测试Metrics组件初始化"""
        component = MetricsComponent(metrics_id=5)
        assert component.metrics_id == 5
        assert component.component_type == "Metrics"
        assert component.component_name == "Metrics_Component_5"
        assert isinstance(component.creation_time, datetime)

    def test_metrics_component_get_metrics_id(self):
        """测试获取metrics ID"""
        component = MetricsComponent(metrics_id=10)
        assert component.get_metrics_id() == 10

    def test_metrics_component_get_info(self):
        """测试获取组件信息"""
        component = MetricsComponent(metrics_id=15)
        info = component.get_info()
        assert info["metrics_id"] == 15
        assert info["component_name"] == "Metrics_Component_15"
        assert info["component_type"] == "Metrics"
        assert "creation_time" in info
        assert info["version"] == "2.0.0"
        assert info["type"] == "unified_features_monitoring_component"

    def test_metrics_component_process_success(self):
        """测试处理数据成功"""
        component = MetricsComponent(metrics_id=20)
        data = {"key": "value", "number": 123}
        result = component.process(data)
        assert result["metrics_id"] == 20
        assert result["status"] == "success"
        assert result["input_data"] == data
        assert "processed_at" in result
        assert result["processing_type"] == "unified_metrics_processing"

    def test_metrics_component_process_with_exception(self):
        """测试处理数据时异常处理"""
        from unittest.mock import patch, MagicMock
        from datetime import datetime
        component = MetricsComponent(metrics_id=5)
        data = {"key": "value"}
        # 模拟datetime.now()在try块中抛出异常，在except块中返回正常值
        mock_datetime_obj = MagicMock()
        call_count = [0]
        def side_effect():
            call_count[0] += 1
            if call_count[0] == 1:  # 第一次调用（try块中）
                raise Exception("模拟异常")
            else:  # 第二次调用（except块中）
                return datetime.now()
        mock_datetime_obj.now.side_effect = side_effect
        mock_datetime_obj.now.return_value.isoformat = lambda: datetime.now().isoformat()
        with patch('src.features.monitoring.metrics_components.datetime', mock_datetime_obj):
            result = component.process(data)
            assert result["metrics_id"] == 5
            assert result["status"] == "error"
            assert "error" in result
            assert "error_type" in result

    def test_metrics_component_get_status(self):
        """测试获取组件状态"""
        component = MetricsComponent(metrics_id=5)
        status = component.get_status()
        assert status["metrics_id"] == 5
        assert status["status"] == "active"
        assert status["health"] == "good"
        assert "creation_time" in status

    def test_metrics_component_implements_interface(self):
        """测试MetricsComponent实现接口"""
        component = MetricsComponent(metrics_id=10)
        assert isinstance(component, IMetricsComponent)
        assert hasattr(component, 'get_info')
        assert hasattr(component, 'process')
        assert hasattr(component, 'get_status')
        assert hasattr(component, 'get_metrics_id')


class TestMetricsComponentFactory:
    """MetricsComponentFactory测试"""

    def test_factory_create_component(self):
        """测试工厂创建组件"""
        component = MetricsComponentFactory.create_component(5)
        assert isinstance(component, MetricsComponent)
        assert component.metrics_id == 5

    def test_factory_create_invalid_id(self):
        """测试工厂创建无效ID组件"""
        with pytest.raises(ValueError, match="不支持的metrics ID"):
            MetricsComponentFactory.create_component(99)

    def test_factory_get_available_metricss(self):
        """测试获取所有可用的metrics ID"""
        available = MetricsComponentFactory.get_available_metricss()
        assert isinstance(available, list)
        assert len(available) == 4
        assert 5 in available
        assert 10 in available
        assert 15 in available
        assert 20 in available

    def test_factory_create_all_metricss(self):
        """测试创建所有可用metrics"""
        all_metricss = MetricsComponentFactory.create_all_metricss()
        assert isinstance(all_metricss, dict)
        assert len(all_metricss) == 4
        for metrics_id, component in all_metricss.items():
            assert isinstance(component, MetricsComponent)
            assert component.metrics_id == metrics_id

    def test_factory_get_factory_info(self):
        """测试获取工厂信息"""
        info = MetricsComponentFactory.get_factory_info()
        assert info["factory_name"] == "MetricsComponentFactory"
        assert info["version"] == "2.0.0"
        assert info["total_metricss"] == 4
        assert len(info["supported_ids"]) == 4
        assert "created_at" in info

    def test_factory_backward_compatibility_functions(self):
        """测试向后兼容函数"""
        comp5 = create_metrics_metrics_component_5()
        assert comp5.metrics_id == 5

        comp10 = create_metrics_metrics_component_10()
        assert comp10.metrics_id == 10

        comp15 = create_metrics_metrics_component_15()
        assert comp15.metrics_id == 15

        comp20 = create_metrics_metrics_component_20()
        assert comp20.metrics_id == 20

