#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monitor组件测试覆盖
测试monitoring/monitor_components.py
"""

import pytest
from datetime import datetime
from typing import Dict, Any
from unittest.mock import patch, MagicMock

from src.features.monitoring.monitor_components import (
    IMonitorComponent,
    MonitorComponent,
    MonitorComponentFactory,
    create_monitor_monitor_component_6,
    create_monitor_monitor_component_11,
    create_monitor_monitor_component_16,
    create_monitor_monitor_component_21,
)


class TestMonitorComponent:
    """Monitor组件测试"""

    def test_monitor_component_initialization(self):
        """测试Monitor组件初始化"""
        component = MonitorComponent(monitor_id=6)
        assert component.monitor_id == 6
        assert component.component_type == "Monitor"
        assert component.component_name == "Monitor_Component_6"
        assert isinstance(component.creation_time, datetime)

    def test_monitor_component_get_monitor_id(self):
        """测试获取monitor ID"""
        component = MonitorComponent(monitor_id=11)
        assert component.get_monitor_id() == 11

    def test_monitor_component_get_info(self):
        """测试获取组件信息"""
        component = MonitorComponent(monitor_id=16)
        info = component.get_info()
        assert info["monitor_id"] == 16
        assert info["component_name"] == "Monitor_Component_16"
        assert info["component_type"] == "Monitor"
        assert "creation_time" in info
        assert info["version"] == "2.0.0"
        assert info["type"] == "unified_features_monitoring_component"

    def test_monitor_component_process_success(self):
        """测试处理数据成功"""
        component = MonitorComponent(monitor_id=21)
        data = {"key": "value", "number": 456}
        result = component.process(data)
        assert result["monitor_id"] == 21
        assert result["status"] == "success"
        assert result["input_data"] == data
        assert "processed_at" in result
        assert result["processing_type"] == "unified_monitor_processing"

    def test_monitor_component_process_with_exception(self):
        """测试处理数据时异常处理"""
        from unittest.mock import patch, MagicMock
        from datetime import datetime
        component = MonitorComponent(monitor_id=6)
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
        with patch('src.features.monitoring.monitor_components.datetime', mock_datetime_obj):
            result = component.process(data)
            assert result["monitor_id"] == 6
            assert result["status"] == "error"
            assert "error" in result
            assert "error_type" in result

    def test_monitor_component_get_status(self):
        """测试获取组件状态"""
        component = MonitorComponent(monitor_id=6)
        status = component.get_status()
        assert status["monitor_id"] == 6
        assert status["status"] == "active"
        assert status["health"] == "good"
        assert "creation_time" in status

    def test_monitor_component_implements_interface(self):
        """测试MonitorComponent实现接口"""
        component = MonitorComponent(monitor_id=11)
        assert isinstance(component, IMonitorComponent)
        assert hasattr(component, 'get_info')
        assert hasattr(component, 'process')
        assert hasattr(component, 'get_status')
        assert hasattr(component, 'get_monitor_id')


class TestMonitorComponentFactory:
    """MonitorComponentFactory测试"""

    def test_factory_create_component(self):
        """测试工厂创建组件"""
        component = MonitorComponentFactory.create_component(6)
        assert isinstance(component, MonitorComponent)
        assert component.monitor_id == 6

    def test_factory_create_invalid_id(self):
        """测试工厂创建无效ID组件"""
        with pytest.raises(ValueError, match="不支持的monitor ID"):
            MonitorComponentFactory.create_component(99)

    def test_factory_get_available_monitors(self):
        """测试获取所有可用的monitor ID"""
        available = MonitorComponentFactory.get_available_monitors()
        assert isinstance(available, list)
        assert len(available) == 5
        assert 1 in available
        assert 6 in available
        assert 11 in available
        assert 16 in available
        assert 21 in available

    def test_factory_create_all_monitors(self):
        """测试创建所有可用monitor"""
        all_monitors = MonitorComponentFactory.create_all_monitors()
        assert isinstance(all_monitors, dict)
        assert len(all_monitors) == 5
        for monitor_id, component in all_monitors.items():
            assert isinstance(component, MonitorComponent)
            assert component.monitor_id == monitor_id

    def test_factory_get_factory_info(self):
        """测试获取工厂信息"""
        info = MonitorComponentFactory.get_factory_info()
        assert info["factory_name"] == "MonitorComponentFactory"
        assert info["version"] == "2.0.0"
        assert info["total_monitors"] == 5
        assert len(info["supported_ids"]) == 5
        assert "created_at" in info

    def test_factory_backward_compatibility_functions(self):
        """测试向后兼容函数"""
        comp6 = create_monitor_monitor_component_6()
        assert comp6.monitor_id == 6

        comp11 = create_monitor_monitor_component_11()
        assert comp11.monitor_id == 11

        comp16 = create_monitor_monitor_component_16()
        assert comp16.monitor_id == 16

        comp21 = create_monitor_monitor_component_21()
        assert comp21.monitor_id == 21

