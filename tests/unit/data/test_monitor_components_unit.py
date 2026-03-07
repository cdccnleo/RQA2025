#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MonitorComponents 单元测试
覆盖MonitorComponent和MonitorComponentFactory的核心功能。
"""

import asyncio
import pandas as pd
from unittest.mock import Mock

# Mock数据管理器模块以绕过复杂的导入问题
mock_data_manager = Mock()
mock_data_manager.DataManager = Mock()
mock_data_manager.DataLoaderError = Exception

# 配置DataManager实例方法
mock_instance = Mock()
mock_instance.validate_all_configs.return_value = True
mock_instance.health_check.return_value = {"status": "healthy"}
mock_instance.store_data.return_value = True
mock_instance.has_data.return_value = True
mock_instance.get_metadata.return_value = {"data_type": "test", "symbol": "X"}
mock_instance.retrieve_data.return_value = pd.DataFrame({"col": [1, 2, 3]})
mock_instance.get_stats.return_value = {"total_items": 1}
mock_instance.validate_data.return_value = {"valid": True}
mock_instance.shutdown.return_value = None

mock_data_manager.DataManager.return_value = mock_instance

# Mock整个模块
import sys
sys.modules["src.data.data_manager"] = mock_data_manager


import pytest
from datetime import datetime
from typing import Dict, Any

from src.data.quality.monitor_components import (
    IMonitorComponent,
    MonitorComponent,
    MonitorComponentFactory,
)


class TestMonitorComponent:
    """MonitorComponent 测试"""

    def test_get_info_and_status(self):
        """测试获取组件信息和状态"""
        component = MonitorComponent(monitor_id=4)
        
        info = component.get_info()
        assert info["monitor_id"] == 4
        assert info["component_name"] == "Monitor_Component_4"
        assert info["component_type"] == "Monitor"
        assert "creation_time" in info
        
        status = component.get_status()
        assert status["monitor_id"] == 4
        assert status["status"] == "active"
        assert status["health"] == "good"

    def test_process_success_and_error(self):
        """测试数据处理成功和错误场景"""
        component = MonitorComponent(monitor_id=9)
        test_data = {"field": "value", "number": 42}
        
        # 正常处理
        result = component.process(test_data)
        assert result["monitor_id"] == 9
        assert result["status"] == "success"
        assert result["input_data"] == test_data
        assert "processed_at" in result
        
        # 验证边界情况处理
        # 测试空数据
        empty_result = component.process({})
        assert empty_result["status"] == "success"
        
        # 测试None数据（会被转换为字典）
        none_result = component.process(None)
        assert none_result["status"] in ["success", "error"]  # process内部会处理
        
        # 测试大量数据
        large_data = {f"key_{i}": f"value_{i}" for i in range(100)}
        large_result = component.process(large_data)
        assert large_result["status"] == "success"

    def test_get_monitor_id(self):
        """测试获取monitor ID"""
        component = MonitorComponent(monitor_id=19)
        assert component.get_monitor_id() == 19


class TestMonitorComponentFactory:
    """MonitorComponentFactory 测试"""

    def test_create_component_with_valid_id(self):
        """测试使用有效ID创建组件"""
        component = MonitorComponentFactory.create_component(4)
        assert isinstance(component, MonitorComponent)
        assert component.monitor_id == 4
        assert component.component_type == "Monitor"

    def test_create_component_with_invalid_id(self):
        """测试使用无效ID创建组件应抛出异常"""
        with pytest.raises(ValueError, match="不支持的monitor ID"):
            MonitorComponentFactory.create_component(999)

    def test_get_available_monitors(self):
        """测试获取所有可用的monitor ID"""
        ids = MonitorComponentFactory.get_available_monitors()
        assert isinstance(ids, list)
        assert len(ids) > 0
        assert all(isinstance(id_val, int) for id_val in ids)

    def test_create_all_monitors(self):
        """测试创建所有monitor组件"""
        all_components = MonitorComponentFactory.create_all_monitors()
        assert isinstance(all_components, dict)
        assert len(all_components) == len(MonitorComponentFactory.SUPPORTED_MONITOR_IDS)
        
        for monitor_id, component in all_components.items():
            assert isinstance(component, MonitorComponent)
            assert component.monitor_id == monitor_id

    def test_get_factory_info(self):
        """测试获取工厂信息"""
        info = MonitorComponentFactory.get_factory_info()
        assert info["factory_name"] == "MonitorComponentFactory"
        assert info["version"] == "2.0.0"
        assert info["total_monitors"] == len(MonitorComponentFactory.SUPPORTED_MONITOR_IDS)
        assert "supported_ids" in info
        assert "created_at" in info


class TestMonitorComponentLegacyFunctions:
    """向后兼容函数的测试"""

    def test_legacy_create_functions(self):
        """测试向后兼容的创建函数"""
        from src.data.quality.monitor_components import (
            create_monitor_monitor_component_4,
            create_monitor_monitor_component_9,
            create_monitor_monitor_component_14,
        )
        
        comp4 = create_monitor_monitor_component_4()
        assert isinstance(comp4, MonitorComponent)
        assert comp4.monitor_id == 4
        
        comp9 = create_monitor_monitor_component_9()
        assert comp9.monitor_id == 9
        
        comp14 = create_monitor_monitor_component_14()
        assert comp14.monitor_id == 14

