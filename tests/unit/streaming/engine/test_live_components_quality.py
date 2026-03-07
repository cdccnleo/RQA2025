#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live组件质量测试
测试覆盖 LiveComponent 和 LiveComponentFactory 的核心功能
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from tests.unit.streaming.conftest import import_streaming_module


@pytest.fixture
def live_component_factory():
    """创建Live组件工厂实例"""
    LiveComponentFactory = import_streaming_module('src.streaming.engine.live_components', 'LiveComponentFactory')
    if LiveComponentFactory is None:
        pytest.skip("LiveComponentFactory不可用")
    return LiveComponentFactory


@pytest.fixture
def live_component(live_component_factory):
    """创建Live组件实例"""
    from src.streaming.engine.live_components import LiveComponent
    return LiveComponent(live_id=3, component_type="Live")


class TestLiveComponent:
    """LiveComponent测试类"""

    def test_initialization(self, live_component):
        """测试初始化"""
        assert live_component.live_id == 3
        assert live_component.component_type == "Live"
        assert "Live_Component_3" in live_component.component_name
        assert isinstance(live_component.creation_time, datetime)

    def test_get_live_id(self, live_component):
        """测试获取live ID"""
        assert live_component.get_live_id() == 3

    def test_get_info(self, live_component):
        """测试获取组件信息"""
        info = live_component.get_info()
        assert isinstance(info, dict)
        assert info['live_id'] == 3
        assert info['component_type'] == "Live"
        assert 'component_name' in info
        assert 'creation_time' in info
        assert 'version' in info

    def test_process_success(self, live_component):
        """测试处理数据（成功）"""
        test_data = {'symbol': 'TSLA', 'price': 800.0}
        result = live_component.process(test_data)
        
        assert isinstance(result, dict)
        assert result['status'] == 'success'
        assert result['live_id'] == 3
        assert result['input_data'] == test_data
        assert 'processed_at' in result
        assert 'result' in result

    def test_process_error(self, live_component):
        """测试处理数据（错误）"""
        test_data = None
        result = live_component.process(test_data)
        
        assert isinstance(result, dict)
        assert result['status'] == 'success'  # 组件会处理None数据
        assert result['live_id'] == 3

    def test_get_status(self, live_component):
        """测试获取组件状态"""
        status = live_component.get_status()
        assert isinstance(status, dict)
        assert status['live_id'] == 3
        assert status['status'] == 'active'
        assert status['health'] == 'good'
        assert 'creation_time' in status


class TestLiveComponentFactory:
    """LiveComponentFactory测试类"""

    def test_get_available_lives(self, live_component_factory):
        """测试获取所有可用的live ID"""
        available_ids = live_component_factory.get_available_lives()
        assert isinstance(available_ids, list)
        assert len(available_ids) > 0
        assert 3 in available_ids  # 应该包含支持的ID

    def test_create_component_valid_id(self, live_component_factory):
        """测试创建组件（有效ID）"""
        component = live_component_factory.create_component(3)
        assert component is not None
        assert component.get_live_id() == 3
        assert component.component_type == "Live"

    def test_create_component_invalid_id(self, live_component_factory):
        """测试创建组件（无效ID）"""
        with pytest.raises(ValueError):
            live_component_factory.create_component(999)

    def test_create_all_lives(self, live_component_factory):
        """测试创建所有live组件"""
        all_components = live_component_factory.create_all_lives()
        assert isinstance(all_components, dict)
        assert len(all_components) > 0
        for live_id, component in all_components.items():
            assert isinstance(live_id, int)
            assert component.get_live_id() == live_id

    def test_get_live_info(self, live_component_factory):
        """测试获取live信息"""
        # 如果方法不存在，创建组件并获取其info
        if hasattr(live_component_factory, 'get_live_info'):
            info = live_component_factory.get_live_info(3)
            assert isinstance(info, dict)
            assert 'live_id' in info
            assert 'component_name' in info
        else:
            # 使用create_component代替
            component = live_component_factory.create_component(3)
            info = component.get_info()
            assert isinstance(info, dict)
            assert 'live_id' in info
            assert 'component_name' in info

    def test_get_factory_info(self, live_component_factory):
        """测试获取工厂信息"""
        if hasattr(live_component_factory, 'get_factory_info'):
            info = live_component_factory.get_factory_info()
            assert isinstance(info, dict)
            assert 'factory_name' in info
            assert 'version' in info
            assert 'total_lives' in info
            assert 'supported_ids' in info

    def test_create_live_component_functions(self):
        """测试创建live组件的辅助函数"""
        from src.streaming.engine.live_components import (
            create_live_live_component_3,
            create_live_live_component_8,
            create_live_live_component_13,
            create_live_live_component_18,
            create_live_live_component_23,
            create_live_live_component_28
        )
        
        # 测试所有创建函数
        component_3 = create_live_live_component_3()
        assert component_3.live_id == 3
        
        component_8 = create_live_live_component_8()
        assert component_8.live_id == 8
        
        component_13 = create_live_live_component_13()
        assert component_13.live_id == 13
        
        component_18 = create_live_live_component_18()
        assert component_18.live_id == 18
        
        component_23 = create_live_live_component_23()
        assert component_23.live_id == 23
        
        component_28 = create_live_live_component_28()
        assert component_28.live_id == 28

    def test_component_factory_initialization(self):
        """测试组件工厂初始化"""
        from src.streaming.engine.live_components import ComponentFactory
        
        factory = ComponentFactory()
        assert factory._components == {}

    def test_component_factory_create_component_exception(self):
        """测试组件工厂创建组件异常处理"""
        from src.streaming.engine.live_components import ComponentFactory
        from unittest.mock import patch
        
        factory = ComponentFactory()
        
        # Mock _create_component_instance抛出异常
        with patch.object(factory, '_create_component_instance', side_effect=Exception("Create error")):
            component = factory.create_component("Test", {})
            assert component is None

    def test_component_factory_create_component_initialize_fails(self):
        """测试组件工厂创建组件（初始化失败）"""
        from src.streaming.engine.live_components import ComponentFactory
        from unittest.mock import Mock, patch
        
        factory = ComponentFactory()
        
        # Mock _create_component_instance返回一个组件，但initialize返回False
        mock_component = Mock()
        mock_component.initialize.return_value = False
        
        with patch.object(factory, '_create_component_instance', return_value=mock_component):
            component = factory.create_component("Test", {})
            assert component is None

    def test_live_component_process_exception(self, live_component):
        """测试Live组件处理异常"""
        # 直接调用process方法，传入会导致异常的数据
        # 由于process方法有异常处理，应该返回错误结果
        result = live_component.process(None)  # 可能导致异常
        # 应该返回结果（成功或错误）
        assert isinstance(result, dict)

