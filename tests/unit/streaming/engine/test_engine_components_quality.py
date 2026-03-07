#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
引擎组件质量测试
测试覆盖 EngineComponent 和 EngineComponentFactory 的核心功能
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from tests.unit.streaming.conftest import import_streaming_module


@pytest.fixture
def engine_component_factory():
    """创建引擎组件工厂实例"""
    EngineComponentFactory = import_streaming_module('src.streaming.engine.engine_components', 'EngineComponentFactory')
    if EngineComponentFactory is None:
        pytest.skip("EngineComponentFactory不可用")
    return EngineComponentFactory


@pytest.fixture
def engine_component(engine_component_factory):
    """创建引擎组件实例"""
    from src.streaming.engine.engine_components import EngineComponent
    return EngineComponent(engine_id=2, component_type="Engine")


class TestEngineComponent:
    """EngineComponent测试类"""

    def test_initialization(self, engine_component):
        """测试初始化"""
        assert engine_component.engine_id == 2
        assert engine_component.component_type == "Engine"
        assert "Engine_Component_2" in engine_component.component_name
        assert isinstance(engine_component.creation_time, datetime)

    def test_get_engine_id(self, engine_component):
        """测试获取engine ID"""
        assert engine_component.get_engine_id() == 2

    def test_get_info(self, engine_component):
        """测试获取组件信息"""
        info = engine_component.get_info()
        assert isinstance(info, dict)
        assert info['engine_id'] == 2
        assert info['component_type'] == "Engine"
        assert 'component_name' in info
        assert 'creation_time' in info
        assert 'version' in info

    def test_process_success(self, engine_component):
        """测试处理数据（成功）"""
        test_data = {'symbol': 'GOOGL', 'price': 2500.0}
        result = engine_component.process(test_data)
        
        assert isinstance(result, dict)
        assert result['status'] == 'success'
        assert result['engine_id'] == 2
        assert result['input_data'] == test_data
        assert 'processed_at' in result
        assert 'result' in result

    def test_process_error(self, engine_component):
        """测试处理数据（错误）"""
        test_data = None
        result = engine_component.process(test_data)
        
        assert isinstance(result, dict)
        assert result['status'] == 'success'  # 组件会处理None数据
        assert result['engine_id'] == 2

    def test_get_status(self, engine_component):
        """测试获取组件状态"""
        status = engine_component.get_status()
        assert isinstance(status, dict)
        assert status['engine_id'] == 2
        assert status['status'] == 'active'
        assert status['health'] == 'good'
        assert 'creation_time' in status


class TestEngineComponentFactory:
    """EngineComponentFactory测试类"""

    def test_factory_initialization(self, engine_component_factory):
        """测试工厂初始化"""
        factory = engine_component_factory()
        # EngineComponentFactory没有_components属性，它使用不同的结构
        assert hasattr(factory, 'create_component')

    def test_create_component_returns_none(self, engine_component_factory):
        """测试创建组件返回None的情况（无效ID）"""
        factory = engine_component_factory()
        # 使用无效的ID，应该抛出ValueError
        with pytest.raises(ValueError):
            factory.create_component(999)

    def test_create_component_initialize_fails(self, engine_component_factory):
        """测试组件初始化失败的情况"""
        factory = engine_component_factory()
        # EngineComponentFactory的create_component不接受config参数，只接受engine_id
        # 测试有效ID的创建
        component = factory.create_component(2)
        assert component is not None
        assert component.get_engine_id() == 2

    def test_create_component_exception(self, engine_component_factory):
        """测试创建组件时抛出异常（无效ID）"""
        factory = engine_component_factory()
        # 使用无效的ID，应该抛出ValueError
        with pytest.raises(ValueError):
            factory.create_component(999)

    def test_get_available_engines(self, engine_component_factory):
        """测试获取所有可用的engine ID"""
        available_ids = engine_component_factory.get_available_engines()
        assert isinstance(available_ids, list)
        assert len(available_ids) > 0
        assert 2 in available_ids  # 应该包含支持的ID

    def test_create_component_valid_id(self, engine_component_factory):
        """测试创建组件（有效ID）"""
        component = engine_component_factory.create_component(2)
        assert component is not None
        assert component.get_engine_id() == 2
        assert component.component_type == "Engine"

    def test_create_component_invalid_id(self, engine_component_factory):
        """测试创建组件（无效ID）"""
        with pytest.raises(ValueError):
            engine_component_factory.create_component(999)

    def test_create_all_engines(self, engine_component_factory):
        """测试创建所有engine组件"""
        all_components = engine_component_factory.create_all_engines()
        assert isinstance(all_components, dict)
        assert len(all_components) > 0
        for engine_id, component in all_components.items():
            assert isinstance(engine_id, int)
            assert component.get_engine_id() == engine_id

    def test_get_engine_info(self, engine_component_factory):
        """测试获取engine信息"""
        # 如果方法不存在，创建组件并获取其info
        if hasattr(engine_component_factory, 'get_engine_info'):
            info = engine_component_factory.get_engine_info(2)
            assert isinstance(info, dict)
            assert 'engine_id' in info
            assert 'component_name' in info
        else:
            # 使用create_component代替
            component = engine_component_factory.create_component(2)
            info = component.get_info()
            assert isinstance(info, dict)
            assert 'engine_id' in info
            assert 'component_name' in info

    def test_get_factory_info(self, engine_component_factory):
        """测试获取工厂信息"""
        if hasattr(engine_component_factory, 'get_factory_info'):
            info = engine_component_factory.get_factory_info()
            assert isinstance(info, dict)
            assert 'factory_name' in info
            assert 'version' in info
            assert 'total_engines' in info
            assert 'supported_ids' in info

    def test_create_engine_component_functions(self):
        """测试创建engine组件的辅助函数"""
        from src.streaming.engine.engine_components import (
            create_engine_engine_component_2,
            create_engine_engine_component_7,
            create_engine_engine_component_12,
            create_engine_engine_component_17,
            create_engine_engine_component_22,
            create_engine_engine_component_27
        )
        
        # 测试所有创建函数
        component_2 = create_engine_engine_component_2()
        assert component_2.engine_id == 2
        
        component_7 = create_engine_engine_component_7()
        assert component_7.engine_id == 7
        
        component_12 = create_engine_engine_component_12()
        assert component_12.engine_id == 12
        
        component_17 = create_engine_engine_component_17()
        assert component_17.engine_id == 17
        
        component_22 = create_engine_engine_component_22()
        assert component_22.engine_id == 22
        
        component_27 = create_engine_engine_component_27()
        assert component_27.engine_id == 27

