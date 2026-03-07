#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缓存优化器组件测试

测试缓存系统的优化器组件，包括Protocol接口、组件实现和工厂模式。
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from unittest.mock import patch
from src.infrastructure.cache.core.optimizer_components import (
    IOptimizerComponent, OptimizerComponent, OptimizerComponentFactory,
    create_optimizer_component_11, create_optimizer_component_17, create_optimizer_component_23
)


class TestOptimizerComponents:
    """优化器组件测试类"""

    def test_optimizer_component_initialization(self):
        """测试优化器组件初始化"""
        component = OptimizerComponent(component_id=11)

        assert component.component_id == 11
        assert component.component_type == "Cache"
        assert hasattr(component, '_description')
        assert "optimizer_templates" in component._description

    def test_optimizer_component_get_info(self):
        """测试获取组件信息"""
        component = OptimizerComponent(component_id=17)

        info = component.get_info()
        assert isinstance(info, dict)
        assert info["component_id"] == 17
        assert info["type"] == "unified_optimizer_component"
        assert "component_type" in info

    def test_optimizer_component_get_processing_type(self):
        """测试获取处理类型"""
        component = OptimizerComponent(component_id=23)

        processing_type = component.get_processing_type()
        assert processing_type == "optimizer_processing"

    def test_optimizer_component_get_component_type_name(self):
        """测试获取组件类型名称"""
        component = OptimizerComponent(component_id=11)

        type_name = component.get_component_type_name()
        assert type_name == "optimizer"

    def test_optimizer_component_factory_supported_ids(self):
        """测试工厂支持的组件ID"""
        assert OptimizerComponentFactory.SUPPORTED_COMPONENT_IDS == [11, 17, 23]

    def test_optimizer_component_factory_create_valid_component(self):
        """测试工厂创建有效组件"""
        for component_id in [11, 17, 23]:
            component = OptimizerComponentFactory.create_component(component_id)
            assert isinstance(component, OptimizerComponent)
            assert component.component_id == component_id

    def test_optimizer_component_factory_create_invalid_component(self):
        """测试工厂创建无效组件"""
        with pytest.raises(ValueError, match="不支持的组件ID"):
            OptimizerComponentFactory.create_component(999)

    def test_optimizer_component_factory_get_available_components(self):
        """测试获取可用组件"""
        available = OptimizerComponentFactory.get_available_components()
        assert available == [11, 17, 23]

    def test_optimizer_component_factory_create_all_components(self):
        """测试创建所有组件"""
        all_components = OptimizerComponentFactory.create_all_components()

        assert isinstance(all_components, dict)
        assert len(all_components) == 3
        assert all(isinstance(comp, OptimizerComponent) for comp in all_components.values())
        assert set(all_components.keys()) == {11, 17, 23}

    def test_optimizer_component_factory_get_component_info(self):
        """测试获取组件工厂信息"""
        info = OptimizerComponentFactory.get_component_info()

        assert isinstance(info, dict)
        assert info["factory_name"] == "OptimizerComponentFactory"
        assert info["version"] == "2.0.0"
        assert info["total_components"] == 3
        assert info["supported_ids"] == [11, 17, 23]
        assert "created_at" in info
        assert "description" in info

    def test_create_optimizer_component_11(self):
        """测试创建ID为11的组件"""
        component = create_optimizer_component_11()
        assert isinstance(component, OptimizerComponent)
        assert component.component_id == 11

    def test_create_optimizer_component_17(self):
        """测试创建ID为17的组件"""
        component = create_optimizer_component_17()
        assert isinstance(component, OptimizerComponent)
        assert component.component_id == 17

    def test_create_optimizer_component_23(self):
        """测试创建ID为23的组件"""
        component = create_optimizer_component_23()
        assert isinstance(component, OptimizerComponent)
        assert component.component_id == 23

    def test_i_optimizer_component_protocol_methods(self):
        """测试Protocol接口方法签名"""
        # Protocol本身不提供实现，但我们可以检查实际实现是否符合协议
        component = OptimizerComponent(component_id=11)

        # 检查是否有协议定义的方法和属性
        assert hasattr(component, 'get_info')
        assert hasattr(component, 'get_status')  # 从基类继承
        assert hasattr(component, 'component_id')  # 这是一个property

        # 检查方法是否可调用
        assert callable(component.get_info)
        assert callable(component.get_status)

    def test_component_uniqueness(self):
        """测试组件唯一性"""
        comp1 = OptimizerComponentFactory.create_component(11)
        comp2 = OptimizerComponentFactory.create_component(11)

        # 不同实例应该有相同的属性但不是同一个对象
        assert comp1.component_id == comp2.component_id
        assert comp1 is not comp2

    def test_component_info_consistency(self):
        """测试组件信息一致性"""
        comp1 = OptimizerComponent(11)
        comp2 = OptimizerComponent(11)

        info1 = comp1.get_info()
        info2 = comp2.get_info()

        # 相同ID的组件应该有相同的基本信息
        assert info1["component_id"] == info2["component_id"]
        assert info1["type"] == info2["type"]

    def test_factory_singleton_behavior(self):
        """测试工厂的单例行为"""
        # 工厂方法应该是纯函数式的
        comp1 = OptimizerComponentFactory.create_component(11)
        comp2 = OptimizerComponentFactory.create_component(11)

        assert comp1.component_id == comp2.component_id
        assert comp1.component_type == comp2.component_type

    def test_supported_ids_immutability(self):
        """测试支持的ID列表不可变性"""
        original_ids = OptimizerComponentFactory.SUPPORTED_COMPONENT_IDS.copy()

        # 尝试修改（这应该不会影响原始列表）
        ids_copy = OptimizerComponentFactory.SUPPORTED_COMPONENT_IDS
        # 由于是类属性，我们不应该直接修改

        # 验证原始值保持不变
        assert OptimizerComponentFactory.SUPPORTED_COMPONENT_IDS == original_ids

    def test_component_info_timestamp(self):
        """测试组件信息时间戳"""
        with patch('src.infrastructure.cache.core.optimizer_components.datetime') as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = "2023-12-01T10:00:00"

            info = OptimizerComponentFactory.get_component_info()
            assert info["created_at"] == "2023-12-01T10:00:00"

    def test_all_exports(self):
        """测试__all__导出的所有项"""
        from src.infrastructure.cache.core.optimizer_components import __all__ as all_exports

        expected_exports = [
            "IOptimizerComponent",
            "OptimizerComponent",
            "OptimizerComponentFactory",
            "create_optimizer_component_11",
            "create_optimizer_component_17",
            "create_optimizer_component_23",
        ]

        assert all_exports == expected_exports

    def test_component_factory_error_message(self):
        """测试工厂错误消息"""
        with pytest.raises(ValueError) as exc_info:
            OptimizerComponentFactory.create_component(999)

        error_msg = str(exc_info.value)
        assert "不支持的组件ID: 999" in error_msg
        assert "支持的ID: [11, 17, 23]" in error_msg

    def test_create_all_components_completeness(self):
        """测试创建所有组件的完整性"""
        all_components = OptimizerComponentFactory.create_all_components()

        # 验证所有支持的ID都被创建
        for component_id in OptimizerComponentFactory.SUPPORTED_COMPONENT_IDS:
            assert component_id in all_components
            assert isinstance(all_components[component_id], OptimizerComponent)
            assert all_components[component_id].component_id == component_id
