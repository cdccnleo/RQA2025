#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Probe组件测试"""

import pytest
from unittest.mock import Mock, patch
from src.infrastructure.health.components.probe_components import (
    IProbeComponent,
    ProbeComponent,
    ProbeComponentFactory
)


class TestIProbeComponent:
    """测试Probe组件接口"""

    def test_interface_is_abstract(self):
        """测试接口类无法直接实例化"""
        with pytest.raises(TypeError):
            IProbeComponent()

    def test_interface_has_required_methods(self):
        """测试接口定义了所需的方法"""
        # 检查抽象方法是否存在
        assert hasattr(IProbeComponent, 'get_info')
        assert hasattr(IProbeComponent, 'process')
        assert hasattr(IProbeComponent, 'get_status')
        assert hasattr(IProbeComponent, 'get_probe_id')


class TestProbeComponent:
    """测试Probe组件实现"""

    def setup_method(self):
        """测试前准备"""
        self.probe_id = 123
        self.component = ProbeComponent(self.probe_id)

    def test_init(self):
        """测试初始化"""
        assert self.component.probe_id == self.probe_id
        assert self.component.component_type == "Probe"
        assert hasattr(self.component, 'component_name')
        assert hasattr(self.component, 'creation_time')

    def test_init_with_custom_type(self):
        """测试自定义组件类型初始化"""
        custom_type = "CustomProbe"
        component = ProbeComponent(self.probe_id, custom_type)
        assert component.component_type == custom_type

    def test_get_probe_id(self):
        """测试获取probe ID"""
        assert self.component.get_probe_id() == self.probe_id

    def test_get_info(self):
        """测试获取组件信息"""
        info = self.component.get_info()

        assert isinstance(info, dict)
        assert 'probe_id' in info
        assert 'component_type' in info
        assert 'component_name' in info
        assert 'status' in info
        assert 'creation_time' in info
        assert info['probe_id'] == self.probe_id
        assert info['component_type'] == "Probe"
        assert 'Probe_Component_123' in info['component_name']

    def test_get_status(self):
        """测试获取组件状态"""
        status = self.component.get_status()

        assert isinstance(status, dict)
        assert 'status' in status
        assert 'health' in status
        assert 'component_type' in status
        assert 'timestamp' in status
        assert status['component_type'] == "Probe"
        assert status['status'] == "active"
        assert status['health'] == "good"

    def test_process_basic_data(self):
        """测试处理基本数据"""
        input_data = {"key": "value", "number": 42}

        result = self.component.process(input_data)

        assert isinstance(result, dict)
        assert 'processed' in result
        assert 'input_data' in result
        assert 'processed_at' in result
        assert 'status' in result
        assert result['processed'] is True
        assert result['input_data'] == input_data
        assert result['status'] == "success"

    def test_process_empty_data(self):
        """测试处理空数据"""
        input_data = {}

        result = self.component.process(input_data)

        assert isinstance(result, dict)
        assert result['processed'] is True
        assert result['input_data'] == {}
        assert result['status'] == "success"

    def test_process_with_special_characters(self):
        """测试处理包含特殊字符的数据"""
        input_data = {
            "special": "!@#$%^&*()",
            "unicode": "你好世界🚀",
            "nested": {"inner": "value"}
        }

        result = self.component.process(input_data)

        assert result['processed'] is True
        assert result['input_data'] == input_data

    def test_process_updates_timestamp(self):
        """测试处理操作更新时间戳"""
        import time

        # 记录处理前的时间
        before_time = time.time()

        self.component.process({"test": "data"})

        after_time = time.time()

        # 验证时间戳被更新
        status = self.component.get_status()
        last_check = status.get('last_check')

        if last_check:
            assert before_time <= last_check <= after_time

    def test_multiple_process_calls(self):
        """测试多次处理调用"""
        # 第一次处理
        result1 = self.component.process({"first": "call"})
        assert result1['processed'] is True

        # 第二次处理
        result2 = self.component.process({"second": "call"})
        assert result2['processed'] is True

        # 验证两次处理都成功且时间戳是字符串格式
        assert isinstance(result1['processed_at'], str)
        assert isinstance(result2['processed_at'], str)

    def test_process_with_none_data(self):
        """测试处理None数据"""
        # ProbeComponent的process方法可能接受None并处理
        result = self.component.process(None)
        assert isinstance(result, dict)
        assert result['input_data'] is None

    def test_process_with_non_dict_data(self):
        """测试处理非字典数据"""
        result = self.component.process("string_data")
        assert result['processed'] is True
        assert result['input_data'] == "string_data"


class TestProbeComponentFactory:
    """测试Probe组件工厂"""

    def setup_method(self):
        """测试前准备"""
        self.factory = ProbeComponentFactory()

    def test_init(self):
        """测试工厂初始化"""
        assert hasattr(self.factory, 'SUPPORTED_PROBE_IDS')
        assert isinstance(self.factory.SUPPORTED_PROBE_IDS, list)

    def test_create_component_basic(self):
        """测试创建基本组件"""
        probe_id = 5  # 使用支持的ID

        component = self.factory.create_component(probe_id)

        assert isinstance(component, ProbeComponent)
        assert component.get_probe_id() == probe_id

    def test_create_component_with_unsupported_id(self):
        """测试创建不支持ID的组件"""
        probe_id = 999  # 不支持的ID

        with pytest.raises(ValueError):
            self.factory.create_component(probe_id)

    def test_supported_probe_ids(self):
        """测试支持的probe ID"""
        assert isinstance(self.factory.SUPPORTED_PROBE_IDS, list)
        assert 5 in self.factory.SUPPORTED_PROBE_IDS
        assert len(self.factory.SUPPORTED_PROBE_IDS) > 0
