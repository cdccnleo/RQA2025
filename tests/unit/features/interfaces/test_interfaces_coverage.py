#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征层接口模块测试覆盖
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from src.features.interfaces.interfaces import IFeaturesComponent


class TestIFeaturesComponent:
    """特征组件接口测试"""

    def test_interface_is_abstract(self):
        """测试接口是抽象类"""
        assert hasattr(IFeaturesComponent, '__abstractmethods__')
        assert 'get_status' in IFeaturesComponent.__abstractmethods__
        assert 'health_check' in IFeaturesComponent.__abstractmethods__

    def test_cannot_instantiate_interface(self):
        """测试不能直接实例化接口"""
        with pytest.raises(TypeError):
            IFeaturesComponent()

    def test_concrete_implementation(self):
        """测试具体实现"""
        class ConcreteComponent(IFeaturesComponent):
            def get_status(self):
                return {"status": "ok"}
            
            def health_check(self):
                return {"healthy": True}
        
        component = ConcreteComponent()
        assert component.get_status() == {"status": "ok"}
        assert component.health_check() == {"healthy": True}

    def test_incomplete_implementation(self):
        """测试不完整实现"""
        class IncompleteComponent(IFeaturesComponent):
            def get_status(self):
                return {"status": "ok"}
            # 缺少health_check方法
        
        with pytest.raises(TypeError):
            IncompleteComponent()

