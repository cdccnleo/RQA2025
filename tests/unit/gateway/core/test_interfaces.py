#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
网关层接口测试

测试目标：提升interfaces.py的覆盖率
"""

import sys
import importlib
from pathlib import Path
import pytest
from unittest.mock import Mock

# 确保Python路径正确配置
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
project_root_str = str(project_root)
src_path_str = str(project_root / "src")

if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

# 动态导入网关接口模块
try:
    interfaces_module = importlib.import_module('src.gateway.core.interfaces')
    IGatewayComponent = getattr(interfaces_module, 'IGatewayComponent', None)
    if IGatewayComponent is None:
        pytest.skip("网关接口模块不可用", allow_module_level=True)
except ImportError:
    pytest.skip("网关接口模块导入失败", allow_module_level=True)


class TestIGatewayComponent:
    """测试网关组件接口"""
    
    def test_interface_cannot_instantiate(self):
        """测试接口不能直接实例化"""
        with pytest.raises(TypeError):
            IGatewayComponent()
    
    def test_interface_has_get_status_method(self):
        """测试接口有get_status方法"""
        assert hasattr(IGatewayComponent, 'get_status')
        assert callable(getattr(IGatewayComponent, 'get_status'))
    
    def test_interface_has_health_check_method(self):
        """测试接口有health_check方法"""
        assert hasattr(IGatewayComponent, 'health_check')
        assert callable(getattr(IGatewayComponent, 'health_check'))
    
    def test_concrete_implementation(self):
        """测试具体实现"""
        class ConcreteGatewayComponent(IGatewayComponent):
            def get_status(self):
                return {"status": "ok"}
            
            def health_check(self):
                return {"health": "healthy"}
        
        component = ConcreteGatewayComponent()
        assert component.get_status() == {"status": "ok"}
        assert component.health_check() == {"health": "healthy"}
    
    def test_incomplete_implementation(self):
        """测试不完整实现"""
        class IncompleteComponent(IGatewayComponent):
            def get_status(self):
                return {"status": "ok"}
            # 缺少health_check方法
        
        with pytest.raises(TypeError):
            IncompleteComponent()

