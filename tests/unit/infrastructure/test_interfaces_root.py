#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施层根目录接口定义组件测试

测试目标：提升interfaces.py的真实覆盖率
实际导入和使用src.infrastructure.interfaces模块
"""

import pytest
from unittest.mock import Mock
import sys
import os

# 直接导入interfaces.py文件（不是interfaces包）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../src/infrastructure'))


class TestIInfrastructureComponent:
    """测试基础设施组件基础接口"""
    
    def test_interface_definition(self):
        """测试接口定义"""
        # 直接导入interfaces.py文件
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "interfaces_module",
            os.path.join(os.path.dirname(__file__), '../../../src/infrastructure/interfaces.py')
        )
        interfaces_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(interfaces_module)
        
        # 接口是抽象类，不能直接实例化
        assert interfaces_module.IInfrastructureComponent is not None
    
    def test_interface_methods(self):
        """测试接口方法定义"""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "interfaces_module",
            os.path.join(os.path.dirname(__file__), '../../../src/infrastructure/interfaces.py')
        )
        interfaces_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(interfaces_module)
        
        # 检查接口方法（抽象方法可能不在dir中，但可以通过hasattr检查）
        assert hasattr(interfaces_module.IInfrastructureComponent, 'get_status')
        assert hasattr(interfaces_module.IInfrastructureComponent, 'health_check')


class TestIConfigManagerComponent:
    """测试配置管理器接口"""
    
    def test_interface_definition(self):
        """测试接口定义"""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "interfaces_module",
            os.path.join(os.path.dirname(__file__), '../../../src/infrastructure/interfaces.py')
        )
        interfaces_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(interfaces_module)
        
        assert interfaces_module.IConfigManagerComponent is not None
    
    def test_interface_methods(self):
        """测试接口方法定义"""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "interfaces_module",
            os.path.join(os.path.dirname(__file__), '../../../src/infrastructure/interfaces.py')
        )
        interfaces_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(interfaces_module)
        
        # 检查接口方法
        assert hasattr(interfaces_module.IConfigManagerComponent, 'get_config')
        assert hasattr(interfaces_module.IConfigManagerComponent, 'set_config')
        assert hasattr(interfaces_module.IConfigManagerComponent, 'reload_config')


class TestICacheManagerComponent:
    """测试缓存管理器接口"""
    
    def test_interface_definition(self):
        """测试接口定义"""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "interfaces_module",
            os.path.join(os.path.dirname(__file__), '../../../src/infrastructure/interfaces.py')
        )
        interfaces_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(interfaces_module)
        
        assert interfaces_module.ICacheManagerComponent is not None
    
    def test_interface_methods(self):
        """测试接口方法定义"""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "interfaces_module",
            os.path.join(os.path.dirname(__file__), '../../../src/infrastructure/interfaces.py')
        )
        interfaces_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(interfaces_module)
        
        # 检查接口方法
        assert hasattr(interfaces_module.ICacheManagerComponent, 'get')
        assert hasattr(interfaces_module.ICacheManagerComponent, 'set')
        assert hasattr(interfaces_module.ICacheManagerComponent, 'delete')
        assert hasattr(interfaces_module.ICacheManagerComponent, 'clear')


class TestILoggerComponent:
    """测试日志管理器接口"""
    
    def test_interface_definition(self):
        """测试接口定义"""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "interfaces_module",
            os.path.join(os.path.dirname(__file__), '../../../src/infrastructure/interfaces.py')
        )
        interfaces_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(interfaces_module)
        
        assert interfaces_module.ILoggerComponent is not None
    
    def test_interface_methods(self):
        """测试接口方法定义"""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "interfaces_module",
            os.path.join(os.path.dirname(__file__), '../../../src/infrastructure/interfaces.py')
        )
        interfaces_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(interfaces_module)
        
        # 检查接口方法
        assert hasattr(interfaces_module.ILoggerComponent, 'info')
        assert hasattr(interfaces_module.ILoggerComponent, 'error')
        assert hasattr(interfaces_module.ILoggerComponent, 'warning')
        assert hasattr(interfaces_module.ILoggerComponent, 'debug')


class TestISecurityManagerComponent:
    """测试安全管理器接口"""
    
    def test_interface_definition(self):
        """测试接口定义"""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "interfaces_module",
            os.path.join(os.path.dirname(__file__), '../../../src/infrastructure/interfaces.py')
        )
        interfaces_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(interfaces_module)
        
        assert interfaces_module.ISecurityManagerComponent is not None
    
    def test_interface_methods(self):
        """测试接口方法定义"""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "interfaces_module",
            os.path.join(os.path.dirname(__file__), '../../../src/infrastructure/interfaces.py')
        )
        interfaces_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(interfaces_module)
        
        # 检查接口方法
        assert hasattr(interfaces_module.ISecurityManagerComponent, 'authenticate')

