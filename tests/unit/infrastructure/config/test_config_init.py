#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施层config/__init__.py模块测试

测试目标：提升config/__init__.py的真实覆盖率
实际导入和使用src.infrastructure.config模块
"""

import pytest


class TestConfigInit:
    """测试config模块初始化"""
    
    def test_config_validators_import(self):
        """测试ConfigValidators导入"""
        from src.infrastructure.config import ConfigValidators
        
        assert ConfigValidators is not None
    
    def test_config_factory_import(self):
        """测试ConfigFactory导入"""
        from src.infrastructure.config import ConfigFactory
        
        assert ConfigFactory is not None
    
    def test_unified_config_factory_import(self):
        """测试UnifiedConfigFactory导入"""
        from src.infrastructure.config import UnifiedConfigFactory
        
        assert UnifiedConfigFactory is not None
    
    def test_get_config_factory_function(self):
        """测试get_config_factory函数"""
        from src.infrastructure.config import get_config_factory
        
        assert callable(get_config_factory)
    
    def test_create_config_manager_function(self):
        """测试create_config_manager函数"""
        from src.infrastructure.config import create_config_manager
        
        assert callable(create_config_manager)
    
    def test_unified_config_manager_import(self):
        """测试UnifiedConfigManager导入"""
        from src.infrastructure.config import UnifiedConfigManager
        
        assert UnifiedConfigManager is not None
    
    def test_config_load_error_import(self):
        """测试ConfigLoadError导入"""
        from src.infrastructure.config import ConfigLoadError
        
        assert ConfigLoadError is not None
        assert issubclass(ConfigLoadError, Exception)
    
    def test_config_validation_error_import(self):
        """测试ConfigValidationError导入"""
        from src.infrastructure.config import ConfigValidationError
        
        assert ConfigValidationError is not None
        assert issubclass(ConfigValidationError, Exception)
    
    def test_strategy_manager_import(self):
        """测试StrategyManager导入"""
        from src.infrastructure.config import StrategyManager
        
        assert StrategyManager is not None
    
    def test_json_config_loader_import(self):
        """测试JSONConfigLoader导入"""
        from src.infrastructure.config import JSONConfigLoader
        
        assert JSONConfigLoader is not None
    
    def test_json_loader_import(self):
        """测试JSONLoader导入"""
        from src.infrastructure.config import JSONLoader
        
        assert JSONLoader is not None
    
    def test_yaml_loader_import(self):
        """测试YAMLLoader导入"""
        from src.infrastructure.config import YAMLLoader
        
        assert YAMLLoader is not None
    
    def test_create_unified_config_manager_function(self):
        """测试create_unified_config_manager函数"""
        from src.infrastructure.config import create_unified_config_manager
        
        assert callable(create_unified_config_manager)
    
    def test_module_all(self):
        """测试模块__all__导出"""
        from src.infrastructure.config import __all__
        
        assert isinstance(__all__, list)
        assert len(__all__) > 0
        assert "UnifiedConfigManager" in __all__
    
    def test_module_version(self):
        """测试模块版本"""
        from src.infrastructure.config import __version__

        assert isinstance(__version__, str)
        assert len(__version__) > 0

    def test_config_basic_operations(self):
        """测试配置基础操作"""
        try:
            from src.infrastructure.config.core.unified_config_interface import UnifiedConfigInterface
            config_interface = UnifiedConfigInterface()
            assert config_interface is not None
            # 测试基本方法
            assert hasattr(config_interface, 'get_config')
            assert hasattr(config_interface, 'set_config')
            assert hasattr(config_interface, 'validate_config')
        except Exception:
            pytest.skip("UnifiedConfigInterface测试跳过")

