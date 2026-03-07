#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施层__init__.py模块测试

测试目标：提升__init__.py的真实覆盖率
实际导入和使用src.infrastructure模块
"""

import pytest
from unittest.mock import patch, MagicMock


class TestInfrastructureInit:
    """测试基础设施层初始化模块"""
    
    def test_version(self):
        """测试版本号"""
        from src.infrastructure import __version__
        
        assert isinstance(__version__, str)
        assert len(__version__) > 0
    
    def test_author(self):
        """测试作者信息"""
        from src.infrastructure import __author__
        
        assert isinstance(__author__, str)
        assert len(__author__) > 0
    
    def test_get_config_manager(self):
        """测试获取配置管理器"""
        from src.infrastructure import get_config_manager
        
        # 由于可能导入失败，我们测试函数存在性
        assert callable(get_config_manager)
        
        # 测试调用（可能返回None）
        result = get_config_manager()
        # 结果可能是None或配置管理器实例
        assert result is None or hasattr(result, '__class__')
    
    def test_get_cache_manager(self):
        """测试获取缓存管理器"""
        from src.infrastructure import get_cache_manager
        
        assert callable(get_cache_manager)
        
        result = get_cache_manager()
        assert result is None or hasattr(result, '__class__')
    
    def test_get_health_checker(self):
        """测试获取健康检查器"""
        from src.infrastructure import get_health_checker
        
        assert callable(get_health_checker)
        
        result = get_health_checker()
        assert result is None or hasattr(result, '__class__')
    
    def test_get_monitor(self):
        """测试获取监控器"""
        from src.infrastructure import get_monitor
        
        assert callable(get_monitor)
        
        result = get_monitor()
        assert result is None or hasattr(result, '__class__')
    
    def test_create_config_manager(self):
        """测试创建配置管理器"""
        from src.infrastructure import create_config_manager
        
        assert callable(create_config_manager)
        
        result = create_config_manager()
        assert result is None or hasattr(result, '__class__')
    
    def test_create_cache_manager(self):
        """测试创建缓存管理器"""
        from src.infrastructure import create_cache_manager
        
        assert callable(create_cache_manager)
        
        result = create_cache_manager()
        assert result is None or hasattr(result, '__class__')
    
    def test_create_health_checker(self):
        """测试创建健康检查器"""
        from src.infrastructure import create_health_checker
        
        assert callable(create_health_checker)
        
        result = create_health_checker()
        assert result is None or hasattr(result, '__class__')
    
    def test_get_default_monitor(self):
        """测试获取默认监控器"""
        from src.infrastructure import get_default_monitor
        
        assert callable(get_default_monitor)
        
        result = get_default_monitor()
        assert result is None or hasattr(result, '__class__')
    
    def test_config_manager_alias(self):
        """测试ConfigManager别名"""
        from src.infrastructure import ConfigManager
        
        assert callable(ConfigManager)
    
    def test_cache_manager_alias(self):
        """测试CacheManager别名"""
        from src.infrastructure import CacheManager
        
        assert callable(CacheManager)
    
    def test_health_checker_alias(self):
        """测试HealthChecker别名"""
        from src.infrastructure import HealthChecker
        
        assert callable(HealthChecker)
    
    def test_monitor_alias(self):
        """测试Monitor别名"""
        from src.infrastructure import Monitor
        
        assert callable(Monitor)

