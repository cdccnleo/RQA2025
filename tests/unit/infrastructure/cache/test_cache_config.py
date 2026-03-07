#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缓存测试专用配置工具

提供统一的测试配置，确保测试性能最优
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
from src.infrastructure.cache.core.cache_configs import CacheConfig


class TestCacheConfigFactory:
    """测试缓存配置工厂"""
    
    @staticmethod
    def get_fast_test_config(**overrides):
        """获取快速测试配置"""
        config_dict = {
            'basic': {'max_size': 100, 'ttl': 300},
            'multi_level': {'level': 'memory', 'memory_max_size': 50, 'memory_ttl': 60},
            'advanced': {
                'enable_compression': False, 
                'enable_preloading': False, 
                'cleanup_interval': 1  # 1秒清理间隔，快速响应
            },
            'smart': {'enable_monitoring': False, 'enable_auto_optimization': False}
        }
        
        # 应用覆盖参数
        for key, value in overrides.items():
            if key in config_dict:
                config_dict[key].update(value)
            else:
                config_dict[key] = value
                
        return CacheConfig.from_dict(config_dict)
    
    @staticmethod
    def get_performance_test_config():
        """获取性能测试配置"""
        return TestCacheConfigFactory.get_fast_test_config(
            advanced={'cleanup_interval': 1, 'enable_compression': False}
        )
