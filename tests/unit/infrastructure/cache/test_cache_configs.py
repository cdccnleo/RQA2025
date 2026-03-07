#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缓存配置单元测试

测试CacheConfig及其子配置类的功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from dataclasses import asdict
from src.infrastructure.cache.core.cache_configs import (
    CacheConfig, BasicCacheConfig, MultiLevelCacheConfig,
    AdvancedCacheConfig, SmartCacheConfig, DistributedCacheConfig,
    CacheLevel, DataType
)


class TestBasicCacheConfig:
    """测试基础缓存配置"""

    def test_default_initialization(self):
        """测试默认初始化"""
        config = BasicCacheConfig()
        assert config.max_size == 1000
        assert config.ttl == 3600
        assert config.strategy.value == "lru"

    def test_custom_initialization(self):
        """测试自定义初始化"""
        config = BasicCacheConfig(max_size=500, ttl=1800)
        assert config.max_size == 500
        assert config.ttl == 1800

    def test_validation_max_size(self):
        """测试最大容量验证"""
        with pytest.raises(ValueError):
            BasicCacheConfig(max_size=0)

        with pytest.raises(ValueError):
            BasicCacheConfig(max_size=-1)

    def test_validation_ttl(self):
        """测试TTL验证"""
        with pytest.raises(ValueError):
            BasicCacheConfig(ttl=0)

        with pytest.raises(ValueError):
            BasicCacheConfig(ttl=-1)


class TestMultiLevelCacheConfig:
    """测试多级缓存配置"""

    def test_default_initialization(self):
        """测试默认初始化"""
        config = MultiLevelCacheConfig()
        assert config.level == CacheLevel.HYBRID
        assert config.memory_max_size == 1000
        assert config.memory_ttl == 30
        assert config.redis_max_size == 10000
        assert config.redis_ttl == 300
        assert config.file_max_size == 100000
        assert config.file_ttl == 3600

    def test_memory_level_config(self):
        """测试内存级别配置"""
        config = MultiLevelCacheConfig(level=CacheLevel.MEMORY)
        assert config.level == CacheLevel.MEMORY

    def test_redis_level_config(self):
        """测试Redis级别配置"""
        config = MultiLevelCacheConfig(level=CacheLevel.REDIS)
        assert config.level == CacheLevel.REDIS

    def test_file_level_config(self):
        """测试文件级别配置"""
        config = MultiLevelCacheConfig(level=CacheLevel.FILE)
        assert config.level == CacheLevel.FILE

    def test_validation_memory_max_size(self):
        """测试内存最大容量验证"""
        with pytest.raises(ValueError):
            MultiLevelCacheConfig(memory_max_size=0)

    def test_validation_memory_ttl(self):
        """测试内存TTL验证"""
        with pytest.raises(ValueError):
            MultiLevelCacheConfig(memory_ttl=0)


class TestAdvancedCacheConfig:
    """测试高级缓存配置"""

    def test_default_initialization(self):
        """测试默认初始化"""
        config = AdvancedCacheConfig()
        assert config.enable_compression == True
        assert config.enable_preloading == True
        assert config.enable_parallel_write == True
        assert config.preload_threshold == 0.8
        assert config.cleanup_interval == 60
        assert config.max_memory_mb == 100

    def test_validation_preload_threshold(self):
        """测试预加载阈值验证"""
        with pytest.raises(ValueError):
            AdvancedCacheConfig(preload_threshold=-0.1)

        with pytest.raises(ValueError):
            AdvancedCacheConfig(preload_threshold=1.5)


class TestSmartCacheConfig:
    """测试智能缓存配置"""

    def test_default_initialization(self):
        """测试默认初始化"""
        config = SmartCacheConfig()
        assert config.enable_monitoring == True
        assert config.enable_auto_optimization == True
        assert config.adaptation_interval == 300

    def test_validation_adaptation_interval(self):
        """测试适应间隔验证"""
        with pytest.raises(ValueError):
            SmartCacheConfig(adaptation_interval=0)


class TestDistributedCacheConfig:
    """测试分布式缓存配置"""

    def test_default_initialization(self):
        """测试默认初始化"""
        config = DistributedCacheConfig()
        assert config.distributed == False
        assert config.redis_host == "localhost"
        assert config.redis_port == 6379
        assert config.cluster_mode == False

    def test_validation_redis_port(self):
        """测试Redis端口验证"""
        with pytest.raises(ValueError):
            DistributedCacheConfig(redis_port=0)

        with pytest.raises(ValueError):
            DistributedCacheConfig(redis_port=65536)


class TestCacheConfig:
    """测试统一缓存配置"""

    def test_default_initialization(self):
        """测试默认初始化"""
        config = CacheConfig()
        assert isinstance(config.basic, BasicCacheConfig)
        assert isinstance(config.multi_level, MultiLevelCacheConfig)
        assert isinstance(config.advanced, AdvancedCacheConfig)
        assert isinstance(config.smart, SmartCacheConfig)
        assert isinstance(config.distributed, DistributedCacheConfig)

    def test_from_dict_basic(self):
        """测试从字典创建基础配置"""
        config_dict = {
            'basic': {'max_size': 500, 'ttl': 1800}
        }
        config = CacheConfig.from_dict(config_dict)
        assert config.basic.max_size == 500
        assert config.basic.ttl == 1800

    def test_from_dict_multi_level(self):
        """测试从字典创建多级配置"""
        config_dict = {
            'multi_level': {
                'level': 'memory',
                'memory_max_size': 2000,
                'memory_ttl': 60
            }
        }
        config = CacheConfig.from_dict(config_dict)
        assert config.multi_level.level == 'memory'  # from_dict返回字符串
        assert config.multi_level.memory_max_size == 2000
        assert config.multi_level.memory_ttl == 60

    def test_from_dict_advanced(self):
        """测试从字典创建高级配置"""
        config_dict = {
            'advanced': {
                'enable_compression': False,
                'enable_preloading': False,
                'max_memory_mb': 200
            }
        }
        config = CacheConfig.from_dict(config_dict)
        assert config.advanced.enable_compression == False
        assert config.advanced.enable_preloading == False
        assert config.advanced.max_memory_mb == 200

    def test_from_dict_smart(self):
        """测试从字典创建智能配置"""
        config_dict = {
            'smart': {
                'enable_monitoring': False,
                'adaptation_interval': 600
            }
        }
        config = CacheConfig.from_dict(config_dict)
        assert config.smart.enable_monitoring == False
        assert config.smart.adaptation_interval == 600

    def test_from_dict_distributed(self):
        """测试从字典创建分布式配置"""
        config_dict = {
            'distributed': {
                'distributed': True,
                'redis_host': 'redis.example.com',
                'redis_port': 6380
            }
        }
        config = CacheConfig.from_dict(config_dict)
        assert config.distributed.distributed == True
        assert config.distributed.redis_host == 'redis.example.com'
        assert config.distributed.redis_port == 6380

    def test_to_dict_conversion(self):
        """测试转换为字典"""
        config = CacheConfig()
        config_dict = config.to_dict()

        assert 'basic' in config_dict
        assert 'multi_level' in config_dict
        assert 'advanced' in config_dict
        assert 'smart' in config_dict
        assert 'distributed' in config_dict

        # 验证基础配置
        assert config_dict['basic']['max_size'] == 1000
        assert config_dict['basic']['ttl'] == 3600

    def test_create_simple_memory_config(self):
        """测试创建简单内存配置"""
        config = CacheConfig.create_simple_memory_config()

        assert config.multi_level.level == CacheLevel.MEMORY
        assert config.advanced.enable_compression == False
        assert config.advanced.enable_preloading == False
        assert config.smart.enable_monitoring == False
        assert config.distributed.distributed == False

    def test_create_production_config(self):
        """测试创建生产环境配置"""
        config = CacheConfig.create_production_config()

        assert config.multi_level.level == CacheLevel.HYBRID
        assert config.basic.max_size == 10000
        assert config.basic.ttl == 7200
        assert config.advanced.enable_compression == True
        assert config.advanced.enable_preloading == True
        assert config.smart.enable_monitoring == True
        assert config.distributed.distributed == True

    def test_dependency_validation_distributed_without_redis(self):
        """测试分布式模式但未配置Redis的验证"""
        config = CacheConfig()
        config.distributed.distributed = True
        config.distributed.redis_host = ""

        with pytest.raises(ValueError, match="启用分布式模式时必须配置redis_host"):
            config._validate_dependencies()

    def test_dependency_validation_preloading_threshold(self):
        """测试预加载阈值验证"""
        config = CacheConfig()
        config.advanced.enable_preloading = True
        config.advanced.preload_threshold = 0

        with pytest.raises(ValueError, match="启用预加载时preload_threshold必须大于0"):
            config._validate_dependencies()

    def test_dependency_validation_compression_memory(self):
        """测试压缩内存限制验证"""
        config = CacheConfig()
        config.advanced.enable_compression = True
        config.advanced.max_memory_mb = 10

        # 这个应该只是警告，不抛出异常
        config._validate_dependencies()

    def test_optimization_defaults_memory_level(self):
        """测试内存级别默认值优化"""
        config = CacheConfig()
        config.multi_level.level = CacheLevel.MEMORY
        config.multi_level.memory_ttl = 400  # 大于300

        config._optimize_defaults()

        # 应该被调整为300
        assert config.multi_level.memory_ttl == 300

    def test_optimization_defaults_redis_level(self):
        """测试Redis级别默认值优化"""
        config = CacheConfig()
        config.multi_level.level = CacheLevel.REDIS
        config.multi_level.redis_ttl = 500  # 小于600

        config._optimize_defaults()

        # 应该被调整为600
        assert config.multi_level.redis_ttl == 600


class TestCacheLevel:
    """测试缓存级别枚举"""

    def test_cache_level_values(self):
        """测试缓存级别枚举值"""
        assert CacheLevel.L1.value == "L1"
        assert CacheLevel.L2.value == "L2"
        assert CacheLevel.L3.value == "L3"
        assert CacheLevel.MEMORY.value == "memory"
        assert CacheLevel.REDIS.value == "redis"
        assert CacheLevel.FILE.value == "file"
        assert CacheLevel.HYBRID.value == "hybrid"

    def test_cache_level_members(self):
        """测试缓存级别枚举成员"""
        assert CacheLevel.MEMORY in CacheLevel
        assert CacheLevel.REDIS in CacheLevel
        assert CacheLevel.FILE in CacheLevel
        assert CacheLevel.HYBRID in CacheLevel


class TestDataType:
    """测试数据类型枚举"""

    def test_data_type_values(self):
        """测试数据类型枚举值"""
        assert DataType.SMALL.value == "small"
        assert DataType.MEDIUM.value == "medium"
        assert DataType.LARGE.value == "large"
        assert DataType.CRITICAL.value == "critical"
        assert DataType.TEMPORARY.value == "temporary"

    def test_data_type_members(self):
        """测试数据类型枚举成员"""
        assert DataType.SMALL in DataType
        assert DataType.MEDIUM in DataType
        assert DataType.LARGE in DataType
        assert DataType.CRITICAL in DataType
        assert DataType.TEMPORARY in DataType


if __name__ == '__main__':
    pytest.main([__file__])
