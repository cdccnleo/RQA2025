"""
performance_config 模块测试

测试性能配置管理功能。
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from unittest.mock import Mock

from src.infrastructure.cache.utils.performance_config import (
    PerformanceConfigManager,
    PerformanceThresholds,
    CacheConfiguration,
    OptimizationConfiguration,
    MonitoringConfiguration,
    OptimizationLevel,
    CachePolicy,
    create_performance_config
)


class TestPerformanceThresholds:
    """测试性能阈值"""

    def test_init_default(self):
        """测试默认初始化"""
        thresholds = PerformanceThresholds()
        assert thresholds.response_time_warning == 1.0
        assert thresholds.response_time_critical == 5.0
        assert thresholds.success_rate_warning == 0.95
        assert thresholds.success_rate_critical == 0.90


class TestCacheConfiguration:
    """测试缓存配置"""

    def test_init_default(self):
        """测试默认初始化"""
        config = CacheConfiguration()
        assert config.enabled is True
        assert config.strategy == CachePolicy.HYBRID
        assert config.max_size == 1000

    def test_strategy_string_conversion(self):
        """测试策略字符串转换"""
        config = CacheConfiguration(strategy="lru_only")
        assert config.strategy == CachePolicy.LRU_ONLY

    def test_invalid_strategy_fallback(self):
        """测试无效策略回退"""
        config = CacheConfiguration(strategy="invalid")
        assert config.strategy == CachePolicy.HYBRID


class TestOptimizationConfiguration:
    """测试优化配置"""

    def test_init_default(self):
        """测试默认初始化"""
        config = OptimizationConfiguration()
        assert config.level == OptimizationLevel.BASIC
        assert config.enable_batch_processing is True
        assert config.batch_size == 10

    def test_level_string_conversion(self):
        """测试级别字符串转换"""
        config = OptimizationConfiguration(level="advanced")
        assert config.level == OptimizationLevel.ADVANCED

    def test_invalid_level_fallback(self):
        """测试无效级别回退"""
        config = OptimizationConfiguration(level="invalid")
        assert config.level == OptimizationLevel.BASIC


class TestMonitoringConfiguration:
    """测试监控配置"""

    def test_init_default(self):
        """测试默认初始化"""
        config = MonitoringConfiguration()
        assert config.enable_performance_monitoring is True
        assert config.performance_sampling_interval == 5
        assert config.performance_history_size == 100


class TestPerformanceConfigManager:
    """测试性能配置管理器"""

    def test_init_default_config(self):
        """测试默认配置初始化"""
        manager = PerformanceConfigManager()
        assert manager.config == {}
        assert isinstance(manager.thresholds, PerformanceThresholds)
        assert isinstance(manager.cache, CacheConfiguration)

    def test_init_custom_config(self):
        """测试自定义配置初始化"""
        custom_config = {
            'thresholds': {'response_time_warning': 2.0},
            'cache': {'max_size': 500}
        }
        manager = PerformanceConfigManager(custom_config)
        assert manager.thresholds.response_time_warning == 2.0
        assert manager.cache.max_size == 500

    def test_get_cache_config(self):
        """测试获取缓存配置"""
        manager = PerformanceConfigManager()
        config = manager.get_cache_config()

        assert isinstance(config, dict)
        assert 'max_size' in config
        assert 'ttl_seconds' in config
        assert 'policy' in config

    def test_get_optimization_config(self):
        """测试获取优化配置"""
        manager = PerformanceConfigManager()
        config = manager.get_optimization_config()

        assert isinstance(config, dict)
        assert 'level' in config
        assert 'batch_processing' in config
        assert 'batch_size' in config

    def test_get_monitoring_config(self):
        """测试获取监控配置"""
        manager = PerformanceConfigManager()
        config = manager.get_monitoring_config()

        assert isinstance(config, dict)
        assert 'performance_monitoring' in config
        assert 'sampling_interval' in config
        assert 'alerts' in config

    def test_is_optimization_enabled(self):
        """测试优化功能启用检查"""
        manager = PerformanceConfigManager()

        # 测试不同级别
        assert manager.is_optimization_enabled('caching') is True
        assert manager.is_optimization_enabled('batch_processing') is True

        # 测试高级优化
        manager.optimization.level = OptimizationLevel.ADVANCED
        assert manager.is_optimization_enabled('parallel_processing') is True

        # 测试无优化
        manager.optimization.level = OptimizationLevel.NONE
        assert manager.is_optimization_enabled('caching') is False

    def test_get_ttl_for_type(self):
        """测试根据类型获取TTL"""
        manager = PerformanceConfigManager()

        assert manager.get_ttl_for_type('health_check') == 30
        assert manager.get_ttl_for_type('performance') == 60
        assert manager.get_ttl_for_type('unknown') == 30  # 默认值

    def test_validate_config_valid(self):
        """测试有效配置验证"""
        manager = PerformanceConfigManager()
        assert manager.validate_config() is True

    def test_validate_config_invalid(self):
        """测试无效配置验证"""
        manager = PerformanceConfigManager()
        manager.cache.max_size = 0  # 无效配置
        assert manager.validate_config() is False

    def test_get_summary(self):
        """测试获取配置摘要"""
        manager = PerformanceConfigManager()
        summary = manager.get_summary()

        assert isinstance(summary, dict)
        assert 'optimization_level' in summary
        assert 'cache_strategy' in summary
        assert 'config_valid' in summary


class TestCreatePerformanceConfig:
    """测试创建性能配置"""

    def test_create_development_config(self):
        """测试创建开发环境配置"""
        manager = create_performance_config('development')

        assert isinstance(manager, PerformanceConfigManager)
        assert manager.thresholds.response_time_warning == 2.0
        assert manager.cache.max_size == 500

    def test_create_production_config(self):
        """测试创建生产环境配置"""
        manager = create_performance_config('production')

        assert isinstance(manager, PerformanceConfigManager)
        assert manager.thresholds.response_time_warning == 0.5
        assert manager.cache.max_size == 2000
        assert manager.optimization.level == OptimizationLevel.ADVANCED

    def test_create_staging_config(self):
        """测试创建暂存环境配置"""
        manager = create_performance_config('staging')  # 不存在的环境，使用默认

        assert isinstance(manager, PerformanceConfigManager)
        assert manager.thresholds.response_time_warning == 2.0  # 默认值

    def test_environment_specific_features(self):
        """测试环境特定功能"""
        manager = create_performance_config('high_performance')

        assert manager.optimization.level == OptimizationLevel.AGGRESSIVE
        assert manager.optimization.enable_parallel_processing is True
        assert manager.optimization.enable_memory_pooling is True
        assert manager.cache.enable_compression is True