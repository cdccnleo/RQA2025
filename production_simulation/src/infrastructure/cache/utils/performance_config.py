"""
performance_config 模块

提供 performance_config 相关功能和接口。
"""


from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional
"""
基础设施层 - 配置管理组件

performance_config 模块

配置管理相关的文件
提供配置管理相关的功能实现。
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
健康检查性能优化配置

定义各种缓存策略、性能阈值和优化参数。
"""


class OptimizationLevel(Enum):

    """优化级别枚举"""
    NONE = "none"           # 无优化
    BASIC = "basic"         # 基础优化
    ADVANCED = "advanced"   # 高级优化
    AGGRESSIVE = "aggressive"  # 激进优化


class CachePolicy(Enum):

    """缓存策略配置枚举"""
    DISABLED = "disabled"   # 禁用缓存
    TTL_ONLY = "ttl_only"   # 仅TTL策略
    LRU_ONLY = "lru_only"   # 仅LRU策略
    HYBRID = "hybrid"       # 混合策略
    ADAPTIVE = "adaptive"   # 自适应策略


@dataclass
class PerformanceThresholds:

    """性能阈值配置"""
    # 响应时间阈值（秒）
    response_time_warning: float = 1.0      # 警告阈值
    response_time_critical: float = 5.0     # 严重阈值

    # 成功率阈值
    success_rate_warning: float = 0.95      # 警告阈值
    success_rate_critical: float = 0.90     # 严重阈值

    # 性能评分阈值
    performance_score_warning: float = 0.8   # 警告阈值
    performance_score_critical: float = 0.6  # 严重阈值

    # 资源使用阈值
    cpu_usage_warning: float = 0.7          # CPU使用率警告
    memory_usage_warning: float = 0.8       # 内存使用率警告
    disk_usage_warning: float = 0.85        # 磁盘使用率警告


@dataclass
class CacheConfiguration:

    """缓存配置"""
    # 基本配置
    enabled: bool = True
    strategy: CachePolicy = CachePolicy.HYBRID

    def __post_init__(self):
        """后初始化处理"""
        # 处理strategy字段的类型转换
        if isinstance(self.strategy, str):
            # 如果是字符串，转换为对应的枚举值
            try:
                self.strategy = CachePolicy(self.strategy)
            except ValueError:
                # 如果转换失败，使用默认值
                self.strategy = CachePolicy.HYBRID

    # 容量配置
    max_size: int = 1000                    # 最大缓存条目数
    max_memory_mb: float = 100.0            # 最大内存使用（MB）

    # TTL配置

    default_ttl_seconds: int = 30           # 默认TTL
    health_check_ttl: int = 30              # 健康检查结果TTL
    performance_ttl: int = 60                # 性能指标TTL
    trend_ttl: int = 300                    # 趋势分析TTL

    # 清理配置
    cleanup_interval_seconds: int = 60      # 清理间隔
    eviction_policy: str = "lru"            # 淘汰策略

    # 高级配置
    enable_compression: bool = False        # 是否启用压缩
    compression_threshold_bytes: int = 1024  # 压缩阈值
    enable_persistence: bool = False        # 是否启用持久化
    persistence_interval_seconds: int = 300  # 持久化间隔


@dataclass
class OptimizationConfiguration:

    """优化配置"""
    # 优化级别
    level: OptimizationLevel = OptimizationLevel.BASIC

    def __post_init__(self):
        """后初始化处理"""
        # 处理level字段的类型转换
        if isinstance(self.level, str):
            # 如果是字符串，转换为对应的枚举值
            try:
                self.level = OptimizationLevel(self.level)
            except ValueError:
                # 如果转换失败，使用默认值
                self.level = OptimizationLevel.BASIC

    # 计算优化
    enable_batch_processing: bool = True    # 启用批处理
    batch_size: int = 10                    # 批处理大小
    enable_parallel_processing: bool = False  # 启用并行处理
    max_workers: int = 4                    # 最大工作线程数

    # 内存优化
    enable_memory_pooling: bool = False     # 启用内存池
    memory_pool_size_mb: float = 50.0       # 内存池大小
    enable_object_reuse: bool = False       # 启用对象重用

    # 算法优化
    enable_approximation: bool = False      # 启用近似计算
    approximation_tolerance: float = 0.01   # 近似容差
    enable_caching_optimization: bool = True  # 启用缓存优化


@dataclass
class MonitoringConfiguration:

    """监控配置"""
    # 性能监控
    enable_performance_monitoring: bool = True
    performance_sampling_interval: int = 5   # 采样间隔（秒）
    performance_history_size: int = 100      # 历史记录大小

    # 资源监控
    enable_resource_monitoring: bool = True
    resource_check_interval: int = 10       # 资源检查间隔
    enable_detailed_metrics: bool = False   # 启用详细指标

    # 告警配置
    enable_alerts: bool = True
    alert_cooldown_seconds: int = 300       # 告警冷却时间
    max_alerts_per_hour: int = 10           # 每小时最大告警数


class PerformanceConfigManager:

    """性能配置管理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化性能配置管理器

        Args:
            config: 配置字典
        """
        self.config = config or {}

        # 初始化各配置组件
        self.thresholds = PerformanceThresholds(**self.config.get('thresholds', {}))
        self.cache = CacheConfiguration(**self.config.get('cache', {}))
        self.optimization = OptimizationConfiguration(**self.config.get('optimization', {}))
        self.monitoring = MonitoringConfiguration(**self.config.get('monitoring', {}))

    def get_cache_config(self) -> Dict[str, Any]:
        """
        获取缓存配置

        Returns:
            缓存配置字典
        """
        return {
            'max_size': self.cache.max_size,
            'ttl_seconds': self.cache.default_ttl_seconds,
            'policy': self.cache.strategy.value,
            'max_memory_mb': self.cache.max_memory_mb,
            'cleanup_interval': self.cache.cleanup_interval_seconds,
            'enable_compression': self.cache.enable_compression
        }

    def get_optimization_config(self) -> Dict[str, Any]:
        """
        获取优化配置

        Returns:
            优化配置字典
        """
        return {
            'level': self.optimization.level.value,
            'batch_processing': self.optimization.enable_batch_processing,
            'batch_size': self.optimization.batch_size,
            'parallel_processing': self.optimization.enable_parallel_processing,
            'max_workers': self.optimization.max_workers,
            'memory_pooling': self.optimization.enable_memory_pooling,
            'object_reuse': self.optimization.enable_object_reuse
        }

    def get_monitoring_config(self) -> Dict[str, Any]:
        """
        获取监控配置

        Returns:
            监控配置字典
        """
        return {
            'performance_monitoring': self.monitoring.enable_performance_monitoring,
            'sampling_interval': self.monitoring.performance_sampling_interval,
            'history_size': self.monitoring.performance_history_size,
            'resource_monitoring': self.monitoring.enable_resource_monitoring,
            'resource_interval': self.monitoring.resource_check_interval,
            'alerts': self.monitoring.enable_alerts
        }

    def is_optimization_enabled(self, feature: str) -> bool:
        """
        检查特定优化功能是否启用

        Args:
            feature: 功能名称

        Returns:
            是否启用
        """
        if self.optimization.level == OptimizationLevel.NONE:
            return False

        if self.optimization.level == OptimizationLevel.BASIC:
            return feature in ['caching', 'batch_processing']

        if self.optimization.level == OptimizationLevel.ADVANCED:
            return feature in ['caching', 'batch_processing', 'parallel_processing', 'memory_pooling']

        if self.optimization.level == OptimizationLevel.AGGRESSIVE:
            return True

        return False

    def get_ttl_for_type(self, data_type: str) -> int:
        """
        根据数据类型获取TTL

        Args:
            data_type: 数据类型

        Returns:
            TTL（秒）
        """
        ttl_map = {
            'health_check': self.cache.health_check_ttl,
            'performance': self.cache.performance_ttl,
            'trend': self.cache.trend_ttl,
            'default': self.cache.default_ttl_seconds
        }
        return ttl_map.get(data_type, self.cache.default_ttl_seconds)

    def validate_config(self) -> bool:
        """
        验证配置有效性

        Returns:
            配置是否有效
        """
        try:
            # 验证阈值配置
            assert 0 < self.thresholds.response_time_warning < self.thresholds.response_time_critical
            assert 0 < self.thresholds.success_rate_critical < self.thresholds.success_rate_warning < 1
            assert 0 < self.thresholds.performance_score_critical < self.thresholds.performance_score_warning < 1

            # 验证缓存配置
            assert self.cache.max_size > 0
            assert self.cache.max_memory_mb > 0
            assert self.cache.default_ttl_seconds > 0

            # 验证优化配置
            assert self.optimization.batch_size > 0
            assert self.optimization.max_workers > 0

            # 验证监控配置
            assert self.monitoring.performance_sampling_interval > 0
            assert self.monitoring.performance_history_size > 0

            return True

        except AssertionError:
            return False

    def get_summary(self) -> Dict[str, Any]:
        """
        获取配置摘要

        Returns:
            配置摘要字典
        """
        return {
            'optimization_level': self.optimization.level.value,
            'cache_strategy': self.cache.strategy.value,
            'cache_enabled': self.cache.enabled,
            'performance_monitoring': self.monitoring.enable_performance_monitoring,
            'alerts_enabled': self.monitoring.enable_alerts,
            'config_valid': self.validate_config()
        }


# 预定义配置模板
PERFORMANCE_CONFIG_TEMPLATES = {
    'development': {
        'thresholds': {
            'response_time_warning': 2.0,
            'response_time_critical': 10.0,
            'success_rate_warning': 0.9,
            'success_rate_critical': 0.8
        },
        'cache': {
            'max_size': 500,
            'max_memory_mb': 50.0,
            'default_ttl_seconds': 60
        },
        'optimization': {
            'level': 'basic'
        },
        'monitoring': {
            'performance_sampling_interval': 10,
            'performance_history_size': 50
        }
    },
    'production': {
        'thresholds': {
            'response_time_warning': 0.5,
            'response_time_critical': 2.0,
            'success_rate_warning': 0.99,
            'success_rate_critical': 0.95
        },
        'cache': {
            'max_size': 2000,
            'max_memory_mb': 200.0,
            'default_ttl_seconds': 15,
            'strategy': 'hybrid'
        },
        'optimization': {
            'level': 'advanced',
            'enable_parallel_processing': True,
            'max_workers': 8
        },
        'monitoring': {
            'performance_sampling_interval': 5,
            'performance_history_size': 200,
            'enable_detailed_metrics': True
        }
    },
    'high_performance': {
        'thresholds': {
            'response_time_warning': 0.1,
            'response_time_critical': 0.5,
            'success_rate_warning': 0.999,
            'success_rate_critical': 0.99
        },
        'cache': {
            'max_size': 5000,
            'max_memory_mb': 500.0,
            'default_ttl_seconds': 5,
            'strategy': 'adaptive',
            'enable_compression': True
        },
        'optimization': {
            'level': 'aggressive',
            'enable_parallel_processing': True,
            'max_workers': 16,
            'enable_memory_pooling': True,
            'enable_object_reuse': True,
            'enable_approximation': True
        },
        'monitoring': {
            'performance_sampling_interval': 1,
            'performance_history_size': 500,
            'enable_detailed_metrics': True
        }
    }
}


def create_performance_config(environment: str = 'development',

                              custom_config: Optional[Dict[str, Any]] = None):
    """
    创建性能配置管理器

    Args:
        environment: 环境名称
        custom_config: 自定义配置

    Returns:
        性能配置管理器实例
    """
    # 获取环境模板
    template = PERFORMANCE_CONFIG_TEMPLATES.get(
        environment, PERFORMANCE_CONFIG_TEMPLATES['development'])

    # 合并自定义配置
    if custom_config:
        # 深度合并配置
        merged_config = _deep_merge(template, custom_config)
    else:
        merged_config = template

    return PerformanceConfigManager(merged_config)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    深度合并配置字典

    Args:
        base: 基础配置
        override: 覆盖配置

    Returns:
        合并后的配置
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result
