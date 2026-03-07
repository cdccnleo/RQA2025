"""
集成配置模块

提供EnhancedDataIntegration的配置类。
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class IntegrationConfig:
    """集成配置"""
    
    # 并行加载配置
    parallel_loading: Dict[str, Any] = None
    # 缓存策略配置
    cache_strategy: Dict[str, Any] = None
    # 质量监控配置
    quality_monitor: Dict[str, Any] = None
    # 数据管理器配置
    data_manager: Dict[str, Any] = None
    # 性能优化配置
    performance_optimization: Dict[str, Any] = None

    def __post_init__(self):
        """初始化默认配置"""
        if self.parallel_loading is None:
            self.parallel_loading = {
                "max_workers": 12,  # 增加工作线程数
                "enable_auto_scaling": True,
                "batch_size": 20,  # 增加批处理大小
                "max_queue_size": 2000,  # 增加队列大小
                "enable_dynamic_threading": True,  # 启用动态线程管理
                "thread_pool_strategy": "adaptive",  # 自适应线程池策略
            }

        if self.cache_strategy is None:
            self.cache_strategy = {
                "approach": "adaptive",
                "max_size": 200 * 1024 * 1024,  # 增加到200MB
                "max_items": 20000,  # 增加缓存项数
                "enable_preload": True,
                "enable_adaptive_ttl": True,
                "enable_cache_warming": True,  # 启用缓存预热
                "preload_strategy": "predictive",  # 预测性预热策略
            }

        if self.quality_monitor is None:
            self.quality_monitor = {
                "enable_alerting": True,
                "enable_trend_analysis": True,
                "enable_advanced_metrics": True,  # 启用高级质量指标
                "enable_anomaly_detection": True,  # 启用异常检测
                "quality_threshold": 0.95,  # 质量阈值
            }

        if self.data_manager is None:
            self.data_manager = {
                "enable_enhanced_features": True,
                "cache_enabled": True,
                "quality_check_enabled": True,
                "enable_performance_monitoring": True,  # 启用性能监控
            }

        if self.performance_optimization is None:
            self.performance_optimization = {
                "enable_financial_optimization": True,  # 启用财务数据优化
                "enable_parallel_optimization": True,  # 启用并行优化
                "enable_memory_optimization": True,  # 启用内存优化
                "enable_connection_pooling": True,  # 启用连接池
                "max_connection_pool_size": 50,  # 连接池大小
                "connection_timeout": 30,  # 连接超时
                "enable_data_compression": True,  # 启用数据压缩
                "compression_level": 6,  # 压缩级别
            }

