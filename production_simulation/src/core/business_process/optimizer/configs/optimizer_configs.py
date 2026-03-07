"""
智能优化器配置类定义

应用参数对象模式，提供类型安全的配置管理
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from enum import Enum


class OptimizationStrategy(Enum):
    """优化策略枚举"""
    PERFORMANCE_FIRST = "performance_first"  # 性能优先
    QUALITY_FIRST = "quality_first"          # 质量优先
    BALANCED = "balanced"                     # 平衡模式
    CUSTOM = "custom"                         # 自定义


# DecisionStrategy 在 decision_engine 中定义，避免循环导入
# 这里用字符串默认值
# class DecisionStrategy(Enum): ...
# 改为使用Any类型，运行时从decision_engine导入


@dataclass
class AnalysisConfig:
    """性能分析配置"""
    analysis_interval: int = 60                    # 分析间隔（秒）
    metrics_retention_days: int = 30               # 指标保留天数
    enable_deep_analysis: bool = True              # 启用深度分析
    historical_data_window: int = 100              # 历史数据窗口
    enable_trend_prediction: bool = True           # 启用趋势预测
    prediction_horizon: int = 10                   # 预测时间范围

    def __post_init__(self):
        """配置后验证"""
        if self.analysis_interval <= 0:
            raise ValueError("analysis_interval必须大于0")
        if self.metrics_retention_days <= 0:
            raise ValueError("metrics_retention_days必须大于0")


@dataclass
class DecisionConfig:
    """智能决策配置"""
    strategy: Any = "balanced"  # DecisionStrategy枚举，默认balanced
    risk_threshold: float = 0.7                    # 风险阈值
    decision_timeout: int = 30                     # 决策超时（秒）
    enable_ml_enhancement: bool = True             # 启用ML增强
    confidence_threshold: float = 0.6              # 置信度阈值
    enable_multi_model: bool = True                # 启用多模型集成
    model_weights: Optional[Dict[str, float]] = None  # 模型权重

    def __post_init__(self):
        """配置后验证"""
        if not 0 <= self.risk_threshold <= 1:
            raise ValueError("risk_threshold必须在0-1之间")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold必须在0-1之间")
        if self.decision_timeout <= 0:
            raise ValueError("decision_timeout必须大于0")


@dataclass
class ExecutionConfig:
    """流程执行配置"""
    max_concurrent_processes: int = 10             # 最大并发流程数
    execution_timeout: int = 300                   # 执行超时（秒）
    enable_retry: bool = True                      # 启用重试
    max_retries: int = 3                           # 最大重试次数
    retry_delay: int = 5                           # 重试延迟（秒）
    parallel_execution: bool = False               # 并行执行
    enable_circuit_breaker: bool = True            # 启用断路器
    circuit_breaker_threshold: int = 5             # 断路器阈值

    def __post_init__(self):
        """配置后验证"""
        if self.max_concurrent_processes <= 0:
            raise ValueError("max_concurrent_processes必须大于0")
        if self.execution_timeout <= 0:
            raise ValueError("execution_timeout必须大于0")
        if self.max_retries < 0:
            raise ValueError("max_retries不能为负数")


@dataclass
class RecommendationConfig:
    """建议生成配置"""
    min_confidence: float = 0.6                    # 最小置信度
    max_recommendations: int = 10                  # 最大建议数
    enable_ai_insights: bool = True                # 启用AI洞察
    priority_threshold: float = 0.7                # 优先级阈值
    enable_auto_implementation: bool = False       # 启用自动实施
    impact_calculation_method: str = "weighted"    # 影响计算方法

    def __post_init__(self):
        """配置后验证"""
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("min_confidence必须在0-1之间")
        if self.max_recommendations <= 0:
            raise ValueError("max_recommendations必须大于0")


@dataclass
class MonitoringConfig:
    """流程监控配置"""
    monitoring_interval: int = 30                  # 监控间隔（秒）
    alert_threshold: Dict[str, float] = field(default_factory=lambda: {
        'execution_time': 300.0,                   # 执行时间告警阈值
        'error_rate': 0.1,                         # 错误率告警阈值
        'resource_usage': 0.8                      # 资源使用告警阈值
    })
    enable_auto_alert: bool = True                 # 启用自动告警
    metrics_retention: int = 1000                  # 指标保留数量
    enable_performance_tracking: bool = True       # 启用性能追踪
    enable_anomaly_detection: bool = True          # 启用异常检测

    def __post_init__(self):
        """配置后验证"""
        if self.monitoring_interval <= 0:
            raise ValueError("monitoring_interval必须大于0")
        if self.metrics_retention <= 0:
            raise ValueError("metrics_retention必须大于0")


@dataclass
class OptimizerConfig:
    """
    优化器主配置类

    整合所有子配置，提供统一的配置接口
    采用参数对象模式，提高可维护性
    """
    # 子配置对象
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    recommendation: RecommendationConfig = field(default_factory=RecommendationConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    # 全局配置
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    enable_ai_enhancement: bool = True             # 启用AI增强
    max_concurrent_processes: int = 10             # 最大并发数
    global_timeout: int = 600                      # 全局超时（秒）
    enable_logging: bool = True                    # 启用日志
    log_level: str = "INFO"                        # 日志级别

    # 额外配置
    custom_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """配置后验证"""
        if self.max_concurrent_processes <= 0:
            raise ValueError("max_concurrent_processes必须大于0")
        if self.global_timeout <= 0:
            raise ValueError("global_timeout必须大于0")

        # 验证日志级别
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.log_level.upper() not in valid_levels:
            raise ValueError(f"log_level必须是{valid_levels}之一")

    @classmethod
    def create_default(cls) -> 'OptimizerConfig':
        """创建默认配置"""
        return cls()

    @classmethod
    def create_high_performance(cls) -> 'OptimizerConfig':
        """创建高性能配置"""
        config = cls()
        config.optimization_strategy = OptimizationStrategy.PERFORMANCE_FIRST
        config.max_concurrent_processes = 20
        config.execution.parallel_execution = True
        config.execution.max_concurrent_processes = 20
        config.monitoring.monitoring_interval = 15  # 更频繁监控
        return config

    @classmethod
    def create_conservative(cls) -> 'OptimizerConfig':
        """创建保守配置"""
        config = cls()
        config.optimization_strategy = OptimizationStrategy.QUALITY_FIRST
        config.decision.strategy = "conservative"  # 使用字符串避免循环导入
        config.decision.risk_threshold = 0.9
        config.decision.confidence_threshold = 0.8
        config.execution.max_retries = 5
        return config

    @classmethod
    def create_ai_optimized(cls) -> 'OptimizerConfig':
        """创建AI优化配置"""
        config = cls()
        config.optimization_strategy = OptimizationStrategy.CUSTOM
        config.decision.strategy = "ai_optimized"  # 使用字符串避免循环导入
        config.decision.enable_ml_enhancement = True
        config.decision.enable_multi_model = True
        config.analysis.enable_deep_analysis = True
        config.analysis.enable_trend_prediction = True
        config.recommendation.enable_ai_insights = True
        return config

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于序列化）"""
        return {
            'analysis': self.analysis.__dict__,
            'decision': {
                'strategy': self.decision.strategy.value,
                **{k: v for k, v in self.decision.__dict__.items() if k != 'strategy'}
            },
            'execution': self.execution.__dict__,
            'recommendation': self.recommendation.__dict__,
            'monitoring': self.monitoring.__dict__,
            'optimization_strategy': self.optimization_strategy.value,
            'enable_ai_enhancement': self.enable_ai_enhancement,
            'max_concurrent_processes': self.max_concurrent_processes,
            'global_timeout': self.global_timeout,
            'enable_logging': self.enable_logging,
            'log_level': self.log_level,
            'custom_config': self.custom_config
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'OptimizerConfig':
        """从字典创建配置（用于反序列化）"""
        # 提取子配置
        analysis = AnalysisConfig(**config_dict.get('analysis', {}))

        decision_dict = config_dict.get('decision', {})
        if 'strategy' in decision_dict and isinstance(decision_dict['strategy'], str):
            decision_dict['strategy'] = DecisionStrategy(decision_dict['strategy'])
        decision = DecisionConfig(**decision_dict)

        execution = ExecutionConfig(**config_dict.get('execution', {}))
        recommendation = RecommendationConfig(**config_dict.get('recommendation', {}))
        monitoring = MonitoringConfig(**config_dict.get('monitoring', {}))

        # 处理优化策略
        strategy = config_dict.get('optimization_strategy', 'balanced')
        if isinstance(strategy, str):
            strategy = OptimizationStrategy(strategy)

        # 创建主配置
        return cls(
            analysis=analysis,
            decision=decision,
            execution=execution,
            recommendation=recommendation,
            monitoring=monitoring,
            optimization_strategy=strategy,
            enable_ai_enhancement=config_dict.get('enable_ai_enhancement', True),
            max_concurrent_processes=config_dict.get('max_concurrent_processes', 10),
            global_timeout=config_dict.get('global_timeout', 600),
            enable_logging=config_dict.get('enable_logging', True),
            log_level=config_dict.get('log_level', 'INFO'),
            custom_config=config_dict.get('custom_config')
        )


# 便捷函数
def create_optimizer_config(**kwargs) -> OptimizerConfig:
    """
    创建优化器配置（便捷函数）

    Args:
        **kwargs: 配置参数

    Returns:
        OptimizerConfig: 优化器配置对象

    Examples:
        >>> # 创建默认配置
        >>> config = create_optimizer_config()

        >>> # 创建自定义配置
        >>> config = create_optimizer_config(
        ...     max_concurrent_processes=20,
        ...     enable_ai_enhancement=True
        ... )

        >>> # 创建高性能配置
        >>> config = OptimizerConfig.create_high_performance()
    """
    return OptimizerConfig(**kwargs)
