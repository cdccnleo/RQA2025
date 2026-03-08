"""
管道配置管理模块

提供管道配置的加载、验证和管理功能，支持YAML/JSON格式配置和各阶段参数配置。

主要功能:
    - 管道配置的加载和保存（YAML/JSON）
    - 各阶段参数配置管理
    - 配置验证和默认值处理
    - 与ModelManager、FeatureManager等模块集成

使用示例:
    >>> config = PipelineConfig.from_yaml("pipeline.yaml")
    >>> stage_config = config.get_stage_config("model_training")
    >>> config.validate()
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class StageType(Enum):
    """
    管道阶段类型枚举
    
    定义ML管道的8个标准阶段。
    
    Attributes:
        DATA_PREPARATION: 数据准备阶段
        FEATURE_ENGINEERING: 特征工程阶段
        MODEL_TRAINING: 模型训练阶段
        MODEL_EVALUATION: 模型评估阶段
        MODEL_VALIDATION: 模型验证阶段
        CANARY_DEPLOYMENT: 金丝雀部署阶段
        FULL_DEPLOYMENT: 全量部署阶段
        MONITORING: 监控阶段
    """
    DATA_PREPARATION = "data_preparation"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    MODEL_EVALUATION = "model_evaluation"
    MODEL_VALIDATION = "model_validation"
    CANARY_DEPLOYMENT = "canary_deployment"
    FULL_DEPLOYMENT = "full_deployment"
    MONITORING = "monitoring"


@dataclass
class StageConfig:
    """
    阶段配置数据类
    
    定义单个管道阶段的配置参数。
    
    Attributes:
        name: 阶段名称
        stage_type: 阶段类型
        enabled: 是否启用
        config: 阶段特定配置字典
        dependencies: 依赖的其他阶段列表
        timeout_seconds: 执行超时时间（秒）
        retry_count: 失败重试次数
        retry_delay_seconds: 重试间隔（秒）
    
    Example:
        >>> config = StageConfig(
        ...     name="model_training",
        ...     stage_type=StageType.MODEL_TRAINING,
        ...     config={"epochs": 100, "batch_size": 32}
        ... )
    """
    name: str
    stage_type: StageType = StageType.DATA_PREPARATION
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    timeout_seconds: int = 3600
    retry_count: int = 3
    retry_delay_seconds: int = 60
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StageConfig':
        """
        从字典创建阶段配置
        
        Args:
            data: 配置字典
        
        Returns:
            StageConfig 实例
        """
        stage_type_str = data.get('stage_type', 'data_preparation')
        try:
            stage_type = StageType(stage_type_str)
        except ValueError:
            stage_type = StageType.DATA_PREPARATION
        
        return cls(
            name=data.get('name', ''),
            stage_type=stage_type,
            enabled=data.get('enabled', True),
            config=data.get('config', {}),
            dependencies=data.get('dependencies', []),
            timeout_seconds=data.get('timeout_seconds', 3600),
            retry_count=data.get('retry_count', 3),
            retry_delay_seconds=data.get('retry_delay_seconds', 60)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            配置字典
        """
        return {
            'name': self.name,
            'stage_type': self.stage_type.value,
            'enabled': self.enabled,
            'config': self.config,
            'dependencies': self.dependencies,
            'timeout_seconds': self.timeout_seconds,
            'retry_count': self.retry_count,
            'retry_delay_seconds': self.retry_delay_seconds
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项
        
        Args:
            key: 配置键
            default: 默认值
        
        Returns:
            配置值
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        设置配置项
        
        Args:
            key: 配置键
            value: 配置值
        """
        self.config[key] = value


@dataclass
class RollbackTriggerConfig:
    """
    回滚触发器配置
    
    定义自动回滚的触发条件。
    
    Attributes:
        metric: 监控指标名称
        threshold: 阈值
        operator: 比较操作符 (greater_than, less_than, decrease, increase)
        duration_minutes: 持续时间（分钟）
    """
    metric: str
    threshold: float
    operator: str = "greater_than"
    duration_minutes: int = 5
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RollbackTriggerConfig':
        """从字典创建配置"""
        return cls(
            metric=data.get('metric', ''),
            threshold=data.get('threshold', 0.0),
            operator=data.get('operator', 'greater_than'),
            duration_minutes=data.get('duration_minutes', 5)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'metric': self.metric,
            'threshold': self.threshold,
            'operator': self.operator,
            'duration_minutes': self.duration_minutes
        }


@dataclass
class RollbackConfig:
    """
    回滚配置
    
    定义管道失败时的回滚策略。
    
    Attributes:
        enabled: 是否启用自动回滚
        strategy: 回滚策略 (immediate, gradual, emergency_stop)
        triggers: 回滚触发器列表
        auto_rollback: 是否自动执行回滚
        notification_channels: 通知渠道列表
    """
    enabled: bool = True
    strategy: str = "immediate"
    triggers: List[RollbackTriggerConfig] = field(default_factory=list)
    auto_rollback: bool = True
    notification_channels: List[str] = field(default_factory=lambda: ['email', 'webhook'])
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RollbackConfig':
        """从字典创建配置"""
        triggers = [
            RollbackTriggerConfig.from_dict(t)
            for t in data.get('triggers', [])
        ]
        return cls(
            enabled=data.get('enabled', True),
            strategy=data.get('strategy', 'immediate'),
            triggers=triggers,
            auto_rollback=data.get('auto_rollback', True),
            notification_channels=data.get('notification_channels', ['email', 'webhook'])
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'enabled': self.enabled,
            'strategy': self.strategy,
            'triggers': [t.to_dict() for t in self.triggers],
            'auto_rollback': self.auto_rollback,
            'notification_channels': self.notification_channels
        }


@dataclass
class MonitoringConfig:
    """
    监控配置
    
    定义管道监控和告警参数。
    
    Attributes:
        enabled: 是否启用监控
        metrics_interval_seconds: 指标收集间隔（秒）
        drift_detection: 是否启用漂移检测
        alert_thresholds: 告警阈值字典
        retention_days: 数据保留天数
    """
    enabled: bool = True
    metrics_interval_seconds: int = 60
    drift_detection: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    retention_days: int = 30
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MonitoringConfig':
        """从字典创建配置"""
        return cls(
            enabled=data.get('enabled', True),
            metrics_interval_seconds=data.get('metrics_interval_seconds', 60),
            drift_detection=data.get('drift_detection', True),
            alert_thresholds=data.get('alert_thresholds', {}),
            retention_days=data.get('retention_days', 30)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'enabled': self.enabled,
            'metrics_interval_seconds': self.metrics_interval_seconds,
            'drift_detection': self.drift_detection,
            'alert_thresholds': self.alert_thresholds,
            'retention_days': self.retention_days
        }


@dataclass
class IntegrationConfig:
    """
    集成配置
    
    定义与外部模块（ModelManager, FeatureManager等）的集成参数。
    
    Attributes:
        model_manager: ModelManager配置
        feature_manager: FeatureManager配置
        data_manager: DataManager配置
        custom_integrations: 自定义集成配置
    """
    model_manager: Dict[str, Any] = field(default_factory=dict)
    feature_manager: Dict[str, Any] = field(default_factory=dict)
    data_manager: Dict[str, Any] = field(default_factory=dict)
    custom_integrations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IntegrationConfig':
        """从字典创建配置"""
        return cls(
            model_manager=data.get('model_manager', {}),
            feature_manager=data.get('feature_manager', {}),
            data_manager=data.get('data_manager', {}),
            custom_integrations=data.get('custom_integrations', {})
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'model_manager': self.model_manager,
            'feature_manager': self.feature_manager,
            'data_manager': self.data_manager,
            'custom_integrations': self.custom_integrations
        }


@dataclass
class PipelineConfig:
    """
    管道配置主类
    
    定义完整的ML管道配置，包含所有阶段和全局设置。
    
    Attributes:
        name: 管道名称
        version: 管道版本
        description: 管道描述
        stages: 阶段配置列表
        rollback: 回滚配置
        monitoring: 监控配置
        integration: 集成配置
        global_config: 全局配置字典
    
    Example:
        >>> config = PipelineConfig(
        ...     name="quant_trading_pipeline",
        ...     version="1.0.0",
        ...     stages=[...]
        ... )
        >>> errors = config.validate()
    """
    name: str
    version: str = "1.0.0"
    description: str = ""
    stages: List[StageConfig] = field(default_factory=list)
    rollback: RollbackConfig = field(default_factory=RollbackConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    integration: IntegrationConfig = field(default_factory=IntegrationConfig)
    global_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后处理"""
        self.logger = logging.getLogger(f"pipeline.config.{self.name}")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineConfig':
        """
        从字典创建配置
        
        Args:
            data: 配置字典
        
        Returns:
            PipelineConfig 实例
        """
        pipeline_data = data.get('pipeline', data)
        
        stages = [
            StageConfig.from_dict(s)
            for s in pipeline_data.get('stages', [])
        ]
        
        return cls(
            name=pipeline_data.get('name', 'unnamed_pipeline'),
            version=pipeline_data.get('version', '1.0.0'),
            description=pipeline_data.get('description', ''),
            stages=stages,
            rollback=RollbackConfig.from_dict(pipeline_data.get('rollback', {})),
            monitoring=MonitoringConfig.from_dict(pipeline_data.get('monitoring', {})),
            integration=IntegrationConfig.from_dict(pipeline_data.get('integration', {})),
            global_config=pipeline_data.get('global_config', {})
        )
    
    @classmethod
    def from_yaml(cls, file_path: Union[str, Path]) -> 'PipelineConfig':
        """
        从YAML文件加载配置
        
        Args:
            file_path: YAML文件路径
        
        Returns:
            PipelineConfig 实例
        
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: YAML解析错误
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required to load YAML config. Install with: pip install pyyaml")
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_json(cls, file_path: Union[str, Path]) -> 'PipelineConfig':
        """
        从JSON文件加载配置
        
        Args:
            file_path: JSON文件路径
        
        Returns:
            PipelineConfig 实例
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            配置字典
        """
        return {
            'pipeline': {
                'name': self.name,
                'version': self.version,
                'description': self.description,
                'stages': [s.to_dict() for s in self.stages],
                'rollback': self.rollback.to_dict(),
                'monitoring': self.monitoring.to_dict(),
                'integration': self.integration.to_dict(),
                'global_config': self.global_config
            }
        }
    
    def to_yaml(self, file_path: Union[str, Path]) -> None:
        """
        保存为YAML文件
        
        Args:
            file_path: 保存路径
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required to save YAML config. Install with: pip install pyyaml")
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
    
    def to_json(self, file_path: Union[str, Path]) -> None:
        """
        保存为JSON文件
        
        Args:
            file_path: 保存路径
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def get_stage_config(self, stage_name: str) -> Optional[StageConfig]:
        """
        获取指定阶段的配置
        
        Args:
            stage_name: 阶段名称
        
        Returns:
            StageConfig 或 None
        """
        for stage in self.stages:
            if stage.name == stage_name:
                return stage
        return None
    
    def get_stage_config_by_type(self, stage_type: StageType) -> Optional[StageConfig]:
        """
        获取指定类型的阶段配置
        
        Args:
            stage_type: 阶段类型
        
        Returns:
            StageConfig 或 None
        """
        for stage in self.stages:
            if stage.stage_type == stage_type:
                return stage
        return None
    
    def add_stage(self, stage_config: StageConfig) -> None:
        """
        添加阶段配置
        
        Args:
            stage_config: 阶段配置
        """
        # 检查是否已存在
        existing = self.get_stage_config(stage_config.name)
        if existing:
            self.logger.warning(f"阶段 {stage_config.name} 已存在，将被覆盖")
            self.stages = [s for s in self.stages if s.name != stage_config.name]
        
        self.stages.append(stage_config)
    
    def remove_stage(self, stage_name: str) -> bool:
        """
        移除阶段配置
        
        Args:
            stage_name: 阶段名称
        
        Returns:
            True 如果成功移除
        """
        original_count = len(self.stages)
        self.stages = [s for s in self.stages if s.name != stage_name]
        return len(self.stages) < original_count
    
    def validate(self) -> List[str]:
        """
        验证配置有效性
        
        Returns:
            错误信息列表，空列表表示验证通过
        """
        errors = []
        
        # 验证管道名称
        if not self.name:
            errors.append("管道名称不能为空")
        
        # 验证版本号格式
        if not self.version or not isinstance(self.version, str):
            errors.append("版本号必须是字符串")
        
        # 验证阶段
        if not self.stages:
            errors.append("至少需要配置一个阶段")
        
        stage_names = set()
        stage_types = set()
        
        for stage in self.stages:
            # 检查重复名称
            if stage.name in stage_names:
                errors.append(f"阶段名称重复: {stage.name}")
            stage_names.add(stage.name)
            
            # 检查重复类型
            if stage.stage_type in stage_types:
                errors.append(f"阶段类型重复: {stage.stage_type.value}")
            stage_types.add(stage.stage_type)
            
            # 检查依赖是否存在
            for dep in stage.dependencies:
                if dep not in stage_names:
                    errors.append(f"阶段 {stage.name} 的依赖 {dep} 不存在或顺序错误")
            
            # 检查超时时间
            if stage.timeout_seconds <= 0:
                errors.append(f"阶段 {stage.name} 的超时时间必须大于0")
            
            # 检查重试次数
            if stage.retry_count < 0:
                errors.append(f"阶段 {stage.name} 的重试次数不能为负数")
        
        # 验证回滚配置
        if self.rollback.enabled:
            valid_strategies = ['immediate', 'gradual', 'emergency_stop']
            if self.rollback.strategy not in valid_strategies:
                errors.append(f"无效的回滚策略: {self.rollback.strategy}")
            
            for trigger in self.rollback.triggers:
                valid_operators = ['greater_than', 'less_than', 'decrease', 'increase']
                if trigger.operator not in valid_operators:
                    errors.append(f"无效的触发器操作符: {trigger.operator}")
        
        # 验证监控配置
        if self.monitoring.enabled:
            if self.monitoring.metrics_interval_seconds <= 0:
                errors.append("指标收集间隔必须大于0")
            if self.monitoring.retention_days <= 0:
                errors.append("数据保留天数必须大于0")
        
        return errors
    
    def get_execution_order(self) -> List[str]:
        """
        获取阶段执行顺序（拓扑排序）
        
        Returns:
            按依赖关系排序的阶段名称列表
        """
        # 构建依赖图
        in_degree = {stage.name: 0 for stage in self.stages}
        dependencies = {stage.name: [] for stage in self.stages}
        
        for stage in self.stages:
            for dep in stage.dependencies:
                if dep in in_degree:
                    dependencies[dep].append(stage.name)
                    in_degree[stage.name] += 1
        
        # 拓扑排序
        queue = [name for name, degree in in_degree.items() if degree == 0]
        execution_order = []
        
        while queue:
            current = queue.pop(0)
            execution_order.append(current)
            
            for dependent in dependencies[current]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        # 检查循环依赖
        if len(execution_order) != len(self.stages):
            raise ValueError("阶段依赖存在循环")
        
        return execution_order
    
    def get_global(self, key: str, default: Any = None) -> Any:
        """
        获取全局配置项
        
        Args:
            key: 配置键
            default: 默认值
        
        Returns:
            配置值
        """
        return self.global_config.get(key, default)
    
    def set_global(self, key: str, value: Any) -> None:
        """
        设置全局配置项
        
        Args:
            key: 配置键
            value: 配置值
        """
        self.global_config[key] = value


def create_default_config() -> PipelineConfig:
    """
    创建默认管道配置
    
    创建包含8个标准阶段的默认配置。
    
    Returns:
        默认PipelineConfig实例
    """
    return PipelineConfig(
        name="quant_trading_ml_pipeline",
        version="1.0.0",
        description="量化交易ML管道默认配置",
        stages=[
            StageConfig(
                name="data_preparation",
                stage_type=StageType.DATA_PREPARATION,
                enabled=True,
                config={
                    "data_sources": ["market_data"],
                    "date_range": "last_90_days",
                    "quality_checks": True,
                    "max_missing_threshold": 10.0
                }
            ),
            StageConfig(
                name="feature_engineering",
                stage_type=StageType.FEATURE_ENGINEERING,
                enabled=True,
                config={
                    "feature_selection": "variance",
                    "standardization": "zscore",
                    "max_features": 100
                },
                dependencies=["data_preparation"]
            ),
            StageConfig(
                name="model_training",
                stage_type=StageType.MODEL_TRAINING,
                enabled=True,
                config={
                    "model_type": "xgboost",
                    "hyperparameter_search": True,
                    "cross_validation_folds": 5
                },
                dependencies=["feature_engineering"]
            ),
            StageConfig(
                name="model_evaluation",
                stage_type=StageType.MODEL_EVALUATION,
                enabled=True,
                config={
                    "metrics": ["accuracy", "f1", "sharpe_ratio"],
                    "backtest": True,
                    "backtest_periods": 3
                },
                dependencies=["model_training"]
            ),
            StageConfig(
                name="model_validation",
                stage_type=StageType.MODEL_VALIDATION,
                enabled=True,
                config={
                    "ab_test": True,
                    "shadow_mode": True,
                    "validation_period_days": 7
                },
                dependencies=["model_evaluation"]
            ),
            StageConfig(
                name="canary_deployment",
                stage_type=StageType.CANARY_DEPLOYMENT,
                enabled=True,
                config={
                    "traffic_percentage": 5,
                    "duration_minutes": 30,
                    "success_criteria": {
                        "error_rate": 0.01,
                        "latency_p99": 100
                    }
                },
                dependencies=["model_validation"]
            ),
            StageConfig(
                name="full_deployment",
                stage_type=StageType.FULL_DEPLOYMENT,
                enabled=True,
                config={
                    "strategy": "blue_green",
                    "health_check": True
                },
                dependencies=["canary_deployment"]
            ),
            StageConfig(
                name="monitoring",
                stage_type=StageType.MONITORING,
                enabled=True,
                config={
                    "metrics_interval": 60,
                    "drift_detection": True,
                    "alert_channels": ["email", "webhook"]
                },
                dependencies=["full_deployment"]
            )
        ],
        rollback=RollbackConfig(
            enabled=True,
            strategy="immediate",
            triggers=[
                RollbackTriggerConfig(
                    metric="accuracy",
                    threshold=0.1,
                    operator="decrease"
                ),
                RollbackTriggerConfig(
                    metric="max_drawdown",
                    threshold=0.15,
                    operator="greater_than"
                ),
                RollbackTriggerConfig(
                    metric="drift_score",
                    threshold=0.5,
                    operator="greater_than"
                )
            ]
        ),
        monitoring=MonitoringConfig(
            enabled=True,
            metrics_interval_seconds=60,
            drift_detection=True,
            alert_thresholds={
                "accuracy": 0.7,
                "sharpe_ratio": 1.0,
                "max_drawdown": 0.15
            },
            retention_days=30
        ),
        integration=IntegrationConfig(
            model_manager={
                "model_registry_path": "./models",
                "versioning_enabled": True
            },
            feature_manager={
                "feature_store_path": "./features",
                "cache_enabled": True
            },
            data_manager={
                "data_source": "postgresql",
                "connection_pool_size": 10
            }
        ),
        global_config={
            "environment": "production",
            "log_level": "INFO",
            "parallel_execution": False
        }
    )


def load_config(config_path: Union[str, Path]) -> PipelineConfig:
    """
    加载管道配置（自动识别文件格式）
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        PipelineConfig 实例
    
    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 不支持的文件格式
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    suffix = config_path.suffix.lower()
    if suffix in ['.yaml', '.yml']:
        return PipelineConfig.from_yaml(config_path)
    elif suffix == '.json':
        return PipelineConfig.from_json(config_path)
    else:
        raise ValueError(f"不支持的配置文件格式: {suffix}")
