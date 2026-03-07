"""
管道配置管理模块

提供管道配置的加载、验证和管理功能，支持YAML/JSON格式配置
"""

import os
import yaml
import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from enum import Enum

from .exceptions import ConfigurationException


class StageType(Enum):
    """管道阶段类型枚举"""
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
    
    Attributes:
        name: 阶段名称
        enabled: 是否启用
        config: 阶段特定配置
        dependencies: 依赖的其他阶段
        timeout_seconds: 阶段执行超时时间
        retry_count: 失败重试次数
        retry_delay_seconds: 重试间隔
    """
    name: str
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    timeout_seconds: int = 3600
    retry_count: int = 3
    retry_delay_seconds: int = 60
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StageConfig':
        """从字典创建配置"""
        return cls(
            name=data.get('name', ''),
            enabled=data.get('enabled', True),
            config=data.get('config', {}),
            dependencies=data.get('dependencies', []),
            timeout_seconds=data.get('timeout_seconds', 3600),
            retry_count=data.get('retry_count', 3),
            retry_delay_seconds=data.get('retry_delay_seconds', 60)
        )


@dataclass
class RollbackTriggerConfig:
    """
    回滚触发器配置
    
    Attributes:
        metric: 监控指标名称
        threshold: 阈值
        operator: 比较操作符 (greater_than, less_than, decrease, increase)
        duration_minutes: 持续时间
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


@dataclass
class RollbackConfig:
    """
    回滚配置
    
    Attributes:
        enabled: 是否启用自动回滚
        strategy: 回滚策略 (immediate, gradual, emergency_stop)
        triggers: 回滚触发器列表
        auto_rollback: 是否自动执行回滚
        notification_channels: 通知渠道
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


@dataclass
class MonitoringConfig:
    """
    监控配置
    
    Attributes:
        enabled: 是否启用监控
        metrics_interval_seconds: 指标收集间隔
        drift_detection: 是否启用漂移检测
        alert_thresholds: 告警阈值
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


@dataclass
class PipelineConfig:
    """
    管道配置主类
    
    Attributes:
        name: 管道名称
        version: 管道版本
        stages: 阶段配置列表
        rollback: 回滚配置
        monitoring: 监控配置
        global_config: 全局配置
    """
    name: str
    version: str = "1.0.0"
    stages: List[StageConfig] = field(default_factory=list)
    rollback: RollbackConfig = field(default_factory=RollbackConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    global_config: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineConfig':
        """从字典创建配置"""
        pipeline_data = data.get('pipeline', data)
        
        stages = [
            StageConfig.from_dict(s) 
            for s in pipeline_data.get('stages', [])
        ]
        
        return cls(
            name=pipeline_data.get('name', 'unnamed_pipeline'),
            version=pipeline_data.get('version', '1.0.0'),
            stages=stages,
            rollback=RollbackConfig.from_dict(pipeline_data.get('rollback', {})),
            monitoring=MonitoringConfig.from_dict(pipeline_data.get('monitoring', {})),
            global_config=pipeline_data.get('global_config', {})
        )
    
    @classmethod
    def from_yaml(cls, file_path: Union[str, Path]) -> 'PipelineConfig':
        """
        从YAML文件加载配置
        
        Args:
            file_path: YAML文件路径
            
        Returns:
            PipelineConfig实例
            
        Raises:
            ConfigurationException: 文件不存在或格式错误
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ConfigurationException(
                message=f"配置文件不存在: {file_path}",
                config_key="file_path"
            )
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            return cls.from_dict(data)
        except yaml.YAMLError as e:
            raise ConfigurationException(
                message=f"YAML解析错误: {e}",
                config_key=str(file_path),
                cause=e
            )
    
    @classmethod
    def from_json(cls, file_path: Union[str, Path]) -> 'PipelineConfig':
        """
        从JSON文件加载配置
        
        Args:
            file_path: JSON文件路径
            
        Returns:
            PipelineConfig实例
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ConfigurationException(
                message=f"配置文件不存在: {file_path}",
                config_key="file_path"
            )
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            raise ConfigurationException(
                message=f"JSON解析错误: {e}",
                config_key=str(file_path),
                cause=e
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'pipeline': {
                'name': self.name,
                'version': self.version,
                'stages': [
                    {
                        'name': s.name,
                        'enabled': s.enabled,
                        'config': s.config,
                        'dependencies': s.dependencies,
                        'timeout_seconds': s.timeout_seconds,
                        'retry_count': s.retry_count,
                        'retry_delay_seconds': s.retry_delay_seconds
                    }
                    for s in self.stages
                ],
                'rollback': {
                    'enabled': self.rollback.enabled,
                    'strategy': self.rollback.strategy,
                    'triggers': [
                        {
                            'metric': t.metric,
                            'threshold': t.threshold,
                            'operator': t.operator,
                            'duration_minutes': t.duration_minutes
                        }
                        for t in self.rollback.triggers
                    ],
                    'auto_rollback': self.rollback.auto_rollback,
                    'notification_channels': self.rollback.notification_channels
                },
                'monitoring': {
                    'enabled': self.monitoring.enabled,
                    'metrics_interval_seconds': self.monitoring.metrics_interval_seconds,
                    'drift_detection': self.monitoring.drift_detection,
                    'alert_thresholds': self.monitoring.alert_thresholds,
                    'retention_days': self.monitoring.retention_days
                },
                'global_config': self.global_config
            }
        }
    
    def to_yaml(self, file_path: Union[str, Path]) -> None:
        """保存为YAML文件"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
    
    def to_json(self, file_path: Union[str, Path]) -> None:
        """保存为JSON文件"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def get_stage_config(self, stage_name: str) -> Optional[StageConfig]:
        """获取指定阶段的配置"""
        for stage in self.stages:
            if stage.name == stage_name:
                return stage
        return None
    
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
        
        # 验证阶段
        if not self.stages:
            errors.append("至少需要配置一个阶段")
        
        stage_names = set()
        for stage in self.stages:
            # 检查重复名称
            if stage.name in stage_names:
                errors.append(f"阶段名称重复: {stage.name}")
            stage_names.add(stage.name)
            
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
        
        return errors


def load_pipeline_config(config_path: Union[str, Path]) -> PipelineConfig:
    """
    加载管道配置（自动识别文件格式）
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        PipelineConfig实例
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise ConfigurationException(
            message=f"配置文件不存在: {config_path}",
            config_key="config_path"
        )
    
    # 根据扩展名选择加载方式
    suffix = config_path.suffix.lower()
    if suffix in ['.yaml', '.yml']:
        return PipelineConfig.from_yaml(config_path)
    elif suffix == '.json':
        return PipelineConfig.from_json(config_path)
    else:
        raise ConfigurationException(
            message=f"不支持的配置文件格式: {suffix}",
            config_key="config_path"
        )


def create_default_config() -> PipelineConfig:
    """创建默认管道配置"""
    return PipelineConfig(
        name="quant_trading_ml_pipeline",
        version="1.0.0",
        stages=[
            StageConfig(
                name="data_preparation",
                enabled=True,
                config={
                    "data_sources": ["market_data"],
                    "date_range": "last_90_days",
                    "quality_checks": True
                }
            ),
            StageConfig(
                name="feature_engineering",
                enabled=True,
                config={
                    "feature_selection": "variance",
                    "standardization": "zscore"
                },
                dependencies=["data_preparation"]
            ),
            StageConfig(
                name="model_training",
                enabled=True,
                config={
                    "model_type": "xgboost",
                    "hyperparameter_search": True
                },
                dependencies=["feature_engineering"]
            ),
            StageConfig(
                name="model_evaluation",
                enabled=True,
                config={
                    "metrics": ["accuracy", "f1", "sharpe_ratio"],
                    "backtest": True
                },
                dependencies=["model_training"]
            ),
            StageConfig(
                name="model_validation",
                enabled=True,
                config={
                    "ab_test": True,
                    "shadow_mode": True
                },
                dependencies=["model_evaluation"]
            ),
            StageConfig(
                name="canary_deployment",
                enabled=True,
                config={
                    "traffic_percentage": 5,
                    "duration_minutes": 30
                },
                dependencies=["model_validation"]
            ),
            StageConfig(
                name="full_deployment",
                enabled=True,
                config={
                    "strategy": "blue_green"
                },
                dependencies=["canary_deployment"]
            ),
            StageConfig(
                name="monitoring",
                enabled=True,
                config={
                    "metrics_interval": 60,
                    "drift_detection": True
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
            }
        )
    )
