"""
ML管道模块

提供端到端的自动化机器学习训练管道
"""

from .controller import MLPipelineController, PipelineContext, PipelineExecutionResult, PipelineStatus
from .config import (
    PipelineConfig,
    StageConfig,
    RollbackConfig,
    RollbackTriggerConfig,
    MonitoringConfig,
    load_pipeline_config,
    create_default_config
)
from .exceptions import (
    PipelineException,
    StageExecutionException,
    StageValidationException,
    DataQualityException,
    ModelTrainingException,
    DeploymentException,
    RollbackException,
    ConfigurationException,
    PipelineErrorCode
)
from .stages import (
    # 基类
    PipelineStage,
    StageResult,
    StageStatus,
    CompositeStage,
    
    # 阶段实现
    DataPreparationStage,
    FeatureEngineeringStage,
    ModelTrainingStage,
    ModelEvaluationStage,
    ModelValidationStage,
    CanaryDeploymentStage,
    FullDeploymentStage,
    MonitoringStage,
    
    # 数据类
    DataQualityReport,
    BacktestResult,
    ValidationResult,
    CanaryMetrics,
    DeploymentStatus,
    MonitoringMetrics
)

__all__ = [
    # 控制器
    "MLPipelineController",
    "PipelineContext",
    "PipelineExecutionResult",
    "PipelineStatus",
    
    # 配置
    "PipelineConfig",
    "StageConfig",
    "RollbackConfig",
    "RollbackTriggerConfig",
    "MonitoringConfig",
    "load_pipeline_config",
    "create_default_config",
    
    # 异常
    "PipelineException",
    "StageExecutionException",
    "StageValidationException",
    "DataQualityException",
    "ModelTrainingException",
    "DeploymentException",
    "RollbackException",
    "ConfigurationException",
    "PipelineErrorCode",
    
    # 阶段基类
    "PipelineStage",
    "StageResult",
    "StageStatus",
    "CompositeStage",
    
    # 阶段实现
    "DataPreparationStage",
    "FeatureEngineeringStage",
    "ModelTrainingStage",
    "ModelEvaluationStage",
    "ModelValidationStage",
    "CanaryDeploymentStage",
    "FullDeploymentStage",
    "MonitoringStage",
    
    # 数据类
    "DataQualityReport",
    "BacktestResult",
    "ValidationResult",
    "CanaryMetrics",
    "DeploymentStatus",
    "MonitoringMetrics"
]
