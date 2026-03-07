"""
管道阶段模块

提供8个标准的ML管道阶段实现
"""

from .base import PipelineStage, StageResult, StageStatus, CompositeStage
from .data_preparation import DataPreparationStage, DataQualityReport
from .feature_engineering import FeatureEngineeringStage
from .model_training import ModelTrainingStage
from .model_evaluation import ModelEvaluationStage, BacktestResult
from .model_validation import ModelValidationStage, ValidationResult
from .canary_deployment import CanaryDeploymentStage, CanaryMetrics
from .full_deployment import FullDeploymentStage, DeploymentStatus
from .monitoring import MonitoringStage, MonitoringMetrics

__all__ = [
    # 基类
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
