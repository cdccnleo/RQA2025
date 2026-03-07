"""
管道异常类定义模块

提供管道执行过程中可能发生的各种异常类型，支持错误分类和处理
"""

from enum import Enum
from typing import Optional, Dict, Any


class PipelineErrorCode(Enum):
    """管道错误代码枚举"""
    # 通用错误
    UNKNOWN_ERROR = "UNKNOWN_ERROR"
    CONFIG_ERROR = "CONFIG_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"
    
    # 阶段错误
    STAGE_EXECUTION_ERROR = "STAGE_EXECUTION_ERROR"
    STAGE_VALIDATION_ERROR = "STAGE_VALIDATION_ERROR"
    STAGE_ROLLBACK_ERROR = "STAGE_ROLLBACK_ERROR"
    
    # 数据错误
    DATA_LOAD_ERROR = "DATA_LOAD_ERROR"
    DATA_QUALITY_ERROR = "DATA_QUALITY_ERROR"
    DATA_MISSING_ERROR = "DATA_MISSING_ERROR"
    
    # 模型错误
    MODEL_TRAINING_ERROR = "MODEL_TRAINING_ERROR"
    MODEL_EVALUATION_ERROR = "MODEL_EVALUATION_ERROR"
    MODEL_LOAD_ERROR = "MODEL_LOAD_ERROR"
    MODEL_SAVE_ERROR = "MODEL_SAVE_ERROR"
    
    # 部署错误
    DEPLOYMENT_ERROR = "DEPLOYMENT_ERROR"
    CANARY_DEPLOYMENT_ERROR = "CANARY_DEPLOYMENT_ERROR"
    ROLLBACK_ERROR = "ROLLBACK_ERROR"
    
    # 监控错误
    MONITORING_ERROR = "MONITORING_ERROR"
    METRICS_ERROR = "METRICS_ERROR"
    ALERT_ERROR = "ALERT_ERROR"


class PipelineException(Exception):
    """
    管道基础异常类
    
    所有管道相关异常的基类，提供统一的错误信息结构和上下文管理
    
    Attributes:
        error_code: 错误代码
        message: 错误信息
        stage_name: 发生错误的阶段名称
        context: 错误上下文信息
    """
    
    def __init__(
        self,
        message: str,
        error_code: PipelineErrorCode = PipelineErrorCode.UNKNOWN_ERROR,
        stage_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """
        初始化管道异常
        
        Args:
            message: 错误描述信息
            error_code: 错误代码，用于分类处理
            stage_name: 发生错误的阶段名称
            context: 额外的上下文信息
            cause: 原始异常
        """
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.stage_name = stage_name
        self.context = context or {}
        self.cause = cause
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将异常转换为字典格式
        
        Returns:
            包含异常信息的字典
        """
        return {
            "error_code": self.error_code.value,
            "message": self.message,
            "stage_name": self.stage_name,
            "context": self.context,
            "cause": str(self.cause) if self.cause else None
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        parts = [f"[{self.error_code.value}] {self.message}"]
        if self.stage_name:
            parts.append(f"Stage: {self.stage_name}")
        if self.cause:
            parts.append(f"Caused by: {self.cause}")
        return " | ".join(parts)


class StageExecutionException(PipelineException):
    """
    阶段执行异常
    
    管道阶段执行过程中发生的错误
    """
    
    def __init__(
        self,
        message: str,
        stage_name: str,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            message=message,
            error_code=PipelineErrorCode.STAGE_EXECUTION_ERROR,
            stage_name=stage_name,
            context=context,
            cause=cause
        )


class StageValidationException(PipelineException):
    """
    阶段验证异常
    
    阶段输出验证失败时抛出
    """
    
    def __init__(
        self,
        message: str,
        stage_name: str,
        validation_errors: Optional[list] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            message=message,
            error_code=PipelineErrorCode.STAGE_VALIDATION_ERROR,
            stage_name=stage_name,
            context=context,
            cause=cause
        )
        self.validation_errors = validation_errors or []


class DataQualityException(PipelineException):
    """
    数据质量异常
    
    数据质量检查不通过时抛出
    """
    
    def __init__(
        self,
        message: str,
        quality_issues: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            message=message,
            error_code=PipelineErrorCode.DATA_QUALITY_ERROR,
            context=context,
            cause=cause
        )
        self.quality_issues = quality_issues


class ModelTrainingException(PipelineException):
    """
    模型训练异常
    
    模型训练过程中发生的错误
    """
    
    def __init__(
        self,
        message: str,
        model_type: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            message=message,
            error_code=PipelineErrorCode.MODEL_TRAINING_ERROR,
            context=context,
            cause=cause
        )
        self.model_type = model_type


class DeploymentException(PipelineException):
    """
    部署异常
    
    模型部署过程中发生的错误
    """
    
    def __init__(
        self,
        message: str,
        deployment_type: Optional[str] = None,
        model_version: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            message=message,
            error_code=PipelineErrorCode.DEPLOYMENT_ERROR,
            context=context,
            cause=cause
        )
        self.deployment_type = deployment_type
        self.model_version = model_version


class RollbackException(PipelineException):
    """
    回滚异常
    
    回滚操作失败时抛出
    """
    
    def __init__(
        self,
        message: str,
        rollback_target: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            message=message,
            error_code=PipelineErrorCode.ROLLBACK_ERROR,
            context=context,
            cause=cause
        )
        self.rollback_target = rollback_target


class ConfigurationException(PipelineException):
    """
    配置异常
    
    管道配置错误时抛出
    """
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            message=message,
            error_code=PipelineErrorCode.CONFIG_ERROR,
            context=context,
            cause=cause
        )
        self.config_key = config_key
