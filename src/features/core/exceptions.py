#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征层异常定义

定义特征层中使用的各种异常类型，提供统一的错误处理机制。
"""

from typing import List, Any, Optional
from enum import Enum


class FeatureErrorType(Enum):

    """特征错误类型枚举"""
    DATA_VALIDATION = "data_validation"
    CONFIG_VALIDATION = "config_validation"
    PROCESSING = "processing"
    STANDARDIZATION = "standardization"
    SELECTION = "selection"
    SENTIMENT = "sentiment"
    TECHNICAL = "technical"
    GENERAL = "general"


class FeatureDataValidationError(ValueError):

    """特征数据验证错误"""

    def __init__(self, message: str, missing_columns: Optional[List[str]] = None,


                 invalid_types: Optional[List[str]] = None, data_shape: Optional[tuple] = None):
        self.message = message
        self.missing_columns = missing_columns or []
        self.invalid_types = invalid_types or []
        self.data_shape = data_shape
        self.error_type = FeatureErrorType.DATA_VALIDATION
        super().__init__(self.message)

    def __str__(self):

        details = []
        if self.missing_columns:
            details.append(f"缺失列: {self.missing_columns}")
        if self.invalid_types:
            details.append(f"无效类型: {self.invalid_types}")
        if self.data_shape:
            details.append(f"数据形状: {self.data_shape}")

        if details:
            return f"{self.message} - {'; '.join(details)}"
        return self.message


class FeatureConfigValidationError(ValueError):

    """特征配置验证错误"""

    def __init__(self, message: str, config_field: Optional[str] = None,


                 expected_value: Any = None, actual_value: Any = None,
                 config_dict: Optional[dict] = None):
        self.message = message
        self.config_field = config_field
        self.expected_value = expected_value
        self.actual_value = actual_value
        self.config_dict = config_dict or {}
        self.error_type = FeatureErrorType.CONFIG_VALIDATION
        super().__init__(self.message)

    def __str__(self):

        details = []
        if self.config_field:
            details.append(f"配置字段: {self.config_field}")
        if self.expected_value is not None:
            details.append(f"期望值: {self.expected_value}")
        if self.actual_value is not None:
            details.append(f"实际值: {self.actual_value}")

        if details:
            return f"{self.message} - {'; '.join(details)}"
        return self.message


class FeatureProcessingError(RuntimeError):

    """特征处理错误"""

    def __init__(self, message: str, processor_name: Optional[str] = None,


                 step: Optional[str] = None, original_error: Optional[Exception] = None,
                 feature_name: Optional[str] = None):
        self.message = message
        self.processor_name = processor_name
        self.step = step
        self.original_error = original_error
        self.feature_name = feature_name
        self.error_type = FeatureErrorType.PROCESSING
        super().__init__(self.message)

    def __str__(self):

        details = []
        if self.processor_name:
            details.append(f"处理器: {self.processor_name}")
        if self.step:
            details.append(f"步骤: {self.step}")
        if self.feature_name:
            details.append(f"特征: {self.feature_name}")
        if self.original_error:
            details.append(f"原始错误: {str(self.original_error)}")

        if details:
            return f"{self.message} - {'; '.join(details)}"
        return self.message


class FeatureStandardizationError(RuntimeError):

    """特征标准化错误"""

    def __init__(self, message: str, method: Optional[str] = None,


                 scaler_path: Optional[str] = None, is_fitted: bool = False):
        self.message = message
        self.method = method
        self.scaler_path = scaler_path
        self.is_fitted = is_fitted
        self.error_type = FeatureErrorType.STANDARDIZATION
        super().__init__(self.message)

    def __str__(self):

        details = []
        if self.method:
            details.append(f"标准化方法: {self.method}")
        if self.scaler_path:
            details.append(f"模型路径: {self.scaler_path}")
        details.append(f"已拟合: {self.is_fitted}")

        if details:
            return f"{self.message} - {'; '.join(details)}"
        return self.message


class FeatureSelectionError(RuntimeError):

    """特征选择错误"""

    def __init__(self, message: str, selection_method: Optional[str] = None,


                 target_column: Optional[str] = None, feature_count: Optional[int] = None):
        self.message = message
        self.selection_method = selection_method
        self.target_column = target_column
        self.feature_count = feature_count
        self.error_type = FeatureErrorType.SELECTION
        super().__init__(self.message)

    def __str__(self):

        details = []
        if self.selection_method:
            details.append(f"选择方法: {self.selection_method}")
        if self.target_column:
            details.append(f"目标列: {self.target_column}")
        if self.feature_count is not None:
            details.append(f"特征数量: {self.feature_count}")

        if details:
            return f"{self.message} - {'; '.join(details)}"
        return self.message


class FeatureSentimentError(RuntimeError):

    """特征情感分析错误"""

    def __init__(self, message: str, text_length: Optional[int] = None,


                 model_type: Optional[str] = None, batch_size: Optional[int] = None):
        self.message = message
        self.text_length = text_length
        self.model_type = model_type
        self.batch_size = batch_size
        self.error_type = FeatureErrorType.SENTIMENT
        super().__init__(self.message)

    def __str__(self):

        details = []
        if self.text_length is not None:
            details.append(f"文本长度: {self.text_length}")
        if self.model_type:
            details.append(f"模型类型: {self.model_type}")
        if self.batch_size is not None:
            details.append(f"批次大小: {self.batch_size}")

        if details:
            return f"{self.message} - {'; '.join(details)}"
        return self.message


class FeatureTechnicalError(RuntimeError):

    """特征技术指标错误"""

    def __init__(self, message: str, indicator_name: Optional[str] = None,


                 period: Optional[int] = None, data_length: Optional[int] = None):
        self.message = message
        self.indicator_name = indicator_name
        self.period = period
        self.data_length = data_length
        self.error_type = FeatureErrorType.TECHNICAL
        super().__init__(self.message)

    def __str__(self):

        details = []
        if self.indicator_name:
            details.append(f"指标名称: {self.indicator_name}")
        if self.period is not None:
            details.append(f"周期: {self.period}")
        if self.data_length is not None:
            details.append(f"数据长度: {self.data_length}")

        if details:
            return f"{self.message} - {'; '.join(details)}"
        return self.message


class FeatureGeneralError(Exception):

    """特征通用错误"""

    def __init__(self, message: str, error_code: Optional[str] = None,


                 context: Optional[dict] = None):
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.error_type = FeatureErrorType.GENERAL
        super().__init__(self.message)

    def __str__(self):

        details = []
        if self.error_code:
            details.append(f"错误代码: {self.error_code}")
        if self.context:
            details.append(f"上下文: {self.context}")

        if details:
            return f"{self.message} - {'; '.join(details)}"
        return self.message


# 异常工厂类

class FeatureExceptionFactory:

    """特征异常工厂类"""

    @staticmethod
    def create_data_validation_error(message: str, **kwargs) -> FeatureDataValidationError:
        """创建数据验证错误"""
        return FeatureDataValidationError(message, **kwargs)

    @staticmethod
    def create_config_validation_error(message: str, **kwargs) -> FeatureConfigValidationError:
        """创建配置验证错误"""
        return FeatureConfigValidationError(message, **kwargs)

    @staticmethod
    def create_processing_error(message: str, **kwargs) -> FeatureProcessingError:
        """创建处理错误"""
        return FeatureProcessingError(message, **kwargs)

    @staticmethod
    def create_standardization_error(message: str, **kwargs) -> FeatureStandardizationError:
        """创建标准化错误"""
        return FeatureStandardizationError(message, **kwargs)

    @staticmethod
    def create_selection_error(message: str, **kwargs) -> FeatureSelectionError:
        """创建选择错误"""
        return FeatureSelectionError(message, **kwargs)

    @staticmethod
    def create_sentiment_error(message: str, **kwargs) -> FeatureSentimentError:
        """创建情感分析错误"""
        return FeatureSentimentError(message, **kwargs)

    @staticmethod
    def create_technical_error(message: str, **kwargs) -> FeatureTechnicalError:
        """创建技术指标错误"""
        return FeatureTechnicalError(message, **kwargs)

    @staticmethod
    def create_general_error(message: str, **kwargs) -> FeatureGeneralError:
        """创建通用错误"""
        return FeatureGeneralError(message, **kwargs)


# 异常处理工具类

class FeatureExceptionHandler:

    """特征异常处理工具类"""

    def __init__(self):

        self.exception_factory = FeatureExceptionFactory()
        self.error_count = 0
        self.error_history = []

    def handle_exception(self, exception: Exception, context: Optional[dict] = None) -> Exception:
        """处理异常，添加上下文信息"""
        self.error_count += 1

        # 记录错误历史
        error_info = {
            "error_id": f"FE_{self.error_count:06d}",
            "error_type": type(exception).__name__,
            "error_message": str(exception),
            "context": context or {},
            "timestamp": self._get_timestamp()
        }
        self.error_history.append(error_info)

        # 根据异常类型添加额外信息
        if isinstance(exception, FeatureDataValidationError):
            return self._enhance_data_validation_error(exception, context)
        elif isinstance(exception, FeatureConfigValidationError):
            return self._enhance_config_validation_error(exception, context)
        elif isinstance(exception, FeatureProcessingError):
            return self._enhance_processing_error(exception, context)
        else:
            return exception

    def _enhance_data_validation_error(self, error: FeatureDataValidationError,


                                       context: Optional[dict]) -> FeatureDataValidationError:
        """增强数据验证错误"""
        if context and "data_shape" in context:
            error.data_shape = context["data_shape"]
        return error

    def _enhance_config_validation_error(self, error: FeatureConfigValidationError,


                                         context: Optional[dict]) -> FeatureConfigValidationError:
        """增强配置验证错误"""
        if context and "config" in context:
            error.config_dict = context["config"]
        return error

    def _enhance_processing_error(self, error: FeatureProcessingError,


                                  context: Optional[dict]) -> FeatureProcessingError:
        """增强处理错误"""
        if context and "processor" in context:
            error.processor_name = context["processor"]
        if context and "step" in context:
            error.step = context["step"]
        return error

    def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()

    def get_error_summary(self) -> dict:
        """获取错误摘要"""
        error_types = {}
        for error_info in self.error_history:
            error_type = error_info["error_type"]
            error_types[error_type] = error_types.get(error_type, 0) + 1

        return {
            "total_errors": self.error_count,
            "error_types": error_types,
            "recent_errors": self.error_history[-10:] if self.error_history else []
        }

    def clear_history(self):
        """清除错误历史"""
        self.error_history.clear()
        self.error_count = 0


# 全局异常处理器实例
feature_exception_handler = FeatureExceptionHandler()


def handle_feature_exception(func):
    """特征异常处理装饰器"""
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):

        try:
            return func(*args, **kwargs)
        except Exception as e:
            # 使用全局异常处理器处理异常
            enhanced_exception = feature_exception_handler.handle_exception(e, {
                "function": func.__name__,
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys())
            })
            raise enhanced_exception

    return wrapper
