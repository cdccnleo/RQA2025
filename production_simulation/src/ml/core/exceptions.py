"""
机器学习层异常处理
ML Layer Exception Handling

定义机器学习相关的异常类和错误处理机制
"""


class MLException(Exception):
    """机器学习基础异常类"""

    def __init__(self, message: str, error_code: int = -1):
        super().__init__(message)
        self.error_code = error_code
        self.message = message


class ModelTrainingError(MLException):
    """模型训练错误"""

    def __init__(self, message: str, model_type: str = None):
        super().__init__(f"模型训练失败 - {model_type}: {message}")
        self.model_type = model_type


class ModelPredictionError(MLException):
    """模型预测错误"""

    def __init__(self, message: str, model_id: str = None):
        super().__init__(f"模型预测失败 - {model_id}: {message}")
        self.model_id = model_id


class DataValidationError(MLException):
    """数据验证错误"""

    def __init__(self, message: str, field: str = None):
        super().__init__(f"数据验证失败 - {field}: {message}")
        self.field = field


class ModelNotFoundError(MLException):
    """模型未找到错误"""

    def __init__(self, model_id: str):
        super().__init__(f"模型未找到: {model_id}", 404)
        self.model_id = model_id


class ConfigurationError(MLException):
    """配置错误"""

    def __init__(self, message: str, config_key: str = None):
        super().__init__(f"配置错误 - {config_key}: {message}")
        self.config_key = config_key


class ResourceExhaustionError(MLException):
    """资源耗尽错误"""

    def __init__(self, message: str, resource_type: str = None):
        super().__init__(f"资源耗尽 - {resource_type}: {message}")
        self.resource_type = resource_type


def handle_ml_exception(func):
    """
    装饰器：统一处理机器学习异常

    Args:
        func: 被装饰的函数

    Returns:
        包装后的函数
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except MLException:
            # 重新抛出ML异常
            raise
        except Exception as e:
            # 将其他异常包装为ML异常
            raise MLException(f"意外错误: {str(e)}") from e

    return wrapper


def validate_data_shape(X, y=None, expected_features=None):
    """
    验证数据形状

    Args:
        X: 特征数据
        y: 目标变量 (可选)
        expected_features: 期望的特征数量 (可选)

    Raises:
        DataValidationError: 数据验证失败
    """
    if X is None or len(X) == 0:
        raise DataValidationError("特征数据不能为空")

    if expected_features and X.shape[1] != expected_features:
        raise DataValidationError(
            f"特征数量不匹配，期望{expected_features}个，实际{X.shape[1]}个",
            "features"
        )

    if y is not None and len(y) != len(X):
        raise DataValidationError(
            f"特征和目标变量长度不匹配，X:{len(X)}, y:{len(y)}",
            "target"
        )


def validate_model_exists(model_dict, model_id):
    """
    验证模型是否存在

    Args:
        model_dict: 模型字典
        model_id: 模型ID

    Raises:
        ModelNotFoundError: 模型不存在
    """
    if model_id not in model_dict:
        raise ModelNotFoundError(model_id)
