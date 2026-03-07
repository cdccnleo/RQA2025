"""配置系统核心接口"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from enum import Enum

class ValidationResult:
    """验证结果"""
    def __init__(self, valid: bool, message: str = "", errors: Optional[List[str]] = None):
        self.valid = valid
        self.message = message
        self.errors = errors or []

class ValidationSeverity(Enum):
    """验证严重程度"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class BaseConfigValidator(ABC):
    """配置验证器基类"""

    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> ValidationResult:
        """验证配置"""
        pass

class TypedConfigBase:
    """类型化配置基类"""

    def __init__(self):
        self._config = {}

    def set_config(self, key: str, value: Any):
        """设置配置项"""
        self._config[key] = value

    def get_config(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        return self._config.get(key, default)

    def validate(self) -> ValidationResult:
        """验证配置"""
        return ValidationResult(True, "Configuration is valid")




