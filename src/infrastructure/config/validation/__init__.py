"""
配置验证模块
版本更新记录：
2024-04-02 v3.6.0 - 验证模块更新
    - 增强Schema验证
    - 新增类型验证
    - 完善验证工厂
"""

from .schema import ConfigValidator
from .typed_config import TypedConfigBase
from .validator_factory import ConfigValidatorFactory

__all__ = [
    'ConfigValidator',
    'TypedConfigBase',
    'ConfigValidatorFactory'
]
