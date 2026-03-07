"""
validators 模块

提供 validators 相关功能和接口。
"""

import logging
from typing import Optional, Dict, Any

from .specialized_validators import (
    TradingHoursValidator, DatabaseConfigValidator, LoggingConfigValidator, NetworkConfigValidator
)
from .validator_base import (
    ValidationSeverity, ValidationType, ValidationResult, ValidationRule,
    IConfigValidator, BaseConfigValidator
)
from .validator_composition import (
    ConfigValidators, UnifiedValidatorFactory, ConfigValidator
)

"""
统一配置验证器

拆分后的validators.py，作为验证器模块的统一入口
所有验证器功能已重构为模块化架构，提高代码组织性和可维护性
"""

logger = logging.getLogger(__name__)

# ==================== 导入所有验证器组件 ====================

# 基础组件

# 专用验证器

# 组合和工厂

# ==================== 兼容性导出 ====================

__all__ = [
    # 基础组件
    'ValidationSeverity',
    'ValidationType',
    'ValidationResult',
    'ValidationRule',
    'IConfigValidator',
    'BaseConfigValidator',

    # 专用验证器
    'TradingHoursValidator',
    'DatabaseConfigValidator',
    'LoggingConfigValidator',
    'NetworkConfigValidator',

    # 组合和工厂
    'ConfigValidators',
    'UnifiedValidatorFactory',
    'ConfigValidator',

    # 验证函数
    'validate_trading_hours',
    'validate_database_config',
    'validate_logging_config',
    'validate_network_config'
]

# ==================== 便捷函数 ====================

# 全局验证器工厂实例（用于单例模式）
_validator_factory_instance = None


def create_validator_factory() -> UnifiedValidatorFactory:
    """创建验证器工厂

    Returns:
        配置的验证器工厂实例
    """
    return UnifiedValidatorFactory()


def create_composite_validator(validator_specs: list) -> ConfigValidators:
    """创建组合验证器

        Args:
        validator_specs: 验证器规格列表

        Returns:
        组合验证器实例
    """
    factory = create_validator_factory()
    return factory.create_composite_validator(validator_specs)


def validate_config(config: dict, validator_specs: Optional[list] = None) -> list:
    """验证配置的便捷函数

    Args:
        config: 要验证的配置字典
        validator_specs: 验证器规格列表，如果为None则使用默认验证器

    Returns:
        验证结果列表
    """
    if validator_specs is None:
        # 默认验证器规格
        validator_specs = [
            {'type': 'database'},
            {'type': 'logging'},
            {'type': 'network'}
        ]

    validator = create_composite_validator(validator_specs)
    return validator.validate(config)

# ==================== 向后兼容性 ====================


# 为保持向后兼容性，提供别名
ConfigValidationResult = ValidationResult
ValidatorFactory = UnifiedValidatorFactory

# ==================== 便捷函数 (向后兼容) ====================


def create_validator(validator_type: str):
    """
    创建指定类型的验证器 (向后兼容)

    Args:
        validator_type: 验证器类型

    Returns:
        验证器实例
    """
    factory = create_validator_factory()
    return factory.create_validator(validator_type)


def create_validator_suite(validator_types: list):
    """
    创建验证器套件 (向后兼容)

    Args:
        validator_types: 验证器类型列表

    Returns:
        验证器套件实例
    """
    factory = create_validator_factory()
    return factory.create_validator_suite(validator_types)


def validate_config_with_suite(config: Dict[str, Any], validator_types: list):
    """
    使用验证器套件验证配置 (向后兼容)

    Args:
        config: 配置字典
        validator_types: 验证器类型列表

    Returns:
        (is_valid, result) 元组
    """
    suite = create_validator_suite(validator_types)
    return suite.validate(config)


def get_validator_factory():
    """
    获取验证器工厂实例 (向后兼容)

    Returns:
        验证器工厂实例
    """
    global _validator_factory_instance
    if _validator_factory_instance is None:
        _validator_factory_instance = create_validator_factory()
    return _validator_factory_instance


def reset_validator_factory():
    """
    重置验证器工厂 (向后兼容)

    主要用于测试清理
    """
    global _validator_factory_instance
    _validator_factory_instance = None

# ==================== 便捷验证函数 ====================


def validate_trading_hours(config: dict) -> bool:
    """
    验证交易时间配置

    Args:
        config: 配置字典

    Returns:
        bool: 验证是否通过
    """
    try:
        from .specialized_validators import TradingHoursValidator

        validator = TradingHoursValidator()
        result = validator.validate(config)

        # 对于交易时间，缺少配置只是警告，不是错误
        return result.is_valid
    except Exception:
        return False


def validate_database_config(config: dict) -> bool:
    """
    验证数据库配置

    Args:
        config: 配置字典

    Returns:
        bool: 验证是否通过
    """
    try:
        if not isinstance(config, dict):
            return False

        db_config = config.get("database", {})
        if not isinstance(db_config, dict):
            return False

        # 检查必需的数据库字段
        required_fields = ["host", "port", "username", "password"]
        for field in required_fields:
            if field not in db_config:
                return False

        # 验证端口范围
        port = db_config.get("port")
        if not isinstance(port, int) or not (1 <= port <= 65535):
            return False

        return True
    except Exception:
        return False


def validate_logging_config(config: dict) -> bool:
    """
    验证日志配置

    Args:
        config: 配置字典

    Returns:
        bool: 验证是否通过
    """
    try:
        if not isinstance(config, dict):
            return False

        logging_config = config.get("logging", {})
        if not isinstance(logging_config, dict):
            return False

        # 检查日志级别
        level = logging_config.get("level")
        if not level:
            return False

        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if level.upper() not in valid_levels:
            return False

        return True
    except Exception:
        return False


def validate_network_config(config: dict) -> bool:
    """
    验证网络配置

    Args:
        config: 配置字典

    Returns:
        bool: 验证是否通过
    """
    try:
        if not isinstance(config, dict):
            return False

        network_config = config.get("network", {})
        if not isinstance(network_config, dict):
            return False

        # 检查端口配置
        port = network_config.get("port")
        if port is not None:
            if not isinstance(port, int) or not (1 <= port <= 65535):
                return False

        # 检查主机配置
        host = network_config.get("host")
        if host is not None and not isinstance(host, str):
            return False

        return True
    except Exception:
        return False




