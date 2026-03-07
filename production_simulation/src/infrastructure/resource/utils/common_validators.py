"""
common_validators 模块

提供 common_validators 相关功能和接口。
"""

from ..config.config_classes import (
    TaskConfig, ProcessConfig, MonitorConfig, AlertConfig,
    ResourceConfig, OptimizationConfig, APIConfig, ResourceMonitorConfig
)
from ..shared_interfaces import ConfigValidator, ConfigurationException
from typing import Dict, List, Any, Optional, Callable

"""
通用配置验证器
用于Phase 7: 代码重复消除和接口标准化

提供统一的配置验证功能，消除重复的验证代码。
"""


class TaskConfigValidator(ConfigValidator):
    """任务配置验证器"""

    def validate_config(self, config: Any) -> bool:
        """验证任务配置"""
        if not super().validate_config(config):
            return False

        if not isinstance(config, TaskConfig):
            self.errors.append("配置必须是TaskConfig实例")
            return False

        # 验证必需字段
        required_fields = ['task_type']
        if not self.validate_required_fields(vars(config), required_fields):
            return False

        # 验证字段类型
        field_types = {
            'priority': int,
            'timeout': int,
            'max_retries': int
        }
        if not self.validate_field_types(vars(config), field_types):
            return False

        # 验证优先级范围
        if hasattr(config, 'priority') and config.priority is not None:
            if not (1 <= config.priority <= 5):
                self.errors.append("优先级必须在1-5之间")

        return True


class ProcessConfigValidator(ConfigValidator):
    """进程配置验证器"""

    def validate_config(self, config: Any) -> bool:
        """验证进程配置"""
        if not super().validate_config(config):
            return False

        if not isinstance(config, ProcessConfig):
            self.errors.append("配置必须是ProcessConfig实例")
            return False

        # 验证必需字段
        required_fields = ['action']
        if not self.validate_required_fields(vars(config), required_fields):
            return False

        # 验证action的有效性
        valid_actions = ['allocate_quota', 'release_quota', 'check_quota', 'monitor_quota_usage',
                         'set_policy', 'generate_report', 'backup_quota_state', 'restore_quota_state',
                         'get_audit_log', 'configure_alerts', 'auto_scale_quota', 'calculate_cost']

        if hasattr(config, 'action') and config.action not in valid_actions:
            self.errors.append(f"action必须是以下之一: {', '.join(valid_actions)}")

        return True


class MonitorConfigValidator(ConfigValidator):
    """监控配置验证器"""

    def validate_config(self, config: Any) -> bool:
        """验证监控配置"""
        if not super().validate_config(config):
            return False

        if not isinstance(config, (MonitorConfig, ResourceMonitorConfig)):
            self.errors.append("配置必须是MonitorConfig或ResourceMonitorConfig实例")
            return False

        # 验证数值范围
        if hasattr(config, 'monitor_interval') and config.monitor_interval is not None:
            if config.monitor_interval < 1:
                self.errors.append("监控间隔必须大于等于1秒")

        if hasattr(config, 'history_size') and config.history_size is not None:
            if config.history_size < 1:
                self.errors.append("历史记录大小必须大于等于1")

        # 验证阈值范围
        threshold_fields = ['cpu_warning', 'memory_warning', 'disk_warning']
        for field in threshold_fields:
            if hasattr(config, field):
                value = getattr(config, field)
                if value is not None and not (0 <= value <= 100):
                    self.errors.append(f"{field}必须在0-100之间")

        return True


class AlertConfigValidator(ConfigValidator):
    """告警配置验证器"""

    def validate_config(self, config: Any) -> bool:
        """验证告警配置"""
        if not super().validate_config(config):
            return False

        if not isinstance(config, AlertConfig):
            self.errors.append("配置必须是AlertConfig实例")
            return False

        # 验证告警级别
        if hasattr(config, 'severity') and config.severity is not None:
            valid_severities = ['low', 'medium', 'high', 'critical']
            if config.severity not in valid_severities:
                self.errors.append(f"severity必须是以下之一: {', '.join(valid_severities)}")

        # 验证通道配置
        if hasattr(config, 'channels') and config.channels is not None:
            if not isinstance(config.channels, list):
                self.errors.append("channels必须是列表")
            else:
                for channel in config.channels:
                    if not isinstance(channel, dict) or 'type' not in channel:
                        self.errors.append("每个channel必须是包含'type'字段的字典")
                        break

        return True


class ResourceConfigValidator(ConfigValidator):
    """资源配置验证器"""

    def validate_config(self, config: Any) -> bool:
        """验证资源配置"""
        if not super().validate_config(config):
            return False

        if not isinstance(config, ResourceConfig):
            self.errors.append("配置必须是ResourceConfig实例")
            return False

        # 验证资源类型
        if hasattr(config, 'resource_types') and config.resource_types is not None:
            if not isinstance(config.resource_types, list):
                self.errors.append("resource_types必须是列表")
            else:
                valid_types = ['cpu', 'memory', 'disk', 'network', 'gpu']
                for res_type in config.resource_types:
                    if res_type not in valid_types:
                        self.errors.append(f"resource_types包含无效类型: {res_type}")
                        break

        return True


class OptimizationConfigValidator(ConfigValidator):
    """优化配置验证器"""

    def validate_config(self, config: Any) -> bool:
        """验证优化配置"""
        if not super().validate_config(config):
            return False

        if not isinstance(config, OptimizationConfig):
            self.errors.append("配置必须是OptimizationConfig实例")
            return False

        # 验证优化类型
        if hasattr(config, 'optimization_type') and config.optimization_type is not None:
            valid_types = ['performance', 'memory', 'cpu', 'network', 'comprehensive']
            if config.optimization_type not in valid_types:
                self.errors.append(f"optimization_type必须是以下之一: {', '.join(valid_types)}")

        # 验证约束条件
        if hasattr(config, 'constraints') and config.constraints is not None:
            if not isinstance(config.constraints, dict):
                self.errors.append("constraints必须是字典")
            else:
                # 验证数值约束
                numeric_constraints = ['cpu_usage', 'memory_usage', 'disk_usage', 'network_usage']
                for constraint in numeric_constraints:
                    if constraint in config.constraints:
                        value = config.constraints[constraint]
                        if not isinstance(value, (int, float)) or value < 0:
                            self.errors.append(f"{constraint}必须是非负数")

        return True


class APIConfigValidator(ConfigValidator):
    """API配置验证器"""

    def validate_config(self, config: Any) -> bool:
        """验证API配置"""
        if not super().validate_config(config):
            return False

        if not isinstance(config, APIConfig):
            self.errors.append("配置必须是APIConfig实例")
            return False

        # 验证基础URL
        if hasattr(config, 'base_url') and config.base_url is not None:
            if not isinstance(config.base_url, str) or not config.base_url.startswith('/'):
                self.errors.append("base_url必须是以'/'开头的字符串")

        # 验证速率限制
        if hasattr(config, 'rate_limit') and config.rate_limit is not None:
            if not isinstance(config.rate_limit, dict):
                self.errors.append("rate_limit必须是字典")
            elif 'requests_per_minute' in config.rate_limit:
                rpm = config.rate_limit['requests_per_minute']
                if not isinstance(rpm, int) or rpm < 1:
                    self.errors.append("requests_per_minute必须是大于等于1的整数")

        return True

# =============================================================================
# 工厂函数
# =============================================================================


def get_config_validator(config_type: str) -> ConfigValidator:
    """
    获取配置验证器工厂函数

    Args:
        config_type: 配置类型 ('task', 'process', 'monitor', 'alert', 'resource', 'optimization', 'api')

    Returns:
        ConfigValidator: 对应的配置验证器实例
    """
    validators = {
        'task': TaskConfigValidator,
        'process': ProcessConfigValidator,
        'monitor': MonitorConfigValidator,
        'alert': AlertConfigValidator,
        'resource': ResourceConfigValidator,
        'optimization': OptimizationConfigValidator,
        'api': APIConfigValidator,
    }

    validator_class = validators.get(config_type.lower())
    if validator_class:
        return validator_class()
    else:
        raise ValueError(f"不支持的配置类型: {config_type}")


def validate_config_safely(config: Any, config_type: str) -> tuple[bool, list[str]]:
    """
    安全验证配置

    Args:
        config: 配置对象
        config_type: 配置类型

    Returns:
        tuple: (是否验证通过, 错误信息列表)
    """
    try:
        validator = get_config_validator(config_type)
        is_valid = validator.validate_config(config)
        errors = validator.get_validation_errors()
        return is_valid, errors
    except Exception as e:
        return False, [f"验证过程中发生错误: {e}"]

# =============================================================================
# 便捷验证函数
# =============================================================================


def validate_task_config(config: TaskConfig) -> tuple[bool, list[str]]:
    """验证任务配置"""
    return validate_config_safely(config, 'task')


def validate_process_config(config: ProcessConfig) -> tuple[bool, list[str]]:
    """验证进程配置"""
    return validate_config_safely(config, 'process')


def validate_monitor_config(config: MonitorConfig) -> tuple[bool, list[str]]:
    """验证监控配置"""
    return validate_config_safely(config, 'monitor')


def validate_alert_config(config: AlertConfig) -> tuple[bool, list[str]]:
    """验证告警配置"""
    return validate_config_safely(config, 'alert')


def validate_resource_config(config: ResourceConfig) -> tuple[bool, list[str]]:
    """验证资源配置"""
    return validate_config_safely(config, 'resource')


def validate_optimization_config(config: OptimizationConfig) -> tuple[bool, list[str]]:
    """验证优化配置"""
    return validate_config_safely(config, 'optimization')


def validate_api_config(config: APIConfig) -> tuple[bool, list[str]]:
    """验证API配置"""
    return validate_config_safely(config, 'api')


__all__ = [
    # 验证器类
    'TaskConfigValidator', 'ProcessConfigValidator', 'MonitorConfigValidator',
    'AlertConfigValidator', 'ResourceConfigValidator', 'OptimizationConfigValidator', 'APIConfigValidator',

    # 工厂函数
    'get_config_validator', 'validate_config_safely',

    # 便捷函数
    'validate_task_config', 'validate_process_config', 'validate_monitor_config',
    'validate_alert_config', 'validate_resource_config', 'validate_optimization_config', 'validate_api_config',
]
