"""
enhanced_config_validator 模块

提供 enhanced_config_validator 相关功能和接口。
"""


import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Callable
import json
import logging
import threading
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版配置验证器
提供配置项的有效性验证和业务规则校验
"""

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """验证级别"""
    STRICT = "strict"
    NORMAL = "normal"
    LAX = "lax"


@dataclass
class ValidationError:
    """验证错误"""
    field: str
    message: str
    severity: str = "error"
    suggestion: str = ""


class EnhancedConfigValidator:
    """增强版配置验证器"""

    def __init__(self, validation_level: ValidationLevel = ValidationLevel.NORMAL):
        self.validation_level = validation_level
        self.custom_validators: Dict[str, Callable] = {}
        self.validation_stats = {
            'total_validations': 0,
            'passed_validations': 0,
            'failed_validations': 0,
            'errors_by_type': {}
        }

    def validate_database_config(self, config: Dict[str, Any]) -> List[ValidationError]:
        """验证数据库配置"""
        errors = []

        # 必需字段检查
        required_fields = ['host', 'port', 'username', 'database']
        for field in required_fields:
            if field not in config:
                errors.append(ValidationError(
                    field=f"database.{field}",
                    message=f"缺少必需的数据库配置字段: {field}",
                    suggestion=f"请添加 {field} 配置项"
                ))

        # 端口验证
        port = config.get('port')
        if port is not None:
            if not isinstance(port, int) or not (1024 <= port <= 65535):
                errors.append(ValidationError(
                    field="database.port",
                    message=f"数据库端口 {port} 无效，应在1024-65535范围内",
                    suggestion="请使用有效的端口号"
                ))

        # 连接池验证
        pool_config = config.get('connection_pool', {})
        min_conn = pool_config.get('min_connections', 1)
        max_conn = pool_config.get('max_connections', 10)

        if min_conn > max_conn:
            errors.append(ValidationError(
                field="database.connection_pool",
                message="最小连接数不能大于最大连接数",
                suggestion="请调整连接池配置"
            ))

        return errors

    def validate_api_config(self, config: Dict[str, Any]) -> List[ValidationError]:
        """验证API配置"""
        errors = []

        # 基础URL验证
        base_url = config.get('base_url', '')
        if base_url and not base_url.startswith(('http://', 'https://')):
            errors.append(ValidationError(
                field="api.base_url",
                message="API基础URL必须以http://或https://开头",
                suggestion="请提供有效的HTTP/HTTPS URL"
            ))

        # 超时验证
        timeout = config.get('timeout', 30)
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            errors.append(ValidationError(
                field="api.timeout",
                message=f"API超时时间 {timeout} 无效，应为正数",
                suggestion="请设置合理的超时时间（建议5-300秒）"
            ))

        # 重试配置验证
        retry_config = config.get('retry', {})
        max_retries = retry_config.get('max_attempts', 3)
        if max_retries < 0 or max_retries > 10:
            errors.append(ValidationError(
                field="api.retry.max_attempts",
                message=f"最大重试次数 {max_retries} 无效，应在0-10范围内",
                suggestion="请设置合理的重试次数"
            ))

        return errors

    def validate_logging_config(self, config: Dict[str, Any]) -> List[ValidationError]:
        """验证日志配置"""
        errors = []

        # 日志级别验证
        level = config.get('level', 'INFO').upper()
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

        if level not in valid_levels:
            errors.append(ValidationError(
                field="logging.level",
                message=f"日志级别 {level} 无效",
                suggestion=f"请使用以下级别之一: {', '.join(valid_levels)}"
            ))

        # 日志文件路径验证
        log_file = config.get('file', '')
        if log_file:
            try:
                # 检查路径是否可写
                log_dir = os.path.dirname(log_file)
                if log_dir and not os.path.exists(log_dir):
                    errors.append(ValidationError(
                        field="logging.file",
                        message=f"日志目录不存在: {log_dir}",
                        suggestion="请创建日志目录或检查路径权限"
                    ))
            except Exception as e:
                errors.append(ValidationError(
                    field="logging.file",
                    message=f"日志文件路径验证失败: {e}",
                    suggestion="请检查文件路径格式"
                ))

        return errors

    def validate_security_config(self, config: Dict[str, Any]) -> List[ValidationError]:
        """验证安全配置"""
        errors = []

        # 加密配置验证
        encryption = config.get('encryption', {})
        if encryption.get('enabled', False):
            key_file = encryption.get('key_file', '')
            if not key_file:
                errors.append(ValidationError(
                    field="security.encryption.key_file",
                    message="启用加密时必须指定密钥文件",
                    suggestion="请提供加密密钥文件路径"
                ))

        # 访问控制验证
        access_control = config.get('access_control', {})
        if access_control.get('enabled', False):
            user_file = access_control.get('user_file', '')
            if not user_file:
                errors.append(ValidationError(
                    field="security.access_control.user_file",
                    message="启用访问控制时必须指定用户文件",
                    suggestion="请提供用户配置文件路径"
                ))

        return errors

    def validate_monitoring_config(self, config: Dict[str, Any]) -> List[ValidationError]:
        """验证监控配置"""
        errors = []

        # Prometheus配置验证
        prometheus = config.get('prometheus', {})
        if prometheus.get('enabled', False):
            port = prometheus.get('port', 9090)
            if not isinstance(port, int) or not (1024 <= port <= 65535):
                errors.append(ValidationError(
                    field="monitoring.prometheus.port",
                    message=f"Prometheus端口 {port} 无效",
                    suggestion="请使用1024-65535范围内的端口"
                ))

        # 告警配置验证
        alerting = config.get('alerting', {})
        if alerting.get('enabled', False):
            email_config = alerting.get('email', {})
            if email_config.get('enabled', False):
                smtp_server = email_config.get('smtp_server', '')
                if not smtp_server:
                    errors.append(ValidationError(
                        field="monitoring.alerting.email.smtp_server",
                        message="启用邮件告警时必须配置SMTP服务器",
                        suggestion="请提供SMTP服务器地址"
                    ))

        return errors

    def validate_full_config(self, config: Dict[str, Any]) -> Dict[str, List[ValidationError]]:
        """验证完整配置"""
        self.validation_stats['total_validations'] += 1

        results = {
            'database': self.validate_database_config(config.get('database', {})),
            'api': self.validate_api_config(config.get('api', {})),
            'logging': self.validate_logging_config(config.get('logging', {})),
            'security': self.validate_security_config(config.get('security', {})),
            'monitoring': self.validate_monitoring_config(config.get('monitoring', {}))
        }

        # 统计错误
        total_errors = sum(len(errors) for errors in results.values())

        if total_errors == 0:
            self.validation_stats['passed_validations'] += 1
        else:
            self.validation_stats['failed_validations'] += 1

        return results

    def add_custom_validator(self, name: str, validator_func: Callable):
        """添加自定义验证器"""
        self.custom_validators[name] = validator_func

    def get_validation_stats(self) -> Dict[str, Any]:
        """获取验证统计信息"""
        return self.validation_stats.copy()

    def reset_stats(self):
        """重置统计信息"""
        self.validation_stats = {
            'total_validations': 0,
            'passed_validations': 0,
            'failed_validations': 0,
            'errors_by_type': {}
        }


# 全局验证器实例
_global_enhanced_validator = None
_enhanced_validator_lock = threading.Lock()


def get_enhanced_config_validator() -> EnhancedConfigValidator:
    """获取全局增强版配置验证器"""
    global _global_enhanced_validator

    if _global_enhanced_validator is None:
        with _enhanced_validator_lock:
            if _global_enhanced_validator is None:
                _global_enhanced_validator = EnhancedConfigValidator()

    return _global_enhanced_validator


def validate_config_file(config_file: str) -> Dict[str, List[ValidationError]]:
    """验证配置文件"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        validator = get_enhanced_config_validator()
        return validator.validate_full_config(config)

    except Exception as e:
        return {
            'file_error': [ValidationError(
                field="config_file",
                message=f"配置文件读取失败: {e}",
                suggestion="请检查文件是否存在且格式正确"
            )]
        }


if __name__ == "__main__":
    # 测试增强版配置验证器
    print("初始化增强版配置验证器...")

    validator = get_enhanced_config_validator()

    # 测试配置数据
    test_config = {
        "database": {
            "host": "localhost",
            "port": 5432,
            "username": "admin",
            "database": "rqa2025",
            "connection_pool": {
                "min_connections": 5,
                "max_connections": 20
            }
        },
        "api": {
            "base_url": "https://api.example.com",
            "timeout": 30,
            "retry": {
                "max_attempts": 3
            }
        },
        "logging": {
            "level": "INFO",
            "file": "/var/log/rqa2025/app.log"
        },
        "security": {
            "encryption": {
                "enabled": True,
                "key_file": "/etc/ssl/private/key.pem"
            }
        },
        "monitoring": {
            "prometheus": {
                "enabled": True,
                "port": 9090
            },
            "alerting": {
                "enabled": True,
                "email": {
                    "enabled": True,
                    "smtp_server": "smtp.example.com"
                }
            }
        }
    }

    print("验证测试配置...")
    results = validator.validate_full_config(test_config)

    print("\\n验证结果:")
    total_errors = 0
    for section, errors in results.items():
        if errors:
            print(f"\\n{section.upper()} 配置问题 ({len(errors)} 个):")
            for error in errors:
                print(f"  ❌ {error.field}: {error.message}")
                if error.suggestion:
                    print(f"     💡 {error.suggestion}")
                total_errors += 1
        else:
            print(f"✅ {section.upper()} 配置验证通过")

    print(f"\\n总体结果: 发现 {total_errors} 个配置问题")

    # 获取统计信息
    stats = validator.get_validation_stats()
    print(f"\\n验证统计: {stats}")

    print("\\n配置验证测试完成！")




