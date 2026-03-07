#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
生产环境配置验证器
用于验证生产环境配置的完整性和正确性
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class ValidationLevel(Enum):
    """验证级别"""
    CRITICAL = "critical"      # 严重错误，必须修复
    ERROR = "error"            # 错误，需要修复
    WARNING = "warning"        # 警告，建议修复
    INFO = "info"              # 信息，仅供参考


@dataclass
class ValidationResult:
    """验证结果"""
    level: ValidationLevel
    message: str
    file_path: str
    line_number: Optional[int] = None
    field_path: Optional[str] = None
    suggestion: Optional[str] = None


class ProductionConfigValidator:
    """生产环境配置验证器"""

    def __init__(self, config_dir: str = "config/production"):
        self.config_dir = Path(config_dir)
        self.logger = self._setup_logging()
        self.validation_results: List[ValidationResult] = []

        # 必需配置文件
        self.required_files = [
            "config.yaml",
            "database.yaml",
            "monitoring.yaml"
        ]

        # 必需环境变量
        self.required_env_vars = [
            "DB_HOST",
            "DB_PASSWORD",
            "REDIS_PASSWORD",
            "SMTP_PASSWORD"
        ]

        # 配置验证规则
        self.validation_rules = {
            "config.yaml": self._validate_main_config,
            "database.yaml": self._validate_database_config,
            "monitoring.yaml": self._validate_monitoring_config
        }

    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def validate_all(self) -> bool:
        """验证所有配置"""
        self.logger.info("开始生产环境配置验证...")

        # 验证配置文件存在性
        self._validate_file_existence()

        # 验证环境变量
        self._validate_environment_variables()

        # 验证配置文件内容
        self._validate_config_contents()

        # 验证配置一致性
        self._validate_config_consistency()

        # 验证安全配置
        self._validate_security_config()

        # 输出验证结果
        self._print_validation_results()

        # 返回验证结果
        critical_errors = any(r.level == ValidationLevel.CRITICAL for r in self.validation_results)
        errors = any(r.level in [ValidationLevel.CRITICAL, ValidationLevel.ERROR]
                     for r in self.validation_results)

        if critical_errors:
            self.logger.error("发现严重错误，配置验证失败！")
            return False
        elif errors:
            self.logger.warning("发现配置错误，需要修复后重新验证")
            return False
        else:
            self.logger.info("配置验证通过！")
            return True

    def _validate_file_existence(self):
        """验证必需配置文件存在性"""
        for file_name in self.required_files:
            file_path = self.config_dir / file_name
            if not file_path.exists():
                self.validation_results.append(ValidationResult(
                    level=ValidationLevel.CRITICAL,
                    message=f"必需配置文件不存在: {file_name}",
                    file_path=str(file_path)
                ))
            else:
                self.logger.info(f"配置文件存在: {file_name}")

    def _validate_environment_variables(self):
        """验证必需环境变量"""
        for env_var in self.required_env_vars:
            if not os.getenv(env_var):
                self.validation_results.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    message=f"必需环境变量未设置: {env_var}",
                    file_path="environment",
                    suggestion=f"请设置环境变量: export {env_var}=<value>"
                ))
            else:
                self.logger.info(f"环境变量已设置: {env_var}")

    def _validate_config_contents(self):
        """验证配置文件内容"""
        for file_name, validator_func in self.validation_rules.items():
            file_path = self.config_dir / file_name
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        config_data = yaml.safe_load(f)

                    if config_data:
                        validator_func(config_data, str(file_path))
                    else:
                        self.validation_results.append(ValidationResult(
                            level=ValidationLevel.ERROR,
                            message=f"配置文件为空或格式错误: {file_name}",
                            file_path=str(file_path)
                        ))
                except Exception as e:
                    self.validation_results.append(ValidationResult(
                        level=ValidationLevel.ERROR,
                        message=f"配置文件解析失败: {file_name}, 错误: {str(e)}",
                        file_path=str(file_path)
                    ))

    def _validate_main_config(self, config: Dict[str, Any], file_path: str):
        """验证主配置文件"""
        # 验证应用配置
        if 'app' not in config:
            self.validation_results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                message="缺少应用配置节",
                file_path=file_path,
                field_path="app"
            ))
        else:
            app_config = config['app']
            if app_config.get('environment') != 'production':
                self.validation_results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    message="应用环境应设置为production",
                    file_path=file_path,
                    field_path="app.environment",
                    suggestion="将environment设置为production"
                ))

            if app_config.get('debug', False):
                self.validation_results.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    message="生产环境不应启用debug模式",
                    file_path=file_path,
                    field_path="app.debug",
                    suggestion="将debug设置为false"
                ))

        # 验证服务器配置
        if 'server' in config:
            server_config = config['server']
            if server_config.get('host') == '0.0.0.0':
                self.validation_results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    message="生产环境建议限制服务器绑定地址",
                    file_path=file_path,
                    field_path="server.host",
                    suggestion="考虑使用具体的IP地址或localhost"
                ))

        # 验证安全配置
        if 'security' in config:
            security_config = config['security']
            if not security_config.get('ssl', {}).get('enabled', False):
                self.validation_results.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    message="生产环境必须启用SSL",
                    file_path=file_path,
                    field_path="security.ssl.enabled",
                    suggestion="将ssl.enabled设置为true"
                ))

    def _validate_database_config(self, config: Dict[str, Any], file_path: str):
        """验证数据库配置文件"""
        if 'database' not in config:
            self.validation_results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                message="缺少数据库配置节",
                file_path=file_path,
                field_path="database"
            ))
            return

        db_config = config['database']

        # 验证连接池配置
        if db_config.get('pool_size', 0) > 50:
            self.validation_results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                message="数据库连接池大小过大",
                file_path=file_path,
                field_path="database.pool_size",
                suggestion="建议将pool_size设置为20-30"
            ))

        # 验证超时配置
        if db_config.get('pool_timeout', 0) < 10:
            self.validation_results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                message="数据库连接池超时时间过短",
                file_path=file_path,
                field_path="database.pool_timeout",
                suggestion="建议将pool_timeout设置为30秒以上"
            ))

    def _validate_monitoring_config(self, config: Dict[str, Any], file_path: str):
        """验证监控配置文件"""
        if 'monitoring' not in config:
            self.validation_results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                message="缺少监控配置节",
                file_path=file_path,
                field_path="monitoring"
            ))
            return

        monitoring_config = config['monitoring']

        # 验证监控启用状态
        if not monitoring_config.get('enabled', False):
            self.validation_results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                message="生产环境必须启用监控",
                file_path=file_path,
                field_path="monitoring.enabled",
                suggestion="将monitoring.enabled设置为true"
            ))

        # 验证告警配置
        if monitoring_config.get('alerting', {}).get('enabled', False):
            alerting_config = monitoring_config['alerting']
            if not alerting_config.get('webhook_url') and not alerting_config.get('email'):
                self.validation_results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    message="告警已启用但未配置通知方式",
                    file_path=file_path,
                    field_path="monitoring.alerting",
                    suggestion="配置webhook_url或email通知"
                ))

    def _validate_config_consistency(self):
        """验证配置一致性"""
        # 检查端口冲突
        used_ports = set()
        for file_name in self.required_files:
            file_path = self.config_dir / file_name
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        config_data = yaml.safe_load(f)

                    # 检查服务器端口
                    if 'server' in config_data and 'port' in config_data['server']:
                        port = config_data['server']['port']
                        if port in used_ports:
                            self.validation_results.append(ValidationResult(
                                level=ValidationLevel.ERROR,
                                message=f"端口冲突: {port}",
                                file_path=str(file_path),
                                field_path="server.port"
                            ))
                        else:
                            used_ports.add(port)

                    # 检查监控端口
                    if 'monitoring' in config_data and 'prometheus' in config_data['monitoring']:
                        prometheus_port = config_data['monitoring']['prometheus'].get('port')
                        if prometheus_port and prometheus_port in used_ports:
                            self.validation_results.append(ValidationResult(
                                level=ValidationLevel.ERROR,
                                message=f"端口冲突: {prometheus_port}",
                                file_path=str(file_path),
                                field_path="monitoring.prometheus.port"
                            ))
                        elif prometheus_port:
                            used_ports.add(prometheus_port)
                except Exception:
                    continue

    def _validate_security_config(self):
        """验证安全配置"""
        # 检查敏感信息
        for file_name in self.required_files:
            file_path = self.config_dir / file_name
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # 检查硬编码的密码
                    if 'password: "123456"' in content or 'password: "admin"' in content:
                        self.validation_results.append(ValidationResult(
                            level=ValidationLevel.CRITICAL,
                            message=f"发现硬编码的默认密码: {file_name}",
                            file_path=str(file_path),
                            suggestion="使用环境变量或安全的密码管理"
                        ))

                    # 检查敏感配置
                    if 'secret_key: "default"' in content:
                        self.validation_results.append(ValidationResult(
                            level=ValidationLevel.CRITICAL,
                            message=f"发现默认密钥: {file_name}",
                            file_path=str(file_path),
                            suggestion="生成安全的密钥并存储在环境变量中"
                        ))
                except Exception:
                    continue

    def _print_validation_results(self):
        """输出验证结果"""
        if not self.validation_results:
            self.logger.info("所有配置验证通过！")
            return

        # 按级别分组
        critical_results = [
            r for r in self.validation_results if r.level == ValidationLevel.CRITICAL]
        error_results = [r for r in self.validation_results if r.level == ValidationLevel.ERROR]
        warning_results = [r for r in self.validation_results if r.level == ValidationLevel.WARNING]
        info_results = [r for r in self.validation_results if r.level == ValidationLevel.INFO]

        # 输出严重错误
        if critical_results:
            self.logger.error(f"\n=== 严重错误 ({len(critical_results)}个) ===")
            for result in critical_results:
                self.logger.error(f"❌ {result.message}")
                if result.file_path != "environment":
                    self.logger.error(f"   文件: {result.file_path}")
                if result.field_path:
                    self.logger.error(f"   字段: {result.field_path}")
                if result.suggestion:
                    self.logger.error(f"   建议: {result.suggestion}")

        # 输出错误
        if error_results:
            self.logger.error(f"\n=== 错误 ({len(error_results)}个) ===")
            for result in error_results:
                self.logger.error(f"❌ {result.message}")
                if result.file_path != "environment":
                    self.logger.error(f"   文件: {result.file_path}")
                if result.field_path:
                    self.logger.error(f"   字段: {result.field_path}")
                if result.suggestion:
                    self.logger.error(f"   建议: {result.suggestion}")

        # 输出警告
        if warning_results:
            self.logger.warning(f"\n=== 警告 ({len(warning_results)}个) ===")
            for result in warning_results:
                self.logger.warning(f"⚠️  {result.message}")
                if result.file_path != "environment":
                    self.logger.warning(f"   文件: {result.file_path}")
                if result.field_path:
                    self.logger.warning(f"   字段: {result.field_path}")
                if result.suggestion:
                    self.logger.warning(f"   建议: {result.suggestion}")

        # 输出信息
        if info_results:
            self.logger.info(f"\n=== 信息 ({len(info_results)}个) ===")
            for result in info_results:
                self.logger.info(f"ℹ️  {result.message}")

        # 输出统计
        total = len(self.validation_results)
        self.logger.info(f"\n=== 验证统计 ===")
        self.logger.info(f"总计: {total}")
        self.logger.info(f"严重错误: {len(critical_results)}")
        self.logger.info(f"错误: {len(error_results)}")
        self.logger.info(f"警告: {len(warning_results)}")
        self.logger.info(f"信息: {len(info_results)}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="生产环境配置验证器")
    parser.add_argument(
        "--config-dir",
        default="config/production",
        help="配置文件目录路径"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细输出"
    )

    args = parser.parse_args()

    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 创建验证器并执行验证
    validator = ProductionConfigValidator(args.config_dir)
    success = validator.validate_all()

    # 设置退出码
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
