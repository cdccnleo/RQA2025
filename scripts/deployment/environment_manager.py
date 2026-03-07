#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
环境变量管理工具
用于管理生产环境的环境变量配置
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import secrets
import string

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class EnvironmentType(Enum):
    """环境类型"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class EnvironmentVariable:
    """环境变量定义"""
    name: str
    value: Optional[str] = None
    description: str = ""
    required: bool = False
    sensitive: bool = False
    default_value: Optional[str] = None
    validation_regex: Optional[str] = None
    example: Optional[str] = None


class EnvironmentManager:
    """环境变量管理器"""

    def __init__(self, environment: EnvironmentType = EnvironmentType.PRODUCTION):
        self.environment = environment
        self.logger = self._setup_logging()
        self.config_dir = Path("config") / environment.value
        self.env_file = Path(f".env.{environment.value}")

        # 定义必需的环境变量
        self.required_variables = self._define_required_variables()

        # 定义可选的环境变量
        self.optional_variables = self._define_optional_variables()

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

    def _define_required_variables(self) -> List[EnvironmentVariable]:
        """定义必需的环境变量"""
        return [
            EnvironmentVariable(
                name="DB_HOST",
                description="数据库主机地址",
                required=True,
                example="localhost"
            ),
            EnvironmentVariable(
                name="DB_PORT",
                description="数据库端口",
                required=True,
                default_value="5432",
                example="5432"
            ),
            EnvironmentVariable(
                name="DB_NAME",
                description="数据库名称",
                required=True,
                example="rqa2025_prod"
            ),
            EnvironmentVariable(
                name="DB_USERNAME",
                description="数据库用户名",
                required=True,
                example="rqa_user"
            ),
            EnvironmentVariable(
                name="DB_PASSWORD",
                description="数据库密码",
                required=True,
                sensitive=True,
                example="your_secure_password"
            ),
            EnvironmentVariable(
                name="REDIS_HOST",
                description="Redis主机地址",
                required=True,
                default_value="localhost",
                example="localhost"
            ),
            EnvironmentVariable(
                name="REDIS_PORT",
                description="Redis端口",
                required=True,
                default_value="6379",
                example="6379"
            ),
            EnvironmentVariable(
                name="REDIS_PASSWORD",
                description="Redis密码",
                required=True,
                sensitive=True,
                example="your_redis_password"
            ),
            EnvironmentVariable(
                name="JWT_SECRET_KEY",
                description="JWT密钥",
                required=True,
                sensitive=True,
                example="your_jwt_secret_key"
            ),
            EnvironmentVariable(
                name="ENCRYPTION_KEY",
                description="加密密钥",
                required=True,
                sensitive=True,
                example="your_encryption_key"
            )
        ]

    def _define_optional_variables(self) -> List[EnvironmentVariable]:
        """定义可选的环境变量"""
        return [
            EnvironmentVariable(
                name="LOG_LEVEL",
                description="日志级别",
                required=False,
                default_value="INFO",
                example="INFO"
            ),
            EnvironmentVariable(
                name="METRICS_ENABLED",
                description="是否启用指标收集",
                required=False,
                default_value="true",
                example="true"
            ),
            EnvironmentVariable(
                name="SMTP_HOST",
                description="SMTP服务器地址",
                required=False,
                example="smtp.gmail.com"
            ),
            EnvironmentVariable(
                name="SMTP_PORT",
                description="SMTP端口",
                required=False,
                default_value="587",
                example="587"
            ),
            EnvironmentVariable(
                name="SMTP_USERNAME",
                description="SMTP用户名",
                required=False,
                example="your_email@gmail.com"
            ),
            EnvironmentVariable(
                name="SMTP_PASSWORD",
                description="SMTP密码",
                required=False,
                sensitive=True,
                example="your_smtp_password"
            ),
            EnvironmentVariable(
                name="ALERT_WEBHOOK_URL",
                description="告警Webhook地址",
                required=False,
                example="https://hooks.slack.com/services/xxx/yyy/zzz"
            ),
            EnvironmentVariable(
                name="GRAFANA_URL",
                description="Grafana地址",
                required=False,
                default_value="http://localhost:3000",
                example="http://localhost:3000"
            ),
            EnvironmentVariable(
                name="PROMETHEUS_PORT",
                description="Prometheus端口",
                required=False,
                default_value="9090",
                example="9090"
            ),
            EnvironmentVariable(
                name="APP_PORT",
                description="应用端口",
                required=False,
                default_value="8080",
                example="8080"
            )
        ]

    def generate_environment_file(self, force: bool = False) -> bool:
        """生成环境变量文件"""
        if self.env_file.exists() and not force:
            self.logger.warning(f"环境文件已存在: {self.env_file}")
            response = input("是否覆盖现有文件? (y/N): ")
            if response.lower() != 'y':
                return False

        try:
            # 创建环境文件内容
            content = self._generate_env_content()

            # 写入文件
            with open(self.env_file, 'w', encoding='utf-8') as f:
                f.write(content)

            self.logger.info(f"环境文件已生成: {self.env_file}")
            return True

        except Exception as e:
            self.logger.error(f"生成环境文件失败: {str(e)}")
            return False

    def _generate_env_content(self) -> str:
        """生成环境文件内容"""
        lines = [
            f"# RQA2025 {self.environment.value.upper()} 环境配置",
            f"# 生成时间: {self._get_current_timestamp()}",
            f"# 环境类型: {self.environment.value}",
            "",
            "# ========================================",
            "# 必需环境变量",
            "# ========================================",
            ""
        ]

        # 添加必需变量
        for var in self.required_variables:
            lines.append(f"# {var.description}")
            if var.example:
                lines.append(f"# 示例: {var.example}")
            if var.sensitive:
                lines.append(f"# 注意: 这是敏感信息，请妥善保管")

            if var.default_value:
                lines.append(f"{var.name}={var.default_value}")
            else:
                lines.append(f"{var.name}=")
            lines.append("")

        # 添加可选变量
        lines.extend([
            "# ========================================",
            "# 可选环境变量",
            "# ========================================",
            ""
        ])

        for var in self.optional_variables:
            lines.append(f"# {var.description}")
            if var.example:
                lines.append(f"# 示例: {var.example}")
            if var.default_value:
                lines.append(f"# 默认值: {var.default_value}")

            if var.default_value:
                lines.append(f"{var.name}={var.default_value}")
            else:
                lines.append(f"{var.name}=")
            lines.append("")

        # 添加说明
        lines.extend([
            "# ========================================",
            "# 使用说明",
            "# ========================================",
            "# 1. 复制此文件为 .env",
            "# 2. 填写必需的环境变量值",
            "# 3. 根据需要修改可选的环境变量值",
            "# 4. 确保敏感信息的安全性",
            "# 5. 在生产环境中使用环境变量或密钥管理服务",
            ""
        ])

        return "\n".join(lines)

    def validate_environment(self) -> Dict[str, Any]:
        """验证环境变量配置"""
        self.logger.info("开始验证环境变量配置...")

        results = {
            "valid": True,
            "missing_required": [],
            "missing_optional": [],
            "validation_errors": [],
            "warnings": []
        }

        # 检查必需变量
        for var in self.required_variables:
            value = os.getenv(var.name)
            if not value:
                results["missing_required"].append(var.name)
                results["valid"] = False
            else:
                # 验证值格式
                if var.validation_regex:
                    import re
                    if not re.match(var.validation_regex, value):
                        results["validation_errors"].append({
                            "variable": var.name,
                            "error": f"值格式不符合要求: {var.validation_regex}"
                        })
                        results["valid"] = False

        # 检查可选变量
        for var in self.optional_variables:
            value = os.getenv(var.name)
            if not value and var.default_value:
                results["missing_optional"].append(var.name)
                results["warnings"].append(f"可选变量 {var.name} 未设置，将使用默认值: {var.default_value}")

        # 输出验证结果
        self._print_validation_results(results)

        return results

    def _print_validation_results(self, results: Dict[str, Any]):
        """输出验证结果"""
        if results["valid"]:
            self.logger.info("✅ 环境变量验证通过！")
        else:
            self.logger.error("❌ 环境变量验证失败！")

        # 输出缺失的必需变量
        if results["missing_required"]:
            self.logger.error(f"\n=== 缺失的必需环境变量 ({len(results['missing_required'])}个) ===")
            for var_name in results["missing_required"]:
                self.logger.error(f"❌ {var_name}")

        # 输出缺失的可选变量
        if results["missing_optional"]:
            self.logger.warning(f"\n=== 缺失的可选环境变量 ({len(results['missing_optional'])}个) ===")
            for var_name in results["missing_optional"]:
                self.logger.warning(f"⚠️  {var_name}")

        # 输出验证错误
        if results["validation_errors"]:
            self.logger.error(f"\n=== 验证错误 ({len(results['validation_errors'])}个) ===")
            for error in results["validation_errors"]:
                self.logger.error(f"❌ {error['variable']}: {error['error']}")

        # 输出警告
        if results["warnings"]:
            self.logger.warning(f"\n=== 警告 ({len(results['warnings'])}个) ===")
            for warning in results["warnings"]:
                self.logger.warning(f"⚠️  {warning}")

    def generate_secrets(self) -> Dict[str, str]:
        """生成安全的密钥"""
        self.logger.info("生成安全的密钥...")

        secrets_dict = {}

        # 生成JWT密钥
        jwt_secret = ''.join(secrets.choice(string.ascii_letters + string.digits)
                             for _ in range(64))
        secrets_dict["JWT_SECRET_KEY"] = jwt_secret

        # 生成加密密钥
        encryption_key = ''.join(secrets.choice(
            string.ascii_letters + string.digits) for _ in range(32))
        secrets_dict["ENCRYPTION_KEY"] = encryption_key

        # 生成数据库密码
        db_password = ''.join(secrets.choice(string.ascii_letters + string.digits)
                              for _ in range(16))
        secrets_dict["DB_PASSWORD"] = db_password

        # 生成Redis密码
        redis_password = ''.join(secrets.choice(
            string.ascii_letters + string.digits) for _ in range(16))
        secrets_dict["REDIS_PASSWORD"] = redis_password

        self.logger.info("密钥生成完成！")
        return secrets_dict

    def update_environment_file(self, updates: Dict[str, str]) -> bool:
        """更新环境文件"""
        if not self.env_file.exists():
            self.logger.error(f"环境文件不存在: {self.env_file}")
            return False

        try:
            # 读取现有内容
            with open(self.env_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # 更新变量值
            updated_lines = []
            updated_vars = set()

            for line in lines:
                if line.strip() and not line.startswith('#') and '=' in line:
                    var_name = line.split('=')[0].strip()
                    if var_name in updates:
                        updated_lines.append(f"{var_name}={updates[var_name]}\n")
                        updated_vars.add(var_name)
                    else:
                        updated_lines.append(line)
                else:
                    updated_lines.append(line)

            # 添加新的变量
            for var_name, value in updates.items():
                if var_name not in updated_vars:
                    updated_lines.append(f"{var_name}={value}\n")

            # 写回文件
            with open(self.env_file, 'w', encoding='utf-8') as f:
                f.writelines(updated_lines)

            self.logger.info(f"环境文件已更新: {self.env_file}")
            return True

        except Exception as e:
            self.logger.error(f"更新环境文件失败: {str(e)}")
            return False

    def export_to_shell(self) -> bool:
        """导出环境变量到当前shell"""
        if not self.env_file.exists():
            self.logger.error(f"环境文件不存在: {self.env_file}")
            return False

        try:
            # 读取环境文件
            with open(self.env_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 解析环境变量
            env_vars = {}
            for line in content.split('\n'):
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    if value:  # 只处理有值的变量
                        env_vars[key.strip()] = value.strip()

            # 导出到当前shell
            for key, value in env_vars.items():
                os.environ[key] = value
                self.logger.info(f"已导出环境变量: {key}")

            self.logger.info(f"成功导出 {len(env_vars)} 个环境变量")
            return True

        except Exception as e:
            self.logger.error(f"导出环境变量失败: {str(e)}")
            return False

    def _get_current_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def create_docker_env_file(self) -> bool:
        """创建Docker环境文件"""
        docker_env_file = Path(f".env.docker.{self.environment.value}")

        try:
            # 读取环境文件
            if not self.env_file.exists():
                self.logger.error(f"环境文件不存在: {self.env_file}")
                return False

            with open(self.env_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 转换为Docker格式
            docker_content = []
            for line in content.split('\n'):
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    if value:  # 只包含有值的变量
                        docker_content.append(f"{key.strip()}={value.strip()}")

            # 写入Docker环境文件
            with open(docker_env_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(docker_content))

            self.logger.info(f"Docker环境文件已创建: {docker_env_file}")
            return True

        except Exception as e:
            self.logger.error(f"创建Docker环境文件失败: {str(e)}")
            return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="环境变量管理工具")
    parser.add_argument(
        "--environment", "-e",
        choices=["development", "staging", "production"],
        default="production",
        help="目标环境"
    )
    parser.add_argument(
        "--action", "-a",
        choices=["generate", "validate", "update", "export", "docker", "secrets"],
        default="generate",
        help="执行的操作"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="强制覆盖现有文件"
    )
    parser.add_argument(
        "--updates",
        nargs='+',
        help="要更新的环境变量 (格式: KEY=VALUE)"
    )

    args = parser.parse_args()

    # 创建环境管理器
    env_type = EnvironmentType(args.environment)
    manager = EnvironmentManager(env_type)

    # 执行相应操作
    if args.action == "generate":
        success = manager.generate_environment_file(args.force)
        if success:
            print(f"\n✅ 环境文件已生成: {manager.env_file}")
            print("请编辑文件并填写必需的环境变量值")
        else:
            print("❌ 环境文件生成失败")
            sys.exit(1)

    elif args.action == "validate":
        results = manager.validate_environment()
        if not results["valid"]:
            sys.exit(1)

    elif args.action == "update":
        if not args.updates:
            print("❌ 请提供要更新的环境变量")
            sys.exit(1)

        updates = {}
        for update in args.updates:
            if '=' in update:
                key, value = update.split('=', 1)
                updates[key.strip()] = value.strip()

        success = manager.update_environment_file(updates)
        if not success:
            sys.exit(1)

    elif args.action == "export":
        success = manager.export_to_shell()
        if not success:
            sys.exit(1)

    elif args.action == "docker":
        success = manager.create_docker_env_file()
        if not success:
            sys.exit(1)

    elif args.action == "secrets":
        secrets = manager.generate_secrets()
        print("\n=== 生成的密钥 ===")
        for key, value in secrets.items():
            print(f"{key}={value}")
        print("\n⚠️  请妥善保管这些密钥，不要提交到版本控制系统")


if __name__ == "__main__":
    main()
