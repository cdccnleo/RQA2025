#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置部署自动化脚本
支持多环境部署、配置验证、回滚等功能
"""
import os
import json
import yaml
import requests
import time
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DeploymentConfig:
    """部署配置"""
    environment: str
    api_base: str
    username: str
    password: str
    config_file: str
    backup_enabled: bool = True
    validate_before_deploy: bool = True
    rollback_on_failure: bool = True
    timeout: int = 30


@dataclass
class DeploymentResult:
    """部署结果"""
    success: bool
    environment: str
    timestamp: datetime
    config_file: str
    backup_file: Optional[str] = None
    error_message: Optional[str] = None
    validation_errors: List[str] = None
    deployment_time: float = 0.0


class ConfigDeploymentAutomation:
    """配置部署自动化"""

    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.session_id = None
        self.backup_dir = "backups/deployments"
        self.deployment_history = []

    def deploy(self) -> DeploymentResult:
        """执行配置部署"""
        start_time = time.time()
        logger.info(f"开始部署配置到环境: {self.config.environment}")

        try:
            # 1. 登录认证
            self._login()

            # 2. 创建备份
            backup_file = None
            if self.config.backup_enabled:
                backup_file = self._create_backup()

            # 3. 验证配置
            validation_errors = []
            if self.config.validate_before_deploy:
                validation_errors = self._validate_config()
                if validation_errors:
                    logger.error(f"配置验证失败: {validation_errors}")
                    return DeploymentResult(
                        success=False,
                        environment=self.config.environment,
                        timestamp=datetime.now(),
                        config_file=self.config.config_file,
                        backup_file=backup_file,
                        error_message="配置验证失败",
                        validation_errors=validation_errors,
                        deployment_time=time.time() - start_time
                    )

            # 4. 部署配置
            self._deploy_config()

            # 5. 验证部署结果
            if not self._verify_deployment():
                if self.config.rollback_on_failure and backup_file:
                    logger.warning("部署验证失败，开始回滚...")
                    self._rollback(backup_file)
                    return DeploymentResult(
                        success=False,
                        environment=self.config.environment,
                        timestamp=datetime.now(),
                        config_file=self.config.config_file,
                        backup_file=backup_file,
                        error_message="部署验证失败，已回滚",
                        deployment_time=time.time() - start_time
                    )
                else:
                    return DeploymentResult(
                        success=False,
                        environment=self.config.environment,
                        timestamp=datetime.now(),
                        config_file=self.config.config_file,
                        backup_file=backup_file,
                        error_message="部署验证失败",
                        deployment_time=time.time() - start_time
                    )

            # 6. 记录部署历史
            result = DeploymentResult(
                success=True,
                environment=self.config.environment,
                timestamp=datetime.now(),
                config_file=self.config.config_file,
                backup_file=backup_file,
                deployment_time=time.time() - start_time
            )

            self.deployment_history.append(result)
            self._save_deployment_history()

            logger.info(f"✅ 配置部署成功到环境: {self.config.environment}")
            return result

        except Exception as e:
            logger.error(f"❌ 配置部署失败: {e}")
            return DeploymentResult(
                success=False,
                environment=self.config.environment,
                timestamp=datetime.now(),
                config_file=self.config.config_file,
                error_message=str(e),
                deployment_time=time.time() - start_time
            )

    def _login(self):
        """登录认证"""
        try:
            response = requests.post(f"{self.config.api_base}/api/login", json={
                "username": self.config.username,
                "password": self.config.password
            }, timeout=self.config.timeout)

            data = response.json()
            if data.get("success"):
                self.session_id = data["session_id"]
                logger.info("✅ 登录成功")
            else:
                raise Exception(f"登录失败: {data.get('detail', '未知错误')}")

        except Exception as e:
            raise Exception(f"登录失败: {e}")

    def _create_backup(self) -> Optional[str]:
        """创建配置备份"""
        try:
            # 获取当前配置
            response = requests.get(f"{self.config.api_base}/api/config",
                                    headers={"Authorization": f"Bearer {self.session_id}"},
                                    timeout=self.config.timeout)

            if response.status_code != 200:
                logger.warning("无法获取当前配置，跳过备份")
                return None

            data = response.json()
            current_config = data.get("config", {})

            # 创建备份目录
            os.makedirs(self.backup_dir, exist_ok=True)

            # 生成备份文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"backup_{self.config.environment}_{timestamp}.json"
            backup_filepath = os.path.join(self.backup_dir, backup_filename)

            # 保存备份
            with open(backup_filepath, 'w', encoding='utf-8') as f:
                json.dump(current_config, f, ensure_ascii=False, indent=2)

            logger.info(f"✅ 配置备份已创建: {backup_filepath}")
            return backup_filepath

        except Exception as e:
            logger.warning(f"创建备份失败: {e}")
            return None

    def _validate_config(self) -> List[str]:
        """验证配置"""
        errors = []

        try:
            # 读取配置文件
            with open(self.config.config_file, 'r', encoding='utf-8') as f:
                if self.config.config_file.endswith('.json'):
                    config_data = json.load(f)
                elif self.config.config_file.endswith('.yaml') or self.config.config_file.endswith('.yml'):
                    config_data = yaml.safe_load(f)
                else:
                    errors.append("不支持的文件格式，仅支持 JSON 和 YAML")
                    return errors

            # 发送到验证API
            response = requests.post(f"{self.config.api_base}/api/config/validate",
                                     headers={"Authorization": f"Bearer {self.session_id}"},
                                     json={"config": config_data},
                                     timeout=self.config.timeout)

            data = response.json()
            if not data.get("success"):
                validation_errors = data.get("errors", [])
                errors.extend(validation_errors)

            # 基本验证
            if not isinstance(config_data, dict):
                errors.append("配置必须是对象格式")

            # 检查必需字段
            required_fields = ["database", "trading", "risk_control"]
            for field in required_fields:
                if field not in config_data:
                    errors.append(f"缺少必需字段: {field}")

            logger.info(f"✅ 配置验证完成，发现 {len(errors)} 个错误")
            return errors

        except Exception as e:
            errors.append(f"配置验证失败: {e}")
            return errors

    def _deploy_config(self):
        """部署配置"""
        try:
            # 读取配置文件
            with open(self.config.config_file, 'r', encoding='utf-8') as f:
                if self.config.config_file.endswith('.json'):
                    config_data = json.load(f)
                elif self.config.config_file.endswith('.yaml') or self.config.config_file.endswith('.yml'):
                    config_data = yaml.safe_load(f)
                else:
                    raise Exception("不支持的文件格式")

            # 批量更新配置
            response = requests.put(f"{self.config.api_base}/api/config/batch",
                                    headers={"Authorization": f"Bearer {self.session_id}"},
                                    json={"config": config_data},
                                    timeout=self.config.timeout)

            data = response.json()
            if not data.get("success"):
                raise Exception(f"配置部署失败: {data.get('message', '未知错误')}")

            logger.info("✅ 配置部署完成")

        except Exception as e:
            raise Exception(f"配置部署失败: {e}")

    def _verify_deployment(self) -> bool:
        """验证部署结果"""
        try:
            # 等待一段时间让配置生效
            time.sleep(2)

            # 获取部署后的配置
            response = requests.get(f"{self.config.api_base}/api/config",
                                    headers={"Authorization": f"Bearer {self.session_id}"},
                                    timeout=self.config.timeout)

            if response.status_code != 200:
                logger.error("无法获取部署后的配置")
                return False

            data = response.json()
            deployed_config = data.get("config", {})

            # 读取原始配置文件
            with open(self.config.config_file, 'r', encoding='utf-8') as f:
                if self.config.config_file.endswith('.json'):
                    original_config = json.load(f)
                elif self.config.config_file.endswith('.yaml') or self.config.config_file.endswith('.yml'):
                    original_config = yaml.safe_load(f)
                else:
                    return False

            # 比较关键配置项
            key_fields = ["database", "trading", "risk_control"]
            for field in key_fields:
                if field in original_config and field in deployed_config:
                    if original_config[field] != deployed_config[field]:
                        logger.error(f"配置验证失败: {field} 字段不匹配")
                        return False

            logger.info("✅ 部署验证通过")
            return True

        except Exception as e:
            logger.error(f"部署验证失败: {e}")
            return False

    def _rollback(self, backup_file: str):
        """回滚配置"""
        try:
            if not backup_file or not os.path.exists(backup_file):
                logger.error("备份文件不存在，无法回滚")
                return False

            # 读取备份配置
            with open(backup_file, 'r', encoding='utf-8') as f:
                backup_config = json.load(f)

            # 恢复配置
            response = requests.put(f"{self.config.api_base}/api/config/batch",
                                    headers={"Authorization": f"Bearer {self.session_id}"},
                                    json={"config": backup_config},
                                    timeout=self.config.timeout)

            data = response.json()
            if data.get("success"):
                logger.info("✅ 配置回滚成功")
                return True
            else:
                logger.error(f"配置回滚失败: {data.get('message', '未知错误')}")
                return False

        except Exception as e:
            logger.error(f"配置回滚失败: {e}")
            return False

    def _save_deployment_history(self):
        """保存部署历史"""
        try:
            history_file = os.path.join(self.backup_dir, "deployment_history.json")

            # 读取现有历史
            history = []
            if os.path.exists(history_file):
                with open(history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)

            # 添加新的部署记录
            history.append({
                "environment": self.config.environment,
                "timestamp": self.deployment_history[-1].timestamp.isoformat(),
                "config_file": self.config.config_file,
                "backup_file": self.deployment_history[-1].backup_file,
                "success": self.deployment_history[-1].success,
                "deployment_time": self.deployment_history[-1].deployment_time,
                "error_message": self.deployment_history[-1].error_message
            })

            # 保存历史记录
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)

            logger.info(f"✅ 部署历史已保存: {history_file}")

        except Exception as e:
            logger.error(f"保存部署历史失败: {e}")

    def get_deployment_status(self) -> Dict:
        """获取部署状态"""
        try:
            response = requests.get(f"{self.config.api_base}/api/health",
                                    timeout=self.config.timeout)

            if response.status_code == 200:
                return {"status": "healthy", "environment": self.config.environment}
            else:
                return {"status": "unhealthy", "environment": self.config.environment}

        except Exception as e:
            return {"status": "error", "environment": self.config.environment, "error": str(e)}


class EnvironmentManager:
    """环境管理器"""

    def __init__(self):
        self.environments = {
            "development": {
                "api_base": "http://localhost:8080",
                "username": "dev_user",
                "password": "dev_pass"
            },
            "staging": {
                "api_base": "http://staging-config.example.com:8080",
                "username": "staging_user",
                "password": "staging_pass"
            },
            "production": {
                "api_base": "http://prod-config.example.com:8080",
                "username": "prod_user",
                "password": "prod_pass"
            }
        }

    def get_environment_config(self, environment: str) -> Dict:
        """获取环境配置"""
        if environment not in self.environments:
            raise ValueError(f"未知环境: {environment}")

        return self.environments[environment]

    def list_environments(self) -> List[str]:
        """列出所有环境"""
        return list(self.environments.keys())

    def validate_environment(self, environment: str) -> bool:
        """验证环境连接"""
        try:
            config = self.get_environment_config(environment)
            response = requests.get(f"{config['api_base']}/api/health", timeout=10)
            return response.status_code == 200
        except Exception:
            return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="配置部署自动化工具")
    parser.add_argument('--environment', '-e', required=True, help='目标环境')
    parser.add_argument('--config-file', '-c', required=True, help='配置文件路径')
    parser.add_argument('--username', '-u', help='用户名')
    parser.add_argument('--password', '-p', help='密码')
    parser.add_argument('--api-base', help='API服务地址')
    parser.add_argument('--no-backup', action='store_true', help='禁用备份')
    parser.add_argument('--no-validate', action='store_true', help='禁用验证')
    parser.add_argument('--no-rollback', action='store_true', help='禁用回滚')
    parser.add_argument('--timeout', type=int, default=30, help='超时时间(秒)')

    args = parser.parse_args()

    # 环境管理器
    env_manager = EnvironmentManager()

    try:
        # 获取环境配置
        env_config = env_manager.get_environment_config(args.environment)

        # 合并命令行参数
        if args.username:
            env_config['username'] = args.username
        if args.password:
            env_config['password'] = args.password
        if args.api_base:
            env_config['api_base'] = args.api_base

        # 创建部署配置
        deployment_config = DeploymentConfig(
            environment=args.environment,
            api_base=env_config['api_base'],
            username=env_config['username'],
            password=env_config['password'],
            config_file=args.config_file,
            backup_enabled=not args.no_backup,
            validate_before_deploy=not args.no_validate,
            rollback_on_failure=not args.no_rollback,
            timeout=args.timeout
        )

        # 验证环境连接
        if not env_manager.validate_environment(args.environment):
            logger.error(f"❌ 无法连接到环境: {args.environment}")
            return

        # 执行部署
        automation = ConfigDeploymentAutomation(deployment_config)
        result = automation.deploy()

        # 输出结果
        if result.success:
            print(f"\n✅ 配置部署成功!")
            print(f"   环境: {result.environment}")
            print(f"   配置文件: {result.config_file}")
            print(f"   部署时间: {result.deployment_time:.2f}秒")
            if result.backup_file:
                print(f"   备份文件: {result.backup_file}")
        else:
            print(f"\n❌ 配置部署失败!")
            print(f"   环境: {result.environment}")
            print(f"   错误信息: {result.error_message}")
            if result.validation_errors:
                print(f"   验证错误: {result.validation_errors}")

    except Exception as e:
        logger.error(f"部署失败: {e}")
        print(f"\n❌ 部署失败: {e}")


if __name__ == "__main__":
    main()
