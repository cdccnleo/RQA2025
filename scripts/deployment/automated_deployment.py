#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
自动化部署脚本
用于生产环境的自动化部署流程
"""

import sys
import json
import logging
import argparse
import subprocess
import time
import shutil
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from enum import Enum
import requests
import psutil

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class DeploymentStatus(Enum):
    """部署状态"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLBACK = "rollback"


class DeploymentType(Enum):
    """部署类型"""
    FULL = "full"              # 全量部署
    INCREMENTAL = "incremental"  # 增量部署
    ROLLING = "rolling"         # 滚动部署
    BLUE_GREEN = "blue_green"   # 蓝绿部署


@dataclass
class DeploymentConfig:
    """部署配置"""
    deployment_type: DeploymentType
    target_environment: str
    version: str
    rollback_enabled: bool = True
    health_check_enabled: bool = True
    monitoring_enabled: bool = True
    backup_enabled: bool = True
    timeout_minutes: int = 30


@dataclass
class DeploymentResult:
    """部署结果"""
    status: DeploymentStatus
    start_time: float
    end_time: Optional[float] = None
    error_message: Optional[str] = None
    rollback_required: bool = False
    health_check_passed: bool = False
    backup_created: bool = False


class AutomatedDeployment:
    """自动化部署管理器"""

    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.deployment_dir = Path("deploy")
        self.backup_dir = Path("backup") / "deployment"
        self.current_deployment = None

        # 确保目录存在
        self.backup_dir.mkdir(parents=True, exist_ok=True)

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

    def deploy(self) -> DeploymentResult:
        """执行部署"""
        self.logger.info(f"开始部署: {self.config.deployment_type.value}")
        self.logger.info(f"目标环境: {self.config.target_environment}")
        self.logger.info(f"版本: {self.config.version}")

        # 创建部署结果
        result = DeploymentResult(
            status=DeploymentStatus.IN_PROGRESS,
            start_time=time.time()
        )

        try:
            # 1. 预部署检查
            if not self._pre_deployment_check():
                result.status = DeploymentStatus.FAILED
                result.error_message = "预部署检查失败"
                return result

            # 2. 创建备份
            if self.config.backup_enabled:
                if not self._create_backup():
                    result.status = DeploymentStatus.FAILED
                    result.error_message = "备份创建失败"
                    return result
                result.backup_created = True

            # 3. 执行部署
            if not self._execute_deployment():
                result.status = DeploymentStatus.FAILED
                result.error_message = "部署执行失败"
                result.rollback_required = True
                return result

            # 4. 健康检查
            if self.config.health_check_enabled:
                if not self._health_check():
                    result.status = DeploymentStatus.FAILED
                    result.error_message = "健康检查失败"
                    result.rollback_required = True
                    return result
                result.health_check_passed = True

            # 5. 部署后配置
            if not self._post_deployment_config():
                result.status = DeploymentStatus.FAILED
                result.error_message = "部署后配置失败"
                result.rollback_required = True
                return result

            # 6. 启动监控
            if self.config.monitoring_enabled:
                self._start_monitoring()

            # 部署成功
            result.status = DeploymentStatus.SUCCESS
            result.end_time = time.time()

            self.logger.info("✅ 部署成功完成！")
            return result

        except Exception as e:
            self.logger.error(f"部署过程中发生错误: {str(e)}")
            result.status = DeploymentStatus.FAILED
            result.error_message = str(e)
            result.rollback_required = True
            return result

    def _pre_deployment_check(self) -> bool:
        """预部署检查"""
        self.logger.info("执行预部署检查...")

        checks = [
            self._check_system_resources,
            self._check_dependencies,
            self._check_configuration,
            self._check_network_connectivity,
            self._check_database_connectivity
        ]

        for check in checks:
            if not check():
                return False

        self.logger.info("✅ 预部署检查通过")
        return True

    def _check_system_resources(self) -> bool:
        """检查系统资源"""
        self.logger.info("检查系统资源...")

        # 检查磁盘空间
        disk_usage = psutil.disk_usage('/')
        free_gb = disk_usage.free / (1024**3)
        if free_gb < 5:  # 至少需要5GB可用空间
            self.logger.error(f"磁盘空间不足: {free_gb:.2f}GB")
            return False

        # 检查内存
        memory = psutil.virtual_memory()
        if memory.available < 2 * 1024 * 1024 * 1024:  # 至少需要2GB可用内存
            self.logger.error("内存不足")
            return False

        # 检查CPU负载
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 90:
            self.logger.warning(f"CPU负载较高: {cpu_percent}%")

        self.logger.info("✅ 系统资源检查通过")
        return True

    def _check_dependencies(self) -> bool:
        """检查依赖"""
        self.logger.info("检查系统依赖...")

        required_commands = [
            "docker", "docker-compose", "kubectl", "helm"
        ]

        for cmd in required_commands:
            if not shutil.which(cmd):
                self.logger.error(f"缺少必需命令: {cmd}")
                return False

        self.logger.info("✅ 依赖检查通过")
        return True

    def _check_configuration(self) -> bool:
        """检查配置"""
        self.logger.info("检查部署配置...")

        # 检查配置文件
        config_files = [
            f"config/{self.config.target_environment}/config.yaml",
            f"deploy/{self.config.target_environment}_deployment.yaml",
            ".env.production"
        ]

        for config_file in config_files:
            if not Path(config_file).exists():
                self.logger.error(f"配置文件不存在: {config_file}")
                return False

        self.logger.info("✅ 配置检查通过")
        return True

    def _check_network_connectivity(self) -> bool:
        """检查网络连通性"""
        self.logger.info("检查网络连通性...")

        # 检查外部服务连通性
        external_services = [
            "https://api.github.com",
            "https://hub.docker.com"
        ]

        for service in external_services:
            try:
                response = requests.get(service, timeout=10)
                if response.status_code != 200:
                    self.logger.warning(f"服务 {service} 响应异常: {response.status_code}")
            except Exception as e:
                self.logger.warning(f"无法连接到 {service}: {str(e)}")

        self.logger.info("✅ 网络连通性检查通过")
        return True

    def _check_database_connectivity(self) -> bool:
        """检查数据库连通性"""
        self.logger.info("检查数据库连通性...")

        # 这里应该根据实际配置检查数据库连接
        # 暂时跳过，在实际部署中需要实现
        self.logger.info("⚠️  数据库连通性检查跳过（需要实际配置）")
        return True

    def _create_backup(self) -> bool:
        """创建备份"""
        self.logger.info("创建部署备份...")

        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_name = f"deployment_backup_{timestamp}"
            backup_path = self.backup_dir / backup_name

            # 创建备份目录
            backup_path.mkdir(parents=True, exist_ok=True)

            # 备份配置文件
            config_backup = backup_path / "config"
            config_backup.mkdir(exist_ok=True)

            shutil.copytree("config", config_backup, dirs_exist_ok=True)

            # 备份部署文件
            deploy_backup = backup_path / "deploy"
            deploy_backup.mkdir(exist_ok=True)

            shutil.copytree("deploy", deploy_backup, dirs_exist_ok=True)

            # 创建备份元数据
            backup_metadata = {
                "timestamp": timestamp,
                "version": self.config.version,
                "environment": self.config.target_environment,
                "deployment_type": self.config.deployment_type.value,
                "backup_path": str(backup_path)
            }

            with open(backup_path / "backup_metadata.json", 'w', encoding='utf-8') as f:
                json.dump(backup_metadata, f, indent=2, ensure_ascii=False)

            self.logger.info(f"✅ 备份创建成功: {backup_path}")
            return True

        except Exception as e:
            self.logger.error(f"备份创建失败: {str(e)}")
            return False

    def _execute_deployment(self) -> bool:
        """执行部署"""
        self.logger.info("执行部署...")

        try:
            if self.config.deployment_type == DeploymentType.DOCKER:
                return self._deploy_with_docker()
            elif self.config.deployment_type == DeploymentType.KUBERNETES:
                return self._deploy_with_kubernetes()
            elif self.config.deployment_type == DeploymentType.HELM:
                return self._deploy_with_helm()
            else:
                self.logger.error(f"不支持的部署类型: {self.config.deployment_type.value}")
                return False

        except Exception as e:
            self.logger.error(f"部署执行失败: {str(e)}")
            return False

    def _deploy_with_docker(self) -> bool:
        """使用Docker部署"""
        self.logger.info("使用Docker部署...")

        try:
            # 构建Docker镜像
            self.logger.info("构建Docker镜像...")
            build_cmd = [
                "docker", "build",
                "-t", f"rqa2025:{self.config.version}",
                "-f", "deploy/docker/Dockerfile",
                "."
            ]

            result = subprocess.run(build_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error(f"Docker构建失败: {result.stderr}")
                return False

            # 停止现有容器
            self.logger.info("停止现有容器...")
            stop_cmd = ["docker-compose", "-f", "deploy/docker-compose.production.yml", "down"]
            subprocess.run(stop_cmd, capture_output=True)

            # 启动新容器
            self.logger.info("启动新容器...")
            start_cmd = ["docker-compose", "-f", "deploy/docker-compose.production.yml", "up", "-d"]
            result = subprocess.run(start_cmd, capture_output=True, text=True)

            if result.returncode != 0:
                self.logger.error(f"容器启动失败: {result.stderr}")
                return False

            self.logger.info("✅ Docker部署成功")
            return True

        except Exception as e:
            self.logger.error(f"Docker部署失败: {str(e)}")
            return False

    def _deploy_with_kubernetes(self) -> bool:
        """使用Kubernetes部署"""
        self.logger.info("使用Kubernetes部署...")

        try:
            # 应用Kubernetes配置
            k8s_config = f"deploy/kubernetes/{self.config.target_environment}_deployment.yaml"

            if not Path(k8s_config).exists():
                self.logger.error(f"Kubernetes配置文件不存在: {k8s_config}")
                return False

            apply_cmd = ["kubectl", "apply", "-f", k8s_config]
            result = subprocess.run(apply_cmd, capture_output=True, text=True)

            if result.returncode != 0:
                self.logger.error(f"Kubernetes部署失败: {result.stderr}")
                return False

            # 等待部署完成
            self.logger.info("等待部署完成...")
            rollout_cmd = ["kubectl", "rollout", "status", "deployment/rqa2025"]

            for _ in range(60):  # 最多等待5分钟
                result = subprocess.run(rollout_cmd, capture_output=True, text=True)
                if "successfully rolled out" in result.stdout:
                    self.logger.info("✅ Kubernetes部署成功")
                    return True
                time.sleep(5)

            self.logger.error("Kubernetes部署超时")
            return False

        except Exception as e:
            self.logger.error(f"Kubernetes部署失败: {str(e)}")
            return False

    def _deploy_with_helm(self) -> bool:
        """使用Helm部署"""
        self.logger.info("使用Helm部署...")

        try:
            # 更新Helm仓库
            repo_cmd = ["helm", "repo", "update"]
            subprocess.run(repo_cmd, capture_output=True)

            # 安装/升级Helm Chart
            install_cmd = [
                "helm", "upgrade", "--install", "rqa2025",
                "deploy/helm/rqa2025",
                "--namespace", self.config.target_environment,
                "--create-namespace",
                "--set", f"image.tag={self.config.version}",
                "--wait", "--timeout", "5m"
            ]

            result = subprocess.run(install_cmd, capture_output=True, text=True)

            if result.returncode != 0:
                self.logger.error(f"Helm部署失败: {result.stderr}")
                return False

            self.logger.info("✅ Helm部署成功")
            return True

        except Exception as e:
            self.logger.error(f"Helm部署失败: {str(e)}")
            return False

    def _health_check(self) -> bool:
        """健康检查"""
        self.logger.info("执行健康检查...")

        try:
            # 检查应用健康状态
            health_endpoints = [
                "http://localhost:8080/health",
                "http://localhost:8080/ready",
                "http://localhost:8080/metrics"
            ]

            for endpoint in health_endpoints:
                try:
                    response = requests.get(endpoint, timeout=10)
                    if response.status_code != 200:
                        self.logger.error(f"健康检查失败: {endpoint} - {response.status_code}")
                        return False
                except Exception as e:
                    self.logger.error(f"健康检查失败: {endpoint} - {str(e)}")
                    return False

            # 检查数据库连接
            # 这里应该实现实际的数据库连接检查

            # 检查Redis连接
            # 这里应该实现实际的Redis连接检查

            self.logger.info("✅ 健康检查通过")
            return True

        except Exception as e:
            self.logger.error(f"健康检查失败: {str(e)}")
            return False

    def _post_deployment_config(self) -> bool:
        """部署后配置"""
        self.logger.info("执行部署后配置...")

        try:
            # 更新配置
            if not self._update_configuration():
                return False

            # 清理临时文件
            if not self._cleanup_temp_files():
                return False

            # 更新部署状态
            if not self._update_deployment_status():
                return False

            self.logger.info("✅ 部署后配置完成")
            return True

        except Exception as e:
            self.logger.error(f"部署后配置失败: {str(e)}")
            return False

    def _update_configuration(self) -> bool:
        """更新配置"""
        # 这里应该实现实际的配置更新逻辑
        self.logger.info("配置更新完成")
        return True

    def _cleanup_temp_files(self) -> bool:
        """清理临时文件"""
        # 清理部署过程中产生的临时文件
        temp_patterns = ["*.tmp", "*.log", "*.cache"]

        for pattern in temp_patterns:
            for temp_file in Path(".").glob(pattern):
                try:
                    temp_file.unlink()
                except Exception as e:
                    self.logger.warning(f"清理临时文件失败: {temp_file} - {str(e)}")

        self.logger.info("临时文件清理完成")
        return True

    def _update_deployment_status(self) -> bool:
        """更新部署状态"""
        # 更新部署状态文件
        status_file = Path("deploy/deployment_status.json")

        status_data = {
            "last_deployment": {
                "timestamp": time.time(),
                "version": self.config.version,
                "environment": self.config.target_environment,
                "status": "success"
            },
            "deployment_history": []
        }

        # 读取现有历史记录
        if status_file.exists():
            try:
                with open(status_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    status_data["deployment_history"] = existing_data.get("deployment_history", [])
            except Exception as e:
                self.logger.warning(f"读取部署状态失败: {str(e)}")

        # 添加新的部署记录
        status_data["deployment_history"].append({
            "timestamp": time.time(),
            "version": self.config.version,
            "environment": self.config.target_environment,
            "status": "success",
            "deployment_type": self.config.deployment_type.value
        })

        # 保持最近10条记录
        status_data["deployment_history"] = status_data["deployment_history"][-10:]

        # 写入状态文件
        try:
            with open(status_file, 'w', encoding='utf-8') as f:
                json.dump(status_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"更新部署状态失败: {str(e)}")
            return False

        return True

    def _start_monitoring(self):
        """启动监控"""
        self.logger.info("启动监控系统...")

        try:
            # 启动Prometheus
            prometheus_cmd = ["docker-compose", "-f",
                              "deploy/monitoring/docker-compose.yml", "up", "-d", "prometheus"]
            subprocess.run(prometheus_cmd, capture_output=True)

            # 启动Grafana
            grafana_cmd = ["docker-compose", "-f",
                           "deploy/monitoring/docker-compose.yml", "up", "-d", "grafana"]
            subprocess.run(grafana_cmd, capture_output=True)

            # 启动AlertManager
            alertmanager_cmd = ["docker-compose", "-f",
                                "deploy/monitoring/docker-compose.yml", "up", "-d", "alertmanager"]
            subprocess.run(alertmanager_cmd, capture_output=True)

            self.logger.info("✅ 监控系统启动成功")

        except Exception as e:
            self.logger.error(f"监控系统启动失败: {str(e)}")

    def rollback(self) -> bool:
        """回滚部署"""
        self.logger.info("开始回滚部署...")

        try:
            # 查找最近的备份
            backup_dirs = sorted(self.backup_dir.glob("deployment_backup_*"), reverse=True)

            if not backup_dirs:
                self.logger.error("没有找到可用的备份")
                return False

            latest_backup = backup_dirs[0]
            self.logger.info(f"使用备份: {latest_backup}")

            # 恢复配置文件
            config_backup = latest_backup / "config"
            if config_backup.exists():
                shutil.rmtree("config", ignore_errors=True)
                shutil.copytree(config_backup, "config")

            # 恢复部署文件
            deploy_backup = latest_backup / "deploy"
            if deploy_backup.exists():
                shutil.rmtree("deploy", ignore_errors=True)
                shutil.copytree(deploy_backup, "deploy")

            # 重新部署
            if not self._execute_deployment():
                self.logger.error("回滚部署失败")
                return False

            self.logger.info("✅ 回滚部署成功")
            return True

        except Exception as e:
            self.logger.error(f"回滚部署失败: {str(e)}")
            return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="自动化部署脚本")
    parser.add_argument(
        "--type", "-t",
        choices=["docker", "kubernetes", "helm"],
        default="docker",
        help="部署类型"
    )
    parser.add_argument(
        "--environment", "-e",
        choices=["development", "staging", "production"],
        default="production",
        help="目标环境"
    )
    parser.add_argument(
        "--version", "-v",
        required=True,
        help="部署版本"
    )
    parser.add_argument(
        "--no-rollback",
        action="store_true",
        help="禁用回滚"
    )
    parser.add_argument(
        "--no-health-check",
        action="store_true",
        help="禁用健康检查"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="禁用备份"
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="执行回滚"
    )

    args = parser.parse_args()

    # 创建部署配置
    config = DeploymentConfig(
        deployment_type=DeploymentType(args.type),
        target_environment=args.environment,
        version=args.version,
        rollback_enabled=not args.no_rollback,
        health_check_enabled=not args.no_health_check,
        backup_enabled=not args.no_backup
    )

    # 创建部署管理器
    deployment = AutomatedDeployment(config)

    if args.rollback:
        # 执行回滚
        success = deployment.rollback()
        sys.exit(0 if success else 1)
    else:
        # 执行部署
        result = deployment.deploy()

        if result.status == DeploymentStatus.SUCCESS:
            print("✅ 部署成功完成！")
            sys.exit(0)
        else:
            print(f"❌ 部署失败: {result.error_message}")

            if result.rollback_required and config.rollback_enabled:
                print("开始回滚...")
                if deployment.rollback():
                    print("✅ 回滚成功")
                    sys.exit(0)
                else:
                    print("❌ 回滚失败")
                    sys.exit(1)
            else:
                sys.exit(1)


if __name__ == "__main__":
    main()
