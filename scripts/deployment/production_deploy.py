#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 生产环境部署脚本
"""

import os
import sys
import yaml
import time
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class DeploymentStatus(Enum):
    """部署状态"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLBACK = "rollback"


@dataclass
class DeploymentStep:
    """部署步骤"""
    name: str
    command: str
    timeout: int = 300
    retries: int = 3
    critical: bool = True
    status: DeploymentStatus = DeploymentStatus.PENDING
    output: str = ""
    error: str = ""


class ProductionDeployer:
    """生产环境部署器"""

    def __init__(self, config_file: str = "config/production/deployment.yaml"):
        self.config_file = Path(config_file)
        self.logger = self._setup_logging()
        self.deployment_config = self._load_deployment_config()
        self.steps: List[DeploymentStep] = []
        self.current_step = 0

        # 设置环境变量
        self._setup_environment()

    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        log_dir = Path("logs/deployment")
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"deployment_{int(time.time())}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger(__name__)

    def _load_deployment_config(self) -> Dict[str, Any]:
        """加载部署配置"""
        if not self.config_file.exists():
            self.logger.warning(f"部署配置文件不存在: {self.config_file}")
            return self._get_default_deployment_config()

        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"加载部署配置失败: {str(e)}")
            return self._get_default_deployment_config()

    def _get_default_deployment_config(self) -> Dict[str, Any]:
        """获取默认部署配置"""
        return {
            "deployment": {
                "strategy": "blue-green",
                "health_check": {
                    "enabled": True,
                    "interval": 30,
                    "timeout": 10,
                    "retries": 3
                },
                "rollback": {
                    "enabled": True,
                    "max_versions": 5
                },
                "scaling": {
                    "auto_scaling": True,
                    "min_instances": 2,
                    "max_instances": 10
                }
            },
            "services": [
                "data-service",
                "features-service",
                "model-service",
                "trading-service",
                "risk-service",
                "api-gateway"
            ]
        }

    def _setup_environment(self):
        """设置环境变量"""
        # 设置生产环境标识
        os.environ["RQA_ENV"] = "production"
        os.environ["RQA_DEPLOYMENT"] = "true"

        # 设置日志级别
        os.environ["LOG_LEVEL"] = "INFO"

        self.logger.info("环境变量设置完成")

    def prepare_deployment(self) -> bool:
        """准备部署"""
        self.logger.info("开始准备部署...")

        try:
            # 验证配置
            if not self._validate_deployment_config():
                return False

            # 创建部署步骤
            self._create_deployment_steps()

            # 检查系统资源
            if not self._check_system_resources():
                return False

            # 备份当前版本
            if not self._backup_current_version():
                return False

            self.logger.info("部署准备完成")
            return True

        except Exception as e:
            self.logger.error(f"部署准备失败: {str(e)}")
            return False

    def _validate_deployment_config(self) -> bool:
        """验证部署配置"""
        self.logger.info("验证部署配置...")

        required_fields = ["deployment", "services"]
        for field in required_fields:
            if field not in self.deployment_config:
                self.logger.error(f"缺少必需的配置字段: {field}")
                return False

        # 验证服务列表
        services = self.deployment_config.get("services", [])
        if not services:
            self.logger.error("服务列表为空")
            return False

        # 检查Docker镜像是否存在
        for service in services:
            if not self._check_docker_image(service):
                self.logger.warning(f"Docker镜像不存在: {service}")

        return True

    def _check_docker_image(self, service: str) -> bool:
        """检查Docker镜像是否存在"""
        try:
            result = subprocess.run(
                ["docker", "images", "-q", f"rqa2025/{service}:latest"],
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode == 0 and result.stdout.strip()
        except Exception as e:
            self.logger.warning(f"检查Docker镜像失败: {str(e)}")
            return False

    def _create_deployment_steps(self):
        """创建部署步骤"""
        self.logger.info("创建部署步骤...")

        # 基础部署步骤
        self.steps = [
            DeploymentStep(
                name="停止旧服务",
                command="docker-compose -f deploy/docker-compose.prod.yml down",
                timeout=120,
                critical=True
            ),
            DeploymentStep(
                name="拉取最新镜像",
                command="docker-compose -f deploy/docker-compose.prod.yml pull",
                timeout=300,
                critical=True
            ),
            DeploymentStep(
                name="启动新服务",
                command="docker-compose -f deploy/docker-compose.prod.yml up -d",
                timeout=300,
                critical=True
            ),
            DeploymentStep(
                name="健康检查",
                command="python scripts/deployment/health_check.py",
                timeout=180,
                critical=True
            ),
            DeploymentStep(
                name="性能测试",
                command="python scripts/testing/run_performance_tests.py",
                timeout=600,
                critical=False
            )
        ]

        self.logger.info(f"创建了 {len(self.steps)} 个部署步骤")

    def _check_system_resources(self) -> bool:
        """检查系统资源"""
        self.logger.info("检查系统资源...")

        try:
            # 检查磁盘空间
            disk_usage = subprocess.run(
                ["df", "-h", "/"],
                capture_output=True,
                text=True,
                timeout=30
            )

            if disk_usage.returncode == 0:
                lines = disk_usage.stdout.strip().split('\n')
                if len(lines) > 1:
                    usage_line = lines[1]
                    usage_parts = usage_line.split()
                    if len(usage_parts) >= 5:
                        usage_percent = usage_parts[4].replace('%', '')
                        if int(usage_percent) > 90:
                            self.logger.warning(f"磁盘使用率过高: {usage_percent}%")

            # 检查内存
            memory_info = subprocess.run(
                ["free", "-h"],
                capture_output=True,
                text=True,
                timeout=30
            )

            if memory_info.returncode == 0:
                self.logger.info("系统资源检查完成")

            return True

        except Exception as e:
            self.logger.warning(f"系统资源检查失败: {str(e)}")
            return True  # 不阻止部署

    def _backup_current_version(self) -> bool:
        """备份当前版本"""
        self.logger.info("备份当前版本...")

        try:
            backup_dir = Path("backups/deployment")
            backup_dir.mkdir(parents=True, exist_ok=True)

            timestamp = int(time.time())
            backup_name = f"production_backup_{timestamp}"
            backup_path = backup_dir / backup_name

            # 创建备份目录
            backup_path.mkdir(exist_ok=True)

            # 备份配置文件
            config_backup = backup_path / "config"
            config_backup.mkdir(exist_ok=True)

            subprocess.run(
                ["cp", "-r", "config/production", str(config_backup)],
                check=True,
                timeout=120
            )

            # 备份部署脚本
            scripts_backup = backup_path / "scripts"
            scripts_backup.mkdir(exist_ok=True)

            subprocess.run(
                ["cp", "-r", "scripts/deployment", str(scripts_backup)],
                check=True,
                timeout=120
            )

            self.logger.info(f"备份完成: {backup_path}")
            return True

        except Exception as e:
            self.logger.error(f"备份失败: {str(e)}")
            return False

    def execute_deployment(self) -> bool:
        """执行部署"""
        self.logger.info("开始执行部署...")

        total_steps = len(self.steps)

        for i, step in enumerate(self.steps):
            self.current_step = i + 1
            self.logger.info(f"执行步骤 {self.current_step}/{total_steps}: {step.name}")

            # 执行步骤
            if not self._execute_step(step):
                if step.critical:
                    self.logger.error(f"关键步骤失败: {step.name}")
                    self.logger.info("开始回滚...")
                    self._rollback_deployment()
                    return False
                else:
                    self.logger.warning(f"非关键步骤失败: {step.name}，继续执行")

            # 步骤间等待
            if i < total_steps - 1:
                time.sleep(5)

        self.logger.info("部署执行完成")
        return True

    def _execute_step(self, step: DeploymentStep) -> bool:
        """执行单个部署步骤"""
        step.status = DeploymentStatus.IN_PROGRESS

        for attempt in range(step.retries):
            try:
                self.logger.info(f"执行命令: {step.command} (尝试 {attempt + 1}/{step.retries})")

                result = subprocess.run(
                    step.command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=step.timeout,
                    env=os.environ.copy()
                )

                step.output = result.stdout
                step.error = result.stderr

                if result.returncode == 0:
                    step.status = DeploymentStatus.SUCCESS
                    self.logger.info(f"步骤 {step.name} 执行成功")
                    return True
                else:
                    self.logger.warning(f"步骤 {step.name} 执行失败 (尝试 {attempt + 1}): {result.stderr}")

                    if attempt < step.retries - 1:
                        time.sleep(10)  # 重试前等待

            except subprocess.TimeoutExpired:
                step.error = f"步骤执行超时 (>{step.timeout}s)"
                self.logger.error(f"步骤 {step.name} 执行超时")

            except Exception as e:
                step.error = str(e)
                self.logger.error(f"步骤 {step.name} 执行异常: {str(e)}")

        step.status = DeploymentStatus.FAILED
        return False

    def _rollback_deployment(self):
        """回滚部署"""
        self.logger.info("开始回滚部署...")

        try:
            # 停止新服务
            subprocess.run(
                ["docker-compose", "-f", "deploy/docker-compose.prod.yml", "down"],
                check=True,
                timeout=120
            )

            # 启动旧服务
            subprocess.run(
                ["docker-compose", "-f", "deploy/docker-compose.prod.yml", "up", "-d"],
                check=True,
                timeout=300
            )

            self.logger.info("回滚完成")

        except Exception as e:
            self.logger.error(f"回滚失败: {str(e)}")

    def generate_deployment_report(self) -> str:
        """生成部署报告"""
        report = ["# 生产环境部署报告\n"]

        # 部署概览
        total_steps = len(self.steps)
        successful_steps = len([s for s in self.steps if s.status == DeploymentStatus.SUCCESS])
        failed_steps = len([s for s in self.steps if s.status == DeploymentStatus.FAILED])

        report.append(f"## 部署概览\n")
        report.append(f"- 总步骤数: {total_steps}")
        report.append(f"- 成功步骤: {successful_steps}")
        report.append(f"- 失败步骤: {failed_steps}")
        report.append(f"- 成功率: {successful_steps/total_steps*100:.1f}%\n")

        # 步骤详情
        report.append("## 步骤详情\n")
        for i, step in enumerate(self.steps):
            status_icon = "✅" if step.status == DeploymentStatus.SUCCESS else "❌"
            report.append(f"### {i+1}. {step.name} {status_icon}")
            report.append(f"- **状态**: {step.status.value}")
            report.append(f"- **命令**: `{step.command}`")
            report.append(f"- **超时**: {step.timeout}s")
            report.append(f"- **重试**: {step.retries}次")
            report.append(f"- **关键性**: {'是' if step.critical else '否'}")

            if step.output:
                report.append(f"- **输出**: ```\n{step.output}\n```")

            if step.error:
                report.append(f"- **错误**: ```\n{step.error}\n```")

            report.append("")

        # 部署结果
        if failed_steps == 0:
            report.append("## 部署结果\n")
            report.append("🎉 **部署成功！** 所有步骤都已完成。")
        else:
            report.append("## 部署结果\n")
            report.append("⚠️ **部署部分失败** 请检查失败的步骤并手动处理。")

        return "\n".join(report)

    def save_deployment_report(self, report: str):
        """保存部署报告"""
        report_dir = Path("reports/deployment")
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time())
        report_file = report_dir / f"deployment_report_{timestamp}.md"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        self.logger.info(f"部署报告已保存: {report_file}")


def main():
    """主函数"""
    deployer = ProductionDeployer()

    # 准备部署
    if not deployer.prepare_deployment():
        print("❌ 部署准备失败")
        sys.exit(1)

    # 执行部署
    if not deployer.execute_deployment():
        print("❌ 部署执行失败")
        sys.exit(1)

    # 生成报告
    report = deployer.generate_deployment_report()
    print(report)

    # 保存报告
    deployer.save_deployment_report(report)

    print("✅ 部署完成！")


if __name__ == "__main__":
    main()
