#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
云原生架构部署脚本
自动化部署RQA2025云原生架构
"""

import sys
import subprocess
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CloudNativeDeployer:
    """云原生架构部署器"""

    def __init__(self, config_path: str = "deploy/kubernetes"):
        self.config_path = Path(config_path)
        self.project_root = Path(__file__).parent.parent.parent
        self.deploy_status = {}

    def check_prerequisites(self) -> bool:
        """检查部署前置条件"""
        logger.info("检查部署前置条件...")

        # 检查Docker
        try:
            result = subprocess.run(['docker', '--version'],
                                    capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("✓ Docker已安装")
            else:
                logger.error("✗ Docker未安装")
                return False
        except FileNotFoundError:
            logger.error("✗ Docker未安装")
            return False

        # 检查kubectl
        try:
            result = subprocess.run(['kubectl', 'version', '--client'],
                                    capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("✓ kubectl已安装")
            else:
                logger.error("✗ kubectl未安装")
                return False
        except FileNotFoundError:
            logger.error("✗ kubectl未安装")
            return False

        return True

    def build_docker_images(self) -> bool:
        """构建Docker镜像"""
        logger.info("构建Docker镜像...")

        images = [
            ("deploy/Dockerfile.engine-service", "rqa2025-engine:latest"),
            ("deploy/Dockerfile.infrastructure-service", "rqa2025-infrastructure:latest"),
            ("deploy/Dockerfile.business-service", "rqa2025-business:latest"),
        ]

        for dockerfile, image_name in images:
            try:
                logger.info(f"构建镜像: {image_name}")
                result = subprocess.run([
                    'docker', 'build', '-f', dockerfile,
                    '-t', image_name, '.'
                ], cwd=self.project_root, capture_output=True, text=True)

                if result.returncode == 0:
                    logger.info(f"✓ 镜像构建成功: {image_name}")
                else:
                    logger.error(f"✗ 镜像构建失败: {image_name}")
                    return False
            except Exception as e:
                logger.error(f"✗ 镜像构建异常: {image_name}, 错误: {e}")
                return False

        return True

    def deploy_kubernetes_resources(self) -> bool:
        """部署Kubernetes资源"""
        logger.info("部署Kubernetes资源...")

        resources = [
            "namespace.yml",
            "configmap.yml",
            "services.yml",
            "deployments.yml",
            "autoscaling.yml"
        ]

        for resource in resources:
            try:
                resource_path = f"deploy/kubernetes/{resource}"
                logger.info(f"部署资源: {resource}")

                result = subprocess.run([
                    'kubectl', 'apply', '-f', resource_path
                ], cwd=self.project_root, capture_output=True, text=True)

                if result.returncode == 0:
                    logger.info(f"✓ 资源部署成功: {resource}")
                else:
                    logger.error(f"✗ 资源部署失败: {resource}")
                    return False
            except Exception as e:
                logger.error(f"✗ 资源部署异常: {resource}, 错误: {e}")
                return False

        return True

    def deploy(self) -> bool:
        """执行完整部署"""
        logger.info("开始云原生架构部署...")

        # 检查前置条件
        if not self.check_prerequisites():
            logger.error("前置条件检查失败")
            return False

        # 构建Docker镜像
        if not self.build_docker_images():
            logger.error("Docker镜像构建失败")
            return False

        # 部署Kubernetes资源
        if not self.deploy_kubernetes_resources():
            logger.error("Kubernetes资源部署失败")
            return False

        logger.info("云原生架构部署完成!")
        return True


def main():
    """主函数"""
    deployer = CloudNativeDeployer()

    if deployer.deploy():
        logger.info("✅ 云原生架构部署成功!")
        sys.exit(0)
    else:
        logger.error("❌ 云原生架构部署失败!")
        sys.exit(1)


if __name__ == "__main__":
    main()
