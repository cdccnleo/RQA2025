#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
生产环境云原生部署脚本
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from typing import Dict, Any
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/production_deployment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProductionDeployer:
    """生产环境部署器"""

    def __init__(self, config_path: str = "deploy/production_cloud_native.yml"):
        self.config_path = config_path
        self.namespace = "rqa2025-cloud-native"
        self.deployment_status = {}

    def check_prerequisites(self) -> bool:
        """检查部署前置条件"""
        logger.info("检查部署前置条件...")

        # 检查kubectl
        try:
            result = subprocess.run(['kubectl', 'version', '--client'],
                                    capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("kubectl未安装或配置错误")
                return False
            logger.info("✓ kubectl可用")
        except FileNotFoundError:
            logger.error("kubectl未安装")
            return False

        # 检查Docker
        try:
            result = subprocess.run(['docker', 'version'],
                                    capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("Docker未安装或配置错误")
                return False
            logger.info("✓ Docker可用")
        except FileNotFoundError:
            logger.error("Docker未安装")
            return False

        logger.info("✓ 所有前置条件检查通过")
        return True

    def deploy_namespace(self) -> bool:
        """部署命名空间"""
        logger.info(f"部署命名空间: {self.namespace}")

        try:
            result = subprocess.run(
                ['kubectl', 'create', 'namespace', self.namespace],
                capture_output=True, text=True
            )

            if result.returncode != 0 and "already exists" not in result.stderr:
                logger.error(f"创建命名空间失败: {result.stderr}")
                return False

            logger.info(f"✓ 命名空间 {self.namespace} 创建成功")
            return True

        except Exception as e:
            logger.error(f"部署命名空间时出错: {e}")
            return False

    def deploy_services(self) -> bool:
        """部署服务"""
        logger.info("部署服务...")

        try:
            result = subprocess.run(
                ['kubectl', 'apply', '-f', self.config_path],
                capture_output=True, text=True
            )

            if result.returncode != 0:
                logger.error(f"部署服务失败: {result.stderr}")
                return False

            logger.info("✓ 服务部署成功")
            return True

        except Exception as e:
            logger.error(f"部署服务时出错: {e}")
            return False

    def check_service_health(self, service_name: str) -> bool:
        """检查服务健康状态"""
        logger.info(f"检查服务健康状态: {service_name}")

        try:
            result = subprocess.run(
                ['kubectl', 'get', 'pods', '-n', self.namespace,
                 '-l', f'app={service_name}', '-o', 'json'],
                capture_output=True, text=True
            )

            if result.returncode != 0:
                logger.error(f"获取Pod状态失败: {result.stderr}")
                return False

            pods_data = json.loads(result.stdout)

            for pod in pods_data['items']:
                pod_name = pod['metadata']['name']
                pod_status = pod['status']['phase']

                if pod_status != 'Running':
                    logger.error(f"Pod {pod_name} 状态异常: {pod_status}")
                    return False

            logger.info(f"✓ 服务 {service_name} 健康检查通过")
            return True

        except Exception as e:
            logger.error(f"检查服务 {service_name} 健康状态时出错: {e}")
            return False

    def check_all_services_health(self) -> bool:
        """检查所有服务健康状态"""
        logger.info("检查所有服务健康状态...")

        services = [
            'rqa2025-backtest',
            'rqa2025-data',
            'rqa2025-intelligent'
        ]

        for service in services:
            if not self.check_service_health(service):
                return False

        logger.info("✓ 所有服务健康检查通过")
        return True

    def rollback_deployment(self) -> bool:
        """回滚部署"""
        logger.warning("开始回滚部署...")

        try:
            deployments = [
                'rqa2025-backtest-service',
                'rqa2025-data-service',
                'rqa2025-intelligent-orchestrator'
            ]

            for deployment in deployments:
                result = subprocess.run(
                    ['kubectl', 'rollout', 'undo', 'deployment', deployment, '-n', self.namespace],
                    capture_output=True, text=True
                )

                if result.returncode != 0:
                    logger.error(f"回滚部署 {deployment} 失败: {result.stderr}")
                    return False

                logger.info(f"✓ 部署 {deployment} 回滚成功")

            logger.info("✓ 所有部署回滚完成")
            return True

        except Exception as e:
            logger.error(f"回滚部署时出错: {e}")
            return False

    def get_deployment_status(self) -> Dict[str, Any]:
        """获取部署状态"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'namespace': self.namespace,
            'services': {},
            'health': {}
        }

        try:
            result = subprocess.run(
                ['kubectl', 'get', 'deployments', '-n', self.namespace, '-o', 'json'],
                capture_output=True, text=True
            )

            if result.returncode == 0:
                deployments = json.loads(result.stdout)
                for deployment in deployments['items']:
                    name = deployment['metadata']['name']
                    status['services'][name] = {
                        'replicas': deployment['spec']['replicas'],
                        'available': deployment['status']['availableReplicas'],
                        'ready': deployment['status']['readyReplicas']
                    }

        except Exception as e:
            logger.error(f"获取部署状态时出错: {e}")

        return status

    def deploy(self) -> bool:
        """执行完整部署流程"""
        logger.info("开始生产环境云原生部署...")

        deployment_steps = [
            ("检查前置条件", self.check_prerequisites),
            ("部署命名空间", self.deploy_namespace),
            ("部署服务", self.deploy_services),
            ("检查服务健康状态", self.check_all_services_health)
        ]

        for step_name, step_func in deployment_steps:
            logger.info(f"执行步骤: {step_name}")

            try:
                if not step_func():
                    logger.error(f"步骤 '{step_name}' 失败，开始回滚...")
                    self.rollback_deployment()
                    return False

                logger.info(f"✓ 步骤 '{step_name}' 完成")

            except Exception as e:
                logger.error(f"步骤 '{step_name}' 执行时出错: {e}")
                logger.error("开始回滚...")
                self.rollback_deployment()
                return False

        # 获取最终部署状态
        final_status = self.get_deployment_status()

        # 保存部署状态
        os.makedirs('logs', exist_ok=True)
        with open('logs/production_deployment_status.json', 'w') as f:
            json.dump(final_status, f, indent=2)

        logger.info("✓ 生产环境云原生部署完成！")
        logger.info(f"部署状态已保存到: logs/production_deployment_status.json")

        return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='生产环境云原生部署脚本')
    parser.add_argument('--config', default='deploy/production_cloud_native.yml',
                        help='部署配置文件路径')
    parser.add_argument('--dry-run', action='store_true',
                        help='仅检查配置，不执行实际部署')
    parser.add_argument('--rollback', action='store_true',
                        help='回滚部署')

    args = parser.parse_args()

    deployer = ProductionDeployer(args.config)

    if args.dry_run:
        logger.info("执行配置检查...")
        if deployer.check_prerequisites():
            logger.info("✓ 配置检查通过")
        else:
            logger.error("✗ 配置检查失败")
            sys.exit(1)
    elif args.rollback:
        logger.info("执行回滚...")
        if deployer.rollback_deployment():
            logger.info("✓ 回滚完成")
        else:
            logger.error("✗ 回滚失败")
            sys.exit(1)
    else:
        if deployer.deploy():
            logger.info("✓ 部署成功完成")
        else:
            logger.error("✗ 部署失败")
            sys.exit(1)


if __name__ == "__main__":
    main()
