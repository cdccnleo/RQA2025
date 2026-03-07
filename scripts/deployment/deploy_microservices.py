#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
微服务部署脚本

自动化部署和管理回测系统微服务
"""

from src.backtest.microservice_architecture import (
    MicroserviceOrchestrator, ServiceConfig, ConfigManager
)
import sys
import argparse
import logging
import asyncio
import yaml
from pathlib import Path
from typing import Dict, Any, List

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MicroserviceDeployer:
    """微服务部署器"""

    def __init__(self, config_path: str = "config/microservices.yml"):
        self.config_path = config_path
        self.config_manager = ConfigManager(config_path)
        self.orchestrator = MicroserviceOrchestrator()

    def load_service_configs(self) -> List[ServiceConfig]:
        """加载服务配置"""
        service_configs = []
        services_config = self.config_manager.config.get('services', {})

        for service_name, config in services_config.items():
            service_config = ServiceConfig(
                name=service_name,
                port=config.get('port', 8000),
                replicas=config.get('replicas', 1),
                cpu_limit=config.get('cpu_limit', '500m'),
                memory_limit=config.get('memory_limit', '512Mi'),
                environment=config.get('environment', {}),
                startup_timeout=config.get('startup_timeout', 300),
                shutdown_timeout=config.get('shutdown_timeout', 60),
                max_retries=config.get('max_retries', 3),
                retry_delay=config.get('retry_delay', 5),
                log_level=config.get('log_level', 'INFO'),
                metrics_enabled=config.get('metrics_enabled', True)
            )
            service_configs.append(service_config)

        return service_configs

    async def deploy_services(self, services: List[str] = None, environment: str = "production"):
        """部署服务"""
        try:
            service_configs = self.load_service_configs()

            if services:
                # 只部署指定的服务
                service_configs = [config for config in service_configs if config.name in services]

            logger.info(f"开始部署 {len(service_configs)} 个服务到 {environment} 环境")

            deployment_results = []
            for config in service_configs:
                try:
                    logger.info(f"部署服务: {config.name}")
                    result = await self._deploy_single_service(config, environment)
                    deployment_results.append({
                        'service': config.name,
                        'status': 'success' if result else 'failed',
                        'config': config
                    })
                except Exception as e:
                    logger.error(f"部署服务 {config.name} 失败: {e}")
                    deployment_results.append({
                        'service': config.name,
                        'status': 'failed',
                        'error': str(e)
                    })

            # 输出部署结果
            self._print_deployment_results(deployment_results)

            return deployment_results

        except Exception as e:
            logger.error(f"部署服务失败: {e}")
            raise

    async def _deploy_single_service(self, config: ServiceConfig, environment: str) -> bool:
        """部署单个服务"""
        try:
            # 构建Docker镜像
            image_tag = f"backtest-{config.name}:latest"
            dockerfile_path = f"dockerfiles/{config.name}"

            # 检查Dockerfile是否存在
            if not Path(dockerfile_path).exists():
                logger.warning(f"Dockerfile不存在: {dockerfile_path}")
                return False

            # 构建镜像
            build_success = self.orchestrator.docker_manager.build_image(
                config.name, dockerfile_path, image_tag
            )

            if not build_success:
                logger.error(f"构建Docker镜像失败: {image_tag}")
                return False

            # 部署到Kubernetes
            deploy_success = self.orchestrator.k8s_manager.deploy_service(
                config, image_tag, namespace=environment
            )

            if not deploy_success:
                logger.error(f"部署到Kubernetes失败: {config.name}")
                return False

            logger.info(f"服务 {config.name} 部署成功")
            return True

        except Exception as e:
            logger.error(f"部署服务 {config.name} 时发生错误: {e}")
            return False

    def _print_deployment_results(self, results: List[Dict[str, Any]]):
        """打印部署结果"""
        print("\n" + "="*50)
        print("部署结果汇总")
        print("="*50)

        success_count = 0
        failed_count = 0

        for result in results:
            status = result['status']
            service = result['service']

            if status == 'success':
                print(f"✅ {service}: 部署成功")
                success_count += 1
            else:
                print(f"❌ {service}: 部署失败")
                if 'error' in result:
                    print(f"   错误: {result['error']}")
                failed_count += 1

        print("-"*50)
        print(f"总计: {len(results)} 个服务")
        print(f"成功: {success_count} 个")
        print(f"失败: {failed_count} 个")
        print("="*50)

    async def scale_service(self, service_name: str, replicas: int):
        """扩展服务"""
        try:
            logger.info(f"扩展服务 {service_name} 到 {replicas} 个副本")
            result = await self.orchestrator.scale_service(service_name, replicas)

            if result:
                logger.info(f"服务 {service_name} 扩展成功")
            else:
                logger.error(f"服务 {service_name} 扩展失败")

            return result

        except Exception as e:
            logger.error(f"扩展服务时发生错误: {e}")
            return False

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        try:
            status = self.orchestrator.get_system_status()
            return status
        except Exception as e:
            logger.error(f"获取系统状态失败: {e}")
            return {}

    async def rollback_service(self, service_name: str, version: str):
        """回滚服务"""
        try:
            logger.info(f"回滚服务 {service_name} 到版本 {version}")

            # 获取服务配置
            service_configs = self.load_service_configs()
            target_config = None
            for config in service_configs:
                if config.name == service_name:
                    target_config = config
                    break

            if not target_config:
                logger.error(f"未找到服务配置: {service_name}")
                return False

            # 构建回滚镜像
            image_tag = f"backtest-{service_name}:{version}"
            dockerfile_path = f"dockerfiles/{service_name}"

            build_success = self.orchestrator.docker_manager.build_image(
                service_name, dockerfile_path, image_tag
            )

            if not build_success:
                logger.error(f"构建回滚镜像失败: {image_tag}")
                return False

            # 部署回滚版本
            deploy_success = self.orchestrator.k8s_manager.deploy_service(
                target_config, image_tag
            )

            if deploy_success:
                logger.info(f"服务 {service_name} 回滚成功")
            else:
                logger.error(f"服务 {service_name} 回滚失败")

            return deploy_success

        except Exception as e:
            logger.error(f"回滚服务时发生错误: {e}")
            return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="微服务部署工具")
    parser.add_argument("action", choices=["deploy", "scale", "status", "rollback"],
                        help="执行的操作")
    parser.add_argument("--config", default="config/microservices.yml",
                        help="配置文件路径")
    parser.add_argument("--services", nargs="+", help="要部署的服务列表")
    parser.add_argument("--environment", default="production",
                        help="部署环境")
    parser.add_argument("--replicas", type=int, help="副本数量（用于scale操作）")
    parser.add_argument("--service-name", help="服务名称（用于scale和rollback操作）")
    parser.add_argument("--version", help="版本号（用于rollback操作）")

    args = parser.parse_args()

    deployer = MicroserviceDeployer(args.config)

    try:
        if args.action == "deploy":
            asyncio.run(deployer.deploy_services(args.services, args.environment))

        elif args.action == "scale":
            if not args.service_name or not args.replicas:
                print("错误: scale操作需要指定--service-name和--replicas参数")
                return
            asyncio.run(deployer.scale_service(args.service_name, args.replicas))

        elif args.action == "status":
            status = deployer.get_system_status()
            print("系统状态:")
            print(yaml.dump(status, default_flow_style=False, allow_unicode=True))

        elif args.action == "rollback":
            if not args.service_name or not args.version:
                print("错误: rollback操作需要指定--service-name和--version参数")
                return
            asyncio.run(deployer.rollback_service(args.service_name, args.version))

    except KeyboardInterrupt:
        print("\n操作被用户中断")
    except Exception as e:
        logger.error(f"执行操作失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
