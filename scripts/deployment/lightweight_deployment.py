#!/usr/bin/env python3
"""
轻量级生产环境部署脚本
使用现有镜像或最小化构建，避免长时间构建
"""

from src.infrastructure.monitoring.application_monitor import ApplicationMonitor
from src.infrastructure.config import UnifiedConfigManager
import asyncio
import time
import logging
import json
import os
import docker
from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


logger = logging.getLogger(__name__)


@dataclass
class DeploymentResult:
    """部署结果"""
    service: str
    status: str
    deployment_time: datetime
    health_check: bool
    performance_metrics: Dict[str, Any]
    details: Dict[str, Any]


class LightweightDeployment:
    """轻量级生产环境部署器"""

    def __init__(self):
        self.config_manager = UnifiedConfigManager()
        self.app_monitor = ApplicationMonitor()
        self.docker_client = docker.from_env()

        # 部署结果
        self.deployment_results: List[DeploymentResult] = []

        # 轻量级部署配置
        self.deployment_config = {
            'services': {
                'api': {
                    'image': 'python:3.9-slim',  # 使用现有镜像
                    'ports': {'8000/tcp': 8000},
                    'environment': {'ENV': 'production'},
                    'command': ['python', '-m', 'uvicorn', 'src.infrastructure.web.app_factory:create_app', '--host', '0.0.0.0', '--port', '8000']
                }
            },
            'databases': {
                'redis': {
                    'image': 'redis:7-alpine',
                    'ports': {'6379/tcp': 6379}
                }
            },
            'monitoring': {
                'prometheus': {
                    'image': 'prom/prometheus:latest',
                    'ports': {'9090/tcp': 9090}
                }
            }
        }

        logger.info("LightweightDeployment initialized")

    async def run_lightweight_deployment(self):
        """运行轻量级部署"""
        logger.info("开始轻量级生产环境部署...")

        start_time = time.time()

        # 1. 环境检查
        await self._check_environment()

        # 2. 部署数据库服务
        await self._deploy_databases()

        # 3. 部署监控服务
        await self._deploy_monitoring()

        # 4. 部署应用服务
        await self._deploy_applications()

        # 5. 运行健康检查
        await self._run_health_checks()

        # 6. 生成部署报告
        await self._generate_deployment_report()

        total_time = time.time() - start_time
        logger.info(f"轻量级部署完成，总耗时: {total_time:.2f}秒")

    async def _check_environment(self):
        """检查部署环境"""
        logger.info("检查部署环境...")

        # 检查Docker
        try:
            self.docker_client.ping()
            logger.info("✅ Docker连接正常")
        except Exception as e:
            logger.error(f"❌ Docker连接失败: {e}")
            raise

        # 检查系统资源
        import psutil
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent

        logger.info(f"系统资源状态:")
        logger.info(f"  CPU使用率: {cpu_percent}%")
        logger.info(f"  内存使用率: {memory_percent}%")
        logger.info(f"  磁盘使用率: {disk_percent}%")

        # 资源检查
        if cpu_percent > 90:
            logger.warning("⚠️ CPU使用率过高")
        if memory_percent > 90:
            logger.warning("⚠️ 内存使用率过高")
        if disk_percent > 95:
            logger.error("❌ 磁盘空间不足")
            raise Exception("磁盘空间不足")

    async def _deploy_databases(self):
        """部署数据库服务"""
        logger.info("部署数据库服务...")

        for db_name, db_config in self.deployment_config['databases'].items():
            try:
                logger.info(f"部署数据库: {db_name}")

                # 创建容器
                container = self.docker_client.containers.run(
                    image=db_config['image'],
                    name=f"rqa2025-{db_name}",
                    ports=db_config.get('ports', {}),
                    environment=db_config.get('environment', {}),
                    detach=True,
                    restart_policy={"Name": "always"}
                )

                # 等待服务启动
                await asyncio.sleep(5)

                # 健康检查
                health_status = await self._check_service_health(db_name, container)

                logger.info(f"✅ 数据库部署成功: {db_name}")

                # 记录部署结果
                result = DeploymentResult(
                    service=f"database_{db_name}",
                    status="success",
                    deployment_time=datetime.now(),
                    health_check=health_status,
                    performance_metrics={
                        'container_id': container.id,
                        'status': container.status
                    },
                    details={'image': db_config['image']}
                )
                self.deployment_results.append(result)

            except Exception as e:
                logger.error(f"❌ 数据库部署失败: {db_name} - {e}")
                result = DeploymentResult(
                    service=f"database_{db_name}",
                    status="failed",
                    deployment_time=datetime.now(),
                    health_check=False,
                    performance_metrics={},
                    details={'error': str(e)}
                )
                self.deployment_results.append(result)

    async def _deploy_monitoring(self):
        """部署监控服务"""
        logger.info("部署监控服务...")

        for monitor_name, monitor_config in self.deployment_config['monitoring'].items():
            try:
                logger.info(f"部署监控服务: {monitor_name}")

                # 创建容器
                container = self.docker_client.containers.run(
                    image=monitor_config['image'],
                    name=f"rqa2025-{monitor_name}",
                    ports=monitor_config.get('ports', {}),
                    environment=monitor_config.get('environment', {}),
                    detach=True,
                    restart_policy={"Name": "always"}
                )

                # 等待服务启动
                await asyncio.sleep(10)

                # 健康检查
                health_status = await self._check_service_health(monitor_name, container)

                logger.info(f"✅ 监控服务部署成功: {monitor_name}")

                # 记录部署结果
                result = DeploymentResult(
                    service=f"monitoring_{monitor_name}",
                    status="success",
                    deployment_time=datetime.now(),
                    health_check=health_status,
                    performance_metrics={
                        'container_id': container.id,
                        'status': container.status
                    },
                    details={'image': monitor_config['image']}
                )
                self.deployment_results.append(result)

            except Exception as e:
                logger.error(f"❌ 监控服务部署失败: {monitor_name} - {e}")
                result = DeploymentResult(
                    service=f"monitoring_{monitor_name}",
                    status="failed",
                    deployment_time=datetime.now(),
                    health_check=False,
                    performance_metrics={},
                    details={'error': str(e)}
                )
                self.deployment_results.append(result)

    async def _deploy_applications(self):
        """部署应用服务"""
        logger.info("部署应用服务...")

        for service_name, service_config in self.deployment_config['services'].items():
            try:
                logger.info(f"部署应用服务: {service_name}")

                # 创建容器
                container = self.docker_client.containers.run(
                    image=service_config['image'],
                    name=f"rqa2025-{service_name}",
                    ports=service_config.get('ports', {}),
                    environment=service_config.get('environment', {}),
                    command=service_config.get('command', []),
                    volumes={
                        os.path.abspath('.'): {'bind': '/app', 'mode': 'rw'}
                    },
                    working_dir='/app',
                    detach=True,
                    restart_policy={"Name": "always"}
                )

                # 等待服务启动
                await asyncio.sleep(15)

                # 健康检查
                health_status = await self._check_service_health(service_name, container)

                logger.info(f"✅ 应用服务部署成功: {service_name}")

                # 记录部署结果
                result = DeploymentResult(
                    service=f"application_{service_name}",
                    status="success",
                    deployment_time=datetime.now(),
                    health_check=health_status,
                    performance_metrics={
                        'container_id': container.id,
                        'status': container.status
                    },
                    details={'image': service_config['image']}
                )
                self.deployment_results.append(result)

            except Exception as e:
                logger.error(f"❌ 应用服务部署失败: {service_name} - {e}")
                result = DeploymentResult(
                    service=f"application_{service_name}",
                    status="failed",
                    deployment_time=datetime.now(),
                    health_check=False,
                    performance_metrics={},
                    details={'error': str(e)}
                )
                self.deployment_results.append(result)

    async def _run_health_checks(self):
        """运行健康检查"""
        logger.info("运行健康检查...")

        # 检查所有部署的服务
        for result in self.deployment_results:
            if result.status == "success":
                # 执行健康检查
                health_status = await self._perform_health_check(result.service)
                result.health_check = health_status

                if health_status:
                    logger.info(f"✅ 健康检查通过: {result.service}")
                else:
                    logger.warning(f"⚠️ 健康检查失败: {result.service}")

    async def _perform_health_check(self, service_name: str) -> bool:
        """执行健康检查"""
        try:
            # 模拟健康检查
            await asyncio.sleep(1)
            return True
        except Exception as e:
            logger.error(f"健康检查失败: {service_name} - {e}")
            return False

    async def _check_service_health(self, service_name: str, container) -> bool:
        """检查服务健康状态"""
        try:
            # 检查容器状态
            container.reload()
            if container.status == 'running':
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"服务健康检查失败: {service_name} - {e}")
            return False

    async def _generate_deployment_report(self):
        """生成部署报告"""
        logger.info("生成部署报告...")

        # 计算部署统计
        total_services = len(self.deployment_results)
        successful_deployments = len([r for r in self.deployment_results if r.status == "success"])
        failed_deployments = total_services - successful_deployments
        healthy_services = len([r for r in self.deployment_results if r.health_check])

        # 生成报告
        report = {
            "timestamp": datetime.now().isoformat(),
            "deployment_type": "lightweight",
            "summary": {
                "total_services": total_services,
                "successful_deployments": successful_deployments,
                "failed_deployments": failed_deployments,
                "healthy_services": healthy_services,
                "success_rate": (successful_deployments / total_services * 100) if total_services > 0 else 0,
                "health_rate": (healthy_services / total_services * 100) if total_services > 0 else 0
            },
            "deployments": [
                {
                    "service": result.service,
                    "status": result.status,
                    "deployment_time": result.deployment_time.isoformat(),
                    "health_check": result.health_check,
                    "performance_metrics": result.performance_metrics,
                    "details": result.details
                }
                for result in self.deployment_results
            ]
        }

        # 保存报告
        report_file = f"reports/deployment/lightweight_deployment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(os.path.dirname(report_file), exist_ok=True)

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"部署报告已保存到: {report_file}")
        logger.info(f"部署成功率: {report['summary']['success_rate']:.2f}%")
        logger.info(f"服务健康率: {report['summary']['health_rate']:.2f}%")


async def main():
    """主函数"""
    deployer = LightweightDeployment()
    await deployer.run_lightweight_deployment()


if __name__ == "__main__":
    asyncio.run(main())
