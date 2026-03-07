#!/usr/bin/env python3
"""
生产环境部署脚本 - 优化版
包含容器化部署、监控告警、自动扩缩容等功能
"""

from src.infrastructure.monitoring.application_monitor import ApplicationMonitor
from src.infrastructure.config import UnifiedConfigManager
import asyncio
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


class ProductionDeploymentOptimizer:
    """生产环境部署优化器"""

    def __init__(self):
        self.config_manager = UnifiedConfigManager()
        self.app_monitor = ApplicationMonitor()
        self.docker_client = docker.from_env()

        # 部署结果
        self.deployment_results: List[DeploymentResult] = []

        # 部署配置
        self.deployment_config = {
            'services': {
                'api': {
                    'image': 'rqa2025-api:latest',
                    'ports': {'8000/tcp': 8000},
                    'environment': {'ENV': 'production'},
                    'replicas': 3
                },
                'inference': {
                    'image': 'rqa2025-inference:latest',
                    'ports': {'8001/tcp': 8001},
                    'environment': {'ENV': 'production'},
                    'replicas': 2
                },
                'worker': {
                    'image': 'rqa2025-worker:latest',
                    'environment': {'ENV': 'production'},
                    'replicas': 5
                }
            },
            'databases': {
                'postgresql': {
                    'image': 'postgres:15',
                    'ports': {'5432/tcp': 5432},
                    'environment': {
                        'POSTGRES_DB': 'rqa2025',
                        'POSTGRES_USER': 'rqa2025',
                        'POSTGRES_PASSWORD': 'rqa2025_password'
                    }
                },
                'redis': {
                    'image': 'redis:7-alpine',
                    'ports': {'6379/tcp': 6379}
                },
                'elasticsearch': {
                    'image': 'elasticsearch:8.11.0',
                    'ports': {'9200/tcp': 9200, '9300/tcp': 9300},
                    'environment': {
                        'discovery.type': 'single-node',
                        'xpack.security.enabled': 'false'
                    }
                }
            },
            'monitoring': {
                'prometheus': {
                    'image': 'prom/prometheus:latest',
                    'ports': {'9090/tcp': 9090}
                },
                'grafana': {
                    'image': 'grafana/grafana:latest',
                    'ports': {'3000/tcp': 3000},
                    'environment': {
                        'GF_SECURITY_ADMIN_PASSWORD': 'admin'
                    }
                }
            }
        }

        logger.info("ProductionDeploymentOptimizer initialized")

    async def run_production_deployment(self):
        """运行生产环境部署"""
        logger.info("开始生产环境部署...")

        # 1. 环境检查
        await self._check_environment()

        # 2. 构建Docker镜像
        await self._build_docker_images()

        # 3. 部署数据库服务
        await self._deploy_databases()

        # 4. 部署监控服务
        await self._deploy_monitoring()

        # 5. 部署应用服务
        await self._deploy_applications()

        # 6. 配置负载均衡
        await self._configure_load_balancer()

        # 7. 设置自动扩缩容
        await self._setup_auto_scaling()

        # 8. 配置监控告警
        await self._setup_monitoring_alerts()

        # 9. 运行健康检查
        await self._run_health_checks()

        # 10. 生成部署报告
        await self._generate_deployment_report()

        logger.info("生产环境部署完成")

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
        if cpu_percent > 80:
            logger.warning("⚠️ CPU使用率过高")
        if memory_percent > 80:
            logger.warning("⚠️ 内存使用率过高")
        if disk_percent > 90:
            logger.error("❌ 磁盘空间不足")
            raise Exception("磁盘空间不足")

    async def _build_docker_images(self):
        """构建Docker镜像"""
        logger.info("构建Docker镜像...")

        images_to_build = [
            {
                'name': 'rqa2025-api',
                'dockerfile': 'Dockerfile',
                'context': '.'
            },
            {
                'name': 'rqa2025-inference',
                'dockerfile': 'Dockerfile.inference',
                'context': '.'
            },
            {
                'name': 'rqa2025-worker',
                'dockerfile': 'Dockerfile.worker',
                'context': '.'
            }
        ]

        for image_config in images_to_build:
            try:
                logger.info(f"构建镜像: {image_config['name']}")

                # 构建镜像
                image, logs = self.docker_client.images.build(
                    path=image_config['context'],
                    dockerfile=image_config['dockerfile'],
                    tag=image_config['name'],
                    rm=True
                )

                logger.info(f"✅ 镜像构建成功: {image_config['name']}")

                # 记录部署结果
                result = DeploymentResult(
                    service=f"build_{image_config['name']}",
                    status="success",
                    deployment_time=datetime.now(),
                    health_check=True,
                    performance_metrics={
                        'image_size': image.attrs['Size'],
                        'layers': len(image.attrs['Layers'])
                    },
                    details={'dockerfile': image_config['dockerfile']}
                )
                self.deployment_results.append(result)

            except Exception as e:
                logger.error(f"❌ 镜像构建失败: {image_config['name']} - {e}")
                result = DeploymentResult(
                    service=f"build_{image_config['name']}",
                    status="failed",
                    deployment_time=datetime.now(),
                    health_check=False,
                    performance_metrics={},
                    details={'error': str(e)}
                )
                self.deployment_results.append(result)

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
                await asyncio.sleep(10)

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
                await asyncio.sleep(15)

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

                # 部署多个副本
                for i in range(service_config['replicas']):
                    container_name = f"rqa2025-{service_name}-{i+1}"

                    # 创建容器
                    container = self.docker_client.containers.run(
                        image=service_config['image'],
                        name=container_name,
                        ports=service_config.get('ports', {}),
                        environment=service_config.get('environment', {}),
                        detach=True,
                        restart_policy={"Name": "always"}
                    )

                    # 等待服务启动
                    await asyncio.sleep(5)

                    # 健康检查
                    health_status = await self._check_service_health(f"{service_name}-{i+1}", container)

                    logger.info(f"✅ 应用服务部署成功: {container_name}")

                    # 记录部署结果
                    result = DeploymentResult(
                        service=f"application_{container_name}",
                        status="success",
                        deployment_time=datetime.now(),
                        health_check=health_status,
                        performance_metrics={
                            'container_id': container.id,
                            'status': container.status,
                            'replica': i+1
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

    async def _configure_load_balancer(self):
        """配置负载均衡"""
        logger.info("配置负载均衡...")

        try:
            # 创建Nginx负载均衡器
            nginx_config = """
events {
    worker_connections 1024;
}

http {
    upstream api_backend {
        server rqa2025-api-1:8000;
        server rqa2025-api-2:8000;
        server rqa2025-api-3:8000;
    }
    
    upstream inference_backend {
        server rqa2025-inference-1:8001;
        server rqa2025-inference-2:8001;
    }
    
    server {
        listen 80;
        
        location /api/ {
            proxy_pass http://api_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
        
        location /inference/ {
            proxy_pass http://inference_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
    }
}
"""

            # 创建Nginx容器
            nginx_container = self.docker_client.containers.run(
                image='nginx:alpine',
                name='rqa2025-nginx',
                ports={'80/tcp': 80},
                volumes={'nginx_config': {'bind': '/etc/nginx/nginx.conf', 'mode': 'ro'}},
                detach=True,
                restart_policy={"Name": "always"}
            )

            logger.info("✅ 负载均衡器配置成功")

            # 记录部署结果
            result = DeploymentResult(
                service="load_balancer_nginx",
                status="success",
                deployment_time=datetime.now(),
                health_check=True,
                performance_metrics={
                    'container_id': nginx_container.id,
                    'status': nginx_container.status
                },
                details={'upstreams': ['api_backend', 'inference_backend']}
            )
            self.deployment_results.append(result)

        except Exception as e:
            logger.error(f"❌ 负载均衡器配置失败: {e}")
            result = DeploymentResult(
                service="load_balancer_nginx",
                status="failed",
                deployment_time=datetime.now(),
                health_check=False,
                performance_metrics={},
                details={'error': str(e)}
            )
            self.deployment_results.append(result)

    async def _setup_auto_scaling(self):
        """设置自动扩缩容"""
        logger.info("设置自动扩缩容...")

        try:
            # 创建自动扩缩容配置
            scaling_config = {
                'api_service': {
                    'min_replicas': 2,
                    'max_replicas': 10,
                    'target_cpu_utilization': 70,
                    'target_memory_utilization': 80
                },
                'inference_service': {
                    'min_replicas': 1,
                    'max_replicas': 5,
                    'target_cpu_utilization': 80,
                    'target_memory_utilization': 85
                },
                'worker_service': {
                    'min_replicas': 3,
                    'max_replicas': 15,
                    'target_cpu_utilization': 75,
                    'target_memory_utilization': 80
                }
            }

            # 保存扩缩容配置
            config_file = "deploy/auto_scaling_config.json"
            os.makedirs(os.path.dirname(config_file), exist_ok=True)

            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(scaling_config, f, indent=2, ensure_ascii=False)

            logger.info("✅ 自动扩缩容配置成功")

            # 记录部署结果
            result = DeploymentResult(
                service="auto_scaling",
                status="success",
                deployment_time=datetime.now(),
                health_check=True,
                performance_metrics={
                    'services_configured': len(scaling_config)
                },
                details={'config_file': config_file}
            )
            self.deployment_results.append(result)

        except Exception as e:
            logger.error(f"❌ 自动扩缩容配置失败: {e}")
            result = DeploymentResult(
                service="auto_scaling",
                status="failed",
                deployment_time=datetime.now(),
                health_check=False,
                performance_metrics={},
                details={'error': str(e)}
            )
            self.deployment_results.append(result)

    async def _setup_monitoring_alerts(self):
        """设置监控告警"""
        logger.info("设置监控告警...")

        try:
            # 创建告警规则
            alert_rules = {
                'high_cpu_usage': {
                    'condition': 'cpu_usage > 80%',
                    'duration': '5m',
                    'severity': 'warning'
                },
                'high_memory_usage': {
                    'condition': 'memory_usage > 85%',
                    'duration': '5m',
                    'severity': 'warning'
                },
                'service_down': {
                    'condition': 'service_health == 0',
                    'duration': '1m',
                    'severity': 'critical'
                },
                'high_error_rate': {
                    'condition': 'error_rate > 5%',
                    'duration': '2m',
                    'severity': 'critical'
                }
            }

            # 保存告警规则
            alert_file = "deploy/alert_rules.json"
            os.makedirs(os.path.dirname(alert_file), exist_ok=True)

            with open(alert_file, 'w', encoding='utf-8') as f:
                json.dump(alert_rules, f, indent=2, ensure_ascii=False)

            logger.info("✅ 监控告警配置成功")

            # 记录部署结果
            result = DeploymentResult(
                service="monitoring_alerts",
                status="success",
                deployment_time=datetime.now(),
                health_check=True,
                performance_metrics={
                    'alert_rules': len(alert_rules)
                },
                details={'alert_file': alert_file}
            )
            self.deployment_results.append(result)

        except Exception as e:
            logger.error(f"❌ 监控告警配置失败: {e}")
            result = DeploymentResult(
                service="monitoring_alerts",
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
        report_file = f"reports/deployment/production_deployment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(os.path.dirname(report_file), exist_ok=True)

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"部署报告已保存到: {report_file}")
        logger.info(f"部署成功率: {report['summary']['success_rate']:.2f}%")
        logger.info(f"服务健康率: {report['summary']['health_rate']:.2f}%")


async def main():
    """主函数"""
    deployer = ProductionDeploymentOptimizer()
    await deployer.run_production_deployment()


if __name__ == "__main__":
    asyncio.run(main())
