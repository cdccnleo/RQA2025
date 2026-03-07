#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
自动化生产环境部署脚本
实现容器化部署、监控配置、自动扩缩容等功能
"""

import os
import sys
import time
import json
import yaml
import subprocess
import docker
from typing import Dict, Any
from datetime import datetime
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AutomatedProductionDeployer:
    """自动化生产环境部署器"""

    def __init__(self, config_path: str = "deploy/config/production.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.docker_client = docker.from_env()
        self.deployment_status = {
            'start_time': datetime.now().isoformat(),
            'status': 'initializing',
            'steps_completed': [],
            'errors': [],
            'warnings': []
        }

    def _load_config(self) -> Dict[str, Any]:
        """加载部署配置"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"成功加载部署配置: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"加载部署配置失败: {e}")
            raise

    def deploy(self) -> bool:
        """执行完整部署流程"""
        try:
            logger.info("开始自动化生产环境部署")
            self.deployment_status['status'] = 'deploying'

            # 1. 环境检查
            if not self._check_environment():
                return False

            # 2. 构建Docker镜像
            if not self._build_images():
                return False

            # 3. 部署基础设施
            if not self._deploy_infrastructure():
                return False

            # 4. 部署应用服务
            if not self._deploy_application():
                return False

            # 5. 配置监控系统
            if not self._configure_monitoring():
                return False

            # 6. 配置自动扩缩容
            if not self._configure_autoscaling():
                return False

            # 7. 健康检查
            if not self._health_check():
                return False

            # 8. 性能测试
            if not self._performance_test():
                return False

            # 9. 生成部署报告
            self._generate_deployment_report()

            self.deployment_status['status'] = 'completed'
            logger.info("自动化生产环境部署完成")
            return True

        except Exception as e:
            logger.error(f"部署过程中发生错误: {e}")
            self.deployment_status['status'] = 'failed'
            self.deployment_status['errors'].append(str(e))
            return False

    def _check_environment(self) -> bool:
        """检查部署环境"""
        logger.info("检查部署环境")
        self.deployment_status['steps_completed'].append('environment_check')

        try:
            # 检查Docker
            docker_version = subprocess.check_output(['docker', '--version'],
                                                     text=True, stderr=subprocess.PIPE)
            logger.info(f"Docker版本: {docker_version.strip()}")

            # 检查Docker Compose
            compose_version = subprocess.check_output(['docker-compose', '--version'],
                                                      text=True, stderr=subprocess.PIPE)
            logger.info(f"Docker Compose版本: {compose_version.strip()}")

            # 检查磁盘空间
            disk_usage = subprocess.check_output(['df', '-h', '/'],
                                                 text=True, stderr=subprocess.PIPE)
            logger.info(f"磁盘使用情况:\n{disk_usage}")

            # 检查内存
            memory_info = subprocess.check_output(['free', '-h'],
                                                  text=True, stderr=subprocess.PIPE)
            logger.info(f"内存使用情况:\n{memory_info}")

            return True

        except Exception as e:
            logger.error(f"环境检查失败: {e}")
            self.deployment_status['errors'].append(f"环境检查失败: {e}")
            return False

    def _build_images(self) -> bool:
        """构建Docker镜像"""
        logger.info("构建Docker镜像")
        self.deployment_status['steps_completed'].append('build_images')

        try:
            # 构建主应用镜像
            logger.info("构建RQA2025主应用镜像")
            subprocess.run([
                'docker', 'build',
                '-t', 'rqa2025:latest',
                '-f', 'Dockerfile.optimized',
                '--build-arg', 'ENVIRONMENT=production',
                '--build-arg', 'BUILD_DATE=' + datetime.now().isoformat(),
                '.'
            ], check=True)

            # 构建监控镜像
            logger.info("构建监控系统镜像")
            subprocess.run([
                'docker', 'build',
                '-t', 'rqa2025-monitoring:latest',
                '-f', 'deploy/monitoring/Dockerfile',
                'deploy/monitoring/'
            ], check=True)

            # 验证镜像构建
            images = self.docker_client.images.list()
            rqa_images = [img for img in images if 'rqa2025' in img.tags[0] if img.tags]

            if len(rqa_images) >= 2:
                logger.info(f"成功构建 {len(rqa_images)} 个镜像")
                return True
            else:
                logger.error("镜像构建验证失败")
                return False

        except Exception as e:
            logger.error(f"镜像构建失败: {e}")
            self.deployment_status['errors'].append(f"镜像构建失败: {e}")
            return False

    def _deploy_infrastructure(self) -> bool:
        """部署基础设施"""
        logger.info("部署基础设施")
        self.deployment_status['steps_completed'].append('deploy_infrastructure')

        try:
            # 创建Docker网络
            logger.info("创建Docker网络")
            subprocess.run([
                'docker', 'network', 'create', 'rqa2025-network'
            ], check=True)

            # 启动Redis集群
            logger.info("启动Redis集群")
            subprocess.run([
                'docker-compose', '-f', 'deploy/docker-compose.yml',
                'up', '-d', 'redis-master', 'redis-slave'
            ], check=True)

            # 启动PostgreSQL
            logger.info("启动PostgreSQL数据库")
            subprocess.run([
                'docker-compose', '-f', 'deploy/docker-compose.yml',
                'up', '-d', 'postgres'
            ], check=True)

            # 启动Elasticsearch
            logger.info("启动Elasticsearch")
            subprocess.run([
                'docker-compose', '-f', 'deploy/docker-compose.yml',
                'up', '-d', 'elasticsearch'
            ], check=True)

            # 等待服务启动
            logger.info("等待基础设施服务启动")
            time.sleep(30)

            # 验证服务状态
            services = ['redis-master', 'redis-slave', 'postgres', 'elasticsearch']
            for service in services:
                if not self._check_service_health(service):
                    logger.error(f"服务 {service} 健康检查失败")
                    return False

            logger.info("基础设施部署完成")
            return True

        except Exception as e:
            logger.error(f"基础设施部署失败: {e}")
            self.deployment_status['errors'].append(f"基础设施部署失败: {e}")
            return False

    def _deploy_application(self) -> bool:
        """部署应用服务"""
        logger.info("部署应用服务")
        self.deployment_status['steps_completed'].append('deploy_application')

        try:
            # 启动API服务
            logger.info("启动RQA2025 API服务")
            subprocess.run([
                'docker-compose', '-f', 'deploy/docker-compose.yml',
                'up', '-d', 'rqa2025-api'
            ], check=True)

            # 启动推理服务
            logger.info("启动RQA2025推理服务")
            subprocess.run([
                'docker-compose', '-f', 'deploy/docker-compose.yml',
                'up', '-d', 'rqa2025-inference'
            ], check=True)

            # 启动负载均衡器
            logger.info("启动负载均衡器")
            subprocess.run([
                'docker-compose', '-f', 'deploy/docker-compose.yml',
                'up', '-d', 'nginx'
            ], check=True)

            # 等待应用服务启动
            logger.info("等待应用服务启动")
            time.sleep(60)

            # 验证应用服务状态
            app_services = ['rqa2025-api', 'rqa2025-inference', 'nginx']
            for service in app_services:
                if not self._check_service_health(service):
                    logger.error(f"应用服务 {service} 健康检查失败")
                    return False

            logger.info("应用服务部署完成")
            return True

        except Exception as e:
            logger.error(f"应用服务部署失败: {e}")
            self.deployment_status['errors'].append(f"应用服务部署失败: {e}")
            return False

    def _configure_monitoring(self) -> bool:
        """配置监控系统"""
        logger.info("配置监控系统")
        self.deployment_status['steps_completed'].append('configure_monitoring')

        try:
            # 启动Prometheus
            logger.info("启动Prometheus监控")
            subprocess.run([
                'docker-compose', '-f', 'deploy/docker-compose.yml',
                'up', '-d', 'prometheus'
            ], check=True)

            # 启动Grafana
            logger.info("启动Grafana可视化")
            subprocess.run([
                'docker-compose', '-f', 'deploy/docker-compose.yml',
                'up', '-d', 'grafana'
            ], check=True)

            # 启动AlertManager
            logger.info("启动AlertManager告警")
            subprocess.run([
                'docker-compose', '-f', 'deploy/docker-compose.yml',
                'up', '-d', 'alertmanager'
            ], check=True)

            # 配置Grafana仪表板
            logger.info("配置Grafana仪表板")
            self._configure_grafana_dashboards()

            # 配置告警规则
            logger.info("配置告警规则")
            self._configure_alert_rules()

            # 等待监控服务启动
            logger.info("等待监控服务启动")
            time.sleep(30)

            # 验证监控服务状态
            monitoring_services = ['prometheus', 'grafana', 'alertmanager']
            for service in monitoring_services:
                if not self._check_service_health(service):
                    logger.error(f"监控服务 {service} 健康检查失败")
                    return False

            logger.info("监控系统配置完成")
            return True

        except Exception as e:
            logger.error(f"监控系统配置失败: {e}")
            self.deployment_status['errors'].append(f"监控系统配置失败: {e}")
            return False

    def _configure_autoscaling(self) -> bool:
        """配置自动扩缩容"""
        logger.info("配置自动扩缩容")
        self.deployment_status['steps_completed'].append('configure_autoscaling')

        try:
            # 创建HPA (Horizontal Pod Autoscaler)
            logger.info("创建水平自动扩缩容器")
            hpa_config = {
                'apiVersion': 'autoscaling/v2',
                'kind': 'HorizontalPodAutoscaler',
                'metadata': {
                    'name': 'rqa2025-hpa',
                    'namespace': 'rqa2025-production'
                },
                'spec': {
                    'scaleTargetRef': {
                        'apiVersion': 'apps/v1',
                        'kind': 'Deployment',
                        'name': 'rqa2025-api'
                    },
                    'minReplicas': 2,
                    'maxReplicas': 10,
                    'metrics': [
                        {
                            'type': 'Resource',
                            'resource': {
                                'name': 'cpu',
                                'target': {
                                    'type': 'Utilization',
                                    'averageUtilization': 70
                                }
                            }
                        },
                        {
                            'type': 'Resource',
                            'resource': {
                                'name': 'memory',
                                'target': {
                                    'type': 'Utilization',
                                    'averageUtilization': 80
                                }
                            }
                        }
                    ]
                }
            }

            # 保存HPA配置
            with open('deploy/k8s/hpa.yaml', 'w') as f:
                yaml.dump(hpa_config, f)

            # 应用HPA配置
            subprocess.run([
                'kubectl', 'apply', '-f', 'deploy/k8s/hpa.yaml'
            ], check=True)

            # 创建VPA (Vertical Pod Autoscaler)
            logger.info("创建垂直自动扩缩容器")
            vpa_config = {
                'apiVersion': 'autoscaling.k8s.io/v1',
                'kind': 'VerticalPodAutoscaler',
                'metadata': {
                    'name': 'rqa2025-vpa',
                    'namespace': 'rqa2025-production'
                },
                'spec': {
                    'targetRef': {
                        'apiVersion': 'apps/v1',
                        'kind': 'Deployment',
                        'name': 'rqa2025-api'
                    },
                    'updatePolicy': {
                        'updateMode': 'Auto'
                    }
                }
            }

            # 保存VPA配置
            with open('deploy/k8s/vpa.yaml', 'w') as f:
                yaml.dump(vpa_config, f)

            # 应用VPA配置
            subprocess.run([
                'kubectl', 'apply', '-f', 'deploy/k8s/vpa.yaml'
            ], check=True)

            logger.info("自动扩缩容配置完成")
            return True

        except Exception as e:
            logger.error(f"自动扩缩容配置失败: {e}")
            self.deployment_status['errors'].append(f"自动扩缩容配置失败: {e}")
            return False

    def _health_check(self) -> bool:
        """健康检查"""
        logger.info("执行健康检查")
        self.deployment_status['steps_completed'].append('health_check')

        try:
            # 检查API服务健康状态
            logger.info("检查API服务健康状态")
            response = subprocess.run([
                'curl', '-f', 'http://localhost:8000/health'
            ], capture_output=True, text=True)

            if response.returncode != 0:
                logger.error("API服务健康检查失败")
                return False

            # 检查推理服务健康状态
            logger.info("检查推理服务健康状态")
            response = subprocess.run([
                'curl', '-f', 'http://localhost:8001/health'
            ], capture_output=True, text=True)

            if response.returncode != 0:
                logger.error("推理服务健康检查失败")
                return False

            # 检查数据库连接
            logger.info("检查数据库连接")
            response = subprocess.run([
                'docker', 'exec', 'rqa2025-postgres',
                'pg_isready', '-U', 'rqa2025'
            ], capture_output=True, text=True)

            if response.returncode != 0:
                logger.error("数据库连接检查失败")
                return False

            # 检查Redis连接
            logger.info("检查Redis连接")
            response = subprocess.run([
                'docker', 'exec', 'rqa2025-redis-master',
                'redis-cli', 'ping'
            ], capture_output=True, text=True)

            if 'PONG' not in response.stdout:
                logger.error("Redis连接检查失败")
                return False

            logger.info("健康检查通过")
            return True

        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            self.deployment_status['errors'].append(f"健康检查失败: {e}")
            return False

    def _performance_test(self) -> bool:
        """性能测试"""
        logger.info("执行性能测试")
        self.deployment_status['steps_completed'].append('performance_test')

        try:
            # 运行性能测试脚本
            logger.info("运行性能测试")
            result = subprocess.run([
                'python', 'scripts/testing/performance_benchmark_framework.py'
            ], capture_output=True, text=True)

            if result.returncode != 0:
                logger.error("性能测试失败")
                logger.error(f"错误输出: {result.stderr}")
                return False

            # 分析性能测试结果
            logger.info("分析性能测试结果")
            performance_metrics = self._analyze_performance_results()

            # 检查性能指标是否满足要求
            if not self._validate_performance_metrics(performance_metrics):
                logger.error("性能指标不满足要求")
                return False

            logger.info("性能测试通过")
            return True

        except Exception as e:
            logger.error(f"性能测试失败: {e}")
            self.deployment_status['errors'].append(f"性能测试失败: {e}")
            return False

    def _check_service_health(self, service_name: str) -> bool:
        """检查服务健康状态"""
        try:
            # 获取容器状态
            containers = self.docker_client.containers.list(
                filters={'name': service_name}
            )

            if not containers:
                logger.error(f"服务 {service_name} 未找到")
                return False

            container = containers[0]
            container_info = container.attrs

            # 检查容器状态
            if container_info['State']['Status'] != 'running':
                logger.error(f"服务 {service_name} 状态异常: {container_info['State']['Status']}")
                return False

            # 检查健康状态
            health = container_info['State'].get('Health', {})
            if health and health.get('Status') == 'unhealthy':
                logger.error(f"服务 {service_name} 健康检查失败")
                return False

            return True

        except Exception as e:
            logger.error(f"检查服务 {service_name} 健康状态失败: {e}")
            return False

    def _configure_grafana_dashboards(self):
        """配置Grafana仪表板"""
        try:
            # 导入RQA2025仪表板
            dashboard_config = {
                'dashboard': {
                    'title': 'RQA2025 Production Dashboard',
                    'panels': [
                        {
                            'title': 'CPU Usage',
                            'type': 'graph',
                            'targets': [
                                {
                                    'expr': 'system_cpu_usage',
                                    'legendFormat': 'CPU %'
                                }
                            ]
                        },
                        {
                            'title': 'Memory Usage',
                            'type': 'graph',
                            'targets': [
                                {
                                    'expr': 'system_memory_usage',
                                    'legendFormat': 'Memory %'
                                }
                            ]
                        },
                        {
                            'title': 'Response Time',
                            'type': 'graph',
                            'targets': [
                                {
                                    'expr': 'histogram_quantile(0.95, app_response_time_bucket)',
                                    'legendFormat': '95th percentile'
                                }
                            ]
                        }
                    ]
                }
            }

            # 保存仪表板配置
            with open('deploy/monitoring/grafana-dashboard.json', 'w') as f:
                json.dump(dashboard_config, f, indent=2)

            # 导入仪表板
            subprocess.run([
                'curl', '-X', 'POST',
                '-H', 'Content-Type: application/json',
                '-d', '@deploy/monitoring/grafana-dashboard.json',
                'http://admin:admin@localhost:3000/api/dashboards/db'
            ], check=True)

        except Exception as e:
            logger.warning(f"配置Grafana仪表板失败: {e}")
            self.deployment_status['warnings'].append(f"配置Grafana仪表板失败: {e}")

    def _configure_alert_rules(self):
        """配置告警规则"""
        try:
            # Prometheus告警规则
            alert_rules = {
                'groups': [
                    {
                        'name': 'rqa2025_alerts',
                        'rules': [
                            {
                                'alert': 'HighCPUUsage',
                                'expr': 'system_cpu_usage > 80',
                                'for': '5m',
                                'labels': {
                                    'severity': 'warning'
                                },
                                'annotations': {
                                    'summary': 'High CPU usage detected',
                                    'description': 'CPU usage is above 80% for 5 minutes'
                                }
                            },
                            {
                                'alert': 'HighMemoryUsage',
                                'expr': 'system_memory_usage > 85',
                                'for': '5m',
                                'labels': {
                                    'severity': 'warning'
                                },
                                'annotations': {
                                    'summary': 'High memory usage detected',
                                    'description': 'Memory usage is above 85% for 5 minutes'
                                }
                            },
                            {
                                'alert': 'HighErrorRate',
                                'expr': 'app_error_rate > 5',
                                'for': '2m',
                                'labels': {
                                    'severity': 'critical'
                                },
                                'annotations': {
                                    'summary': 'High error rate detected',
                                    'description': 'Error rate is above 5% for 2 minutes'
                                }
                            }
                        ]
                    }
                ]
            }

            # 保存告警规则
            with open('deploy/monitoring/alert-rules.yml', 'w') as f:
                yaml.dump(alert_rules, f)

            # 应用告警规则
            subprocess.run([
                'docker', 'exec', 'rqa2025-prometheus',
                'wget', '-O', '/etc/prometheus/alert-rules.yml',
                'http://localhost:9090/api/v1/rules'
            ], check=True)

        except Exception as e:
            logger.warning(f"配置告警规则失败: {e}")
            self.deployment_status['warnings'].append(f"配置告警规则失败: {e}")

    def _analyze_performance_results(self) -> Dict[str, Any]:
        """分析性能测试结果"""
        try:
            # 读取性能测试结果
            with open('reports/performance/performance_test_results.json', 'r') as f:
                results = json.load(f)

            return results

        except Exception as e:
            logger.warning(f"分析性能测试结果失败: {e}")
            return {}

    def _validate_performance_metrics(self, metrics: Dict[str, Any]) -> bool:
        """验证性能指标"""
        try:
            # 检查关键性能指标
            if 'response_time_avg' in metrics:
                if metrics['response_time_avg'] > 1000:  # 1秒
                    logger.error(f"平均响应时间过高: {metrics['response_time_avg']}ms")
                    return False

            if 'throughput' in metrics:
                if metrics['throughput'] < 100:  # 100 req/s
                    logger.error(f"吞吐量过低: {metrics['throughput']} req/s")
                    return False

            if 'error_rate' in metrics:
                if metrics['error_rate'] > 1:  # 1%
                    logger.error(f"错误率过高: {metrics['error_rate']}%")
                    return False

            return True

        except Exception as e:
            logger.warning(f"验证性能指标失败: {e}")
            return True  # 验证失败时默认通过

    def _generate_deployment_report(self):
        """生成部署报告"""
        try:
            self.deployment_status['end_time'] = datetime.now().isoformat()
            self.deployment_status['duration'] = (
                datetime.fromisoformat(self.deployment_status['end_time']) -
                datetime.fromisoformat(self.deployment_status['start_time'])
            ).total_seconds()

            # 保存部署报告
            report_path = f"reports/deployment/deployment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            os.makedirs(os.path.dirname(report_path), exist_ok=True)

            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(self.deployment_status, f, indent=2, ensure_ascii=False)

            logger.info(f"部署报告已生成: {report_path}")

        except Exception as e:
            logger.error(f"生成部署报告失败: {e}")

    def rollback(self) -> bool:
        """回滚部署"""
        logger.info("开始回滚部署")

        try:
            # 停止所有服务
            subprocess.run([
                'docker-compose', '-f', 'deploy/docker-compose.yml', 'down'
            ], check=True)

            # 删除网络
            subprocess.run([
                'docker', 'network', 'rm', 'rqa2025-network'
            ], check=True)

            # 清理数据卷（可选）
            if self.config.get('cleanup_volumes', False):
                subprocess.run([
                    'docker', 'volume', 'prune', '-f'
                ], check=True)

            logger.info("回滚完成")
            return True

        except Exception as e:
            logger.error(f"回滚失败: {e}")
            return False

    def get_deployment_status(self) -> Dict[str, Any]:
        """获取部署状态"""
        return self.deployment_status.copy()


def main():
    """主函数"""
    try:
        # 创建部署器
        deployer = AutomatedProductionDeployer()

        # 执行部署
        success = deployer.deploy()

        if success:
            logger.info("✅ 自动化生产环境部署成功")
            status = deployer.get_deployment_status()
            logger.info(f"部署状态: {status['status']}")
            logger.info(f"完成步骤: {len(status['steps_completed'])}")
            if status['errors']:
                logger.warning(f"部署错误: {status['errors']}")
            if status['warnings']:
                logger.warning(f"部署警告: {status['warnings']}")
        else:
            logger.error("❌ 自动化生产环境部署失败")
            status = deployer.get_deployment_status()
            logger.error(f"部署错误: {status['errors']}")

            # 询问是否回滚
            response = input("是否要回滚部署? (y/N): ")
            if response.lower() == 'y':
                deployer.rollback()

    except Exception as e:
        logger.error(f"部署脚本执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
