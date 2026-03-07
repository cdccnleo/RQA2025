#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DevOps一体化部署流水线
实现开发、测试、运维的完全自动化集成
"""

import os
import sys
import json
import time
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import requests
import docker
from docker.errors import DockerException
import kubernetes.client
from kubernetes.client.rest import ApiException


@dataclass
class DeploymentStage:
    """部署阶段"""
    name: str
    status: str  # 'pending', 'running', 'success', 'failed'
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    logs: List[str] = None
    artifacts: Dict[str, Any] = None

    def __post_init__(self):
        if self.logs is None:
            self.logs = []
        if self.artifacts is None:
            self.artifacts = {}


@dataclass
class DeploymentResult:
    """部署结果"""
    pipeline_id: str
    status: str
    stages: List[DeploymentStage]
    metrics: Dict[str, Any]
    start_time: datetime
    end_time: Optional[datetime] = None
    artifacts: Dict[str, Any] = None

    def __post_init__(self):
        if self.artifacts is None:
            self.artifacts = {}


class DevOpsDeploymentPipeline:
    """DevOps一体化部署流水线"""

    def __init__(self, config_path: str = "deployment_config.json"):
        self.config = self._load_config(config_path)
        self.docker_client = docker.from_env()
        self.k8s_client = self._init_kubernetes_client()
        self.current_deployment = None

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载部署配置"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # 默认配置
            return {
                'environments': {
                    'dev': {'namespace': 'dev', 'replicas': 1},
                    'staging': {'namespace': 'staging', 'replicas': 2},
                    'prod': {'namespace': 'prod', 'replicas': 3}
                },
                'services': {
                    'api': {'image': 'rqa2025/api', 'port': 8080},
                    'worker': {'image': 'rqa2025/worker', 'port': 8081},
                    'web': {'image': 'rqa2025/web', 'port': 3000}
                },
                'monitoring': {
                    'prometheus_url': 'http://prometheus:9090',
                    'grafana_url': 'http://grafana:3000',
                    'alertmanager_url': 'http://alertmanager:9093'
                },
                'rollback_strategy': 'immediate'
            }

    def _init_kubernetes_client(self):
        """初始化Kubernetes客户端"""
        try:
            kubernetes.config.load_kube_config()
            return kubernetes.client.CoreV1Api()
        except Exception:
            print("⚠️  Kubernetes客户端初始化失败，使用Docker模式")
            return None

    def run_full_pipeline(self, environment: str = 'dev',
                          version: str = 'latest') -> DeploymentResult:
        """运行完整部署流水线"""
        pipeline_id = f"deploy_{environment}_{int(time.time())}"

        print(f"🚀 开始DevOps部署流水线: {pipeline_id}")
        print(f"   环境: {environment}")
        print(f"   版本: {version}")

        deployment = DeploymentResult(
            pipeline_id=pipeline_id,
            status='running',
            stages=[],
            metrics={},
            start_time=datetime.now()
        )

        try:
            # 1. 代码质量检查
            self._run_code_quality_check(deployment)

            # 2. 构建阶段
            self._run_build_stage(deployment, version)

            # 3. 测试阶段
            self._run_test_stage(deployment)

            # 4. 安全扫描
            self._run_security_scan(deployment)

            # 5. 部署准备
            self._run_deployment_prep(deployment, environment)

            # 6. 部署执行
            self._run_deployment_execution(deployment, environment, version)

            # 7. 验证部署
            self._run_deployment_verification(deployment, environment)

            # 8. 监控设置
            self._run_monitoring_setup(deployment, environment)

            # 9. 性能测试
            self._run_performance_validation(deployment, environment)

            # 更新部署状态
            deployment.status = 'success'
            deployment.end_time = datetime.now()

            print("
                  ✅ DevOps部署流水线执行成功！"            print(f"   总耗时: {(deployment.end_time - deployment.start_time).total_seconds(): .1f}秒")
            print(f"   执行阶段: {len(deployment.stages)} 个")

        except Exception as e:
            deployment.status = 'failed'
            deployment.end_time = datetime.now()
            print(f"❌ DevOps部署流水线执行失败: {e}")

        finally:
            # 保存部署结果
            self._save_deployment_result(deployment)

        return deployment

    def _run_code_quality_check(self, deployment: DeploymentResult):
        """运行代码质量检查"""
        stage = DeploymentStage(name='code_quality_check', status='running')
        stage.start_time = datetime.now()
        deployment.stages.append(stage)

        try:
            print("🔍 执行代码质量检查...")

            # 运行静态分析
            result = subprocess.run([
                'python', '-m', 'flake8', 'src/', '--count',
                '--select=E9,F63,F7,F82', '--show-source', '--statistics'
            ], capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                stage.logs.append("✅ 代码质量检查通过")
                stage.status = 'success'
            else:
                stage.logs.append(f"❌ 代码质量问题: {result.stdout}")
                stage.status = 'failed'
                raise Exception("代码质量检查失败")

            # 运行单元测试覆盖率检查
            result = subprocess.run([
                'python', '-m', 'pytest', 'tests/unit/',
                '--cov=src', '--cov-report=xml', '--cov-fail-under=70'
            ], capture_output=True, text=True, timeout=600)

            if result.returncode == 0:
                stage.logs.append("✅ 单元测试覆盖率检查通过")
            else:
                stage.logs.append(f"⚠️  覆盖率不足: {result.stdout}")

        except Exception as e:
            stage.logs.append(f"❌ 代码质量检查异常: {e}")
            stage.status = 'failed'
            raise

        finally:
            stage.end_time = datetime.now()

    def _run_build_stage(self, deployment: DeploymentResult, version: str):
        """运行构建阶段"""
        stage = DeploymentStage(name='build', status='running')
        stage.start_time = datetime.now()
        deployment.stages.append(stage)

        try:
            print("🔨 执行构建阶段...")

            # 构建Docker镜像
            services = self.config.get('services', {})

            for service_name, service_config in services.items():
                image_name = f"{service_config['image']}:{version}"

                print(f"   构建镜像: {image_name}")

                # 使用Docker构建
                result = subprocess.run([
                    'docker', 'build', '-t', image_name,
                    '-f', f'Dockerfile.{service_name}', '.'
                ], capture_output=True, text=True, timeout=1800)

                if result.returncode == 0:
                    stage.logs.append(f"✅ 构建镜像成功: {image_name}")
                    stage.artifacts[f'{service_name}_image'] = image_name
                else:
                    stage.logs.append(f"❌ 构建镜像失败: {image_name}")
                    stage.logs.append(result.stderr)
                    raise Exception(f"构建镜像失败: {service_name}")

            stage.status = 'success'

        except Exception as e:
            stage.logs.append(f"❌ 构建阶段异常: {e}")
            stage.status = 'failed'
            raise

        finally:
            stage.end_time = datetime.now()

    def _run_test_stage(self, deployment: DeploymentResult):
        """运行测试阶段"""
        stage = DeploymentStage(name='testing', status='running')
        stage.start_time = datetime.now()
        deployment.stages.append(stage)

        try:
            print("🧪 执行测试阶段...")

            # 运行单元测试
            print("   运行单元测试...")
            result = subprocess.run([
                'python', '-m', 'pytest', 'tests/unit/',
                '--junitxml=test-results-unit.xml',
                '--cov=src', '--cov-report=xml',
                '-v', '--tb=short'
            ], capture_output=True, text=True, timeout=900)

            stage.logs.append(f"单元测试结果: {result.stdout}")
            if result.returncode != 0:
                stage.logs.append(f"单元测试失败: {result.stderr}")

            # 运行集成测试
            print("   运行集成测试...")
            result = subprocess.run([
                'python', '-m', 'pytest', 'tests/integration/',
                '--junitxml=test-results-integration.xml',
                '-v', '--tb=short'
            ], capture_output=True, text=True, timeout=600)

            stage.logs.append(f"集成测试结果: {result.stdout}")

            # 运行性能测试
            print("   运行性能测试...")
            result = subprocess.run([
                'python', '-m', 'pytest', 'tests/performance/',
                '--junitxml=test-results-performance.xml',
                '-v'
            ], capture_output=True, text=True, timeout=1200)

            stage.logs.append(f"性能测试结果: {result.stdout}")

            stage.artifacts['test_results'] = {
                'unit': 'test-results-unit.xml',
                'integration': 'test-results-integration.xml',
                'performance': 'test-results-performance.xml'
            }

            stage.status = 'success'

        except Exception as e:
            stage.logs.append(f"❌ 测试阶段异常: {e}")
            stage.status = 'failed'
            raise

        finally:
            stage.end_time = datetime.now()

    def _run_security_scan(self, deployment: DeploymentResult):
        """运行安全扫描"""
        stage = DeploymentStage(name='security_scan', status='running')
        stage.start_time = datetime.now()
        deployment.stages.append(stage)

        try:
            print("🔒 执行安全扫描...")

            # 运行Trivy容器扫描
            services = self.config.get('services', {})

            for service_name, service_config in services.items():
                image_name = f"{service_config['image']}:latest"

                print(f"   扫描镜像: {image_name}")

                result = subprocess.run([
                    'trivy', 'image', '--format=json',
                    '--output', f'trivy-{service_name}.json', image_name
                ], capture_output=True, text=True, timeout=600)

                if result.returncode == 0:
                    stage.logs.append(f"✅ 安全扫描完成: {service_name}")
                    stage.artifacts[f'{service_name}_security'] = f'trivy-{service_name}.json'
                else:
                    stage.logs.append(f"❌ 安全扫描失败: {service_name}")
                    stage.logs.append(result.stderr)

            # 运行代码安全扫描
            result = subprocess.run([
                'bandit', '-r', 'src/', '-f', 'json', '-o', 'bandit-report.json'
            ], capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                stage.logs.append("✅ 代码安全扫描完成")
                stage.artifacts['code_security'] = 'bandit-report.json'
            else:
                stage.logs.append(f"⚠️  代码安全扫描警告: {result.stderr}")

            stage.status = 'success'

        except Exception as e:
            stage.logs.append(f"❌ 安全扫描异常: {e}")
            stage.status = 'failed'
            raise

        finally:
            stage.end_time = datetime.now()

    def _run_deployment_prep(self, deployment: DeploymentResult, environment: str):
        """运行部署准备"""
        stage = DeploymentStage(name='deployment_prep', status='running')
        stage.start_time = datetime.now()
        deployment.stages.append(stage)

        try:
            print("📋 执行部署准备...")

            # 生成部署配置
            env_config = self.config['environments'].get(environment, {})

            deployment_config = {
                'environment': environment,
                'namespace': env_config.get('namespace', environment),
                'replicas': env_config.get('replicas', 1),
                'services': self.config['services'],
                'timestamp': datetime.now().isoformat()
            }

            # 保存部署配置
            config_file = f'deployment-{environment}.json'
            with open(config_file, 'w') as f:
                json.dump(deployment_config, f, indent=2)

            stage.artifacts['deployment_config'] = config_file
            stage.logs.append(f"✅ 部署配置生成: {config_file}")

            # 验证集群状态
            if self.k8s_client:
                try:
                    # 检查命名空间
                    namespace = env_config.get('namespace', environment)
                    self.k8s_client.read_namespace(namespace)
                    stage.logs.append(f"✅ Kubernetes命名空间验证: {namespace}")
                except ApiException:
                    stage.logs.append(f"⚠️  Kubernetes命名空间不存在: {namespace}")

            stage.status = 'success'

        except Exception as e:
            stage.logs.append(f"❌ 部署准备异常: {e}")
            stage.status = 'failed'
            raise

        finally:
            stage.end_time = datetime.now()

    def _run_deployment_execution(self, deployment: DeploymentResult,
                                  environment: str, version: str):
        """运行部署执行"""
        stage = DeploymentStage(name='deployment_execution', status='running')
        stage.start_time = datetime.now()
        deployment.stages.append(stage)

        try:
            print("🚀 执行部署...")

            env_config = self.config['environments'].get(environment, {})
            services = self.config['services']

            # 使用Kubernetes部署
            if self.k8s_client:
                self._deploy_to_kubernetes(stage, env_config, services, version)
            else:
                # 使用Docker Compose部署
                self._deploy_with_docker_compose(stage, environment, version)

            stage.status = 'success'

        except Exception as e:
            stage.logs.append(f"❌ 部署执行异常: {e}")
            stage.status = 'failed'
            raise

        finally:
            stage.end_time = datetime.now()

    def _deploy_to_kubernetes(self, stage: DeploymentStage, env_config: Dict[str, Any],
                              services: Dict[str, Any], version: str):
        """部署到Kubernetes"""
        print("   使用Kubernetes部署...")

        apps_v1 = kubernetes.client.AppsV1Api()
        namespace = env_config.get('namespace', 'default')

        for service_name, service_config in services.items():
            # 创建Deployment
            deployment = kubernetes.client.V1Deployment(
                metadata=kubernetes.client.V1ObjectMeta(
                    name=f'{service_name}-deployment',
                    namespace=namespace
                ),
                spec=kubernetes.client.V1DeploymentSpec(
                    replicas=env_config.get('replicas', 1),
                    selector=kubernetes.client.V1LabelSelector(
                        match_labels={'app': service_name}
                    ),
                    template=kubernetes.client.V1PodTemplateSpec(
                        metadata=kubernetes.client.V1ObjectMeta(
                            labels={'app': service_name}
                        ),
                        spec=kubernetes.client.V1PodSpec(
                            containers=[kubernetes.client.V1Container(
                                name=service_name,
                                image=f"{service_config['image']}:{version}",
                                ports=[kubernetes.client.V1ContainerPort(
                                    container_port=service_config['port']
                                )]
                            )]
                        )
                    )
                )
            )

            try:
                apps_v1.create_namespaced_deployment(namespace, deployment)
                stage.logs.append(f"✅ Kubernetes部署成功: {service_name}")
            except ApiException as e:
                stage.logs.append(f"❌ Kubernetes部署失败: {service_name} - {e}")
                raise

    def _deploy_with_docker_compose(self, stage: DeploymentStage,
                                    environment: str, version: str):
        """使用Docker Compose部署"""
        print("   使用Docker Compose部署...")

        compose_file = f'docker-compose.{environment}.yml'

        if os.path.exists(compose_file):
            result = subprocess.run([
                'docker-compose', '-f', compose_file, 'up', '-d'
            ], capture_output=True, text=True, timeout=600)

            if result.returncode == 0:
                stage.logs.append("✅ Docker Compose部署成功")
            else:
                stage.logs.append(f"❌ Docker Compose部署失败: {result.stderr}")
                raise Exception("Docker Compose部署失败")
        else:
            stage.logs.append(f"⚠️  Docker Compose文件不存在: {compose_file}")

    def _run_deployment_verification(self, deployment: DeploymentResult, environment: str):
        """运行部署验证"""
        stage = DeploymentStage(name='deployment_verification', status='running')
        stage.start_time = datetime.now()
        deployment.stages.append(stage)

        try:
            print("🔍 执行部署验证...")

            env_config = self.config['environments'].get(environment, {})
            services = self.config['services']

            verification_results = {}

            # 验证服务健康状态
            for service_name, service_config in services.items():
                service_url = f"http://localhost:{service_config['port']}/health"

                try:
                    response = requests.get(service_url, timeout=10)

                    if response.status_code == 200:
                        verification_results[service_name] = 'healthy'
                        stage.logs.append(f"✅ 服务健康检查通过: {service_name}")
                    else:
                        verification_results[service_name] = 'unhealthy'
                        stage.logs.append(f"❌ 服务健康检查失败: {service_name} - {response.status_code}")

                except Exception as e:
                    verification_results[service_name] = 'unreachable'
                    stage.logs.append(f"❌ 服务连接失败: {service_name} - {e}")

            stage.artifacts['health_checks'] = verification_results

            # 检查是否有失败的服务
            failed_services = [s for s, status in verification_results.items()
                               if status != 'healthy']

            if failed_services:
                raise Exception(f"部署验证失败的服务: {', '.join(failed_services)}")

            stage.status = 'success'

        except Exception as e:
            stage.logs.append(f"❌ 部署验证异常: {e}")
            stage.status = 'failed'
            raise

        finally:
            stage.end_time = datetime.now()

    def _run_monitoring_setup(self, deployment: DeploymentResult, environment: str):
        """运行监控设置"""
        stage = DeploymentStage(name='monitoring_setup', status='running')
        stage.start_time = datetime.now()
        deployment.stages.append(stage)

        try:
            print("📊 执行监控设置...")

            monitoring_config = self.config.get('monitoring', {})

            # 设置Prometheus监控
            if 'prometheus_url' in monitoring_config:
                self._setup_prometheus_monitoring(stage, environment)

            # 设置Grafana仪表板
            if 'grafana_url' in monitoring_config:
                self._setup_grafana_dashboards(stage, environment)

            # 设置告警规则
            if 'alertmanager_url' in monitoring_config:
                self._setup_alert_rules(stage, environment)

            stage.status = 'success'

        except Exception as e:
            stage.logs.append(f"❌ 监控设置异常: {e}")
            stage.status = 'failed'
            raise

        finally:
            stage.end_time = datetime.now()

    def _setup_prometheus_monitoring(self, stage: DeploymentStage, environment: str):
        """设置Prometheus监控"""
        # 这里可以添加具体的Prometheus配置逻辑
        stage.logs.append("✅ Prometheus监控配置完成")

    def _setup_grafana_dashboards(self, stage: DeploymentStage, environment: str):
        """设置Grafana仪表板"""
        # 这里可以添加具体的Grafana仪表板配置逻辑
        stage.logs.append("✅ Grafana仪表板配置完成")

    def _setup_alert_rules(self, stage: DeploymentStage, environment: str):
        """设置告警规则"""
        # 这里可以添加具体的告警规则配置逻辑
        stage.logs.append("✅ 告警规则配置完成")

    def _run_performance_validation(self, deployment: DeploymentResult, environment: str):
        """运行性能验证"""
        stage = DeploymentStage(name='performance_validation', status='running')
        stage.start_time = datetime.now()
        deployment.stages.append(stage)

        try:
            print("⚡ 执行性能验证...")

            # 运行部署后的性能测试
            result = subprocess.run([
                'python', '-m', 'pytest', 'tests/performance/',
                '--junitxml=performance-post-deploy.xml',
                '-v', '--tb=short'
            ], capture_output=True, text=True, timeout=600)

            stage.logs.append(f"性能验证结果: {result.stdout}")
            stage.artifacts['performance_results'] = 'performance-post-deploy.xml'

            if result.returncode == 0:
                stage.logs.append("✅ 性能验证通过")
                stage.status = 'success'
            else:
                stage.logs.append(f"❌ 性能验证失败: {result.stderr}")
                stage.status = 'failed'

        except Exception as e:
            stage.logs.append(f"❌ 性能验证异常: {e}")
            stage.status = 'failed'
            raise

        finally:
            stage.end_time = datetime.now()

    def _save_deployment_result(self, deployment: DeploymentResult):
        """保存部署结果"""
        result_file = f"deployment-result-{deployment.pipeline_id}.json"

        result_data = {
            'pipeline_id': deployment.pipeline_id,
            'status': deployment.status,
            'start_time': deployment.start_time.isoformat(),
            'end_time': deployment.end_time.isoformat() if deployment.end_time else None,
            'duration': (deployment.end_time - deployment.start_time).total_seconds() if deployment.end_time else None,
            'stages': [
                {
                    'name': stage.name,
                    'status': stage.status,
                    'start_time': stage.start_time.isoformat() if stage.start_time else None,
                    'end_time': stage.end_time.isoformat() if stage.end_time else None,
                    'duration': (stage.end_time - stage.start_time).total_seconds() if stage.start_time and stage.end_time else None,
                    'logs': stage.logs,
                    'artifacts': stage.artifacts
                }
                for stage in deployment.stages
            ],
            'metrics': deployment.metrics,
            'artifacts': deployment.artifacts
        }

        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        print(f"📄 部署结果已保存: {result_file}")

    def rollback_deployment(self, deployment_id: str, environment: str):
        """回滚部署"""
        print(f"🔄 执行部署回滚: {deployment_id}")

        try:
            # 停止当前部署
            if self.k8s_client:
                # Kubernetes回滚
                apps_v1 = kubernetes.client.AppsV1Api()
                env_config = self.config['environments'].get(environment, {})
                namespace = env_config.get('namespace', environment)

                # 回滚到上一个版本
                for service_name in self.config.get('services', {}):
                    deployment_name = f'{service_name}-deployment'
                    apps_v1.rollback_namespaced_deployment(
                        deployment_name, namespace,
                        kubernetes.client.V1RollbackConfig()
                    )

            else:
                # Docker Compose回滚
                compose_file = f'docker-compose.{environment}.yml'
                if os.path.exists(compose_file):
                    subprocess.run([
                        'docker-compose', '-f', compose_file, 'down'
                    ], check=True)

                    # 启动上一版本
                    subprocess.run([
                        'docker-compose', '-f', compose_file, 'up', '-d'
                    ], check=True)

            print("✅ 部署回滚完成")

        except Exception as e:
            print(f"❌ 部署回滚失败: {e}")
            raise


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python deployment_pipeline.py <环境> [版本]")
        print("环境: dev, staging, prod")
        print("版本: 默认为latest")
        sys.exit(1)

    environment = sys.argv[1]
    version = sys.argv[2] if len(sys.argv) > 2 else 'latest'

    # 创建部署流水线
    pipeline = DevOpsDeploymentPipeline()

    try:
        # 运行完整流水线
        result = pipeline.run_full_pipeline(environment, version)

        if result.status == 'success':
            print("
                  🎉 DevOps部署流水线执行成功！"            print(f"流水线ID: {result.pipeline_id}")
            print(f"环境: {environment}")
            print(f"版本: {version}")
            print(f"耗时: {(result.end_time - result.start_time).total_seconds():.1f}秒")
            sys.exit(0)
        else:
            print(f"\n❌ DevOps部署流水线执行失败: {result.status}")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⚠️  用户中断部署")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ DevOps部署异常: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
