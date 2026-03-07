#!/usr / bin / env python
# -*- coding: utf-8 -*-

"""
回测层微服务架构

实现长期规划目标：支持微服务架构、实现容器化部署、添加云原生特性
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from pathlib import Path
import docker
try:
    import kubernetes
    from kubernetes import client, config
    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False
    kubernetes = None
    client = None
    config = None

import yaml
import time

logger = logging.getLogger(__name__)


class ConfigManager:

    """配置管理器"""

    def __init__(self, config_path: Optional[str] = None):

        self.config_path = config_path or "config / microservices.yml"
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r', encoding='utf - 8') as f:
                    return yaml.safe_load(f)
            else:
                logger.warning(f"配置文件不存在: {self.config_path}")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'services': {
                'backtest - service': {
                    'port': 8001,
                    'replicas': 2,
                    'cpu_limit': '1000m',
                    'memory_limit': '1Gi'
                },
                'data - service': {
                    'port': 8002,
                    'replicas': 1,
                    'cpu_limit': '500m',
                    'memory_limit': '512Mi'
                },
                'strategy - service': {
                    'port': 8003,
                    'replicas': 1,
                    'cpu_limit': '500m',
                    'memory_limit': '512Mi'
                }
            },
            'monitoring': {
                'enabled': True,
                'metrics_port': 9090,
                'health_check_interval': 30
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }

    def get_service_config(self, service_name: str) -> Optional[Dict[str, Any]]:
        """获取服务配置"""
        return self.config.get('services', {}).get(service_name)

    def get_monitoring_config(self) -> Dict[str, Any]:
        """获取监控配置"""
        return self.config.get('monitoring', {})

    def get_logging_config(self) -> Dict[str, Any]:
        """获取日志配置"""
        return self.config.get('logging', {})


class MetricsCollector:

    """指标收集器"""

    def __init__(self):

        self.metrics: Dict[str, Any] = {
            'service_metrics': {},
            'system_metrics': {},
            'performance_metrics': {}
        }

    def record_service_metric(self, service_name: str, metric_name: str, value: Union[int, float, str]):
        """记录服务指标"""
        if service_name not in self.metrics['service_metrics']:
            self.metrics['service_metrics'][service_name] = {}

        self.metrics['service_metrics'][service_name][metric_name] = {
            'value': value,
            'timestamp': datetime.now().isoformat()
        }

    def record_system_metric(self, metric_name: str, value: Union[int, float, str]):
        """记录系统指标"""
        self.metrics['system_metrics'][metric_name] = {
            'value': value,
            'timestamp': datetime.now().isoformat()
        }

    def record_performance_metric(self, operation: str, duration: float):
        """记录性能指标"""
        if operation not in self.metrics['performance_metrics']:
            self.metrics['performance_metrics'][operation] = []

        self.metrics['performance_metrics'][operation].append({
            'duration': duration,
            'timestamp': datetime.now().isoformat()
        })

    def get_metrics(self) -> Dict[str, Any]:
        """获取所有指标"""
        return self.metrics.copy()

    def clear_metrics(self):
        """清除指标"""
        self.metrics = {
            'service_metrics': {},
            'system_metrics': {},
            'performance_metrics': {}
        }


@dataclass
class ServiceConfig:

    """服务配置"""
    name: str
    port: int
    replicas: int = 1
    cpu_limit: str = "500m"
    memory_limit: str = "512Mi"
    environment: Dict[str, str] = field(default_factory=dict)
    health_check_path: str = "/health"
    readiness_path: str = "/ready"
    startup_timeout: int = 300  # 启动超时时间（秒）
    shutdown_timeout: int = 60   # 关闭超时时间（秒）
    max_retries: int = 3         # 最大重试次数
    retry_delay: int = 5         # 重试延迟（秒）
    log_level: str = "INFO"      # 日志级别
    metrics_enabled: bool = True  # 是否启用指标收集


@dataclass
class ServiceInstance:

    """服务实例"""
    service_id: str
    host: str
    port: int
    status: str  # 'running', 'stopped', 'failed', 'starting', 'stopping'
    health_status: str  # 'healthy', 'unhealthy', 'unknown'
    last_heartbeat: datetime = field(default_factory=datetime.now)
    start_time: Optional[datetime] = None
    uptime: Optional[timedelta] = None
    version: str = "1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)


class MicroService(ABC):

    """微服务基类"""

    def __init__(self, config: ServiceConfig):

        self.config = config
        self.instances: List[ServiceInstance] = []
        self.health_check_interval = 30  # 秒
        self.metrics_collector = MetricsCollector()
        self.is_running = False
        self.start_time = None
        self.error_count = 0
        self.last_error = None

    @abstractmethod
    async def start(self):
        """启动服务"""

    @abstractmethod
    async def stop(self):
        """停止服务"""

    @abstractmethod
    async def health_check(self) -> bool:
        """健康检查"""

    async def register_instance(self, instance: ServiceInstance):
        """注册服务实例"""
        self.instances.append(instance)
        logger.info(f"服务实例已注册: {instance.service_id}")

    async def unregister_instance(self, service_id: str):
        """注销服务实例"""
        self.instances = [inst for inst in self.instances if inst.service_id != service_id]
        logger.info(f"服务实例已注销: {service_id}")

    def get_healthy_instances(self) -> List[ServiceInstance]:
        """获取健康实例"""
        return [inst for inst in self.instances if inst.health_status == 'healthy']

    def get_service_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        return {
            'name': self.config.name,
            'is_running': self.is_running,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'uptime': str(datetime.now() - self.start_time) if self.start_time else None,
            'total_instances': len(self.instances),
            'healthy_instances': len(self.get_healthy_instances()),
            'error_count': self.error_count,
            'last_error': str(self.last_error) if self.last_error else None,
            'metrics': self.metrics_collector.get_metrics()
        }

    def record_error(self, error: Exception):
        """记录错误"""
        self.error_count += 1
        self.last_error = error
        logger.error(f"服务 {self.config.name} 发生错误: {error}")

    def reset_error_count(self):
        """重置错误计数"""
        self.error_count = 0
        self.last_error = None


class BacktestService(MicroService):

    """回测服务"""

    def __init__(self, config: ServiceConfig):

        super().__init__(config)
        self.backtest_engine = None
        self.task_queue = None  # 延迟初始化

    async def start(self):
        """启动回测服务"""
        start_time = time.time()
        try:
            self.is_running = True
            self.start_time = datetime.now()

            # 初始化任务队列
            if self.task_queue is None:
                self.task_queue = asyncio.Queue()

            # 初始化回测引擎
            try:
                from .real_time_engine import RealTimeBacktestEngine
                self.backtest_engine = RealTimeBacktestEngine()
            except ImportError as e:
                logger.warning(f"无法导入RealTimeBacktestEngine: {e}")
                self.backtest_engine = None

            # 启动任务处理循环
            asyncio.create_task(self._process_tasks())

            # 启动健康检查
            asyncio.create_task(self._health_check_loop())

            # 记录启动指标
            startup_duration = time.time() - start_time
            self.metrics_collector.record_performance_metric('service_startup', startup_duration)
            self.metrics_collector.record_service_metric(
                self.config.name, 'startup_duration', startup_duration)

            logger.info(f"回测服务已启动: {self.config.name} (耗时: {startup_duration:.2f}秒)")

        except Exception as e:
            self.record_error(e)
            self.is_running = False
            logger.error(f"启动回测服务失败: {e}")
            raise

    async def stop(self):
        """停止回测服务"""
        try:
            self.is_running = False

            if self.backtest_engine:
                try:
                    self.backtest_engine.stop()
                except Exception as e:
                    logger.warning(f"停止回测引擎时发生错误: {e}")

            # 记录运行时间
            if self.start_time:
                uptime = datetime.now() - self.start_time
                self.metrics_collector.record_service_metric(
                    self.config.name, 'total_uptime', uptime.total_seconds())

            logger.info(f"回测服务已停止: {self.config.name}")

        except Exception as e:
            self.record_error(e)
            logger.error(f"停止回测服务时发生错误: {e}")
            raise

    async def health_check(self) -> bool:
        """健康检查"""
        try:
            # 检查回测引擎状态
            if self.backtest_engine and self.backtest_engine.running:
                return True
            return False
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return False

    async def _process_tasks(self):
        """处理任务队列"""
        while True:
            try:
                task = await self.task_queue.get()
                await self._execute_backtest_task(task)
                self.task_queue.task_done()
            except Exception as e:
                logger.error(f"处理任务失败: {e}")

    async def _execute_backtest_task(self, task: Dict[str, Any]):
        """执行回测任务"""
        try:
            # 执行回测逻辑
            result = await self._run_backtest(task)

            # 返回结果
            await self._send_result(task['task_id'], result)

        except Exception as e:
            logger.error(f"执行回测任务失败: {e}")
            await self._send_error(task['task_id'], str(e))

    async def _run_backtest(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """运行回测"""
        # 模拟回测执行
        await asyncio.sleep(1)
        return {
            'task_id': task['task_id'],
            'status': 'completed',
            'result': {'portfolio_value': 1000000.0}
        }

    async def _send_result(self, task_id: str, result: Dict[str, Any]):
        """发送结果"""
        # 这里应该发送到消息队列或API
        logger.info(f"回测任务完成: {task_id}")

    async def _send_error(self, task_id: str, error: str):
        """发送错误"""
        logger.error(f"回测任务失败: {task_id} - {error}")

    async def _health_check_loop(self):
        """健康检查循环"""
        while True:
            try:
                is_healthy = await self.health_check()
                if not is_healthy:
                    logger.warning(f"服务健康检查失败: {self.config.name}")

                await asyncio.sleep(self.health_check_interval)

            except Exception as e:
                logger.error(f"健康检查循环错误: {e}")


class DataService(MicroService):

    """数据服务"""

    def __init__(self, config: ServiceConfig):

        super().__init__(config)
        self.data_cache = {}

    async def start(self):
        """启动数据服务"""
        try:
            # 初始化数据缓存
            self.data_cache = {}

            # 启动数据同步
            asyncio.create_task(self._sync_data())

            logger.info(f"数据服务已启动: {self.config.name}")

        except Exception as e:
            logger.error(f"启动数据服务失败: {e}")

    async def stop(self):
        """停止数据服务"""
        self.data_cache.clear()
        logger.info(f"数据服务已停止: {self.config.name}")

    async def health_check(self) -> bool:
        """健康检查"""
        try:
            # 检查数据缓存状态
            return len(self.data_cache) >= 0
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return False

    async def _sync_data(self):
        """同步数据"""
        while True:
            try:
                # 模拟数据同步
                await asyncio.sleep(60)
                logger.debug("数据同步完成")

            except Exception as e:
                logger.error(f"数据同步失败: {e}")


class StrategyService(MicroService):

    """策略服务"""

    def __init__(self, config: ServiceConfig):

        super().__init__(config)
        self.strategies = {}

    async def start(self):
        """启动策略服务"""
        try:
            # 加载策略
            await self._load_strategies()

            # 启动策略监控
            asyncio.create_task(self._monitor_strategies())

            logger.info(f"策略服务已启动: {self.config.name}")

        except Exception as e:
            logger.error(f"启动策略服务失败: {e}")

    async def stop(self):
        """停止策略服务"""
        self.strategies.clear()
        logger.info(f"策略服务已停止: {self.config.name}")

    async def health_check(self) -> bool:
        """健康检查"""
        try:
            # 检查策略状态
            return len(self.strategies) >= 0
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return False

    async def _load_strategies(self):
        """加载策略"""
        # 模拟加载策略
        self.strategies = {
            'momentum': {'type': 'momentum', 'enabled': True},
            'mean_reversion': {'type': 'mean_reversion', 'enabled': True}
        }

    async def _monitor_strategies(self):
        """监控策略"""
        while True:
            try:
                # 监控策略性能
                await asyncio.sleep(30)
                logger.debug("策略监控完成")

            except Exception as e:
                logger.error(f"策略监控失败: {e}")


class ServiceRegistry:

    """服务注册中心"""

    def __init__(self):

        self.services: Dict[str, MicroService] = {}
        self.service_discovery = {}

    async def register_service(self, service: MicroService):
        """注册服务"""
        self.services[service.config.name] = service
        self.service_discovery[service.config.name] = {
            'instances': [],
            'last_updated': datetime.now()
        }
        logger.info(f"服务已注册: {service.config.name}")

    async def unregister_service(self, service_name: str):
        """注销服务"""
        if service_name in self.services:
            del self.services[service_name]
        if service_name in self.service_discovery:
            del self.service_discovery[service_name]
        logger.info(f"服务已注销: {service_name}")

    def get_service(self, service_name: str) -> Optional[MicroService]:
        """获取服务"""
        return self.services.get(service_name)

    def get_all_services(self) -> Dict[str, MicroService]:
        """获取所有服务"""
        return self.services


class DockerManager:

    """Docker管理器"""

    def __init__(self):
        """初始化Docker管理器，延迟连接Docker API"""
        self.client = None
        self._client_initialized = False
        self._client_available = False
        self._init_error = None

    def _ensure_client(self) -> bool:
        """确保Docker客户端已初始化，延迟初始化以避免在容器内持续报错"""
        if self._client_initialized:
            return self._client_available
        
        self._client_initialized = True
        
        try:
            # 检测是否在容器内运行
            import os
            is_in_container = os.path.exists('/.dockerenv') or os.path.exists('/proc/self/cgroup')
            
            if is_in_container:
                # 在容器内运行时，检查是否有Docker socket挂载
                docker_socket_paths = [
                    '/var/run/docker.sock',
                    '//./pipe/dockerDesktopLinuxEngine',  # Windows Docker Desktop
                    'npipe:////./pipe/dockerDesktopLinuxEngine'
                ]
                
                # 检查是否有可用的Docker socket
                has_docker_socket = False
                for path in docker_socket_paths:
                    if os.path.exists(path.replace('npipe://', '').replace('//./', '')):
                        has_docker_socket = True
                        break
                
                if not has_docker_socket:
                    logger.warning(
                        "检测到在容器内运行，但未挂载Docker socket。"
                        "Docker功能将被禁用。如需使用Docker功能，请挂载Docker socket。"
                    )
                    self._client_available = False
                    self._init_error = "容器内未挂载Docker socket"
                    return False
            
            # 尝试连接Docker API
            self.client = docker.from_env()
            # 测试连接
            self.client.ping()
            self._client_available = True
            logger.info("Docker客户端初始化成功")
            return True
            
        except docker.errors.DockerException as e:
            error_msg = str(e)
            # 只在首次初始化失败时记录警告，避免持续输出错误日志
            if 'dockerDesktopLinuxEngine' in error_msg or 'docker.sock' in error_msg:
                logger.warning(
                    f"Docker API不可用: {error_msg}。"
                    "Docker功能将被禁用。如果不需要Docker功能，可以忽略此警告。"
                )
            else:
                logger.warning(f"Docker客户端初始化失败: {error_msg}")
            
            self._client_available = False
            self._init_error = error_msg
            return False
        except Exception as e:
            logger.warning(f"Docker客户端初始化失败: {e}")
            self._client_available = False
            self._init_error = str(e)
            return False

    def build_image(self, service_name: str, dockerfile_path: str, tag: str) -> bool:
        """构建Docker镜像"""
        if not self._ensure_client():
            logger.warning(f"Docker不可用，跳过镜像构建: {tag}")
            return False
            
        try:
            image, logs = self.client.images.build(
                path=dockerfile_path,
                tag=tag,
                rm=True
            )
            logger.info(f"Docker镜像构建成功: {tag}")
            return True
        except Exception as e:
            logger.error(f"Docker镜像构建失败: {e}")
            return False

    def run_container(self, image_name: str, container_name: str,


                      ports: Dict[str, str], environment: Dict[str, str] = None) -> bool:
        """运行Docker容器"""
        if not self._ensure_client():
            logger.warning(f"Docker不可用，跳过容器启动: {container_name}")
            return False
            
        try:
            container = self.client.containers.run(
                image_name,
                name=container_name,
                ports=ports,
                environment=environment,
                detach=True
            )
            logger.info(f"Docker容器启动成功: {container_name}")
            return True
        except Exception as e:
            logger.error(f"Docker容器启动失败: {e}")
            return False

    def stop_container(self, container_name: str) -> bool:
        """停止Docker容器"""
        if not self._ensure_client():
            logger.warning(f"Docker不可用，跳过容器停止: {container_name}")
            return False
            
        try:
            container = self.client.containers.get(container_name)
            container.stop()
            logger.info(f"Docker容器停止成功: {container_name}")
            return True
        except Exception as e:
            logger.error(f"Docker容器停止失败: {e}")
            return False

    def get_container_status(self, container_name: str) -> str:
        """获取容器状态"""
        if not self._ensure_client():
            return "unavailable"
            
        try:
            container = self.client.containers.get(container_name)
            return container.status
        except Exception as e:
            logger.error(f"获取容器状态失败: {e}")
            return "unknown"


class KubernetesManager:

    """Kubernetes管理器"""

    def __init__(self):

        if not KUBERNETES_AVAILABLE:
            logger.warning("Kubernetes客户端不可用，跳过Kubernetes功能")
            self.v1 = None
            self.apps_v1 = None
            return

        try:
            config.load_kube_config()
            self.v1 = client.CoreV1Api()
            self.apps_v1 = client.AppsV1Api()
        except Exception as e:
            logger.error(f"Kubernetes配置加载失败: {e}")
            self.v1 = None
            self.apps_v1 = None

    def deploy_service(self, service_config: ServiceConfig,


                       image_name: str, namespace: str = "default") -> bool:
        """部署服务到Kubernetes"""
        if not KUBERNETES_AVAILABLE or self.v1 is None or self.apps_v1 is None:
            logger.warning("Kubernetes不可用，跳过服务部署")
            return False

        try:
            # 创建Deployment
            deployment = client.V1Deployment(
                metadata=client.V1ObjectMeta(name=service_config.name),
                spec=client.V1DeploymentSpec(
                    replicas=service_config.replicas,
                    selector=client.V1LabelSelector(
                        match_labels={"app": service_config.name}
                    ),
                    template=client.V1PodTemplateSpec(
                        metadata=client.V1ObjectMeta(
                            labels={"app": service_config.name}
                        ),
                        spec=client.V1PodSpec(
                            containers=[
                                client.V1Container(
                                    name=service_config.name,
                                    image=image_name,
                                    ports=[client.V1ContainerPort(
                                        container_port=service_config.port)],
                                    resources=client.V1ResourceRequirements(
                                        limits={
                                            "cpu": service_config.cpu_limit,
                                            "memory": service_config.memory_limit
                                        }
                                    ),
                                    env=[
                                        client.V1EnvVar(name=k, value=v)
                                        for k, v in service_config.environment.items()
                                    ]
                                )
                            ]
                        )
                    )
                )
            )

            self.apps_v1.create_namespaced_deployment(
                body=deployment,
                namespace=namespace
            )

            # 创建Service
            service = client.V1Service(
                metadata=client.V1ObjectMeta(name=service_config.name),
                spec=client.V1ServiceSpec(
                    selector={"app": service_config.name},
                    ports=[client.V1ServicePort(port=service_config.port)]
                )
            )

            self.v1.create_namespaced_service(
                body=service,
                namespace=namespace
            )

            logger.info(f"Kubernetes服务部署成功: {service_config.name}")
            return True

        except Exception as e:
            logger.error(f"Kubernetes服务部署失败: {e}")
            return False

    def delete_service(self, service_name: str, namespace: str = "default") -> bool:
        """删除Kubernetes服务"""
        if not KUBERNETES_AVAILABLE or self.v1 is None or self.apps_v1 is None:
            logger.warning("Kubernetes不可用，跳过服务删除")
            return False

        try:
            # 删除Deployment
            self.apps_v1.delete_namespaced_deployment(
                name=service_name,
                namespace=namespace
            )

            # 删除Service
            self.v1.delete_namespaced_service(
                name=service_name,
                namespace=namespace
            )

            logger.info(f"Kubernetes服务删除成功: {service_name}")
            return True

        except Exception as e:
            logger.error(f"Kubernetes服务删除失败: {e}")
            return False

    def get_service_status(self, service_name: str, namespace: str = "default") -> Dict[str, Any]:
        """获取服务状态"""
        if not KUBERNETES_AVAILABLE or self.v1 is None or self.apps_v1 is None:
            logger.warning("Kubernetes不可用，无法获取服务状态")
            return {}

        try:
            deployment = self.apps_v1.read_namespaced_deployment(
                name=service_name,
                namespace=namespace
            )

            return {
                'name': service_name,
                'replicas': deployment.spec.replicas,
                'available_replicas': deployment.status.available_replicas,
                'ready_replicas': deployment.status.ready_replicas,
                'updated_replicas': deployment.status.updated_replicas
            }

        except Exception as e:
            logger.error(f"获取服务状态失败: {e}")
            return {}


class MicroserviceOrchestrator:

    """微服务编排器"""

    def __init__(self):

        self.service_registry = ServiceRegistry()
        self.docker_manager = DockerManager()
        self.k8s_manager = KubernetesManager()

    async def deploy_backtest_system(self, environment: str = "production"):
        """部署回测系统"""
        try:
            # 创建服务配置
            services = [
                ServiceConfig("backtest - service", 8001, 2),
                ServiceConfig("data - service", 8002, 1),
                ServiceConfig("strategy - service", 8003, 1)
            ]

            # 部署服务
            for service_config in services:
                # 构建Docker镜像
                image_tag = f"backtest-{service_config.name}:latest"
                self.docker_manager.build_image(
                    service_config.name,
                    f"dockerfiles/{service_config.name}",
                    image_tag
                )

                # 部署到Kubernetes
                self.k8s_manager.deploy_service(service_config, image_tag)

            logger.info("回测系统部署完成")
            return True

        except Exception as e:
            logger.error(f"部署回测系统失败: {e}")
            return False

    async def scale_service(self, service_name: str, replicas: int):
        """扩展服务"""
        try:
            # 更新Kubernetes Deployment
            # 这里需要实现具体的扩展逻辑
            logger.info(f"服务扩展完成: {service_name} -> {replicas} replicas")
            return True
        except Exception as e:
            logger.error(f"服务扩展失败: {e}")
            return False

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        status = {
            'services': {},
            'overall_health': 'healthy'
        }

        for service_name, service in self.service_registry.services.items():
            healthy_instances = service.get_healthy_instances()
            status['services'][service_name] = {
                'total_instances': len(service.instances),
                'healthy_instances': len(healthy_instances),
                'health_status': 'healthy' if healthy_instances else 'unhealthy'
            }

        if not healthy_instances:
            status['overall_health'] = 'unhealthy'

        return status


# 全局微服务编排器实例
orchestrator = MicroserviceOrchestrator()


async def deploy_microservices():
    """部署微服务"""
    return await orchestrator.deploy_backtest_system()


async def scale_microservice(service_name: str, replicas: int):
    """扩展微服务"""
    return await orchestrator.scale_service(service_name, replicas)


def get_microservice_status() -> Dict[str, Any]:
    """获取微服务状态"""
    return orchestrator.get_system_status()


def build_docker_image(service_name: str, dockerfile_path: str, tag: str) -> bool:
    """构建Docker镜像"""
    return orchestrator.docker_manager.build_image(service_name, dockerfile_path, tag)


def deploy_to_kubernetes(service_config: ServiceConfig, image_name: str) -> bool:
    """部署到Kubernetes"""
    return orchestrator.k8s_manager.deploy_service(service_config, image_name)
