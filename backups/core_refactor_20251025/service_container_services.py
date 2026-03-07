"""
服务容器管理
提供统一的服务访问接口和服务管理功能
"""

from typing import Dict, Any, Optional, List, Callable, Type
import logging
import threading
import time
import json
import os
from dataclasses import dataclass, asdict

from .container import DependencyContainer, Lifecycle, ServiceHealth
from .service_container.unified_container_interface import IServiceContainer, ServiceScope, ServiceStatus
from ...infrastructure.container.unified_container_interface import IServiceScope
from src.infrastructure.health.infrastructure.load_balancer import LoadBalancingStrategy

logger = logging.getLogger(__name__)


@dataclass
class ServiceConfig:

    """服务配置"""
    name: str
    service_type: Optional[Type] = None
    implementation: Optional[Type] = None
    factory: Optional[Callable] = None
    lifecycle: Lifecycle = Lifecycle.SINGLETON
    version: str = "1.0.0"
    dependencies: List[str] = None
    health_check: Optional[Callable] = None
    health_check_interval: int = 300  # 5分钟
    max_instances: int = 1
    load_balancing_strategy: str = "round_robin"
    weight: int = 1
    enabled: bool = True
    auto_start: bool = True
    config: Dict[str, Any] = None

    def __post_init__(self):

        if self.dependencies is None:
            self.dependencies = []
        if self.config is None:
            self.config = {}


@dataclass
class ServiceInstance:

    """服务实例"""
    name: str
    instance: Any
    status: ServiceStatus
    created_time: float
    last_health_check: Optional[float] = None
    health_status: ServiceHealth = ServiceHealth.UNKNOWN
    connections: int = 0
    weight: int = 1
    metadata: Dict[str, Any] = None

    def __post_init__(self):

        if self.metadata is None:
            self.metadata = {}


class ServiceContainer(IServiceContainer):

    """服务容器管理 - 增强版"""

    def __init__(self, config_dir: str = "config / services"):

        self.container = DependencyContainer()
        self.config_dir = config_dir
        self.service_configs: Dict[str, ServiceConfig] = {}
        self.service_instances: Dict[str, List[ServiceInstance]] = {}
        self.service_status: Dict[str, ServiceStatus] = {}
        self.load_balancers: Dict[str, 'LoadBalancer'] = {}

        # 细粒度锁机制 - 提高并发性能
        self._service_lock = threading.RLock()    # 服务级锁
        self._instance_lock = threading.RLock()   # 实例级锁
        self._config_lock = threading.RLock()     # 配置级锁
        self._metrics_lock = threading.RLock()    # 指标级锁

        # 向后兼容的全局锁
        self.lock = self._service_lock

        # 监控和统计
        self.metrics = {
            'total_services': 0,
            'running_services': 0,
            'stopped_services': 0,
            'error_services': 0,
            'total_instances': 0
        }

        # 加载配置
        self._load_configs()

        # 启动监控线程
        self._monitoring_enabled = True
        self._monitor_thread = threading.Thread(target=self._monitor_services, daemon=True)
        self._monitor_thread.start()

        logger.info("服务容器管理初始化完成")

    def _load_configs(self):
        """加载服务配置"""
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)
            return

        for filename in os.listdir(self.config_dir):
            if filename.endswith('.json'):
                config_path = os.path.join(self.config_dir, filename)
                try:
                    with open(config_path, 'r', encoding='utf - 8') as f:
                        config_data = json.load(f)
                        service_config = ServiceConfig(**config_data)
                        self.service_configs[service_config.name] = service_config
                        logger.info(f"加载服务配置: {service_config.name}")
                except Exception as e:
                    logger.error(f"加载服务配置失败 {filename}: {e}")

    def register_service(self, config: ServiceConfig) -> bool:
        """注册服务"""
        with self._config_lock:
            try:
                # 检查服务是否已存在
                if config.name in self.service_configs:
                    logger.warning(f"服务 {config.name} 已存在，将被覆盖")

                # 保存配置
                self.service_configs[config.name] = config
                self._save_config(config)

                logger.info(f"注册服务配置: {config.name}")
            except Exception as e:
                logger.error(f"保存服务配置失败: {config.name}, 错误: {e}")
                return False

        with self._service_lock:
            try:
                # 初始化服务实例列表和状态
                self.service_instances[config.name] = []
                self.service_status[config.name] = ServiceStatus.STOPPED

                # 创建负载均衡器
                if config.max_instances > 1:
                    self.load_balancers[config.name] = LoadBalancer(config.load_balancing_strategy)

                # 如果配置为自动启动，则启动服务
                if config.auto_start and config.enabled:
                    self.start_service(config.name)

                logger.info(f"注册服务实例: {config.name}")
                return True

            except Exception as e:
                logger.error(f"注册服务实例失败: {config.name}, 错误: {e}")
                return False

    def unregister_service(self, name: str) -> bool:
        """注销服务"""
        with self.lock:
            try:
                if name not in self.service_configs:
                    logger.warning(f"服务 {name} 不存在")
                    return False

                # 停止服务
                self.stop_service(name)

                # 清理资源
                if name in self.service_instances:
                    for instance in self.service_instances[name]:
                        self._cleanup_instance(instance)
                    del self.service_instances[name]

                if name in self.service_configs:
                    del self.service_configs[name]

                if name in self.service_status:
                    del self.service_status[name]

                if name in self.load_balancers:
                    del self.load_balancers[name]

                logger.info(f"注销服务: {name}")
                return True

            except Exception as e:
                logger.error(f"注销服务失败: {name}, 错误: {e}")
                return False

    def start_service(self, name: str) -> bool:
        """启动服务"""
        with self.lock:
            try:
                if name not in self.service_configs:
                    logger.error(f"服务 {name} 不存在")
                    return False

                config = self.service_configs[name]
                if not config.enabled:
                    logger.warning(f"服务 {name} 已禁用")
                    return False

                # 更新状态
                self.service_status[name] = ServiceStatus.STARTING

                # 创建服务实例
                instances = []
                for i in range(config.max_instances):
                    instance = self._create_service_instance(config, i)
                    instances.append(instance)

                self.service_instances[name] = instances
                self.service_status[name] = ServiceStatus.RUNNING

                logger.info(f"启动服务: {name}, 实例数: {len(instances)}")
                return True

            except Exception as e:
                logger.error(f"启动服务失败: {name}, 错误: {e}")
                self.service_status[name] = ServiceStatus.ERROR
                return False

    def stop_service(self, name: str) -> bool:
        """停止服务"""
        with self.lock:
            try:
                if name not in self.service_instances:
                    logger.warning(f"服务 {name} 未运行")
                    return True

                # 更新状态
                self.service_status[name] = ServiceStatus.STOPPING

                # 停止所有实例
                instances = self.service_instances[name]
                for instance in instances:
                    self._cleanup_instance(instance)

                # 清空实例列表
                self.service_instances[name] = []
                self.service_status[name] = ServiceStatus.STOPPED

                logger.info(f"停止服务: {name}")
                return True

            except Exception as e:
                logger.error(f"停止服务失败: {name}, 错误: {e}")
                self.service_status[name] = ServiceStatus.ERROR
                return False

    def get_service(self, name: str) -> Optional[Any]:
        """获取服务实例"""
        with self._instance_lock:
            if name not in self.service_instances:
                return None

            instances = self.service_instances[name]
            if not instances:
                return None

            # 如果只有一个实例，直接返回
            if len(instances) == 1:
                return instances[0].instance

            # 如果有多个实例，使用负载均衡
            if name in self.load_balancers:
                load_balancer = self.load_balancers[name]
                return load_balancer.get_instance(instances)

            # 默认返回第一个实例
            return instances[0].instance

    def get_service_status(self, name: str) -> Optional[ServiceStatus]:
        """获取服务状态"""
        return self.service_status.get(name)

    def get_service_info(self, name: str) -> Optional[Dict[str, Any]]:
        """获取服务信息"""
        with self._config_lock:
            if name not in self.service_configs:
                return None
            config = self.service_configs[name]

        with self._instance_lock:
            instances = self.service_instances.get(name, [])

        status = self.service_status.get(name)

        return {
            'name': name,
            'config': asdict(config),
            'status': status.value if status else None,
            'instances': [
                {
                    'instance_id': i,
                    'status': instance.status.value,
                    'health_status': instance.health_status.value,
                    'created_time': instance.created_time,
                    'connections': instance.connections
                }
                for i, instance in enumerate(instances)
            ],
            'total_instances': len(instances)
        }

    def list_services(self) -> List[Dict[str, Any]]:
        """列出所有服务"""
        return [self.get_service_info(name) for name in self.service_configs.keys()]

    def get_services_by_status(self, status: ServiceStatus) -> List[str]:
        """根据状态获取服务"""
        return [name for name, s in self.service_status.items() if s == status]

    def get_services_by_type(self, service_type: Type) -> List[str]:
        """根据类型获取服务"""
        return [name for name, config in self.service_configs.items()
                if config.service_type == service_type]

    def update_service_config(self, name: str, **kwargs) -> bool:
        """更新服务配置"""
        with self.lock:
            try:
                if name not in self.service_configs:
                    logger.error(f"服务 {name} 不存在")
                    return False

                config = self.service_configs[name]

                # 更新配置字段
                for key, value in kwargs.items():
                    if hasattr(config, key):
                        setattr(config, key, value)

                # 保存更新后的配置
                self._save_config(config)

                logger.info(f"更新服务配置: {name}")
                return True

            except Exception as e:
                logger.error(f"更新服务配置失败: {name}, 错误: {e}")
                return False

    def _create_service_instance(self, config: ServiceConfig, index: int) -> ServiceInstance:
        """创建服务实例"""
        try:
            # 创建实例
            if config.factory:
                instance = config.factory()
            elif config.service_type:
                instance = config.service_type()
            else:
                raise ValueError(f"无法创建服务实例: {config.name}")

            # 创建服务实例对象
            service_instance = ServiceInstance(
                name=f"{config.name}_{index}",
                instance=instance,
                status=ServiceStatus.RUNNING,
                created_time=time.time(),
                weight=config.weight
            )

            return service_instance

        except Exception as e:
            logger.error(f"创建服务实例失败: {config.name}_{index}, 错误: {e}")
            raise

    def _cleanup_instance(self, instance: ServiceInstance):
        """清理服务实例"""
        try:
            if hasattr(instance.instance, 'stop'):
                instance.instance.stop()
            instance.status = ServiceStatus.STOPPED
        except Exception as e:
            logger.error(f"清理服务实例失败: {instance.name}, 错误: {e}")

    def _get_health_status(self, instances: List[ServiceInstance]) -> ServiceHealth:
        """获取服务健康状态"""
        if not instances:
            return ServiceHealth.UNKNOWN

        healthy_count = 0
        for instance in instances:
            if instance.health_status == ServiceHealth.HEALTHY:
                healthy_count += 1

        if healthy_count == len(instances):
            return ServiceHealth.HEALTHY
        elif healthy_count > 0:
            return ServiceHealth.UNHEALTHY
        else:
            return ServiceHealth.UNHEALTHY

    def _save_config(self, config: ServiceConfig):
        """保存服务配置"""
        try:
            if not os.path.exists(self.config_dir):
                os.makedirs(self.config_dir)

            config_file = os.path.join(self.config_dir, f"{config.name}.json")
            with open(config_file, 'w') as f:
                json.dump(asdict(config), f, indent=2)
        except Exception as e:
            logger.error(f"保存服务配置失败: {config.name}, 错误: {e}")

    def _monitor_services(self):
        """监控服务"""
        while self._monitoring_enabled:
            try:
                for name, config in self.service_configs.items():
                    if config.enabled:
                        self._check_service_health(name, config)

                time.sleep(30)  # 30秒检查一次

            except Exception as e:
                logger.error(f"服务监控异常: {e}")
                time.sleep(60)  # 出错后等待1分钟再检查

    def _check_service_health(self, name: str, config: ServiceConfig):
        """检查服务健康状态"""
        try:
            instances = self.service_instances.get(name, [])
            if not instances:
                return

            for instance in instances:
                if config.health_check:
                    try:
                        is_healthy = config.health_check(instance.instance)
                        instance.health_status = ServiceHealth.HEALTHY if is_healthy else ServiceHealth.UNHEALTHY
                    except Exception as e:
                        logger.error(f"健康检查失败: {instance.name}, 错误: {e}")
                        instance.health_status = ServiceHealth.UNHEALTHY

                instance.last_health_check = time.time()

        except Exception as e:
            logger.error(f"检查服务健康状态失败: {name}, 错误: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """获取服务指标"""
        with self._metrics_lock:
            with self._config_lock:
                total_services = len(self.service_configs)
                running_services = len(self.get_services_by_status(ServiceStatus.RUNNING))
                stopped_services = len(self.get_services_by_status(ServiceStatus.STOPPED))
                error_services = len(self.get_services_by_status(ServiceStatus.ERROR))

            with self._instance_lock:
                total_instances = sum(len(instances)
                                      for instances in self.service_instances.values())

            return {
                'total_services': total_services,
                'running_services': running_services,
                'stopped_services': stopped_services,
                'error_services': error_services,
                'total_instances': total_instances
            }

    def shutdown(self):
        """关闭服务容器"""
        self._monitoring_enabled = False

        with self._service_lock:
            # 停止所有服务
            for name in list(self.service_configs.keys()):
                self.stop_service(name)

            logger.info("服务容器已关闭")

    # 实现统一服务容器接口的其他方法

    def register_instance(self, service_type: Type, instance: Any, name: Optional[str] = None) -> bool:
        """注册服务实例"""
        try:
            # 创建服务配置
            service_name = name or service_type.__name__
            config = ServiceConfig(
                name=service_name,
                service_type=service_type,
                lifecycle=Lifecycle.SINGLETON
            )

            # 创建服务实例
            service_instance = ServiceInstance(
                name=service_name,
                instance=instance,
                config=config,
                status=ServiceStatus.RUNNING
            )

            # 注册到内部容器
            self.container.register(service_name, instance)

            with self._service_lock:
                self.service_configs[service_name] = config
                self.service_instances[service_name] = service_instance

            logger.info(f"服务实例 {service_name} 已注册")
            return True

        except Exception as e:
            logger.error(f"注册服务实例失败: {e}")
            return False

    def unregister(self, service_type: Type, name: Optional[str] = None) -> bool:
        """注销服务"""
        service_name = name or service_type.__name__

        with self._service_lock:
            if service_name in self.service_configs:
                # 停止服务
                self.stop_service(service_name)

                # 从容器中移除
                del self.service_configs[service_name]
                if service_name in self.service_instances:
                    del self.service_instances[service_name]

                logger.info(f"服务 {service_name} 已注销")
                return True

        return False

    def is_registered(self, service_type: Type, name: Optional[str] = None) -> bool:
        """检查服务是否已注册"""
        service_name = name or service_type.__name__
        with self._service_lock:
            return service_name in self.service_configs

    def get_service_info(self, service_type: Type, name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """获取服务信息"""
        service_name = name or service_type.__name__

        with self._service_lock:
            if service_name in self.service_configs:
                config = self.service_configs[service_name]
                instance = self.service_instances.get(service_name)

                return {
                    'name': service_name,
                    'type': service_type,
                    'lifecycle': config.lifecycle,
                    'status': instance.status if instance else ServiceStatus.STOPPED,
                    'dependencies': getattr(config, 'dependencies', []),
                    'metadata': getattr(config, 'metadata', {})
                }

        return None

    def get_all_services(self) -> Dict[str, List[Dict[str, Any]]]:
        """获取所有已注册的服务"""
        result = {}

        with self._service_lock:
            for service_name, config in self.service_configs.items():
                service_type_name = config.service_type.__name__ if config.service_type else 'unknown'
                if service_type_name not in result:
                    result[service_type_name] = []

                instance = self.service_instances.get(service_name)
                result[service_type_name].append({
                    'name': service_name,
                    'type': config.service_type,
                    'lifecycle': config.lifecycle,
                    'status': instance.status if instance else ServiceStatus.STOPPED,
                    'dependencies': getattr(config, 'dependencies', []),
                    'metadata': getattr(config, 'metadata', {})
                })

        return result

    def create_scope(self, scope_type: ServiceScope) -> 'IServiceScope':
        """创建服务作用域"""
        # 简化实现，返回一个基本的scope对象
        return ServiceScopeContext(scope_type, self)

    def begin_scope(self, scope_type: ServiceScope) -> 'IServiceScope':
        """开始一个新的作用域"""
        return self.create_scope(scope_type)

    def get_service_status(self, service_type: Type, name: Optional[str] = None) -> ServiceStatus:
        """获取服务状态"""
        service_name = name or service_type.__name__

        with self._service_lock:
            instance = self.service_instances.get(service_name)
            if instance:
                return instance.status

        return ServiceStatus.STOPPED

    def initialize_service(self, service_type: Type, name: Optional[str] = None) -> bool:
        """初始化服务"""
        service_name = name or service_type.__name__
        return self.start_service(service_name)

    def dispose_service(self, service_type: Type, name: Optional[str] = None) -> bool:
        """销毁服务"""
        service_name = name or service_type.__name__
        return self.stop_service(service_name)

    def get_service_dependencies(self, service_type: Type, name: Optional[str] = None) -> List[Type]:
        """获取服务依赖"""
        service_info = self.get_service_info(service_type, name)
        if service_info:
            return service_info.get('dependencies', [])
        return []

    def validate_service(self, service_type: Type, name: Optional[str] = None) -> Dict[str, Any]:
        """验证服务配置"""
        service_name = name or service_type.__name__

        with self._service_lock:
            if service_name not in self.service_configs:
                return {'valid': False, 'errors': ['服务未注册']}

            config = self.service_configs[service_name]
            errors = []

            # 基本验证
            if not config.name:
                errors.append('服务名称不能为空')

            if not config.service_type:
                errors.append('服务类型不能为空')

            return {
                'valid': len(errors) == 0,
                'errors': errors,
                'warnings': []
            }

    def get_service_metrics(self, service_type: Type = None, name: Optional[str] = None) -> Dict[str, Any]:
        """获取服务指标"""
        metrics = {
            'total_services': len(self.service_configs),
            'running_services': sum(1 for inst in self.service_instances.values() if inst.status == ServiceStatus.RUNNING),
            'stopped_services': sum(1 for inst in self.service_instances.values() if inst.status == ServiceStatus.STOPPED),
            'error_services': sum(1 for inst in self.service_instances.values() if inst.status == ServiceStatus.ERROR)
        }

        if service_type or name:
            service_name = name or service_type.__name__ if service_type else None
            if service_name and service_name in self.service_instances:
                instance = self.service_instances[service_name]
                metrics['service_specific'] = {
                    'name': service_name,
                    'status': instance.status.value,
                    'uptime': getattr(instance, 'uptime', 0),
                    'restart_count': getattr(instance, 'restart_count', 0)
                }

        return metrics

    def enable_monitoring(self, enabled: bool = True) -> None:
        """启用/禁用监控"""
        self._monitoring_enabled = enabled
        if enabled:
            self._start_monitoring()
        else:
            self._monitoring_enabled = False

    def clear_cache(self) -> None:
        """清空服务实例缓存"""
        with self._service_lock:
            # 清理容器缓存
            self.container.clear()

            # 重新创建实例（对于transient服务）
            for service_name, config in self.service_configs.items():
                if config.lifecycle == Lifecycle.TRANSIENT:
                    if service_name in self.service_instances:
                        del self.service_instances[service_name]

        logger.info("服务实例缓存已清空")

    def dispose(self) -> None:
        """销毁容器，清理所有资源"""
        self.shutdown()


class ServiceScopeContext:
    """服务作用域上下文"""

    def __init__(self, scope_type: ServiceScope, container: ServiceContainer):
        self.scope_type = scope_type
        self.container = container
        self.scoped_instances = {}

    def resolve(self, service_type: Type, name: Optional[str] = None) -> Any:
        """在当前作用域内解析服务"""
        # 对于scoped生命周期，在作用域内创建新的实例
        service_name = name or service_type.__name__

        if service_name not in self.scoped_instances:
            # 从容器中解析服务
            instance = self.container.resolve(service_type, name)
            if instance:
                self.scoped_instances[service_name] = instance

        return self.scoped_instances.get(service_name)

    def dispose(self) -> None:
        """销毁作用域"""
        # 清理作用域内的实例（对于scoped生命周期）
        for instance in self.scoped_instances.values():
            if hasattr(instance, 'dispose'):
                instance.dispose()

        self.scoped_instances.clear()

    def get_scope_type(self) -> ServiceScope:
        """获取作用域类型"""
        return self.scope_type

    def get_services_in_scope(self) -> List[Dict[str, Any]]:
        """获取作用域内的服务"""
        return [
            {'name': name, 'instance': instance}
            for name, instance in self.scoped_instances.items()
        ]


class LoadBalancer:

    """负载均衡器"""

    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN):

        self.strategy = strategy
        self.current_index = 0
        self.lock = threading.Lock()

    def get_instance(self, instances: List[ServiceInstance]) -> Optional[Any]:
        """获取服务实例"""
        if not instances:
            return None

        with self.lock:
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                return self._round_robin(instances)
            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return self._least_connections(instances)
            elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                return self._weighted_round_robin(instances)
            elif self.strategy == LoadBalancingStrategy.RANDOM:
                return self._random(instances)
            else:
                return instances[0].instance

    def _round_robin(self, instances: List[ServiceInstance]) -> Any:
        """轮询策略"""
        if not instances:
            return None

        instance = instances[self.current_index % len(instances)]
        self.current_index += 1
        return instance.instance

    def _least_connections(self, instances: List[ServiceInstance]) -> Any:
        """最少连接策略"""
        if not instances:
            return None

        min_connections = float('inf')
        selected_instance = None

        for instance in instances:
            if instance.connections < min_connections:
                min_connections = instance.connections
                selected_instance = instance

        if selected_instance:
            selected_instance.connections += 1
            return selected_instance.instance

        return None

    def _weighted_round_robin(self, instances: List[ServiceInstance]) -> Any:
        """加权轮询策略"""
        if not instances:
            return None

        # 简化的加权轮询实现
        total_weight = sum(instance.weight for instance in instances)
        if total_weight == 0:
            return instances[0].instance

        current_weight = self.current_index % total_weight
        weight_sum = 0

        for instance in instances:
            weight_sum += instance.weight
            if current_weight < weight_sum:
                self.current_index += 1
                return instance.instance

        return instances[0].instance

    def _random(self, instances: List[ServiceInstance]) -> Any:
        """随机策略"""
        if not instances:
            return None

        import secrets
        selected_instance = secrets.choice(instances)
        return selected_instance.instance
