#!/usr/bin/env python3
"""
服务注册发现

实现微服务架构中的服务注册、服务发现、健康检查和负载均衡
"""

import logging
import json
import time
import threading
import socket
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import uuid

logger = logging.getLogger(__name__)

# 尝试导入Redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis不可用，将使用内存存储")


class ServiceStatus(Enum):

    """服务状态"""
    UP = "up"
    DOWN = "down"
    MAINTENANCE = "maintenance"
    UNKNOWN = "unknown"


class DiscoveryType(Enum):

    """发现类型"""
    DNS = "dns"
    STATIC = "static"
    REDIS = "redis"
    ETCD = "etcd"
    CONSUL = "consul"


@dataclass
class ServiceInstance:

    """服务实例"""
    service_id: str
    service_name: str
    host: str
    port: int
    protocol: str = "http"
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: ServiceStatus = ServiceStatus.UP
    registered_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    ttl: int = 30  # 生存时间(秒)
    weight: int = 1  # 负载权重
    version: str = "1.0.0"

    @property
    def url(self) -> str:
        """服务URL"""
        return f"{self.protocol}://{self.host}:{self.port}"

    @property
    def is_expired(self) -> bool:
        """是否过期"""
        return (datetime.now() - self.last_heartbeat).seconds > self.ttl

    @property
    def health_check_url(self) -> Optional[str]:
        """健康检查URL"""
        health_path = self.metadata.get('health_check_path', '/health')
        return f"{self.url}{health_path}"


@dataclass
class ServiceRegistry:

    """服务注册表"""
    service_name: str
    instances: List[ServiceInstance] = field(default_factory=list)

    @property
    def healthy_instances(self) -> List[ServiceInstance]:
        """健康的实例"""
        return [inst for inst in self.instances if inst.status == ServiceStatus.UP and not inst.is_expired]

    @property
    def total_weight(self) -> int:
        """总权重"""
        return sum(inst.weight for inst in self.healthy_instances)


class HealthChecker:

    """健康检查器"""

    def __init__(self, check_interval: int = 10, timeout: int = 5):

        self.check_interval = check_interval
        self.timeout = timeout
        self.executor = ThreadPoolExecutor(max_workers=10)

    def start_health_check(self, registry: Dict[str, ServiceRegistry],


                           callback: Optional[Callable[[str, ServiceInstance, bool], None]] = None):
        """启动健康检查"""

        def health_check_worker():

            while True:
                try:
                    self._perform_health_checks(registry, callback)
                    time.sleep(self.check_interval)
                except Exception as e:
                    logger.error(f"健康检查异常: {e}")
                    time.sleep(self.check_interval)

        thread = threading.Thread(target=health_check_worker, daemon=True)
        thread.start()
        logger.info("健康检查线程已启动")

    def _perform_health_checks(self, registry: Dict[str, ServiceRegistry],


                               callback: Optional[Callable] = None):
        """执行健康检查"""
        futures = []

        # 收集所有需要检查的实例
        for service_name, service_registry in registry.items():
            for instance in service_registry.instances:
                if instance.status != ServiceStatus.MAINTENANCE:
                    future = self.executor.submit(
                        self._check_instance_health,
                        service_name,
                        instance,
                        callback
                    )
                    futures.append(future)

        # 等待所有检查完成
        for future in futures:
            try:
                future.result(timeout=self.timeout + 1)
            except Exception as e:
                logger.error(f"健康检查任务异常: {e}")

    def _check_instance_health(self, service_name: str, instance: ServiceInstance,


                               callback: Optional[Callable] = None) -> bool:
        """检查实例健康状态"""
        try:
            if not instance.health_check_url:
                # 如果没有健康检查URL，使用简单的连接测试
                return self._check_tcp_connection(instance.host, instance.port)

            # HTTP健康检查
            import requests
            response = requests.get(
                instance.health_check_url,
                timeout=self.timeout,
                headers={'User - Agent': 'ServiceDiscovery - HealthCheck'}
            )

            is_healthy = response.status_code == 200
            previous_status = instance.status

            if is_healthy:
                instance.status = ServiceStatus.UP
                instance.last_heartbeat = datetime.now()
            else:
                instance.status = ServiceStatus.DOWN

            # 如果状态发生变化，调用回调
            if callback and previous_status != instance.status:
                callback(service_name, instance, is_healthy)

            return is_healthy

        except Exception as e:
            logger.warning(f"健康检查失败 {instance.service_id}: {e}")
            instance.status = ServiceStatus.DOWN

            if callback:
                callback(service_name, instance, False)

            return False

    def _check_tcp_connection(self, host: str, port: int) -> bool:
        """检查TCP连接"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception:
            return False


class LoadBalancer:

    """负载均衡器"""

    def __init__(self, algorithm: str = "round_robin"):

        self.algorithm = algorithm
        self.current_index = 0
        self.lock = threading.Lock()

    def select_instance(self, instances: List[ServiceInstance]) -> Optional[ServiceInstance]:
        """选择服务实例"""
        healthy_instances = [inst for inst in instances if inst.status == ServiceStatus.UP]

        if not healthy_instances:
            return None

        with self.lock:
            if self.algorithm == "round_robin":
                instance = healthy_instances[self.current_index % len(healthy_instances)]
                self.current_index += 1
                return instance
            elif self.algorithm == "weighted_round_robin":
                return self._weighted_round_robin(healthy_instances)
            elif self.algorithm == "random":
                import secrets
                return secrets.choice(healthy_instances)
            elif self.algorithm == "least_connections":
                # 简化实现，实际应该跟踪连接数
                return healthy_instances[0]
            else:
                return healthy_instances[0]

    def _weighted_round_robin(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """加权轮询"""
        total_weight = sum(inst.weight for inst in instances)
        if total_weight == 0:
            return instances[0]

        # 简单实现：根据权重随机选择
        import secrets
        r = secrets.uniform(0, total_weight)
        current_weight = 0

        for instance in instances:
            current_weight += instance.weight
            if r <= current_weight:
                return instance

        return instances[0]


class ServiceDiscovery:

    """服务发现"""

    def __init__(self, discovery_type: DiscoveryType = DiscoveryType.REDIS,


                 config: Optional[Dict[str, Any]] = None):
        self.discovery_type = discovery_type
        self.config = config or {}

        # 服务注册表
        self.registry: Dict[str, ServiceRegistry] = {}
        self.lock = threading.RLock()

        # 健康检查器
        self.health_checker = HealthChecker(
            check_interval=self.config.get('health_check_interval', 10),
            timeout=self.config.get('health_check_timeout', 5)
        )

        # 负载均衡器
        self.load_balancer = LoadBalancer(
            algorithm=self.config.get('load_balance_algorithm', 'round_robin')
        )

        # 存储后端
        self.storage = self._initialize_storage()

        # 启动健康检查
        self.health_checker.start_health_check(self.registry, self._health_check_callback)

        logger.info(f"服务发现初始化完成，使用 {discovery_type.value} 后端")

    def _initialize_storage(self):
        """初始化存储后端"""
        if self.discovery_type == DiscoveryType.REDIS and REDIS_AVAILABLE:
            return RedisStorage(self.config)
        elif self.discovery_type == DiscoveryType.STATIC:
            return StaticStorage(self.config)
        else:
            return MemoryStorage(self.config)

    def register_service(self, instance: ServiceInstance) -> bool:
        """注册服务"""
        with self.lock:
            try:
                # 添加到本地注册表
                if instance.service_name not in self.registry:
                    self.registry[instance.service_name] = ServiceRegistry(instance.service_name)

                self.registry[instance.service_name].instances.append(instance)

                # 持久化存储
                if hasattr(self.storage, 'save_instance'):
                    self.storage.save_instance(instance)

                logger.info(f"服务已注册: {instance.service_id} ({instance.service_name})")
                return True

            except Exception as e:
                logger.error(f"服务注册失败: {e}")
                return False

    def unregister_service(self, service_id: str, service_name: str) -> bool:
        """注销服务"""
        with self.lock:
            try:
                if service_name in self.registry:
                    instances = self.registry[service_name].instances
                    self.registry[service_name].instances = [
                        inst for inst in instances if inst.service_id != service_id
                    ]

                    # 持久化存储
                    if hasattr(self.storage, 'delete_instance'):
                        self.storage.delete_instance(service_id, service_name)

                logger.info(f"服务已注销: {service_id} ({service_name})")
                return True

            except Exception as e:
                logger.error(f"服务注销失败: {e}")
                return False

    def discover_service(self, service_name: str) -> Optional[ServiceInstance]:
        """发现服务"""
        with self.lock:
            try:
                service_registry = self.registry.get(service_name)
                if not service_registry:
                    # 尝试从存储加载
                    if hasattr(self.storage, 'load_service'):
                        instances = self.storage.load_service(service_name)
                        if instances:
                            self.registry[service_name] = ServiceRegistry(service_name, instances)
                            service_registry = self.registry[service_name]

                if not service_registry:
                    return None

                return self.load_balancer.select_instance(service_registry.instances)

            except Exception as e:
                logger.error(f"服务发现失败: {e}")
                return None

    def get_service_instances(self, service_name: str) -> List[ServiceInstance]:
        """获取服务的所有实例"""
        with self.lock:
            service_registry = self.registry.get(service_name)
            if service_registry:
                return service_registry.instances.copy()
            return []

    def get_all_services(self) -> Dict[str, List[ServiceInstance]]:
        """获取所有服务"""
        with self.lock:
            return {
                name: registry.instances.copy()
                for name, registry in self.registry.items()
            }

    def update_service_status(self, service_id: str, service_name: str, status: ServiceStatus):
        """更新服务状态"""
        with self.lock:
            try:
                service_registry = self.registry.get(service_name)
                if service_registry:
                    for instance in service_registry.instances:
                        if instance.service_id == service_id:
                            instance.status = status
                            logger.info(f"服务状态已更新: {service_id} -> {status.value}")
                            return True
                return False

            except Exception as e:
                logger.error(f"更新服务状态失败: {e}")
                return False

    def send_heartbeat(self, service_id: str, service_name: str):
        """发送心跳"""
        with self.lock:
            try:
                service_registry = self.registry.get(service_name)
                if service_registry:
                    for instance in service_registry.instances:
                        if instance.service_id == service_id:
                            instance.last_heartbeat = datetime.now()
                            return True
                return False

            except Exception as e:
                logger.error(f"发送心跳失败: {e}")
                return False

    def _health_check_callback(self, service_name: str, instance: ServiceInstance, is_healthy: bool):
        """健康检查回调"""
        status = ServiceStatus.UP if is_healthy else ServiceStatus.DOWN
        logger.info(f"健康检查结果: {instance.service_id} ({service_name}) -> {status.value}")

        # 更新存储
        if hasattr(self.storage, 'update_instance_status'):
            try:
                self.storage.update_instance_status(instance.service_id, service_name, status)
            except Exception as e:
                logger.error(f"更新存储状态失败: {e}")

    def cleanup_expired_services(self):
        """清理过期服务"""
        with self.lock:
            try:
                for service_name, service_registry in self.registry.items():
                    original_count = len(service_registry.instances)
                    service_registry.instances = [
                        inst for inst in service_registry.instances if not inst.is_expired
                    ]

                    removed_count = original_count - len(service_registry.instances)
                    if removed_count > 0:
                        logger.info(f"清理过期服务: {service_name} 移除了 {removed_count} 个实例")

            except Exception as e:
                logger.error(f"清理过期服务失败: {e}")

    def get_service_stats(self) -> Dict[str, Any]:
        """获取服务统计"""
        with self.lock:
            stats = {
                'total_services': len(self.registry),
                'total_instances': sum(len(reg.instances) for reg in self.registry.values()),
                'healthy_instances': sum(len(reg.healthy_instances) for reg in self.registry.values()),
                'services': {}
            }

            for name, registry in self.registry.items():
                stats['services'][name] = {
                    'total_instances': len(registry.instances),
                    'healthy_instances': len(registry.healthy_instances),
                    'total_weight': registry.total_weight
                }

            return stats


class MemoryStorage:

    """内存存储"""

    def __init__(self, config: Dict[str, Any]):

        self.instances: Dict[str, List[ServiceInstance]] = {}

    def save_instance(self, instance: ServiceInstance):
        """保存实例"""
        if instance.service_name not in self.instances:
            self.instances[instance.service_name] = []
        self.instances[instance.service_name].append(instance)

    def delete_instance(self, service_id: str, service_name: str):
        """删除实例"""
        if service_name in self.instances:
            self.instances[service_name] = [
                inst for inst in self.instances[service_name] if inst.service_id != service_id
            ]

    def load_service(self, service_name: str) -> List[ServiceInstance]:
        """加载服务"""
        return self.instances.get(service_name, [])

    def update_instance_status(self, service_id: str, service_name: str, status: ServiceStatus):
        """更新实例状态"""
        if service_name in self.instances:
            for instance in self.instances[service_name]:
                if instance.service_id == service_id:
                    instance.status = status


class RedisStorage:

    """Redis存储"""

    def __init__(self, config: Dict[str, Any]):

        if not REDIS_AVAILABLE:
            raise ImportError("Redis不可用")

        self.redis = redis.Redis(
            host=config.get('redis_host', 'localhost'),
            port=config.get('redis_port', 6379),
            db=config.get('redis_db', 0),
            password=config.get('redis_password')
        )

        self.key_prefix = config.get('redis_key_prefix', 'service_discovery')

    def _get_service_key(self, service_name: str) -> str:
        """获取服务键"""
        return f"{self.key_prefix}:service:{service_name}"

    def _get_instance_key(self, service_id: str, service_name: str) -> str:
        """获取实例键"""
        return f"{self.key_prefix}:instance:{service_name}:{service_id}"

    def save_instance(self, instance: ServiceInstance):
        """保存实例"""
        try:
            instance_key = self._get_instance_key(instance.service_id, instance.service_name)
            service_key = self._get_service_key(instance.service_name)

            # 保存实例详情
            instance_data = {
                'service_id': instance.service_id,
                'service_name': instance.service_name,
                'host': instance.host,
                'port': instance.port,
                'protocol': instance.protocol,
                'metadata': json.dumps(instance.metadata),
                'status': instance.status.value,
                'registered_at': instance.registered_at.isoformat(),
                'last_heartbeat': instance.last_heartbeat.isoformat(),
                'ttl': instance.ttl,
                'weight': instance.weight,
                'version': instance.version
            }

            self.redis.hmset(instance_key, instance_data)
            self.redis.expire(instance_key, instance.ttl)

            # 添加到服务集合
            self.redis.sadd(service_key, instance.service_id)

        except Exception as e:
            logger.error(f"保存实例到Redis失败: {e}")

    def delete_instance(self, service_id: str, service_name: str):
        """删除实例"""
        try:
            instance_key = self._get_instance_key(service_id, service_name)
            service_key = self._get_service_key(service_name)

            self.redis.delete(instance_key)
            self.redis.srem(service_key, service_id)

        except Exception as e:
            logger.error(f"从Redis删除实例失败: {e}")

    def load_service(self, service_name: str) -> List[ServiceInstance]:
        """加载服务"""
        try:
            service_key = self._get_service_key(service_name)
            instance_ids = self.redis.smembers(service_key)

            instances = []
            for instance_id in instance_ids:
                instance_id = instance_id.decode() if isinstance(instance_id, bytes) else instance_id
                instance_key = self._get_instance_key(instance_id, service_name)

                instance_data = self.redis.hgetall(instance_key)
                if instance_data:
                    instance = self._deserialize_instance(instance_data)
                    instances.append(instance)

            return instances

        except Exception as e:
            logger.error(f"从Redis加载服务失败: {e}")
            return []

    def update_instance_status(self, service_id: str, service_name: str, status: ServiceStatus):
        """更新实例状态"""
        try:
            instance_key = self._get_instance_key(service_id, service_name)
            self.redis.hset(instance_key, 'status', status.value)

        except Exception as e:
            logger.error(f"更新Redis实例状态失败: {e}")

    def _deserialize_instance(self, data: Dict) -> ServiceInstance:
        """反序列化实例"""
        metadata = json.loads(data.get(b'metadata', b'{}').decode()) if isinstance(
            data.get(b'metadata'), bytes) else data.get('metadata', {})

        return ServiceInstance(
            service_id=data.get(b'service_id', b'').decode() if isinstance(
                data.get(b'service_id'), bytes) else data.get('service_id', ''),
            service_name=data.get(b'service_name', b'').decode() if isinstance(
                data.get(b'service_name'), bytes) else data.get('service_name', ''),
            host=data.get(b'host', b'').decode() if isinstance(
                data.get(b'host'), bytes) else data.get('host', ''),
            port=int(data.get(b'port', b'0').decode() if isinstance(
                data.get(b'port'), bytes) else data.get('port', 0)),
            protocol=data.get(b'protocol', b'http').decode() if isinstance(
                data.get(b'protocol'), bytes) else data.get('protocol', 'http'),
            metadata=metadata,
            status=ServiceStatus(data.get(b'status', b'up').decode() if isinstance(
                data.get(b'status'), bytes) else data.get('status', 'up')),
            ttl=int(data.get(b'ttl', b'30').decode() if isinstance(
                data.get(b'ttl'), bytes) else data.get('ttl', 30)),
            weight=int(data.get(b'weight', b'1').decode() if isinstance(
                data.get(b'weight'), bytes) else data.get('weight', 1)),
            version=data.get(b'version', b'1.0.0').decode() if isinstance(
                data.get(b'version'), bytes) else data.get('version', '1.0.0')
        )


class StaticStorage:

    """静态存储"""

    def __init__(self, config: Dict[str, Any]):

        self.services = config.get('static_services', {})

    def load_service(self, service_name: str) -> List[ServiceInstance]:
        """加载服务"""
        service_config = self.services.get(service_name, [])
        instances = []

        for config in service_config:
            instance = ServiceInstance(
                service_id=config.get('service_id', str(uuid.uuid4())),
                service_name=service_name,
                host=config['host'],
                port=config['port'],
                protocol=config.get('protocol', 'http'),
                metadata=config.get('metadata', {}),
                weight=config.get('weight', 1),
                version=config.get('version', '1.0.0')
            )
            instances.append(instance)

        return instances
