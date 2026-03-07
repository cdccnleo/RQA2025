"""
服务发现客户端

负责服务发现、缓存管理和负载均衡。

从service_discovery.py中提取以改善代码组织。

Author: RQA2025 Development Team
Date: 2025-11-01
"""

import logging
import time
import threading
import uuid
from typing import Dict, List, Optional
from enum import Enum

from .service_registry import (
    ServiceInstance,
    ServiceRegistry,
    ServiceStatus
)

logger = logging.getLogger(__name__)


class LoadBalanceStrategy(Enum):
    """负载均衡策略"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RANDOM = "random"
    CONSISTENT_HASH = "consistent_hash"
    HEALTH_BASED = "health_based"


class LoadBalancer:
    """
    负载均衡器
    
    支持多种负载均衡策略。
    """

    def __init__(self, strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN):
        self.strategy = strategy
        self._round_robin_index = 0
        self._connection_counts: Dict[str, int] = {}
        self._lock = threading.Lock()

    def select_instance(self, instances: List[ServiceInstance]) -> Optional[ServiceInstance]:
        """选择服务实例"""
        if not instances:
            return None

        if len(instances) == 1:
            return instances[0]

        with self._lock:
            if self.strategy == LoadBalanceStrategy.ROUND_ROBIN:
                return self._round_robin_select(instances)
            elif self.strategy == LoadBalanceStrategy.RANDOM:
                return self._random_select(instances)
            elif self.strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
                return self._least_connections_select(instances)
            elif self.strategy == LoadBalanceStrategy.WEIGHTED_ROUND_ROBIN:
                return self._weighted_round_robin_select(instances)
            elif self.strategy == LoadBalanceStrategy.HEALTH_BASED:
                return self._health_based_select(instances)
            else:
                return self._round_robin_select(instances)

    def _round_robin_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """轮询选择"""
        instance = instances[self._round_robin_index % len(instances)]
        self._round_robin_index += 1
        return instance

    def _random_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """随机选择"""
        import secrets
        return secrets.choice(instances)

    def _least_connections_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """最少连接数选择"""
        min_connections = float('inf')
        selected_instance = instances[0]

        for instance in instances:
            connections = self._connection_counts.get(instance.service_id, 0)
            if connections < min_connections:
                min_connections = connections
                selected_instance = instance

        return selected_instance

    def _weighted_round_robin_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """加权轮询选择"""
        # 简化实现：根据权重创建加权列表
        weighted_instances = []
        for instance in instances:
            weight = instance.weight or 100
            weighted_instances.extend([instance] * (weight // 10))

        if weighted_instances:
            return self._round_robin_select(weighted_instances)
        else:
            return instances[0]

    def _health_based_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """基于健康状态选择"""
        # 优先选择健康的服务
        healthy_instances = [
            instance for instance in instances
            if instance.status == ServiceStatus.HEALTHY
        ]

        if healthy_instances:
            return self._round_robin_select(healthy_instances)
        else:
            return instances[0]

    def increment_connections(self, service_id: str):
        """增加连接数"""
        with self._lock:
            self._connection_counts[service_id] = self._connection_counts.get(service_id, 0) + 1

    def decrement_connections(self, service_id: str):
        """减少连接数"""
        with self._lock:
            if service_id in self._connection_counts:
                self._connection_counts[service_id] = max(
                    0, self._connection_counts[service_id] - 1)


class ServiceDiscoveryClient:
    """
    服务发现客户端
    
    负责:
    1. 服务发现和缓存
    2. 负载均衡
    3. 连接管理
    """

    def __init__(self, registry: ServiceRegistry,
                 client_id: Optional[str] = None):
        self.registry = registry
        self.client_id = client_id or str(uuid.uuid4())
        self._service_cache: Dict[str, List[ServiceInstance]] = {}
        self._cache_lock = threading.RLock()
        self._cache_ttl = 60  # 缓存TTL为60秒
        self._cache_timestamps: Dict[str, float] = {}

        # 负载均衡器
        self._load_balancers: Dict[str, LoadBalancer] = {}

        logger.info(f"服务发现客户端初始化: {self.client_id}")

    def discover(self, service_name: str,
                 strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN,
                 use_cache: bool = True) -> Optional[ServiceInstance]:
        """发现并返回一个服务实例"""
        services = self.discover_all(service_name, use_cache)

        if not services:
            return None

        # 获取负载均衡器
        if service_name not in self._load_balancers:
            self._load_balancers[service_name] = LoadBalancer(strategy)

        lb = self._load_balancers[service_name]
        return lb.select_instance(services)

    def discover_all(self, service_name: str,
                     use_cache: bool = True) -> List[ServiceInstance]:
        """发现所有服务实例"""
        if use_cache and self._is_cache_valid(service_name):
            with self._cache_lock:
                return self._service_cache.get(service_name, []).copy()

        # 从注册中心获取服务
        services = self.registry.get_healthy_services(service_name)

        # 更新缓存
        if use_cache:
            with self._cache_lock:
                self._service_cache[service_name] = services.copy()
                self._cache_timestamps[service_name] = time.time()

        return services

    def invalidate_cache(self, service_name: Optional[str] = None):
        """使缓存失效"""
        with self._cache_lock:
            if service_name:
                if service_name in self._service_cache:
                    del self._service_cache[service_name]
                if service_name in self._cache_timestamps:
                    del self._cache_timestamps[service_name]
            else:
                self._service_cache.clear()
                self._cache_timestamps.clear()

    def _is_cache_valid(self, service_name: str) -> bool:
        """检查缓存是否有效"""
        if service_name not in self._cache_timestamps:
            return False

        cache_time = self._cache_timestamps[service_name]
        return time.time() - cache_time < self._cache_ttl

    def set_cache_ttl(self, ttl: int):
        """设置缓存TTL"""
        self._cache_ttl = ttl

    def get_load_balancer(self, service_name: str) -> Optional[LoadBalancer]:
        """获取指定服务的负载均衡器"""
        return self._load_balancers.get(service_name)


__all__ = [
    'LoadBalanceStrategy',
    'LoadBalancer',
    'ServiceDiscoveryClient'
]

