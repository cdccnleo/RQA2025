#!/usr/bin/env python3
"""
RQA2025 负载均衡器组件

提供企业级的负载均衡功能，支持多种负载均衡算法，
实现服务实例的智能分配和健康检查。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time
import threading
import logging
import random
import hashlib

from ...base import ComponentStatus, ComponentHealth
from ...patterns.standard_interface_template import StandardComponent

logger = logging.getLogger(__name__)


class LoadBalancingAlgorithm(Enum):
    """负载均衡算法枚举"""
    ROUND_ROBIN = "round_robin"           # 轮询
    WEIGHTED_ROUND_ROBIN = "weighted_rr"  # 加权轮询
    LEAST_CONNECTIONS = "least_conn"      # 最少连接
    WEIGHTED_LEAST_CONNECTIONS = "weighted_lc"  # 加权最少连接
    RANDOM = "random"                     # 随机
    IP_HASH = "ip_hash"                   # IP哈希
    LEAST_RESPONSE_TIME = "least_rt"      # 最少响应时间


@dataclass
class ServiceInstance:
    """服务实例"""
    id: str
    host: str
    port: int
    weight: int = 1
    active_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    response_time: float = 0.0
    last_health_check: datetime = field(default_factory=datetime.now)
    is_healthy: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def address(self) -> str:
        """获取实例地址"""
        return f"{self.host}:{self.port}"

    @property
    def failure_rate(self) -> float:
        """计算失败率"""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests

    @property
    def load_score(self) -> float:
        """计算负载评分 (0-1, 越高负载越重)"""
        # 基于连接数、响应时间、失败率计算综合负载
        conn_score = min(self.active_connections / 10.0, 1.0)  # 假设最大10个连接
        rt_score = min(self.response_time / 1000.0, 1.0)      # 假设最大1秒响应时间
        failure_score = self.failure_rate

        return (conn_score * 0.5 + rt_score * 0.3 + failure_score * 0.2)


@dataclass
class LoadBalancingStats:
    """负载均衡统计"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    current_instances: int = 0
    healthy_instances: int = 0
    unhealthy_instances: int = 0


class LoadBalancingStrategy(ABC):
    """负载均衡策略基类"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def select_instance(self, instances: List[ServiceInstance],
                        request_context: Optional[Dict[str, Any]] = None) -> Optional[ServiceInstance]:
        """选择服务实例"""

    @abstractmethod
    def get_algorithm_type(self) -> LoadBalancingAlgorithm:
        """获取算法类型"""


class RoundRobinStrategy(LoadBalancingStrategy):
    """轮询策略"""

    def __init__(self):
        super().__init__("Round Robin")
        self._current_index = 0
        self._lock = threading.Lock()

    def select_instance(self, instances: List[ServiceInstance],
                        request_context: Optional[Dict[str, Any]] = None) -> Optional[ServiceInstance]:
        """轮询选择实例"""
        healthy_instances = [inst for inst in instances if inst.is_healthy]
        if not healthy_instances:
            return None

        with self._lock:
            instance = healthy_instances[self._current_index % len(healthy_instances)]
            self._current_index += 1
            return instance

    def get_algorithm_type(self) -> LoadBalancingAlgorithm:
        return LoadBalancingAlgorithm.ROUND_ROBIN


class WeightedRoundRobinStrategy(LoadBalancingStrategy):
    """加权轮询策略"""

    def __init__(self):
        super().__init__("Weighted Round Robin")
        self._current_weight = 0
        self._lock = threading.Lock()

    def select_instance(self, instances: List[ServiceInstance],
                        request_context: Optional[Dict[str, Any]] = None) -> Optional[ServiceInstance]:
        """加权轮询选择实例"""
        healthy_instances = [inst for inst in instances if inst.is_healthy and inst.weight > 0]
        if not healthy_instances:
            return None

        with self._lock:
            total_weight = sum(inst.weight for inst in healthy_instances)

            while True:
                for instance in healthy_instances:
                    if self._current_weight < instance.weight:
                        self._current_weight += 1
                        return instance

                self._current_weight -= total_weight
                if self._current_weight <= 0:
                    self._current_weight = 0
                    break

        return healthy_instances[0]  # 默认返回第一个


class LeastConnectionsStrategy(LoadBalancingStrategy):
    """最少连接策略"""

    def __init__(self):
        super().__init__("Least Connections")

    def select_instance(self, instances: List[ServiceInstance],
                        request_context: Optional[Dict[str, Any]] = None) -> Optional[ServiceInstance]:
        """选择连接数最少的实例"""
        healthy_instances = [inst for inst in instances if inst.is_healthy]
        if not healthy_instances:
            return None

        return min(healthy_instances, key=lambda inst: inst.active_connections)


class RandomStrategy(LoadBalancingStrategy):
    """随机策略"""

    def __init__(self):
        super().__init__("Random")

    def select_instance(self, instances: List[ServiceInstance],
                        request_context: Optional[Dict[str, Any]] = None) -> Optional[ServiceInstance]:
        """随机选择实例"""
        healthy_instances = [inst for inst in instances if inst.is_healthy]
        if not healthy_instances:
            return None

        return random.choice(healthy_instances)

    def get_algorithm_type(self) -> LoadBalancingAlgorithm:
        return LoadBalancingAlgorithm.RANDOM


class IpHashStrategy(LoadBalancingStrategy):
    """IP哈希策略"""

    def __init__(self):
        super().__init__("IP Hash")

    def select_instance(self, instances: List[ServiceInstance],
                        request_context: Optional[Dict[str, Any]] = None) -> Optional[ServiceInstance]:
        """基于IP哈希选择实例"""
        healthy_instances = [inst for inst in instances if inst.is_healthy]
        if not healthy_instances:
            return None

        # 从请求上下文中获取客户端IP
        client_ip = request_context.get(
            'client_ip', '127.0.0.1') if request_context else '127.0.0.1'

        # 计算哈希值
        hash_value = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
        index = hash_value % len(healthy_instances)

        return healthy_instances[index]

    def get_algorithm_type(self) -> LoadBalancingAlgorithm:
        return LoadBalancingAlgorithm.IP_HASH


class LoadBalancer(StandardComponent):
    """企业级负载均衡器"""

    def __init__(self, service_name: str, algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.ROUND_ROBIN):
        """初始化负载均衡器

        Args:
            service_name: 服务名称
            algorithm: 负载均衡算法
        """
        super().__init__(service_name, "2.0.0", f"{service_name}负载均衡器")
        self._instances: Dict[str, ServiceInstance] = {}
        self._algorithm = algorithm
        self._strategy = self._create_strategy(algorithm)
        self._stats = LoadBalancingStats()

        # 健康检查配置
        self.health_check_interval = 30  # 秒
        self.health_check_timeout = 5    # 秒
        self.unhealthy_threshold = 3     # 连续失败次数
        self.healthy_threshold = 2       # 连续成功次数

        # 监控和线程安全
        self._lock = threading.RLock()
        self._health_check_thread: Optional[threading.Thread] = None
        self._running = False

        # 事件回调
        self._instance_change_callbacks: List[Callable] = []
        self._health_change_callbacks: List[Callable] = []

    def _create_strategy(self, algorithm: LoadBalancingAlgorithm) -> LoadBalancingStrategy:
        """创建负载均衡策略"""
        strategies = {
            LoadBalancingAlgorithm.ROUND_ROBIN: RoundRobinStrategy,
            LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN: WeightedRoundRobinStrategy,
            LoadBalancingAlgorithm.LEAST_CONNECTIONS: LeastConnectionsStrategy,
            LoadBalancingAlgorithm.RANDOM: RandomStrategy,
            LoadBalancingAlgorithm.IP_HASH: IpHashStrategy,
        }

        strategy_class = strategies.get(algorithm)
        if not strategy_class:
            raise ValueError(f"不支持的负载均衡算法: {algorithm}")

        return strategy_class()

    def initialize(self) -> bool:
        """初始化负载均衡器"""
        try:
            self.set_status(ComponentStatus.INITIALIZING)

            # 启动健康检查线程
            self._running = True
            self._health_check_thread = threading.Thread(
                target=self._health_check_loop,
                name=f"{self.service_name}_health_check",
                daemon=True
            )
            self._health_check_thread.start()

            self.set_status(ComponentStatus.INITIALIZED)
            self.set_health(ComponentHealth.HEALTHY)

            logger.info(f"负载均衡器 {self.service_name} 初始化完成")
            return True

        except Exception as e:
            self.set_status(ComponentStatus.ERROR)
            self.set_health(ComponentHealth.UNHEALTHY)
            logger.error(f"负载均衡器 {self.service_name} 初始化失败: {e}")
            return False

    def shutdown(self) -> bool:
        """关闭负载均衡器"""
        try:
            self.set_status(ComponentStatus.STOPPING)
            self._running = False

            if self._health_check_thread and self._health_check_thread.is_alive():
                self._health_check_thread.join(timeout=5)

            self.set_status(ComponentStatus.STOPPED)
            logger.info(f"负载均衡器 {self.service_name} 已关闭")
            return True

        except Exception as e:
            logger.error(f"负载均衡器 {self.service_name} 关闭失败: {e}")
            return False

    def register_instance(self, instance_id: str, host: str, port: int,
                          weight: int = 1, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """注册服务实例"""
        try:
            with self._lock:
                if instance_id in self._instances:
                    logger.warning(f"服务实例 {instance_id} 已存在，将更新信息")
                    return self.update_instance(instance_id, host, port, weight, metadata)

                instance = ServiceInstance(
                    id=instance_id,
                    host=host,
                    port=port,
                    weight=weight,
                    metadata=metadata or {}
                )

                self._instances[instance_id] = instance
                self._update_stats()
                self._notify_instance_change("register", instance)

                logger.info(f"注册服务实例: {instance_id} -> {instance.address}")
                return True

        except Exception as e:
            logger.error(f"注册服务实例失败 {instance_id}: {e}")
            return False

    def unregister_instance(self, instance_id: str) -> bool:
        """注销服务实例"""
        try:
            with self._lock:
                if instance_id not in self._instances:
                    logger.warning(f"服务实例 {instance_id} 不存在")
                    return False

                instance = self._instances.pop(instance_id)
                self._update_stats()
                self._notify_instance_change("unregister", instance)

                logger.info(f"注销服务实例: {instance_id}")
                return True

        except Exception as e:
            logger.error(f"注销服务实例失败 {instance_id}: {e}")
            return False

    def update_instance(self, instance_id: str, host: Optional[str] = None,
                        port: Optional[int] = None, weight: Optional[int] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> bool:
        """更新服务实例信息"""
        try:
            with self._lock:
                if instance_id not in self._instances:
                    return False

                instance = self._instances[instance_id]

                if host is not None:
                    instance.host = host
                if port is not None:
                    instance.port = port
                if weight is not None:
                    instance.weight = weight
                if metadata is not None:
                    instance.metadata.update(metadata)

                logger.info(f"更新服务实例: {instance_id}")
                return True

        except Exception as e:
            logger.error(f"更新服务实例失败 {instance_id}: {e}")
            return False

    def select_instance(self, request_context: Optional[Dict[str, Any]] = None) -> Optional[ServiceInstance]:
        """选择服务实例"""
        try:
            with self._lock:
                instances = list(self._instances.values())
                if not instances:
                    return None

                selected = self._strategy.select_instance(instances, request_context)
                if selected:
                    selected.total_requests += 1
                    selected.active_connections += 1
                    self._stats.total_requests += 1

                return selected

        except Exception as e:
            logger.error(f"选择服务实例失败: {e}")
            self._stats.failed_requests += 1
            return None

    def release_connection(self, instance_id: str):
        """释放连接"""
        try:
            with self._lock:
                if instance_id in self._instances:
                    instance = self._instances[instance_id]
                    if instance.active_connections > 0:
                        instance.active_connections -= 1
        except Exception as e:
            logger.error(f"释放连接失败 {instance_id}: {e}")

    def record_response_time(self, instance_id: str, response_time: float, success: bool = True):
        """记录响应时间"""
        try:
            with self._lock:
                if instance_id in self._instances:
                    instance = self._instances[instance_id]
                    instance.response_time = response_time

                    if success:
                        self._stats.successful_requests += 1
                    else:
                        instance.failed_requests += 1
                        self._stats.failed_requests += 1

                    # 更新平均响应时间
                    total_requests = self._stats.successful_requests + self._stats.failed_requests
                    if total_requests > 0:
                        # 简单移动平均
                        self._stats.avg_response_time = (
                            self._stats.avg_response_time * 0.9 + response_time * 0.1
                        )

        except Exception as e:
            logger.error(f"记录响应时间失败 {instance_id}: {e}")

    def get_instances(self) -> List[ServiceInstance]:
        """获取所有服务实例"""
        with self._lock:
            return list(self._instances.values())

    def get_healthy_instances(self) -> List[ServiceInstance]:
        """获取健康的服务实例"""
        with self._lock:
            return [inst for inst in self._instances.values() if inst.is_healthy]

    def get_stats(self) -> LoadBalancingStats:
        """获取负载均衡统计信息"""
        with self._lock:
            return LoadBalancingStats(
                total_requests=self._stats.total_requests,
                successful_requests=self._stats.successful_requests,
                failed_requests=self._stats.failed_requests,
                avg_response_time=self._stats.avg_response_time,
                current_instances=len(self._instances),
                healthy_instances=len(
                    [inst for inst in self._instances.values() if inst.is_healthy]),
                unhealthy_instances=len(
                    [inst for inst in self._instances.values() if not inst.is_healthy])
            )

    def set_algorithm(self, algorithm: LoadBalancingAlgorithm):
        """设置负载均衡算法"""
        with self._lock:
            if algorithm != self._algorithm:
                self._algorithm = algorithm
                self._strategy = self._create_strategy(algorithm)
                logger.info(f"负载均衡器 {self.service_name} 算法变更为: {algorithm.value}")

    def add_instance_change_callback(self, callback: Callable):
        """添加实例变更回调"""
        self._instance_change_callbacks.append(callback)

    def add_health_change_callback(self, callback: Callable):
        """添加健康状态变更回调"""
        self._health_change_callbacks.append(callback)

    def _update_stats(self):
        """更新统计信息"""
        self._stats.current_instances = len(self._instances)
        self._stats.healthy_instances = len(
            [inst for inst in self._instances.values() if inst.is_healthy])
        self._stats.unhealthy_instances = len(
            [inst for inst in self._instances.values() if not inst.is_healthy])

    def _notify_instance_change(self, action: str, instance: ServiceInstance):
        """通知实例变更"""
        for callback in self._instance_change_callbacks:
            try:
                callback(action, instance)
            except Exception as e:
                logger.error(f"实例变更回调执行失败: {e}")

    def _notify_health_change(self, instance: ServiceInstance, old_status: bool):
        """通知健康状态变更"""
        for callback in self._health_change_callbacks:
            try:
                callback(instance, old_status)
            except Exception as e:
                logger.error(f"健康状态变更回调执行失败: {e}")

    def _health_check_loop(self):
        """健康检查循环"""
        while self._running:
            try:
                self._perform_health_checks()
                time.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"健康检查循环异常: {e}")
                time.sleep(5)  # 出错时等待5秒再继续

    def _perform_health_checks(self):
        """执行健康检查"""
        for instance in list(self._instances.values()):
            try:
                is_healthy = self._check_instance_health(instance)

                old_status = instance.is_healthy
                if is_healthy != old_status:
                    instance.is_healthy = is_healthy
                    instance.last_health_check = datetime.now()
                    self._notify_health_change(instance, old_status)
                    self._update_stats()

                    status_str = "健康" if is_healthy else "不健康"
                    logger.info(f"服务实例 {instance.id} 状态变更为: {status_str}")

            except Exception as e:
                logger.error(f"检查实例 {instance.id} 健康状态失败: {e}")

    def _check_instance_health(self, instance: ServiceInstance) -> bool:
        """检查服务实例健康状态"""
        # 这里应该实现实际的健康检查逻辑
        # 例如HTTP请求、TCP连接等
        # 目前使用模拟检查
        try:
            # 模拟网络检查延迟
            time.sleep(0.001)

            # 简单检查：基于失败率判断
            if instance.failure_rate > 0.5:  # 失败率超过50%认为不健康
                return False

            # 检查最后健康检查时间
            if datetime.now() - instance.last_health_check > timedelta(minutes=5):
                return False

            return True

        except Exception:
            return False

    def _perform_health_check(self) -> Dict[str, Any]:
        """执行健康检查（StandardComponent要求）"""
        try:
            healthy_count = len([inst for inst in self._instances.values() if inst.is_healthy])
            total_count = len(self._instances)

            health_status = {
                'component_name': self.name,  # 使用StandardComponent的name属性
                'status': 'healthy' if healthy_count > 0 else 'unhealthy',
                'healthy_instances': healthy_count,
                'total_instances': total_count,
                'availability': healthy_count / total_count if total_count > 0 else 0.0,
                'last_check': datetime.now().isoformat(),
                'active_connections': sum(inst.active_connections for inst in self._instances.values()),
                'total_requests': self._stats.total_requests,
                'success_rate': (self._stats.successful_requests /
                                 max(self._stats.total_requests, 1))
            }

            return health_status

        except Exception as e:
            logger.error(f"负载均衡器健康检查失败: {e}")
            return {
                'component_name': self.name,  # 使用StandardComponent的name属性
                'status': 'error',
                'error': str(e),
                'last_check': datetime.now().isoformat()
            }
