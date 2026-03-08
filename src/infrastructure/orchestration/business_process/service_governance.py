#!/usr/bin/env python3
"""
服务治理组件

提供数据采集服务治理能力：
1. 服务发现和注册
2. 依赖管理和健康检查
3. 负载均衡和故障转移
4. 服务监控和告警
5. 配置管理和服务协调
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
import threading
import json

from src.infrastructure.logging.core.unified_logger import get_unified_logger

logger = get_unified_logger(__name__)


@dataclass
class ServiceInfo:
    """服务信息"""
    service_id: str
    service_type: str
    service_name: str
    host: str
    port: int
    protocol: str = "http"
    status: str = "unknown"
    health_check_url: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    health_score: float = 0.0
    response_time: float = 0.0


@dataclass
class ServiceDependency:
    """服务依赖关系"""
    service_id: str
    dependency_id: str
    dependency_type: str  # "hard", "soft", "optional"
    required: bool = True
    timeout: int = 30
    retry_count: int = 3
    circuit_breaker_enabled: bool = True


class ServiceRegistry:
    """服务注册表"""

    def __init__(self):
        self.services: Dict[str, ServiceInfo] = {}
        self.dependencies: Dict[str, List[ServiceDependency]] = {}
        self.service_types: Dict[str, List[str]] = {}  # service_type -> [service_ids]
        self._lock = threading.RLock()

    def register_service(self, service_info: ServiceInfo) -> bool:
        """注册服务"""
        with self._lock:
            try:
                self.services[service_info.service_id] = service_info

                # 更新服务类型索引
                if service_info.service_type not in self.service_types:
                    self.service_types[service_info.service_type] = []
                if service_info.service_id not in self.service_types[service_info.service_type]:
                    self.service_types[service_info.service_type].append(service_info.service_id)

                # 初始化依赖列表
                if service_info.service_id not in self.dependencies:
                    self.dependencies[service_info.service_id] = []

                logger.info(f"服务注册成功: {service_info.service_id} ({service_info.service_type})")
                return True

            except Exception as e:
                logger.error(f"服务注册失败: {e}")
                return False

    def unregister_service(self, service_id: str) -> bool:
        """注销服务"""
        with self._lock:
            try:
                if service_id in self.services:
                    service_info = self.services[service_id]
                    service_type = service_info.service_type

                    # 从服务列表中移除
                    del self.services[service_id]

                    # 从服务类型索引中移除
                    if service_type in self.service_types and service_id in self.service_types[service_type]:
                        self.service_types[service_type].remove(service_id)

                    # 清理依赖关系
                    if service_id in self.dependencies:
                        del self.dependencies[service_id]

                    # 清理其他服务的依赖引用
                    for svc_id, deps in self.dependencies.items():
                        self.dependencies[svc_id] = [
                            dep for dep in deps if dep.dependency_id != service_id
                        ]

                    logger.info(f"服务注销成功: {service_id}")
                    return True
                return False

            except Exception as e:
                logger.error(f"服务注销失败: {e}")
                return False

    def get_service(self, service_id: str) -> Optional[ServiceInfo]:
        """获取服务信息"""
        with self._lock:
            return self.services.get(service_id)

    def get_services_by_type(self, service_type: str) -> List[ServiceInfo]:
        """按类型获取服务列表"""
        with self._lock:
            service_ids = self.service_types.get(service_type, [])
            return [self.services[sid] for sid in service_ids if sid in self.services]

    def get_all_services(self) -> List[ServiceInfo]:
        """获取所有服务"""
        with self._lock:
            return list(self.services.values())

    def update_service_status(self, service_id: str, status: str,
                            health_score: float = 0.0,
                            response_time: float = 0.0) -> bool:
        """更新服务状态"""
        with self._lock:
            if service_id in self.services:
                self.services[service_id].status = status
                self.services[service_id].health_score = health_score
                self.services[service_id].response_time = response_time
                self.services[service_id].last_heartbeat = datetime.now()
                return True
            return False

    def add_dependency(self, service_id: str, dependency: ServiceDependency) -> bool:
        """添加服务依赖"""
        with self._lock:
            if service_id not in self.dependencies:
                self.dependencies[service_id] = []
            self.dependencies[service_id].append(dependency)
            return True

    def get_dependencies(self, service_id: str) -> List[ServiceDependency]:
        """获取服务依赖"""
        with self._lock:
            return self.dependencies.get(service_id, [])

    def get_service_stats(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        with self._lock:
            total_services = len(self.services)
            healthy_services = len([s for s in self.services.values() if s.status == "healthy"])
            service_types_count = len(self.service_types)

            return {
                "total_services": total_services,
                "healthy_services": healthy_services,
                "unhealthy_services": total_services - healthy_services,
                "service_types": service_types_count,
                "health_rate": healthy_services / total_services if total_services > 0 else 0
            }


class ServiceDiscovery:
    """服务发现"""

    def __init__(self, registry: ServiceRegistry):
        self.registry = registry
        self.discovery_strategies: Dict[str, Callable] = {}

    def register_strategy(self, strategy_name: str, strategy_func: Callable):
        """注册发现策略"""
        self.discovery_strategies[strategy_name] = strategy_func

    def discover_services(self, service_type: str,
                         criteria: Optional[Dict[str, Any]] = None) -> List[ServiceInfo]:
        """发现服务"""
        services = self.registry.get_services_by_type(service_type)

        if not criteria:
            return services

        # 应用筛选条件
        filtered_services = []
        for service in services:
            if self._matches_criteria(service, criteria):
                filtered_services.append(service)

        return filtered_services

    def _matches_criteria(self, service: ServiceInfo, criteria: Dict[str, Any]) -> bool:
        """检查服务是否匹配条件"""
        for key, value in criteria.items():
            if key == "status" and service.status != value:
                return False
            elif key == "health_score_min" and service.health_score < value:
                return False
            elif key == "health_score_max" and service.health_score > value:
                return False
            elif key == "response_time_max" and service.response_time > value:
                return False
            elif key == "metadata" and isinstance(value, dict):
                for meta_key, meta_value in value.items():
                    if service.metadata.get(meta_key) != meta_value:
                        return False

        return True


class LoadBalancer:
    """负载均衡器"""

    def __init__(self, registry: ServiceRegistry):
        self.registry = registry
        self.algorithms = {
            "round_robin": self._round_robin_select,
            "weighted_round_robin": self._weighted_round_robin_select,
            "least_connections": self._least_connections_select,
            "random": self._random_select,
            "health_based": self._health_based_select
        }
        self.current_index: Dict[str, int] = {}

    def select_service(self, service_type: str,
                      algorithm: str = "round_robin",
                      criteria: Optional[Dict[str, Any]] = None) -> Optional[ServiceInfo]:
        """选择服务实例"""
        services = self.registry.get_services_by_type(service_type)

        if not services:
            return None

        # 筛选可用服务
        available_services = [s for s in services if s.status == "healthy"]
        if not available_services:
            return None

        # 应用额外筛选条件
        if criteria:
            from src.infrastructure.orchestration.business_process.service_governance import ServiceDiscovery
            discovery = ServiceDiscovery(self.registry)
            available_services = discovery.discover_services(service_type, criteria)

        if not available_services:
            return None

        # 使用指定算法选择服务
        if algorithm in self.algorithms:
            return self.algorithms[algorithm](available_services, service_type)
        else:
            return available_services[0]

    def _round_robin_select(self, services: List[ServiceInfo], service_type: str) -> ServiceInfo:
        """轮询选择"""
        if service_type not in self.current_index:
            self.current_index[service_type] = 0

        service = services[self.current_index[service_type] % len(services)]
        self.current_index[service_type] = (self.current_index[service_type] + 1) % len(services)
        return service

    def _weighted_round_robin_select(self, services: List[ServiceInfo], service_type: str) -> ServiceInfo:
        """加权轮询选择"""
        # 基于健康评分进行加权选择
        total_weight = sum(s.health_score for s in services)
        if total_weight == 0:
            return services[0]

        import random
        rand_weight = random.uniform(0, total_weight)
        current_weight = 0

        for service in services:
            current_weight += service.health_score
            if rand_weight <= current_weight:
                return service

        return services[0]

    def _least_connections_select(self, services: List[ServiceInfo], service_type: str) -> ServiceInfo:
        """最少连接选择"""
        # 这里可以基于连接数或响应时间选择
        # 暂时基于响应时间选择
        return min(services, key=lambda s: s.response_time)

    def _random_select(self, services: List[ServiceInfo], service_type: str) -> ServiceInfo:
        """随机选择"""
        import random
        return random.choice(services)

    def _health_based_select(self, services: List[ServiceInfo], service_type: str) -> ServiceInfo:
        """基于健康状态选择"""
        # 优先选择健康评分最高的服务
        return max(services, key=lambda s: s.health_score)


class HealthChecker:
    """健康检查器"""

    def __init__(self, registry: ServiceRegistry):
        self.registry = registry
        self.check_interval = 30  # 30秒检查一次
        self.timeout = 10  # 10秒超时
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start_health_checks(self):
        """启动健康检查"""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._health_check_loop())

        logger.info("健康检查服务已启动")

    async def stop_health_checks(self):
        """停止健康检查"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info("健康检查服务已停止")

    async def _health_check_loop(self):
        """健康检查循环"""
        while self._running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"健康检查循环异常: {e}")
                await asyncio.sleep(self.check_interval)

    async def _perform_health_checks(self):
        """执行健康检查"""
        services = self.registry.get_all_services()

        for service in services:
            try:
                health_status = await self._check_service_health(service)

                # 更新服务状态
                self.registry.update_service_status(
                    service.service_id,
                    health_status["status"],
                    health_status["health_score"],
                    health_status["response_time"]
                )

            except Exception as e:
                logger.error(f"健康检查失败 {service.service_id}: {e}")
                self.registry.update_service_status(service.service_id, "unhealthy", 0.0, 999.0)

    async def _check_service_health(self, service: ServiceInfo) -> Dict[str, Any]:
        """检查单个服务健康状态"""
        import aiohttp

        start_time = time.time()

        try:
            if service.health_check_url:
                # 使用自定义健康检查URL
                check_url = service.health_check_url
            else:
                # 构造默认健康检查URL
                check_url = f"{service.protocol}://{service.host}:{service.port}/health"

            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(check_url) as response:
                    response_time = (time.time() - start_time) * 1000  # 毫秒

                    if response.status == 200:
                        try:
                            data = await response.json()
                            health_score = data.get("health_score", 0.8)
                        except:
                            health_score = 0.8

                        return {
                            "status": "healthy",
                            "health_score": health_score,
                            "response_time": response_time
                        }
                    else:
                        return {
                            "status": "unhealthy",
                            "health_score": 0.0,
                            "response_time": response_time
                        }

        except asyncio.TimeoutError:
            response_time = self.timeout * 1000
            return {
                "status": "unhealthy",
                "health_score": 0.0,
                "response_time": response_time
            }
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return {
                "status": "unhealthy",
                "health_score": 0.0,
                "response_time": response_time
            }


class ServiceGovernanceManager:
    """服务治理管理器"""

    def __init__(self):
        self.registry = ServiceRegistry()
        self.discovery = ServiceDiscovery(self.registry)
        self.load_balancer = LoadBalancer(self.registry)
        self.health_checker = HealthChecker(self.registry)

        # 注册默认服务发现策略
        self._register_default_strategies()

    def _register_default_strategies(self):
        """注册默认服务发现策略"""

        def healthy_services_strategy(service_type: str, criteria: Dict[str, Any]) -> List[ServiceInfo]:
            """只返回健康的服务"""
            criteria = criteria or {}
            criteria["status"] = "healthy"
            return self.discovery.discover_services(service_type, criteria)

        def high_performance_strategy(service_type: str, criteria: Dict[str, Any]) -> List[ServiceInfo]:
            """返回高性能服务（响应时间短）"""
            criteria = criteria or {}
            criteria["response_time_max"] = 500  # 500ms以内
            criteria["status"] = "healthy"
            return self.discovery.discover_services(service_type, criteria)

        self.discovery.register_strategy("healthy", healthy_services_strategy)
        self.discovery.register_strategy("high_performance", high_performance_strategy)

    async def initialize(self):
        """初始化服务治理"""
        # 注册核心服务
        await self._register_core_services()

        # 启动健康检查
        await self.health_checker.start_health_checks()

        logger.info("服务治理管理器初始化完成")

    async def shutdown(self):
        """关闭服务治理"""
        await self.health_checker.stop_health_checks()
        logger.info("服务治理管理器已关闭")

    async def _register_core_services(self):
        """注册核心服务"""

        # 注册网关服务
        gateway_service = ServiceInfo(
            service_id="gateway",
            service_type="api_gateway",
            service_name="API网关",
            host="localhost",
            port=8000,
            protocol="http",
            health_check_url="http://localhost:8000/health",
            status="healthy",
            metadata={"version": "1.0.0", "environment": "production"}
        )
        self.registry.register_service(gateway_service)

        # 注册数据层服务
        data_service = ServiceInfo(
            service_id="data_layer",
            service_type="data_service",
            service_name="数据层服务",
            host="localhost",
            port=8001,
            protocol="http",
            health_check_url="http://localhost:8001/health",
            status="healthy",
            metadata={"version": "1.0.0", "capabilities": ["storage", "processing", "cache"]}
        )
        self.registry.register_service(data_service)

        # 注册缓存服务
        cache_service = ServiceInfo(
            service_id="cache_service",
            service_type="cache",
            service_name="缓存服务",
            host="localhost",
            port=6379,
            protocol="redis",
            status="healthy",
            metadata={"type": "redis", "max_memory": "1GB"}
        )
        self.registry.register_service(cache_service)

        # 设置服务依赖关系
        self.registry.add_dependency("gateway", ServiceDependency(
            service_id="gateway",
            dependency_id="data_layer",
            dependency_type="hard",
            required=True
        ))

        self.registry.add_dependency("data_layer", ServiceDependency(
            service_id="data_layer",
            dependency_id="cache_service",
            dependency_type="soft",
            required=False
        ))

    def get_service_endpoint(self, service_type: str,
                           algorithm: str = "health_based",
                           criteria: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """获取服务端点"""
        service = self.load_balancer.select_service(service_type, algorithm, criteria)
        if service:
            return f"{service.protocol}://{service.host}:{service.port}"
        return None

    def get_service_stats(self) -> Dict[str, Any]:
        """获取服务治理统计信息"""
        registry_stats = self.registry.get_service_stats()

        return {
            "registry": registry_stats,
            "health_check_running": self.health_checker._running,
            "discovery_strategies": list(self.discovery.discovery_strategies.keys()),
            "load_balancer_algorithms": list(self.load_balancer.algorithms.keys())
        }
