#!/usr/bin/env python3
"""
服务发现和负载均衡组件

为数据采集编排器提供微服务发现和负载均衡能力：
1. 服务注册和发现
2. 负载均衡策略
3. 健康检查
4. 故障转移
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import aiohttp
import json
import hashlib

from src.infrastructure.logging.core.unified_logger import get_unified_logger

logger = get_unified_logger(__name__)


@dataclass
class ServiceInstance:
    """服务实例信息"""
    service_name: str
    instance_id: str
    host: str
    port: int
    protocol: str = "http"
    status: str = "healthy"
    last_heartbeat: datetime = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.last_heartbeat is None:
            self.last_heartbeat = datetime.now()
        if self.metadata is None:
            self.metadata = {}

    @property
    def url(self) -> str:
        """获取服务URL"""
        return f"{self.protocol}://{self.host}:{self.port}"

    def is_healthy(self) -> bool:
        """检查服务是否健康"""
        if self.status != "healthy":
            return False

        # 检查心跳超时（30秒）
        if self.last_heartbeat and (datetime.now() - self.last_heartbeat) > timedelta(seconds=30):
            self.status = "unhealthy"
            return False

        return True


class LoadBalancer:
    """负载均衡器"""

    def __init__(self, strategy: str = "round_robin"):
        self.strategy = strategy
        self.current_index: Dict[str, int] = {}
        self.weights: Dict[str, Dict[str, int]] = {}

    def select_instance(self, service_name: str, instances: List[ServiceInstance]) -> Optional[ServiceInstance]:
        """选择服务实例"""
        healthy_instances = [inst for inst in instances if inst.is_healthy()]

        if not healthy_instances:
            logger.warning(f"服务 {service_name} 没有健康的实例")
            return None

        if self.strategy == "round_robin":
            return self._round_robin_select(service_name, healthy_instances)
        elif self.strategy == "weighted_round_robin":
            return self._weighted_round_robin_select(service_name, healthy_instances)
        elif self.strategy == "least_connections":
            return self._least_connections_select(healthy_instances)
        elif self.strategy == "random":
            return self._random_select(healthy_instances)
        else:
            return healthy_instances[0]

    def _round_robin_select(self, service_name: str, instances: List[ServiceInstance]) -> ServiceInstance:
        """轮询选择"""
        current_idx = self.current_index.get(service_name, 0)
        instance = instances[current_idx % len(instances)]
        self.current_index[service_name] = (current_idx + 1) % len(instances)
        return instance

    def _weighted_round_robin_select(self, service_name: str, instances: List[ServiceInstance]) -> ServiceInstance:
        """加权轮询选择"""
        total_weight = sum(self.weights.get(service_name, {}).get(inst.instance_id, 1) for inst in instances)

        if total_weight == 0:
            return instances[0]

        current_weight = self.current_index.get(service_name, 0) % total_weight

        weight_sum = 0
        for instance in instances:
            weight = self.weights.get(service_name, {}).get(instance.instance_id, 1)
            weight_sum += weight

            if current_weight < weight_sum:
                self.current_index[service_name] = current_weight + 1
                return instance

        return instances[0]

    def _least_connections_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """最少连接选择"""
        # 简化实现，返回第一个健康实例
        return min(instances, key=lambda x: x.metadata.get('connections', 0))

    def _random_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """随机选择"""
        import random
        return random.choice(instances)


class ServiceRegistry:
    """服务注册表"""

    def __init__(self):
        self.services: Dict[str, List[ServiceInstance]] = {}
        self.health_check_interval = 30  # 30秒健康检查间隔
        self._health_check_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    async def start_health_checks(self):
        """启动健康检查"""
        if self._health_check_task is None:
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            logger.info("服务健康检查已启动")

    async def stop_health_checks(self):
        """停止健康检查"""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            logger.info("服务健康检查已停止")

    async def _health_check_loop(self):
        """健康检查循环"""
        while True:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"健康检查循环异常: {e}")
                await asyncio.sleep(5)

    async def _perform_health_checks(self):
        """执行健康检查"""
        async with self._lock:
            for service_name, instances in self.services.items():
                for instance in instances:
                    try:
                        await self._check_instance_health(instance)
                    except Exception as e:
                        logger.error(f"检查服务实例健康状态失败 {instance.instance_id}: {e}")
                        instance.status = "unhealthy"

    async def _check_instance_health(self, instance: ServiceInstance):
        """检查单个实例健康状态"""
        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                health_url = f"{instance.url}/health"
                async with session.get(health_url) as response:
                    if response.status == 200:
                        instance.status = "healthy"
                        instance.last_heartbeat = datetime.now()
                    else:
                        instance.status = "unhealthy"
                        logger.warning(f"服务实例 {instance.instance_id} 健康检查失败: HTTP {response.status}")
        except Exception as e:
            instance.status = "unhealthy"
            logger.warning(f"服务实例 {instance.instance_id} 健康检查异常: {e}")

    async def register_service(self, service_name: str, instance: ServiceInstance):
        """注册服务实例"""
        async with self._lock:
            if service_name not in self.services:
                self.services[service_name] = []

            # 检查是否已存在相同的实例
            existing_instance = None
            for inst in self.services[service_name]:
                if inst.instance_id == instance.instance_id:
                    existing_instance = inst
                    break

            if existing_instance:
                # 更新现有实例
                existing_instance.host = instance.host
                existing_instance.port = instance.port
                existing_instance.status = instance.status
                existing_instance.metadata = instance.metadata
                existing_instance.last_heartbeat = datetime.now()
                logger.info(f"更新服务实例: {service_name}/{instance.instance_id}")
            else:
                # 添加新实例
                instance.last_heartbeat = datetime.now()
                self.services[service_name].append(instance)
                logger.info(f"注册服务实例: {service_name}/{instance.instance_id}")

    async def unregister_service(self, service_name: str, instance_id: str):
        """注销服务实例"""
        async with self._lock:
            if service_name in self.services:
                self.services[service_name] = [
                    inst for inst in self.services[service_name]
                    if inst.instance_id != instance_id
                ]
                logger.info(f"注销服务实例: {service_name}/{instance_id}")

    def get_service_instances(self, service_name: str) -> List[ServiceInstance]:
        """获取服务实例"""
        return self.services.get(service_name, [])

    def get_all_services(self) -> Dict[str, List[ServiceInstance]]:
        """获取所有服务"""
        return self.services.copy()

    async def discover_service(self, service_name: str) -> Optional[ServiceInstance]:
        """发现服务实例"""
        instances = self.get_service_instances(service_name)
        if not instances:
            return None

        # 使用负载均衡选择实例
        load_balancer = LoadBalancer("round_robin")
        return load_balancer.select_instance(service_name, instances)


class ServiceDiscovery:
    """服务发现组件"""

    def __init__(self):
        self.registry = ServiceRegistry()
        self.load_balancer = LoadBalancer("round_robin")
        self._discovery_task: Optional[asyncio.Task] = None
        self.service_endpoints = {
            "data-service": ["data-service:8000"],
            "strategy-service": ["strategy-service:8001"],
            "trading-service": ["trading-service:8002"],
            "risk-service": ["risk-service:8003"],
            "rqa2025-app": ["rqa2025-app:8000"]
        }

    async def start_discovery(self):
        """启动服务发现"""
        await self.registry.start_health_checks()
        await self._register_known_services()
        logger.info("服务发现已启动")

    async def stop_discovery(self):
        """停止服务发现"""
        await self.registry.stop_health_checks()
        logger.info("服务发现已停止")

    async def _register_known_services(self):
        """注册已知服务"""
        for service_name, endpoints in self.service_endpoints.items():
            for endpoint in endpoints:
                try:
                    host, port_str = endpoint.split(":")
                    port = int(port_str)

                    instance_id = f"{service_name}-{host}-{port}"

                    instance = ServiceInstance(
                        service_name=service_name,
                        instance_id=instance_id,
                        host=host,
                        port=port,
                        metadata={"auto_registered": True}
                    )

                    await self.registry.register_service(service_name, instance)
                    logger.info(f"自动注册服务实例: {service_name} -> {endpoint}")

                except Exception as e:
                    logger.error(f"注册服务 {service_name} 失败: {e}")

    async def discover_service(self, service_name: str) -> Optional[ServiceInstance]:
        """发现服务"""
        return await self.registry.discover_service(service_name)

    async def call_service(self, service_name: str, endpoint: str, method: str = "GET",
                          data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """调用服务"""
        try:
            instance = await self.discover_service(service_name)
            if not instance:
                logger.error(f"无法发现服务: {service_name}")
                return None

            url = f"{instance.url}{endpoint}"

            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                if method.upper() == "GET":
                    async with session.get(url) as response:
                        return await self._process_response(response)
                elif method.upper() == "POST":
                    headers = {"Content-Type": "application/json"}
                    json_data = json.dumps(data) if data else None
                    async with session.post(url, data=json_data, headers=headers) as response:
                        return await self._process_response(response)
                elif method.upper() == "PUT":
                    headers = {"Content-Type": "application/json"}
                    json_data = json.dumps(data) if data else None
                    async with session.put(url, data=json_data, headers=headers) as response:
                        return await self._process_response(response)
                else:
                    logger.error(f"不支持的HTTP方法: {method}")
                    return None

        except Exception as e:
            logger.error(f"调用服务 {service_name} 失败: {e}")
            return None

    async def _process_response(self, response) -> Dict[str, Any]:
        """处理HTTP响应"""
        try:
            if response.status == 200:
                content_type = response.headers.get("Content-Type", "")
                if "application/json" in content_type:
                    return await response.json()
                else:
                    return {"text": await response.text()}
            else:
                return {
                    "error": f"HTTP {response.status}",
                    "status_code": response.status
                }
        except Exception as e:
            return {"error": str(e)}

    def get_service_stats(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        stats = {}
        all_services = self.registry.get_all_services()

        for service_name, instances in all_services.items():
            healthy_count = len([inst for inst in instances if inst.is_healthy()])
            total_count = len(instances)

            stats[service_name] = {
                "total_instances": total_count,
                "healthy_instances": healthy_count,
                "unhealthy_instances": total_count - healthy_count,
                "instances": [
                    {
                        "id": inst.instance_id,
                        "host": inst.host,
                        "port": inst.port,
                        "status": inst.status,
                        "last_heartbeat": inst.last_heartbeat.isoformat() if inst.last_heartbeat else None
                    }
                    for inst in instances
                ]
            }

        return stats


# 全局服务发现实例
_service_discovery_instance: Optional[ServiceDiscovery] = None


def get_service_discovery() -> ServiceDiscovery:
    """获取服务发现实例"""
    global _service_discovery_instance
    if _service_discovery_instance is None:
        _service_discovery_instance = ServiceDiscovery()
    return _service_discovery_instance
