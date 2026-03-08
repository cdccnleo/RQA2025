"""
服务注册中心

负责服务实例的注册、注销和健康检查。

从service_discovery.py中提取以改善代码组织。

Author: RQA2025 Development Team
Date: 2025-11-01
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """服务状态枚举"""
    STARTING = "starting"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    MAINTENANCE = "maintenance"
    STOPPING = "stopping"


@dataclass
class ServiceInstance:
    """服务实例信息"""
    service_id: str
    service_name: str
    host: str
    port: int
    status: ServiceStatus = ServiceStatus.STARTING
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    health_check_url: Optional[str] = None
    health_check_interval: int = 30
    weight: int = 100
    created_at: Optional[float] = None
    last_heartbeat: Optional[float] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.tags is None:
            self.tags = []
        if self.created_at is None:
            self.created_at = time.time()
        if self.last_heartbeat is None:
            self.last_heartbeat = time.time()


@dataclass
class HealthCheckResult:
    """健康检查结果"""
    service_id: str
    is_healthy: bool
    response_time: float
    error_message: Optional[str] = None
    check_time: Optional[float] = None

    def __post_init__(self):
        if self.check_time is None:
            self.check_time = time.time()


class ServiceRegistry:
    """
    服务注册中心
    
    负责:
    1. 服务实例注册和注销
    2. 服务健康检查
    3. 服务状态管理
    4. 服务查询
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._services: Dict[str, ServiceInstance] = {}
        self._service_groups: Dict[str, List[str]] = {}
        self._health_checks: Dict[str, HealthCheckResult] = {}
        self._lock = threading.RLock()

        # 健康检查配置
        self.health_check_interval = self.config.get('health_check_interval', 30)
        self.health_check_timeout = self.config.get('health_check_timeout', 10)
        self.max_failed_checks = self.config.get('max_failed_checks', 3)

        # 后台任务
        self._running = True
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True
        )
        self._health_check_thread.start()

        logger.info("服务注册中心初始化完成")

    def register(self, service: ServiceInstance) -> bool:
        """注册服务实例"""
        with self._lock:
            try:
                self._services[service.service_id] = service

                # 加入服务组
                if service.service_name not in self._service_groups:
                    self._service_groups[service.service_name] = []
                if service.service_id not in self._service_groups[service.service_name]:
                    self._service_groups[service.service_name].append(service.service_id)

                logger.info(f"服务注册成功: {service.service_name} ({service.service_id})")
                return True

            except Exception as e:
                logger.error(f"服务注册失败: {e}")
                return False

    def deregister(self, service_id: str) -> bool:
        """注销服务实例"""
        with self._lock:
            try:
                if service_id not in self._services:
                    return False

                service = self._services[service_id]

                # 从服务组移除
                if service.service_name in self._service_groups:
                    if service_id in self._service_groups[service.service_name]:
                        self._service_groups[service.service_name].remove(service_id)

                # 删除服务
                del self._services[service_id]

                # 删除健康检查记录
                if service_id in self._health_checks:
                    del self._health_checks[service_id]

                logger.info(f"服务注销成功: {service.service_name} ({service_id})")
                return True

            except Exception as e:
                logger.error(f"服务注销失败: {e}")
                return False

    def get_service(self, service_id: str) -> Optional[ServiceInstance]:
        """获取服务实例"""
        with self._lock:
            return self._services.get(service_id)

    def get_services_by_name(self, service_name: str) -> List[ServiceInstance]:
        """根据服务名获取所有实例"""
        with self._lock:
            if service_name not in self._service_groups:
                return []
            
            return [
                self._services[service_id]
                for service_id in self._service_groups[service_name]
                if service_id in self._services
            ]

    def get_healthy_services(self, service_name: str) -> List[ServiceInstance]:
        """获取健康的服务实例"""
        with self._lock:
            services = self.get_services_by_name(service_name)
            return [
                service for service in services
                if service.status == ServiceStatus.HEALTHY
            ]

    def update_heartbeat(self, service_id: str) -> bool:
        """更新服务心跳"""
        with self._lock:
            if service_id in self._services:
                self._services[service_id].last_heartbeat = time.time()
                return True
            return False

    def update_status(self, service_id: str, status: ServiceStatus) -> bool:
        """更新服务状态"""
        with self._lock:
            if service_id in self._services:
                self._services[service_id].status = status
                logger.info(f"服务状态更新: {service_id} -> {status.value}")
                return True
            return False

    def perform_health_check(self, service_id: str) -> HealthCheckResult:
        """执行健康检查"""
        service = self.get_service(service_id)
        
        if not service:
            return HealthCheckResult(
                service_id=service_id,
                is_healthy=False,
                response_time=0.0,
                error_message="服务不存在"
            )

        try:
            start_time = time.time()
            
            # 检查心跳超时
            heartbeat_age = time.time() - service.last_heartbeat
            if heartbeat_age > self.health_check_interval * 2:
                return HealthCheckResult(
                    service_id=service_id,
                    is_healthy=False,
                    response_time=time.time() - start_time,
                    error_message=f"心跳超时: {heartbeat_age:.1f}秒"
                )

            # 简单的健康检查
            response_time = time.time() - start_time
            return HealthCheckResult(
                service_id=service_id,
                is_healthy=True,
                response_time=response_time
            )

        except Exception as e:
            return HealthCheckResult(
                service_id=service_id,
                is_healthy=False,
                response_time=0.0,
                error_message=str(e)
            )

    def _health_check_loop(self):
        """健康检查循环"""
        while self._running:
            try:
                with self._lock:
                    service_ids = list(self._services.keys())
                
                for service_id in service_ids:
                    result = self.perform_health_check(service_id)
                    
                    with self._lock:
                        self._health_checks[service_id] = result
                        
                        # 更新服务状态
                        if service_id in self._services:
                            if result.is_healthy:
                                self._services[service_id].status = ServiceStatus.HEALTHY
                            else:
                                self._services[service_id].status = ServiceStatus.UNHEALTHY
                
                time.sleep(self.health_check_interval)

            except Exception as e:
                logger.error(f"健康检查循环异常: {e}")
                time.sleep(5)

    def get_all_services(self) -> List[ServiceInstance]:
        """获取所有服务实例"""
        with self._lock:
            return list(self._services.values())

    def get_service_count(self) -> int:
        """获取服务数量"""
        with self._lock:
            return len(self._services)

    def get_service_names(self) -> List[str]:
        """获取所有服务名称"""
        with self._lock:
            return list(self._service_groups.keys())

    def shutdown(self):
        """关闭服务注册中心"""
        self._running = False
        if self._health_check_thread.is_alive():
            self._health_check_thread.join(timeout=5)
        logger.info("服务注册中心已关闭")


__all__ = [
    'ServiceStatus',
    'ServiceInstance',
    'HealthCheckResult',
    'ServiceRegistry'
]

