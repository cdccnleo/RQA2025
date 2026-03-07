"""
basic_health_checker 模块

提供 basic_health_checker 相关功能和接口。
"""

import logging

import time

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from ..models.health_result import HealthCheckResult, CheckType
from ..models.health_status import HealthStatus
from typing import Dict, Any, Optional, Callable, Tuple
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施层 - 健康检查组件

basic_health_checker 模块

提供基础的健康检查功能实现。

基础健康检查器实现
提供基本的健康检查功能，遵循统一接口规范。
"""

logger = logging.getLogger(__name__)


@dataclass
class ServiceHealthProfile:
    """服务健康档案"""

    name: str
    check_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    last_check_time: Optional[datetime] = None
    average_response_time: float = 0.0
    status: str = "unknown"


class IHealthChecker(ABC):
    @abstractmethod
    def check_health(self, service_name: str):
        pass

    @abstractmethod
    def register_service(self, service_name: str, check_func: Callable) -> None:
        pass

    @abstractmethod
    def unregister_service(self, service_name: str) -> None:
        pass


class BasicHealthChecker(IHealthChecker):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._checkers: Dict[str, Callable] = {}
        self._services: Dict[str, ServiceHealthProfile] = defaultdict(ServiceHealthProfile)

    def register_service(self, name: str, check_func: Callable) -> None:
        if not callable(check_func):
            raise ValueError(f"Check function for {name} must be callable")

        # 创建服务健康档案
        if name not in self._services:
            self._services[name] = ServiceHealthProfile(name)

        self._checkers[name] = check_func
        logger.info(f"Registered health check for service: {name}")

    def unregister_service(self, service_name: str) -> None:
        if service_name in self._checkers:
            del self._checkers[service_name]
        if service_name in self._services:
            del self._services[service_name]
        logger.info(f"Unregistered health check for service: {service_name}")

    def check_service(self, name: str, timeout: int = 5) -> Dict[str, Any]:
        """检查单个服务的健康状态"""
        # 验证服务是否存在
        if not self._validate_service_exists(name):
            return {"status": "error", "message": f"Service {name} not found"}

        try:
            # 执行服务检查
            result, response_time = self._execute_service_check(name)

            # 创建并返回成功的检查结果
            return self._create_success_check_result(name, result, response_time)

        except Exception as e:
            # 创建并返回错误的检查结果
            return self._create_error_check_result(name, e)

    def _validate_service_exists(self, name: str) -> bool:
        """验证服务是否存在"""
        return name in self._checkers

    def _execute_service_check(self, name: str) -> Tuple[bool, float]:
        """执行服务健康检查"""
        start_time = time.time()
        result = self._checkers[name]()
        response_time = time.time() - start_time
        return result, response_time

    def _create_success_check_result(self, name: str, result: bool, response_time: float) -> Dict[str, Any]:
        """创建成功的检查结果"""
        # 更新服务健康档案（手动更新，避免调用不存在的方法）
        if name in self._services:
            profile = self._services[name]
            profile.check_count += 1
            if result:
                profile.success_count += 1
            else:
                profile.failure_count += 1
            profile.last_check_time = datetime.now()
            profile.status = "healthy" if result else "unhealthy"

        return {
            'status': 'up' if result else 'unhealthy',
            'response_time': response_time,
            'timestamp': datetime.now().isoformat(),
            'details': {'result': result}
        }

    def _create_error_check_result(self, name: str, error: Exception) -> Dict[str, Any]:
        """创建错误的检查结果"""
        # 更新服务健康档案（手动更新）
        if name in self._services:
            profile = self._services[name]
            profile.check_count += 1
            profile.failure_count += 1
            profile.last_check_time = datetime.now()
            profile.status = "error"

        return {
            'status': 'error',
            'error': str(error),
            'timestamp': datetime.now().isoformat()
        }

    def _update_service_health_record(self, name: str, check_result: HealthCheckResult):
        """更新服务健康档案"""
        if name in self._services:
            self._services[name].add_check_result(check_result)

    def check_health(self) -> Dict[str, Any]:
        service_results = {}

        for service_name in self._checkers:
            try:
                result = self.check_service(service_name)
                service_results[service_name] = result
            except Exception as e:
                service_results[service_name] = {
                    'status': 'error',
                    'error': str(e)
                }

        overall_status = 'healthy'
        for result in service_results.values():
            if result.get('status') in ['error', 'down']:
                overall_status = 'unhealthy'
                break

        return {
            'overall_status': overall_status,
            'services': service_results,
            'timestamp': datetime.now().isoformat()
        }

    def generate_status_report(self) -> Dict[str, Any]:
        """生成状态报告"""
        return self.check_health()

    def check_component(self, component_name: str) -> Dict[str, Any]:
        """检查组件健康状态"""
        return self.check_service(component_name)

    def perform_health_check(self) -> Dict[str, Any]:
        """别名方法，用于兼容测试"""
        result = self.check_health()
        # 返回测试期望的格式
        return {
            'healthy': result['overall_status'] == 'healthy',
            'status': result['overall_status'],
            'services': result['services'],
            'timestamp': result['timestamp']
        }
