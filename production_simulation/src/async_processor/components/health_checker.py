"""
Health Checker Module
健康检查器模块

此文件作为主入口，导入并导出各个模块的组件。

重构说明(2025-11-01):
- health_models.py: 健康检查数据模型
- health_checker.py: 健康检查器(本文件)

Author: RQA2025 Development Team
Date: 2025-11-01
"""

import logging
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
import threading
import time
import requests
import socket

from .health_models import HealthStatus, ComponentType, HealthCheck

logger = logging.getLogger(__name__)


# 继续保留HealthChecker类
class HealthChecker:

    """
    Health Check Class
    健康检查类

    Represents a health check for a specific component
    表示对特定组件的健康检查
    """

    def __init__(self,


                 component_name: str,
                 component_type: ComponentType,
                 check_function: Callable,
                 interval: int = 30,
                 timeout: float = 5.0,
                 retries: int = 3):
        """
        Initialize a health check
        初始化健康检查

        Args:
            component_name: Name of the component to check
                           要检查的组件名称
            component_type: Type of the component
                           组件类型
            check_function: Function to perform the health check
                           执行健康检查的函数
            interval: Check interval in seconds
                     检查间隔（秒）
            timeout: Timeout for each check
                    每次检查的超时时间
            retries: Number of retries on failure
                    失败时的重试次数
        """
        self.component_name = component_name
        self.component_type = component_type
        self.check_function = check_function
        self.interval = interval
        self.timeout = timeout
        self.retries = retries

        # Status tracking
        self.last_check = None
        self.last_status = HealthStatus.UNKNOWN
        self.consecutive_failures = 0
        self.total_checks = 0
        self.successful_checks = 0

        # History
        self.status_history: List[Dict[str, Any]] = []
        self.max_history_size = 100

    def perform_check(self) -> Dict[str, Any]:
        """
        Perform the health check
        执行健康检查

        Returns:
            dict: Health check result
                  健康检查结果
        """
        self.total_checks += 1
        start_time = time.time()

        result = {
            'component_name': self.component_name,
            'component_type': self.component_type.value,
            'timestamp': datetime.now(),
            'status': HealthStatus.UNKNOWN,
            'response_time': 0.0,
            'details': {},
            'error': None
        }

        try:
            # Perform the actual check with retries
            for attempt in range(self.retries + 1):
                try:
                    check_result = self.check_function()

                    result['response_time'] = time.time() - start_time
                    result['details'] = check_result.get('details', {})
                    result['status'] = HealthStatus.HEALTHY if check_result.get(
                        'healthy', False) else HealthStatus.UNHEALTHY

                    if result['status'] == HealthStatus.HEALTHY:
                        self.successful_checks += 1
                        self.consecutive_failures = 0
                        break
                    else:
                        if attempt < self.retries:
                            time.sleep(0.5)  # Brief pause before retry
                            continue

                except Exception as e:
                    result['error'] = str(e)
                    if attempt < self.retries:
                        time.sleep(0.5)  # Brief pause before retry
                        continue

            # Update failure tracking
            if result['status'] != HealthStatus.HEALTHY:
                self.consecutive_failures += 1
                if self.consecutive_failures >= 3:
                    result['status'] = HealthStatus.UNHEALTHY
                elif self.consecutive_failures >= 1:
                    result['status'] = HealthStatus.DEGRADED

        except Exception as e:
            result['error'] = str(e)
            result['status'] = HealthStatus.UNHEALTHY
            self.consecutive_failures += 1

        # Update tracking
        self.last_check = result['timestamp']
        self.last_status = result['status']

        # Add to history
        self.status_history.append(result)
        if len(self.status_history) > self.max_history_size:
            self.status_history = self.status_history[-self.max_history_size:]

        return result

    def get_status_summary(self) -> Dict[str, Any]:
        """
        Get status summary for this health check
        获取此健康检查的状态摘要

        Returns:
            dict: Status summary
                  状态摘要
        """
        return {
            'component_name': self.component_name,
            'component_type': self.component_type.value,
            'last_status': self.last_status.value if self.last_status else 'unknown',
            'last_check': self.last_check.isoformat() if self.last_check else None,
            'consecutive_failures': self.consecutive_failures,
            'total_checks': self.total_checks,
            'success_rate': (self.successful_checks / max(self.total_checks, 1)) * 100,
            'recent_history': [h['status'] for h in self.status_history[-10:]]
        }


class HealthChecker:

    """
    Health Checker Class
    健康检查器类

    Manages health checks for multiple components and services
    管理多个组件和服务的健康检查
    """

    def __init__(self, checker_name: str = "default_health_checker"):
        """
        Initialize the health checker
        初始化健康检查器

        Args:
            checker_name: Name of this health checker
                         此健康检查器的名称
        """
        self.checker_name = checker_name
        self.health_checks: Dict[str, HealthCheck] = {}
        self.is_running = False
        self.checking_thread = None

        logger.info(f"Health checker {checker_name} initialized")

    def add_health_check(self,


                         component_name: str,
                         component_type: ComponentType,
                         check_function: Callable,
                         **kwargs) -> None:
        """
        Add a health check for a component
        为组件添加健康检查

        Args:
            component_name: Name of the component
                           组件名称
            component_type: Type of the component
                           组件类型
            check_function: Function to perform health check
                           执行健康检查的函数
            **kwargs: Additional arguments for HealthCheck
                     HealthCheck的其他参数
        """
        health_check = HealthCheck(
            component_name, component_type, check_function, **kwargs)
        self.health_checks[component_name] = health_check
        logger.info(f"Added health check for {component_name} ({component_type.value})")

    def remove_health_check(self, component_name: str) -> bool:
        """
        Remove a health check
        移除健康检查

        Args:
            component_name: Name of the component
                           组件名称

        Returns:
            bool: True if removed successfully, False otherwise
                  移除成功返回True，否则返回False
        """
        if component_name in self.health_checks:
            del self.health_checks[component_name]
            logger.info(f"Removed health check for {component_name}")
            return True
        return False

    def start_checking(self) -> bool:
        """
        Start periodic health checking
        开始定期健康检查

        Returns:
            bool: True if started successfully, False otherwise
                  启动成功返回True，否则返回False
        """
        if self.is_running:
            logger.warning(f"{self.checker_name} is already running")
            return False

        try:
            self.is_running = True
            self.checking_thread = threading.Thread(
                target=self._checking_loop, daemon=True)
            self.checking_thread.start()
            logger.info(f"Health checking started for {self.checker_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to start health checking: {str(e)}")
            self.is_running = False
            return False

    def stop_checking(self) -> bool:
        """
        Stop periodic health checking
        停止定期健康检查

        Returns:
            bool: True if stopped successfully, False otherwise
                  停止成功返回True，否则返回False
        """
        if not self.is_running:
            logger.warning(f"{self.checker_name} is not running")
            return False

        try:
            self.is_running = False
            if self.checking_thread and self.checking_thread.is_alive():
                self.checking_thread.join(timeout=5.0)
            logger.info(f"Health checking stopped for {self.checker_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to stop health checking: {str(e)}")
            return False

    def perform_health_check(self, component_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform health check(s)
        执行健康检查

        Args:
            component_name: Specific component to check (None for all)
                           要检查的特定组件（None表示全部）

        Returns:
            dict: Health check results
                  健康检查结果
        """
        results = {
            'timestamp': datetime.now(),
            'overall_status': HealthStatus.HEALTHY,
            'component_results': {}
        }

        try:
            components_to_check = [component_name] if component_name else list(
                self.health_checks.keys())

            for comp_name in components_to_check:
                if comp_name in self.health_checks:
                    check_result = self.health_checks[comp_name].perform_check()
                    results['component_results'][comp_name] = check_result

                    # Update overall status
                    if check_result['status'] == HealthStatus.UNHEALTHY:
                        results['overall_status'] = HealthStatus.UNHEALTHY
                    elif (check_result['status'] == HealthStatus.DEGRADED
                          and results['overall_status'] == HealthStatus.HEALTHY):
                        results['overall_status'] = HealthStatus.DEGRADED

            results['total_components'] = len(results['component_results'])
            results['healthy_components'] = sum(1 for r in results['component_results'].values()
                                                if r['status'] == HealthStatus.HEALTHY)

        except Exception as e:
            logger.error(f"Failed to perform health check: {str(e)}")
            results['error'] = str(e)

        return results

    def get_overall_health_status(self) -> Dict[str, Any]:
        """
        Get overall health status of all components
        获取所有组件的整体健康状态

        Returns:
            dict: Overall health status
                  整体健康状态
        """
        all_results = self.perform_health_check()

        return {
            'checker_name': self.checker_name,
            'overall_status': all_results['overall_status'].value,
            'total_components': all_results['total_components'],
            'healthy_components': all_results['healthy_components'],
            'unhealthy_components': all_results['total_components'] - all_results['healthy_components'],
            'component_summaries': {
                name: check.get_status_summary()
                for name, check in self.health_checks.items()
            },
            'last_check': all_results['timestamp']
        }

    def _checking_loop(self) -> None:
        """
        Main checking loop for periodic health checks
        定期健康检查的主要循环
        """
        logger.info(f"Health checking loop started for {self.checker_name}")

        while self.is_running:
            try:
                # Perform health checks for all components
                self.perform_health_check()

                # Wait before next check cycle (use the minimum interval)
                if self.health_checks:
                    min_interval = min(
                        check.interval for check in self.health_checks.values())
                    time.sleep(min_interval)
                else:
                    time.sleep(30)  # Default interval

            except Exception as e:
                logger.error(f"Health checking loop error: {str(e)}")
                time.sleep(30)

        logger.info(f"Health checking loop stopped for {self.checker_name}")


# Pre - built health check functions for common components


def check_http_service(url: str, timeout: float = 5.0) -> Dict[str, Any]:
    """
    Check HTTP service health
    检查HTTP服务健康状态

    Args:
        url: Service URL to check
            要检查的服务URL

    Returns:
        dict: Health check result
              健康检查结果
    """
    try:
        response = requests.get(url, timeout=timeout)
        return {
            'healthy': response.status_code == 200,
            'details': {
                'status_code': response.status_code,
                'response_time': response.elapsed.total_seconds(),
                'url': url
            }
        }
    except Exception as e:
        return {
            'healthy': False,
            'details': {'error': str(e), 'url': url}
        }


def check_tcp_service(host: str, port: int, timeout: float = 5.0) -> Dict[str, Any]:
    """
    Check TCP service health
    检查TCP服务健康状态

    Args:
        host: Host to check
             要检查的主机
        port: Port to check
             要检查的端口

    Returns:
        dict: Health check result
              健康检查结果
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()

        return {
            'healthy': result == 0,
            'details': {'host': host, 'port': port, 'connection_result': result}
        }
    except Exception as e:
        return {
            'healthy': False,
            'details': {'error': str(e), 'host': host, 'port': port}
        }


def check_database_connection(connection_string: str) -> Dict[str, Any]:
    """
    Check database connection health
    检查数据库连接健康状态

    Args:
        connection_string: Database connection string
                           数据库连接字符串

    Returns:
        dict: Health check result
              健康检查结果
    """
    # Placeholder implementation - in real usage, this would connect to actual database
    try:
        # Simulate database connection check
        time.sleep(0.1)  # Simulate connection time
        return {
            'healthy': True,  # Assume healthy for demo
            'details': {'connection_string': connection_string.replace('password=****', 'password=****')}
        }
    except Exception as e:
        return {
            'healthy': False,
            'details': {'error': str(e)}
        }


def check_filesystem(path: str) -> Dict[str, Any]:
    """
    Check filesystem health
    检查文件系统健康状态

    Args:
        path: Filesystem path to check
             要检查的文件系统路径

    Returns:
        dict: Health check result
              健康检查结果
    """
    try:
        import os
        stat = os.statvfs(path) if hasattr(os, 'statvfs') else None

        if stat:
            # Unix - like systems
            total = stat.f_blocks * stat.f_frsize
            free = stat.f_available * stat.f_frsize
            used_percent = ((total - free) / total) * 100

            return {
                'healthy': used_percent < 90,  # Consider unhealthy if >90% used
                'details': {
                    'path': path,
                    'total_bytes': total,
                    'free_bytes': free,
                    'used_percent': used_percent
                }
            }
        else:
            # Fallback for systems without statvfs
            return {
                'healthy': os.path.exists(path),
                'details': {'path': path, 'exists': os.path.exists(path)}
            }

    except Exception as e:
        return {
            'healthy': False,
            'details': {'error': str(e), 'path': path}
        }


# Global health checker instance
# 全局健康检查器实例
health_checker = HealthChecker()

__all__ = [
    'HealthStatus',
    'ComponentType',
    'HealthCheck',
    'HealthChecker',
    'health_checker',
    'check_http_service',
    'check_tcp_service',
    'check_database_connection',
    'check_filesystem'
]
