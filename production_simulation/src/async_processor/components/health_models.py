"""
健康检查数据模型

健康检查相关的枚举和数据类。

从health_checker.py中提取以改善代码组织。

Author: RQA2025 Development Team
Date: 2025-11-01
"""

import logging
from typing import Any, Dict, List, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """Component type enumeration"""
    SERVICE = "service"
    DATABASE = "database"
    CACHE = "cache"
    QUEUE = "queue"
    API = "api"
    FILESYSTEM = "filesystem"
    NETWORK = "network"


class HealthCheck:
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
        """Execute health check with retry logic"""
        from datetime import datetime
        import time
        
        check_start = datetime.now()
        result = None
        
        for attempt in range(self.retries + 1):
            try:
                result = self.check_function()
                status = HealthStatus.HEALTHY if result.get('healthy', False) else HealthStatus.UNHEALTHY
                break
            except Exception as e:
                logger.warning(f"Health check failed (attempt {attempt + 1}/{self.retries + 1}): {e}")
                status = HealthStatus.UNHEALTHY
                if attempt < self.retries:
                    time.sleep(0.5)
        
        # Update tracking
        self.total_checks += 1
        self.last_check = datetime.now()
        self.last_status = status
        
        if status == HealthStatus.HEALTHY:
            self.successful_checks += 1
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
        
        # Record in history
        check_result = {
            'timestamp': check_start.isoformat(),
            'status': status.value,
            'duration': (datetime.now() - check_start).total_seconds(),
            'details': result if result else {'error': 'Check failed'}
        }
        
        self.status_history.append(check_result)
        if len(self.status_history) > self.max_history_size:
            self.status_history.pop(0)
        
        return check_result

    def get_status(self) -> HealthStatus:
        """Get current health status"""
        return self.last_status

    def get_statistics(self) -> Dict[str, Any]:
        """Get health check statistics"""
        return {
            'component_name': self.component_name,
            'component_type': self.component_type.value,
            'total_checks': self.total_checks,
            'successful_checks': self.successful_checks,
            'success_rate': (self.successful_checks / max(self.total_checks, 1)) * 100,
            'consecutive_failures': self.consecutive_failures,
            'last_status': self.last_status.value,
            'last_check': self.last_check.isoformat() if self.last_check else None
        }


__all__ = ['HealthStatus', 'ComponentType', 'HealthCheck']

