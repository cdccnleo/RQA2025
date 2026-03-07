
import threading

from ...core.shared_interfaces import ILogger, StandardLogger
from dataclasses import dataclass, field
from typing import Dict, List, Optional
"""
健康检查管理器

职责：管理健康检查项的注册、配置和生命周期
"""


@dataclass
class HealthCheck:
    """健康检查项"""
    name: str
    description: str
    check_function: callable
    enabled: bool = True
    timeout: int = 30  # 秒
    interval: int = 60  # 检查间隔(秒)
    tags: List[str] = field(default_factory=list)


class HealthCheckManager:
    """
    健康检查管理器

    职责：管理健康检查项的注册、配置和生命周期
    """

    def __init__(self, logger: Optional[ILogger] = None):
        self.logger = logger or StandardLogger(f"{self.__class__.__name__}")
        self._checks: Dict[str, HealthCheck] = {}
        self._lock = threading.RLock()

    def register_check(self, check: HealthCheck) -> bool:
        """注册健康检查项"""
        with self._lock:
            if check.name in self._checks:
                self.logger.log_warning(f"健康检查项已存在: {check.name}")
                return False

            self._checks[check.name] = check
            self.logger.log_info(f"已注册健康检查项: {check.name}")
            return True

    def unregister_check(self, check_name: str) -> bool:
        """注销健康检查项"""
        with self._lock:
            if check_name not in self._checks:
                self.logger.log_warning(f"健康检查项不存在: {check_name}")
                return False

            del self._checks[check_name]
            self.logger.log_info(f"已注销健康检查项: {check_name}")
            return True

    def get_check(self, check_name: str) -> Optional[HealthCheck]:
        """获取健康检查项"""
        with self._lock:
            return self._checks.get(check_name)

    def get_all_checks(self) -> List[HealthCheck]:
        """获取所有健康检查项"""
        with self._lock:
            return list(self._checks.values())

    def enable_check(self, check_name: str) -> bool:
        """启用健康检查项"""
        with self._lock:
            check = self._checks.get(check_name)
            if check:
                check.enabled = True
                self.logger.log_info(f"已启用健康检查项: {check_name}")
                return True
            return False

    def disable_check(self, check_name: str) -> bool:
        """禁用健康检查项"""
        with self._lock:
            check = self._checks.get(check_name)
            if check:
                check.enabled = False
                self.logger.log_info(f"已禁用健康检查项: {check_name}")
                return True
            return False

    def get_enabled_checks(self) -> List[HealthCheck]:
        """获取启用的健康检查项"""
        with self._lock:
            return [check for check in self._checks.values() if check.enabled]
