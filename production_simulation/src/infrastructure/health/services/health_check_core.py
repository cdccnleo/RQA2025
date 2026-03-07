"""
health_check_core 模块

提供 health_check_core 相关功能和接口。
"""

import logging

import time

from ..core.interfaces import IUnifiedInfrastructureInterface
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List
"""
RQA2025 基础设施层健康检查核心实现

提供健康检查的核心功能和基础实现。
"""


class HealthStatus(Enum):
    """健康状态枚举"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """健康检查结果"""
    status: HealthStatus
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: float = None
    duration_ms: float = 0.0

    def __post_init__(self):

        if self.timestamp is None:
            self.timestamp = time.time()


class IHealthCheckProviderComponent(ABC):
    """HealthCheckProvider组件接口

    定义HealthCheckProvider功能的核心抽象接口。
    """

    @abstractmethod
    def check_health(self) -> HealthCheckResult:
        """执行健康检查"""

    @abstractmethod
    def get_name(self) -> str:
        """获取健康检查提供者名称"""


class HealthCheckCore(IUnifiedInfrastructureInterface):
    """健康检查核心类"""

    def __init__(self, name: str = "HealthCheckCore"):

        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.providers: Dict[str, IHealthCheckProviderComponent] = {}
        self.check_history: List[HealthCheckResult] = []
        self.max_history_size = 100
        self._initialized = False
        self._check_count = 0
        self._last_check_time = None

    def register_provider(self, provider: IHealthCheckProviderComponent) -> bool:
        """注册健康检查提供者"""
        try:
            provider_name = provider.get_name()
            self.providers[provider_name] = provider
            self.logger.info(f"注册健康检查提供者: {provider_name}")
            return True
        except Exception as e:
            self.logger.error(f"注册健康检查提供者失败: {e}")
            return False

    def unregister_provider(self, provider_name: str) -> bool:
        """注销健康检查提供者"""
        if provider_name in self.providers:
            del self.providers[provider_name]
            self.logger.info(f"注销健康检查提供者: {provider_name}")
            return True
        return False

    def check_all_health(self) -> Dict[str, HealthCheckResult]:
        """检查所有提供者的健康状态"""
        results = {}

        for provider_name, provider in self.providers.items():
            try:
                start_time = time.time()
                result = provider.check_health()
                result.duration_ms = (time.time() - start_time) * 1000
                results[provider_name] = result

                # 记录到历史
                self._add_to_history(result)

                self.logger.debug(f"健康检查完成: {provider_name} -> {result.status.value}")

            except Exception as e:
                error_result = HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message=f"健康检查异常: {str(e)}",
                    details={"error": str(e)},
                    duration_ms=0.0
                )

                results[provider_name] = error_result
                self.logger.error(f"健康检查异常: {provider_name} - {e}")

        return results

    def check_provider_health(self, provider_name: str) -> Optional[HealthCheckResult]:
        """检查指定提供者的健康状态"""
        if provider_name not in self.providers:
            self.logger.warning(f"健康检查提供者不存在: {provider_name}")
            return None

        try:
            start_time = time.time()
            result = self.providers[provider_name].check_health()
            result.duration_ms = (time.time() - start_time) * 1000

            # 记录到历史
            self._add_to_history(result)

            return result

        except Exception as e:
            error_result = HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"健康检查异常: {str(e)}",
                details={"error": str(e)},
                duration_ms=0.0
            )

            self.logger.error(f"健康检查异常: {provider_name} - {e}")
            return error_result

    def get_overall_health(self) -> HealthStatus:
        """获取整体健康状态"""
        if not self.providers:
            return HealthStatus.UNKNOWN

        results = self.check_all_health()

        # 统计各状态数量
        status_counts = {}
        for result in results.values():
            status = result.status
            status_counts[status] = status_counts.get(status, 0) + 1

        # 判断整体状态
        if HealthStatus.UNHEALTHY in status_counts:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in status_counts:
            return HealthStatus.DEGRADED
        elif HealthStatus.HEALTHY in status_counts:
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN

    def get_health_summary(self) -> Dict[str, Any]:
        """获取健康状态摘要"""
        overall_status = self.get_overall_health()
        results = self.check_all_health()

        # 统计各状态数量
        status_counts = {}
        for result in results.values():
            status = result.status
            status_counts[status.value] = status_counts.get(status.value, 0) + 1

        return {
            "overall_status": overall_status.value,
            "total_providers": len(self.providers),
            "status_counts": status_counts,
            "last_check": time.time(),
            "providers": {
                name: {
                    "status": result.status.value,
                    "message": result.message,
                    "duration_ms": result.duration_ms,
                    "timestamp": result.timestamp
                }
                for name, result in results.items()
            }
        }

    def _add_to_history(self, result: HealthCheckResult):
        """添加检查结果到历史记录"""
        self.check_history.append(result)

        # 限制历史记录大小
        if len(self.check_history) > self.max_history_size:
            self.check_history.pop(0)

    def get_check_history(self, limit: Optional[int] = None) -> List[HealthCheckResult]:
        """获取检查历史记录"""
        if limit is None:
            return self.check_history.copy()
        return self.check_history[-limit:].copy()

    def clear_history(self):
        """清空检查历史记录"""
        self.check_history.clear()
        self.logger.info("清空健康检查历史记录")

    def get_provider_names(self) -> List[str]:
        """获取所有提供者名称"""
        return list(self.providers.keys())

    def has_provider(self, provider_name: str) -> bool:
        """检查是否存在指定的提供者"""
        return provider_name in self.providers

    # IUnifiedInfrastructureInterface 实现
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """初始化健康检查核心

        Args:
            config: 配置参数

        Returns:
            bool: 初始化是否成功
        """
        try:
            self.logger.info(f"开始初始化HealthCheckCore: {self.name}")

            if config:
                self.max_history_size = config.get('max_history_size', self.max_history_size)
                # 可以在这里处理其他配置参数

            self._initialized = True
            self.logger.info(f"HealthCheckCore初始化成功: {self.name}")
            return True

        except Exception as e:
            self.logger.error(f"HealthCheckCore初始化失败: {str(e)}", exc_info=True)
            self._initialized = False
            return False

    def get_component_info(self) -> Dict[str, Any]:
        """获取组件信息

        Returns:
            Dict[str, Any]: 组件信息
        """
        try:
            self.logger.debug(f"获取HealthCheckCore组件信息: {self.name}")

            return {
                "component_type": "HealthCheckCore",
                "name": self.name,
                "initialized": self._initialized,
                "providers_count": len(self.providers),
                "check_history_size": len(self.check_history),
                "max_history_size": self.max_history_size,
                "check_count": self._check_count,
                "last_check_time": self._last_check_time.isoformat() if self._last_check_time else None,
                "provider_names": list(self.providers.keys())
            }
        except Exception as e:
            self.logger.error(f"获取HealthCheckCore组件信息失败: {str(e)}")
            return {"error": str(e)}

    def is_healthy(self) -> bool:
        """检查组件健康状态

        Returns:
            bool: 组件是否健康
        """
        try:
            self.logger.debug(f"检查HealthCheckCore健康状态: {self.name}")

            # 检查基本状态
            if not self._initialized:
                self.logger.warning(f"HealthCheckCore未初始化: {self.name}")
                return False

            # 检查是否有提供者
            if not self.providers:
                self.logger.warning(f"HealthCheckCore没有注册的提供者: {self.name}")
                return False

            # 检查提供者是否都健康
            unhealthy_providers = []
            for name, provider in self.providers.items():
                try:
                    if hasattr(provider, 'is_healthy') and not provider.is_healthy():
                        unhealthy_providers.append(name)
                except Exception as e:
                    self.logger.warning(f"检查提供者健康状态失败 {name}: {str(e)}")
                    unhealthy_providers.append(name)

            if unhealthy_providers:
                self.logger.warning(f"发现不健康的提供者: {unhealthy_providers}")
                return False

            # 检查是否能够执行基本操作
            try:
                self.get_component_info()
                return True
            except Exception as e:
                self.logger.error(f"HealthCheckCore健康检查失败: {str(e)}")
                return False

        except Exception as e:
            self.logger.error(f"HealthCheckCore健康检查异常: {str(e)}")
            return False

    def get_metrics(self) -> Dict[str, Any]:
        """获取组件指标

        Returns:
            Dict[str, Any]: 组件指标数据
        """
        try:
            self.logger.debug(f"获取HealthCheckCore指标: {self.name}")
            
            return {
                "component_metrics": self._get_component_metrics(),
                "providers_metrics": self._get_providers_metrics(),
                "history_metrics": self._get_history_metrics(),
                "performance_metrics": self._get_performance_metrics()
            }
        except Exception as e:
            self.logger.error(f"获取HealthCheckCore指标失败: {str(e)}")
            return {"error": str(e)}

    def _get_component_metrics(self) -> Dict[str, Any]:
        """获取组件基础指标"""
        uptime_seconds = 0
        if self._last_check_time:
            uptime_seconds = (datetime.now() - self._last_check_time).total_seconds()
        
        return {
            "name": self.name,
            "initialized": self._initialized,
            "check_count": self._check_count,
            "last_check_time": self._last_check_time.isoformat() if self._last_check_time else None,
            "uptime_seconds": uptime_seconds
        }

    def _get_providers_metrics(self) -> Dict[str, Any]:
        """获取提供者相关指标"""
        provider_health = self._calculate_provider_health()
        
        return {
            "total_providers": len(self.providers),
            "healthy_providers": sum(provider_health.values()),
            "provider_health_status": provider_health
        }

    def _calculate_provider_health(self) -> Dict[str, bool]:
        """计算提供者健康状态"""
        provider_health = {}
        for name, provider in self.providers.items():
            try:
                is_healthy = hasattr(provider, 'is_healthy') and provider.is_healthy()
                provider_health[name] = is_healthy
            except Exception:
                provider_health[name] = False
        return provider_health

    def _get_history_metrics(self) -> Dict[str, Any]:
        """获取历史记录相关指标"""
        healthy_checks = sum(
            1 for result in self.check_history if result.status == HealthStatus.HEALTHY)
        total_checks = len(self.check_history)
        success_rate = healthy_checks / total_checks if total_checks > 0 else 0
        
        history_utilization = 0
        if self.max_history_size > 0:
            history_utilization = len(self.check_history) / self.max_history_size
        
        return {
            "total_history": len(self.check_history),
            "max_history": self.max_history_size,
            "healthy_checks": healthy_checks,
            "success_rate": success_rate,
            "history_utilization": history_utilization
        }

    def _get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        now = datetime.now()
        last_check = self._last_check_time or now
        time_diff_seconds = max(1, (now - last_check).total_seconds())
        checks_per_minute = self._check_count / (time_diff_seconds / 60)
        
        return {
            "checks_per_minute": checks_per_minute
        }

    def cleanup(self) -> bool:
        """清理组件资源

        Returns:
            bool: 清理是否成功
        """
        try:
            self.logger.info(f"开始清理HealthCheckCore资源: {self.name}")

            # 清理提供者
            self.providers.clear()

            # 清理历史记录
            self.check_history.clear()

            # 重置计数器
            self._check_count = 0
            self._last_check_time = None

            # 保持初始化状态，但清理运行时数据
            self.logger.info(f"HealthCheckCore资源清理完成: {self.name}")
            return True

        except Exception as e:
            self.logger.error(f"HealthCheckCore资源清理失败: {str(e)}", exc_info=True)
            return False
