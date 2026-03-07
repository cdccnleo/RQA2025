"""
health 模块

提供 health 相关功能和接口。
"""

import logging


from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from src.infrastructure.health.core.exceptions import ValidationError, HealthInfrastructureError
from typing import Any, Optional, Dict, List, Callable
logger = logging.getLogger(__name__)
"""
基础设施层 - 健康检查组件

health 模块

健康检查相关的文件
提供健康检查相关的功能实现。
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
健康检查统一接口
定义健康检查系统的标准接口，支持多种健康检查策略和状态管理
"""


class HealthStatus(Enum):

    """健康状态枚举"""
    HEALTHY = "healthy"          # 健康
    DEGRADED = "degraded"        # 降级
    UNHEALTHY = "unhealthy"      # 不健康
    UNKNOWN = "unknown"          # 未知


class HealthCheckType(Enum):

    """健康检查类型"""
    LIVENESS = "liveness"        # 存活检查
    READINESS = "readiness"      # 就绪检查
    STARTUP = "startup"          # 启动检查
    CUSTOM = "custom"            # 自定义检查


@dataclass
class HealthCheckResult:

    """健康检查结果"""
    name: str
    status: HealthStatus
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None
    duration_ms: float = 0.0
    error: Optional[Exception] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """数据验证"""
        self._validate_data()
        # 设置默认时间戳
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def _validate_data(self) -> None:
        """验证数据完整性和正确性"""
        try:
            # 验证name
            if not self.name or not isinstance(self.name, str):
                raise ValidationError(
                    "健康检查名称不能为空且必须是字符串",
                    field="name",
                    value=self.name,
                    validation_rule="非空字符串"
                )

            # 验证status
            if not isinstance(self.status, HealthStatus):
                raise ValidationError(
                    "健康状态必须是HealthStatus枚举值",
                    field="status",
                    value=self.status,
                    validation_rule="HealthStatus枚举值"
                )

            # 验证message
            if not self.message or not isinstance(self.message, str):
                raise ValidationError(
                    "消息不能为空且必须是字符串",
                    field="message",
                    value=self.message,
                    validation_rule="非空字符串"
                )

            # 验证duration_ms
            if not isinstance(self.duration_ms, (int, float)) or self.duration_ms < 0:
                raise ValidationError(
                    "持续时间必须是非负数",
                    field="duration_ms",
                    value=self.duration_ms,
                    validation_rule="非负数"
                )

            # 验证details
            if self.details is not None and not isinstance(self.details, dict):
                raise ValidationError(
                    "详细信息必须是字典或None",
                    field="details",
                    value=self.details,
                    validation_rule="字典或None"
                )

            # 验证metadata
            if self.metadata is not None and not isinstance(self.metadata, dict):
                raise ValidationError(
                    "元数据必须是字典或None",
                    field="metadata",
                    value=self.metadata,
                    validation_rule="字典或None"
                )

        except ValidationError:
            raise  # 重新抛出验证错误
        except Exception as e:
            logger.error(f"健康检查结果数据验证失败: {e}", exc_info=True)
            raise HealthInfrastructureError(f"数据验证失败: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        try:
            return {
                "name": self.name,
                "status": self.status.value,
                "message": self.message,
                "details": self.details or {},
                "timestamp": self.timestamp.isoformat() if self.timestamp else None,
                "duration_ms": self.duration_ms,
                "error": str(self.error) if self.error else None,
                "metadata": self.metadata or {}
            }
        except Exception as e:
            logger.error(f"转换为字典格式失败: {e}", exc_info=True)
            raise HealthInfrastructureError(f"序列化失败: {e}")

    def is_healthy(self) -> bool:
        """检查是否健康"""
        return self.status == HealthStatus.HEALTHY

    def is_degraded(self) -> bool:
        """检查是否降级"""
        return self.status == HealthStatus.DEGRADED

    def is_unhealthy(self) -> bool:
        """检查是否不健康"""
        return self.status == HealthStatus.UNHEALTHY

    def has_error(self) -> bool:
        """检查是否有错误"""
        return self.error is not None


@dataclass
class HealthSummary:

    """健康状态摘要"""
    overall_status: HealthStatus
    total_checks: int
    healthy_checks: int
    degraded_checks: int
    unhealthy_checks: int
    unknown_checks: int
    last_check_time: datetime
    response_time_ms: float
    details: Dict[str, HealthCheckResult]

    def __post_init__(self):
        """数据验证"""
        self._validate_data()

    def _validate_data(self) -> None:
        """验证数据完整性和正确性"""
        try:
            # 验证overall_status
            if not isinstance(self.overall_status, HealthStatus):
                raise ValidationError(
                    "整体状态必须是HealthStatus枚举值",
                    field="overall_status",
                    value=self.overall_status,
                    validation_rule="HealthStatus枚举值"
                )

            # 验证检查计数
            count_fields = ['total_checks', 'healthy_checks', 'degraded_checks',
                            'unhealthy_checks', 'unknown_checks']
            for field in count_fields:
                value = getattr(self, field)
                if not isinstance(value, int) or value < 0:
                    raise ValidationError(
                        f"{field}必须是非负整数",
                        field=field,
                        value=value,
                        validation_rule="非负整数"
                    )

            # 验证计数总和
            total_from_parts = (self.healthy_checks + self.degraded_checks +
                                self.unhealthy_checks + self.unknown_checks)
            if total_from_parts != self.total_checks:
                raise ValidationError(
                    "各状态检查数之和必须等于总检查数",
                    field="total_checks",
                    value=self.total_checks,
                    validation_rule=f"各部分之和应为 {total_from_parts}"
                )

            # 验证last_check_time
            if not isinstance(self.last_check_time, datetime):
                raise ValidationError(
                    "最后检查时间必须是datetime对象",
                    field="last_check_time",
                    value=self.last_check_time,
                    validation_rule="datetime对象"
                )

            # 验证response_time_ms
            if not isinstance(self.response_time_ms, (int, float)) or self.response_time_ms < 0:
                raise ValidationError(
                    "响应时间必须是非负数",
                    field="response_time_ms",
                    value=self.response_time_ms,
                    validation_rule="非负数"
                )

            # 验证details
            if not isinstance(self.details, dict):
                raise ValidationError(
                    "详细信息必须是字典",
                    field="details",
                    value=self.details,
                    validation_rule="字典"
                )

            # 验证details中的值
            for key, value in self.details.items():
                if not isinstance(value, HealthCheckResult):
                    raise ValidationError(
                        f"details['{key}']必须是HealthCheckResult实例",
                        field=f"details['{key}']",
                        value=value,
                        validation_rule="HealthCheckResult实例"
                    )

        except ValidationError:
            raise  # 重新抛出验证错误
        except Exception as e:
            logger.error(f"健康摘要数据验证失败: {e}", exc_info=True)
            raise HealthInfrastructureError(f"数据验证失败: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        try:
            return {
                "overall_status": self.overall_status.value,
                "total_checks": self.total_checks,
                "healthy_checks": self.healthy_checks,
                "degraded_checks": self.degraded_checks,
                "unhealthy_checks": self.unhealthy_checks,
                "unknown_checks": self.unknown_checks,
                "last_check_time": self.last_check_time.isoformat(),
                "response_time_ms": self.response_time_ms,
                "details": {k: v.to_dict() for k, v in self.details.items()}
            }
        except Exception as e:
            logger.error(f"转换为字典格式失败: {e}", exc_info=True)
            raise HealthInfrastructureError(f"序列化失败: {e}")

    def get_health_percentage(self) -> float:
        """获取健康百分比"""
        if self.total_checks == 0:
            return 0.0
        return (self.healthy_checks / self.total_checks) * 100.0

    def has_critical_issues(self) -> bool:
        """检查是否有严重问题"""
        return self.unhealthy_checks > 0 or self.unknown_checks > 0


class HealthCheckInterface(ABC):

    """健康检查标准接口"""

    @abstractmethod
    def check_health(self) -> HealthCheckResult:
        """
        执行健康检查

        Returns:
            健康检查结果
        """

    @abstractmethod
    def get_name(self) -> str:
        """
        获取健康检查名称

        Returns:
            健康检查名称
        """

    @abstractmethod
    def get_type(self) -> HealthCheckType:
        """
        获取健康检查类型

        Returns:
            健康检查类型
        """

    @abstractmethod
    def is_enabled(self) -> bool:
        """
        检查是否启用

        Returns:
            是否启用
        """

    @abstractmethod
    def set_enabled(self, enabled: bool) -> None:
        """
        设置启用状态

        Args:
            enabled: 是否启用
        """


class AsyncHealthCheckInterface(ABC):

    """异步健康检查接口"""

    @abstractmethod
    async def check_health_async(self) -> HealthCheckResult:
        """
        异步执行健康检查

        Returns:
            健康检查结果
        """


class HealthCheckerInterface(ABC):

    """健康检查器接口"""

    @abstractmethod
    def add_check(self, check: HealthCheckInterface) -> bool:
        """
        添加健康检查

        Args:
            check: 健康检查实例

        Returns:
            是否添加成功
        """

    @abstractmethod
    def remove_check(self, name: str) -> bool:
        """
        移除健康检查

        Args:
            name: 健康检查名称

        Returns:
            是否移除成功
        """

    @abstractmethod
    def get_check(self, name: str) -> Optional[HealthCheckInterface]:
        """
        获取指定的健康检查

        Args:
            name: 健康检查名称

        Returns:
            健康检查实例
        """

    @abstractmethod
    def list_checks(self) -> List[HealthCheckInterface]:
        """
        列出所有健康检查

        Returns:
            健康检查列表
        """

    @abstractmethod
    def run_checks(self) -> HealthSummary:
        """
        运行所有健康检查

        Returns:
            健康状态摘要
        """

    @abstractmethod
    def run_check(self, name: str) -> Optional[HealthCheckResult]:
        """
        运行指定的健康检查

        Args:
            name: 健康检查名称

        Returns:
            健康检查结果
        """


class AsyncHealthCheckerInterface(ABC):

    """异步健康检查器接口"""

    @abstractmethod
    async def run_checks_async(self) -> HealthSummary:
        """
        异步运行所有健康检查

        Returns:
            健康状态摘要
        """

    @abstractmethod
    async def run_check_async(self, name: str) -> Optional[HealthCheckResult]:
        """
        异步运行指定的健康检查

        Args:
            name: 健康检查名称

        Returns:
            健康检查结果
        """


class HealthReporterInterface(ABC):

    """健康状态报告器接口"""

    @abstractmethod
    def report_health(self, summary: HealthSummary) -> bool:
        """
        报告健康状态

        Args:
            summary: 健康状态摘要

        Returns:
            是否报告成功
        """

    @abstractmethod
    def get_health_endpoint(self) -> str:
        """
        获取健康检查端点

        Returns:
            健康检查端点URL
        """

    @abstractmethod
    def set_health_endpoint(self, endpoint: str) -> None:
        """
        设置健康检查端点

        Args:
            endpoint: 健康检查端点URL
        """


class HealthMonitorInterface(ABC):

    """健康监控器接口"""

    @abstractmethod
    def start_monitoring(self, interval_seconds: int = 30) -> bool:
        """
        开始健康监控

        Args:
            interval_seconds: 监控间隔（秒）

        Returns:
            是否开始成功
        """

    @abstractmethod
    def stop_monitoring(self) -> bool:
        """
        停止健康监控

        Returns:
            是否停止成功
        """

    @abstractmethod
    def is_monitoring(self) -> bool:
        """
        检查是否正在监控

        Returns:
            是否正在监控
        """

    @abstractmethod
    def set_alert_callback(self, callback: Callable[[HealthSummary], None]) -> None:
        """
        设置告警回调函数

        Args:
            callback: 告警回调函数
        """

    @abstractmethod
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """
        获取监控统计信息

        Returns:
            监控统计信息
        """
