#!/usr/bin/env python3
"""
统一弹性机制接口

定义弹性层统一接口，确保所有弹性组件实现统一的API。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass
from datetime import datetime


class ResilienceLevel(Enum):
    """弹性级别"""
    NONE = "none"           # 无弹性
    BASIC = "basic"         # 基础弹性
    ADVANCED = "advanced"   # 高级弹性
    ENTERPRISE = "enterprise"  # 企业级弹性


class FailureType(Enum):
    """故障类型"""
    NETWORK = "network"         # 网络故障
    DATABASE = "database"       # 数据库故障
    SERVICE = "service"         # 服务故障
    RESOURCE = "resource"       # 资源故障
    CONFIGURATION = "configuration"  # 配置故障
    EXTERNAL = "external"       # 外部依赖故障


class CircuitBreakerState(Enum):
    """熔断器状态"""
    CLOSED = "closed"       # 关闭（正常）
    OPEN = "open"          # 开启（熔断）
    HALF_OPEN = "half_open"  # 半开（测试恢复）


class DegradationStrategy(Enum):
    """降级策略"""
    DISABLE_FEATURE = "disable_feature"     # 禁用功能
    REDUCE_FREQUENCY = "reduce_frequency"   # 降低频率
    LIMIT_CONCURRENCY = "limit_concurrency"  # 限制并发
    USE_CACHE = "use_cache"                 # 使用缓存
    FALLBACK = "fallback"                   # 降级到备用方案


class RecoveryAction(Enum):
    """恢复动作"""
    RESTART = "restart"             # 重启
    SCALE_UP = "scale_up"          # 扩容
    FAILOVER = "failover"          # 故障转移
    ROLLBACK = "rollback"          # 回滚
    RELOAD_CONFIG = "reload_config"  # 重新加载配置


@dataclass
class HealthStatus:
    """
    健康状态数据类

    表示组件或服务的健康状态。
    """
    component: str
    status: str
    timestamp: datetime
    response_time: Optional[float] = None
    error_count: int = 0
    success_count: int = 0
    last_error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CircuitBreakerConfig:
    """
    熔断器配置数据类

    定义熔断器的配置参数。
    """
    failure_threshold: int = 5        # 失败阈值
    recovery_timeout: int = 60        # 恢复超时(秒)
    expected_exception: Exception = Exception  # 预期异常类型
    success_threshold: int = 2        # 成功阈值（半开状态）
    timeout: float = 10.0            # 超时时间(秒)
    name: str = "default"             # 熔断器名称


@dataclass
class DegradationRule:
    """
    降级规则数据类

    定义降级规则的配置。
    """
    rule_id: str
    condition: str  # 降级条件
    strategy: DegradationStrategy
    priority: int = 1
    enabled: bool = True
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ResilienceEvent:
    """
    弹性事件数据类

    表示弹性机制中的事件。
    """
    event_id: str
    event_type: str
    component: str
    timestamp: datetime
    details: Dict[str, Any]
    severity: str = "info"

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class ICircuitBreaker(ABC):
    """
    熔断器接口

    所有熔断器实现必须遵循此接口，确保API的一致性。
    """

    @abstractmethod
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        执行函数调用（带熔断保护）

        Args:
            func: 要执行的函数
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            函数执行结果

        Raises:
            CircuitBreakerError: 熔断器打开时抛出
        """

    @abstractmethod
    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """
        异步执行函数调用（带熔断保护）

        Args:
            func: 要执行的异步函数
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            函数执行结果
        """

    @abstractmethod
    def get_state(self) -> CircuitBreakerState:
        """
        获取熔断器状态

        Returns:
            熔断器状态
        """

    @abstractmethod
    def reset(self) -> None:
        """重置熔断器状态"""

    @abstractmethod
    def get_failure_count(self) -> int:
        """
        获取失败次数

        Returns:
            失败次数
        """

    @abstractmethod
    def get_success_count(self) -> int:
        """
        获取成功次数

        Returns:
            成功次数
        """

    @abstractmethod
    def get_config(self) -> CircuitBreakerConfig:
        """
        获取熔断器配置

        Returns:
            熔断器配置
        """

    @abstractmethod
    def update_config(self, config: CircuitBreakerConfig) -> bool:
        """
        更新熔断器配置

        Args:
            config: 新的熔断器配置

        Returns:
            是否更新成功
        """


class IHealthChecker(ABC):
    """
    健康检查器接口
    """

    @abstractmethod
    def check_health(self, component: str) -> HealthStatus:
        """
        检查组件健康状态

        Args:
            component: 组件名称

        Returns:
            健康状态
        """

    @abstractmethod
    async def check_health_async(self, component: str) -> HealthStatus:
        """
        异步检查组件健康状态

        Args:
            component: 组件名称

        Returns:
            健康状态
        """

    @abstractmethod
    def register_check(self, component: str, check_func: Callable) -> bool:
        """
        注册健康检查函数

        Args:
            component: 组件名称
            check_func: 健康检查函数

        Returns:
            是否注册成功
        """

    @abstractmethod
    def unregister_check(self, component: str) -> bool:
        """
        注销健康检查函数

        Args:
            component: 组件名称

        Returns:
            是否注销成功
        """

    @abstractmethod
    def get_health_history(self, component: str, hours: int = 24) -> List[HealthStatus]:
        """
        获取健康检查历史

        Args:
            component: 组件名称
            hours: 历史时长(小时)

        Returns:
            健康状态历史列表
        """

    @abstractmethod
    def get_overall_health_score(self) -> float:
        """
        获取整体健康评分

        Returns:
            健康评分(0-100)
        """

    @abstractmethod
    def set_health_thresholds(self, thresholds: Dict[str, Any]) -> bool:
        """
        设置健康阈值

        Args:
            thresholds: 健康阈值字典

        Returns:
            是否设置成功
        """


class IDegradationManager(ABC):
    """
    降级管理器接口
    """

    @abstractmethod
    def add_degradation_rule(self, rule: DegradationRule) -> bool:
        """
        添加降级规则

        Args:
            rule: 降级规则

        Returns:
            是否添加成功
        """

    @abstractmethod
    def remove_degradation_rule(self, rule_id: str) -> bool:
        """
        移除降级规则

        Args:
            rule_id: 降级规则ID

        Returns:
            是否移除成功
        """

    @abstractmethod
    def enable_degradation_rule(self, rule_id: str) -> bool:
        """
        启用降级规则

        Args:
            rule_id: 降级规则ID

        Returns:
            是否启用成功
        """

    @abstractmethod
    def disable_degradation_rule(self, rule_id: str) -> bool:
        """
        禁用降级规则

        Args:
            rule_id: 降级规则ID

        Returns:
            是否禁用成功
        """

    @abstractmethod
    def evaluate_degradation_conditions(self) -> List[str]:
        """
        评估降级条件

        Returns:
            需要触发的降级规则ID列表
        """

    @abstractmethod
    def apply_degradation(self, rule_id: str) -> bool:
        """
        应用降级规则

        Args:
            rule_id: 降级规则ID

        Returns:
            是否应用成功
        """

    @abstractmethod
    def remove_degradation(self, rule_id: str) -> bool:
        """
        移除降级应用

        Args:
            rule_id: 降级规则ID

        Returns:
            是否移除成功
        """

    @abstractmethod
    def get_active_degradations(self) -> List[str]:
        """
        获取活跃的降级规则

        Returns:
            活跃降级规则ID列表
        """

    @abstractmethod
    def get_degradation_status(self) -> Dict[str, Any]:
        """
        获取降级状态

        Returns:
            降级状态字典
        """


class IRecoveryManager(ABC):
    """
    恢复管理器接口
    """

    @abstractmethod
    def initiate_recovery(self, component: str, failure_type: FailureType) -> bool:
        """
        启动恢复过程

        Args:
            component: 组件名称
            failure_type: 故障类型

        Returns:
            是否启动成功
        """

    @abstractmethod
    def check_recovery_status(self, recovery_id: str) -> Dict[str, Any]:
        """
        检查恢复状态

        Args:
            recovery_id: 恢复任务ID

        Returns:
            恢复状态字典
        """

    @abstractmethod
    def cancel_recovery(self, recovery_id: str) -> bool:
        """
        取消恢复过程

        Args:
            recovery_id: 恢复任务ID

        Returns:
            是否取消成功
        """

    @abstractmethod
    def add_recovery_strategy(self, failure_type: FailureType,
                              strategy: List[RecoveryAction]) -> bool:
        """
        添加恢复策略

        Args:
            failure_type: 故障类型
            strategy: 恢复动作列表

        Returns:
            是否添加成功
        """

    @abstractmethod
    def get_recovery_history(self, component: str, days: int = 7) -> List[Dict[str, Any]]:
        """
        获取恢复历史

        Args:
            component: 组件名称
            days: 历史天数

        Returns:
            恢复历史列表
        """

    @abstractmethod
    def test_recovery_procedures(self, component: str) -> Dict[str, Any]:
        """
        测试恢复流程

        Args:
            component: 组件名称

        Returns:
            测试结果字典
        """


class ILoadBalancer(ABC):
    """
    负载均衡器接口
    """

    @abstractmethod
    def get_instance(self, service_name: str, request_context: Dict[str, Any] = None) -> Optional[str]:
        """
        获取服务实例

        Args:
            service_name: 服务名称
            request_context: 请求上下文

        Returns:
            服务实例标识符
        """

    @abstractmethod
    def register_instance(self, service_name: str, instance_id: str,
                          metadata: Dict[str, Any] = None) -> bool:
        """
        注册服务实例

        Args:
            service_name: 服务名称
            instance_id: 服务实例ID
            metadata: 元数据

        Returns:
            是否注册成功
        """

    @abstractmethod
    def unregister_instance(self, service_name: str, instance_id: str) -> bool:
        """
        注销服务实例

        Args:
            service_name: 服务名称
            instance_id: 服务实例ID

        Returns:
            是否注销成功
        """

    @abstractmethod
    def update_instance_health(self, service_name: str, instance_id: str,
                               healthy: bool) -> bool:
        """
        更新实例健康状态

        Args:
            service_name: 服务名称
            instance_id: 服务实例ID
            healthy: 是否健康

        Returns:
            是否更新成功
        """

    @abstractmethod
    def get_service_instances(self, service_name: str) -> List[Dict[str, Any]]:
        """
        获取服务实例列表

        Args:
            service_name: 服务名称

        Returns:
            服务实例信息列表
        """

    @abstractmethod
    def set_load_balancing_strategy(self, strategy: str) -> bool:
        """
        设置负载均衡策略

        Args:
            strategy: 策略名称

        Returns:
            是否设置成功
        """


class IResilienceManager(ABC):
    """
    弹性管理器统一接口

    所有弹性管理器实现必须遵循此接口，确保API的一致性。
    """

    @abstractmethod
    def get_resilience_level(self) -> ResilienceLevel:
        """
        获取弹性级别

        Returns:
            弹性级别
        """

    @abstractmethod
    def set_resilience_level(self, level: ResilienceLevel) -> bool:
        """
        设置弹性级别

        Args:
            level: 弹性级别

        Returns:
            是否设置成功
        """

    @abstractmethod
    def handle_failure(self, component: str, failure_type: FailureType,
                       failure_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理故障

        Args:
            component: 组件名称
            failure_type: 故障类型
            failure_details: 故障详细信息

        Returns:
            处理结果字典
        """

    @abstractmethod
    def get_system_resilience_score(self) -> float:
        """
        获取系统弹性评分

        Returns:
            弹性评分(0-100)
        """

    @abstractmethod
    def enable_auto_recovery(self, enabled: bool = True) -> None:
        """
        启用/禁用自动恢复

        Args:
            enabled: 是否启用
        """

    @abstractmethod
    def is_auto_recovery_enabled(self) -> bool:
        """
        检查自动恢复是否启用

        Returns:
            是否启用
        """

    @abstractmethod
    def get_failure_statistics(self, days: int = 7) -> Dict[str, Any]:
        """
        获取故障统计信息

        Args:
            days: 统计天数

        Returns:
            故障统计字典
        """

    @abstractmethod
    def get_resilience_events(self, hours: int = 24) -> List[ResilienceEvent]:
        """
        获取弹性事件

        Args:
            hours: 事件时长(小时)

        Returns:
            弹性事件列表
        """

    @abstractmethod
    def simulate_failure(self, component: str, failure_type: FailureType) -> Dict[str, Any]:
        """
        模拟故障（测试用）

        Args:
            component: 组件名称
            failure_type: 故障类型

        Returns:
            模拟结果字典
        """

    @abstractmethod
    def test_resilience_mechanisms(self) -> Dict[str, Any]:
        """
        测试弹性机制

        Returns:
            测试结果字典
        """

    @abstractmethod
    def generate_resilience_report(self) -> Dict[str, Any]:
        """
        生成弹性报告

        Returns:
            弹性报告字典
        """
