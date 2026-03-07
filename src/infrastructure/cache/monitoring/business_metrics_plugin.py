
# 模块级别的默认registry
import threading

from prometheus_client import Gauge, Counter, CollectorRegistry, generate_latest
from typing import Dict, Any, Optional
"""
基础设施层 - 工具组件组件

business_metrics_plugin 模块

通用工具组件
提供工具组件相关的功能实现。
"""

business_registry = CollectorRegistry()

# 模块级别的Prometheus指标（用于兼容性）
strategy_return_gauge = Gauge(
    'strategy_return',
    'Strategy return value',
    ['strategy'],
    registry=business_registry
)

active_users_gauge = Gauge(
    'active_users',
    'Number of active users',
    registry=business_registry
)

strategy_call_counter = Counter(
    'strategy_calls_total',
    'Total number of strategy calls',
    ['strategy'],
    registry=business_registry
)


class BusinessMetricsPlugin:
    """
    business_metrics_plugin - 缓存系统

    职责说明：
    负责数据缓存、内存管理、缓存策略和性能优化

    核心职责：
    - 内存缓存管理
    - Redis缓存操作
    - 缓存策略实现
    - 缓存性能监控
    - 缓存数据同步
    - 缓存失效处理

    相关接口：
    - ICacheComponent
    - ICacheManager
    - ICacheStrategy
    """

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """初始化业务指标收集器"""
        self.registry = registry or CollectorRegistry()
        self.strategy_returns = {}
        self.active_users = 0
        self.strategy_calls = {}
        self._lock = threading.Lock()

        # 初始化Prometheus指标
        self._init_prometheus_metrics()

    def _init_prometheus_metrics(self):
        """初始化Prometheus指标"""
        self.strategy_return_gauge = Gauge(
            'strategy_return',
            'Strategy return value',
            ['strategy'],
            registry=self.registry
        )

        self.active_users_gauge = Gauge(
            'active_users',
            'Number of active users',
            registry=self.registry
        )

        self.strategy_calls_counter = Counter(
            'strategy_calls_total',
            'Total number of strategy calls',
            ['strategy'],
            registry=self.registry
        )

    def update_strategy_return(self, strategy: str, value: float):
        """更新策略收益率"""
        with self._lock:
            self.strategy_returns[strategy] = value
            self.strategy_return_gauge.labels(strategy=strategy).set(value)

    def update_active_users(self, count: int):
        """更新活跃用户数"""
        with self._lock:
            self.active_users = count
            self.active_users_gauge.set(count)

    def inc_strategy_call(self, strategy: str):
        """增加策略调用次数"""
        with self._lock:
            self.strategy_calls[strategy] = self.strategy_calls.get(strategy, 0) + 1
            self.strategy_calls_counter.labels(strategy=strategy).inc()

    def get_metrics(self, as_prometheus: bool = False) -> Any:
        """获取指标信息

        Args:
            as_prometheus: 是否返回Prometheus文本格式
        """
        if as_prometheus:
            return self.get_metrics_prometheus()
        return self.get_metrics_dict()

    def get_metrics_prometheus(self) -> bytes:
        """以Prometheus格式导出指标"""
        return generate_latest(self.registry)

    def get_metrics_dict(self) -> Dict[str, Any]:
        """获取指标字典"""
        with self._lock:
            return {
                'strategy_returns': self.strategy_returns.copy(),
                'active_users': self.active_users,
                'strategy_calls': self.strategy_calls.copy()
            }

    def get_strategy_return(self, strategy: str) -> float:
        """获取策略收益率"""
        return self.strategy_returns.get(strategy, 0.0)

    def get_active_users(self) -> int:
        """获取活跃用户数"""
        return self.active_users

    def get_strategy_calls(self, strategy: str) -> int:
        """获取策略调用次数"""
        return self.strategy_calls.get(strategy, 0)

    def collect_metrics(self) -> Dict[str, Any]:
        """收集指标（别名方法，兼容测试用例）"""
        return self.get_metrics_dict()

    def increment_strategy_calls(self, strategy: str):
        """增加策略调用计数（别名方法，兼容测试用例）"""
        return self.inc_strategy_call(strategy)

# 兼容性函数


def update_strategy_return(strategy: str, value: float):
    """更新策略收益率（兼容性函数）"""
    global _default_collector
    if '_default_collector' not in globals():
        _default_collector = BusinessMetricsPlugin()
    _default_collector.update_strategy_return(strategy, value)
    # 同时更新模块级别的指标
    strategy_return_gauge.labels(strategy=strategy).set(value)


def update_active_users(count: int):
    """更新活跃用户数（兼容性函数）"""
    global _default_collector
    if '_default_collector' not in globals():
        _default_collector = BusinessMetricsPlugin()
    _default_collector.update_active_users(count)
    # 同时更新模块级别的指标
    active_users_gauge.set(count)


def inc_strategy_call(strategy: str):
    """增加策略调用次数（兼容性函数）"""
    global _default_collector
    if '_default_collector' not in globals():
        _default_collector = BusinessMetricsPlugin()
    _default_collector.inc_strategy_call(strategy)
    # 同时更新模块级别的指标
    strategy_call_counter.labels(strategy=strategy).inc()


def business_metrics():
    """获取业务指标（兼容性函数）"""
    return generate_latest(business_registry)
