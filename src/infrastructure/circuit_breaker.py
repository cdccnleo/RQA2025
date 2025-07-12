from prometheus_client import Gauge, Counter, REGISTRY
from enum import Enum, auto
import threading
from typing import Optional

class CircuitState(Enum):
    CLOSED = auto()
    HALF_OPEN = auto()
    OPEN = auto()

class CircuitBreaker:
    """增强监控的熔断器实现"""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        registry=None,
        **kwargs
    ):
        """
        初始化熔断器

        Args:
            name: 熔断器名称(用于监控区分)
            failure_threshold: 触发熔断的失败次数
            recovery_timeout: 恢复超时(秒)
            registry: Prometheus注册表，默认为全局REGISTRY
            kwargs: 其他配置参数
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._lock = threading.Lock()
        
        # 使用传入的registry或默认全局registry
        self._registry = registry or REGISTRY
        
        # 初始化监控指标为实例属性
        self._init_metrics()

    def _init_metrics(self):
        """初始化Prometheus监控指标"""
        try:
            self.state_gauge = Gauge(
                'circuit_breaker_state', 'Circuit breaker state',
                ['service_name'], registry=self._registry
            )
        except ValueError:
            # 如果指标已存在，尝试获取现有指标
            self.state_gauge = Gauge(
                'circuit_breaker_state', 'Circuit breaker state',
                ['service_name'], registry=self._registry
            )
        
        try:
            self.state_change_counter = Counter(
                'circuit_breaker_state_changes',
                'Total circuit breaker state changes',
                ['name', 'from_state', 'to_state'], registry=self._registry
            )
        except ValueError:
            self.state_change_counter = Counter(
                'circuit_breaker_state_changes',
                'Total circuit breaker state changes',
                ['name', 'from_state', 'to_state'], registry=self._registry
            )
        
        try:
            self.trip_counter = Counter(
                'circuit_breaker_trips',
                'Total circuit breaker trips',
                ['name', 'reason'], registry=self._registry
            )
        except ValueError:
            self.trip_counter = Counter(
                'circuit_breaker_trips',
                'Total circuit breaker trips',
                ['name', 'reason'], registry=self._registry
            )

        # 初始化状态指标
        self.state_gauge.labels(service_name=self.name).set(0)  # 初始状态: CLOSED

    @property
    def state(self) -> CircuitState:
        """获取当前状态"""
        return self._state

    @state.setter
    def state(self, new_state: CircuitState):
        """更新状态并记录变更"""
        with self._lock:
            old_state = self._state.name
            self._state = new_state

            # 更新状态指标
            state_value = {
                CircuitState.CLOSED: 0,
                CircuitState.HALF_OPEN: 1,
                CircuitState.OPEN: 2
            }[new_state]
            self.state_gauge.labels(service_name=self.name).set(state_value)

            # 记录状态变更
            self.state_change_counter.labels(
                name=self.name,
                from_state=old_state,
                to_state=new_state.name
            ).inc()

    def trip(self, reason: Optional[str] = None):
        """触发熔断"""
        with self._lock:
            if self._state != CircuitState.OPEN:
                # 记录熔断事件
                self.trip_counter.labels(
                    name=self.name,
                    reason=reason or "unknown"
                ).inc()

                # 更新状态
                self.state = CircuitState.OPEN

    def record_failure(self, error: Optional[Exception] = None):
        """记录失败并检查是否需要熔断"""
        with self._lock:
            self._failure_count += 1
            if self._failure_count >= self.failure_threshold:
                self.trip(reason=str(error) if error else "threshold_exceeded")

    def reset(self):
        """重置熔断器"""
        with self._lock:
            self._failure_count = 0
            self.state = CircuitState.CLOSED

    def can_execute(self) -> bool:
        """检查是否可以执行操作"""
        return self._state != CircuitState.OPEN

    def get_failure_count(self) -> int:
        """获取失败次数"""
        return self._failure_count

    def get_state_name(self) -> str:
        """获取状态名称"""
        return self._state.name
