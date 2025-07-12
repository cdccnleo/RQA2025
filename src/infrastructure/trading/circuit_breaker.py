from prometheus_client import Gauge, Counter
from datetime import datetime, timedelta
from typing import Optional, Dict
import threading

def is_trading_session_active(trading_hours: dict = None) -> bool:
    """检查当前是否在交易时段内"""
    if not trading_hours:
        return False
        
    now = datetime.now().time()
    for period in trading_hours.values():
        start = datetime.strptime(period['start'], '%H:%M').time()
        end = datetime.strptime(period['end'], '%H:%M').time()
        if start <= now < end:
            return True
    return False


class InstrumentedCircuitBreaker:
    """增强版熔断器（生产级实现）"""

    STATE_CLOSED = 0
    STATE_HALF_OPEN = 1
    STATE_OPEN = 2

    def __init__(self,
                 name: str,
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 trading_hours: Optional[Dict] = None):
        """
        Args:
            name: 熔断器名称(按功能或品种命名)
            failure_threshold: 触发熔断的失败次数阈值
            recovery_timeout: 恢复超时时间(秒)
            trading_hours: 交易时段配置 {
                'morning': {'start': '09:30', 'end': '11:30'},
                'afternoon': {'start': '13:00', 'end': '15:00'}
            }
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.trading_hours = trading_hours or {}

        # 运行时状态
        self._state = self.STATE_CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._lock = threading.RLock()

        # 监控指标
        self.state_gauge = Gauge(
            'circuit_breaker_state',
            'Current state of circuit breaker',
            ['breaker_name', 'instrument']
        )
        self.failure_counter = Counter(
            'circuit_breaker_failures',
            'Total failure counts',
            ['breaker_name', 'instrument', 'error_type']
        )
        self.state_change_counter = Counter(
            'circuit_breaker_state_changes',
            'Total state transition counts',
            ['breaker_name', 'from_state', 'to_state']
        )

    @property
    def state(self) -> int:
        """获取当前状态"""
        return self._state

    def record_failure(self, error_type: str = "default", instrument: str = "default"):
        """记录失败事件"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.now()
            self.failure_counter.labels(
                breaker_name=self.name,
                instrument=instrument,
                error_type=error_type
            ).inc()

            if (self._state == self.STATE_CLOSED and
                self._failure_count >= self.failure_threshold):
                self._trip()

    def record_success(self):
        """记录成功事件"""
        with self._lock:
            if self._state == self.STATE_HALF_OPEN:
                self._reset()
            elif self._state == self.STATE_CLOSED:
                self._failure_count = 0

    def _trip(self):
        """触发熔断"""
        prev_state = self._state
        self._state = self.STATE_OPEN
        self._update_metrics(prev_state)

        # 发送实时告警
        self._send_alert(
            action="TRIPPED",
            details={
                "threshold": self.failure_threshold,
                "failure_count": self._failure_count,
                "last_error_time": self._last_failure_time.isoformat()
            }
        )

        # 设置恢复检查定时器
        threading.Timer(
            self._get_effective_timeout(),
            self._attempt_recovery
        ).start()

    def _attempt_recovery(self):
        """尝试恢复"""
        if self._state == self.STATE_OPEN:
            with self._lock:
                if self._should_attempt_recovery():
                    prev_state = self._state
                    self._state = self.STATE_HALF_OPEN
                    self._update_metrics(prev_state)

    def _should_attempt_recovery(self) -> bool:
        """判断是否允许尝试恢复"""
        if not self.trading_hours:
            return True

        now = datetime.now().time()
        for period in self.trading_hours.values():
            start = datetime.strptime(period['start'], '%H:%M').time()
            end = datetime.strptime(period['end'], '%H:%M').time()
            if start <= now < end:
                return False
        return True

    def _reset(self):
        """重置熔断器"""
        prev_state = self._state
        self._state = self.STATE_CLOSED
        self._failure_count = 0
        self._update_metrics(prev_state)

        # 发送恢复通知
        self._send_alert(
            action="RESET",
            details={"reset_time": datetime.now().isoformat()}
        )

    def _update_metrics(self, prev_state: int):
        """更新监控指标"""
        self.state_gauge.labels(
            breaker_name=self.name,
            instrument="global"
        ).set(self._state)

        self.state_change_counter.labels(
            breaker_name=self.name,
            from_state=prev_state,
            to_state=self._state
        ).inc()

    def _send_alert(self, action: str, details: Dict):
        """发送告警通知"""
        # 实现与告警系统的集成
        alert_payload = {
            "breaker": self.name,
            "action": action,
            "timestamp": datetime.now().isoformat(),
            "details": details
        }
        # alert_system.send("CIRCUIT_BREAKER", alert_payload)

    def _get_effective_timeout(self) -> int:
        """获取实际恢复超时时间（考虑交易时段）"""
        if not self.trading_hours or self._state != self.STATE_OPEN:
            return self.recovery_timeout

        now = datetime.now()
        for period in self.trading_hours.values():
            start = datetime.strptime(period['start'], '%H:%M').time()
            end = datetime.strptime(period['end'], '%H:%M').time()

            if now.time() < start:
                # 当前时间早于交易时段开始
                wait_seconds = (datetime.combine(now.date(), start) - now).seconds
                return min(wait_seconds, self.recovery_timeout)

            elif now.time() >= end:
                # 当前时间晚于交易时段结束
                next_start = datetime.strptime(
                    next(iter(self.trading_hours.values()))['start'],
                    '%H:%M'
                ).time()
                next_day = now.date() + timedelta(days=1)
                wait_seconds = (datetime.combine(next_day, next_start) - now).seconds
                return min(wait_seconds, self.recovery_timeout)

        return self.recovery_timeout

    def manual_reset(self):
        """手动重置熔断器（供运维使用）"""
        with self._lock:
            if self._state != self.STATE_CLOSED:
                self._reset()
                # 记录手动操作审计日志
                self._log_audit_event("MANUAL_RESET")

    def _log_audit_event(self, action: str):
        """记录审计日志"""
        audit_log = {
            "timestamp": datetime.now().isoformat(),
            "breaker": self.name,
            "action": action,
            "state": self._state,
            "operator": getattr(self, '_operator', 'system')
        }
        # audit_logger.log(audit_log)

class CircuitBreakerManager:
    """熔断器集中管理器"""

    def __init__(self):
        self.breakers: Dict[str, InstrumentedCircuitBreaker] = {}

    def get_breaker(self,
                   name: str,
                   **kwargs) -> InstrumentedCircuitBreaker:
        """获取或创建熔断器"""
        if name not in self.breakers:
            self.breakers[name] = InstrumentedCircuitBreaker(name, **kwargs)
        return self.breakers[name]

    def get_all_status(self) -> Dict:
        """获取所有熔断器状态"""
        return {
            name: {
                "state": breaker.state,
                "failure_count": breaker._failure_count,
                "last_failure": breaker._last_failure_time.isoformat()
                if breaker._last_failure_time else None
            }
            for name, breaker in self.breakers.items()
        }
