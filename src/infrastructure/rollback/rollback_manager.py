"""
回滚管理器模块

负责自动回滚的触发、执行和验证
"""

from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import threading
import logging


class RollbackStrategy(Enum):
    """回滚策略"""
    IMMEDIATE = "immediate"      # 立即回滚
    GRADUAL = "gradual"          # 渐进回滚
    EMERGENCY_STOP = "emergency_stop"  # 紧急停止


class RollbackTriggerType(Enum):
    """回滚触发类型"""
    METRIC_THRESHOLD = "metric_threshold"
    ERROR_RATE = "error_rate"
    LATENCY_SPIKE = "latency_spike"
    MANUAL = "manual"
    SCHEDULED = "scheduled"


@dataclass
class RollbackTrigger:
    """回滚触发器"""
    trigger_type: RollbackTriggerType
    metric: str
    threshold: float
    operator: str  # greater_than, less_than, increase, decrease
    duration_minutes: int = 5
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trigger_type": self.trigger_type.value,
            "metric": self.metric,
            "threshold": self.threshold,
            "operator": self.operator,
            "duration_minutes": self.duration_minutes,
            "enabled": self.enabled
        }


@dataclass
class RollbackRecord:
    """回滚记录"""
    rollback_id: str
    deployment_id: str
    strategy: RollbackStrategy
    trigger: RollbackTrigger
    start_time: datetime
    end_time: Optional[datetime] = None
    success: bool = False
    error_message: Optional[str] = None
    from_version: Optional[str] = None
    to_version: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rollback_id": self.rollback_id,
            "deployment_id": self.deployment_id,
            "strategy": self.strategy.value,
            "trigger": self.trigger.to_dict(),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "success": self.success,
            "error_message": self.error_message,
            "from_version": self.from_version,
            "to_version": self.to_version
        }


class RollbackManager:
    """
    回滚管理器
    
    功能：
    - 回滚触发器管理
    - 自动回滚执行
    - 回滚策略选择
    - 回滚历史记录
    """
    
    def __init__(self):
        self.logger = logging.getLogger("rollback.manager")
        self._triggers: List[RollbackTrigger] = []
        self._records: List[RollbackRecord] = []
        self._handlers: Dict[str, Callable[[str, str], bool]] = {}  # deployment_id -> handler
        self._lock = threading.Lock()
        self._metric_history: Dict[str, List[tuple]] = {}  # metric -> [(timestamp, value)]
        self._active_rollbacks: Dict[str, RollbackRecord] = {}  # deployment_id -> record
    
    def add_trigger(self, trigger: RollbackTrigger) -> None:
        """添加回滚触发器"""
        with self._lock:
            self._triggers.append(trigger)
        self.logger.info(f"添加回滚触发器: {trigger.metric} {trigger.operator} {trigger.threshold}")
    
    def remove_trigger(self, metric: str) -> bool:
        """移除回滚触发器"""
        with self._lock:
            original_len = len(self._triggers)
            self._triggers = [t for t in self._triggers if t.metric != metric]
            return len(self._triggers) < original_len
    
    def register_deployment_handler(self, deployment_id: str, handler: Callable[[str, str], bool]) -> None:
        """
        注册部署回滚处理器
        
        Args:
            deployment_id: 部署ID
            handler: 回滚处理函数，参数为(deployment_id, target_version)，返回是否成功
        """
        self._handlers[deployment_id] = handler
    
    def evaluate_metric(self, deployment_id: str, metric_name: str, value: float) -> Optional[RollbackRecord]:
        """
        评估指标是否触发回滚
        
        Args:
            deployment_id: 部署ID
            metric_name: 指标名称
            value: 指标值
            
        Returns:
            如果触发回滚，返回回滚记录，否则返回None
        """
        # 保存历史数据
        if metric_name not in self._metric_history:
            self._metric_history[metric_name] = []
        self._metric_history[metric_name].append((datetime.now(), value))
        
        # 限制历史数据量
        if len(self._metric_history[metric_name]) > 1000:
            self._metric_history[metric_name] = self._metric_history[metric_name][-500:]
        
        with self._lock:
            for trigger in self._triggers:
                if not trigger.enabled:
                    continue
                
                if trigger.metric != metric_name:
                    continue
                
                # 检查触发条件
                if self._check_trigger(trigger, value, metric_name):
                    self.logger.warning(
                        f"触发回滚条件: {metric_name}={value:.4f} {trigger.operator} {trigger.threshold}"
                    )
                    
                    # 执行回滚
                    record = self._execute_rollback(deployment_id, trigger)
                    return record
        
        return None
    
    def _check_trigger(self, trigger: RollbackTrigger, value: float, metric_name: str) -> bool:
        """检查触发条件"""
        op = trigger.operator
        
        if op == "greater_than":
            return value > trigger.threshold
        elif op == "less_than":
            return value < trigger.threshold
        elif op == "increase":
            # 检查相比历史值是否增加超过阈值
            history = self._metric_history.get(metric_name, [])
            if len(history) >= 2:
                prev_value = history[-2][1]
                increase_pct = (value - prev_value) / prev_value if prev_value != 0 else 0
                return increase_pct > trigger.threshold
            return False
        elif op == "decrease":
            # 检查相比历史值是否减少超过阈值
            history = self._metric_history.get(metric_name, [])
            if len(history) >= 2:
                prev_value = history[-2][1]
                decrease_pct = (prev_value - value) / prev_value if prev_value != 0 else 0
                return decrease_pct > trigger.threshold
            return False
        
        return False
    
    def _execute_rollback(self, deployment_id: str, trigger: RollbackTrigger) -> RollbackRecord:
        """执行回滚"""
        import uuid
        
        rollback_id = f"rollback_{uuid.uuid4().hex[:8]}"
        
        # 确定回滚策略
        strategy = self._determine_strategy(trigger)
        
        record = RollbackRecord(
            rollback_id=rollback_id,
            deployment_id=deployment_id,
            strategy=strategy,
            trigger=trigger,
            start_time=datetime.now()
        )
        
        self._active_rollbacks[deployment_id] = record
        
        try:
            self.logger.info(f"开始执行回滚: {rollback_id}, 策略: {strategy.value}")
            
            # 获取处理器
            handler = self._handlers.get(deployment_id)
            if handler is None:
                raise ValueError(f"未找到部署 {deployment_id} 的回滚处理器")
            
            # 获取目标版本（上一个稳定版本）
            target_version = self._get_previous_version(deployment_id)
            record.to_version = target_version
            
            # 执行回滚
            if strategy == RollbackStrategy.IMMEDIATE:
                success = self._rollback_immediate(deployment_id, target_version, handler)
            elif strategy == RollbackStrategy.GRADUAL:
                success = self._rollback_gradual(deployment_id, target_version, handler)
            elif strategy == RollbackStrategy.EMERGENCY_STOP:
                success = self._rollback_emergency(deployment_id, handler)
            else:
                success = False
            
            record.success = success
            record.end_time = datetime.now()
            
            if success:
                self.logger.info(f"回滚成功: {rollback_id}")
            else:
                self.logger.error(f"回滚失败: {rollback_id}")
            
        except Exception as e:
            record.success = False
            record.error_message = str(e)
            record.end_time = datetime.now()
            self.logger.error(f"回滚异常: {e}")
        
        finally:
            # 保存记录
            with self._lock:
                self._records.append(record)
            
            # 清理活动回滚
            if deployment_id in self._active_rollbacks:
                del self._active_rollbacks[deployment_id]
        
        return record
    
    def _determine_strategy(self, trigger: RollbackTrigger) -> RollbackStrategy:
        """确定回滚策略"""
        # 根据触发器类型和严重程度选择策略
        if trigger.trigger_type == RollbackTriggerType.ERROR_RATE:
            if trigger.threshold > 0.1:  # 错误率超过10%
                return RollbackStrategy.IMMEDIATE
            else:
                return RollbackStrategy.GRADUAL
        elif trigger.trigger_type == RollbackTriggerType.LATENCY_SPIKE:
            return RollbackStrategy.GRADUAL
        elif trigger.trigger_type == RollbackTriggerType.METRIC_THRESHOLD:
            if trigger.metric in ["accuracy", "sharpe_ratio"]:
                return RollbackStrategy.IMMEDIATE
        
        return RollbackStrategy.IMMEDIATE
    
    def _rollback_immediate(self, deployment_id: str, target_version: str, handler: Callable) -> bool:
        """立即回滚"""
        self.logger.info(f"执行立即回滚到版本: {target_version}")
        return handler(deployment_id, target_version)
    
    def _rollback_gradual(self, deployment_id: str, target_version: str, handler: Callable) -> bool:
        """渐进回滚"""
        self.logger.info(f"执行渐进回滚到版本: {target_version}")
        
        # 模拟渐进式流量切换
        traffic_steps = [75, 50, 25, 0]
        
        for percentage in traffic_steps:
            self.logger.info(f"切换 {percentage}% 流量到新版本")
            # 实际应调整流量比例
            import time
            time.sleep(1)
        
        # 最后执行完全回滚
        return handler(deployment_id, target_version)
    
    def _rollback_emergency(self, deployment_id: str, handler: Callable) -> bool:
        """紧急停止"""
        self.logger.info("执行紧急停止")
        
        # 立即停止所有流量
        # 实际应切断流量或切换到备用系统
        
        return True
    
    def _get_previous_version(self, deployment_id: str) -> Optional[str]:
        """获取上一个稳定版本"""
        # 从回滚记录中找到上一个成功的版本
        for record in reversed(self._records):
            if record.deployment_id == deployment_id and record.success:
                return record.to_version
        
        return None
    
    def manual_rollback(self, deployment_id: str, target_version: Optional[str] = None) -> RollbackRecord:
        """手动触发回滚"""
        trigger = RollbackTrigger(
            trigger_type=RollbackTriggerType.MANUAL,
            metric="manual",
            threshold=0,
            operator="manual",
            enabled=True
        )
        
        record = self._execute_rollback(deployment_id, trigger)
        
        # 如果指定了目标版本，覆盖自动检测的版本
        if target_version:
            record.to_version = target_version
        
        return record
    
    def get_records(
        self,
        deployment_id: Optional[str] = None,
        success: Optional[bool] = None
    ) -> List[RollbackRecord]:
        """获取回滚记录"""
        with self._lock:
            records = self._records.copy()
        
        if deployment_id:
            records = [r for r in records if r.deployment_id == deployment_id]
        if success is not None:
            records = [r for r in records if r.success == success]
        
        return records
    
    def get_active_rollbacks(self) -> Dict[str, RollbackRecord]:
        """获取活动回滚"""
        return self._active_rollbacks.copy()
    
    def export_records(self) -> List[Dict[str, Any]]:
        """导出回滚记录"""
        return [record.to_dict() for record in self._records]


# 默认回滚触发器配置
DEFAULT_ROLLBACK_TRIGGERS = [
    RollbackTrigger(
        trigger_type=RollbackTriggerType.METRIC_THRESHOLD,
        metric="accuracy",
        threshold=0.1,
        operator="decrease",
        duration_minutes=5
    ),
    RollbackTrigger(
        trigger_type=RollbackTriggerType.METRIC_THRESHOLD,
        metric="max_drawdown",
        threshold=0.15,
        operator="greater_than",
        duration_minutes=5
    ),
    RollbackTrigger(
        trigger_type=RollbackTriggerType.ERROR_RATE,
        metric="error_rate",
        threshold=0.05,
        operator="greater_than",
        duration_minutes=3
    ),
    RollbackTrigger(
        trigger_type=RollbackTriggerType.LATENCY_SPIKE,
        metric="latency_p95",
        threshold=300,
        operator="greater_than",
        duration_minutes=5
    ),
    RollbackTrigger(
        trigger_type=RollbackTriggerType.METRIC_THRESHOLD,
        metric="drift_score",
        threshold=0.5,
        operator="greater_than",
        duration_minutes=15
    )
]


# 全局回滚管理器实例
_global_rollback_manager: Optional[RollbackManager] = None


def get_rollback_manager() -> RollbackManager:
    """获取全局回滚管理器"""
    global _global_rollback_manager
    if _global_rollback_manager is None:
        _global_rollback_manager = RollbackManager()
        # 添加默认触发器
        for trigger in DEFAULT_ROLLBACK_TRIGGERS:
            _global_rollback_manager.add_trigger(trigger)
    return _global_rollback_manager


def reset_rollback_manager() -> None:
    """重置全局回滚管理器"""
    global _global_rollback_manager
    _global_rollback_manager = None
