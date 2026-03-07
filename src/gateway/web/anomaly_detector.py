"""
异常检测器模块
提供策略执行异常检测功能，包括延迟异常、错误率异常、信号频率异常等
符合量化交易系统合规要求
"""

import time
import asyncio
import logging
import statistics
from typing import Dict, Any, List, Optional, Callable, Deque
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

# 使用统一日志系统
try:
    from src.infrastructure.logging.core.unified_logger import get_unified_logger
    logger = get_unified_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """异常类型"""
    HIGH_LATENCY = "high_latency"
    VERY_HIGH_LATENCY = "very_high_latency"
    HIGH_ERROR_RATE = "high_error_rate"
    VERY_HIGH_ERROR_RATE = "very_high_error_rate"
    SIGNAL_SPIKE = "signal_spike"
    SIGNAL_DROP = "signal_drop"
    POSITION_ABNORMAL = "position_abnormal"


class AnomalySeverity(Enum):
    """异常严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AnomalyEvent:
    """异常事件"""
    strategy_id: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    message: str
    timestamp: float
    details: Dict[str, Any]
    value: float
    threshold: float


@dataclass
class AnomalyRule:
    """异常检测规则"""
    rule_id: str
    anomaly_type: AnomalyType
    threshold: float
    duration: int  # 持续时间（秒）
    severity: AnomalySeverity
    enabled: bool = True
    description: str = ""


class SlidingWindow:
    """滑动窗口统计"""
    
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.data: Deque[tuple] = deque(maxlen=window_size)  # (timestamp, value)
        
    def add(self, value: float) -> None:
        """添加数据点"""
        self.data.append((time.time(), value))
        
    def get_values(self, duration: Optional[int] = None) -> List[float]:
        """获取窗口内的值"""
        if duration is None:
            return [v for _, v in self.data]
        
        cutoff = time.time() - duration
        return [v for t, v in self.data if t >= cutoff]
        
    def get_average(self, duration: Optional[int] = None) -> float:
        """获取平均值"""
        values = self.get_values(duration)
        return statistics.mean(values) if values else 0.0
        
    def get_std_dev(self, duration: Optional[int] = None) -> float:
        """获取标准差"""
        values = self.get_values(duration)
        return statistics.stdev(values) if len(values) > 1 else 0.0
        
    def get_max(self, duration: Optional[int] = None) -> float:
        """获取最大值"""
        values = self.get_values(duration)
        return max(values) if values else 0.0
        
    def get_min(self, duration: Optional[int] = None) -> float:
        """获取最小值"""
        values = self.get_values(duration)
        return min(values) if values else 0.0


class AnomalyDetector:
    """
    异常检测器
    
    职责：
    1. 延迟异常检测
    2. 错误率异常检测
    3. 信号频率异常检测
    4. 仓位异常检测
    
    检测算法：
    - 滑动窗口统计
    - 标准差检测
    - 趋势分析
    - 阈值检测
    """
    
    # 默认异常检测规则
    DEFAULT_RULES = [
        AnomalyRule(
            rule_id="high_latency",
            anomaly_type=AnomalyType.HIGH_LATENCY,
            threshold=1000.0,  # 1000ms
            duration=60,
            severity=AnomalySeverity.MEDIUM,
            description="延迟超过1000ms"
        ),
        AnomalyRule(
            rule_id="very_high_latency",
            anomaly_type=AnomalyType.VERY_HIGH_LATENCY,
            threshold=5000.0,  # 5000ms
            duration=30,
            severity=AnomalySeverity.HIGH,
            description="延迟超过5000ms"
        ),
        AnomalyRule(
            rule_id="high_error_rate",
            anomaly_type=AnomalyType.HIGH_ERROR_RATE,
            threshold=0.05,  # 5%
            duration=300,
            severity=AnomalySeverity.MEDIUM,
            description="错误率超过5%"
        ),
        AnomalyRule(
            rule_id="very_high_error_rate",
            anomaly_type=AnomalyType.VERY_HIGH_ERROR_RATE,
            threshold=0.20,  # 20%
            duration=60,
            severity=AnomalySeverity.CRITICAL,
            description="错误率超过20%"
        ),
        AnomalyRule(
            rule_id="signal_spike",
            anomaly_type=AnomalyType.SIGNAL_SPIKE,
            threshold=100.0,  # 100个信号
            duration=60,
            severity=AnomalySeverity.MEDIUM,
            description="信号数量激增"
        ),
        AnomalyRule(
            rule_id="signal_drop",
            anomaly_type=AnomalyType.SIGNAL_DROP,
            threshold=-50.0,  # 下降50%
            duration=300,
            severity=AnomalySeverity.MEDIUM,
            description="信号数量骤降"
        ),
        AnomalyRule(
            rule_id="position_abnormal",
            anomaly_type=AnomalyType.POSITION_ABNORMAL,
            threshold=2.0,  # 2倍标准差
            duration=60,
            severity=AnomalySeverity.HIGH,
            description="仓位异常波动"
        )
    ]
    
    # 滑动窗口大小
    WINDOW_SIZE = 1000
    
    def __init__(self):
        # 异常检测规则
        self._rules: Dict[str, AnomalyRule] = {}
        
        # 策略数据窗口
        self._latency_windows: Dict[str, SlidingWindow] = {}
        self._error_rate_windows: Dict[str, SlidingWindow] = {}
        self._signal_count_windows: Dict[str, SlidingWindow] = {}
        self._position_windows: Dict[str, SlidingWindow] = {}
        
        # 历史信号计数（用于检测信号下降）
        self._last_signal_counts: Dict[str, int] = {}
        self._last_signal_time: Dict[str, float] = {}
        
        # 回调函数
        self._anomaly_callbacks: List[Callable[[AnomalyEvent], None]] = []
        
        # 异常事件历史
        self._anomaly_history: Dict[str, List[AnomalyEvent]] = {}
        
        # 初始化默认规则
        self._init_default_rules()
        
        logger.info("异常检测器初始化完成")
        
    def _init_default_rules(self):
        """初始化默认规则"""
        for rule in self.DEFAULT_RULES:
            self._rules[rule.rule_id] = rule
            
    def register_strategy(self, strategy_id: str) -> None:
        """注册策略到异常检测器"""
        if strategy_id in self._latency_windows:
            logger.warning(f"策略 {strategy_id} 已在异常检测器中")
            return
            
        # 初始化数据窗口
        self._latency_windows[strategy_id] = SlidingWindow(self.WINDOW_SIZE)
        self._error_rate_windows[strategy_id] = SlidingWindow(self.WINDOW_SIZE)
        self._signal_count_windows[strategy_id] = SlidingWindow(self.WINDOW_SIZE)
        self._position_windows[strategy_id] = SlidingWindow(self.WINDOW_SIZE)
        
        # 初始化历史记录
        self._last_signal_counts[strategy_id] = 0
        self._last_signal_time[strategy_id] = time.time()
        self._anomaly_history[strategy_id] = []
        
        logger.info(f"策略 {strategy_id} 已注册到异常检测器")
        
    def unregister_strategy(self, strategy_id: str) -> None:
        """取消策略异常检测"""
        if strategy_id not in self._latency_windows:
            return
            
        del self._latency_windows[strategy_id]
        del self._error_rate_windows[strategy_id]
        del self._signal_count_windows[strategy_id]
        del self._position_windows[strategy_id]
        del self._last_signal_counts[strategy_id]
        del self._last_signal_time[strategy_id]
        del self._anomaly_history[strategy_id]
        
        logger.info(f"策略 {strategy_id} 已从异常检测器移除")
        
    def add_latency_data(self, strategy_id: str, latency_ms: float) -> None:
        """添加延迟数据"""
        if strategy_id not in self._latency_windows:
            self.register_strategy(strategy_id)
            
        self._latency_windows[strategy_id].add(latency_ms)
        
        # 检测延迟异常
        self._detect_latency_anomaly(strategy_id, latency_ms)
        
    def add_error_data(self, strategy_id: str, error_count: int, total_count: int) -> None:
        """添加错误数据"""
        if strategy_id not in self._error_rate_windows:
            self.register_strategy(strategy_id)
            
        error_rate = error_count / max(total_count, 1)
        self._error_rate_windows[strategy_id].add(error_rate)
        
        # 检测错误率异常
        self._detect_error_rate_anomaly(strategy_id, error_rate)
        
    def add_signal_data(self, strategy_id: str, signal_count: int) -> None:
        """添加信号数据"""
        if strategy_id not in self._signal_count_windows:
            self.register_strategy(strategy_id)
            
        self._signal_count_windows[strategy_id].add(signal_count)
        
        # 检测信号数量异常
        self._detect_signal_anomaly(strategy_id, signal_count)
        
    def add_position_data(self, strategy_id: str, position_value: float) -> None:
        """添加仓位数据"""
        if strategy_id not in self._position_windows:
            self.register_strategy(strategy_id)
            
        self._position_windows[strategy_id].add(position_value)
        
        # 检测仓位异常
        self._detect_position_anomaly(strategy_id, position_value)
        
    def _detect_latency_anomaly(self, strategy_id: str, current_latency: float) -> None:
        """检测延迟异常"""
        # 检测高延迟
        rule = self._rules.get("high_latency")
        if rule and rule.enabled and current_latency > rule.threshold:
            self._trigger_anomaly(
                strategy_id,
                rule,
                current_latency,
                f"延迟异常: 当前{current_latency:.0f}ms，超过阈值{rule.threshold:.0f}ms"
            )
            
        # 检测极高延迟
        rule = self._rules.get("very_high_latency")
        if rule and rule.enabled and current_latency > rule.threshold:
            self._trigger_anomaly(
                strategy_id,
                rule,
                current_latency,
                f"严重延迟异常: 当前{current_latency:.0f}ms，超过阈值{rule.threshold:.0f}ms"
            )
            
    def _detect_error_rate_anomaly(self, strategy_id: str, current_error_rate: float) -> None:
        """检测错误率异常"""
        # 检测高错误率
        rule = self._rules.get("high_error_rate")
        if rule and rule.enabled and current_error_rate > rule.threshold:
            self._trigger_anomaly(
                strategy_id,
                rule,
                current_error_rate,
                f"错误率异常: 当前{current_error_rate:.2%}，超过阈值{rule.threshold:.2%}"
            )
            
        # 检测极高错误率
        rule = self._rules.get("very_high_error_rate")
        if rule and rule.enabled and current_error_rate > rule.threshold:
            self._trigger_anomaly(
                strategy_id,
                rule,
                current_error_rate,
                f"严重错误率异常: 当前{current_error_rate:.2%}，超过阈值{rule.threshold:.2%}"
            )
            
    def _detect_signal_anomaly(self, strategy_id: str, current_count: int) -> None:
        """检测信号数量异常"""
        last_count = self._last_signal_counts.get(strategy_id, 0)
        last_time = self._last_signal_time.get(strategy_id, time.time())
        
        # 计算变化率
        time_diff = time.time() - last_time
        if time_diff > 0 and last_count > 0:
            change_rate = (current_count - last_count) / last_count * 100
            
            # 检测信号激增
            rule = self._rules.get("signal_spike")
            if rule and rule.enabled and change_rate > rule.threshold:
                self._trigger_anomaly(
                    strategy_id,
                    rule,
                    change_rate,
                    f"信号数量激增: 增长{change_rate:.1f}%，超过阈值{rule.threshold:.1f}%"
                )
                
            # 检测信号骤降
            rule = self._rules.get("signal_drop")
            if rule and rule.enabled and change_rate < rule.threshold:
                self._trigger_anomaly(
                    strategy_id,
                    rule,
                    change_rate,
                    f"信号数量骤降: 下降{abs(change_rate):.1f}%，超过阈值{abs(rule.threshold):.1f}%"
                )
                
        # 更新历史记录
        self._last_signal_counts[strategy_id] = current_count
        self._last_signal_time[strategy_id] = time.time()
        
    def _detect_position_anomaly(self, strategy_id: str, current_position: float) -> None:
        """检测仓位异常（基于标准差）"""
        window = self._position_windows.get(strategy_id)
        if not window:
            return
            
        rule = self._rules.get("position_abnormal")
        if not rule or not rule.enabled:
            return
            
        # 计算统计值
        values = window.get_values(rule.duration)
        if len(values) < 10:  # 需要足够的数据点
            return
            
        mean = statistics.mean(values)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0
        
        if std_dev == 0:
            return
            
        # 计算Z-score
        z_score = abs(current_position - mean) / std_dev
        
        if z_score > rule.threshold:
            self._trigger_anomaly(
                strategy_id,
                rule,
                z_score,
                f"仓位异常波动: Z-score={z_score:.2f}，超过阈值{rule.threshold:.2f}"
            )
            
    def _trigger_anomaly(
        self, 
        strategy_id: str, 
        rule: AnomalyRule, 
        value: float,
        message: str
    ) -> None:
        """触发异常事件"""
        # 检查是否已经存在相同类型的未恢复异常
        recent_anomalies = [
            a for a in self._anomaly_history.get(strategy_id, [])
            if a.anomaly_type == rule.anomaly_type
            and time.time() - a.timestamp < 300  # 5分钟内不重复触发
        ]
        
        if recent_anomalies:
            return
            
        event = AnomalyEvent(
            strategy_id=strategy_id,
            anomaly_type=rule.anomaly_type,
            severity=rule.severity,
            message=message,
            timestamp=time.time(),
            details={
                "rule_id": rule.rule_id,
                "threshold": rule.threshold,
                "duration": rule.duration,
                "description": rule.description
            },
            value=value,
            threshold=rule.threshold
        )
        
        # 保存到历史
        if strategy_id not in self._anomaly_history:
            self._anomaly_history[strategy_id] = []
        self._anomaly_history[strategy_id].append(event)
        
        # 限制历史记录长度
        max_history = 100
        if len(self._anomaly_history[strategy_id]) > max_history:
            self._anomaly_history[strategy_id] = self._anomaly_history[strategy_id][-max_history:]
        
        logger.warning(f"异常检测: {message}")
        
        # 触发回调
        for callback in self._anomaly_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"异常回调执行失败: {e}")
                
    def add_anomaly_callback(self, callback: Callable[[AnomalyEvent], None]) -> None:
        """添加异常事件回调"""
        self._anomaly_callbacks.append(callback)
        
    def remove_anomaly_callback(self, callback: Callable[[AnomalyEvent], None]) -> None:
        """移除异常事件回调"""
        if callback in self._anomaly_callbacks:
            self._anomaly_callbacks.remove(callback)
            
    def add_rule(self, rule: AnomalyRule) -> None:
        """添加异常检测规则"""
        self._rules[rule.rule_id] = rule
        logger.info(f"异常检测规则已添加: {rule.rule_id}")
        
    def remove_rule(self, rule_id: str) -> None:
        """移除异常检测规则"""
        if rule_id in self._rules:
            del self._rules[rule_id]
            logger.info(f"异常检测规则已移除: {rule_id}")
            
    def enable_rule(self, rule_id: str) -> None:
        """启用规则"""
        if rule_id in self._rules:
            self._rules[rule_id].enabled = True
            
    def disable_rule(self, rule_id: str) -> None:
        """禁用规则"""
        if rule_id in self._rules:
            self._rules[rule_id].enabled = False
            
    def get_anomaly_history(
        self, 
        strategy_id: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        severity: Optional[AnomalySeverity] = None
    ) -> List[AnomalyEvent]:
        """获取异常历史记录"""
        if strategy_id not in self._anomaly_history:
            return []
            
        events = self._anomaly_history[strategy_id]
        
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        if severity:
            events = [e for e in events if e.severity == severity]
            
        return events
        
    def get_statistics(self, strategy_id: str) -> Dict[str, Any]:
        """获取异常统计信息"""
        if strategy_id not in self._anomaly_history:
            return {}
            
        events = self._anomaly_history[strategy_id]
        
        # 按严重程度统计
        severity_counts = {}
        for severity in AnomalySeverity:
            severity_counts[severity.value] = sum(
                1 for e in events if e.severity == severity
            )
            
        # 按类型统计
        type_counts = {}
        for anomaly_type in AnomalyType:
            type_counts[anomaly_type.value] = sum(
                1 for e in events if e.anomaly_type == anomaly_type
            )
            
        return {
            "total_anomalies": len(events),
            "severity_counts": severity_counts,
            "type_counts": type_counts,
            "last_anomaly_time": events[-1].timestamp if events else None
        }


# 全局异常检测器实例
_anomaly_detector: Optional[AnomalyDetector] = None


def get_anomaly_detector() -> AnomalyDetector:
    """获取全局异常检测器实例（单例模式）"""
    global _anomaly_detector
    if _anomaly_detector is None:
        _anomaly_detector = AnomalyDetector()
    return _anomaly_detector
