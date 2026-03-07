"""
策略执行监控器
提供策略执行状态监控、性能指标收集、监控规则管理等功能
符合量化交易系统合规要求，参考风险控制层 RealTimeRiskMonitor 设计
"""

import time
import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

# 使用统一日志系统
try:
    from src.infrastructure.logging.core.unified_logger import get_unified_logger
    logger = get_unified_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """执行状态"""
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"
    STARTING = "starting"
    STOPPING = "stopping"


@dataclass
class ExecutionMetrics:
    """执行指标"""
    strategy_id: str
    latency_ms: float = 0.0
    throughput: float = 0.0
    signal_count: int = 0
    position_count: int = 0
    error_count: int = 0
    last_update: float = field(default_factory=time.time)
    cpu_usage: float = 0.0
    memory_usage: float = 0.0


@dataclass
class MonitoringRule:
    """监控规则"""
    rule_id: str
    rule_type: str
    threshold: float
    duration: int
    severity: str
    enabled: bool = True
    created_at: float = field(default_factory=time.time)


class ExecutionMonitor:
    """
    策略执行监控器
    
    职责：
    1. 实时监控策略执行状态
    2. 监控执行性能指标
    3. 管理监控规则和阈值
    4. 触发监控告警
    
    参考架构：风险控制层 RealTimeRiskMonitor
    """
    
    # 默认监控配置
    DEFAULT_CHECK_INTERVAL = 5  # 默认检查间隔5秒
    DEFAULT_METRICS_WINDOW = 300  # 默认指标窗口300秒
    
    def __init__(self):
        # 监控的策略集合
        self._monitored_strategies: Dict[str, Dict[str, Any]] = {}
        
        # 执行指标缓存
        self._metrics_cache: Dict[str, ExecutionMetrics] = {}
        
        # 监控规则
        self._monitoring_rules: Dict[str, MonitoringRule] = {}
        
        # 回调函数
        self._status_callbacks: List[Callable[[str, ExecutionStatus], None]] = []
        self._metrics_callbacks: List[Callable[[ExecutionMetrics], None]] = []
        
        # 运行状态
        self._running = False
        self._check_task: Optional[asyncio.Task] = None
        
        # 初始化默认规则
        self._init_default_rules()
        
        logger.info("执行监控器初始化完成")
        
    def _init_default_rules(self):
        """初始化默认监控规则"""
        default_rules = [
            MonitoringRule(
                rule_id="high_latency",
                rule_type="latency",
                threshold=1000.0,  # 1000ms
                duration=60,
                severity="warning"
            ),
            MonitoringRule(
                rule_id="very_high_latency",
                rule_type="latency",
                threshold=5000.0,  # 5000ms
                duration=30,
                severity="critical"
            ),
            MonitoringRule(
                rule_id="high_error_rate",
                rule_type="error_rate",
                threshold=0.05,  # 5%
                duration=300,
                severity="warning"
            ),
            MonitoringRule(
                rule_id="signal_spike",
                rule_type="signal_count",
                threshold=100.0,
                duration=60,
                severity="warning"
            )
        ]
        
        for rule in default_rules:
            self._monitoring_rules[rule.rule_id] = rule
            
    def register_strategy(
        self, 
        strategy_id: str, 
        strategy_name: str,
        check_interval: Optional[int] = None,
        custom_rules: Optional[List[MonitoringRule]] = None
    ) -> None:
        """
        注册策略到监控器
        
        Args:
            strategy_id: 策略ID
            strategy_name: 策略名称
            check_interval: 检查间隔（秒）
            custom_rules: 自定义监控规则
        """
        if strategy_id in self._monitored_strategies:
            logger.warning(f"策略 {strategy_id} 已在监控列表中")
            return
            
        self._monitored_strategies[strategy_id] = {
            "strategy_id": strategy_id,
            "strategy_name": strategy_name,
            "status": ExecutionStatus.STOPPED,
            "check_interval": check_interval or self.DEFAULT_CHECK_INTERVAL,
            "custom_rules": custom_rules or [],
            "last_check": 0,
            "metrics_history": [],
            "alerts": [],
            "enabled": True
        }
        
        # 初始化指标缓存
        self._metrics_cache[strategy_id] = ExecutionMetrics(strategy_id=strategy_id)
        
        logger.info(f"策略 {strategy_id} ({strategy_name}) 已注册到执行监控器")
        
    def unregister_strategy(self, strategy_id: str) -> None:
        """取消策略监控"""
        if strategy_id not in self._monitored_strategies:
            logger.warning(f"策略 {strategy_id} 不在监控列表中")
            return
            
        del self._monitored_strategies[strategy_id]
        if strategy_id in self._metrics_cache:
            del self._metrics_cache[strategy_id]
            
        logger.info(f"策略 {strategy_id} 已从执行监控器移除")
        
    def update_strategy_status(
        self, 
        strategy_id: str, 
        status: ExecutionStatus,
        reason: Optional[str] = None
    ) -> None:
        """
        更新策略执行状态
        
        Args:
            strategy_id: 策略ID
            status: 执行状态
            reason: 状态变更原因
        """
        if strategy_id not in self._monitored_strategies:
            logger.warning(f"策略 {strategy_id} 未注册到监控器")
            return
            
        old_status = self._monitored_strategies[strategy_id]["status"]
        self._monitored_strategies[strategy_id]["status"] = status
        self._monitored_strategies[strategy_id]["last_status_change"] = time.time()
        
        logger.info(
            f"策略 {strategy_id} 状态变更: {old_status.value} -> {status.value}, "
            f"原因: {reason or 'N/A'}"
        )
        
        # 触发状态变更回调
        for callback in self._status_callbacks:
            try:
                callback(strategy_id, status)
            except Exception as e:
                logger.error(f"状态回调执行失败: {e}")
                
    def update_metrics(
        self, 
        strategy_id: str, 
        latency_ms: Optional[float] = None,
        throughput: Optional[float] = None,
        signal_count: Optional[int] = None,
        position_count: Optional[int] = None,
        error_count: Optional[int] = None,
        cpu_usage: Optional[float] = None,
        memory_usage: Optional[float] = None
    ) -> None:
        """
        更新执行指标
        
        Args:
            strategy_id: 策略ID
            latency_ms: 延迟（毫秒）
            throughput: 吞吐量
            signal_count: 信号数量
            position_count: 持仓数量
            error_count: 错误数量
            cpu_usage: CPU使用率
            memory_usage: 内存使用率
        """
        if strategy_id not in self._metrics_cache:
            logger.warning(f"策略 {strategy_id} 未注册到监控器")
            return
            
        metrics = self._metrics_cache[strategy_id]
        
        if latency_ms is not None:
            metrics.latency_ms = latency_ms
        if throughput is not None:
            metrics.throughput = throughput
        if signal_count is not None:
            metrics.signal_count = signal_count
        if position_count is not None:
            metrics.position_count = position_count
        if error_count is not None:
            metrics.error_count = error_count
        if cpu_usage is not None:
            metrics.cpu_usage = cpu_usage
        if memory_usage is not None:
            metrics.memory_usage = memory_usage
            
        metrics.last_update = time.time()
        
        # 保存到历史记录
        if strategy_id in self._monitored_strategies:
            history = self._monitored_strategies[strategy_id]["metrics_history"]
            history.append({
                "timestamp": metrics.last_update,
                "latency_ms": metrics.latency_ms,
                "throughput": metrics.throughput,
                "signal_count": metrics.signal_count,
                "position_count": metrics.position_count,
                "error_count": metrics.error_count
            })
            
            # 限制历史记录长度
            max_history = 1000
            if len(history) > max_history:
                self._monitored_strategies[strategy_id]["metrics_history"] = history[-max_history:]
        
        # 触发指标更新回调
        for callback in self._metrics_callbacks:
            try:
                callback(metrics)
            except Exception as e:
                logger.error(f"指标回调执行失败: {e}")
                
    def get_execution_status(self, strategy_id: str) -> Optional[ExecutionStatus]:
        """获取策略执行状态"""
        if strategy_id not in self._monitored_strategies:
            return None
        return self._monitored_strategies[strategy_id]["status"]
        
    def get_metrics(self, strategy_id: str) -> Optional[ExecutionMetrics]:
        """获取策略执行指标"""
        return self._metrics_cache.get(strategy_id)
        
    def get_all_metrics(self) -> Dict[str, ExecutionMetrics]:
        """获取所有策略的执行指标"""
        return self._metrics_cache.copy()
        
    def get_metrics_history(
        self, 
        strategy_id: str, 
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        获取指标历史记录
        
        Args:
            strategy_id: 策略ID
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            历史记录列表
        """
        if strategy_id not in self._monitored_strategies:
            return []
            
        history = self._monitored_strategies[strategy_id]["metrics_history"]
        
        if start_time:
            history = [h for h in history if h["timestamp"] >= start_time]
        if end_time:
            history = [h for h in history if h["timestamp"] <= end_time]
            
        return history
        
    def add_monitoring_rule(self, rule: MonitoringRule) -> None:
        """添加监控规则"""
        self._monitoring_rules[rule.rule_id] = rule
        logger.info(f"监控规则已添加: {rule.rule_id}")
        
    def remove_monitoring_rule(self, rule_id: str) -> None:
        """移除监控规则"""
        if rule_id in self._monitoring_rules:
            del self._monitoring_rules[rule_id]
            logger.info(f"监控规则已移除: {rule_id}")
            
    def enable_rule(self, rule_id: str) -> None:
        """启用监控规则"""
        if rule_id in self._monitoring_rules:
            self._monitoring_rules[rule_id].enabled = True
            logger.info(f"监控规则已启用: {rule_id}")
            
    def disable_rule(self, rule_id: str) -> None:
        """禁用监控规则"""
        if rule_id in self._monitoring_rules:
            self._monitoring_rules[rule_id].enabled = False
            logger.info(f"监控规则已禁用: {rule_id}")
            
    def add_status_callback(self, callback: Callable[[str, ExecutionStatus], None]) -> None:
        """添加状态变更回调"""
        self._status_callbacks.append(callback)
        
    def remove_status_callback(self, callback: Callable[[str, ExecutionStatus], None]) -> None:
        """移除状态变更回调"""
        if callback in self._status_callbacks:
            self._status_callbacks.remove(callback)
            
    def add_metrics_callback(self, callback: Callable[[ExecutionMetrics], None]) -> None:
        """添加指标更新回调"""
        self._metrics_callbacks.append(callback)
        
    def remove_metrics_callback(self, callback: Callable[[ExecutionMetrics], None]) -> None:
        """移除指标更新回调"""
        if callback in self._metrics_callbacks:
            self._metrics_callbacks.remove(callback)
            
    async def start_monitoring(self) -> None:
        """启动监控"""
        if self._running:
            logger.warning("监控器已在运行中")
            return
            
        self._running = True
        self._check_task = asyncio.create_task(self._monitoring_loop())
        logger.info("执行监控器已启动")
        
    async def stop_monitoring(self) -> None:
        """停止监控"""
        if not self._running:
            return
            
        self._running = False
        
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
                
        logger.info("执行监控器已停止")
        
    async def _monitoring_loop(self) -> None:
        """监控循环"""
        while self._running:
            try:
                await self._perform_monitoring_check()
                await asyncio.sleep(self.DEFAULT_CHECK_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"监控检查失败: {e}")
                await asyncio.sleep(self.DEFAULT_CHECK_INTERVAL)
                
    async def _perform_monitoring_check(self) -> None:
        """执行监控检查"""
        for strategy_id, config in self._monitored_strategies.items():
            if not config["enabled"]:
                continue
                
            try:
                # 检查策略状态
                await self._check_strategy_status(strategy_id)
                
                # 检查指标阈值
                await self._check_metrics_thresholds(strategy_id)
                
                # 更新最后检查时间
                config["last_check"] = time.time()
                
            except Exception as e:
                logger.error(f"策略 {strategy_id} 监控检查失败: {e}")
                
    async def _check_strategy_status(self, strategy_id: str) -> None:
        """检查策略状态"""
        config = self._monitored_strategies[strategy_id]
        status = config["status"]
        
        # 检查长时间处于starting状态
        if status == ExecutionStatus.STARTING:
            last_change = config.get("last_status_change", 0)
            if time.time() - last_change > 30:  # 30秒超时
                logger.warning(f"策略 {strategy_id} 启动超时")
                self.update_strategy_status(
                    strategy_id, 
                    ExecutionStatus.ERROR,
                    "启动超时"
                )
                
        # 检查长时间处于stopping状态
        elif status == ExecutionStatus.STOPPING:
            last_change = config.get("last_status_change", 0)
            if time.time() - last_change > 30:  # 30秒超时
                logger.warning(f"策略 {strategy_id} 停止超时")
                self.update_strategy_status(
                    strategy_id,
                    ExecutionStatus.ERROR,
                    "停止超时"
                )
                
    async def _check_metrics_thresholds(self, strategy_id: str) -> None:
        """检查指标阈值"""
        metrics = self._metrics_cache.get(strategy_id)
        if not metrics:
            return
            
        for rule_id, rule in self._monitoring_rules.items():
            if not rule.enabled:
                continue
                
            try:
                if rule.rule_type == "latency" and metrics.latency_ms > rule.threshold:
                    self._trigger_alert(strategy_id, rule, metrics.latency_ms)
                    
                elif rule.rule_type == "error_rate":
                    total_signals = max(metrics.signal_count, 1)
                    error_rate = metrics.error_count / total_signals
                    if error_rate > rule.threshold:
                        self._trigger_alert(strategy_id, rule, error_rate)
                        
                elif rule.rule_type == "signal_count" and metrics.signal_count > rule.threshold:
                    self._trigger_alert(strategy_id, rule, metrics.signal_count)
                    
            except Exception as e:
                logger.error(f"检查规则 {rule_id} 失败: {e}")
                
    def _trigger_alert(self, strategy_id: str, rule: MonitoringRule, value: float) -> None:
        """触发告警"""
        alert = {
            "strategy_id": strategy_id,
            "rule_id": rule.rule_id,
            "rule_type": rule.rule_type,
            "threshold": rule.threshold,
            "actual_value": value,
            "severity": rule.severity,
            "timestamp": time.time()
        }
        
        # 保存告警
        if strategy_id in self._monitored_strategies:
            self._monitored_strategies[strategy_id]["alerts"].append(alert)
            
        logger.warning(
            f"策略 {strategy_id} 触发告警: {rule.rule_id}, "
            f"阈值: {rule.threshold}, 实际值: {value}, 级别: {rule.severity}"
        )
        
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """获取监控摘要"""
        total_strategies = len(self._monitored_strategies)
        running_count = sum(
            1 for s in self._monitored_strategies.values()
            if s["status"] == ExecutionStatus.RUNNING
        )
        error_count = sum(
            1 for s in self._monitored_strategies.values()
            if s["status"] == ExecutionStatus.ERROR
        )
        
        return {
            "total_strategies": total_strategies,
            "running_count": running_count,
            "error_count": error_count,
            "monitoring_rules_count": len(self._monitoring_rules),
            "enabled_rules_count": sum(
                1 for r in self._monitoring_rules.values() if r.enabled
            ),
            "is_running": self._running,
            "last_check": max(
                (s["last_check"] for s in self._monitored_strategies.values()),
                default=0
            )
        }


# 全局执行监控器实例
_execution_monitor: Optional[ExecutionMonitor] = None


def get_execution_monitor() -> ExecutionMonitor:
    """获取全局执行监控器实例（单例模式）"""
    global _execution_monitor
    if _execution_monitor is None:
        _execution_monitor = ExecutionMonitor()
    return _execution_monitor
