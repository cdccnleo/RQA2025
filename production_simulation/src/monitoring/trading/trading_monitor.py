#!/usr/bin/env python3
"""
交易监控系统
监控交易策略性能、系统健康状态和风险指标
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import threading
import time
import psutil
from collections import deque

# 尝试导入GPUtil，如果失败则设置为None
try:
    import GPUtil
except ImportError:
    GPUtil = None

from src.infrastructure.logging.core.interfaces import get_logger

logger = logging.getLogger(__name__)


class AlertLevel(Enum):

    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    ERROR = "error"


class MonitorType(Enum):

    """监控类型"""
    PERFORMANCE = "performance"
    STRATEGY = "strategy"
    RISK = "risk"
    SYSTEM = "system"
    MARKET = "market"


@dataclass
class Alert:

    """告警信息"""
    alert_id: str
    monitor_type: MonitorType
    level: AlertLevel
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    resolved: bool = False
    resolved_time: Optional[datetime] = None


@dataclass
class PerformanceMetrics:

    """性能指标"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float]
    disk_usage: float
    network_io: Dict[str, float]
    response_time: float


@dataclass
class StrategyMetrics:

    """策略指标"""
    strategy_name: str
    timestamp: datetime
    total_signals: int
    profitable_signals: int
    win_rate: float
    total_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int


@dataclass
class RiskMetrics:

    """风险指标"""
    timestamp: datetime
    portfolio_value: float
    position_value: float
    total_exposure: float
    margin_usage: float
    var_95: float
    concentration_ratio: float
    leverage_ratio: float


class TradingMonitor:

    """交易监控系统"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        self.config = config or {}

        # 配置参数
        self.monitoring_interval = self.config.get('monitoring_interval', 60)  # 秒
        self.metrics_retention = self.config.get('metrics_retention', 3600)  # 1小时
        self.alert_thresholds = self.config.get('alert_thresholds', {})

        # 数据存储
        self.performance_history = deque(maxlen=1000)
        self.strategy_history: Dict[str, deque] = {}
        self.risk_history = deque(maxlen=1000)
        self.alerts: List[Alert] = []

        # 监控状态
        self.running = False
        self.monitor_thread = None
        self.alert_thread = None

        # 回调函数
        self.on_alert = None

        self.logger = get_logger(__name__)

    def start_monitoring(self):
        """启动监控"""
        if self.running:
            self.logger.warning("监控系统已在运行中")
            return

        self.running = True

        # 启动监控线程
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.alert_thread = threading.Thread(target=self._alert_processing_loop)

        self.monitor_thread.daemon = True
        self.alert_thread.daemon = True

        self.monitor_thread.start()
        self.alert_thread.start()

        self.logger.info("交易监控系统已启动")

    def stop_monitoring(self):
        """停止监控"""
        self.running = False

        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)

        if self.alert_thread and self.alert_thread.is_alive():
            self.alert_thread.join(timeout=5)

        self.logger.info("交易监控系统已停止")

    def record_performance_metrics(self):
        """记录性能指标"""
        try:
            # CPU使用率
            cpu_usage = psutil.cpu_percent(interval=1)

            # 内存使用率
            memory = psutil.virtual_memory()
            memory_usage = memory.percent

            # GPU使用率
            gpu_usage = None
            try:
                # 尝试导入GPUtil
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_usage = gpus[0].load * 100
            except ImportError:
                # 如果没有GPUtil模块，跳过GPU监控
                pass
            except BaseException:
                pass

            # 磁盘使用率
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent

            # 网络I / O
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }

            # 响应时间（模拟）
            import random
            response_time = random.uniform(0.1, 2.0)  # 毫秒

            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                gpu_usage=gpu_usage,
                disk_usage=disk_usage,
                network_io=network_io,
                response_time=response_time
            )

            self.performance_history.append(metrics)

        except Exception as e:
            self.logger.error(f"记录性能指标失败: {e}")

    def record_strategy_metrics(self, strategy_name: str, metrics: Dict[str, Any]):
        """记录策略指标"""
        try:
            strategy_metrics = StrategyMetrics(
                strategy_name=strategy_name,
                timestamp=datetime.now(),
                total_signals=metrics.get('total_signals', 0),
                profitable_signals=metrics.get('profitable_signals', 0),
                win_rate=metrics.get('win_rate', 0.0),
                total_pnl=metrics.get('total_pnl', 0.0),
                sharpe_ratio=metrics.get('sharpe_ratio', 0.0),
                max_drawdown=metrics.get('max_drawdown', 0.0),
                total_trades=metrics.get('total_trades', 0)
            )

            if strategy_name not in self.strategy_history:
                self.strategy_history[strategy_name] = deque(maxlen=1000)

            self.strategy_history[strategy_name].append(strategy_metrics)

        except Exception as e:
            self.logger.error(f"记录策略指标失败 {strategy_name}: {e}")

    def record_risk_metrics(self, metrics: Dict[str, Any]):
        """记录风险指标"""
        try:
            risk_metrics = RiskMetrics(
                timestamp=datetime.now(),
                portfolio_value=metrics.get('portfolio_value', 0.0),
                position_value=metrics.get('position_value', 0.0),
                total_exposure=metrics.get('total_exposure', 0.0),
                margin_usage=metrics.get('margin_usage', 0.0),
                var_95=metrics.get('var_95', 0.0),
                concentration_ratio=metrics.get('concentration_ratio', 0.0),
                leverage_ratio=metrics.get('leverage_ratio', 0.0)
            )

            self.risk_history.append(risk_metrics)

        except Exception as e:
            self.logger.error(f"记录风险指标失败: {e}")

    def check_alerts(self):
        """检查告警条件"""
        self._check_performance_alerts()
        self._check_strategy_alerts()
        self._check_risk_alerts()

    def _check_performance_alerts(self):
        """检查性能告警"""
        if len(self.performance_history) < 2:
            return

        latest = self.performance_history[-1]

        # CPU使用率告警
        if latest.cpu_usage > self.alert_thresholds.get('cpu_threshold', 90):
            self._create_alert(
                monitor_type=MonitorType.PERFORMANCE,
                level=AlertLevel.WARNING,
                message=f"CPU使用率过高: {latest.cpu_usage:.1f}%",
                details={'cpu_usage': latest.cpu_usage}
            )

        # 内存使用率告警
        if latest.memory_usage > self.alert_thresholds.get('memory_threshold', 90):
            self._create_alert(
                monitor_type=MonitorType.PERFORMANCE,
                level=AlertLevel.WARNING,
                message=f"内存使用率过高: {latest.memory_usage:.1f}%",
                details={'memory_usage': latest.memory_usage}
            )

        # 响应时间告警
        if latest.response_time > self.alert_thresholds.get('response_time_threshold', 1.0):
            self._create_alert(
                monitor_type=MonitorType.PERFORMANCE,
                level=AlertLevel.WARNING,
                message=f"系统响应时间过长: {latest.response_time:.2f}ms",
                details={'response_time': latest.response_time}
            )

    def _check_strategy_alerts(self):
        """检查策略告警"""
        for strategy_name, history in self.strategy_history.items():
            if len(history) < 2:
                continue

            latest = history[-1]

            # 胜率告警
            if latest.win_rate < self.alert_thresholds.get('min_win_rate', 0.4):
                self._create_alert(
                    monitor_type=MonitorType.STRATEGY,
                    level=AlertLevel.WARNING,
                    message=f"策略 {strategy_name} 胜率过低: {latest.win_rate:.2%}",
                    details={'strategy': strategy_name, 'win_rate': latest.win_rate}
                )

            # 最大回撤告警
            if latest.max_drawdown > self.alert_thresholds.get('max_drawdown_threshold', 0.2):
                self._create_alert(
                    monitor_type=MonitorType.STRATEGY,
                    level=AlertLevel.CRITICAL,
                    message=f"策略 {strategy_name} 最大回撤过大: {latest.max_drawdown:.2%}",
                    details={'strategy': strategy_name, 'max_drawdown': latest.max_drawdown}
                )

            # 夏普比率告警
            if latest.sharpe_ratio < self.alert_thresholds.get('min_sharpe_ratio', 0.5):
                self._create_alert(
                    monitor_type=MonitorType.STRATEGY,
                    level=AlertLevel.WARNING,
                    message=f"策略 {strategy_name} 夏普比率过低: {latest.sharpe_ratio:.2f}",
                    details={'strategy': strategy_name, 'sharpe_ratio': latest.sharpe_ratio}
                )

    def _check_risk_alerts(self):
        """检查风险告警"""
        if len(self.risk_history) < 1:
            return

        latest = self.risk_history[-1]

        # 保证金使用率告警
        if latest.margin_usage > self.alert_thresholds.get('margin_usage_threshold', 0.8):
            self._create_alert(
                monitor_type=MonitorType.RISK,
                level=AlertLevel.CRITICAL,
                message=f"保证金使用率过高: {latest.margin_usage:.2%}",
                details={'margin_usage': latest.margin_usage}
            )

        # 集中度告警
        if latest.concentration_ratio > self.alert_thresholds.get('concentration_threshold', 0.5):
            self._create_alert(
                monitor_type=MonitorType.RISK,
                level=AlertLevel.WARNING,
                message=f"投资组合集中度过高: {latest.concentration_ratio:.2%}",
                details={'concentration_ratio': latest.concentration_ratio}
            )

        # VaR告警
        if latest.var_95 > self.alert_thresholds.get('var_threshold', 0.1):
            self._create_alert(
                monitor_type=MonitorType.RISK,
                level=AlertLevel.WARNING,
                message=f"VaR(95%) 过高: {latest.var_95:.2%}",
                details={'var_95': latest.var_95}
            )

    def _create_alert(self, monitor_type: MonitorType, level: AlertLevel,


                      message: str, details: Dict[str, Any]):
        """创建告警"""
        alert = Alert(
            alert_id=f"ALERT_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            monitor_type=monitor_type,
            level=level,
            message=message,
            details=details,
            timestamp=datetime.now()
        )

        self.alerts.append(alert)

        # 调用告警回调
        if self.on_alert:
            self.on_alert(alert)

        self.logger.warning(f"告警创建: {message}")

    def _monitoring_loop(self):
        """监控循环"""
        while self.running:
            try:
                # 记录性能指标
                self.record_performance_metrics()

                # 检查告警
                self.check_alerts()

                # 清理过期数据
                self._cleanup_old_data()

                time.sleep(self.monitoring_interval)

            except Exception as e:
                self.logger.error(f"监控循环异常: {e}")
                time.sleep(5)

    def _alert_processing_loop(self):
        """告警处理循环"""
        while self.running:
            try:
                # 处理告警逻辑（可以扩展为自动处理）
                self._process_alerts()
                time.sleep(30)  # 每30秒处理一次

            except Exception as e:
                self.logger.error(f"告警处理循环异常: {e}")
                time.sleep(5)

    def _process_alerts(self):
        """处理告警"""
        # 这里可以实现告警的自动处理逻辑
        # 例如：发送邮件、短信、自动调整参数等

    def _cleanup_old_data(self):
        """清理过期数据"""
        cutoff_time = datetime.now() - timedelta(seconds=self.metrics_retention)

        # 清理告警
        self.alerts = [alert for alert in self.alerts
                       if alert.timestamp > cutoff_time]

    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if len(self.performance_history) == 0:
            return {}

        recent_metrics = list(self.performance_history)[-10:]  # 最近10个数据点

        return {
            'cpu_usage_avg': np.mean([m.cpu_usage for m in recent_metrics]),
            'memory_usage_avg': np.mean([m.memory_usage for m in recent_metrics]),
            'response_time_avg': np.mean([m.response_time for m in recent_metrics]),
            'data_points': len(self.performance_history)
        }

    def get_strategy_summary(self) -> Dict[str, Dict[str, Any]]:
        """获取策略摘要"""
        summary = {}

        for strategy_name, history in self.strategy_history.items():
            if len(history) == 0:
                continue

            recent_metrics = list(history)[-10:]

            summary[strategy_name] = {
                'win_rate_avg': np.mean([m.win_rate for m in recent_metrics]),
                'total_pnl': sum(m.total_pnl for m in recent_metrics),
                'sharpe_ratio_avg': np.mean([m.sharpe_ratio for m in recent_metrics]),
                'max_drawdown_max': max([m.max_drawdown for m in recent_metrics]),
                'data_points': len(history)
            }

        return summary

    def get_risk_summary(self) -> Dict[str, Any]:
        """获取风险摘要"""
        if len(self.risk_history) == 0:
            return {}

        recent_metrics = list(self.risk_history)[-10:]

        return {
            'portfolio_value_avg': np.mean([m.portfolio_value for m in recent_metrics]),
            'margin_usage_avg': np.mean([m.margin_usage for m in recent_metrics]),
            'concentration_ratio_avg': np.mean([m.concentration_ratio for m in recent_metrics]),
            'var_95_avg': np.mean([m.var_95 for m in recent_metrics]),
            'data_points': len(self.risk_history)
        }

    def get_alert_summary(self) -> Dict[str, int]:
        """获取告警摘要"""
        summary = {}
        for alert in self.alerts:
            key = f"{alert.monitor_type.value}_{alert.level.value}"
            summary[key] = summary.get(key, 0) + 1
        return summary

    def set_alert_callback(self, callback: Callable[[Alert], None]):
        """设置告警回调函数"""
        self.on_alert = callback

    def get_all_alerts(self, resolved: Optional[bool] = None) -> List[Alert]:
        """获取所有告警"""
        if resolved is None:
            return self.alerts.copy()
        else:
            return [alert for alert in self.alerts if alert.resolved == resolved]

    def resolve_alert(self, alert_id: str):
        """解决告警"""
        for alert in self.alerts:
            if alert.alert_id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_time = datetime.now()
                self.logger.info(f"告警已解决: {alert_id}")
                break
