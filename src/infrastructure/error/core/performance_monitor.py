"""
performance_monitor 模块

提供 performance_monitor 相关功能和接口。
"""

import logging

import statistics
import threading
import time

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
"""
基础设施层 - 性能监控器

监控错误处理器的性能指标，包括响应时间、吞吐量、错误率等。
提供性能分析和优化建议。
"""

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """性能指标"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_response_time: float = 0.0
    response_times: List[float] = field(default_factory=list)
    error_counts: Dict[str, int] = field(default_factory=dict)
    throughput_history: List[float] = field(default_factory=list)

    # 计算属性
    @property
    def error_rate(self) -> float:
        """错误率"""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests

    @property
    def avg_response_time(self) -> float:
        """平均响应时间"""
        if self.total_requests == 0:
            return 0.0
        return self.total_response_time / self.total_requests

    @property
    def median_response_time(self) -> float:
        """中位数响应时间"""
        if not self.response_times:
            return 0.0
        return statistics.median(self.response_times)

    @property
    def p95_response_time(self) -> float:
        """95%响应时间"""
        if len(self.response_times) < 20:  # 需要足够的数据
            return self.avg_response_time
        return statistics.quantiles(self.response_times, n=20)[18]  # 95th percentile

    @property
    def throughput_per_second(self) -> float:
        """每秒吞吐量"""
        if not self.throughput_history:
            return 0.0
        return sum(self.throughput_history) / len(self.throughput_history)

    @property
    def min_response_time(self) -> float:
        """最小响应时间"""
        if not self.response_times:
            return 0.0
        return min(self.response_times)

    @property
    def max_response_time(self) -> float:
        """最大响应时间"""
        if not self.response_times:
            return 0.0
        return max(self.response_times)


@dataclass
class AlertConfig:
    """告警配置参数对象"""
    alert_type: str
    severity: str
    message: str
    metrics: Dict[str, Any]
    threshold: Any
    actual_value: Any


@dataclass
class PerformanceAlert:
    """性能告警"""
    alert_type: str
    severity: str
    message: str
    metrics: Dict[str, Any]
    timestamp: float
    threshold: Any
    actual_value: Any

    @classmethod
    def from_config(cls, config: AlertConfig, timestamp: float) -> 'PerformanceAlert':
        """从配置对象创建告警实例"""
        return cls(
            alert_type=config.alert_type,
            severity=config.severity,
            message=config.message,
            metrics=config.metrics,
            timestamp=timestamp,
            threshold=config.threshold,
            actual_value=config.actual_value
        )


class MetricsCollector:
    """指标收集器 - 专门负责收集和存储性能指标"""
    
    def __init__(self, max_history_size: int = 10000):
        self._metrics: Dict[str, PerformanceMetrics] = defaultdict(PerformanceMetrics)
        self._max_history_size = max_history_size
        self._lock = threading.Lock()
        self._throughput_window_start = time.time()
        self._throughput_window_count = 0

    def record_request(self, handler_name: str, response_time: float, success: bool, error_type: Optional[str] = None) -> None:
        """记录请求"""
        with self._lock:
            metrics = self._metrics[handler_name]

            metrics.total_requests += 1
            metrics.total_response_time += response_time

            if success:
                metrics.successful_requests += 1
            else:
                metrics.failed_requests += 1
                if error_type:
                    metrics.error_counts[error_type] = metrics.error_counts.get(error_type, 0) + 1

            # 记录响应时间（限制历史大小）
            metrics.response_times.append(response_time)
            if len(metrics.response_times) > self._max_history_size:
                metrics.response_times.pop(0)

            # 记录吞吐量历史
            self._throughput_window_count += 1
            current_time = time.time()
            if current_time - self._throughput_window_start >= 60:  # 每分钟更新一次吞吐量
                throughput = self._throughput_window_count / 60
                metrics.throughput_history.append(throughput)
                if len(metrics.throughput_history) > 60:  # 保留1小时的历史
                    metrics.throughput_history.pop(0)
                
                self._throughput_window_start = current_time
                self._throughput_window_count = 0

    def get_metrics(self, handler_name: str) -> PerformanceMetrics:
        """获取性能指标"""
        with self._lock:
            return self._metrics.get(handler_name, PerformanceMetrics())

    def get_all_metrics(self) -> Dict[str, PerformanceMetrics]:
        """获取所有性能指标"""
        with self._lock:
            return self._metrics.copy()

    def reset_metrics(self, handler_name: Optional[str] = None) -> None:
        """重置性能指标"""
        with self._lock:
            if handler_name:
                self._reset_handler_metrics(handler_name)
            else:
                self._reset_all_metrics()

    def _reset_handler_metrics(self, handler_name: str) -> None:
        """重置指定处理器的指标"""
        if handler_name in self._metrics:
            self._metrics[handler_name] = PerformanceMetrics()

    def _reset_all_metrics(self) -> None:
        """重置所有指标"""
        self._metrics.clear()


class AlertManager:
    """告警管理器 - 专门负责告警检测和处理"""
    
    def __init__(self, alert_check_interval: int = 60):
        self._alerts: List[PerformanceAlert] = []
        self._alert_callbacks: List[Callable[[PerformanceAlert], None]] = []
        self._lock = threading.Lock()
        self._alert_check_interval = alert_check_interval
        
        # 告警阈值配置
        self._alert_thresholds = {
            'error_rate_threshold': 0.1,  # 10%错误率
            'response_time_threshold': 5.0,  # 5秒响应时间
            'throughput_drop_threshold': 0.5  # 50%吞吐量下降
        }

    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]) -> None:
        """添加告警回调"""
        self._alert_callbacks.append(callback)

    def set_alert_threshold(self, threshold_name: str, value: Any) -> None:
        """设置告警阈值"""
        self._alert_thresholds[threshold_name] = value

    def start_alert_check_loop(self, metrics_collector: MetricsCollector) -> None:
        """启动告警检查循环"""
        def alert_loop():
            while True:
                try:
                    self._check_alerts(metrics_collector)
                except Exception as e:
                    logger.error(f"告警检查出错: {e}")
                time.sleep(self._alert_check_interval)
        
        alert_thread = threading.Thread(target=alert_loop, daemon=True)
        alert_thread.start()

    def _check_alerts(self, metrics_collector: MetricsCollector) -> None:
        """检查告警条件"""
        current_time = time.time()
        all_metrics = metrics_collector.get_all_metrics()
        
        for handler_name, metrics in all_metrics.items():
            self._check_handler_alerts(handler_name, metrics, current_time)

    def _check_handler_alerts(self, handler_name: str, metrics: PerformanceMetrics, current_time: float) -> None:
        """检查单个处理器的告警条件"""
        self._check_error_rate_alert(handler_name, metrics, current_time)
        self._check_response_time_alerts(handler_name, metrics, current_time)
        self._check_throughput_drop_alert(handler_name, metrics, current_time)

    def _check_error_rate_alert(self, handler_name: str, metrics: PerformanceMetrics, current_time: float) -> None:
        """检查错误率告警"""
        threshold = self._alert_thresholds['error_rate_threshold']
        if metrics.error_rate <= threshold:
            return
        
        config = AlertConfig(
            alert_type='high_error_rate',
            severity='high',
            message=f'{handler_name}错误率过高: {metrics.error_rate:.2%}',
            metrics={'error_rate': metrics.error_rate},
            threshold=threshold,
            actual_value=metrics.error_rate
        )
        self._trigger_alert(PerformanceAlert.from_config(config, current_time))

    def _check_response_time_alerts(self, handler_name: str, metrics: PerformanceMetrics, current_time: float) -> None:
        """检查响应时间告警"""
        threshold = self._alert_thresholds['response_time_threshold']
        
        # 检查平均响应时间
        if metrics.avg_response_time > threshold:
            config = AlertConfig(
                alert_type='high_response_time',
                severity='medium',
                message=f'{handler_name}平均响应时间过长: {metrics.avg_response_time:.2f}s',
                metrics={'avg_response_time': metrics.avg_response_time},
                threshold=threshold,
                actual_value=metrics.avg_response_time
            )
            self._trigger_alert(PerformanceAlert.from_config(config, current_time))

        # 检查95%响应时间
        if metrics.p95_response_time > threshold * 2:
            config = AlertConfig(
                alert_type='high_p95_response_time',
                severity='high',
                message=f'{handler_name}95%响应时间过长: {metrics.p95_response_time:.2f}s',
                metrics={'p95_response_time': metrics.p95_response_time},
                threshold=threshold * 2,
                actual_value=metrics.p95_response_time
            )
            self._trigger_alert(PerformanceAlert.from_config(config, current_time))

    def _check_throughput_drop_alert(self, handler_name: str, metrics: PerformanceMetrics, current_time: float) -> None:
        """检查吞吐量下降告警"""
        # 早期返回：需要至少2个历史数据点
        if len(metrics.throughput_history) < 2:
            return
            
        threshold = self._alert_thresholds['throughput_drop_threshold']
        current_throughput = metrics.throughput_per_second
        previous_throughput = metrics.throughput_history[-2]
        
        if previous_throughput > 0 and current_throughput / previous_throughput < threshold:
            config = AlertConfig(
                alert_type='throughput_drop',
                severity='medium',
                message=f'{handler_name}吞吐量下降: {current_throughput:.2f} req/s (之前: {previous_throughput:.2f})',
                metrics={'current_throughput': current_throughput, 'previous_throughput': previous_throughput},
                threshold=threshold,
                actual_value=current_throughput / previous_throughput
            )
            self._trigger_alert(PerformanceAlert.from_config(config, current_time))

    def _trigger_alert(self, alert: PerformanceAlert) -> None:
        """触发告警"""
        with self._lock:
            self._alerts.append(alert)
            
            # 限制告警历史大小
            if len(self._alerts) > 1000:
                self._alerts.pop(0)
                
            # 调用回调函数
            for callback in self._alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"告警回调执行失败: {e}")

    def get_alerts(self, limit: int = 50) -> List[PerformanceAlert]:
        """获取告警列表"""
        with self._lock:
            return self._alerts[-limit:] if limit > 0 else self._alerts.copy()


class PerformanceMonitor:
    """
    性能监控器 - 重构后的主类，使用组合模式

    协调指标收集器、告警管理器和报告生成器的工作。
    """

    def __init__(self, max_history_size: int = 10000, alert_check_interval: int = 60, test_mode: bool = False):
        self._test_mode = test_mode
        
        # 使用组合模式
        self._metrics_collector = MetricsCollector(max_history_size)
        self._alert_manager = AlertManager(alert_check_interval)
        
        # 启动告警检查
        if not test_mode:
            self._alert_manager.start_alert_check_loop(self._metrics_collector)

    def record_request(self, handler_name: str, response_time: float, success: bool, error_type: Optional[str] = None) -> None:
        """记录请求 - 委托给指标收集器"""
        self._metrics_collector.record_request(handler_name, response_time, success, error_type)

    def record_handler_performance(self, handler_name: str, response_time: float, success: bool, error_type: Optional[str] = None) -> None:
        """记录处理器性能（兼容旧接口）"""
        self.record_request(handler_name, response_time, success, error_type)

    def get_metrics(self, handler_name: str) -> PerformanceMetrics:
        """获取性能指标"""
        return self._metrics_collector.get_metrics(handler_name)

    def get_all_metrics(self) -> Dict[str, PerformanceMetrics]:
        """获取所有性能指标"""
        return self._metrics_collector.get_all_metrics()

    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]) -> None:
        """添加告警回调"""
        self._alert_manager.add_alert_callback(callback)

    def set_alert_threshold(self, threshold_name: str, value: Any) -> None:
        """设置告警阈值"""
        self._alert_manager.set_alert_threshold(threshold_name, value)

    def get_alerts(self, limit: int = 50) -> List[PerformanceAlert]:
        """获取告警列表"""
        return self._alert_manager.get_alerts(limit)

    def check_alerts(self) -> None:
        """手动触发告警检查"""
        self._alert_manager._check_alerts(self._metrics_collector)

    def reset_metrics(self, handler_name: Optional[str] = None) -> None:
        """重置性能指标"""
        self._metrics_collector.reset_metrics(handler_name)

    def get_performance_report(self, handler_name: Optional[str] = None) -> Dict[str, Any]:
        """生成性能报告"""
        if handler_name:
            return self._get_handler_report(handler_name)
        else:
            return self._get_summary_report()

    def _get_handler_report(self, handler_name: str) -> Dict[str, Any]:
        """获取单个处理器的性能报告"""
        metrics = self._metrics_collector.get_metrics(handler_name)
        if metrics.total_requests == 0:
            return {'error': f'处理器 {handler_name} 不存在或无数据'}

        return {
            'handler_name': handler_name,
            'total_requests': metrics.total_requests,
            'successful_requests': metrics.successful_requests,
            'failed_requests': metrics.failed_requests,
            'error_rate': metrics.error_rate,
            'avg_response_time': metrics.avg_response_time,
            'median_response_time': metrics.median_response_time,
            'p95_response_time': metrics.p95_response_time,
            'throughput_per_second': metrics.throughput_per_second,
            'top_errors': sorted(metrics.error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        }

    def _get_summary_report(self) -> Dict[str, Any]:
        """获取汇总性能报告"""
        all_metrics = self._metrics_collector.get_all_metrics()
        total_requests = sum(m.total_requests for m in all_metrics.values())
        total_errors = sum(m.failed_requests for m in all_metrics.values())

        return {
            'total_handlers': len(all_metrics),
            'total_requests': total_requests,
            'total_errors': total_errors,
            'overall_error_rate': total_errors / total_requests if total_requests > 0 else 0,
            'handler_performance': self._build_handler_performance_summary(),
            'active_alerts': self._count_active_alerts()
        }

    def _build_handler_performance_summary(self) -> Dict[str, Dict[str, Any]]:
        """构建处理器性能汇总"""
        all_metrics = self._metrics_collector.get_all_metrics()
        return {
            name: {
                'requests': metrics.total_requests,
                'error_rate': metrics.error_rate,
                'avg_response_time': metrics.avg_response_time,
                'throughput': metrics.throughput_per_second
            }
            for name, metrics in all_metrics.items()
        }

    def get_optimization_suggestions(self, handler_name: str) -> List[str]:
        """获取优化建议"""
        metrics = self.get_metrics(handler_name)
        suggestions = []

        if metrics.error_rate > 0.05:  # 5%错误率
            suggestions.append("建议检查错误处理逻辑，降低错误率")

        if metrics.avg_response_time > 2.0:  # 2秒平均响应时间
            suggestions.append("建议优化处理逻辑，降低响应时间")

        if metrics.p95_response_time > 5.0:  # 5秒95%响应时间
            suggestions.append("建议检查性能瓶颈，优化慢请求")

        if len(metrics.error_counts) > 10:  # 多种错误类型
            top_errors = sorted(metrics.error_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            suggestions.append(
                f"重点处理高频错误: {', '.join(f'{err}({count})' for err, count in top_errors)}")

        if not suggestions:
            suggestions.append("性能表现良好，继续保持")

        return suggestions

    def _count_active_alerts(self) -> int:
        """统计活跃告警数量"""
        current_time = time.time()
        alerts = self._alert_manager.get_alerts(limit=0)  # 获取所有告警
        return len([
            alert for alert in alerts 
            if current_time - alert.timestamp < 3600
        ])


# 全局性能监控器实例
_global_performance_monitor: Optional[PerformanceMonitor] = None


def get_global_performance_monitor() -> PerformanceMonitor:
    """获取全局性能监控器"""
    global _global_performance_monitor
    if _global_performance_monitor is None:
        _global_performance_monitor = PerformanceMonitor()
    return _global_performance_monitor


def record_handler_performance(handler_name: str, response_time: float, success: bool, error_type: Optional[str] = None) -> None:
    """便捷函数：记录处理器性能"""
    get_global_performance_monitor().record_request(handler_name, response_time, success, error_type)
