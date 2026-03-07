
import threading
import time

from dataclasses import dataclass, asdict
from enum import Enum
from infrastructure.logging.core.interfaces import get_logger_pool
from typing import Dict, Any, Optional, List

# 导入新的组件
try:
    from ..components.logger_pool_stats_collector import LoggerPoolStatsCollector
    from ..components.logger_pool_alert_manager import LoggerPoolAlertManager
    from ..components.logger_pool_metrics_exporter import LoggerPoolMetricsExporter
    from ..components.logger_pool_monitoring_loop import LoggerPoolMonitoringLoop
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入Logger池监控组件: {e}")
    COMPONENTS_AVAILABLE = False
# -*- coding: utf-8 -*-
"""
RQA2025 基础设施层 - Logger池监控组件

提供Logger对象池的实时监控、性能指标收集和告警功能。
集成到统一监控系统中，支持Prometheus指标导出。
"""


class LoggerPoolMetric(Enum):
    """Logger池监控指标类型"""

    POOL_SIZE = "pool_size"
    MAX_SIZE = "max_size"
    CREATED_COUNT = "created_count"
    HIT_COUNT = "hit_count"
    HIT_RATE = "hit_rate"
    EVICTION_COUNT = "eviction_count"
    MEMORY_USAGE = "memory_usage"
    AVG_ACCESS_TIME = "avg_access_time"


@dataclass
class LoggerPoolStats:
    """Logger池统计数据"""

    pool_size: int
    max_size: int
    created_count: int
    hit_count: int
    hit_rate: float
    logger_count: int
    total_access_count: int
    avg_access_time: float
    memory_usage_mb: float
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)


class LoggerPoolMonitor:
    """
    Logger池监控器

    提供Logger对象池的全面监控功能，包括：
    - 实时性能指标收集
    - 历史数据统计
    - 异常检测和告警
    - Prometheus指标导出
    """

    def __init__(self, pool_name: str = "default", collection_interval: int = 60):
        """
        初始化Logger池监控器

        Args:
            pool_name: 池名称
            collection_interval: 收集间隔(秒)
        """
        self.pool_name = pool_name
        self.collection_interval = collection_interval

        # 告警配置
        self.alert_thresholds = {
            'hit_rate_low': 0.8,      # 命中率低于80%告警
            'pool_usage_high': 0.9,   # 池使用率高于90%告警
            'memory_high': 100.0,     # 内存使用高于100MB告警
        }

        # 监控线程
        self.monitoring_thread: Optional[threading.Thread] = None
        self.running = False
        self._lock = threading.RLock()

        # 初始化子组件
        self._init_components()

        # 为了向后兼容，保留原有属性
        self.current_stats: Optional[LoggerPoolStats] = None
        self.history_stats: List[LoggerPoolStats] = []
        self.max_history_size = 1000
        self.access_times: List[float] = []
        self.max_access_times_size = 1000
        
        # 获取Logger池实例
        self.logger_pool = get_logger_pool()

        # 初始化监控
        self._collect_initial_stats()
    
    def _init_components(self) -> None:
        """初始化子组件"""
        if COMPONENTS_AVAILABLE:
            self._stats_collector = LoggerPoolStatsCollector(self.pool_name)
            self._alert_manager = LoggerPoolAlertManager(self.pool_name, self.alert_thresholds)
            self._metrics_exporter = LoggerPoolMetricsExporter(self.pool_name)
            self._monitoring_loop_manager = LoggerPoolMonitoringLoop(self.pool_name, self.collection_interval)
        else:
            self._stats_collector = None
            self._alert_manager = None
            self._metrics_exporter = None
            self._monitoring_loop_manager = None

    def start_monitoring(self):
        """启动监控"""
        with self._lock:
            if self.running:
                return

            self.running = True
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                name=f"LoggerPoolMonitor-{self.pool_name}",
                daemon=True
            )
            self.monitoring_thread.start()

    def stop_monitoring(self):
        """停止监控"""
        with self._lock:
            self.running = False
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=5)

    def _monitoring_loop(self):
        """监控循环"""
        while self.running:
            try:
                # 使用新的组件进行监控
                if hasattr(self, '_monitoring_loop_manager') and self._monitoring_loop_manager and COMPONENTS_AVAILABLE:
                    # 使用新的监控循环管理器
                    stats = self._monitoring_loop_manager.collect_current_stats()
                    
                    # 更新当前统计（向后兼容）
                    with self._lock:
                        self.current_stats = stats
                        self.history_stats = self._monitoring_loop_manager.get_history_stats()
                        self.access_times = self._monitoring_loop_manager.get_current_access_times()
                    
                    # 检查告警
                    if self._alert_manager and stats:
                        self._alert_manager.check_alerts(stats)
                        
                elif self._stats_collector:
                    # 使用原有的组件方式
                    stats = self._stats_collector.collect_current_stats()
                    
                    # 更新当前统计（向后兼容）
                    with self._lock:
                        self.current_stats = stats
                        self.history_stats = self._stats_collector.get_history_stats()
                        self.access_times = self._stats_collector.get_current_access_times()
                    
                    # 检查告警
                    if self._alert_manager and stats:
                        self._alert_manager.check_alerts(stats)
                else:
                    # 回退到原有方法
                    self._collect_stats()
                    self._check_alerts()
                    
                time.sleep(self.collection_interval)
            except Exception as e:
                print(f"Logger池监控错误: {e}")
                time.sleep(self.collection_interval)

    def _collect_initial_stats(self):
        """收集初始统计信息"""
        try:
            # 使用新的监控循环管理器
            if hasattr(self, '_monitoring_loop_manager') and self._monitoring_loop_manager:
                self._monitoring_loop_manager.collect_initial_stats()
                # 同步数据到兼容属性
                with self._lock:
                    self.current_stats = self._monitoring_loop_manager.get_current_stats()
                    self.history_stats = self._monitoring_loop_manager.get_history_stats()
                    self.access_times = self._monitoring_loop_manager.get_current_access_times()
            else:
                # 回退到原有方法
                self._collect_stats()
        except Exception as e:
            print(f"收集初始Logger池统计失败: {e}")
        finally:
            # 如果仍未收集到任何统计信息，则尝试使用回退逻辑
            if not self.current_stats:
                try:
                    self._collect_stats()
                except Exception:
                    # 保持静默，避免初始化失败
                    pass

            # 确保历史列表至少包含当前统计，避免后续测试依赖失败
            if self.current_stats and not self.history_stats:
                self.history_stats.append(self.current_stats)
 
    def _collect_stats(self):
        """收集当前统计信息"""
        try:
            if not self.logger_pool or not hasattr(self.logger_pool, "get_stats"):
                pool_stats = {}
            else:
                pool_stats = self.logger_pool.get_stats() or {}
            usage_stats = pool_stats.get('usage_stats', {})

            # 计算平均访问时间
            avg_access_time = 0.0
            if self.access_times:
                avg_access_time = sum(self.access_times) / len(self.access_times)

            # 估算内存使用 (简化计算)
            memory_usage = self._estimate_memory_usage(pool_stats)

            # 创建统计对象（填充缺省值确保数据完整）
            stats = LoggerPoolStats(
                pool_size=pool_stats.get('pool_size', 0) or 0,
                max_size=pool_stats.get('max_size', 0) or 0,
                created_count=pool_stats.get('created_count', 0) or 0,
                hit_count=pool_stats.get('hit_count', 0) or 0,
                hit_rate=pool_stats.get('hit_rate', 0.0) or 0.0,
                logger_count=len(pool_stats.get('loggers', [])),
                total_access_count=sum(
                    stat.get('access_count', 0)
                    for stat in usage_stats.values()
                ),
                avg_access_time=avg_access_time,
                memory_usage_mb=memory_usage,
                timestamp=time.time()
            )

            with self._lock:
                self.current_stats = stats
                self.history_stats.append(stats)

                # 限制历史数据大小
                if len(self.history_stats) > self.max_history_size:
                    self.history_stats.pop(0)

        except Exception as e:
            print(f"收集Logger池统计失败: {e}")

    def _estimate_memory_usage(self, pool_stats: Dict[str, Any]) -> float:
        """
        估算内存使用量 (MB)

        这是一个简化的估算，实际内存使用可能更高
        """
        try:
            pool_size = pool_stats.get('pool_size', 0)

            # 每个Logger实例的估算内存占用
            # BaseLogger实例 + 处理器 + 格式化器 + 缓存
            memory_per_logger = 2.0  # MB (估算值)

            total_memory = pool_size * memory_per_logger

            # 添加历史数据的内存占用
            history_memory = len(self.history_stats) * 0.1  # 每条历史记录0.1MB

            return total_memory + history_memory

        except Exception:
            return 0.0

    def _check_alerts(self):
        """检查告警条件"""
        if not self.current_stats:
            return

        stats = self.current_stats

        # 命中率告警
        if stats.hit_rate < self.alert_thresholds['hit_rate_low']:
            self._trigger_alert(
                'hit_rate_low',
                f"Logger池命中率过低: {stats.hit_rate:.2f} < {self.alert_thresholds['hit_rate_low']}",
                severity='warning'
            )

        # 池使用率告警
        usage_rate = stats.pool_size / stats.max_size if stats.max_size > 0 else 0
        if usage_rate > self.alert_thresholds['pool_usage_high']:
            self._trigger_alert(
                'pool_usage_high',
                f"Logger池使用率过高: {usage_rate:.2f} > {self.alert_thresholds['pool_usage_high']}",
                severity='warning'
            )

        # 内存使用告警
        if stats.memory_usage_mb > self.alert_thresholds['memory_high']:
            self._trigger_alert(
                'memory_high',
                f"Logger池内存使用过高: {stats.memory_usage_mb:.1f}MB > {self.alert_thresholds['memory_high']}MB",
                severity='error'
            )

    def _trigger_alert(self, alert_type: str, message: str, severity: str):
        """触发告警"""
        alert_data = {
            'alert_type': alert_type,
            'message': message,
            'severity': severity,
            'pool_name': self.pool_name,
            'timestamp': time.time(),
            'stats': self.current_stats.to_dict() if self.current_stats else None
        }

        # 这里可以集成到实际的告警系统中
        print(f"🔥 Logger池告警 [{severity.upper()}]: {message}")

        # TODO: 集成到统一的告警系统
        # alert_system.trigger_alert(alert_data)

    def get_current_stats(self) -> Optional[LoggerPoolStats]:
        """获取当前统计信息"""
        with self._lock:
            return self.current_stats

    def get_history_stats(self, limit: int = 100) -> List[LoggerPoolStats]:
        """获取历史统计信息"""
        with self._lock:
            return self.history_stats[-limit:] if limit > 0 else self.history_stats.copy()

    def get_metrics_for_prometheus(self) -> str:
        """
        生成Prometheus格式的指标

        Returns:
            Prometheus格式的指标字符串
        """
        if not self.current_stats:
            return ""

        # 使用新的指标导出器组件
        if self._metrics_exporter:
            return self._metrics_exporter.export_prometheus_metrics(self.current_stats)
        
        # 回退到原有方法
        stats = self.current_stats
        pool_name = self.pool_name.replace('-', '_')

        metrics = [
            f'# HELP logger_pool_size Logger pool current size',
            f'# TYPE logger_pool_size gauge',
            f'logger_pool_size{{pool="{pool_name}"}} {stats.pool_size}',

            f'# HELP logger_pool_max_size Logger pool maximum size',
            f'# TYPE logger_pool_max_size gauge',
            f'logger_pool_max_size{{pool="{pool_name}"}} {stats.max_size}',

            f'# HELP logger_pool_created_count Total loggers created',
            f'# TYPE logger_pool_created_count counter',
            f'logger_pool_created_count{{pool="{pool_name}"}} {stats.created_count}',

            f'# HELP logger_pool_hit_count Total cache hits',
            f'# TYPE logger_pool_hit_count counter',
            f'logger_pool_hit_count{{pool="{pool_name}"}} {stats.hit_count}',

            f'# HELP logger_pool_hit_rate Cache hit rate (0.0-1.0)',
            f'# TYPE logger_pool_hit_rate gauge',
            f'logger_pool_hit_rate{{pool="{pool_name}"}} {stats.hit_rate}',

            f'# HELP logger_pool_memory_usage_mb Memory usage in MB',
            f'# TYPE logger_pool_memory_usage_mb gauge',
            f'logger_pool_memory_usage_mb{{pool="{pool_name}"}} {stats.memory_usage_mb}',
        ]

        return '\n'.join(metrics) + '\n'

    def record_access_time(self, access_time: float):
        """
        记录访问时间（用于性能监控）

        Args:
            access_time: 访问时间(秒)
        """
        # 优先使用新的监控循环管理器
        if hasattr(self, '_monitoring_loop_manager') and self._monitoring_loop_manager:
            self._monitoring_loop_manager.update_access_time(access_time)
            # 同步到兼容属性
            with self._lock:
                self.access_times = self._monitoring_loop_manager.get_current_access_times()
        elif self._stats_collector:
            # 使用统计收集器组件
            self._stats_collector.record_access_time(access_time)
            # 更新本地记录（向后兼容）
            with self._lock:
                self.access_times.append(access_time)
                if len(self.access_times) > self.max_access_times_size:
                    self.access_times.pop(0)
        else:
            # 回退到基础方式
            with self._lock:
                self.access_times.append(access_time)
                if len(self.access_times) > self.max_access_times_size:
                    self.access_times.pop(0)

    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能汇总报告"""
        with self._lock:
            if not self.current_stats:
                return {}

            stats = self.current_stats

            # 计算性能趋势
            recent_stats = self.history_stats[-10:] if len(
                self.history_stats) >= 10 else self.history_stats

            hit_rate_trend = 0.0
            if len(recent_stats) >= 2:
                old_rate = recent_stats[0].hit_rate
                new_rate = recent_stats[-1].hit_rate
                hit_rate_trend = (new_rate - old_rate) / old_rate if old_rate > 0 else 0

            return {
                'pool_name': self.pool_name,
                'current_stats': stats.to_dict(),
                'performance_metrics': {
                    'hit_rate_trend': hit_rate_trend,
                    'avg_access_time_ms': stats.avg_access_time * 1000,
                    'pool_utilization': stats.pool_size / stats.max_size if stats.max_size > 0 else 0,
                    'memory_efficiency': 'good' if stats.memory_usage_mb < 50 else 'high',
                },
                'alert_status': self._get_alert_status(),
                'recommendations': self._generate_recommendations()
            }

    def _get_alert_status(self) -> Dict[str, bool]:
        """获取告警状态"""
        if not self.current_stats:
            return {}

        stats = self.current_stats
        usage_rate = stats.pool_size / stats.max_size if stats.max_size > 0 else 0

        return {
            'hit_rate_low': stats.hit_rate < self.alert_thresholds['hit_rate_low'],
            'pool_usage_high': usage_rate > self.alert_thresholds['pool_usage_high'],
            'memory_high': stats.memory_usage_mb > self.alert_thresholds['memory_high'],
        }

    def _generate_recommendations(self) -> List[str]:
        """生成优化建议"""
        recommendations = []

        if not self.current_stats:
            return recommendations

        stats = self.current_stats
        alert_status = self._get_alert_status()

        if alert_status.get('hit_rate_low'):
            recommendations.append("考虑增加池大小以提高命中率")

        if alert_status.get('pool_usage_high'):
            recommendations.append("池使用率过高，考虑增加max_size")

        if alert_status.get('memory_high'):
            recommendations.append("内存使用过高，考虑优化Logger实例")

        if stats.hit_rate > 0.95:
            recommendations.append("命中率优秀，性能表现良好")

        return recommendations


# 全局监控实例
_logger_pool_monitor: Optional[LoggerPoolMonitor] = None
_monitor_lock = threading.Lock()


def get_logger_pool_monitor(pool_name: str = "default") -> LoggerPoolMonitor:
    """
    获取Logger池监控器实例

    Args:
        pool_name: 池名称

    Returns:
        LoggerPoolMonitor: 监控器实例
    """
    global _logger_pool_monitor

    with _monitor_lock:
        if _logger_pool_monitor is None:
            _logger_pool_monitor = LoggerPoolMonitor(pool_name=pool_name)
            _logger_pool_monitor.start_monitoring()

        return _logger_pool_monitor


def start_logger_pool_monitoring(pool_name: str = "default"):
    """启动Logger池监控"""
    monitor = get_logger_pool_monitor(pool_name)
    monitor.start_monitoring()


def stop_logger_pool_monitoring():
    """停止Logger池监控"""
    global _logger_pool_monitor

    with _monitor_lock:
        if _logger_pool_monitor:
            _logger_pool_monitor.stop_monitoring()
            _logger_pool_monitor = None


def get_logger_pool_metrics() -> Dict[str, Any]:
    """
    获取Logger池监控指标

    Returns:
        监控指标字典
    """
    monitor = get_logger_pool_monitor()
    return monitor.get_performance_summary()


# 在模块导入时自动启动监控
try:
    start_logger_pool_monitoring()
except Exception as e:
    print(f"启动Logger池监控失败: {e}")
