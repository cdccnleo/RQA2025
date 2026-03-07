"""
性能监控模块 - 数据层性能监控
使用单例模式确保全局唯一实例
"""
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
from collections import defaultdict, deque


def _resolve_logger():
    """解析基础设施日志，如缺失则回退到标准 logging。"""
    try:
        from src.infrastructure.logging import get_infrastructure_logger as infra_logger

        return infra_logger
    except ImportError:  # pragma: no cover - 在 fallback 测试中覆盖
        import logging

        def _fallback_logger(name: str):
            logger = logging.getLogger(name)
            logger.warning("无法导入基础设施层日志，使用标准logging")
            return logger

        return _fallback_logger


get_infrastructure_logger = _resolve_logger()

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - 环境缺失时的降级日志
    psutil = None

logger = get_infrastructure_logger('__name__')


@dataclass
class PerformanceMetric:

    """性能指标"""
    name: str
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceAlert:

    """性能告警"""
    level: str  # 'info', 'warning', 'error', 'critical'
    message: str
    metric_name: str
    threshold: float
    current_value: float
    timestamp: datetime = field(default_factory=datetime.now)


class PerformanceMonitor:

    """
    性能监控器 - 单例模式
    监控数据层的各种性能指标
    """
    
    # 单例实例
    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """确保只有一个实例"""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, max_history: int = 1000):
        """
        初始化性能监控器（仅第一次调用有效）

        Args:
            max_history: 最大历史记录数
        """
        # 避免重复初始化
        if self._initialized:
            return
            
        self.max_history = max_history
        self._lock = threading.RLock()

        # 性能指标存储
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))

        # 告警配置
        self.alert_thresholds = {
            'cache_hit_rate': {'warning': 0.8, 'error': 0.6, 'critical': 0.4},
            'data_load_time': {'warning': 5.0, 'error': 10.0, 'critical': 30.0},
            'memory_usage': {'warning': 0.8, 'error': 0.9, 'critical': 0.95},
            'error_rate': {'warning': 0.05, 'error': 0.1, 'critical': 0.2}
        }

        # 告警历史
        self.alerts: List[PerformanceAlert] = []

        # 监控状态
        self.is_monitoring = False
        self.monitor_thread = None
        
        # 标记已初始化
        self._initialized = True

        logger.info("PerformanceMonitor initialized (singleton)")


def get_performance_monitor() -> PerformanceMonitor:
    """
    获取 PerformanceMonitor 单例实例
    
    Returns:
        PerformanceMonitor: 全局唯一的性能监控器实例
    """
    return PerformanceMonitor()

    def start_monitoring(self):
        """开始监控"""
        with self._lock:
            if not self.is_monitoring:
                self.is_monitoring = True
                self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
                self.monitor_thread.start()
                logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """停止监控"""
        with self._lock:
            self.is_monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=5)
            logger.info("Performance monitoring stopped")

    def record_metric(self, name: str, value: float, unit: str = "", metadata: Dict[str, Any] = None):
        """
        记录性能指标（内存+数据库持久化）

        Args:
            name: 指标名称
            value: 指标值
            unit: 单位
            metadata: 元数据
        """
        with self._lock:
            metric = PerformanceMetric(
                name=name,
                value=value,
                unit=unit,
                metadata=metadata or {}
            )
            self.metrics[name].append(metric)

            # 检查告警
            self._check_alerts(name, value)
            
            # 异步保存到数据库（不阻塞主流程）
            threading.Thread(
                target=self._persist_metric_to_db,
                args=(name, value, unit, metadata),
                daemon=True
            ).start()
    
    def _persist_metric_to_db(self, name: str, value: float, unit: str, metadata: Dict[str, Any]):
        """
        将指标持久化到数据库
        
        Args:
            name: 指标名称
            value: 指标值
            unit: 单位
            metadata: 元数据
        """
        try:
            from src.gateway.web.postgresql_persistence import get_db_connection
            
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # 插入指标数据
            insert_query = """
                INSERT INTO performance_metrics (
                    metric_name, metric_value, unit, metadata, recorded_at
                ) VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
            """
            
            import json
            cursor.execute(insert_query, (
                name,
                value,
                unit,
                json.dumps(metadata or {})
            ))
            
            conn.commit()
            logger.debug(f"✅ 指标已持久化: {name}={value}{unit}")
            
        except Exception as e:
            # 数据库持久化失败不影响主流程
            logger.debug(f"指标持久化失败（非关键）: {e}")
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def record_cache_hit_rate(self, hit_rate: float):
        """记录缓存命中率"""
        self.record_metric('cache_hit_rate', hit_rate, '%')

    def record_data_load_time(self, load_time: float):
        """记录数据加载时间"""
        self.record_metric('data_load_time', load_time, 'seconds')

    def record_memory_usage(self, usage_percent: float):
        """记录内存使用率"""
        self.record_metric('memory_usage', usage_percent, '%')

    def record_error_rate(self, error_rate: float):
        """记录错误率"""
        self.record_metric('error_rate', error_rate, '%')

    def record_throughput(self, operations_per_second: float):
        """记录吞吐量"""
        self.record_metric('throughput', operations_per_second, 'ops / s')

    def get_metric_history(self, name: str, hours: int = 24) -> List[PerformanceMetric]:
        """
        获取指标历史

        Args:
            name: 指标名称
            hours: 历史小时数

        Returns:
            List[PerformanceMetric]: 指标历史
        """
        with self._lock:
            if name not in self.metrics:
                return []

            cutoff_time = datetime.now() - timedelta(hours=hours)
            return [
                metric for metric in self.metrics[name]
                if metric.timestamp >= cutoff_time
            ]

    def get_current_metric(self, name: str) -> Optional[PerformanceMetric]:
        """
        获取当前指标值

        Args:
            name: 指标名称

        Returns:
            Optional[PerformanceMetric]: 当前指标
        """
        with self._lock:
            if name not in self.metrics or not self.metrics[name]:
                return None
            return self.metrics[name][-1]

    def get_metric_statistics(self, name: str, hours: int = 24) -> Dict[str, float]:
        """
        获取指标统计信息

        Args:
            name: 指标名称
            hours: 统计小时数

        Returns:
            Dict[str, float]: 统计信息
        """
        history = self.get_metric_history(name, hours)
        if not history:
            return {}

        values = [metric.value for metric in history]
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'latest': values[-1] if values else 0
        }

    def get_all_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """
        获取所有指标摘要

        Returns:
            Dict[str, Dict[str, float]]: 所有指标摘要
        """
        with self._lock:
            summary = {}
            for name in self.metrics:
                summary[name] = self.get_metric_statistics(name, hours=1)
            return summary

    def get_recent_alerts(self, hours: int = 24) -> List[PerformanceAlert]:
        """
        获取最近的告警

        Args:
            hours: 小时数

        Returns:
            List[PerformanceAlert]: 告警列表
        """
        with self._lock:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            return [
                alert for alert in self.alerts
                if alert.timestamp >= cutoff_time
            ]

    def set_alert_threshold(self, metric_name: str, level: str, threshold: float):
        """
        设置告警阈值

        Args:
            metric_name: 指标名称
            level: 告警级别
            threshold: 阈值
        """
        with self._lock:
            if metric_name not in self.alert_thresholds:
                self.alert_thresholds[metric_name] = {}
            self.alert_thresholds[metric_name][level] = threshold

    def _check_alerts(self, metric_name: str, value: float):
        """检查告警"""
        if metric_name not in self.alert_thresholds:
            return

        thresholds = self.alert_thresholds[metric_name]

        for level, threshold in thresholds.items():
            if self._should_alert(metric_name, value, threshold, level):
                alert = PerformanceAlert(
                    level=level,
                    message=f"{metric_name} exceeded {level} threshold: {value} > {threshold}",
                    metric_name=metric_name,
                    threshold=threshold,
                    current_value=value
                )
                self.alerts.append(alert)
                logger.warning(f"Performance alert: {alert.message}")

    def _should_alert(self, metric_name: str, value: float, threshold: float, level: str) -> bool:
        """判断是否应该告警"""
        # 根据指标类型判断告警条件
        if metric_name in ['cache_hit_rate', 'memory_usage']:
            # 低于阈值告警
            return value < threshold
        elif metric_name in ['data_load_time', 'error_rate']:
            # 高于阈值告警
            return value > threshold
        else:
            # 默认高于阈值告警
            return value > threshold

    def _monitor_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                # 监控系统资源
                self._monitor_system_resources()

                # 清理旧告警
                self._cleanup_old_alerts()

                time.sleep(60)  # 每分钟检查一次
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                time.sleep(60)

    def _monitor_system_resources(self):
        """监控系统资源"""
        if psutil is None:
            logger.warning("psutil not available, system resource monitoring disabled")
            return

        try:
            memory = psutil.virtual_memory()
            self.record_memory_usage(memory.percent)

            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric('cpu_usage', cpu_percent, '%')

            disk = psutil.disk_usage('/')
            self.record_metric('disk_usage', disk.percent, '%')

        except Exception as e:
            logger.error(f"Error monitoring system resources: {e}")

    def _cleanup_old_alerts(self):
        """清理旧告警"""
        with self._lock:
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.alerts = [
                alert for alert in self.alerts
                if alert.timestamp >= cutoff_time
            ]

    def export_metrics(self, format: str = 'json') -> str:
        """
        导出指标数据

        Args:
            format: 导出格式 ('json', 'csv')

        Returns:
            str: 导出的数据
        """
        import json
        import csv
        from io import StringIO

        if format == 'json':
            data = {
                'metrics': {},
                'alerts': [
                    {
                        'level': alert.level,
                        'message': alert.message,
                        'metric_name': alert.metric_name,
                        'threshold': alert.threshold,
                        'current_value': alert.current_value,
                        'timestamp': alert.timestamp.isoformat()
                    }
                    for alert in self.alerts
                ]
            }

            for name, metrics in self.metrics.items():
                data['metrics'][name] = [
                    {
                        'value': metric.value,
                        'unit': metric.unit,
                        'timestamp': metric.timestamp.isoformat(),
                        'metadata': metric.metadata
                    }
                    for metric in metrics
                ]

            return json.dumps(data, indent=2)

        elif format == 'csv':
            output = StringIO()
            writer = csv.writer(output)

            # 写入表头
            writer.writerow(['metric_name', 'value', 'unit', 'timestamp', 'metadata'])

            # 写入数据
            for name, metrics in self.metrics.items():
                for metric in metrics:
                    writer.writerow([
                        name,
                        metric.value,
                        metric.unit,
                        metric.timestamp.isoformat(),
                        json.dumps(metric.metadata)
                    ])

            return output.getvalue()

        else:
            raise ValueError(f"Unsupported format: {format}")

    def get_performance_report(self) -> Dict[str, Any]:
        """
        获取性能报告

        Returns:
            Dict[str, Any]: 性能报告
        """
        with self._lock:
            report = {
                'timestamp': datetime.now().isoformat(),
                'metrics_summary': self.get_all_metrics_summary(),
                'recent_alerts': [
                    {
                        'level': alert.level,
                        'message': alert.message,
                        'metric_name': alert.metric_name,
                        'timestamp': alert.timestamp.isoformat()
                    }
                    for alert in self.get_recent_alerts(hours=1)
                ],
                'alert_count': len(self.alerts),
                'monitoring_status': 'active' if self.is_monitoring else 'inactive'
            }

            return report
