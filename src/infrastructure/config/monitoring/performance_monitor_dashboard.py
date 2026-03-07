
import datetime
import threading
import time
from pathlib import Path

from typing import Dict, Any, Optional
from .anomaly_detector import AnomalyDetector
from .core import PerformanceMonitorDashboardCore
from .performance_predictor import PerformancePredictor
from .trend_analyzer import TrendAnalyzer
"""统一性能监控面板"""


class PerformanceMonitorDashboard:
    """统一性能监控面板 - 整合所有监控功能"""

    def __init__(self, storage_path: str = "config/performance",
                 retention_days: int = 30,
                 enable_system_monitoring: bool = True):
        """初始化统一监控面板"""
        self.storage_path = Path(storage_path)
        self.retention_days = retention_days
        self.core = PerformanceMonitorDashboardCore(str(self.storage_path), retention_days)
        self.anomaly_detector = AnomalyDetector()
        self.trend_analyzer = TrendAnalyzer()
        self.performance_predictor = PerformancePredictor()
        self.enable_system_monitoring = enable_system_monitoring
        self._system_resources = []  # 系统资源数据列表
        self._monitoring_enabled = False  # 监控是否启用
        self._monitoring_active = False  # 监控是否激活
        self._metrics_list = []  # 指标数据列表
        self._operation_stats = {}  # 操作统计数据
        self._monitor_thread = None  # 监控线程
        self._shutdown_event: Optional[threading.Event] = None

    @property
    def _metrics(self):
        """访问核心的指标数据"""
        return self._metrics_list

    def start_monitoring(self):
        """启动监控"""
        self.core.start()
        self._monitoring_active = True
        self._monitoring_enabled = True
        if self.enable_system_monitoring:
            self._shutdown_event = threading.Event()
            self._monitor_thread = threading.Thread(target=self._system_monitor_loop, daemon=True, name="PerformanceSystemMonitor")
            self._monitor_thread.start()
            # 立即采集一次，确保测试可见
            self._collect_system_resources()

    def stop_monitoring(self):
        """停止监控"""
        self.core.stop()
        self._monitoring_active = False
        self._monitoring_enabled = False
        if self._shutdown_event is not None:
            self._shutdown_event.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2)
        self._monitor_thread = None
        self._shutdown_event = None
        self._system_resources.clear()

    def record_operation(self, operation_type: str, duration: float, success: bool = True, metadata: Optional[Dict[str, Any]] = None):
        """记录操作"""
        result = self.core.record_operation(operation_type, duration, success, metadata)
        # 添加到本地指标列表
        self._metrics_list.append({
            "operation_type": operation_type,
            "duration": duration,
            "success": success,
            "metadata": metadata
        })
        return result if result is not None else True

    def get_operation_stats(self) -> Dict[str, Any]:
        """获取操作统计"""
        return self.core.get_operation_stats()

    def _system_monitor_loop(self):
        """后台系统资源监控循环"""
        while self._shutdown_event and not self._shutdown_event.is_set():
            self._collect_system_resources()
            time.sleep(1)

    def _collect_system_resources(self):
        """收集系统资源信息"""
        try:
            import psutil

            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            net = psutil.net_io_counters()

            snapshot = {
                'timestamp': datetime.datetime.now().isoformat(),
                'cpu_percent': cpu_percent,
                'memory_percent': getattr(memory, 'percent', None),
                'disk_percent': getattr(disk, 'percent', None),
                'bytes_sent': getattr(net, 'bytes_sent', None),
                'bytes_recv': getattr(net, 'bytes_recv', None),
            }
            self._system_resources.append(snapshot)
        except Exception:
            # 忽略系统资源采集异常以保证监控不受影响
            pass

    def get_system_health_status(self) -> Dict[str, Any]:
        """获取系统健康状态"""
        return self.core.get_system_health_status()

    def detect_anomalies(self, metric_name: str = None) -> Dict[str, Any]:
        """检测异常"""
        return self.anomaly_detector.detect_anomaly(metric_name) if metric_name else {}

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            "status": "healthy",
            "monitoring_enabled": self._monitoring_enabled,
            "system_monitoring": self.enable_system_monitoring,
            "timestamp": datetime.datetime.now().isoformat()
        }

    def analyze_trends(self, metric_name: str = None) -> Dict[str, Any]:
        """分析趋势"""
        return self.trend_analyzer.analyze_trend(metric_name) if metric_name else {}

    def predict_performance(self, metric_name: str = None, hours_ahead: int = 1) -> Dict[str, Any]:
        """预测性能"""
        return self.performance_predictor.predict_next_value(metric_name) if metric_name else {}

    def get_monitoring_stats(self) -> Dict[str, Any]:
        """获取监控统计"""
        return {
            "core_stats": self.core.get_operation_stats(),
            "anomalies": self.detect_anomalies(),
            "trends": self.analyze_trends(),
            "predictions": self.predict_performance()
        }

    def get_memory_leak_detection(self) -> Dict[str, Any]:
        """获取内存泄漏检测结果"""
        return self.core.get_memory_leak_detection()

    def get_connection_pool_metrics(self) -> Dict[str, Any]:
        """获取连接池监控指标"""
        return self.core.get_connection_pool_metrics()

    def get_cache_efficiency_metrics(self) -> Dict[str, Any]:
        """获取缓存效率监控指标"""
        return self.core.get_cache_efficiency_metrics()

    def get_business_metrics(self) -> Dict[str, Any]:
        """获取业务指标监控"""
        return self.core.get_business_metrics()

    def get_security_metrics(self) -> Dict[str, Any]:
        """获取安全监控指标"""
        return self.core.get_security_metrics()

    def get_comprehensive_health_report(self) -> Dict[str, Any]:
        """获取综合健康报告"""
        return self.core.get_comprehensive_health_report()

    def get_enhanced_monitoring_stats(self) -> Dict[str, Any]:
        """获取增强的监控统计信息"""
        return {
            "basic_stats": self.get_monitoring_stats(),
            "system_health": self.get_system_health_status(),
            "memory_leaks": self.get_memory_leak_detection(),
            "connections": self.get_connection_pool_metrics(),
            "cache_efficiency": self.get_cache_efficiency_metrics(),
            "business_metrics": self.get_business_metrics(),
            "security_metrics": self.get_security_metrics(),
            "comprehensive_health": self.get_comprehensive_health_report()
        }

    # 保持向后兼容的别名
    get_performance_report = get_monitoring_stats
    get_performance_summary = get_monitoring_stats




