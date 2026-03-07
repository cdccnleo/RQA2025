
import psutil
# 跨层级导入：infrastructure层组件

from datetime import datetime
from src.infrastructure.interfaces.standard_interfaces import IMonitor, ServiceStatus
from typing import Dict, Any, Optional, List, Tuple
"""
统一监控器适配器

将现有的UnifiedMonitor适配到统一接口，解决接口不一致问题
"""


class UnifiedMonitor:
    """简化的统一监控器实现"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.is_running = False
        self._metrics: Dict[str, List[Dict[str, Any]]] = {}
        self._alerts: List[Dict[str, Any]] = []

    def start(self) -> None:
        """启动监控"""
        self.is_running = True

    def stop(self) -> None:
        """停止监控"""
        self.is_running = False

    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """记录指标数据"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'value': value,
            'tags': tags or {}
        }
        self._metrics.setdefault(name, []).append(entry)

    def record_alert(self, level: str, message: str, tags: Optional[Dict[str, str]] = None) -> None:
        """记录告警"""
        self._alerts.append({
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
            'tags': tags or {}
        })

    def get_status(self) -> ServiceStatus:
        """获取状态"""
        return ServiceStatus.RUNNING if self.is_running else ServiceStatus.STOPPED

    def get_metrics(self, name: str, time_range=None):
        """获取指标数据"""
        if name and name in self._metrics:
            return list(self._metrics[name])
        elif not name:
            # 返回所有指标
            return [item for sublist in self._metrics.values() for item in sublist]
        return []

    def get_alerts(self, level=None):
        """获取告警数据"""
        if level:
            return [alert for alert in self._alerts if alert.get('level') == level]
        return list(self._alerts)


class UnifiedMonitorAdapter(IMonitor):

    """统一监控器适配器"""

    def __init__(self, **kwargs):
        """初始化适配器"""
        self._monitor = UnifiedMonitor(config=kwargs)
        self._status = ServiceStatus.STOPPED

    def start(self) -> None:
        """启动监控"""
        try:
            self._monitor.start()
            self._status = ServiceStatus.RUNNING
        except Exception:
            self._status = ServiceStatus.ERROR

    def stop(self) -> None:
        """停止监控"""
        try:
            self._monitor.stop()
            self._status = ServiceStatus.STOPPED
        except Exception:
            self._status = ServiceStatus.ERROR

    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """记录指标"""
        self._monitor.record_metric(name, value, tags)

    def record_alert(self, level: str, message: str, tags: Optional[Dict[str, str]] = None) -> None:
        """记录告警"""
        self._monitor.record_alert(level, message, tags)

    def get_metrics(self, name: str, time_range: Optional[Tuple[datetime, datetime]] = None) -> List[Dict]:
        """获取指标数据"""
        return self._monitor.get_metrics(name, time_range)

    def get_alerts(self, level: Optional[str] = None) -> List[Dict]:
        """获取告警数据"""
        return self._monitor.get_alerts(level)

    def get_status(self) -> ServiceStatus:
        """获取监控状态"""
        return self._status

    def collect_metrics(self) -> Dict[str, Any]:
        """
        收集监控指标

        Returns:
            Dict: 监控指标数据
        """
        try:
            # 获取基本的系统指标
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            return {
                'timestamp': datetime.now().isoformat(),
                'status': 'running' if self._monitor.is_running else 'stopped',
                'metrics_count': len(self.get_metrics('', None)),
                'alerts_count': len(self.get_alerts()),
                'cpu': {'percent': cpu_percent},
                'memory': {'percent': memory.percent, 'used': memory.used, 'total': memory.total},
                'disk': {'percent': disk.percent, 'used': disk.used, 'total': disk.total},
                'network': {'bytes_sent': network.bytes_sent, 'bytes_recv': network.bytes_recv}
            }
        except ImportError:
            # 如果没有psutil，返回基本信息
            return {
                'timestamp': datetime.now().isoformat(),
                'status': 'running' if self._monitor.is_running else 'stopped',
                'metrics_count': len(self.get_metrics('', None)),
                'alerts_count': len(self.get_alerts()),
                'cpu': {'percent': 0.0},
                'memory': {'percent': 0.0, 'used': 0, 'total': 0},
                'disk': {'percent': 0.0, 'used': 0, 'total': 0},
                'network': {'bytes_sent': 0, 'bytes_recv': 0}
            }

    def get_system_status(self) -> Dict[str, Any]:
        """
        获取系统状态

        Returns:
            Dict: 系统状态信息
        """
        return {
            'status': self._status.value,
            'monitor_running': self._monitor.is_running,
            'overall_health': 'healthy' if self._monitor.is_running else 'unhealthy',
            'components': {'monitor': 'running' if self._monitor.is_running else 'stopped'},
            'timestamp': datetime.now().isoformat()
        }

    def register_monitor(self, monitor: Any) -> None:
        """
        注册监控器

        Args:
            monitor: 要注册的监控器
        """
        # 在这个简化实现中，我们只是记录注册操作
        if not hasattr(self, '_registered_monitors'):
            self._registered_monitors = []
        self._registered_monitors.append(monitor)

    def get_registered_monitors(self) -> List[Any]:
        """
        获取已注册的监控器列表

        Returns:
            List[Any]: 已注册的监控器列表
        """
        return getattr(self, '_registered_monitors', [])

    def get_underlying_monitor(self) -> UnifiedMonitor:
        """获取底层监控器实例"""
        return self._monitor
