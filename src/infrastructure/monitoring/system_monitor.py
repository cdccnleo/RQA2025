import os
import psutil
import platform
import time
import threading
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
import logging
import socket

logger = logging.getLogger(__name__)

class SystemMonitor:
    """系统资源监控器"""

    def __init__(
        self,
        check_interval: float = 60.0,
        alert_handlers: Optional[List[Callable[[str, Dict], None]]] = None,
        psutil_mock: Optional[Any] = None,
        os_mock: Optional[Any] = None,
        socket_mock: Optional[Any] = None
    ):
        """
        初始化系统监控器

        Args:
            check_interval: 监控检查间隔(秒)
            alert_handlers: 告警处理器列表
            psutil_mock: 可选的psutil mock，用于测试时注入mock对象
            os_mock: 可选的os mock，用于测试时注入mock对象
            socket_mock: 可选的socket mock，用于测试时注入mock对象
        """
        self.check_interval = check_interval
        self.alert_handlers = alert_handlers or []

        # 测试钩子：允许注入mock的依赖
        if psutil_mock is not None:
            self.psutil = psutil_mock
        else:
            import psutil
            self.psutil = psutil
            
        if os_mock is not None:
            self.os = os_mock
        else:
            import os
            self.os = os
            
        if socket_mock is not None:
            self.socket = socket_mock
        else:
            import socket
            self.socket = socket

        self._monitoring = False
        self._monitor_thread = None
        self._stats: List[Dict] = []

        # 系统基础信息
        self.system_info = self._get_system_info()

    def _get_system_info(self) -> Dict:
        """获取系统基础信息"""
        return {
            'hostname': self.socket.gethostname(),
            'platform': platform.platform(),
            'system': platform.system(),
            'release': platform.release(),
            'processor': platform.processor(),
            'cpu_count': self.psutil.cpu_count(),
            'boot_time': datetime.fromtimestamp(self.psutil.boot_time()).isoformat(),
            'python_version': platform.python_version()
        }

    def start_monitoring(self) -> None:
        """启动系统监控"""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("System monitoring started")

    def stop_monitoring(self) -> None:
        """停止系统监控"""
        if not self._monitoring:
            return

        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=self.check_interval * 2)
        logger.info("System monitoring stopped")

    def _monitor_loop(self) -> None:
        """监控主循环"""
        while self._monitoring:
            try:
                stats = self._collect_system_stats()

                # 保存统计数据
                self._stats.append(stats)
                if len(self._stats) > 1000:  # 限制历史数据量
                    self._stats = self._stats[-1000:]

                # 检查系统状态
                self._check_system_status(stats)

            except Exception as e:
                logger.error(f"System monitoring error: {e}")

            time.sleep(self.check_interval)

    def _collect_system_stats(self) -> Dict:
        """收集系统统计数据"""
        # CPU使用率
        cpu_percent = self.psutil.cpu_percent(interval=1)

        # 内存使用
        mem = self.psutil.virtual_memory()

        # 磁盘使用
        disk = self.psutil.disk_usage('/')

        # 网络IO
        net_io = self.psutil.net_io_counters()

        # 系统负载(仅Linux)
        load_avg = self._get_load_avg()

        # 进程数量
        process_count = len(self.psutil.pids())

        return {
            'timestamp': datetime.now().isoformat(),
            'cpu': {
                'percent': cpu_percent,
                'load_avg': load_avg
            },
            'memory': {
                'total': mem.total,
                'available': mem.available,
                'used': mem.used,
                'percent': mem.percent
            },
            'disk': {
                'total': disk.total,
                'used': disk.used,
                'free': disk.free,
                'percent': disk.percent
            },
            'network': {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            },
            'process': {
                'count': process_count
            }
        }

    def _get_load_avg(self) -> Optional[List[float]]:
        """获取系统负载(仅Linux)"""
        try:
            if hasattr(self.os, 'getloadavg'):
                return list(self.os.getloadavg())
        except Exception:
            pass
        return None

    def _check_system_status(self, stats: Dict) -> None:
        """检查系统状态并触发告警"""
        alerts = []

        # CPU检查
        if stats['cpu']['percent'] > 90:
            alerts.append({
                'type': 'cpu',
                'level': 'critical',
                'message': f"High CPU usage: {stats['cpu']['percent']}%",
                'value': stats['cpu']['percent'],
                'threshold': 90
            })

        # 内存检查
        if stats['memory']['percent'] > 90:
            alerts.append({
                'type': 'memory',
                'level': 'critical',
                'message': f"High memory usage: {stats['memory']['percent']}%",
                'value': stats['memory']['percent'],
                'threshold': 90
            })

        # 磁盘检查
        if stats['disk']['percent'] > 90:
            alerts.append({
                'type': 'disk',
                'level': 'critical',
                'message': f"High disk usage: {stats['disk']['percent']}%",
                'value': stats['disk']['percent'],
                'threshold': 90
            })

        # 触发告警
        for alert in alerts:
            self._trigger_alert(alert)

    def _trigger_alert(self, alert: Dict) -> None:
        """触发系统告警"""
        for handler in self.alert_handlers:
            try:
                handler('system', alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

    def get_stats(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> List[Dict]:
        """
        获取系统监控历史数据

        Args:
            start_time: 开始时间(ISO格式)
            end_time: 结束时间(ISO格式)

        Returns:
            List[Dict]: 系统监控数据列表
        """
        if not start_time and not end_time:
            return self._stats.copy()

        stats = self._stats.copy()

        if start_time:
            stats = [s for s in stats if s['timestamp'] >= start_time]

        if end_time:
            stats = [s for s in stats if s['timestamp'] <= end_time]

        return stats

    def get_summary(self) -> Dict:
        """
        获取系统监控摘要

        Returns:
            Dict: 系统监控摘要
        """
        if not self._stats:
            return {}

        # CPU统计
        cpu_values = [s['cpu']['percent'] for s in self._stats]
        cpu_avg = sum(cpu_values) / len(cpu_values)
        cpu_max = max(cpu_values)

        # 内存统计
        mem_values = [s['memory']['percent'] for s in self._stats]
        mem_avg = sum(mem_values) / len(mem_values)
        mem_max = max(mem_values)

        # 磁盘统计
        disk_values = [s['disk']['percent'] for s in self._stats]
        disk_avg = sum(disk_values) / len(disk_values)
        disk_max = max(disk_values)

        return {
            'system_info': self.system_info,
            'cpu': {
                'avg': cpu_avg,
                'max': cpu_max
            },
            'memory': {
                'avg': mem_avg,
                'max': mem_max
            },
            'disk': {
                'avg': disk_avg,
                'max': disk_max
            },
            'period': {
                'start': self._stats[0]['timestamp'],
                'end': self._stats[-1]['timestamp'],
                'count': len(self._stats)
            }
        }


class ResourceMonitor:
    """资源监控装饰器，用于记录方法执行的资源使用情况"""
    
    def __init__(self, monitor: Optional[SystemMonitor] = None):
        self.monitor = monitor or SystemMonitor()
        
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            # 记录开始前的资源状态
            start_stats = self.monitor._collect_system_stats()
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
            finally:
                # 记录结束后的资源状态
                end_stats = self.monitor._collect_system_stats()
                end_time = time.time()
                
                # 计算资源使用差异
                usage = {
                    'function': func.__name__,
                    'duration': end_time - start_time,
                    'cpu_delta': end_stats['cpu']['percent'] - start_stats['cpu']['percent'],
                    'memory_delta': end_stats['memory']['used'] - start_stats['memory']['used'],
                    'timestamp': datetime.now().isoformat()
                }
                
                logger.info(f"Resource usage for {func.__name__}: {usage}")
                
            return result
            
        return wrapper
