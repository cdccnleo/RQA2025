import os
import psutil

class ResourceAllocationError(Exception):
    """资源分配异常"""
    pass

import time
import threading
from typing import Dict, List, Optional, Callable
from datetime import datetime
import logging
import platform

logger = logging.getLogger(__name__)

class ResourceManager:
    """系统资源管理器"""

    def __init__(
        self,
        cpu_threshold: float = 90.0,
        mem_threshold: float = 90.0,
        disk_threshold: float = 90.0,
        check_interval: float = 5.0,
        alert_handlers: Optional[List[Callable]] = None
    ):
        """
        初始化资源管理器

        Args:
            cpu_threshold: CPU使用率告警阈值(%)
            mem_threshold: 内存使用率告警阈值(%)
            disk_threshold: 磁盘使用率告警阈值(%)
            check_interval: 资源检查间隔(秒)
            alert_handlers: 告警处理器列表
        """
        self.cpu_threshold = cpu_threshold
        self.mem_threshold = mem_threshold
        self.disk_threshold = disk_threshold
        self.check_interval = check_interval
        self.alert_handlers = alert_handlers or []

        self._monitoring = False
        self._monitor_thread = None
        self._stats: List[Dict] = []
        self._lock = threading.Lock()
        
        # 策略资源配额管理
        self.strategy_quotas: Dict[str, Dict] = {}  # {strategy_name: quota_config}
        self.strategy_usage: Dict[str, Dict] = {}  # {strategy_name: resource_usage}
        
        # 初始化GPU管理器
        self.gpu_manager = self._init_gpu_manager()

    def _init_gpu_manager(self):
        """初始化GPU管理器"""
        try:
            import torch
            if torch.cuda.is_available():
                from .gpu_manager import GPUMonitor
                return GPUMonitor()
        except ImportError:
            logger.warning("PyTorch not available, GPU monitoring disabled")
        return None

    def start_monitoring(self) -> None:
        """启动资源监控"""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("Resource monitoring started")

    def stop_monitoring(self) -> None:
        """停止资源监控"""
        if not self._monitoring:
            return

        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=self.check_interval * 2)
        logger.info("Resource monitoring stopped")

    def _monitor_loop(self) -> None:
        """资源监控循环"""
        while self._monitoring:
            try:
                stats = self.get_current_stats()

                with self._lock:
                    self._stats.append(stats)
                    if len(self._stats) > 1000:  # 限制历史数据量
                        self._stats = self._stats[-1000:]

                # 检查资源阈值
                self._check_thresholds(stats)

            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")

            time.sleep(self.check_interval)

    def get_current_stats(self) -> Dict:
        """
        获取当前资源使用情况

        Returns:
            Dict: 资源统计信息
        """
        timestamp = datetime.now().isoformat()

        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)

        # 内存使用
        mem = psutil.virtual_memory()

        # 磁盘使用
        disk = psutil.disk_usage('/')

        # 网络IO
        net_io = psutil.net_io_counters()

        # GPU使用(如果可用)
        gpu_stats = self._get_gpu_stats()

        return {
            'timestamp': timestamp,
            'cpu': {
                'percent': cpu_percent,
                'cores': psutil.cpu_count(),
                'load_avg': self._get_load_avg()
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
                'bytes_recv': net_io.bytes_recv
            },
            'gpu': gpu_stats
        }

    def _get_gpu_stats(self) -> Optional[List[Dict]]:
        """获取GPU统计信息"""
        if not self.gpu_manager:
            return None

        try:
            return self.gpu_manager.get_gpu_stats()
        except Exception as e:
            logger.error(f"Failed to get GPU stats: {e}")
            return None

    def _get_load_avg(self) -> Optional[List[float]]:
        """获取系统负载"""
        try:
            if platform.system() == 'Linux':
                return list(os.getloadavg())
        except Exception:
            pass
        return None

    def _check_thresholds(self, stats: Dict) -> None:
        """检查资源使用是否超过阈值"""
        alerts = []

        # CPU检查
        if stats['cpu']['percent'] > self.cpu_threshold:
            alerts.append({
                'type': 'cpu',
                'level': 'warning',
                'message': f"CPU usage {stats['cpu']['percent']}% exceeds threshold {self.cpu_threshold}%",
                'value': stats['cpu']['percent'],
                'threshold': self.cpu_threshold
            })

        # 内存检查
        if stats['memory']['percent'] > self.mem_threshold:
            alerts.append({
                'type': 'memory',
                'level': 'warning',
                'message': f"Memory usage {stats['memory']['percent']}% exceeds threshold {self.mem_threshold}%",
                'value': stats['memory']['percent'],
                'threshold': self.mem_threshold
            })

        # 磁盘检查
        if stats['disk']['percent'] > self.disk_threshold:
            alerts.append({
                'type': 'disk',
                'level': 'warning',
                'message': f"Disk usage {stats['disk']['percent']}% exceeds threshold {self.disk_threshold}%",
                'value': stats['disk']['percent'],
                'threshold': self.disk_threshold
            })

        # 触发告警
        for alert in alerts:
            self._trigger_alert(alert)

    def _trigger_alert(self, alert: Dict) -> None:
        """触发资源告警"""
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

    def get_stats(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> List[Dict]:
        """
        获取资源使用历史数据

        Args:
            start_time: 开始时间(ISO格式)
            end_time: 结束时间(ISO格式)

        Returns:
            List[Dict]: 资源使用统计列表
        """
        with self._lock:
            stats = self._stats.copy()

        if start_time:
            stats = [s for s in stats if s['timestamp'] >= start_time]

        if end_time:
            stats = [s for s in stats if s['timestamp'] <= end_time]

        return stats

    def get_summary(self) -> Dict:
        """
        获取资源使用摘要

        Returns:
            Dict: 资源使用摘要
        """
        with self._lock:
            stats = self._stats.copy()

        if not stats:
            return {}

        # CPU统计
        cpu_values = [s['cpu']['percent'] for s in stats]
        cpu_avg = sum(cpu_values) / len(cpu_values)
        cpu_max = max(cpu_values)

        # 内存统计
        mem_values = [s['memory']['percent'] for s in stats]
        mem_avg = sum(mem_values) / len(mem_values)
        mem_max = max(mem_values)

        # 磁盘统计
        disk_values = [s['disk']['percent'] for s in stats]
        disk_avg = sum(disk_values) / len(disk_values)
        disk_max = max(disk_values)

        return {
            'cpu': {
                'avg': cpu_avg,
                'max': cpu_max,
                'threshold': self.cpu_threshold
            },
            'memory': {
                'avg': mem_avg,
                'max': mem_max,
                'threshold': self.mem_threshold
            },
            'disk': {
                'avg': disk_avg,
                'max': disk_max,
                'threshold': self.disk_threshold
            },
            'period': {
                'start': stats[0]['timestamp'],
                'end': stats[-1]['timestamp'],
                'count': len(stats)
            }
        }
