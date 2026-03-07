"""
生产环境系统指标收集器组件

负责收集系统指标，包括CPU、内存、磁盘、网络等信息。
"""

import os
import psutil
from datetime import datetime
from typing import Dict, Any, Optional


class ProductionSystemMetricsCollector:
    """生产环境系统指标收集器"""
    
    def __init__(self):
        """初始化指标收集器"""
        pass
    
    def collect_system_info(self) -> Dict[str, Any]:
        """收集系统基本信息"""
        try:
            return {
                'hostname': self._get_hostname(),
                'platform': self._get_platform(),
                'cpu_count': psutil.cpu_count(),
                'total_memory': self._get_total_memory_gb(),
                'total_disk': self._get_total_disk_gb(),
                'python_version': self._get_python_version()
            }
        except Exception as e:
            return {
                'error': f"Failed to collect system info: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }
    
    def collect_system_metrics(self) -> Dict[str, Any]:
        """收集系统指标"""
        try:
            return {
                'timestamp': datetime.now().isoformat(),
                'cpu': self._collect_cpu_metrics(),
                'memory': self._collect_memory_metrics(),
                'disk': self._collect_disk_metrics(),
                'network': self._collect_network_metrics(),
                'process': self._collect_process_metrics()
            }
        except Exception as e:
            return {
                'timestamp': datetime.now().isoformat(),
                'error': f"Failed to collect metrics: {str(e)}"
            }
    
    def _get_hostname(self) -> str:
        """获取主机名"""
        try:
            return os.uname().nodename if hasattr(os, 'uname') else 'unknown'
        except Exception:
            return 'unknown'
    
    def _get_platform(self) -> str:
        """获取平台信息"""
        try:
            return os.uname().sysname if hasattr(os, 'uname') else 'unknown'
        except Exception:
            return 'unknown'
    
    def _get_total_memory_gb(self) -> float:
        """获取总内存(GB)"""
        try:
            return psutil.virtual_memory().total / (1024**3)
        except Exception:
            return 0.0
    
    def _get_total_disk_gb(self) -> float:
        """获取总磁盘空间(GB)"""
        try:
            return psutil.disk_usage('/').total / (1024**3)
        except Exception:
            return 0.0
    
    def _get_python_version(self) -> str:
        """获取Python版本"""
        try:
            return f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
        except Exception:
            return "unknown"
    
    def _collect_cpu_metrics(self) -> Dict[str, Any]:
        """收集CPU指标"""
        try:
            return {
                'percent': psutil.cpu_percent(interval=1),
                'count': psutil.cpu_count()
            }
        except Exception as e:
            return {'percent': 0, 'count': 0, 'error': str(e)}
    
    def _collect_memory_metrics(self) -> Dict[str, Any]:
        """收集内存指标"""
        try:
            memory = psutil.virtual_memory()
            process = psutil.Process(os.getpid())
            process_memory = process.memory_info().rss / (1024**2)  # MB
            
            return {
                'percent': memory.percent,
                'used_mb': memory.used / (1024**2),
                'available_mb': memory.available / (1024**2),
                'process_mb': process_memory
            }
        except Exception as e:
            return {'percent': 0, 'error': str(e)}
    
    def _collect_disk_metrics(self) -> Dict[str, Any]:
        """收集磁盘指标"""
        try:
            disk = psutil.disk_usage('/')
            return {
                'percent': disk.percent,
                'used_gb': disk.used / (1024**3),
                'free_gb': disk.free / (1024**3)
            }
        except Exception as e:
            return {'percent': 0, 'error': str(e)}
    
    def _collect_network_metrics(self) -> Dict[str, Any]:
        """收集网络指标"""
        try:
            network = psutil.net_io_counters()
            return {
                'bytes_sent_mb': network.bytes_sent / (1024**2),
                'bytes_recv_mb': network.bytes_recv / (1024**2)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _collect_process_metrics(self) -> Dict[str, Any]:
        """收集进程指标"""
        try:
            process = psutil.Process(os.getpid())
            return {
                'cpu_percent': process.cpu_percent(interval=1),
                'memory_mb': process.memory_info().rss / (1024**2),
                'threads': process.num_threads()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def get_metrics_summary(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """获取指标摘要"""
        try:
            return {
                'timestamp': metrics.get('timestamp'),
                'cpu_usage': metrics.get('cpu', {}).get('percent', 0),
                'memory_usage': metrics.get('memory', {}).get('percent', 0),
                'disk_usage': metrics.get('disk', {}).get('percent', 0),
                'has_errors': 'error' in str(metrics)
            }
        except Exception:
            return {}

