
import psutil
import threading

from .shared_interfaces import ILogger, IErrorHandler, StandardLogger, BaseErrorHandler
from datetime import datetime
from typing import Dict, Optional, Any
"""
系统资源分析器

Phase 3: 质量提升 - 文件拆分优化

负责分析系统资源状态，包括CPU、内存、线程和I/O资源。
"""


class SystemResourceAnalyzer:
    """系统资源分析器"""

    def __init__(self, logger: Optional[ILogger] = None,
                 error_handler: Optional[IErrorHandler] = None):

        self.logger = logger or StandardLogger(self.__class__.__name__)
        self.error_handler = error_handler or BaseErrorHandler()

    def get_system_resources(self, analysis_depth: str = "basic") -> Dict[str, Any]:
        """获取系统资源状态"""
        try:
            resources = {
                "timestamp": datetime.now().isoformat(),
                "cpu": self._get_cpu_resources(analysis_depth),
                "memory": self._get_memory_resources(analysis_depth),
                "threads": self._get_thread_resources(analysis_depth),
                "io": self._get_io_resources(analysis_depth)
            }

            return resources

        except Exception as e:
            self.error_handler.handle_error(e, {"context": "获取系统资源状态失败"})
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    def _get_cpu_resources(self, depth: str) -> Dict[str, Any]:
        """获取CPU资源信息"""
        try:
            cpu_info = {
                "usage_percent": psutil.cpu_percent(interval=1),
                "count": psutil.cpu_count(),
                "count_logical": psutil.cpu_count(logical=True)
            }

            if depth in ["detailed", "comprehensive"]:
                cpu_freq = psutil.cpu_freq()
                if cpu_freq:
                    if hasattr(cpu_freq, 'current'):
                        # cpu_freq返回对象的情况
                        cpu_info["frequency"] = {
                            "current": cpu_freq.current,
                            "min": cpu_freq.min,
                            "max": cpu_freq.max
                        }
                    elif isinstance(cpu_freq, list) and len(cpu_freq) > 0:
                        # cpu_freq返回列表的情况
                        freq_info = cpu_freq[0] if hasattr(cpu_freq[0], 'current') else None
                        if freq_info:
                            cpu_info["frequency"] = {
                                "current": freq_info.current,
                                "min": freq_info.min,
                                "max": freq_info.max
                            }
                        else:
                            cpu_info["frequency"] = None
                    else:
                        cpu_info["frequency"] = None
                else:
                    cpu_info["frequency"] = None

                # CPU使用率历史（最近5次测量）
                cpu_info["usage_history"] = []
                for _ in range(5):
                    cpu_info["usage_history"].append(psutil.cpu_percent(interval=0.1))

            return cpu_info

        except Exception as e:
            self.error_handler.handle_error(e, {"context": "获取CPU资源信息失败"})
            return {"error": str(e)}

    def _get_memory_resources(self, depth: str) -> Dict[str, Any]:
        """获取内存资源信息"""
        try:
            memory = psutil.virtual_memory()
            memory_info = {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "usage_percent": memory.percent,
                "free": memory.free,
                "total_gb": memory.total / (1024**3),
                "used_gb": memory.used / (1024**3),
                "free_gb": memory.free / (1024**3),
                "available_gb": memory.available / (1024**3)
            }

            if depth in ["detailed", "comprehensive"]:
                swap = psutil.swap_memory()
                memory_info["swap"] = {
                    "total": swap.total,
                    "used": swap.used,
                    "free": swap.free,
                    "usage_percent": swap.percent
                }
                # 为了向后兼容，也在顶层提供swap相关字段
                memory_info["swap_percent"] = swap.percent
                try:
                    memory_info["swap_total_gb"] = swap.total / (1024**3)
                    memory_info["swap_used_gb"] = swap.used / (1024**3)
                    memory_info["swap_free_gb"] = swap.free / (1024**3)
                except (TypeError, AttributeError):
                    # 处理Mock对象的情况
                    memory_info["swap_total_gb"] = 0
                    memory_info["swap_used_gb"] = 0
                    memory_info["swap_free_gb"] = 0

            return memory_info

        except Exception as e:
            self.error_handler.handle_error(e, {"context": "获取内存资源信息失败"})
            return {"error": str(e)}

    def _get_thread_resources(self, depth: str) -> Dict[str, Any]:
        """获取线程资源信息"""
        try:
            current_process = psutil.Process()
            threads = current_process.threads()

            thread_info = {
                "process_thread_count": len(threads),
                "system_thread_count": threading.active_count()
            }

            if depth in ["detailed", "comprehensive"]:
                thread_info["thread_details"] = []
                for thread in threads[:10]:  # 只获取前10个线程的详细信息
                    thread_info["thread_details"].append({
                        "id": thread.id,
                        "user_time": thread.user_time,
                        "system_time": thread.system_time
                    })

            return thread_info

        except Exception as e:
            self.error_handler.handle_error(e, {"context": "获取线程资源信息失败"})
            return {"error": str(e)}

    def _get_io_resources(self, depth: str) -> Dict[str, Any]:
        """获取I/O资源信息"""
        try:
            io_counters = psutil.disk_io_counters()
            net_counters = psutil.net_io_counters()

            io_info = {
                "disk": {
                    "read_count": io_counters.read_count if io_counters else 0,
                    "write_count": io_counters.write_count if io_counters else 0,
                    "read_bytes": io_counters.read_bytes if io_counters else 0,
                    "write_bytes": io_counters.write_bytes if io_counters else 0
                },
                "network": {
                    "bytes_sent": net_counters.bytes_sent if net_counters else 0,
                    "bytes_recv": net_counters.bytes_recv if net_counters else 0,
                    "packets_sent": net_counters.packets_sent if net_counters else 0,
                    "packets_recv": net_counters.packets_recv if net_counters else 0
                }
            }

            if depth in ["detailed", "comprehensive"]:
                # 获取磁盘分区信息
                disk_partitions = []
                for partition in psutil.disk_partitions():
                    try:
                        usage = psutil.disk_usage(partition.mountpoint)
                        disk_partitions.append({
                            "device": partition.device,
                            "mountpoint": partition.mountpoint,
                            "fstype": partition.fstype,
                            "total": usage.total,
                            "used": usage.used,
                            "free": usage.free,
                            "usage_percent": usage.percent
                        })
                    except Exception as e:
                        continue

                io_info["disk_partitions"] = disk_partitions[:5]  # 只保留前5个分区

            return io_info

        except Exception as e:
            self.error_handler.handle_error(e, {"context": "获取I/O资源信息失败"})
            return {"error": str(e)}

    def get_resource_summary(self) -> Dict[str, Any]:
        """获取资源汇总信息"""
        resources = self.get_system_resources("basic")

        if "error" in resources:
            return resources

        return {
            "cpu_usage": resources["cpu"].get("usage_percent", 0),
            "memory_usage": resources["memory"].get("usage_percent", 0),
            "thread_count": resources["threads"].get("process_thread_count", 0),
            "disk_read_bytes": resources["io"]["disk"].get("read_bytes", 0),
            "disk_write_bytes": resources["io"]["disk"].get("write_bytes", 0),
            "network_bytes_sent": resources["io"]["network"].get("bytes_sent", 0),
            "network_bytes_recv": resources["io"]["network"].get("bytes_recv", 0),
            "timestamp": resources["timestamp"]
        }
