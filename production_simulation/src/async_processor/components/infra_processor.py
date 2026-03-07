"""
Infrastructure Processor Module
基础设施处理器模块

This module provides infrastructure - level processing capabilities for async operations
此模块为异步操作提供基础设施级别的处理能力

Author: RQA2025 Development Team
Date: 2025 - 01 - 28
"""

import logging
from typing import Any, Dict, List
from datetime import datetime
from enum import Enum
import threading
import time
import psutil
import platform
import socket

logger = logging.getLogger(__name__)


class InfraComponent(Enum):

    """Infrastructure components"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    PROCESSES = "processes"
    SYSTEM = "system"


class InfrastructureProcessor:

    """
    Infrastructure Processor Class
    基础设施处理器类

    Provides infrastructure monitoring and processing capabilities
    提供基础设施监控和处理能力
    """

    def __init__(self, processor_name: str = "default_infra_processor"):
        """
        Initialize the infrastructure processor
        初始化基础设施处理器

        Args:
            processor_name: Name of this processor
                          此处理器的名称
        """
        self.processor_name = processor_name
        self.is_running = False
        self.monitoring_thread = None
        self.processed_data = {}
        self.alerts = []
        self.monitoring_interval = 30  # seconds

        # Thresholds for alerts
        self.thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_percent': 90.0,
            'network_errors': 10
        }

        logger.info(f"Infrastructure processor {processor_name} initialized")

    def start_monitoring(self) -> bool:
        """
        Start infrastructure monitoring
        开始基础设施监控

        Returns:
            bool: True if started successfully, False otherwise
                  启动成功返回True，否则返回False
        """
        if self.is_running:
            logger.warning(f"{self.processor_name} is already running")
            return False

        try:
            self.is_running = True
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            logger.info(f"Infrastructure monitoring started for {self.processor_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to start infrastructure monitoring: {str(e)}")
            self.is_running = False
            return False

    def stop_monitoring(self) -> bool:
        """
        Stop infrastructure monitoring
        停止基础设施监控

        Returns:
            bool: True if stopped successfully, False otherwise
                  停止成功返回True，否则返回False
        """
        if not self.is_running:
            logger.warning(f"{self.processor_name} is not running")
            return False

        try:
            self.is_running = False
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5.0)
            logger.info(f"Infrastructure monitoring stopped for {self.processor_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to stop infrastructure monitoring: {str(e)}")
            return False

    def collect_system_info(self) -> Dict[str, Any]:
        """
        Collect comprehensive system information
        收集全面的系统信息

        Returns:
            dict: System information data
                  系统信息数据
        """
        try:
            system_info = {
                'timestamp': datetime.now(),
                'hostname': socket.gethostname(),
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'cpu': self._get_cpu_info(),
                'memory': self._get_memory_info(),
                'disk': self._get_disk_info(),
                'network': self._get_network_info(),
                'processes': self._get_process_info()
            }

            return system_info

        except Exception as e:
            logger.error(f"Failed to collect system info: {str(e)}")
            return {}

    def _get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU information"""
        try:
            return {
                'physical_cores': psutil.cpu_count(logical=False),
                'logical_cores': psutil.cpu_count(logical=True),
                'usage_percent': psutil.cpu_percent(interval=0.1),
                'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {}
            }
        except Exception as e:
            logger.error(f"Failed to get CPU info: {str(e)}")
            return {}

    def _get_memory_info(self) -> Dict[str, Any]:
        """Get memory information"""
        try:
            memory = psutil.virtual_memory()
            return {
                'total_gb': memory.total / (1024 ** 3),
                'used_gb': memory.used / (1024 ** 3),
                'free_gb': memory.free / (1024 ** 3),
                'usage_percent': memory.percent
            }
        except Exception as e:
            logger.error(f"Failed to get memory info: {str(e)}")
            return {}

    def _get_disk_info(self) -> Dict[str, Any]:
        """Get disk information"""
        try:
            disk_usage = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()

            return {
                'total_gb': disk_usage.total / (1024 ** 3),
                'used_gb': disk_usage.used / (1024 ** 3),
                'free_gb': disk_usage.free / (1024 ** 3),
                'usage_percent': disk_usage.percent,
                'io_counters': disk_io._asdict() if disk_io else {}
            }
        except Exception as e:
            logger.error(f"Failed to get disk info: {str(e)}")
            return {}

    def _get_network_info(self) -> Dict[str, Any]:
        """Get network information"""
        try:
            net_io = psutil.net_io_counters()
            net_if_addrs = psutil.net_if_addrs()

            return {
                'io_counters': net_io._asdict() if net_io else {},
                'interfaces': {name: [addr._asdict() for addr in addrs]
                               for name, addrs in net_if_addrs.items()}
            }
        except Exception as e:
            logger.error(f"Failed to get network info: {str(e)}")
            return {}

    def _get_process_info(self) -> Dict[str, Any]:
        """Get process information"""
        try:
            current_process = psutil.Process()
            return {
                'pid': current_process.pid,
                'name': current_process.name(),
                'cpu_percent': current_process.cpu_percent(),
                'memory_info': current_process.memory_info()._asdict(),
                'num_threads': current_process.num_threads(),
                'num_fds': getattr(current_process, 'num_fds', lambda: 0)()
            }
        except Exception as e:
            logger.error(f"Failed to get process info: {str(e)}")
            return {}

    def check_thresholds(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check metrics against thresholds and generate alerts
        检查指标是否超过阈值并生成警报

        Args:
            metrics: System metrics data
                    系统指标数据

        Returns:
            list: List of alerts generated
                  生成的警报列表
        """
        alerts = []

        try:
            # CPU threshold check
            cpu_percent = metrics.get('cpu', {}).get('usage_percent', 0)
            if cpu_percent > self.thresholds['cpu_percent']:
                alerts.append({
                    'component': 'cpu',
                    'alert_type': 'high_usage',
                    'value': cpu_percent,
                    'threshold': self.thresholds['cpu_percent'],
                    'message': f"CPU usage {cpu_percent:.1f}% exceeds threshold {self.thresholds['cpu_percent']}%",
                    'timestamp': datetime.now()
                })

            # Memory threshold check
            memory_percent = metrics.get('memory', {}).get('usage_percent', 0)
            if memory_percent > self.thresholds['memory_percent']:
                alerts.append({
                    'component': 'memory',
                    'alert_type': 'high_usage',
                    'value': memory_percent,
                    'threshold': self.thresholds['memory_percent'],
                    'message': f"Memory usage {memory_percent:.1f}% exceeds threshold {self.thresholds['memory_percent']}%",
                    'timestamp': datetime.now()
                })

            # Disk threshold check
            disk_percent = metrics.get('disk', {}).get('usage_percent', 0)
            if disk_percent > self.thresholds['disk_percent']:
                alerts.append({
                    'component': 'disk',
                    'alert_type': 'high_usage',
                    'value': disk_percent,
                    'threshold': self.thresholds['disk_percent'],
                    'message': f"Disk usage {disk_percent:.1f}% exceeds threshold {self.thresholds['disk_percent']}%",
                    'timestamp': datetime.now()
                })

        except Exception as e:
            logger.error(f"Failed to check thresholds: {str(e)}")

        return alerts

    def process_infrastructure_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process infrastructure data and generate insights
        处理基础设施数据并生成洞察

        Args:
            data: Infrastructure data to process
                 要处理的基础设施数据

        Returns:
            dict: Processed data with insights
                  包含洞察的处理后数据
        """
        try:
            processed_data = {
                'original_data': data,
                'processed_at': datetime.now(),
                'insights': {},
                'recommendations': []
            }

            # Generate insights
            if 'cpu' in data:
                cpu_usage = data['cpu'].get('usage_percent', 0)
                if cpu_usage > 70:
                    processed_data['insights']['cpu'] = "High CPU utilization detected"
                    processed_data['recommendations'].append(
                        "Consider optimizing CPU - intensive operations")
                else:
                    processed_data['insights']['cpu'] = "CPU utilization is normal"

            if 'memory' in data:
                memory_usage = data['memory'].get('usage_percent', 0)
                if memory_usage > 80:
                    processed_data['insights']['memory'] = "High memory utilization detected"
                    processed_data['recommendations'].append(
                        "Consider memory optimization or increasing RAM")
                else:
                    processed_data['insights']['memory'] = "Memory utilization is normal"

            return processed_data

        except Exception as e:
            logger.error(f"Failed to process infrastructure data: {str(e)}")
            return {'error': str(e)}

    def _monitoring_loop(self) -> None:
        """
        Main monitoring loop
        主要的监控循环
        """
        logger.info(f"Infrastructure monitoring loop started for {self.processor_name}")

        while self.is_running:
            try:
                # Collect system information
                system_info = self.collect_system_info()

                # Check thresholds and generate alerts
                alerts = self.check_thresholds(system_info)
                self.alerts.extend(alerts)

                # Process the data
                processed_data = self.process_infrastructure_data(system_info)
                self.processed_data[datetime.now()] = processed_data

                # Log alerts
                for alert in alerts:
                    logger.warning(f"Infrastructure alert: {alert['message']}")

                # Clean up old data (keep last 100 entries)
                if len(self.processed_data) > 100:
                    oldest_keys = sorted(list(self.processed_data.keys()))[:-100]
                    for key in oldest_keys:
                        del self.processed_data[key]

                # Clean up old alerts (keep last 50 alerts)
                if len(self.alerts) > 50:
                    self.alerts = self.alerts[-50:]

                # Wait before next monitoring cycle
                time.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error(f"Infrastructure monitoring loop error: {str(e)}")
                time.sleep(self.monitoring_interval)

        logger.info(f"Infrastructure monitoring loop stopped for {self.processor_name}")

    def get_infrastructure_status(self) -> Dict[str, Any]:
        """
        Get current infrastructure status
        获取当前基础设施状态

        Returns:
            dict: Infrastructure status information
                  基础设施状态信息
        """
        return {
            'processor_name': self.processor_name,
            'is_running': self.is_running,
            'last_system_info': self.collect_system_info(),
            'active_alerts': len([a for a in self.alerts
                                 if (datetime.now() - a['timestamp']).seconds < 300]),  # Last 5 minutes
            'total_alerts': len(self.alerts),
            'monitoring_interval': self.monitoring_interval,
            'thresholds': self.thresholds
        }

    def set_thresholds(self, component: str, value: float) -> bool:
        """
        Set threshold for a specific component
        为特定组件设置阈值

        Args:
            component: Component name (cpu, memory, disk)
                      组件名称 (cpu, memory, disk)
            value: Threshold value
                  阈值

        Returns:
            bool: True if set successfully, False otherwise
                  设置成功返回True，否则返回False
        """
        if component in self.thresholds:
            self.thresholds[component] = value
            logger.info(f"Threshold for {component} set to {value}")
            return True
        return False


# Global infrastructure processor instance
# 全局基础设施处理器实例
infra_processor = InfrastructureProcessor()

__all__ = ['InfraComponent', 'InfrastructureProcessor', 'infra_processor']
