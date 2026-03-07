"""
resource_manager 模块

提供 resource_manager 相关功能和接口。
"""

import logging

import psutil
import threading
import time

from ..config.config_classes import ResourceMonitorConfig
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 Resource Manager
资源管理器 - 管理系统资源分配和监控
"""

logger = logging.getLogger(__name__)


class CoreResourceManager:
    """
    资源管理器

    使用配置驱动的方式管理系统资源分配和监控
    """

    def __init__(self, config: Optional[ResourceMonitorConfig] = None):
        """
        初始化资源管理器

        Args:
            config: 资源监控配置
        """
        self.config = config or ResourceMonitorConfig()
        self.logger = logging.getLogger(__name__)

        # 初始化监控状态
        self._monitoring = True
        self._monitor_thread = None
        self._resource_history = []
        self._lock = threading.Lock()

        # 启动监控线程
        self._start_monitoring()

    def start_monitoring(self):
        """启动资源监控（公共方法）"""
        self._start_monitoring()

    def _start_monitoring(self):
        """启动资源监控"""
        if self._monitor_thread is None:
            self._monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
            self._monitor_thread.start()
        self._monitoring = True

    def _monitor_resources(self):
        """监控资源使用情况（后台线程）"""
        while self._monitoring:
            try:
                self._collect_and_store_resource_info()
            except Exception as e:
                self.logger.error(f"资源监控错误: {e}")

            time.sleep(self.config.monitor_interval)

    def _collect_and_store_resource_info(self):
        """收集并存储一次资源信息"""
        # 根据配置收集资源信息
        resource_info = self._collect_resource_info()
        
        # 使用高精度时间戳确保唯一性
        import time as time_module
        timestamp = datetime.now()
        # 添加微秒级延迟确保时间戳不同
        time_module.sleep(0.001)
        resource_info['timestamp'] = timestamp.isoformat()

        with self._lock:
            self._resource_history.append(resource_info)
            # 根据配置保留历史记录
            if len(self._resource_history) > self.config.history_size:
                self._resource_history = self._resource_history[-self.config.history_size:]

    def _collect_resource_info(self) -> Dict[str, Any]:
        """
        收集资源信息

        根据配置决定收集哪些资源指标
        """
        resource_info = {}

        try:
            # CPU监控
            if self.config.enable_cpu_monitoring:
                # 使用非阻塞模式获取CPU使用率（interval=None 返回上次调用后的平均值）
                # 首次调用返回0，后续返回自上次调用以来的平均值
                cpu_percent = psutil.cpu_percent(interval=None)
                resource_info['cpu_percent'] = round(cpu_percent, self.config.precision)

            # 内存监控
            if self.config.enable_memory_monitoring:
                memory = psutil.virtual_memory()
                resource_info.update({
                    'memory_percent': round(memory.percent, self.config.precision),
                    'memory_used_gb': round(memory.used / (1024**3), self.config.precision),
                    'memory_total_gb': round(memory.total / (1024**3), self.config.precision)
                })

            # 磁盘监控
            if self.config.enable_disk_monitoring:
                disk = psutil.disk_usage('/')
                resource_info.update({
                    'disk_percent': round(disk.percent, self.config.precision),
                    'disk_used_gb': round(disk.used / (1024**3), self.config.precision),
                    'disk_total_gb': round(disk.total / (1024**3), self.config.precision)
                })

        except Exception as e:
            self.logger.error(f"收集资源信息失败: {e}")

        return resource_info

    def get_current_usage(self) -> Dict[str, Any]:
        """
        获取当前资源使用情况

        使用配置驱动的方式收集实时资源信息
        """
        try:
            # 使用相同的资源收集逻辑
            current_info = self._collect_resource_info()
            current_info['timestamp'] = datetime.now().isoformat()

            # 添加 disk_usage 字段以保持向后兼容
            if 'disk_percent' in current_info:
                current_info['disk_usage'] = {
                    'percent': current_info['disk_percent'],
                    'used_gb': current_info.get('disk_used_gb', 0),
                    'total_gb': current_info.get('disk_total_gb', 0)
                }

            # 添加健康状态检查
            current_info.update(self._get_health_status(current_info))

            return current_info

        except Exception as e:
            self.logger.error(f"获取资源使用情况失败: {e}")
            return self._create_error_response()

    def _get_health_status(self, resource_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据配置的阈值获取健康状态

        Args:
            resource_info: 资源使用信息

        Returns:
            Dict: 健康状态信息
        """
        health_status = {
            'overall_health': 'healthy',
            'warnings': [],
            'alerts': []
        }

        # 检查CPU使用率
        if self.config.enable_cpu_monitoring and 'cpu_percent' in resource_info:
            cpu_usage = resource_info['cpu_percent']
            if cpu_usage >= self.config.thresholds['cpu_warning']:
                health_status['warnings'].append(f'CPU使用率过高: {cpu_usage}%')
                health_status['overall_health'] = 'warning'

        # 检查内存使用率
        if self.config.enable_memory_monitoring and 'memory_percent' in resource_info:
            memory_usage = resource_info['memory_percent']
            if memory_usage >= self.config.thresholds['memory_warning']:
                health_status['alerts'].append(f'内存使用率过高: {memory_usage}%')
                health_status['overall_health'] = 'critical'

        # 检查磁盘使用率
        if self.config.enable_disk_monitoring and 'disk_percent' in resource_info:
            disk_usage = resource_info['disk_percent']
            if disk_usage >= self.config.thresholds['disk_warning']:
                health_status['warnings'].append(f'磁盘使用率过高: {disk_usage}%')
                if health_status['overall_health'] == 'healthy':
                    health_status['overall_health'] = 'warning'

        return health_status

    def _create_error_response(self) -> Dict[str, Any]:
        """创建错误响应"""
        return {
            'cpu_percent': 0.0,
            'memory_percent': 0.0,
            'memory_used_gb': 0.0,
            'memory_total_gb': 0.0,
            'disk_percent': 0.0,
            'disk_used_gb': 0.0,
            'disk_total_gb': 0.0,
            'timestamp': datetime.now().isoformat(),
            'overall_health': 'unknown',
            'error': '无法获取资源信息'
        }

    def get_usage_history(self, hours: int = 1) -> Dict[str, Any]:
        """获取资源使用历史"""
        with self._lock:
            # 获取指定时间范围内的历史数据
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_history = [
                record for record in self._resource_history
                if datetime.fromisoformat(record['timestamp']) > cutoff_time
            ]

            return {
                'history': recent_history,
                'count': len(recent_history),
                'time_range_hours': hours
            }

    # 兼容性方法（向后兼容测试）
    def get_resource_usage(self) -> Dict[str, Any]:
        """获取资源使用情况（兼容性方法）"""
        return self.get_current_usage()

    def get_resource_history(self, limit: Optional[int] = None) -> list:
        """
        获取资源历史记录（兼容性方法）
        
        Args:
            limit: 限制返回的记录数量
        """
        with self._lock:
            history = list(self._resource_history)
            if limit is not None and limit > 0:
                return history[-limit:]
            return history

    def get_cpu_usage(self) -> float:
        """获取CPU使用率"""
        try:
            return psutil.cpu_percent(interval=None)
        except Exception as e:
            self.logger.error(f"获取CPU使用率失败: {e}")
            return 0.0

    def get_memory_usage(self) -> Dict[str, Any]:
        """获取内存使用情况"""
        try:
            memory = psutil.virtual_memory()
            return {
                'total': memory.total,
                'used': memory.used,
                'free': memory.available,
                'percent': memory.percent
            }
        except Exception as e:
            self.logger.error(f"获取内存使用情况失败: {e}")
            return {'total': 0, 'used': 0, 'free': 0, 'percent': 0}

    def get_disk_usage(self, path: str = '/') -> Dict[str, Any]:
        """获取磁盘使用情况"""
        try:
            disk = psutil.disk_usage(path)
            return {
                'total': disk.total,
                'used': disk.used,
                'free': disk.free,
                'percent': disk.percent
            }
        except Exception as e:
            self.logger.error(f"获取磁盘使用情况失败: {e}")
            return {'total': 0, 'used': 0, 'free': 0, 'percent': 0}

    def get_resource_summary(self) -> Dict[str, Any]:
        """获取资源摘要"""
        return {
            'current_usage': self.get_current_usage(),
            'history_count': len(self._resource_history),
            'alerts': self._check_alerts() if hasattr(self, '_check_alerts') else []
        }

    def _check_alerts(self) -> list:
        """检查告警"""
        alerts = []
        try:
            current_usage = self.get_current_usage()
            
            # 检查CPU告警
            if 'cpu_percent' in current_usage:
                if current_usage['cpu_percent'] >= self.config.alert_threshold.get('cpu', 90.0):
                    alerts.append(f"CPU使用率过高: {current_usage['cpu_percent']}%")
            
            # 检查内存告警
            if 'memory_percent' in current_usage:
                if current_usage['memory_percent'] >= self.config.alert_threshold.get('memory', 85.0):
                    alerts.append(f"内存使用率过高: {current_usage['memory_percent']}%")
            
            # 检查磁盘告警
            if 'disk_percent' in current_usage:
                if current_usage['disk_percent'] >= self.config.alert_threshold.get('disk', 80.0):
                    alerts.append(f"磁盘使用率过高: {current_usage['disk_percent']}%")
        
        except Exception as e:
            self.logger.error(f"检查告警失败: {e}")
        
        return alerts

    def clear_alerts(self) -> None:
        """清除所有告警"""
        # CoreResourceManager不存储持久告警，这里只是兼容性方法
        pass

    def update_resource_stats(self, stats: Dict[str, Any]) -> None:
        """更新资源统计信息"""
        # 在CoreResourceManager中，统计信息通过监控自动更新
        # 这里提供手动更新接口用于测试
        if hasattr(self, '_last_stats'):
            self._last_stats.update(stats)
        else:
            self._last_stats = stats.copy()

    def _check_cpu_threshold(self, cpu_percent: float) -> bool:
        """检查CPU是否超过阈值"""
        return cpu_percent >= self.config.alert_threshold.get('cpu', 90.0)

    def _check_memory_threshold(self, memory_percent: float) -> bool:
        """检查内存是否超过阈值"""
        return memory_percent >= self.config.alert_threshold.get('memory', 85.0)

    def get_resource_limits(self) -> Dict[str, Any]:
        """获取资源限制"""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        return {
            'cpu_cores': psutil.cpu_count(),
            'cpu_logical': psutil.cpu_count(logical=True),
            'memory_total_gb': round(memory.total / (1024**3), 2),
            'disk_total_gb': round(disk.total / (1024**3), 2),
            'memory_limit_percent': 90.0,  # 内存使用率上限
            'cpu_limit_percent': 95.0     # CPU使用率上限
        }

    def check_resource_health(self) -> Dict[str, Any]:
        """检查资源健康状态"""
        current_usage = self.get_current_usage()
        limits = self.get_resource_limits()

        health_status = {
            'overall_health': 'healthy',
            'issues': [],
            'recommendations': []
        }

        # 检查CPU使用率
        if current_usage['cpu_percent'] > limits['cpu_limit_percent']:
            health_status['issues'].append(f"CPU使用率过高: {current_usage['cpu_percent']}%")
            health_status['recommendations'].append("考虑增加CPU资源或优化CPU密集型任务")

        # 检查内存使用率
        if current_usage['memory_percent'] > limits['memory_limit_percent']:
            health_status['issues'].append(f"内存使用率过高: {current_usage['memory_percent']}%")
            health_status['recommendations'].append("考虑增加内存资源或优化内存使用")

        # 检查磁盘使用率
        if current_usage['disk_percent'] > 90.0:
            health_status['issues'].append(f"磁盘使用率过高: {current_usage['disk_percent']}%")
            health_status['recommendations'].append("考虑清理磁盘空间或增加存储资源")

        if health_status['issues']:
            health_status['overall_health'] = 'warning'

        return health_status

    def stop_monitoring(self):
        """停止资源监控"""
        self._monitoring = False
        # 只有当监控线程存在且不是当前线程时才join
        if (self._monitor_thread and 
            self._monitor_thread.is_alive() and 
            self._monitor_thread != threading.current_thread()):
            self._monitor_thread.join(timeout=5.0)

    def __del__(self):
        """析构函数"""
        try:
            self.stop_monitoring()
        except Exception:
            # 忽略析构时的任何错误
            pass
