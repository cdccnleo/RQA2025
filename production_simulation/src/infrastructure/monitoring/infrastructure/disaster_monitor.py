
import psutil
import threading
import time

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Callable
#!/usr/bin/env python3
"""
RQA2025 灾难监控组件

提供系统灾难检测和恢复监控功能。
"""


class DisasterLevel(Enum):
    """灾难级别"""
    NONE = "none"
    WARNING = "warning"
    CRITICAL = "critical"
    DISASTER = "disaster"


class DisasterType(Enum):
    """灾难类型"""
    MEMORY_OVERLOAD = "memory_overload"
    CPU_OVERLOAD = "cpu_overload"
    DISK_FULL = "disk_full"
    NETWORK_FAILURE = "network_failure"
    PROCESS_CRASH = "process_crash"
    SERVICE_UNAVAILABLE = "service_unavailable"


@dataclass
class DisasterEvent:
    """灾难事件"""
    disaster_type: DisasterType
    level: DisasterLevel
    message: str
    timestamp: float
    details: Dict[str, Any]


class DisasterMonitor:
    """灾难监控器"""

    def __init__(self):
        self.disaster_events: List[DisasterEvent] = []
        self.max_events = 1000
        self.thresholds = {
            'memory_percent': 90.0,
            'cpu_percent': 95.0,
            'disk_percent': 95.0
        }
        self.handlers: List[Callable[[DisasterEvent], None]] = []
        self._lock = threading.RLock()
        self.monitoring_active = False

    def add_handler(self, handler: Callable[[DisasterEvent], None]) -> None:
        """
        添加灾难事件处理器

        Args:
            handler: 事件处理函数
        """
        with self._lock:
            self.handlers.append(handler)

    def remove_handler(self, handler: Callable[[DisasterEvent], None]) -> None:
        """
        移除灾难事件处理器

        Args:
            handler: 事件处理函数
        """
        with self._lock:
            if handler in self.handlers:
                self.handlers.remove(handler)

    def check_system_health(self) -> Dict[str, Any]:
        """
        检查系统健康状态

        Returns:
            Dict[str, Any]: 系统健康状态
        """
        try:
            # 收集系统指标
            system_metrics = self._collect_system_metrics()
            
            # 检查灾难条件
            disasters = self._check_disaster_conditions(system_metrics)
            
            # 构建健康状态报告
            health_status = {
                'cpu_percent': system_metrics['cpu_percent'],
                'memory_percent': system_metrics['memory_percent'],
                'disk_percent': system_metrics['disk_percent'],
                'timestamp': time.time(),
                'disasters': disasters
            }

            return health_status

        except Exception as e:
            return {
                'error': str(e),
                'timestamp': time.time(),
                'disasters': []
            }
    
    def _collect_system_metrics(self) -> Dict[str, float]:
        """收集系统指标"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'disk_percent': disk.percent
        }
    
    def _check_disaster_conditions(self, metrics: Dict[str, float]) -> List[DisasterEvent]:
        """检查灾难条件"""
        disasters = []
        
        # 检查内存使用率
        if metrics['memory_percent'] > self.thresholds['memory_percent']:
            disasters.append(self._create_memory_disaster(metrics['memory_percent']))
        
        # 检查CPU使用率
        if metrics['cpu_percent'] > self.thresholds['cpu_percent']:
            disasters.append(self._create_cpu_disaster(metrics['cpu_percent']))
        
        # 检查磁盘使用率
        if metrics['disk_percent'] > self.thresholds['disk_percent']:
            disasters.append(self._create_disk_disaster(metrics['disk_percent']))
        
        # 记录所有灾难事件
        for disaster in disasters:
            self._record_disaster(disaster)
        
        return disasters
    
    def _create_memory_disaster(self, memory_percent: float) -> DisasterEvent:
        """创建内存灾难事件"""
        return DisasterEvent(
            disaster_type=DisasterType.MEMORY_OVERLOAD,
            level=DisasterLevel.CRITICAL,
            message=f"内存使用率过高: {memory_percent:.1f}%",
            timestamp=time.time(),
            details={'memory_percent': memory_percent}
        )
    
    def _create_cpu_disaster(self, cpu_percent: float) -> DisasterEvent:
        """创建CPU灾难事件"""
        return DisasterEvent(
            disaster_type=DisasterType.CPU_OVERLOAD,
            level=DisasterLevel.CRITICAL,
            message=f"CPU使用率过高: {cpu_percent:.1f}%",
            timestamp=time.time(),
            details={'cpu_percent': cpu_percent}
        )
    
    def _create_disk_disaster(self, disk_percent: float) -> DisasterEvent:
        """创建磁盘灾难事件"""
        return DisasterEvent(
            disaster_type=DisasterType.DISK_FULL,
            level=DisasterLevel.CRITICAL,
            message=f"磁盘使用率过高: {disk_percent:.1f}%",
            timestamp=time.time(),
            details={'disk_percent': disk_percent}
        )

    def _record_disaster(self, disaster: DisasterEvent) -> None:
        """
        记录灾难事件

        Args:
            disaster: 灾难事件
        """
        with self._lock:
            self.disaster_events.append(disaster)
            if len(self.disaster_events) > self.max_events:
                self.disaster_events.pop(0)

            # 触发处理器
            for handler in self.handlers:
                try:
                    handler(disaster)
                except Exception as e:
                    print(f"灾难处理器执行失败: {e}")

    def get_recent_disasters(self, limit: int = 50) -> List[DisasterEvent]:
        """
        获取最近的灾难事件

        Args:
            limit: 返回的最大事件数

        Returns:
            List[DisasterEvent]: 灾难事件列表
        """
        with self._lock:
            return self.disaster_events[-limit:]

    def get_disaster_stats(self) -> Dict[str, Any]:
        """
        获取灾难统计信息

        Returns:
            Dict[str, Any]: 灾难统计
        """
        with self._lock:
            if not self.disaster_events:
                return {
                    'total_disasters': 0,
                    'disasters_by_type': {},
                    'disasters_by_level': {},
                    'latest_disaster': None
                }

            disasters_by_type = {}
            disasters_by_level = {}

            for event in self.disaster_events:
                disasters_by_type[event.disaster_type.value] = \
                    disasters_by_type.get(event.disaster_type.value, 0) + 1
                disasters_by_level[event.level.value] = \
                    disasters_by_level.get(event.level.value, 0) + 1

            return {
                'total_disasters': len(self.disaster_events),
                'disasters_by_type': disasters_by_type,
                'disasters_by_level': disasters_by_level,
                'latest_disaster': self.disaster_events[-1] if self.disaster_events else None
            }

    def clear_disasters(self) -> None:
        """清空灾难记录"""
        with self._lock:
            self.disaster_events.clear()

    def set_threshold(self, metric: str, value: float) -> None:
        """
        设置阈值

        Args:
            metric: 指标名称
            value: 阈值
        """
        self.thresholds[metric] = value
