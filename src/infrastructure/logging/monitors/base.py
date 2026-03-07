
from ..core.exceptions import LogMonitorError
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
"""
基础设施层 - 日志监控器基础实现

定义日志监控器的基础接口和实现。
"""


class ILogMonitor(ABC):
    """日志监控器接口"""

    @abstractmethod
    def check_health(self) -> Dict[str, Any]:
        """检查健康状态"""

    @abstractmethod
    def collect_metrics(self) -> Dict[str, Any]:
        """收集指标"""

    @abstractmethod
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """检测异常"""

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """获取监控器状态"""


class BaseMonitor(ILogMonitor):
    """基础日志监控器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化基础监控器

        Args:
            config: 监控器配置
        """
        self.config = config or {}
        self.name = self.config.get('name', self.__class__.__name__)
        self.enabled = self.config.get('enabled', True)
        
        # 验证并设置 interval
        configured_interval = self.config.get('interval', 60)
        self.interval = configured_interval if configured_interval > 0 else 60
        
        # 验证并设置 retention_days
        configured_retention = self.config.get('retention_days', 7)
        self.retention_days = configured_retention if configured_retention > 0 else 7

        # 同步字段值到配置字典以保持一致性
        self.config['interval'] = self.interval
        self.config['retention_days'] = self.retention_days

        # 状态数据
        self.last_check_time = None
        self.metrics_history: List[Dict[str, Any]] = []
        self.anomalies: List[Dict[str, Any]] = []
        self.health_status = 'unknown'

    def check_health(self) -> Dict[str, Any]:
        """检查健康状态"""
        if not self.enabled:
            return {'status': 'disabled', 'enabled': False}

        try:
            health_data = self._check_health()
            self.health_status = health_data.get('status', 'unknown')
            self.last_check_time = datetime.now()

            return health_data
        except Exception as e:
            self.health_status = 'error'
            raise LogMonitorError(f"Health check failed: {e}")

    def collect_metrics(self) -> Dict[str, Any]:
        """收集指标"""
        if not self.enabled:
            return {}

        try:
            metrics = self._collect_metrics()

            # 添加时间戳
            metrics['timestamp'] = datetime.now().isoformat()
            metrics['monitor_name'] = self.name

            # 保存到历史记录
            self.metrics_history.append(metrics)

            # 清理过期数据
            self._cleanup_old_data()

            return metrics
        except Exception as e:
            raise LogMonitorError(f"Metrics collection failed: {e}")

    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """检测异常"""
        if not self.enabled:
            return []

        try:
            new_anomalies = self._detect_anomalies()

            # 添加到异常列表
            for anomaly in new_anomalies:
                anomaly['timestamp'] = datetime.now().isoformat()
                anomaly['monitor_name'] = self.name
                self.anomalies.append(anomaly)

            # 只返回新检测到的异常
            return new_anomalies
        except Exception as e:
            raise LogMonitorError(f"Anomaly detection failed: {e}")

    def get_status(self) -> Dict[str, Any]:
        """获取监控器状态"""
        status = {
            'name': self.name,
            'enabled': self.enabled,
            'health_status': self.health_status,
            'interval': self.interval,
            'retention_days': self.retention_days,
            'metrics_count': len(self.metrics_history),
            'anomalies_count': len(self.anomalies),
            'type': self.__class__.__name__
        }
        if self.last_check_time is not None:
            status['last_check_time'] = self.last_check_time.isoformat()
        return status

    def _cleanup_old_data(self) -> None:
        """清理过期数据"""
        cutoff_time = datetime.now() - timedelta(days=self.retention_days)

        # 清理指标历史
        self.metrics_history = [
            m for m in self.metrics_history
            if datetime.fromisoformat(m['timestamp']) > cutoff_time
        ]

        # 清理异常记录
        self.anomalies = [
            a for a in self.anomalies
            if datetime.fromisoformat(a['timestamp']) > cutoff_time
        ]

    # 子类需要实现的抽象方法
    @abstractmethod
    def _check_health(self) -> Dict[str, Any]:
        """实际的健康检查逻辑"""

    @abstractmethod
    def _collect_metrics(self) -> Dict[str, Any]:
        """实际的指标收集逻辑"""

    @abstractmethod
    def _detect_anomalies(self) -> List[Dict[str, Any]]:
        """实际的异常检测逻辑"""
