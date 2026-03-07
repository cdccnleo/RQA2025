#!/usr/bin/env python3
"""
数据采集质量监控和告警系统

监控数据采集的各项指标，提供质量评估和异常告警功能。
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertType(Enum):
    """告警类型"""
    DATA_COLLECTION_FAILED = "data_collection_failed"
    DATA_QUALITY_LOW = "data_quality_low"
    API_RATE_LIMIT = "api_rate_limit"
    NETWORK_ERROR = "network_error"
    DATA_STALE = "data_stale"
    SYSTEM_TIME_DRIFT = "system_time_drift"

@dataclass
class Alert:
    """告警信息"""
    alert_id: str
    alert_type: AlertType
    level: AlertLevel
    message: str
    source_id: str
    timestamp: float
    details: Dict[str, Any]
    resolved: bool = False
    resolved_at: Optional[float] = None

@dataclass
class DataCollectionMetrics:
    """数据采集指标"""
    source_id: str
    collection_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_records: int = 0
    avg_collection_time: float = 0.0
    last_collection_time: Optional[float] = None
    last_success_time: Optional[float] = None
    consecutive_failures: int = 0
    data_quality_score: float = 0.0
    api_call_count: int = 0
    network_error_count: int = 0
    rate_limit_count: int = 0

class DataCollectionMonitor:
    """
    数据采集监控器

    提供以下功能：
    - 实时指标收集和计算
    - 质量评估和阈值监控
    - 异常检测和告警生成
    - 监控报告生成
    """

    def __init__(self, alert_callback: Optional[Callable[[Alert], None]] = None):
        """
        初始化监控器

        Args:
            alert_callback: 告警回调函数
        """
        self.alert_callback = alert_callback
        self.metrics: Dict[str, DataCollectionMetrics] = {}
        self.alerts: List[Alert] = []
        self.alert_thresholds = {
            'max_consecutive_failures': 3,
            'min_success_rate': 0.8,  # 80%
            'min_data_quality_score': 0.7,  # 70%
            'max_collection_time': 300,  # 5分钟
            'data_stale_threshold': 3600,  # 1小时
            'max_network_errors': 5,
            'max_rate_limits': 3
        }
        self.monitoring_enabled = True

    def record_collection_attempt(self, source_id: str, success: bool,
                                collection_time: float, record_count: int = 0,
                                error_type: Optional[str] = None):
        """
        记录数据采集尝试

        Args:
            source_id: 数据源ID
            success: 是否成功
            collection_time: 采集耗时（秒）
            record_count: 采集到的记录数
            error_type: 错误类型（网络错误、API限制等）
        """
        if not self.monitoring_enabled:
            return

        if source_id not in self.metrics:
            self.metrics[source_id] = DataCollectionMetrics(source_id=source_id)

        metrics = self.metrics[source_id]
        metrics.collection_count += 1
        metrics.last_collection_time = time.time()

        if success:
            metrics.success_count += 1
            metrics.last_success_time = time.time()
            metrics.consecutive_failures = 0
            metrics.total_records += record_count

            # 更新平均采集时间
            if metrics.collection_count == 1:
                metrics.avg_collection_time = collection_time
            else:
                metrics.avg_collection_time = (
                    (metrics.avg_collection_time * (metrics.collection_count - 1)) + collection_time
                ) / metrics.collection_count

        else:
            metrics.failure_count += 1
            metrics.consecutive_failures += 1

            # 记录错误类型
            if error_type:
                if 'network' in error_type.lower() or 'connection' in error_type.lower():
                    metrics.network_error_count += 1
                elif 'rate' in error_type.lower() or 'limit' in error_type.lower():
                    metrics.rate_limit_count += 1

        # 检查告警条件
        self._check_alerts(source_id)

    def update_data_quality(self, source_id: str, quality_score: float):
        """
        更新数据质量评分

        Args:
            source_id: 数据源ID
            quality_score: 质量评分（0.0-1.0）
        """
        if source_id not in self.metrics:
            self.metrics[source_id] = DataCollectionMetrics(source_id=source_id)

        self.metrics[source_id].data_quality_score = quality_score

        # 检查质量告警
        if quality_score < self.alert_thresholds['min_data_quality_score']:
            self._create_alert(
                AlertType.DATA_QUALITY_LOW,
                AlertLevel.WARNING,
                f"数据质量评分过低: {quality_score:.2f}",
                source_id,
                {'quality_score': quality_score, 'threshold': self.alert_thresholds['min_data_quality_score']}
            )

    def _check_alerts(self, source_id: str):
        """检查告警条件"""
        metrics = self.metrics[source_id]

        # 检查连续失败
        if metrics.consecutive_failures >= self.alert_thresholds['max_consecutive_failures']:
            self._create_alert(
                AlertType.DATA_COLLECTION_FAILED,
                AlertLevel.ERROR,
                f"连续采集失败 {metrics.consecutive_failures} 次",
                source_id,
                {'consecutive_failures': metrics.consecutive_failures}
            )

        # 检查成功率
        if metrics.collection_count >= 10:  # 至少10次采集后开始检查
            success_rate = metrics.success_count / metrics.collection_count
            if success_rate < self.alert_thresholds['min_success_rate']:
                self._create_alert(
                    AlertType.DATA_COLLECTION_FAILED,
                    AlertLevel.WARNING,
                    f"采集成功率过低: {success_rate:.1%}",
                    source_id,
                    {'success_rate': success_rate, 'threshold': self.alert_thresholds['min_success_rate']}
                )

        # 检查采集时间
        if metrics.avg_collection_time > self.alert_thresholds['max_collection_time']:
            self._create_alert(
                AlertType.DATA_COLLECTION_FAILED,
                AlertLevel.WARNING,
                f"平均采集时间过长: {metrics.avg_collection_time:.1f}秒",
                source_id,
                {'avg_collection_time': metrics.avg_collection_time}
            )

        # 检查网络错误
        if metrics.network_error_count >= self.alert_thresholds['max_network_errors']:
            self._create_alert(
                AlertType.NETWORK_ERROR,
                AlertLevel.ERROR,
                f"网络错误次数过多: {metrics.network_error_count}",
                source_id,
                {'network_error_count': metrics.network_error_count}
            )

        # 检查API限制
        if metrics.rate_limit_count >= self.alert_thresholds['max_rate_limits']:
            self._create_alert(
                AlertType.API_RATE_LIMIT,
                AlertLevel.WARNING,
                f"API频率限制次数过多: {metrics.rate_limit_count}",
                source_id,
                {'rate_limit_count': metrics.rate_limit_count}
            )

        # 检查数据新鲜度
        if metrics.last_success_time:
            time_since_last_success = time.time() - metrics.last_success_time
            if time_since_last_success > self.alert_thresholds['data_stale_threshold']:
                self._create_alert(
                    AlertType.DATA_STALE,
                    AlertLevel.WARNING,
                    f"数据长时间未更新: {time_since_last_success/3600:.1f}小时",
                    source_id,
                    {'hours_since_last_success': time_since_last_success/3600}
                )

    def _create_alert(self, alert_type: AlertType, level: AlertLevel,
                     message: str, source_id: str, details: Dict[str, Any]):
        """创建告警"""
        alert_id = f"{alert_type.value}_{source_id}_{int(time.time())}"

        # 检查是否已存在相同类型的活跃告警
        for existing_alert in self.alerts:
            if (existing_alert.alert_type == alert_type and
                existing_alert.source_id == source_id and
                not existing_alert.resolved):
                # 更新现有告警
                existing_alert.timestamp = time.time()
                existing_alert.details.update(details)
                logger.info(f"告警已更新: {alert_id}")
                return

        alert = Alert(
            alert_id=alert_id,
            alert_type=alert_type,
            level=level,
            message=message,
            source_id=source_id,
            timestamp=time.time(),
            details=details
        )

        self.alerts.append(alert)

        logger.warning(f"生成告警: {alert_id} - {message}")

        if self.alert_callback:
            try:
                self.alert_callback(alert)
            except Exception as e:
                logger.error(f"告警回调执行失败: {e}")

    def resolve_alert(self, alert_id: str):
        """解决告警"""
        for alert in self.alerts:
            if alert.alert_id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = time.time()
                logger.info(f"告警已解决: {alert_id}")
                break

    def get_metrics(self, source_id: Optional[str] = None) -> Dict[str, Any]:
        """
        获取监控指标

        Args:
            source_id: 数据源ID，如果为None返回所有指标

        Returns:
            Dict[str, Any]: 监控指标
        """
        if source_id:
            return self.metrics.get(source_id, DataCollectionMetrics(source_id)).__dict__
        else:
            return {sid: metrics.__dict__ for sid, metrics in self.metrics.items()}

    def get_alerts(self, resolved: bool = False, source_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取告警列表

        Args:
            resolved: 是否包含已解决的告警
            source_id: 数据源ID过滤

        Returns:
            List[Dict[str, Any]]: 告警列表
        """
        alerts = []
        for alert in self.alerts:
            if alert.resolved != resolved:
                continue
            if source_id and alert.source_id != source_id:
                continue

            alert_dict = {
                'alert_id': alert.alert_id,
                'alert_type': alert.alert_type.value,
                'level': alert.level.value,
                'message': alert.message,
                'source_id': alert.source_id,
                'timestamp': alert.timestamp,
                'details': alert.details,
                'resolved': alert.resolved
            }
            if alert.resolved_at:
                alert_dict['resolved_at'] = alert.resolved_at

            alerts.append(alert_dict)

        return alerts

    def get_health_report(self) -> Dict[str, Any]:
        """
        获取健康报告

        Returns:
            Dict[str, Any]: 健康报告
        """
        total_sources = len(self.metrics)
        healthy_sources = 0
        total_collections = 0
        total_successes = 0

        for metrics in self.metrics.values():
            total_collections += metrics.collection_count
            total_successes += metrics.success_count

            # 判断数据源是否健康
            if metrics.collection_count > 0:
                success_rate = metrics.success_count / metrics.collection_count
                is_healthy = (
                    success_rate >= self.alert_thresholds['min_success_rate'] and
                    metrics.consecutive_failures == 0 and
                    metrics.data_quality_score >= self.alert_thresholds['min_data_quality_score']
                )
                if is_healthy:
                    healthy_sources += 1

        overall_success_rate = (total_successes / total_collections * 100) if total_collections > 0 else 0

        # 获取活跃告警
        active_alerts = [a for a in self.alerts if not a.resolved]
        critical_alerts = [a for a in active_alerts if a.level in [AlertLevel.ERROR, AlertLevel.CRITICAL]]

        health_score = 100
        if critical_alerts:
            health_score -= len(critical_alerts) * 20
        if total_sources > 0 and healthy_sources < total_sources:
            health_score -= (total_sources - healthy_sources) * 10

        health_score = max(0, min(100, health_score))

        return {
            'overall_health_score': health_score,
            'total_sources': total_sources,
            'healthy_sources': healthy_sources,
            'overall_success_rate': f"{overall_success_rate:.1f}%",
            'total_collections': total_collections,
            'active_alerts': len(active_alerts),
            'critical_alerts': len(critical_alerts),
            'alerts_by_level': {
                'info': len([a for a in active_alerts if a.level == AlertLevel.INFO]),
                'warning': len([a for a in active_alerts if a.level == AlertLevel.WARNING]),
                'error': len([a for a in active_alerts if a.level == AlertLevel.ERROR]),
                'critical': len([a for a in active_alerts if a.level == AlertLevel.CRITICAL])
            }
        }

    def cleanup_old_alerts(self, max_age_days: int = 30):
        """
        清理旧的已解决告警

        Args:
            max_age_days: 最大保留天数
        """
        cutoff_time = time.time() - (max_age_days * 24 * 3600)

        self.alerts = [
            alert for alert in self.alerts
            if not alert.resolved or alert.resolved_at > cutoff_time
        ]

        logger.info(f"已清理旧告警，当前告警数量: {len(self.alerts)}")

# 全局监控器实例
_monitor = None

def get_data_collection_monitor(alert_callback: Optional[Callable[[Alert], None]] = None) -> DataCollectionMonitor:
    """
    获取数据采集监控器实例（单例模式）

    Args:
        alert_callback: 告警回调函数

    Returns:
        DataCollectionMonitor: 监控器实例
    """
    global _monitor
    if _monitor is None:
        _monitor = DataCollectionMonitor(alert_callback=alert_callback)
        logger.info("数据采集监控器已初始化")
    return _monitor