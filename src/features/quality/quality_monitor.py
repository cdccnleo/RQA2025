"""
特征质量监控器
用于监控特征质量变化并发送告警
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QualityAlert:
    """质量告警"""
    alert_id: str
    feature_id: int
    feature_name: str
    alert_type: str  # 'quality_drop', 'low_quality', 'anomaly'
    severity: str    # 'info', 'warning', 'critical'
    message: str
    old_score: Optional[float]
    new_score: Optional[float]
    created_at: datetime


class QualityMonitor:
    """特征质量监控器"""
    
    # 告警阈值配置
    ALERT_THRESHOLDS = {
        'quality_drop': 0.1,      # 质量下降超过10%触发告警
        'low_quality': 0.7,       # 质量评分低于0.7触发告警
        'critical_quality': 0.5,  # 质量评分低于0.5触发严重告警
    }
    
    def __init__(self):
        self._alert_history: List[QualityAlert] = []
        self._max_history_size = 1000
        self._alert_cooldown = timedelta(minutes=30)  # 告警冷却时间
        self._last_alert_time: Dict[str, datetime] = {}
    
    def check_quality_change(
        self,
        feature_id: int,
        feature_name: str,
        old_score: float,
        new_score: float
    ) -> Optional[QualityAlert]:
        """
        检查质量变化，必要时生成告警
        
        Args:
            feature_id: 特征ID
            feature_name: 特征名称
            old_score: 旧的质量评分
            new_score: 新的质量评分
        
        Returns:
            如果有告警则返回告警对象，否则返回None
        """
        # 首先检查低质量（优先级更高）
        if new_score < self.ALERT_THRESHOLDS['critical_quality']:
            return self._create_alert(
                feature_id=feature_id,
                feature_name=feature_name,
                alert_type='low_quality',
                severity='critical',
                message=f"特征 {feature_name} 质量严重偏低 ({new_score:.3f})，"
                       f"建议检查数据源或重新计算",
                old_score=old_score,
                new_score=new_score
            )
        elif new_score < self.ALERT_THRESHOLDS['low_quality']:
            return self._create_alert(
                feature_id=feature_id,
                feature_name=feature_name,
                alert_type='low_quality',
                severity='warning',
                message=f"特征 {feature_name} 质量偏低 ({new_score:.3f})，"
                       f"建议关注",
                old_score=old_score,
                new_score=new_score
            )
        
        # 然后检查质量下降（仅在质量正常时检查）
        if old_score > 0:
            quality_drop = old_score - new_score
            if quality_drop >= self.ALERT_THRESHOLDS['quality_drop']:
                return self._create_alert(
                    feature_id=feature_id,
                    feature_name=feature_name,
                    alert_type='quality_drop',
                    severity='warning' if quality_drop < 0.2 else 'critical',
                    message=f"特征 {feature_name} 质量下降 {quality_drop:.1%} "
                           f"(从 {old_score:.3f} 降至 {new_score:.3f})",
                    old_score=old_score,
                    new_score=new_score
                )
        
        return None
    
    def _create_alert(
        self,
        feature_id: int,
        feature_name: str,
        alert_type: str,
        severity: str,
        message: str,
        old_score: Optional[float],
        new_score: Optional[float]
    ) -> Optional[QualityAlert]:
        """创建告警"""
        # 检查冷却时间
        alert_key = f"{feature_id}_{alert_type}"
        last_alert = self._last_alert_time.get(alert_key)
        
        if last_alert and datetime.now() - last_alert < self._alert_cooldown:
            logger.debug(f"告警 {alert_key} 在冷却期内，跳过")
            return None
        
        # 创建告警
        alert = QualityAlert(
            alert_id=f"{alert_key}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            feature_id=feature_id,
            feature_name=feature_name,
            alert_type=alert_type,
            severity=severity,
            message=message,
            old_score=old_score,
            new_score=new_score,
            created_at=datetime.now()
        )
        
        # 更新告警历史
        self._alert_history.append(alert)
        if len(self._alert_history) > self._max_history_size:
            self._alert_history = self._alert_history[-self._max_history_size:]
        
        # 更新最后告警时间
        self._last_alert_time[alert_key] = datetime.now()
        
        # 记录告警
        log_method = getattr(logger, severity, logger.info)
        log_method(f"[质量告警] {message}")
        
        return alert
    
    def get_recent_alerts(
        self,
        hours: int = 24,
        severity: Optional[str] = None
    ) -> List[QualityAlert]:
        """
        获取最近的告警
        
        Args:
            hours: 最近多少小时内的告警
            severity: 按严重程度过滤
        
        Returns:
            告警列表
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        alerts = [
            alert for alert in self._alert_history
            if alert.created_at >= cutoff_time
        ]
        
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        
        return sorted(alerts, key=lambda x: x.created_at, reverse=True)
    
    def get_alert_statistics(self, days: int = 7) -> Dict:
        """
        获取告警统计
        
        Args:
            days: 统计天数
        
        Returns:
            告警统计数据
        """
        cutoff_time = datetime.now() - timedelta(days=days)
        recent_alerts = [
            alert for alert in self._alert_history
            if alert.created_at >= cutoff_time
        ]
        
        # 按类型统计
        type_counts = {}
        severity_counts = {}
        
        for alert in recent_alerts:
            type_counts[alert.alert_type] = type_counts.get(alert.alert_type, 0) + 1
            severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1
        
        return {
            'total_alerts': len(recent_alerts),
            'type_distribution': type_counts,
            'severity_distribution': severity_counts,
            'period_days': days
        }
    
    def clear_history(self):
        """清除告警历史"""
        self._alert_history.clear()
        self._last_alert_time.clear()
        logger.info("质量告警历史已清除")


# 全局质量监控器实例
_quality_monitor: Optional[QualityMonitor] = None


def get_quality_monitor() -> QualityMonitor:
    """获取全局质量监控器实例"""
    global _quality_monitor
    if _quality_monitor is None:
        _quality_monitor = QualityMonitor()
    return _quality_monitor
