#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
告警服务实现
Alert Service Implementation

提供告警规则管理、告警触发检测和告警通知功能。
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
from strategy.interfaces.monitoring_interfaces import (
    IAlertService, Alert, AlertRule, AlertLevel
)
from strategy.core.integration.business_adapters import get_unified_adapter_factory

logger = logging.getLogger(__name__)


class AlertService(IAlertService):

    """
    告警服务
    Alert Service

    管理告警规则、检测告警触发并发送通知。
    """

    def __init__(self):
        """初始化告警服务"""
        self.adapter_factory = get_unified_adapter_factory()

        # 告警规则存储
        self.alert_rules: Dict[str, AlertRule] = {}

        # 活跃告警存储
        self.active_alerts: List[Alert] = []

        # 历史告警存储
        self.alert_history: List[Alert] = []

        # 最大历史告警数量
        self.max_history_alerts = 10000

        logger.info("告警服务初始化完成")

    async def create_alert_rule(self, rule: AlertRule) -> bool:
        """
        创建告警规则

        Args:
            rule: 告警规则

        Returns:
            bool: 创建是否成功
        """
        try:
            if rule.rule_id in self.alert_rules:
                logger.warning(f"告警规则 {rule.rule_id} 已存在")
                return False

            # 验证规则
            if not await self._validate_alert_rule(rule):
                logger.error(f"告警规则验证失败: {rule.rule_id}")
                return False

            self.alert_rules[rule.rule_id] = rule

            # 发布事件
            await self._publish_event("alert_rule_created", {
                "rule_id": rule.rule_id,
                "strategy_id": rule.strategy_id,
                "metric_name": rule.metric_name,
                "level": rule.level.value,
                "timestamp": datetime.now().isoformat()
            })

            logger.info(f"告警规则创建成功: {rule.rule_id}")
            return True

        except Exception as e:
            logger.error(f"创建告警规则失败: {e}")
            return False

    async def update_alert_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """
        更新告警规则

        Args:
            rule_id: 规则ID
            updates: 更新内容

        Returns:
            bool: 更新是否成功
        """
        try:
            if rule_id not in self.alert_rules:
                logger.warning(f"告警规则 {rule_id} 不存在")
                return False

            rule = self.alert_rules[rule_id]
            original_rule = rule.__dict__.copy()

            # 应用更新
            for key, value in updates.items():
                if hasattr(rule, key):
                    setattr(rule, key, value)

            # 验证更新后的规则
            if not await self._validate_alert_rule(rule):
                logger.error(f"更新后的告警规则验证失败: {rule_id}")
                return False

            # 发布事件
            await self._publish_event("alert_rule_updated", {
                "rule_id": rule_id,
                "strategy_id": rule.strategy_id,
                "updates": updates,
                "timestamp": datetime.now().isoformat()
            })

            logger.info(f"告警规则更新成功: {rule_id}")
            return True

        except Exception as e:
            logger.error(f"更新告警规则失败: {e}")
            return False

    async def delete_alert_rule(self, rule_id: str) -> bool:
        """
        删除告警规则

        Args:
            rule_id: 规则ID

        Returns:
            bool: 删除是否成功
        """
        try:
            if rule_id not in self.alert_rules:
                logger.warning(f"告警规则 {rule_id} 不存在")
                return False

            rule = self.alert_rules[rule_id]

            # 删除规则
            del self.alert_rules[rule_id]

            # 同时删除相关的活跃告警
            self.active_alerts = [
                alert for alert in self.active_alerts
                if alert.rule_id != rule_id
            ]

            # 发布事件
            await self._publish_event("alert_rule_deleted", {
                "rule_id": rule_id,
                "strategy_id": rule.strategy_id,
                "timestamp": datetime.now().isoformat()
            })

            logger.info(f"告警规则删除成功: {rule_id}")
            return True

        except Exception as e:
            logger.error(f"删除告警规则失败: {e}")
            return False

    async def _validate_alert_rule(self, rule: AlertRule) -> bool:
        """
        验证告警规则

        Args:
            rule: 告警规则

        Returns:
            bool: 规则是否有效
        """
        # 基本验证
        if not rule.rule_id or not rule.strategy_id or not rule.metric_name:
            return False

        if rule.condition not in ['>', '<', '>=', '<=', '==', '!=']:
            return False

        if not isinstance(rule.level, AlertLevel):
            return False

        if rule.cooldown_minutes < 0:
            return False

        return True

    def get_alert_rules(self, strategy_id: str) -> List[AlertRule]:
        """
        获取告警规则

        Args:
            strategy_id: 策略ID

        Returns:
            List[AlertRule]: 告警规则列表
        """
        return [
            rule for rule in self.alert_rules.values()
            if rule.strategy_id == strategy_id and rule.enabled
        ]

    def get_active_alerts(self, strategy_id: Optional[str] = None) -> List[Alert]:
        """
        获取活跃告警

        Args:
            strategy_id: 策略ID过滤器

        Returns:
            List[Alert]: 活跃告警列表
        """
        alerts = [alert for alert in self.active_alerts if not alert.resolved]

        if strategy_id:
            alerts = [alert for alert in alerts if alert.strategy_id == strategy_id]

        return alerts

    async def resolve_alert(self, alert_id: str, resolution_notes: Optional[str] = None) -> bool:
        """
        解决告警

        Args:
            alert_id: 告警ID
            resolution_notes: 解决说明

        Returns:
            bool: 解决是否成功
        """
        try:
            # 查找告警
            alert = None
            for a in self.active_alerts:
                if a.alert_id == alert_id:
                    alert = a
                    break

            if not alert:
                logger.warning(f"告警 {alert_id} 不存在或已解决")
                return False

            # 标记为已解决
            alert.resolved = True
            alert.resolved_at = datetime.now()

            # 添加到历史记录
            self.alert_history.append(alert)

            # 从活跃告警中移除
            self.active_alerts.remove(alert)

            # 限制历史记录数量
            if len(self.alert_history) > self.max_history_alerts:
                self.alert_history = self.alert_history[-self.max_history_alerts:]

            # 发布事件
            await self._publish_event("alert_resolved", {
                "alert_id": alert_id,
                "strategy_id": alert.strategy_id,
                "resolution_notes": resolution_notes,
                "timestamp": datetime.now().isoformat()
            })

            logger.info(f"告警已解决: {alert_id}")
            return True

        except Exception as e:
            logger.error(f"解决告警失败: {e}")
            return False

    async def _publish_event(self, event_type: str, event_data: Dict[str, Any]):
        """
        发布事件

        Args:
            event_type: 事件类型
            event_data: 事件数据
        """
        try:
            event_bus_adapter = self.adapter_factory.get_adapter("event_bus")
            await event_bus_adapter.publish_event({
                "event_type": f"alert_{event_type}",
                "data": event_data,
                "source": "alert_service",
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"事件发布异常: {e}")

    def get_alert_history(self, strategy_id: str,


                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None) -> List[Alert]:
        """
        获取告警历史

        Args:
            strategy_id: 策略ID
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            List[Alert]: 告警历史列表
        """
        try:
            # 筛选策略相关的告警
            alerts = [alert for alert in self.alert_history if alert.strategy_id == strategy_id]

            # 时间范围筛选
            if start_time:
                alerts = [alert for alert in alerts if alert.timestamp >= start_time]

            if end_time:
                alerts = [alert for alert in alerts if alert.timestamp <= end_time]

            # 按时间排序
            alerts.sort(key=lambda x: x.timestamp, reverse=True)

            return alerts

        except Exception as e:
            logger.error(f"获取告警历史失败: {e}")
            return []

    def get_alert_statistics(self, strategy_id: str,


                             days: int = 7) -> Dict[str, Any]:
        """
        获取告警统计信息

        Args:
            strategy_id: 策略ID
            days: 统计天数

        Returns:
            Dict[str, Any]: 告警统计
        """
        try:
            cutoff_time = datetime.now() - timedelta(days=days)

            # 获取时间范围内的告警
            alerts = self.get_alert_history(strategy_id, cutoff_time)

            # 统计信息
            stats = {
                'total_alerts': len(alerts),
                'active_alerts': len(self.get_active_alerts(strategy_id)),
                'resolved_alerts': len([a for a in alerts if a.resolved]),
                'by_level': {},
                'by_rule': {},
                'avg_resolution_time': 0.0
            }

            # 按级别统计
            for level in AlertLevel:
                level_alerts = [a for a in alerts if a.level == level]
                stats['by_level'][level.value] = len(level_alerts)

            # 按规则统计
            rule_counts = {}
            for alert in alerts:
                rule_counts[alert.rule_id] = rule_counts.get(alert.rule_id, 0) + 1
            stats['by_rule'] = rule_counts

            # 计算平均解决时间
            resolved_alerts = [a for a in alerts if a.resolved and a.resolved_at]
            if resolved_alerts:
                resolution_times = [
                    (a.resolved_at - a.timestamp).total_seconds() / 3600  # 小时
                    for a in resolved_alerts
                ]
                stats['avg_resolution_time'] = sum(resolution_times) / len(resolution_times)

            return stats

        except Exception as e:
            logger.error(f"获取告警统计失败: {e}")
            return {}

    def cleanup_old_alerts(self, days_to_keep: int = 90) -> int:
        """
        清理旧告警

        Args:
            days_to_keep: 保留天数

        Returns:
            int: 删除的告警数量
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            original_count = len(self.alert_history)

            # 保留指定天数内的告警
            self.alert_history = [
                alert for alert in self.alert_history
                if alert.timestamp >= cutoff_date
            ]

            deleted_count = original_count - len(self.alert_history)
            logger.info(f"已清理 {deleted_count} 个旧告警")

            return deleted_count

        except Exception as e:
            logger.error(f"清理旧告警失败: {e}")
            return 0


# 导出类
__all__ = [
    'AlertService'
]
