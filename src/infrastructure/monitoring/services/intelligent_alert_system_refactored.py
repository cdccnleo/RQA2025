#!/usr/bin/env python3
"""
RQA2025 基础设施层智能告警系统 (重构版)

提供完整的监控告警规则配置、触发和通知功能。

重构说明:
- 拆分为多个职责单一的组件
- 使用参数对象模式替换长参数列表
- 提高代码可维护性和可测试性
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json
import logging

from ..core.parameter_objects import AlertRuleConfig, AlertConditionConfig
from ..core.constants import (
    NOTIFICATION_MAX_RETRIES, NOTIFICATION_RETRY_DELAY,
    ALERT_LEVEL_INFO, ALERT_LEVEL_WARNING, ALERT_LEVEL_ERROR, ALERT_LEVEL_CRITICAL
)
from ..core.exceptions import MonitoringException, NotificationError

logger = logging.getLogger(__name__)


class AlertRuleManager:
    """
    告警规则管理器

    专门负责告警规则的增删改查和管理。
    """

    def __init__(self):
        """初始化告警规则管理器"""
        self.rules: Dict[str, AlertRuleConfig] = {}
        self._init_default_rules()

    def _init_default_rules(self):
        """初始化默认告警规则"""
        # 这里可以初始化一些默认规则
        pass

    def add_rule(self, rule: AlertRuleConfig) -> bool:
        """
        添加告警规则

        Args:
            rule: 告警规则配置

        Returns:
            bool: 是否成功添加
        """
        try:
            if rule.rule_id in self.rules:
                logger.warning(f"告警规则已存在: {rule.rule_id}")
                return False

            self.rules[rule.rule_id] = rule
            logger.info(f"告警规则已添加: {rule.rule_id}")
            return True

        except Exception as e:
            logger.error(f"添加告警规则失败: {e}")
            return False

    def remove_rule(self, rule_id: str) -> bool:
        """
        移除告警规则

        Args:
            rule_id: 规则ID

        Returns:
            bool: 是否成功移除
        """
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"告警规则已移除: {rule_id}")
            return True
        return False

    def get_rule(self, rule_id: str) -> Optional[AlertRuleConfig]:
        """
        获取告警规则

        Args:
            rule_id: 规则ID

        Returns:
            Optional[AlertRuleConfig]: 告警规则配置
        """
        return self.rules.get(rule_id)

    def get_all_rules(self) -> List[AlertRuleConfig]:
        """
        获取所有告警规则

        Returns:
            List[AlertRuleConfig]: 告警规则列表
        """
        return list(self.rules.values())

    def enable_rule(self, rule_id: str) -> bool:
        """
        启用告警规则

        Args:
            rule_id: 规则ID

        Returns:
            bool: 是否成功启用
        """
        rule = self.rules.get(rule_id)
        if rule:
            rule.enabled = True
            logger.info(f"告警规则已启用: {rule_id}")
            return True
        return False

    def disable_rule(self, rule_id: str) -> bool:
        """
        禁用告警规则

        Args:
            rule_id: 规则ID

        Returns:
            bool: 是否成功禁用
        """
        rule = self.rules.get(rule_id)
        if rule:
            rule.enabled = False
            logger.info(f"告警规则已禁用: {rule_id}")
            return True
        return False


class AlertProcessor:
    """
    告警处理器

    专门负责告警的评估、触发和处理逻辑。
    """

    def __init__(self, rule_manager: AlertRuleManager):
        """
        初始化告警处理器

        Args:
            rule_manager: 告警规则管理器
        """
        self.rule_manager = rule_manager
        self.last_trigger_times: Dict[str, datetime] = {}

    def process_alerts(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        处理告警

        Args:
            data: 监控数据

        Returns:
            List[Dict[str, Any]]: 触发的告警列表
        """
        triggered_alerts = []

        for rule in self.rule_manager.get_all_rules():
            if not rule.enabled:
                continue

            if self._should_trigger_alert(rule, data):
                alert = self._create_alert(rule, data)
                triggered_alerts.append(alert)

        return triggered_alerts

    def _should_trigger_alert(self, rule: AlertRuleConfig, data: Dict[str, Any]) -> bool:
        """
        判断是否应该触发告警

        Args:
            rule: 告警规则
            data: 监控数据

        Returns:
            bool: 是否触发告警
        """
        try:
            # 检查冷却时间
            if not self._is_cooldown_expired(rule.rule_id, rule.cooldown):
                return False

            # 评估告警条件
            return self._evaluate_condition(rule.condition, data)

        except Exception as e:
            logger.error(f"评估告警规则失败 {rule.rule_id}: {e}")
            return False

    def _evaluate_condition(self, condition: AlertConditionConfig, data: Dict[str, Any]) -> bool:
        """
        评估告警条件

        Args:
            condition: 告警条件
            data: 监控数据

        Returns:
            bool: 条件是否满足
        """
        if condition.field not in data:
            return False

        actual_value = data[condition.field]

        if condition.operator == "gt":
            return actual_value > condition.value
        elif condition.operator == "lt":
            return actual_value < condition.value
        elif condition.operator == "eq":
            return actual_value == condition.value
        elif condition.operator == "ne":
            return actual_value != condition.value
        elif condition.operator == "ge":
            return actual_value >= condition.value
        elif condition.operator == "le":
            return actual_value <= condition.value
        else:
            logger.warning(f"不支持的操作符: {condition.operator}")
            return False

    def _create_alert(self, rule: AlertRuleConfig, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建告警

        Args:
            rule: 告警规则
            data: 监控数据

        Returns:
            Dict[str, Any]: 告警信息
        """
        alert = {
            'alert_id': f"{rule.rule_id}_{int(datetime.now().timestamp())}",
            'rule_id': rule.rule_id,
            'rule_name': rule.name,
            'description': rule.description,
            'level': rule.level,
            'status': 'active',
            'triggered_at': datetime.now().isoformat(),
            'data': data.copy(),
            'channels': rule.channels.copy(),
            'acknowledged': False
        }

        # 更新最后触发时间
        self.last_trigger_times[rule.rule_id] = datetime.now()

        return alert

    def _is_cooldown_expired(self, rule_id: str, cooldown_seconds: int) -> bool:
        """
        检查冷却时间是否已过期

        Args:
            rule_id: 规则ID
            cooldown_seconds: 冷却时间（秒）

        Returns:
            bool: 冷却时间是否已过期
        """
        last_trigger_time = self.last_trigger_times.get(rule_id)
        if not last_trigger_time:
            return True

        elapsed = datetime.now() - last_trigger_time
        return elapsed.total_seconds() >= cooldown_seconds


class AlertHistoryManager:
    """
    告警历史管理器

    专门负责告警历史的存储、查询和管理。
    """

    def __init__(self, max_history_size: int = 1000):
        """
        初始化告警历史管理器

        Args:
            max_history_size: 最大历史记录数量
        """
        self.alert_history: List[Dict[str, Any]] = []
        self.max_history_size = max_history_size

    def add_alert(self, alert: Dict[str, Any]):
        """
        添加告警到历史

        Args:
            alert: 告警信息
        """
        self.alert_history.append(alert)

        # 限制历史记录大小
        if len(self.alert_history) > self.max_history_size:
            self.alert_history = self.alert_history[-self.max_history_size:]

        logger.warning(f"告警已记录: {alert['rule_name']} - {alert['description']}")

    def get_alert_history(self, limit: int = 100, level: Optional[str] = None,
                         status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取告警历史

        Args:
            limit: 返回的最大记录数
            level: 告警级别过滤
            status: 告警状态过滤

        Returns:
            List[Dict[str, Any]]: 告警历史列表
        """
        filtered_alerts = self.alert_history

        # 按级别过滤
        if level:
            filtered_alerts = [a for a in filtered_alerts if a.get('level') == level]

        # 按状态过滤
        if status:
            filtered_alerts = [a for a in filtered_alerts if a.get('status') == status]

        # 排序（最新的在前）
        filtered_alerts.sort(key=lambda x: x.get('triggered_at', ''), reverse=True)

        return filtered_alerts[:limit]

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """
        获取活跃告警

        Returns:
            List[Dict[str, Any]]: 活跃告警列表
        """
        return [alert for alert in self.alert_history if alert.get('status') == 'active']

    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        确认告警

        Args:
            alert_id: 告警ID

        Returns:
            bool: 是否成功确认
        """
        for alert in self.alert_history:
            if alert.get('alert_id') == alert_id:
                alert['status'] = 'acknowledged'
                alert['acknowledged_at'] = datetime.now().isoformat()
                logger.info(f"告警已确认: {alert_id}")
                return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """
        解决告警

        Args:
            alert_id: 告警ID

        Returns:
            bool: 是否成功解决
        """
        for alert in self.alert_history:
            if alert.get('alert_id') == alert_id:
                alert['status'] = 'resolved'
                alert['resolved_at'] = datetime.now().isoformat()
                logger.info(f"告警已解决: {alert_id}")
                return True
        return False

    def get_alert_statistics(self) -> Dict[str, Any]:
        """
        获取告警统计信息

        Returns:
            Dict[str, Any]: 告警统计
        """
        total_alerts = len(self.alert_history)
        active_alerts = len(self.get_active_alerts())

        # 按级别统计
        level_counts = {}
        for alert in self.alert_history:
            level = alert.get('level', 'unknown')
            level_counts[level] = level_counts.get(level, 0) + 1

        # 按状态统计
        status_counts = {}
        for alert in self.alert_history:
            status = alert.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            'total_alerts': total_alerts,
            'active_alerts': active_alerts,
            'level_distribution': level_counts,
            'status_distribution': status_counts,
            'generated_at': datetime.now().isoformat()
        }


class NotificationManager:
    """
    通知管理器

    专门负责告警通知的发送和管理。
    """

    def __init__(self):
        """初始化通知管理器"""
        self.notification_channels = {
            'console': self._notify_console,
            'email': self._notify_email,
            'webhook': self._notify_webhook,
            'slack': self._notify_slack
        }

    def send_notification(self, alert: Dict[str, Any]) -> bool:
        """
        发送通知

        Args:
            alert: 告警信息

        Returns:
            bool: 是否发送成功
        """
        channels = alert.get('channels', ['console'])
        success = True

        for channel in channels:
            try:
                if channel in self.notification_channels:
                    self.notification_channels[channel](alert)
                else:
                    logger.warning(f"不支持的通知渠道: {channel}")
            except Exception as e:
                logger.error(f"发送通知失败 ({channel}): {e}")
                success = False

        return success

    def _notify_console(self, alert: Dict[str, Any]):
        """
        控制台通知

        Args:
            alert: 告警信息
        """
        level = alert.get('level', 'info').upper()
        message = f"[{level}] {alert.get('rule_name', 'Unknown')}: {alert.get('description', '')}"
        print(message)

    def _notify_email(self, alert: Dict[str, Any]):
        """
        邮件通知

        Args:
            alert: 告警信息
        """
        # 这里应该实现邮件发送逻辑
        logger.info(f"邮件通知: {alert.get('rule_name')}")

    def _notify_webhook(self, alert: Dict[str, Any]):
        """
        Webhook通知

        Args:
            alert: 告警信息
        """
        # 这里应该实现Webhook调用逻辑
        logger.info(f"Webhook通知: {alert.get('rule_name')}")

    def _notify_slack(self, alert: Dict[str, Any]):
        """
        Slack通知

        Args:
            alert: 告警信息
        """
        # 这里应该实现Slack消息发送逻辑
        logger.info(f"Slack通知: {alert.get('rule_name')}")


class IntelligentAlertSystemRefactored:
    """
    智能告警系统 (重构版)

    使用组件化架构的智能告警系统，提供：
    - 模块化的告警管理组件
    - 灵活的通知渠道
    - 完整的告警历史管理
    """

    def __init__(self):
        """初始化智能告警系统"""
        # 初始化各个组件
        self.rule_manager = AlertRuleManager()
        self.alert_processor = AlertProcessor(self.rule_manager)
        self.history_manager = AlertHistoryManager()
        self.notification_manager = NotificationManager()

    def process_monitoring_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        处理监控数据并生成告警

        Args:
            data: 监控数据

        Returns:
            List[Dict[str, Any]]: 生成的告警列表
        """
        try:
            # 处理告警
            alerts = self.alert_processor.process_alerts(data)

            # 记录告警历史并发送通知
            for alert in alerts:
                self.history_manager.add_alert(alert)
                self.notification_manager.send_notification(alert)

            return alerts

        except Exception as e:
            logger.error(f"处理监控数据失败: {e}")
            return []

    def add_alert_rule(self, rule: AlertRuleConfig) -> bool:
        """
        添加告警规则

        Args:
            rule: 告警规则配置

        Returns:
            bool: 是否成功添加
        """
        return self.rule_manager.add_rule(rule)

    def remove_alert_rule(self, rule_id: str) -> bool:
        """
        移除告警规则

        Args:
            rule_id: 规则ID

        Returns:
            bool: 是否成功移除
        """
        return self.rule_manager.remove_rule(rule_id)

    def get_alert_rule(self, rule_id: str) -> Optional[AlertRuleConfig]:
        """
        获取告警规则

        Args:
            rule_id: 规则ID

        Returns:
            Optional[AlertRuleConfig]: 告警规则配置
        """
        return self.rule_manager.get_rule(rule_id)

    def get_all_alert_rules(self) -> List[AlertRuleConfig]:
        """
        获取所有告警规则

        Returns:
            List[AlertRuleConfig]: 告警规则列表
        """
        return self.rule_manager.get_all_rules()

    def get_alert_history(self, limit: int = 100, level: Optional[str] = None,
                         status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取告警历史

        Args:
            limit: 返回的最大记录数
            level: 告警级别过滤
            status: 告警状态过滤

        Returns:
            List[Dict[str, Any]]: 告警历史列表
        """
        return self.history_manager.get_alert_history(limit, level, status)

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """
        获取活跃告警

        Returns:
            List[Dict[str, Any]]: 活跃告警列表
        """
        return self.history_manager.get_active_alerts()

    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        确认告警

        Args:
            alert_id: 告警ID

        Returns:
            bool: 是否成功确认
        """
        return self.history_manager.acknowledge_alert(alert_id)

    def resolve_alert(self, alert_id: str) -> bool:
        """
        解决告警

        Args:
            alert_id: 告警ID

        Returns:
            bool: 是否成功解决
        """
        return self.history_manager.resolve_alert(alert_id)

    def get_alert_statistics(self) -> Dict[str, Any]:
        """
        获取告警统计信息

        Returns:
            Dict[str, Any]: 告警统计
        """
        return self.history_manager.get_alert_statistics()

    def export_alert_history(self, file_path: str, format_type: str = 'json') -> bool:
        """
        导出告警历史

        Args:
            file_path: 导出文件路径
            format_type: 导出格式

        Returns:
            bool: 是否成功导出
        """
        try:
            alert_history = self.get_alert_history(limit=10000)  # 导出所有历史

            if format_type == 'json':
                import json
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'export_time': datetime.now().isoformat(),
                        'total_alerts': len(alert_history),
                        'alerts': alert_history
                    }, f, indent=2, ensure_ascii=False)

            elif format_type == 'csv':
                import csv
                if alert_history:
                    fieldnames = alert_history[0].keys()
                    with open(file_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(alert_history)

            else:
                logger.error(f"不支持的导出格式: {format_type}")
                return False

            logger.info(f"告警历史已导出到: {file_path}")
            return True

        except Exception as e:
            logger.error(f"导出告警历史失败: {e}")
            return False
