#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 审计规则引擎

专门负责审计规则的定义、执行和触发逻辑
从AuditLoggingManager中分离出来，提高代码组织性
"""

import logging
import re
import threading
from typing import Any, Dict, List, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from src.infrastructure.security.audit.audit_events import AuditEventType, AuditSeverity


class RuleAction(Enum):
    """规则动作"""
    LOG = "log"                    # 记录日志
    ALERT = "alert"                # 触发告警
    BLOCK = "block"               # 阻止操作
    NOTIFY = "notify"             # 发送通知
    ESCALATE = "escalate"         # 升级处理


class RuleConditionType(Enum):
    """规则条件类型"""
    EVENT_TYPE = "event_type"          # 事件类型
    SEVERITY = "severity"             # 严重程度
    USER_ID = "user_id"              # 用户ID
    RESOURCE = "resource"            # 资源
    ACTION = "action"               # 操作
    RESULT = "result"               # 结果
    RISK_SCORE = "risk_score"       # 风险分数
    TIME_WINDOW = "time_window"     # 时间窗口
    FREQUENCY = "frequency"         # 频率
    PATTERN = "pattern"            # 模式匹配


@dataclass
class RuleCondition:
    """规则条件"""
    condition_type: RuleConditionType
    operator: str  # eq, ne, gt, lt, contains, regex, etc.
    value: Any
    case_sensitive: bool = True

    def matches(self, event: 'AuditEvent') -> bool:
        """检查事件是否匹配条件"""
        if event is None:
            return False

        event_value = self._get_event_value(event)

        if event_value is None:
            return False

        candidate = self._normalize_value(self.value)
        target = self._normalize_value(event_value)

        if self.operator == "eq":
            return target == candidate
        elif self.operator == "ne":
            return target != candidate
        elif self.operator == "gt":
            return target > candidate
        elif self.operator == "gte":
            return target >= candidate
        elif self.operator == "lt":
            return target < candidate
        elif self.operator == "lte":
            return target <= candidate
        elif self.operator == "contains":
            if isinstance(target, str) and isinstance(candidate, str):
                if self.case_sensitive:
                    return candidate in target
                return candidate.lower() in target.lower()
            return False
        elif self.operator == "regex":
            if isinstance(target, str):
                flags = 0 if self.case_sensitive else re.IGNORECASE
                pattern = self.value if isinstance(self.value, str) else str(self.value)
                return bool(re.search(pattern, target, flags))
            return False
        elif self.operator == "in":
            if isinstance(self.value, (list, tuple, set)):
                normalized_collection = {self._normalize_value(item) for item in self.value}
                return target in normalized_collection
            return False

        return False

    def _get_event_value(self, event: 'AuditEvent') -> Any:
        """获取事件中的对应值"""
        if self.condition_type == RuleConditionType.EVENT_TYPE:
            return event.event_type
        elif self.condition_type == RuleConditionType.SEVERITY:
            return event.severity
        elif self.condition_type == RuleConditionType.USER_ID:
            return event.user_id
        elif self.condition_type == RuleConditionType.RESOURCE:
            return event.resource
        elif self.condition_type == RuleConditionType.ACTION:
            return event.action or event.details.get('action')
        elif self.condition_type == RuleConditionType.RESULT:
            if event.result:
                return event.result
            if 'result' in event.details:
                return event.details.get('result')
            if 'success' in event.details:
                return 'success' if event.details['success'] else 'failure'
            return None
        elif self.condition_type == RuleConditionType.RISK_SCORE:
            return event.risk_score
        elif self.condition_type == RuleConditionType.TIME_WINDOW:
            return event.timestamp
        elif self.condition_type == RuleConditionType.PATTERN:
            # 模式匹配通常用于资源或详情
            return event.resource or str(event.details)
        elif self.condition_type == RuleConditionType.FREQUENCY:
            return event.details.get('frequency')

        return None

    def _normalize_value(self, value: Any) -> Any:
        """统一比较值，处理大小写与枚举类型"""
        if isinstance(value, Enum):
            base = value.name if self.case_sensitive else value.name.lower()
            return base
        if isinstance(value, str):
            return value if self.case_sensitive else value.lower()
        return value


@dataclass
class AuditRule:
    """审计规则"""
    rule_id: str
    name: str
    description: str
    conditions: List[RuleCondition] = field(default_factory=list)
    actions: List[RuleAction] = field(default_factory=list)
    priority: int = 1
    enabled: bool = True
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def matches(self, event: 'AuditEvent') -> bool:
        """检查事件是否匹配规则"""
        if not self.enabled:
            return False

        # 所有条件都必须匹配 (AND逻辑)
        for condition in self.conditions:
            if not condition.matches(event):
                return False

        return True

    def execute_actions(self, event: 'AuditEvent') -> List[Dict[str, Any]]:
        """执行规则动作"""
        if not self.enabled:
            return []

        results = []

        for action in self.actions:
            action_enum = action
            if not isinstance(action_enum, RuleAction):
                try:
                    action_enum = RuleAction[str(action).upper()]
                except KeyError:
                    action_enum = None

            result = {
                'rule_id': self.rule_id,
                'rule_name': self.name,
                'action': action_enum.value if isinstance(action_enum, RuleAction) else str(action),
                'event_id': event.event_id,
                'timestamp': datetime.now().isoformat(),
                'result': 'success'
            }

            try:
                if action_enum == RuleAction.LOG:
                    self._execute_log_action(event)
                elif action_enum == RuleAction.ALERT:
                    self._execute_alert_action(event)
                elif action_enum == RuleAction.BLOCK:
                    self._execute_block_action(event)
                elif action_enum == RuleAction.NOTIFY:
                    self._execute_notify_action(event)
                elif action_enum == RuleAction.ESCALATE:
                    self._execute_escalate_action(event)
                else:
                    result['result'] = 'skipped'
                    result['error'] = f"Unsupported action: {action}"
            except Exception as e:
                result['result'] = 'error'
                result['error'] = str(e)
                action_label = action_enum.value if isinstance(action_enum, RuleAction) else str(action)
                logging.error(f"Failed to execute action {action_label} for rule {self.rule_id}: {e}")

            results.append(result)

        return results

    def _execute_log_action(self, event: 'AuditEvent') -> None:
        """执行日志记录动作"""
        logging.info(f"Rule {self.rule_id} triggered: {self.name} for event {event.event_id}")

    def _execute_alert_action(self, event: 'AuditEvent') -> None:
        """执行告警动作"""
        logging.warning(f"ALERT: Rule {self.rule_id} ({self.name}) triggered for high-risk event {event.event_id}")

    def _execute_block_action(self, event: 'AuditEvent') -> None:
        """执行阻止动作"""
        logging.error(f"BLOCK: Rule {self.rule_id} ({self.name}) blocked operation for event {event.event_id}")

    def _execute_notify_action(self, event: 'AuditEvent') -> None:
        """执行通知动作"""
        logging.info(f"NOTIFICATION: Rule {self.rule_id} ({self.name}) sent notification for event {event.event_id}")

    def _execute_escalate_action(self, event: 'AuditEvent') -> None:
        """执行升级动作"""
        logging.critical(f"ESCALATION: Rule {self.rule_id} ({self.name}) escalated event {event.event_id}")


class AuditRuleEngine:
    """审计规则引擎"""

    def __init__(self):
        self._rules: Dict[str, AuditRule] = {}
        self._rule_groups: Dict[str, List[str]] = {}
        self._lock = threading.RLock()
        self._stats = {
            'rules_evaluated': 0,
            'rules_triggered': 0,
            'actions_executed': 0,
            'total_evaluations': 0,
            'total_triggers': 0,
            'total_rules': 0,
            'enabled_rules': 0
        }

    def add_rule(self, rule: AuditRule) -> None:
        """添加规则"""
        with self._lock:
            self._rules[rule.rule_id] = rule
            logging.info(f"Added audit rule: {rule.name} ({rule.rule_id})")
            self._stats['total_rules'] = len(self._rules)
            self._stats['enabled_rules'] = len([r for r in self._rules.values() if r.enabled])

    def remove_rule(self, rule_id: str) -> bool:
        """移除规则"""
        with self._lock:
            if rule_id in self._rules:
                del self._rules[rule_id]
                self._stats['total_rules'] = len(self._rules)
                self._stats['enabled_rules'] = len([r for r in self._rules.values() if r.enabled])
                return True
            return False

    def enable_rule(self, rule_id: str) -> bool:
        """启用规则"""
        with self._lock:
            if rule_id in self._rules:
                self._rules[rule_id].enabled = True
                self._stats['enabled_rules'] = len([r for r in self._rules.values() if r.enabled])
                return True
            return False

    def disable_rule(self, rule_id: str) -> bool:
        """禁用规则"""
        with self._lock:
            if rule_id in self._rules:
                self._rules[rule_id].enabled = False
                self._stats['enabled_rules'] = len([r for r in self._rules.values() if r.enabled])
                return True
            return False

    def evaluate_event(self, event: 'AuditEvent') -> List[Dict[str, Any]]:
        """评估事件并执行匹配的规则"""
        with self._lock:
            results = []
            triggered_rules = []

            # 查找匹配的规则
            for rule in self._rules.values():
                self._stats['rules_evaluated'] += 1
                self._stats['total_evaluations'] += 1

                if rule.matches(event):
                    triggered_rules.append(rule)
                    self._stats['rules_triggered'] += 1
                    self._stats['total_triggers'] += 1

            # 按优先级排序并执行
            triggered_rules.sort(key=lambda r: r.priority, reverse=True)

            for rule in triggered_rules:
                rule_results = rule.execute_actions(event)
                results.extend(rule_results)
                self._stats['actions_executed'] += len(rule_results)

            return results

    def get_rule(self, rule_id: str) -> Optional[AuditRule]:
        """获取规则"""
        with self._lock:
            return self._rules.get(rule_id)

    def list_rules(self, enabled_only: bool = False) -> List[AuditRule]:
        """列出规则"""
        with self._lock:
            rules = list(self._rules.values())
            if enabled_only:
                rules = [r for r in rules if r.enabled]
            return rules

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            stats_copy = self._stats.copy()
            stats_copy['total_rules'] = len(self._rules)
            stats_copy['enabled_rules'] = len([r for r in self._rules.values() if r.enabled])
            return stats_copy

    def create_rule_group(self, group_name: str, rule_ids: List[str]) -> None:
        """创建规则组"""
        with self._lock:
            self._rule_groups[group_name] = rule_ids.copy()
            logging.info(f"Created rule group: {group_name} with {len(rule_ids)} rules")

    def evaluate_event_with_group(self, event: 'AuditEvent', group_name: str) -> List[Dict[str, Any]]:
        """使用规则组评估事件"""
        with self._lock:
            if group_name not in self._rule_groups:
                return []

            group_rule_ids = self._rule_groups[group_name]
            group_rules = [self._rules[rule_id] for rule_id in group_rule_ids if rule_id in self._rules]

            results = []
            for rule in group_rules:
                self._stats['rules_evaluated'] += 1
                self._stats['total_evaluations'] += 1

                if rule.enabled and rule.matches(event):
                    self._stats['rules_triggered'] += 1
                    self._stats['total_triggers'] += 1
                    rule_results = rule.execute_actions(event)
                    results.extend(rule_results)
                    self._stats['actions_executed'] += len(rule_results)

            return results

    def clear_stats(self) -> None:
        """清除统计信息"""
        with self._lock:
            self._stats = {
                'rules_evaluated': 0,
                'rules_triggered': 0,
                'actions_executed': 0,
                'total_evaluations': 0,
                'total_triggers': 0,
                'total_rules': len(self._rules),
                'enabled_rules': len([r for r in self._rules.values() if r.enabled])
            }


# 预定义规则模板
class AuditRuleTemplates:
    """审计规则模板"""

    @staticmethod
    def create_failed_login_rule() -> AuditRule:
        """创建失败登录规则"""
        return AuditRule(
            rule_id="failed_login_detection",
            name="Failed Login Detection",
            description="Detect repeated failed login attempts",
            conditions=[
                RuleCondition(RuleConditionType.EVENT_TYPE, "eq", AuditEventType.ACCESS),
                RuleCondition(RuleConditionType.ACTION, "eq", "login"),
                RuleCondition(RuleConditionType.RESOURCE, "contains", "/login"),
                RuleCondition(RuleConditionType.RESULT, "eq", "failure"),
            ],
            actions=[RuleAction.ALERT, RuleAction.LOG],
            priority=8,
            tags={"security", "authentication"}
        )

    @staticmethod
    def create_high_risk_operation_rule() -> AuditRule:
        """创建高风险操作规则"""
        return AuditRule(
            rule_id="high_risk_operation",
            name="High Risk Operation Detection",
            description="Monitor operations that are considered high risk",
            conditions=[
                RuleCondition(RuleConditionType.RISK_SCORE, "gte", 0.8),
                RuleCondition(RuleConditionType.SEVERITY, "in", [AuditSeverity.HIGH, AuditSeverity.CRITICAL])
            ],
            actions=[RuleAction.ESCALATE, RuleAction.NOTIFY, RuleAction.LOG],
            priority=9,
            tags={"security", "risk"}
        )

    @staticmethod
    def create_suspicious_resource_access_rule() -> AuditRule:
        """创建可疑资源访问规则"""
        return AuditRule(
            rule_id="suspicious_resource_access",
            name="Suspicious Resource Access Detection",
            description="Detect suspicious access to sensitive resources",
            conditions=[
                RuleCondition(RuleConditionType.RESOURCE, "regex", r".*(password|secret|admin).*"),
                RuleCondition(RuleConditionType.SEVERITY, "eq", AuditSeverity.HIGH)
            ],
            actions=[RuleAction.ALERT, RuleAction.LOG],
            priority=7,
            tags={"security", "resource"}
        )

    @staticmethod
    def create_compliance_violation_rule() -> AuditRule:
        """创建合规违规规则"""
        return AuditRule(
            rule_id="compliance_violation",
            name="Compliance Violation Detection",
            description="Detect potential compliance violations",
            conditions=[
                RuleCondition(RuleConditionType.EVENT_TYPE, "eq", AuditEventType.COMPLIANCE),
                RuleCondition(RuleConditionType.RESULT, "eq", "violation")
            ],
            actions=[RuleAction.ESCALATE, RuleAction.NOTIFY, RuleAction.LOG],
            priority=10,
            tags={"compliance", "violation"}
        )
