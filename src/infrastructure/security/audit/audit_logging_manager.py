#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 审计日志管理器

实现全面的操作审计和合规日志功能
提供安全监控、异常检测和合规报告
"""

import json
import logging
import threading
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from collections import defaultdict, deque
import re


class AuditEventType(Enum):

    """审计事件类型"""
    SECURITY = "security"      # 安全事件
    ACCESS = "access"         # 访问事件
    DATA_OPERATION = "data_operation"  # 数据操作
    CONFIG_CHANGE = "config_change"    # 配置变更
    USER_MANAGEMENT = "user_management"  # 用户管理
    SYSTEM_EVENT = "system_event"      # 系统事件
    COMPLIANCE = "compliance"   # 合规事件


class AuditSeverity(Enum):

    """审计事件严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """审计事件"""
    event_id: str
    event_type: AuditEventType
    severity: AuditSeverity
    timestamp: datetime
    user_id: Optional[str] = None
    action: str = ""
    resource: Optional[str] = None
    result: str = ""
    session_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    location: Optional[str] = None
    risk_score: float = 0.0
    tags: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'severity': self.severity.value,
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
            'session_id': self.session_id,
            'resource': self.resource,
            'action': self.action,
            'result': self.result,
            'details': self.details,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'location': self.location,
            'risk_score': self.risk_score,
            'tags': list(self.tags)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEvent':
        """从字典创建"""
        return cls(
            event_id=data['event_id'],
            event_type=AuditEventType(data['event_type']),
            severity=AuditSeverity(data['severity']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            user_id=data.get('user_id'),
            action=data.get('action', ""),
            resource=data.get('resource'),
            result=data.get('result', ""),
            session_id=data.get('session_id'),
            details=data.get('details', {}),
            ip_address=data.get('ip_address'),
            user_agent=data.get('user_agent'),
            location=data.get('location'),
            risk_score=data.get('risk_score', 0.0),
            tags=set(data.get('tags', []))
        )


@dataclass
class AuditRule:
    """审计规则"""
    rule_id: str
    name: str
    description: str = ""
    conditions: Dict[str, Any] = field(default_factory=dict)
    actions: List[str] = field(default_factory=list)
    enabled: bool = True
    priority: int = 0
    cooldown_period: int = 5
    severity_threshold: Optional[AuditSeverity] = None
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    event_pattern: Optional[Dict[str, Any]] = None  # 兼容旧参数命名
    cooldown_minutes: Optional[int] = None  # 兼容旧参数命名

    def __post_init__(self) -> None:
        # 兼容旧配置：event_pattern -> conditions
        if self.event_pattern and not self.conditions:
            self.conditions = dict(self.event_pattern)

        # 兼容旧参数 cooldown_minutes
        if self.cooldown_minutes is not None and self.cooldown_period == 5:
            self.cooldown_period = max(0, self.cooldown_minutes * 60)

        # 如果 actions 未指定，确保至少记录日志
        if not self.actions:
            self.actions = ["log"]

    def matches_event(self, event: AuditEvent) -> bool:
        """检查事件是否匹配规则"""
        if not self.enabled:
            return False

        cond = self.conditions or self.event_pattern or {}

        event_type_condition = cond.get('event_type')
        if event_type_condition:
            expected_type = (
                event_type_condition.value
                if isinstance(event_type_condition, AuditEventType)
                else str(event_type_condition)
            )
            if event.event_type.value != expected_type:
                return False

        severity_cond = cond.get('severity') or cond.get('min_severity')
        if severity_cond:
            target = (
                severity_cond
                if isinstance(severity_cond, AuditSeverity)
                else AuditSeverity(str(severity_cond))
            )
            if event.severity.value != target.value:
                return False

        result_cond = cond.get('result')
        if result_cond and event.result != result_cond:
                return False

        action_cond = cond.get('action')
        if action_cond and event.action != action_cond:
            return False

        resource_pattern = cond.get('resource_pattern')
        if resource_pattern and event.resource:
            if not re.match(str(resource_pattern), str(event.resource)):
                return False

        min_risk = cond.get('min_risk_score')
        if min_risk is not None and event.risk_score < float(min_risk):
            return False

        user_pattern = cond.get('user_pattern')
        if user_pattern and event.user_id:
            if not re.match(str(user_pattern), str(event.user_id)):
                return False

        return True

    def should_trigger(self, event: AuditEvent) -> bool:
        """检查是否应该触发规则"""
        if not self.matches_event(event):
            return False

        # 检查冷却时间
        if self.last_triggered:
            cooldown_end = self.last_triggered + timedelta(seconds=self.cooldown_period)
            if datetime.now() < cooldown_end:
                return False

        # 记录预触发时间，避免短时间内重复触发
        self.last_triggered = datetime.now()

        return True

    def trigger(self, event: AuditEvent) -> List[str]:
        """触发规则"""
        self.last_triggered = datetime.now()
        self.trigger_count += 1
        return self.actions.copy()


@dataclass
class ComplianceReport:
    """合规报告"""
    report_id: str
    report_type: str
    start_date: datetime
    end_date: datetime
    generated_at: datetime
    overall_status: str = "compliant"
    findings: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    compliance_score: float = 0.0
    risk_assessment: str = "low"
    report_period: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.report_period is None:
            self.report_period = {
                "start": self.start_date.isoformat(),
                "end": self.end_date.isoformat(),
                "days": max(1, (self.end_date - self.start_date).days or 1),
            }


class _RuleCollection(list):
    """用于保持测试期望的可变规则集合，同时确保与管理器状态同步"""

    def __init__(self, manager: "AuditLoggingManager") -> None:
        super().__init__()
        self._manager = manager

    def append(self, rule: AuditRule) -> None:  # type: ignore[override]
        self._manager._register_rule(rule)

    def extend(self, rules) -> None:  # type: ignore[override]
        for rule in rules:
            self._manager._register_rule(rule)

    def remove(self, rule: AuditRule) -> None:  # type: ignore[override]
        if rule.rule_id in self._manager.audit_rules:
            del self._manager.audit_rules[rule.rule_id]
        super().remove(rule)


class AuditLoggingManager:

    """
    审计日志管理器

    提供全面的审计和合规功能：
    - 结构化事件记录
    - 实时安全监控
    - 异常检测和告警
    - 合规报告生成
    - 审计数据分析
    """

    def __init__(self, log_path: Optional[str] = None, enable_realtime_monitoring: bool = True):
        """
        初始化审计日志管理器

        Args:
            log_path: 日志存储路径
            enable_realtime_monitoring: 是否启用实时监控
        """
        try:
            self.log_path = Path(log_path or "data/security/audit_logs")
            self.log_path.mkdir(parents=True, exist_ok=True)

            # 日志文件
            self.current_log_file = self.log_path / f"audit_{datetime.now().strftime('%Y%m%d')}.log"
            self.archive_path = self.log_path / "archive"

            # 审计规则
            self.audit_rules: Dict[str, AuditRule] = {}
            self.rules: _RuleCollection = _RuleCollection(self)
            self._initialize_default_rules()

            # 事件队列和处理
            self.event_queue: deque = deque(maxlen=10000)
            self.processed_events: deque = deque(maxlen=50000)

            # 统计信息
            self.event_stats = defaultdict(int)
            self.user_activity = defaultdict(lambda: defaultdict(int))
            self.resource_access = defaultdict(lambda: defaultdict(int))
            self.statistics: Dict[str, Any] = {
                "total_events": 0,
                "security_events": 0,
                "access_events": 0,
                "data_events": 0,
                "compliance_events": 0,
                "system_events": 0,
            }

            # 监控线程
            self.monitoring_thread: Optional[threading.Thread] = None
            self.stop_monitoring = threading.Event()

            if enable_realtime_monitoring:
                self._start_monitoring()

            logging.info("审计日志管理器初始化完成")

        except Exception as e:
            # 确保基本属性被初始化，即使初始化失败
            logging.error(f"审计日志管理器初始化失败: {e}")
            # 初始化基本属性以避免__del__出错
            self.monitoring_thread = None
            self.stop_monitoring = threading.Event()
            raise

    def _initialize_default_rules(self):
        """初始化默认审计规则"""
        default_rules_config = [
            {
                'rule_id': 'failed_login_alert',
                'name': '多次登录失败告警',
                'description': '检测用户多次登录失败，可能表示暴力破解攻击',
                'event_pattern': {
                    'event_type': 'security',
                    'result': 'failure',
                    'action': 'login'
                },
                'severity_threshold': AuditSeverity.HIGH,
                'actions': ['alert', 'log', 'notify'],
                'cooldown_minutes': 10
            },
            {
                'rule_id': 'suspicious_access',
                'name': '可疑访问模式',
                'description': '检测异常的访问模式，如非工作时间大量访问',
                'event_pattern': {
                    'event_type': 'access',
                    'min_risk_score': 0.7
                },
                'severity_threshold': AuditSeverity.MEDIUM,
                'actions': ['alert', 'log'],
                'cooldown_minutes': 15
            },
            {
                'rule_id': 'sensitive_data_access',
                'name': '敏感数据访问',
                'description': '监控对敏感数据的访问',
                'event_pattern': {
                    'event_type': 'data_operation',
                    'resource_pattern': r'.*sensitive.*|.*confidential.*'
                },
                'severity_threshold': AuditSeverity.MEDIUM,
                'actions': ['log', 'audit'],
                'cooldown_minutes': 1
            },
            {
                'rule_id': 'permission_change',
                'name': '权限变更监控',
                'description': '监控用户权限的变更',
                'event_pattern': {
                    'event_type': 'user_management',
                    'action': 'permission_change'
                },
                'severity_threshold': AuditSeverity.MEDIUM,
                'actions': ['log', 'notify'],
                'cooldown_minutes': 0
            }
        ]

        # 创建审计规则对象
        for rule_config in default_rules_config:
            rule = AuditRule(**rule_config)
            self._register_rule(rule)

    # =========================================================================
    # 事件记录
    # =========================================================================

    def log_event(
        self,
        event_type: AuditEventType,
        severity: AuditSeverity,
        user_id: Optional[str],
        action: str,
        result: str,
        resource: Optional[str] = None,
        session_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        location: Optional[str] = None,
        risk_score: float = 0.0,
        tags: Optional[Set[str]] = None,
        source_ip: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **extra_fields: Any,
    ) -> str:
        """
        记录审计事件

        Args:
            event_type: 事件类型
            severity: 严重程度
            user_id: 用户ID
            action: 操作
            result: 结果
            resource: 资源
            session_id: 会话ID
            details: 详细信息
            ip_address: IP地址
            user_agent: 用户代理
            location: 地理位置
            risk_score: 风险分数
            tags: 标签
            source_ip: 来源IP地址（别名）
            metadata: 额外元数据
            extra_fields: 其他附加字段

        Returns:
            事件ID
        """
        event_id = f"evt_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

        # 确保event_type是AuditEventType枚举
        if isinstance(event_type, str):
            event_type = AuditEventType(event_type)

        if isinstance(severity, str):
            severity = AuditSeverity(severity)

        event = AuditEvent(
            event_id=event_id,
            event_type=event_type,
            severity=severity,
            timestamp=datetime.now(),
            user_id=user_id,
            session_id=session_id,
            resource=resource,
            action=action,
            result=result,
            details={},
            ip_address=ip_address or source_ip,
            user_agent=user_agent,
            location=location,
            risk_score=risk_score,
            tags=tags or set()
        )

        if details:
            event.details.update(details)

        if metadata:
            event.details.setdefault("metadata", metadata)

        if extra_fields:
            event.details.setdefault("extra", {}).update(extra_fields)

        # 添加到队列
        self.event_queue.append(event)

        # 更新统计
        self._update_statistics(event)

        return event_id

    def log_security_event(self, user_id: Optional[str], action: str, result: str,


                           details: Optional[Dict[str, Any]] = None,
                           ip_address: Optional[str] = None, risk_score: float = 0.0) -> str:
        """
        记录安全事件

        Args:
            user_id: 用户ID
            action: 操作
            result: 结果
            details: 详细信息
            ip_address: IP地址
            risk_score: 风险分数

        Returns:
            事件ID
        """
        severity = AuditSeverity.HIGH if risk_score > 0.7 else AuditSeverity.MEDIUM
        return self.log_event(
            event_type=AuditEventType.SECURITY,
            severity=severity,
            user_id=user_id,
            action=action,
            result=result,
            details=details,
            ip_address=ip_address,
            risk_score=risk_score,
            tags={'security'}
        )

    def log_access_event(
        self,
        user_id: str,
        resource: str,
        action: str,
        result: str,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        risk_score: float = 0.0,
        details: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        记录访问事件

        Args:
            user_id: 用户ID
            resource: 资源
            action: 操作
            result: 结果
            session_id: 会话ID
            ip_address: IP地址
            risk_score: 风险分数

        Returns:
            事件ID
        """
        severity = AuditSeverity.HIGH if result == 'denied' and risk_score > 0.5 else AuditSeverity.LOW
        return self.log_event(
            event_type=AuditEventType.ACCESS,
            severity=severity,
            user_id=user_id,
            resource=resource,
            action=action,
            result=result,
            session_id=session_id,
            ip_address=ip_address,
             details=details,
            risk_score=risk_score,
            tags={'access'}
        )

    def log_data_operation(
        self,
        user_id: str,
        operation: str,
        resource: str,
        result: str,
        record_count: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        记录数据操作事件

        Args:
            user_id: 用户ID
            operation: 操作
            resource: 资源
            result: 结果
            details: 详细信息

        Returns:
            事件ID
        """
        resource_lower = resource.lower() if isinstance(resource, str) else ""
        risk_score = 0.3 if 'sensitive' in resource_lower else 0.1
        severity = AuditSeverity.MEDIUM if risk_score > 0.2 else AuditSeverity.LOW
        event_details = dict(details or {})
        if record_count is not None:
            event_details.setdefault("record_count", record_count)

        return self.log_event(
            event_type=AuditEventType.DATA_OPERATION,
            severity=severity,
            user_id=user_id,
            resource=resource,
            action=operation,
            result=result,
            details=event_details,
            risk_score=risk_score,
            tags={'data', 'operation'}
        )

    # =========================================================================
    # 事件处理和监控
    # =========================================================================

    def _start_monitoring(self):
        """启动监控"""
        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="audit_monitor"
        )
        self.monitoring_thread.start()

    def _monitoring_loop(self):
        """监控循环"""
        while not self.stop_monitoring.is_set():
            try:
                # 处理事件队列
                self._process_event_queue()

                # 检查审计规则
                self._check_audit_rules()

                # 定期归档
                self._check_log_rotation()

                # 等待
                threading.Event().wait(1)  # 1秒检查间隔

            except Exception as e:
                logging.error(f"审计监控错误: {e}")
                threading.Event().wait(5)

    def _process_event_queue(self):
        """处理事件队列"""
        while self.event_queue:
            event = self.event_queue.popleft()

            try:
                # 写入日志文件
                self._write_event_to_log(event)

                # 添加到已处理事件
                self.processed_events.append(event)

                # 限制已处理事件数量
                if len(self.processed_events) > 50000:
                    self.processed_events.popleft()

            except Exception as e:
                logging.error(f"处理审计事件失败: {e}")

    def _write_event_to_log(self, event: AuditEvent):
        """将事件写入日志文件"""
        try:
            # 确保日志目录存在（兼容临时目录被清空的场景）
            self.current_log_file.parent.mkdir(parents=True, exist_ok=True)
            if not self.current_log_file.exists():
                self.current_log_file.touch()

            with open(self.current_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event.to_dict(), ensure_ascii=False) + '\n')
        except Exception as e:
            logging.error(f"写入审计日志失败: {e}")

    def _check_audit_rules(self):
        """检查审计规则"""
        # 获取最近的事件
        recent_events = list(self.processed_events)[-10:]  # 检查最近10个事件

        for event in recent_events:
            for rule in self.audit_rules.values():
                if rule.should_trigger(event):
                    actions = rule.trigger(event)
                    self._execute_rule_actions(rule, event, actions)

    def _execute_rule_actions(self, rule: AuditRule, event: AuditEvent, actions: List[str]):
        """执行规则动作"""
        for action in actions:
            try:
                if action == 'alert':
                    self._send_alert(rule, event)
                elif action == 'log':
                    logging.warning(f"审计规则触发: {rule.name} - {event.action}")
                elif action == 'notify':
                    self._send_notification(rule, event)
                elif action == 'block':
                    self._block_action(event)
                elif action == 'audit':
                    self._escalate_audit(rule, event)
            except Exception as e:
                logging.error(f"执行规则动作失败: {action}, {e}")

    def _send_alert(self, rule: AuditRule, event: AuditEvent):
        """发送告警"""
        alert_message = f"审计规则告警: {rule.name}\n" \
            f"事件: {event.action}\n" \
            f"用户: {event.user_id}\n" \
            f"资源: {event.resource}\n" \
            f"时间: {event.timestamp.isoformat()}"

        logging.warning(alert_message)

        # 这里可以集成邮件、短信或其他告警系统
        # self.alert_system.send_alert(alert_message, severity=rule.severity_threshold.value)

    def _send_notification(self, rule: AuditRule, event: AuditEvent):
        """发送通知"""
        notification_message = f"审计通知: {rule.name} - {event.action}"
        logging.info(notification_message)

    def _block_action(self, event: AuditEvent):
        """阻止操作"""
        # 实现阻断逻辑

    def _escalate_audit(self, rule: AuditRule, event: AuditEvent):
        """升级审计"""
        # 增加监控级别或触发更详细的审计

    def _check_log_rotation(self):
        """检查日志轮转"""
        # 每天轮转日志文件
        current_date = datetime.now().strftime('%Y%m%d')
        expected_filename = f"audit_{current_date}.log"

        if self.current_log_file.name != expected_filename:
            # 创建归档目录
            self.archive_path.mkdir(exist_ok=True)

            # 移动当前日志文件到归档目录
            archive_name = f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            archive_file = self.archive_path / archive_name

            try:
                if self.current_log_file.exists():
                    self.current_log_file.rename(archive_file)
            except Exception as e:
                logging.error(f"日志轮转失败: {e}")

            # 创建新的日志文件
            self.current_log_file = self.log_path / expected_filename

    def _update_statistics(self, event: AuditEvent):
        """更新统计信息"""
        # 事件类型统计
        self.event_stats[event.event_type.value] += 1
        self.event_stats[f"{event.event_type.value}_{event.result}"] += 1

        # 全局统计
        self.statistics["total_events"] += 1
        if event.event_type == AuditEventType.SECURITY:
            self.statistics["security_events"] += 1
        elif event.event_type == AuditEventType.ACCESS:
            self.statistics["access_events"] += 1
        elif event.event_type == AuditEventType.DATA_OPERATION:
            self.statistics["data_events"] += 1
        elif event.event_type == AuditEventType.COMPLIANCE:
            self.statistics["compliance_events"] += 1
        elif event.event_type == AuditEventType.SYSTEM_EVENT:
            self.statistics["system_events"] += 1

        # 用户活动统计
        if event.user_id:
            self.user_activity[event.user_id][event.action] += 1
            self.user_activity[event.user_id][f"{event.event_type.value}_count"] += 1

        # 资源访问统计
        if event.resource:
            self.resource_access[event.resource][event.action] += 1
            self.resource_access[event.resource][f"{event.event_type.value}_count"] += 1

    # =========================================================================
    # 查询和分析
    # =========================================================================

    def query_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        result: Optional[str] = None,
        event_id: Optional[str] = None,
        severity: Optional[AuditSeverity] = None,
        tags: Optional[Set[str]] = None,
        limit: Optional[int] = None,
    ) -> List[AuditEvent]:
        """
        查询审计事件

        Args:
            start_time: 开始时间
            end_time: 结束时间
            event_type: 事件类型
            user_id: 用户ID
            resource: 资源
            result: 结果
            event_id: 事件ID
            severity: 严重程度
            tags: 标签过滤
            limit: 限制条数，为None时返回全部

        Returns:
            审计事件列表
        """
        # 步骤1: 初始化时间范围
        start_time_dt, end_time_dt = self._initialize_time_range(start_time, end_time)

        # 步骤2: 查询事件
        events = self._filter_events(
            start_time_dt,
            end_time_dt,
            event_type,
            user_id,
            resource,
            result,
            event_id,
            severity,
            tags,
            limit,
        )

        return events

    def _initialize_time_range(self, start_time: Optional[datetime], end_time: Optional[datetime]) -> tuple:
        """初始化时间范围"""
        if not start_time:
            start_time = datetime.now() - timedelta(days=7)
        if not end_time:
            end_time = datetime.now()
        return start_time, end_time

    def _filter_events(
        self,
        start_time: datetime,
        end_time: datetime,
        event_type: Optional[AuditEventType],
        user_id: Optional[str],
        resource: Optional[str],
        result: Optional[str],
        event_id: Optional[str],
        severity: Optional[AuditSeverity],
        tags: Optional[Set[str]],
        limit: Optional[int],
    ) -> List[AuditEvent]:
        """过滤事件"""
        events = []
        if isinstance(severity, str):
            severity = AuditSeverity(severity)
        if tags and not isinstance(tags, set):
            tags = set(tags)

        # 从已处理事件中查询
        for event in reversed(self.processed_events):
            if limit is not None and len(events) >= limit:
                break

            if not self._is_event_in_time_range(event, start_time, end_time):
                continue

            if event_id and event.event_id != event_id:
                continue

            if not self._matches_event_type(event, event_type):
                continue

            if not self._matches_user_id(event, user_id):
                continue

            if not self._matches_resource(event, resource):
                continue

            if not self._matches_result(event, result):
                continue

            if severity and event.severity != severity:
                continue

            if tags:
                if not event.tags:
                    continue
                if not set(tags).issubset(event.tags):
                    continue

            events.append(event)

        return events

    def _is_event_in_time_range(self, event: AuditEvent, start_time: datetime, end_time: datetime) -> bool:
        """检查事件是否在时间范围内"""
        return start_time <= event.timestamp <= end_time

    def _matches_event_type(self, event: AuditEvent, event_type: Optional[AuditEventType]) -> bool:
        """检查事件类型是否匹配"""
        if event_type is None:
            return True
        if isinstance(event_type, str):
            event_type = AuditEventType(event_type)
        return event.event_type == event_type

    def _matches_user_id(self, event: AuditEvent, user_id: Optional[str]) -> bool:
        """检查用户ID是否匹配"""
        return not user_id or event.user_id == user_id

    def _matches_resource(self, event: AuditEvent, resource: Optional[str]) -> bool:
        """检查资源是否匹配"""
        return not resource or event.resource == resource

    def _matches_result(self, event: AuditEvent, result: Optional[str]) -> bool:
        """检查结果是否匹配"""
        return not result or event.result == result

    def get_security_report(self, days: int = 7) -> Dict[str, Any]:
        """
        获取安全报告

        Args:
            days: 报告天数

        Returns:
            安全报告
        """
        # 步骤1: 初始化时间范围
        start_time, end_time = self._calculate_report_time_range(days)

        # 步骤2: 查询事件
        events = self.query_events(start_time=start_time, end_time=end_time)

        # 步骤3: 分析安全事件
        security_events, failed_logins, suspicious_access = self._analyze_security_events(events)

        # 步骤4: 评估用户风险
        user_risks = self._assess_user_risks(events)

        # 步骤5: 生成报告
        return self._generate_security_report_dict(start_time, end_time, days, events, 
                                                  security_events, failed_logins, suspicious_access, user_risks)

    def _calculate_report_time_range(self, days: int) -> tuple:
        """计算报告时间范围"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        return start_time, end_time

    def _analyze_security_events(self, events: List[AuditEvent]) -> tuple:
        """分析安全事件"""
        security_events = [e for e in events if e.event_type == AuditEventType.SECURITY]
        failed_logins = [e for e in events if e.action == 'login' and e.result == 'failure']
        suspicious_access = [e for e in events if e.risk_score > 0.7]
        return security_events, failed_logins, suspicious_access

    def _assess_user_risks(self, events: List[AuditEvent]) -> Dict[str, Dict[str, Any]]:
        """评估用户风险"""
        user_risks = {}
        for event in events:
            if event.user_id:
                if event.user_id not in user_risks:
                    user_risks[event.user_id] = {'events': 0, 'failures': 0, 'risk_score': 0.0}

                user_risks[event.user_id]['events'] += 1
                if event.result == 'failure':
                    user_risks[event.user_id]['failures'] += 1

                # 计算风险分数
                failure_rate = user_risks[event.user_id]['failures'] / \
                    user_risks[event.user_id]['events']
                user_risks[event.user_id]['risk_score'] = min(failure_rate * 2, 1.0)
        return user_risks

    def _generate_security_report_dict(
        self,
        start_time: datetime,
        end_time: datetime,
        days: int,
        events: List[AuditEvent],
        security_events: List[AuditEvent],
        failed_logins: List[AuditEvent],
        suspicious_access: List[AuditEvent],
        user_risks: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """生成安全报告字典"""
        summary = {
            'total_events': len(events),
            'security_events': len(security_events),
            'failed_logins': len(failed_logins),
            'suspicious_access': len(suspicious_access),
        }
        risk_section = {
            'high_risk_users': [uid for uid, risk in user_risks.items() if risk['risk_score'] > 0.7],
            'medium_risk_users': [uid for uid, risk in user_risks.items() if 0.4 < risk['risk_score'] <= 0.7],
            'user_risks': user_risks,
        }
        recommendations = self._generate_security_recommendations(summary, risk_section)

        return {
            'report_period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'days': days,
            },
            'generated_at': datetime.now().isoformat(),
            'total_events': summary['total_events'],
            'security_events': summary['security_events'],
            'failed_logins': summary['failed_logins'],
            'suspicious_access': summary['suspicious_access'],
            'summary': summary,
            'risk_assessment': risk_section,
            'recommendations': recommendations,
        }

    def _generate_security_recommendations(
        self,
        summary: Dict[str, Any],
        risk_section: Dict[str, Any],
    ) -> List[str]:
        """基于统计结果生成安全建议"""
        recommendations: List[str] = []

        if summary['failed_logins'] > 0:
            recommendations.append("审查登录失败记录，必要时启用多因素认证")

        if summary['suspicious_access'] > 0:
            recommendations.append("增加对可疑访问的监控并验证关联用户身份")

        if risk_section['high_risk_users']:
            recommendations.append("对高风险用户执行额外安全审查和提醒")

        if not recommendations:
            recommendations.append("未发现显著风险，继续保持现有安全策略")

        return recommendations

    def get_compliance_report(self, report_type: str = "general", days: int = 30) -> ComplianceReport:
        """
        生成合规报告

        Args:
            report_type: 报告类型
            days: 报告天数

        Returns:
            合规报告
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        # 获取事件数据
        events = self.query_events(start_time=start_time, end_time=end_time, limit=None)

        # 执行合规检查
        findings = self._perform_compliance_checks(events, report_type)
        if report_type == "detailed" and not findings:
            findings.append({
                "type": "summary",
                "severity": "low",
                "description": "未发现显著合规问题",
                "recommendation": "保持现有安全与合规流程"
            })

        metrics = self._calculate_compliance_metrics(events)
        compliance_score = self._calculate_compliance_score(findings, events)
        metrics["compliance_score"] = compliance_score

        # 风险评估
        risk_assessment = self._assess_compliance_risk(compliance_score)
        recommendations = self._generate_compliance_recommendations(findings)
        overall_status = (
            "compliant" if compliance_score >= 90 else
            "warning" if compliance_score >= 70 else
            "critical"
        )

        return ComplianceReport(
            report_id=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            report_type=report_type,
            start_date=start_time,
            end_date=end_time,
            generated_at=datetime.now(),
            overall_status=overall_status,
            findings=findings,
            metrics=metrics,
            recommendations=recommendations,
            compliance_score=compliance_score,
            risk_assessment=risk_assessment
        )

    def _perform_compliance_checks(self, events: List[AuditEvent], report_type: str) -> List[Dict[str, Any]]:
        """执行合规检查"""
        findings = []

        if report_type in ["general", "security"]:
            findings.extend(self._check_security_compliance(events))
            findings.extend(self._check_data_protection_compliance(events))

        if report_type in ["general", "audit"]:
            findings.extend(self._check_audit_compliance(events))

        return findings

    def _calculate_compliance_metrics(self, events: List[AuditEvent]) -> Dict[str, Any]:
        """计算合规指标"""
        metrics: Dict[str, Any] = {
            "total_events": len(events),
            "security_events": 0,
            "access_events": 0,
            "data_operations": 0,
            "compliance_events": 0,
            "policy_updates": 0,
            "failed_events": 0,
        }

        for event in events:
            if event.event_type == AuditEventType.SECURITY:
                metrics["security_events"] += 1
            elif event.event_type == AuditEventType.ACCESS:
                metrics["access_events"] += 1
            elif event.event_type == AuditEventType.DATA_OPERATION:
                metrics["data_operations"] += 1
            elif event.event_type == AuditEventType.COMPLIANCE:
                metrics["compliance_events"] += 1

            if event.result in {"failure", "denied", "error"}:
                metrics["failed_events"] += 1

            if event.action and "policy" in event.action:
                metrics["policy_updates"] += 1

        return metrics

    def _check_security_compliance(self, events: List[AuditEvent]) -> List[Dict[str, Any]]:
        """检查安全合规性"""
        findings = []

        # 访问失败率检查
        failed_access = len([e for e in events if e.result == 'denied'])
        if failed_access > len(events) * 0.1:  # 失败访问超过10%
            findings.append({
                'type': 'security',
                'severity': 'high',
                'description': f'访问失败率过高: {failed_access}/{len(events)}',
                'recommendation': '检查访问控制策略和用户权限配置'
            })

        # 高风险事件检查
        high_risk_events = [e for e in events if e.risk_score > 0.8]
        if high_risk_events:
            findings.append({
                'type': 'security',
                'severity': 'high',
                'description': f'检测到 {len(high_risk_events)} 个高风险事件',
                'recommendation': '立即调查高风险事件并加强监控'
            })

        return findings

    def _check_data_protection_compliance(self, events: List[AuditEvent]) -> List[Dict[str, Any]]:
        """检查数据保护合规性"""
        findings = []

        # 敏感数据访问检查
        sensitive_access = [e for e in events if 'sensitive' in str(e.resource).lower()]
        if sensitive_access:
            findings.append({
                'type': 'data_protection',
                'severity': 'medium',
                'description': f'检测到 {len(sensitive_access)} 次敏感数据访问',
                'recommendation': '确保敏感数据访问有适当的审计和控制'
            })

        return findings

    def _check_audit_compliance(self, events: List[AuditEvent]) -> List[Dict[str, Any]]:
        """检查审计合规性"""
        findings = []

        # 审计覆盖率检查
        unique_resources = set(e.resource for e in events if e.resource)
        if len(unique_resources) < 5:  # 假设至少应该有5个不同的资源
            findings.append({
                'type': 'audit',
                'severity': 'medium',
                'description': f'审计覆盖的资源类型较少: {len(unique_resources)}',
                'recommendation': '扩展审计规则以覆盖更多资源类型'
            })

        return findings

    def _generate_compliance_recommendations(self, findings: List[Dict[str, Any]]) -> List[str]:
        """生成合规建议"""
        recommendations = []

        if not findings:
            recommendations.append("合规状态良好，继续保持当前的安全实践")
            return recommendations

        # 基于发现的问题生成建议
        high_severity_findings = [f for f in findings if f.get('severity') == 'high']
        if high_severity_findings:
            recommendations.append("优先处理高严重性问题，立即采取纠正措施")

        if len(findings) > 5:
            recommendations.append("审查和优化审计规则配置，减少误报")

        # 基于发现类型生成特定建议
        security_findings = [f for f in findings if f.get('type') == 'security']
        if security_findings:
            recommendations.append("加强安全监控和访问控制策略")

        data_protection_findings = [f for f in findings if f.get('type') == 'data_protection']
        if data_protection_findings:
            recommendations.append("加强敏感数据保护措施和访问控制")

        return recommendations

    def _calculate_compliance_score(self, findings: List[Dict[str, Any]], events: List[AuditEvent]) -> float:
        """计算合规分数"""
        if not events:
            return 100.0

        base_score = 100.0

        # 根据发现的问题扣分
        for finding in findings:
            severity = finding.get('severity', 'low')
            if severity == 'critical':
                base_score -= 30
            elif severity == 'high':
                base_score -= 20
            elif severity == 'medium':
                base_score -= 10
            elif severity == 'low':
                base_score -= 5

        # 确保分数在0-100之间
        return max(0.0, min(100.0, base_score))

    def _assess_compliance_risk(self, compliance_score: float) -> str:
        """评估合规风险"""
        if compliance_score >= 90:
            return "low"
        elif compliance_score >= 70:
            return "medium"
        elif compliance_score >= 50:
            return "high"
        else:
            return "critical"

    # =========================================================================
    # 审计规则管理
    # =========================================================================

    def add_audit_rule(self, rule: AuditRule):
        """
        添加审计规则

        Args:
            rule: 审计规则
        """
        self._register_rule(rule)
        logging.info(f"审计规则已添加: {rule.name}")

    def remove_audit_rule(self, rule_id: str):
        """
        移除审计规则

        Args:
            rule_id: 规则ID
        """
        rule = self.audit_rules.pop(rule_id, None)
        if rule:
            try:
                list.remove(self.rules, rule)
            except ValueError:
                pass
            logging.info(f"审计规则已移除: {rule_id}")

    def enable_audit_rule(self, rule_id: str):
        """
        启用审计规则

        Args:
            rule_id: 规则ID
        """
        if rule_id in self.audit_rules:
            self.audit_rules[rule_id].enabled = True
            logging.info(f"审计规则已启用: {rule_id}")

    def disable_audit_rule(self, rule_id: str):
        """
        禁用审计规则

        Args:
            rule_id: 规则ID
        """
        if rule_id in self.audit_rules:
            self.audit_rules[rule_id].enabled = False
            logging.info(f"审计规则已禁用: {rule_id}")

    # =========================================================================
    # 内部辅助方法
    # =========================================================================

    def _register_rule(self, rule: AuditRule) -> None:
        """内部注册规则，保持列表与字典同步"""
        self.audit_rules[rule.rule_id] = rule
        if rule not in self.rules:
            list.append(self.rules, rule)

    # =========================================================================
    # 统计和报告
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息

        Returns:
            统计信息
        """
        return {
            'event_statistics': dict(self.event_stats),
            'user_activity': dict(self.user_activity),
            'resource_access': dict(self.resource_access),
            'audit_rules': {
                'total': len(self.audit_rules),
                'enabled': len([r for r in self.audit_rules.values() if r.enabled]),
                'triggered': sum(r.trigger_count for r in self.audit_rules.values())
            },
            'queue_status': {
                'pending_events': len(self.event_queue),
                'processed_events': len(self.processed_events)
            },
            'timestamp': datetime.now().isoformat()
        }

    def cleanup_old_logs(self, days_to_keep: int = 90):
        """
        清理旧的日志文件

        Args:
            days_to_keep: 保留天数
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        try:
            for log_file in self.log_path.glob("audit_*.log"):
                if log_file != self.current_log_file:
                    # 检查文件修改时间
                    file_date_str = log_file.stem.split('_')[1]  # audit_YYYYMMDD
                    try:
                        file_date = datetime.strptime(file_date_str, '%Y % m % d')
                        if file_date < cutoff_date:
                            log_file.unlink()
                            logging.info(f"清理旧日志文件: {log_file.name}")
                    except ValueError:
                        continue

        except Exception as e:
            logging.error(f"清理日志文件失败: {e}")

    def shutdown(self):
        """关闭审计日志管理器"""
        # 停止监控
        self.stop_monitoring.set()
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)

        # 处理剩余事件
        self._process_event_queue()

        logging.info("审计日志管理器已关闭")

    def __del__(self):
        """析构函数"""
        self.shutdown()
