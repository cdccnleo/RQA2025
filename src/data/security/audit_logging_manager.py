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
    user_id: Optional[str]
    session_id: Optional[str]
    resource: Optional[str]
    action: str
    result: str  # "success", "failure", "denied", etc.
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
            session_id=data.get('session_id'),
            resource=data.get('resource'),
            action=data['action'],
            result=data['result'],
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
    description: str
    event_pattern: Dict[str, Any]  # 事件匹配模式
    severity_threshold: AuditSeverity
    actions: List[str]  # 触发动作：alert, block, log, notify
    enabled: bool = True
    cooldown_minutes: int = 5
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0

    def matches_event(self, event: AuditEvent) -> bool:
        """检查事件是否匹配规则"""
        if not self.enabled:
            return False

        # 检查事件类型
        if 'event_type' in self.event_pattern:
            if event.event_type.value != self.event_pattern['event_type']:
                return False

        # 检查严重程度
        if 'min_severity' in self.event_pattern:
            min_severity = AuditSeverity(self.event_pattern['min_severity'])
            if event.severity.value not in ['high', 'critical'] and event.severity != min_severity:
                return False

        # 检查结果
        if 'result' in self.event_pattern:
            if event.result != self.event_pattern['result']:
                return False

        # 检查资源模式
        if 'resource_pattern' in self.event_pattern and event.resource:
            pattern = self.event_pattern['resource_pattern']
            if not re.match(pattern, event.resource):
                return False

        # 检查风险分数
        if 'min_risk_score' in self.event_pattern:
            if event.risk_score < self.event_pattern['min_risk_score']:
                return False

        return True

    def should_trigger(self, event: AuditEvent) -> bool:
        """检查是否应该触发规则"""
        if not self.matches_event(event):
            return False

        # 检查冷却时间
        if self.last_triggered:
            cooldown_end = self.last_triggered + timedelta(minutes=self.cooldown_minutes)
            if datetime.now() < cooldown_end:
                return False

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
    period_start: datetime
    period_end: datetime
    generated_at: datetime
    findings: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    compliance_score: float = 0.0
    risk_assessment: str = "low"


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
        self.log_path = Path(log_path or "data/security/audit_logs")
        self.log_path.mkdir(parents=True, exist_ok=True)

        # 日志文件
        self.current_log_file = self.log_path / f"audit_{datetime.now().strftime('%Y%m%d')}.log"
        self.archive_path = self.log_path / "archive"

        # 审计规则
        self.audit_rules: Dict[str, AuditRule] = {}
        self._initialize_default_rules()

        # 事件队列和处理
        self.event_queue: deque = deque(maxlen=10000)
        self.processed_events: deque = deque(maxlen=50000)

        # 统计信息
        self.event_stats = defaultdict(int)
        self.user_activity = defaultdict(lambda: defaultdict(int))
        self.resource_access = defaultdict(lambda: defaultdict(int))

        # 监控线程
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()

        if enable_realtime_monitoring:
            self._start_monitoring()

        logging.info("审计日志管理器初始化完成")

    def _initialize_default_rules(self):
        """初始化默认审计规则"""
        # 失败登录告警
        self.audit_rules['failed_login_alert'] = AuditRule(
            rule_id='failed_login_alert',
            name='多次登录失败告警',
            description='检测用户多次登录失败，可能表示暴力破解攻击',
            event_pattern={
                'event_type': 'security',
                'result': 'failure',
                'action': 'login'
            },
            severity_threshold=AuditSeverity.HIGH,
            actions=['alert', 'log', 'notify'],
            cooldown_minutes=10
        )

        # 异常访问模式
        self.audit_rules['suspicious_access'] = AuditRule(
            rule_id='suspicious_access',
            name='可疑访问模式',
            description='检测异常的访问模式，如非工作时间大量访问',
            event_pattern={
                'event_type': 'access',
                'min_risk_score': 0.7
            },
            severity_threshold=AuditSeverity.MEDIUM,
            actions=['alert', 'log'],
            cooldown_minutes=15
        )

        # 敏感数据访问
        self.audit_rules['sensitive_data_access'] = AuditRule(
            rule_id='sensitive_data_access',
            name='敏感数据访问',
            description='监控对敏感数据的访问',
            event_pattern={
                'event_type': 'data_operation',
                'resource_pattern': r'.*sensitive.*|.*confidential.*'
            },
            severity_threshold=AuditSeverity.MEDIUM,
            actions=['log', 'audit'],
            cooldown_minutes=1
        )

        # 权限变更
        self.audit_rules['permission_change'] = AuditRule(
            rule_id='permission_change',
            name='权限变更监控',
            description='监控用户权限的变更',
            event_pattern={
                'event_type': 'user_management',
                'action': 'permission_change'
            },
            severity_threshold=AuditSeverity.MEDIUM,
            actions=['log', 'notify'],
            cooldown_minutes=0
        )

    # =========================================================================
    # 事件记录
    # =========================================================================

    def log_event(self, event_type: AuditEventType, severity: AuditSeverity,


                  user_id: Optional[str], action: str, result: str,
                  resource: Optional[str] = None, session_id: Optional[str] = None,
                  details: Optional[Dict[str, Any]] = None,
                  ip_address: Optional[str] = None, user_agent: Optional[str] = None,
                  location: Optional[str] = None, risk_score: float = 0.0,
                  tags: Optional[Set[str]] = None) -> str:
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

        Returns:
            事件ID
        """
        event_id = f"evt_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

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
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent,
            location=location,
            risk_score=risk_score,
            tags=tags or set()
        )

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

    def log_access_event(self, user_id: str, resource: str, action: str, result: str,


                         session_id: Optional[str] = None, ip_address: Optional[str] = None,
                         risk_score: float = 0.0) -> str:
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
            risk_score=risk_score,
            tags={'access'}
        )

    def log_data_operation(self, user_id: str, operation: str, resource: str,


                           result: str, details: Optional[Dict[str, Any]] = None) -> str:
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
        risk_score = 0.3 if 'sensitive' in resource.lower() else 0.1
        severity = AuditSeverity.MEDIUM if risk_score > 0.2 else AuditSeverity.LOW

        return self.log_event(
            event_type=AuditEventType.DATA_OPERATION,
            severity=severity,
            user_id=user_id,
            resource=resource,
            action=operation,
            result=result,
            details=details,
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
            with open(self.current_log_file, 'a', encoding='utf - 8') as f:
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
        # 实现通知逻辑

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

    def query_events(self, start_time: Optional[datetime] = None,


                     end_time: Optional[datetime] = None,
                     event_type: Optional[AuditEventType] = None,
                     user_id: Optional[str] = None,
                     resource: Optional[str] = None,
                     result: Optional[str] = None,
                     limit: int = 100) -> List[AuditEvent]:
        """
        查询审计事件

        Args:
            start_time: 开始时间
            end_time: 结束时间
            event_type: 事件类型
            user_id: 用户ID
            resource: 资源
            result: 结果
            limit: 限制条数

        Returns:
            审计事件列表
        """
        if not start_time:
            start_time = datetime.now() - timedelta(days=7)
        if not end_time:
            end_time = datetime.now()

        events = []

        # 从已处理事件中查询
        for event in reversed(self.processed_events):
            if len(events) >= limit:
                break

            if event.timestamp < start_time or event.timestamp > end_time:
                continue

            if event_type and event.event_type != event_type:
                continue

            if user_id and event.user_id != user_id:
                continue

            if resource and event.resource != resource:
                continue

            if result and event.result != result:
                continue

            events.append(event)

        return events

    def get_security_report(self, days: int = 7) -> Dict[str, Any]:
        """
        获取安全报告

        Args:
            days: 报告天数

        Returns:
            安全报告
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        events = self.query_events(start_time=start_time, end_time=end_time)

        # 分析安全事件
        security_events = [e for e in events if e.event_type == AuditEventType.SECURITY]
        failed_logins = [e for e in events if e.action == 'login' and e.result == 'failure']
        suspicious_access = [e for e in events if e.risk_score > 0.7]

        # 用户风险评估
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

        return {
            'period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'days': days
            },
            'summary': {
                'total_events': len(events),
                'security_events': len(security_events),
                'failed_logins': len(failed_logins),
                'suspicious_access': len(suspicious_access)
            },
            'risk_assessment': {
                'high_risk_users': [uid for uid, risk in user_risks.items() if risk['risk_score'] > 0.7],
                'medium_risk_users': [uid for uid, risk in user_risks.items() if 0.4 < risk['risk_score'] <= 0.7],
                'user_risks': user_risks
            },
            'generated_at': datetime.now().isoformat()
        }

    def get_compliance_report(self, report_type: str = "general",


                              days: int = 30) -> ComplianceReport:
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

        events = self.query_events(start_time=start_time, end_time=end_time)

        findings = []
        recommendations = []
        compliance_score = 100.0

        # 检查各种合规要求
        if report_type == "general" or report_type == "security":
            # 安全合规检查
            failed_access = len([e for e in events if e.result == 'denied'])
            if failed_access > len(events) * 0.1:  # 失败访问超过10%
                findings.append({
                    'type': 'security',
                    'severity': 'high',
                    'description': f'访问失败率过高: {failed_access}/{len(events)}',
                    'recommendation': '检查访问控制策略和用户权限配置'
                })
                compliance_score -= 20

            # 敏感数据访问检查
            sensitive_access = [e for e in events if 'sensitive' in str(e.resource).lower()]
            if len(sensitive_access) > 0:
                findings.append({
                    'type': 'data_protection',
                    'severity': 'medium',
                    'description': f'检测到 {len(sensitive_access)} 次敏感数据访问',
                    'recommendation': '确保敏感数据访问有适当的审计和控制'
                })

        # 生成建议
        if compliance_score < 80:
            recommendations.append("加强安全监控和访问控制")
        if len(findings) > 5:
            recommendations.append("审查和优化审计规则配置")

        # 风险评估
        risk_assessment = "low"
        if compliance_score < 60:
            risk_assessment = "high"
        elif compliance_score < 80:
            risk_assessment = "medium"

        return ComplianceReport(
            report_id=f"report_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            report_type=report_type,
            period_start=start_time,
            period_end=end_time,
            generated_at=datetime.now(),
            findings=findings,
            recommendations=recommendations,
            compliance_score=compliance_score,
            risk_assessment=risk_assessment
        )

    # =========================================================================
    # 审计规则管理
    # =========================================================================

    def add_audit_rule(self, rule: AuditRule):
        """
        添加审计规则

        Args:
            rule: 审计规则
        """
        self.audit_rules[rule.rule_id] = rule
        logging.info(f"审计规则已添加: {rule.name}")

    def remove_audit_rule(self, rule_id: str):
        """
        移除审计规则

        Args:
            rule_id: 规则ID
        """
        if rule_id in self.audit_rules:
            del self.audit_rules[rule_id]
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
                        file_date = datetime.strptime(file_date_str, '%Y%m%d')
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
