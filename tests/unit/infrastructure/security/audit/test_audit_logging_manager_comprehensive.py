#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
审计日志管理器综合测试
测试AuditLoggingManager的核心功能、审计规则、合规报告等
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import json
import time
import threading
from unittest.mock import patch, MagicMock, mock_open
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import os

from src.infrastructure.security.audit.audit_logging_manager import (
    AuditEventType, AuditSeverity, AuditEvent, AuditRule,
    ComplianceReport, AuditLoggingManager
)


@pytest.fixture
def temp_log_dir():
    """创建临时日志目录"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # 清理
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def audit_manager(temp_log_dir):
    """创建审计日志管理器实例"""
    manager = AuditLoggingManager(
        log_path=temp_log_dir,
        enable_realtime_monitoring=False  # 测试时禁用实时监控
    )
    return manager


class TestAuditEventType:
    """测试AuditEventType枚举"""

    def test_audit_event_type_values(self):
        """测试审计事件类型枚举值"""
        assert AuditEventType.SECURITY.value == "security"
        assert AuditEventType.ACCESS.value == "access"
        assert AuditEventType.DATA_OPERATION.value == "data_operation"
        assert AuditEventType.CONFIG_CHANGE.value == "config_change"
        assert AuditEventType.USER_MANAGEMENT.value == "user_management"
        assert AuditEventType.SYSTEM_EVENT.value == "system_event"
        assert AuditEventType.COMPLIANCE.value == "compliance"

    def test_audit_event_type_membership(self):
        """测试审计事件类型成员"""
        event_types = [
            AuditEventType.SECURITY, AuditEventType.ACCESS,
            AuditEventType.DATA_OPERATION, AuditEventType.CONFIG_CHANGE,
            AuditEventType.USER_MANAGEMENT, AuditEventType.SYSTEM_EVENT,
            AuditEventType.COMPLIANCE
        ]
        assert len(event_types) == 7
        assert AuditEventType.SECURITY in event_types


class TestAuditSeverity:
    """测试AuditSeverity枚举"""

    def test_audit_severity_values(self):
        """测试审计严重程度枚举值"""
        assert AuditSeverity.LOW.value == "low"
        assert AuditSeverity.MEDIUM.value == "medium"
        assert AuditSeverity.HIGH.value == "high"
        assert AuditSeverity.CRITICAL.value == "critical"

    def test_audit_severity_order(self):
        """测试审计严重程度顺序"""
        severities = [AuditSeverity.LOW, AuditSeverity.MEDIUM, AuditSeverity.HIGH, AuditSeverity.CRITICAL]

        # 验证所有严重程度值都不同
        values = [s.value for s in severities]
        assert len(set(values)) == len(values)

        # 验证枚举成员存在
        assert AuditSeverity.LOW in severities
        assert AuditSeverity.CRITICAL in severities


class TestAuditEvent:
    """测试AuditEvent类"""

    def test_audit_event_creation(self):
        """测试审计事件创建"""
        event = AuditEvent(
            event_id="test_event_123",
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.HIGH,
            timestamp=datetime.now(),
            user_id="user123",
            action="login",
            resource="system",
            result="success",
            details={"ip": "192.168.1.1", "user_agent": "Test Browser"},
            ip_address="192.168.1.1",
            session_id="session_123"
        )

        assert event.event_id == "test_event_123"
        assert event.event_type == AuditEventType.SECURITY
        assert event.severity == AuditSeverity.HIGH
        assert event.user_id == "user123"
        assert event.action == "login"
        assert event.resource == "system"
        assert event.result == "success"
        assert event.details == {"ip": "192.168.1.1", "user_agent": "Test Browser"}
        assert event.ip_address == "192.168.1.1"
        assert event.session_id == "session_123"

    def test_audit_event_to_dict(self):
        """测试审计事件序列化"""
        timestamp = datetime.now()
        event = AuditEvent(
            event_id="test_event_123",
            event_type=AuditEventType.ACCESS,
            severity=AuditSeverity.MEDIUM,
            timestamp=timestamp,
            user_id="user123",
            action="read",
            resource="file.txt",
            result="success"
        )

        event_dict = event.to_dict()
        assert event_dict["event_id"] == "test_event_123"
        assert event_dict["event_type"] == "access"
        assert event_dict["severity"] == "medium"
        assert event_dict["user_id"] == "user123"
        assert event_dict["action"] == "read"
        assert event_dict["resource"] == "file.txt"
        assert event_dict["result"] == "success"

    def test_audit_event_from_dict(self):
        """测试从字典创建审计事件"""
        timestamp_str = "2025-10-27T10:00:00"
        event_dict = {
            "event_id": "test_event_123",
            "event_type": "security",
            "severity": "high",
            "timestamp": timestamp_str,
            "user_id": "user123",
            "action": "login",
            "resource": "system",
            "result": "success",
            "details": {"ip": "192.168.1.1"},
            "source_ip": "192.168.1.1",
            "session_id": "session_123"
        }

        event = AuditEvent.from_dict(event_dict)
        assert event.event_id == "test_event_123"
        assert event.event_type == AuditEventType.SECURITY
        assert event.severity == AuditSeverity.HIGH
        assert event.user_id == "user123"
        assert event.action == "login"
        assert event.result == "success"


class TestAuditRule:
    """测试AuditRule类"""

    def test_audit_rule_creation(self):
        """测试审计规则创建"""
        conditions = {
            "event_type": AuditEventType.SECURITY,
            "severity": AuditSeverity.HIGH,
            "user_pattern": "admin_*",
            "resource_pattern": "system/*",
            "time_window": 3600
        }

        rule = AuditRule(
            rule_id="test_rule_001",
            name="High Security Events",
            description="Monitor high severity security events",
            conditions=conditions,
            actions=["alert", "log", "notify"],
            enabled=True,
            priority=1
        )

        assert rule.rule_id == "test_rule_001"
        assert rule.name == "High Security Events"
        assert rule.description == "Monitor high severity security events"
        assert rule.conditions == conditions
        assert rule.actions == ["alert", "log", "notify"]
        assert rule.enabled is True
        assert rule.priority == 1

    def test_audit_rule_matches_event(self):
        """测试审计规则匹配事件"""
        rule = AuditRule(
            rule_id="test_rule",
            name="Security Alert",
            description="Alert on security events",
            conditions={
                "event_type": AuditEventType.SECURITY,
                "severity": AuditSeverity.HIGH
            },
            actions=["alert"],
            enabled=True
        )

        # 匹配的事件
        matching_event = AuditEvent(
            event_id="event1",
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.HIGH,
            timestamp=datetime.now(),
            user_id="user1",
            action="login",
            resource="system",
            result="failed"
        )

        # 不匹配的事件
        non_matching_event = AuditEvent(
            event_id="event2",
            event_type=AuditEventType.ACCESS,
            severity=AuditSeverity.LOW,
            timestamp=datetime.now(),
            user_id="user2",
            action="read",
            resource="file.txt",
            result="success"
        )

        assert rule.matches_event(matching_event)
        assert not rule.matches_event(non_matching_event)

    def test_audit_rule_should_trigger(self):
        """测试审计规则是否应该触发"""
        rule = AuditRule(
            rule_id="test_rule",
            name="Test Rule",
            description="Test rule",
            conditions={},
            actions=["alert"],
            enabled=True,
            cooldown_period=60  # 60秒冷却期
        )

        event = AuditEvent(
            event_id="event1",
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.HIGH,
            timestamp=datetime.now(),
            user_id="user1",
            action="login",
            resource="system",
            result="success"
        )

        # 第一次应该触发
        assert rule.should_trigger(event)

        # 立即再次触发应该被冷却期阻止
        assert not rule.should_trigger(event)

        # 手动重置最后触发时间以测试
        rule.last_triggered = datetime.now() - timedelta(seconds=70)
        assert rule.should_trigger(event)

    def test_audit_rule_trigger(self):
        """测试审计规则触发"""
        rule = AuditRule(
            rule_id="test_rule",
            name="Test Rule",
            description="Test rule",
            conditions={},
            actions=["alert", "log", "notify"],
            enabled=True
        )

        event = AuditEvent(
            event_id="event1",
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.HIGH,
            timestamp=datetime.now(),
            user_id="user1",
            action="login",
            resource="system",
            result="success"
        )

        actions = rule.trigger(event)
        assert actions == ["alert", "log", "notify"]
        assert rule.last_triggered is not None


class TestComplianceReport:
    """测试ComplianceReport类"""

    def test_compliance_report_creation(self):
        """测试合规报告创建"""
        findings = [
            {
                "type": "security",
                "severity": "high",
                "description": "Multiple failed login attempts",
                "recommendation": "Implement account lockout policy"
            }
        ]

        metrics = {
            "total_events": 1000,
            "security_events": 50,
            "compliance_score": 85.5
        }

        report = ComplianceReport(
            report_id="compliance_001",
            report_type="security",
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            overall_status="warning",
            findings=findings,
            metrics=metrics,
            generated_at=datetime.now()
        )

        assert report.report_id == "compliance_001"
        assert report.report_type == "security"
        assert report.overall_status == "warning"
        assert report.findings == findings
        assert report.metrics == metrics


class TestAuditLoggingManager:
    """测试AuditLoggingManager类"""

    def test_initialization(self, audit_manager):
        """测试审计日志管理器初始化"""
        assert audit_manager.log_path is not None
        assert hasattr(audit_manager, 'audit_rules')
        assert hasattr(audit_manager, 'event_queue')
        assert hasattr(audit_manager, 'event_stats')
        assert audit_manager.monitoring_thread is None  # 因为我们禁用了实时监控

    def test_log_event(self, audit_manager):
        """测试记录审计事件"""
        event_id = audit_manager.log_event(
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.MEDIUM,
            user_id="test_user",
            action="login",
            resource="system",
            result="success",
            details={"ip": "192.168.1.1"},
            source_ip="192.168.1.1",
            session_id="session_123"
        )

        assert event_id is not None
        assert len(event_id) > 0

        # 验证事件被记录
        assert len(audit_manager.event_queue) > 0

    def test_log_security_event(self, audit_manager):
        """测试记录安全事件"""
        event_id = audit_manager.log_security_event(
            user_id="test_user",
            action="password_change",
            result="success",
            details={"old_password_hash": "hash123"}
        )

        assert event_id is not None

        # 验证事件队列中有事件
        assert len(audit_manager.event_queue) > 0

    def test_log_access_event(self, audit_manager):
        """测试记录访问事件"""
        event_id = audit_manager.log_access_event(
            user_id="test_user",
            resource="file.txt",
            action="read",
            result="success",
            details={"file_size": 1024}
        )

        assert event_id is not None
        assert len(audit_manager.event_queue) > 0

    def test_log_data_operation(self, audit_manager):
        """测试记录数据操作事件"""
        event_id = audit_manager.log_data_operation(
            user_id="test_user",
            operation="export",
            resource="customer_data",
            result="success",
            record_count=1000,
            details={"format": "csv"}
        )

        assert event_id is not None
        assert len(audit_manager.event_queue) > 0

    def test_query_events(self, audit_manager):
        """测试查询审计事件"""
        # 先记录一些事件
        audit_manager.log_event(
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.HIGH,
            user_id="user1",
            action="login",
            resource="system",
            result="success"
        )

        audit_manager.log_event(
            event_type=AuditEventType.ACCESS,
            severity=AuditSeverity.MEDIUM,
            user_id="user2",
            action="read",
            resource="file.txt",
            result="success"
        )

        # 强制处理事件队列
        audit_manager._process_event_queue()

        # 查询事件
        events = audit_manager.query_events()
        assert len(events) >= 2

        # 按用户查询
        user1_events = audit_manager.query_events(user_id="user1")
        assert len(user1_events) >= 1
        assert all(event.user_id == "user1" for event in user1_events)

        # 按事件类型查询
        security_events = audit_manager.query_events(event_type=AuditEventType.SECURITY)
        assert len(security_events) >= 1
        assert all(event.event_type == AuditEventType.SECURITY for event in security_events)

    def test_get_security_report(self, audit_manager):
        """测试获取安全报告"""
        # 记录一些安全事件
        audit_manager.log_security_event("user1", "login", "success")
        audit_manager.log_security_event("user2", "login", "failed")
        audit_manager.log_security_event("user1", "password_change", "success")

        # 处理事件队列
        audit_manager._process_event_queue()

        # 获取安全报告
        report = audit_manager.get_security_report(days=1)

        assert "report_period" in report
        assert "total_events" in report
        assert "security_events" in report
        assert "risk_assessment" in report
        assert "recommendations" in report

    def test_get_compliance_report(self, audit_manager):
        """测试获取合规报告"""
        # 记录一些事件
        audit_manager.log_event(
            event_type=AuditEventType.COMPLIANCE,
            severity=AuditSeverity.MEDIUM,
            user_id="admin",
            action="policy_update",
            resource="security_policy",
            result="success"
        )

        # 处理事件队列
        audit_manager._process_event_queue()

        # 获取合规报告
        compliance_report = audit_manager.get_compliance_report()

        assert isinstance(compliance_report, ComplianceReport)
        assert compliance_report.report_id is not None
        assert compliance_report.report_type == "general"
        assert compliance_report.findings is not None
        assert compliance_report.metrics is not None

    def test_audit_rules_processing(self, audit_manager):
        """测试审计规则处理"""
        # 创建一个测试规则
        test_rule = AuditRule(
            rule_id="test_alert_rule",
            name="Test Alert Rule",
            description="Rule for testing alerts",
            conditions={
                "event_type": AuditEventType.SECURITY,
                "severity": AuditSeverity.CRITICAL
            },
            actions=["alert"],
            enabled=True
        )

        audit_manager.rules.append(test_rule)

        # 记录匹配的事件
        audit_manager.log_event(
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.CRITICAL,
            user_id="test_user",
            action="unauthorized_access",
            resource="sensitive_data",
            result="blocked"
        )

        # 处理事件队列和规则检查
        audit_manager._process_event_queue()
        audit_manager._check_audit_rules()

        # 验证规则被触发
        assert test_rule.last_triggered is not None

    @patch('src.infrastructure.security.audit.audit_logging_manager.logging.warning')
    def test_alert_sending(self, mock_warning, audit_manager):
        """测试警报发送"""
        rule = AuditRule(
            rule_id="alert_test",
            name="Alert Test",
            description="Test alert sending",
            event_pattern={},
            severity_threshold=AuditSeverity.CRITICAL,
            actions=["alert"],
            enabled=True
        )

        event = AuditEvent(
            event_id="alert_event",
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.CRITICAL,
            timestamp=datetime.now(),
            session_id="test_session",
            user_id="test_user",
            action="breach",
            resource="system",
            result="detected"
        )

        audit_manager._send_alert(rule, event)

        # 验证日志调用
        mock_warning.assert_called()

    @patch('src.infrastructure.security.audit.audit_logging_manager.logging.info')
    def test_notification_sending(self, mock_info, audit_manager):
        """测试通知发送"""
        rule = AuditRule(
            rule_id="notify_test",
            name="Notify Test",
            description="Test notification sending",
            event_pattern={},
            severity_threshold=AuditSeverity.HIGH,
            actions=["notify"],
            enabled=True
        )

        event = AuditEvent(
            event_id="notify_event",
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.HIGH,
            timestamp=datetime.now(),
            session_id="admin_session",
            user_id="admin",
            action="config_change",
            resource="firewall",
            result="success"
        )

        audit_manager._send_notification(rule, event)

        # 验证日志调用
        mock_info.assert_called()

    def test_statistics_update(self, audit_manager):
        """测试统计信息更新"""
        initial_stats = audit_manager.statistics.copy()

        # 记录事件
        audit_manager.log_event(
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.HIGH,
            user_id="test_user",
            action="login",
            resource="system",
            result="success"
        )

        audit_manager._process_event_queue()

        # 验证统计信息已更新
        assert audit_manager.statistics["total_events"] >= initial_stats.get("total_events", 0) + 1
        assert audit_manager.statistics["security_events"] >= initial_stats.get("security_events", 0) + 1

    def test_log_rotation(self, audit_manager):
        """测试日志轮转"""
        # 记录多个事件以触发轮转检查
        for i in range(150):  # 超过默认的100个事件限制
            audit_manager.log_event(
                event_type=AuditEventType.SYSTEM_EVENT,
                severity=AuditSeverity.LOW,
                user_id=f"user{i}",
                action="heartbeat",
                resource="system",
                result="success"
            )

        # 处理事件队列
        audit_manager._process_event_queue()

        # 检查日志轮转（这会检查是否创建了新的日志文件）
        audit_manager._check_log_rotation()

        # 验证事件总数在合理范围内（可能有轮转）
        total_events = len(list(audit_manager.query_events()))
        assert total_events <= 200  # 应该有一定限制

    def test_concurrent_logging(self, audit_manager):
        """测试并发日志记录"""
        import threading
        import time

        results = []
        errors = []

        def log_worker(worker_id):
            try:
                for i in range(10):
                    event_id = audit_manager.log_event(
                        event_type=AuditEventType.ACCESS,
                        severity=AuditSeverity.LOW,
                        user_id=f"worker_{worker_id}",
                        action=f"action_{i}",
                        resource=f"resource_{i}",
                        result="success"
                    )
                    results.append(event_id)
                    time.sleep(0.001)  # 小延迟以增加并发性
            except Exception as e:
                errors.append(str(e))

        # 启动多个线程
        threads = []
        num_threads = 5
        for i in range(num_threads):
            t = threading.Thread(target=log_worker, args=(i,))
            threads.append(t)
            t.start()

        # 等待所有线程完成
        for t in threads:
            t.join()

        # 验证结果
        assert len(results) == num_threads * 10  # 每个线程10个事件
        assert len(errors) == 0  # 不应该有错误

        # 处理事件队列
        audit_manager._process_event_queue()

        # 验证所有事件都被记录
        events = audit_manager.query_events()
        assert len(events) >= num_threads * 10

    def test_error_handling(self, audit_manager):
        """测试错误处理"""
        # 测试无效参数
        with pytest.raises(Exception):
            audit_manager.log_event(
                event_type="invalid_type",  # 应该是枚举
                severity=AuditSeverity.HIGH,
                user_id="test",
                action="test",
                resource="test",
                result="success"
            )

    def test_large_scale_events(self, audit_manager):
        """测试大规模事件处理"""
        # 记录大量事件
        num_events = 1000
        for i in range(num_events):
            audit_manager.log_event(
                event_type=AuditEventType.SYSTEM_EVENT,
                severity=AuditSeverity.LOW,
                user_id=f"user_{i % 10}",  # 10个不同用户
                action="routine_check",
                resource="system",
                result="success"
            )

        # 处理事件队列
        audit_manager._process_event_queue()

        # 验证事件数量
        events = audit_manager.query_events()
        assert len(events) >= num_events

        # 测试查询性能（应该在合理时间内完成）
        import time
        start_time = time.time()
        filtered_events = audit_manager.query_events(
            event_type=AuditEventType.SYSTEM_EVENT,
            user_id="user_1"
        )
        query_time = time.time() - start_time

        # 查询应该在1秒内完成
        assert query_time < 1.0
        assert len(filtered_events) >= 100  # user_1应该有很多事件

    def test_rule_engine_complex_conditions(self, audit_manager):
        """测试复杂规则条件"""
        # 创建复杂规则
        complex_rule = AuditRule(
            rule_id="complex_rule",
            name="Complex Security Rule",
            description="Rule with multiple conditions",
            conditions={
                "event_type": AuditEventType.SECURITY,
                "severity": AuditSeverity.HIGH,
                "user_pattern": "admin_*",
                "resource_pattern": "system/*",
                "result": "failed"
            },
            actions=["alert", "block"],
            enabled=True
        )

        audit_manager.rules.append(complex_rule)

        # 测试匹配的事件
        matching_event = AuditEvent(
            event_id="match_1",
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.HIGH,
            timestamp=datetime.now(),
            user_id="admin_user",
            action="access",
            resource="system/config",
            result="failed"
        )

        # 测试不匹配的事件
        non_matching_events = [
            AuditEvent("no_match_1", AuditEventType.ACCESS, AuditSeverity.HIGH,
                      datetime.now(), "admin_user", "read", "file.txt", "failed"),
            AuditEvent("no_match_2", AuditEventType.SECURITY, AuditSeverity.LOW,
                      datetime.now(), "admin_user", "access", "system/config", "failed"),
            AuditEvent("no_match_3", AuditEventType.SECURITY, AuditSeverity.HIGH,
                      datetime.now(), "regular_user", "access", "system/config", "failed"),
            AuditEvent("no_match_4", AuditEventType.SECURITY, AuditSeverity.HIGH,
                      datetime.now(), "admin_user", "access", "user/profile", "failed"),
            AuditEvent("no_match_5", AuditEventType.SECURITY, AuditSeverity.HIGH,
                      datetime.now(), "admin_user", "access", "system/config", "success")
        ]

        assert complex_rule.matches_event(matching_event)
        for event in non_matching_events:
            assert not complex_rule.matches_event(event)

    def test_compliance_reporting_detailed(self, audit_manager):
        """测试详细合规报告"""
        # 创建各种合规相关事件
        compliance_events = [
            (AuditEventType.COMPLIANCE, "policy_update", "success"),
            (AuditEventType.SECURITY, "password_change", "success"),
            (AuditEventType.ACCESS, "privileged_access", "granted"),
            (AuditEventType.USER_MANAGEMENT, "role_assignment", "success"),
            (AuditEventType.DATA_OPERATION, "data_export", "success"),
        ]

        for event_type, action, result in compliance_events:
            audit_manager.log_event(
                event_type=event_type,
                severity=AuditSeverity.MEDIUM,
                user_id="compliance_user",
                action=action,
                resource="system",
                result=result
            )

        # 处理事件
        audit_manager._process_event_queue()

        # 生成详细合规报告
        report = audit_manager.get_compliance_report("detailed", days=1)

        assert report.report_type == "detailed"
        assert len(report.findings) > 0
        assert "total_events" in report.metrics
        assert "compliance_score" in report.metrics

    def test_audit_data_integrity(self, audit_manager):
        """测试审计数据完整性"""
        # 记录事件
        original_event_id = audit_manager.log_event(
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.HIGH,
            user_id="integrity_test",
            action="test_action",
            resource="test_resource",
            result="success",
            details={"test": "data"}
        )

        # 处理队列
        audit_manager._process_event_queue()

        # 查询事件
        events = audit_manager.query_events(event_id=original_event_id)

        assert len(events) == 1
        event = events[0]

        # 验证数据完整性
        assert event.event_id == original_event_id
        assert event.event_type == AuditEventType.SECURITY
        assert event.severity == AuditSeverity.HIGH
        assert event.user_id == "integrity_test"
        assert event.action == "test_action"
        assert event.resource == "test_resource"
        assert event.result == "success"
        assert event.details == {"test": "data"}

    def test_performance_under_load(self, audit_manager):
        """测试负载下的性能"""
        import time

        # 测试批量记录性能
        num_events = 500
        start_time = time.time()

        for i in range(num_events):
            audit_manager.log_event(
                event_type=AuditEventType.SYSTEM_EVENT,
                severity=AuditSeverity.LOW,
                user_id=f"perf_user_{i % 5}",
                action="performance_test",
                resource="system",
                result="success"
            )

        logging_time = time.time() - start_time

        # 处理队列
        process_start = time.time()
        audit_manager._process_event_queue()
        process_time = time.time() - process_start

        # 查询性能
        query_start = time.time()
        events = audit_manager.query_events()
        query_time = time.time() - query_start

        # 验证性能指标
        assert logging_time < 5.0  # 记录500个事件应该在5秒内完成
        assert process_time < 2.0  # 处理队列应该在2秒内完成
        assert query_time < 1.0    # 查询应该在1秒内完成
        assert len(events) >= num_events
