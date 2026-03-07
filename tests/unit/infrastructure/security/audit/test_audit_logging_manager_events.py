#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AuditLoggingManager 事件记录补充测试
"""

from datetime import datetime
from pathlib import Path

import pytest

from src.infrastructure.security.audit.audit_logging_manager import (
    AuditLoggingManager,
    AuditEventType,
    AuditSeverity,
)


@pytest.fixture()
def audit_manager(tmp_path: Path) -> AuditLoggingManager:
    return AuditLoggingManager(log_path=str(tmp_path), enable_realtime_monitoring=False)


def test_log_event_populates_details_and_statistics(audit_manager: AuditLoggingManager) -> None:
    event_id = audit_manager.log_event(
        event_type="security",  # 验证字符串枚举转换
        severity="critical",
        user_id="user-1",
        action="override",
        result="failure",
        resource="db://prod/accounts",
        ip_address="10.0.0.5",
        details={"reason": "manual override"},
        metadata={"source": "api-gateway"},
        source_ip="10.0.0.5",
        correlation_id="abc-123",
    )

    event = audit_manager.event_queue[-1]
    assert event.event_id == event_id
    assert event.event_type == AuditEventType.SECURITY
    assert event.severity == AuditSeverity.CRITICAL
    assert event.details["reason"] == "manual override"
    assert event.details["metadata"] == {"source": "api-gateway"}
    assert event.details["extra"]["correlation_id"] == "abc-123"
    assert audit_manager.statistics["total_events"] == 1
    assert audit_manager.statistics["security_events"] == 1


def test_log_access_event_severity_and_queue(audit_manager: AuditLoggingManager) -> None:
    event_id = audit_manager.log_access_event(
        user_id="user-2",
        resource="/api/private",
        action="GET",
        result="denied",
        session_id="sess-001",
        risk_score=0.75,
    )

    event = audit_manager.event_queue[-1]
    assert event.event_id == event_id
    assert event.event_type == AuditEventType.ACCESS
    # denied 且 risk_score > 0.5 -> HIGH
    assert event.severity == AuditSeverity.HIGH
    assert event.session_id == "sess-001"
    assert audit_manager.statistics["access_events"] == 1


def test_log_data_operation_records_counts(audit_manager: AuditLoggingManager) -> None:
    event_id = audit_manager.log_data_operation(
        user_id="user-3",
        operation="export",
        resource="s3://bucket/sensitive/report.csv",
        result="success",
        record_count=42,
        details={"ft": datetime(2025, 1, 1)},
    )

    event = audit_manager.event_queue[-1]
    assert event.event_id == event_id
    assert event.event_type == AuditEventType.DATA_OPERATION
    assert event.details["record_count"] == 42
    assert isinstance(event.details["ft"], datetime)
    assert event.risk_score == pytest.approx(0.3)
    assert event.severity == AuditSeverity.MEDIUM
    assert audit_manager.statistics["data_events"] == 1

