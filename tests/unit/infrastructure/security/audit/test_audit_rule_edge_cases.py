#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
审计规则补充测试

验证 AuditRule / AuditEvent 中兼容字段、正则匹配与冷却逻辑等低覆盖分支。
"""

from datetime import datetime, timedelta

from src.infrastructure.security.audit.audit_logging_manager import (
    AuditEvent,
    AuditEventType,
    AuditRule,
    AuditSeverity,
)


def _make_event(**overrides):
    defaults = dict(
        event_id="evt-001",
        event_type=AuditEventType.SECURITY,
        severity=AuditSeverity.HIGH,
        timestamp=datetime(2025, 1, 1, 12, 0, 0),
        user_id="user-123",
        action="login",
        resource="db://prod/users",
        result="failed",
        risk_score=9.5,
    )
    defaults.update(overrides)
    return AuditEvent(**defaults)


def test_audit_rule_legacy_fields_and_defaults():
    """event_pattern 与 cooldown_minutes 应兼容旧配置并补齐默认动作。"""
    rule = AuditRule(
        rule_id="legacy",
        name="Legacy Rule",
        event_pattern={"event_type": "security", "result": "failed"},
        actions=[],
        cooldown_minutes=1,
    )

    assert rule.conditions == {"event_type": "security", "result": "failed"}
    assert rule.cooldown_period == 60  # 1 分钟转换为秒
    assert rule.actions == ["log"]  # 未指定动作时自动补全

    event = _make_event()
    assert rule.matches_event(event) is True


def test_audit_rule_matches_event_regex_and_thresholds():
    """resource/user 正则、最小风险阈值均满足时应匹配成功。"""
    rule = AuditRule(
        rule_id="regex",
        name="Regex Rule",
        conditions={
            "resource_pattern": r"db://.+/users",
            "user_pattern": r"user-\d+",
            "min_risk_score": 5,
            "result": "failed",
        },
    )

    assert rule.matches_event(_make_event()) is True
    assert rule.matches_event(_make_event(user_id="guest")) is False


def test_audit_rule_should_trigger_respects_cooldown():
    """should_trigger 应考虑冷却时间，过期后重新触发。"""
    rule = AuditRule(
        rule_id="cooldown",
        name="Cooldown Rule",
        conditions={"event_type": AuditEventType.SECURITY},
        cooldown_period=2,
    )
    event = _make_event()

    assert rule.should_trigger(event) is True
    # 冷却期内重复调用应拒绝
    assert rule.should_trigger(event) is False

    # 人为回拨 last_triggered 到 3 秒前，验证冷却到期后可再次触发
    rule.last_triggered = datetime.now() - timedelta(seconds=3)
    assert rule.should_trigger(event) is True


def test_audit_event_from_dict_roundtrip():
    """from_dict 应将列表标签恢复为集合。"""
    raw = {
        "event_id": "evt-raw",
        "event_type": "security",
        "severity": "critical",
        "timestamp": "2025-01-01T00:00:00",
        "tags": ["urgent", "security"],
    }

    event = AuditEvent.from_dict(raw)
    assert isinstance(event.tags, set)
    assert event.tags == {"urgent", "security"}
    assert sorted(event.to_dict()["tags"]) == ["security", "urgent"]

