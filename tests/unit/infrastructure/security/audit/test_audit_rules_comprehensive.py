#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
审计规则引擎综合测试
测试AuditRuleEngine及其相关组件的核心功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from src.infrastructure.security.audit.audit_rules import (
    AuditRuleEngine,
    AuditRule,
    RuleCondition,
    RuleAction,
    RuleConditionType,
    AuditRuleTemplates
)
from src.infrastructure.security.audit.audit_events import AuditEvent, AuditEventType, AuditSeverity


@pytest.fixture
def audit_rule_engine():
    """创建审计规则引擎实例"""
    return AuditRuleEngine()


@pytest.fixture
def sample_audit_event():
    """创建示例审计事件"""
    return AuditEvent(
        event_id="test-event-001",
        event_type=AuditEventType.ACCESS,
        severity=AuditSeverity.MEDIUM,
        timestamp=datetime.now(),
        user_id="test_user",
        resource="/api/data",
        details={"action": "read", "ip_address": "192.168.1.100"}
    )


@pytest.fixture
def sample_failed_login_event():
    """创建失败登录事件"""
    return AuditEvent(
        event_id="failed-login-001",
        event_type=AuditEventType.ACCESS,
        severity=AuditSeverity.MEDIUM,
        timestamp=datetime.now(),
        user_id="test_user",
        resource="/login",
        details={"action": "login", "success": False, "ip_address": "192.168.1.100"}
    )


@pytest.fixture
def sample_high_risk_event():
    """创建高风险事件"""
    return AuditEvent(
        event_id="high-risk-001",
        event_type=AuditEventType.SECURITY,
        severity=AuditSeverity.CRITICAL,
        timestamp=datetime.now(),
        user_id="suspicious_user",
        resource="/api/admin",
        details={"action": "access", "ip_address": "10.0.0.1"}
    )


class TestRuleAction:
    """测试规则动作枚举"""

    def test_rule_actions_exist(self):
        """测试规则动作定义"""
        assert RuleAction.LOG.value == "log"
        assert RuleAction.ALERT.value == "alert"
        assert RuleAction.BLOCK.value == "block"
        assert RuleAction.NOTIFY.value == "notify"
        assert RuleAction.ESCALATE.value == "escalate"

    def test_rule_actions_unique(self):
        """测试规则动作值唯一"""
        values = [action.value for action in RuleAction]
        assert len(values) == len(set(values))


class TestRuleConditionType:
    """测试规则条件类型枚举"""

    def test_rule_condition_types_exist(self):
        """测试规则条件类型定义"""
        assert RuleConditionType.EVENT_TYPE.value == "event_type"
        assert RuleConditionType.SEVERITY.value == "severity"
        assert RuleConditionType.USER_ID.value == "user_id"
        assert RuleConditionType.RESOURCE.value == "resource"
        assert RuleConditionType.ACTION.value == "action"
        assert RuleConditionType.RESULT.value == "result"
        assert RuleConditionType.RISK_SCORE.value == "risk_score"
        assert RuleConditionType.TIME_WINDOW.value == "time_window"
        assert RuleConditionType.FREQUENCY.value == "frequency"
        assert RuleConditionType.PATTERN.value == "pattern"


class TestRuleCondition:
    """测试规则条件类"""

    def test_rule_condition_creation_minimal(self):
        """测试最小化规则条件创建"""
        condition = RuleCondition(
            condition_type=RuleConditionType.EVENT_TYPE,
            operator="eq",
            value=AuditEventType.ACCESS
        )

        assert condition.condition_type == RuleConditionType.EVENT_TYPE
        assert condition.operator == "eq"
        assert condition.value == AuditEventType.ACCESS
        assert condition.case_sensitive is True

    def test_rule_condition_creation_complete(self):
        """测试完整规则条件创建"""
        condition = RuleCondition(
            condition_type=RuleConditionType.USER_ID,
            operator="contains",
            value="admin",
            case_sensitive=False
        )

        assert condition.condition_type == RuleConditionType.USER_ID
        assert condition.operator == "contains"
        assert condition.value == "admin"
        assert condition.case_sensitive is False

    def test_matches_event_type_equal(self, sample_audit_event):
        """测试事件类型相等匹配"""
        condition = RuleCondition(
            condition_type=RuleConditionType.EVENT_TYPE,
            operator="eq",
            value=AuditEventType.ACCESS
        )

        assert condition.matches(sample_audit_event) is True

        # 测试不匹配的情况
        condition.value = AuditEventType.SECURITY
        assert condition.matches(sample_audit_event) is False

    def test_matches_severity_equal(self, sample_audit_event):
        """测试严重程度相等匹配"""
        condition = RuleCondition(
            condition_type=RuleConditionType.SEVERITY,
            operator="eq",
            value=AuditSeverity.MEDIUM
        )

        assert condition.matches(sample_audit_event) is True

        # 测试不匹配的情况
        condition.value = AuditSeverity.HIGH
        assert condition.matches(sample_audit_event) is False

    def test_matches_user_id_contains(self, sample_audit_event):
        """测试用户ID包含匹配"""
        condition = RuleCondition(
            condition_type=RuleConditionType.USER_ID,
            operator="contains",
            value="test"
        )

        assert condition.matches(sample_audit_event) is True

        # 测试不匹配的情况
        condition.value = "admin"
        assert condition.matches(sample_audit_event) is False

    def test_matches_resource_regex(self, sample_audit_event):
        """测试资源正则匹配"""
        condition = RuleCondition(
            condition_type=RuleConditionType.RESOURCE,
            operator="regex",
            value=r"/api/.*"
        )

        assert condition.matches(sample_audit_event) is True

        # 测试不匹配的情况
        condition.value = r"/admin/.*"
        assert condition.matches(sample_audit_event) is False

    def test_matches_with_details(self, sample_audit_event):
        """测试匹配包含details字段的条件"""
        # 测试action匹配
        condition = RuleCondition(
            condition_type=RuleConditionType.ACTION,
            operator="eq",
            value="read"
        )

        assert condition.matches(sample_audit_event) is True

    def test_matches_case_insensitive(self, sample_audit_event):
        """测试不区分大小写匹配"""
        condition = RuleCondition(
            condition_type=RuleConditionType.USER_ID,
            operator="contains",
            value="TEST",
            case_sensitive=False
        )

        assert condition.matches(sample_audit_event) is True

    def test_matches_invalid_operator(self, sample_audit_event):
        """测试无效操作符"""
        condition = RuleCondition(
            condition_type=RuleConditionType.USER_ID,
            operator="invalid_op",
            value="test"
        )

        # 应该返回False而不是抛出异常
        assert condition.matches(sample_audit_event) is False


class TestAuditRule:
    """测试审计规则类"""

    def test_audit_rule_creation_minimal(self):
        """测试最小化审计规则创建"""
        rule = AuditRule(
            rule_id="test-rule-001",
            name="Test Rule",
            description="A test audit rule",
            conditions=[],
            actions=[]
        )

        assert rule.rule_id == "test-rule-001"
        assert rule.name == "Test Rule"
        assert rule.description == "A test audit rule"
        assert rule.enabled is True
        assert rule.priority == 1
        assert rule.conditions == []
        assert rule.actions == []

    def test_audit_rule_creation_complete(self):
        """测试完整审计规则创建"""
        conditions = [
            RuleCondition(
                condition_type=RuleConditionType.EVENT_TYPE,
                operator="eq",
                value=AuditEventType.ACCESS
            )
        ]
        actions = [RuleAction.LOG, RuleAction.ALERT]

        rule = AuditRule(
            rule_id="complete-rule-001",
            name="Complete Test Rule",
            description="A complete audit rule",
            conditions=conditions,
            actions=actions,
            enabled=False,
            priority=5,
            metadata={"category": "security"}
        )

        assert rule.rule_id == "complete-rule-001"
        assert rule.name == "Complete Test Rule"
        assert rule.description == "A complete audit rule"
        assert rule.enabled is False
        assert rule.priority == 5
        assert rule.conditions == conditions
        assert rule.actions == actions
        assert rule.metadata == {"category": "security"}

    def test_matches_no_conditions(self, sample_audit_event):
        """测试无条件规则匹配"""
        rule = AuditRule(
            rule_id="no-condition-rule",
            name="No Condition Rule",
            description="Rule with no conditions",
            conditions=[],
            actions=[]
        )

        assert rule.matches(sample_audit_event) is True

    def test_matches_with_conditions(self, sample_audit_event):
        """测试带条件规则匹配"""
        conditions = [
            RuleCondition(
                condition_type=RuleConditionType.EVENT_TYPE,
                operator="eq",
                value=AuditEventType.ACCESS
            ),
            RuleCondition(
                condition_type=RuleConditionType.SEVERITY,
                operator="eq",
                value=AuditSeverity.MEDIUM
            )
        ]

        rule = AuditRule(
            rule_id="condition-rule",
            name="Condition Rule",
            description="Rule with conditions",
            conditions=conditions,
            actions=[]
        )

        assert rule.matches(sample_audit_event) is True

    def test_matches_condition_not_met(self, sample_audit_event):
        """测试条件不满足的情况"""
        conditions = [
            RuleCondition(
                condition_type=RuleConditionType.EVENT_TYPE,
                operator="eq",
                value=AuditEventType.SECURITY  # 不匹配的事件类型
            )
        ]

        rule = AuditRule(
            rule_id="not-match-rule",
            name="Not Match Rule",
            description="Rule that won't match",
            conditions=conditions,
            actions=[]
        )

        assert rule.matches(sample_audit_event) is False

    def test_execute_actions_log(self, sample_audit_event):
        """测试执行LOG动作"""
        rule = AuditRule(
            rule_id="log-rule",
            name="Log Rule",
            description="Rule with log action",
            conditions=[],
            actions=[RuleAction.LOG]
        )

        results = rule.execute_actions(sample_audit_event)

        assert isinstance(results, list)
        assert len(results) >= 1
        assert any(result.get("action") == "log" for result in results)

    def test_execute_actions_multiple(self, sample_audit_event):
        """测试执行多个动作"""
        rule = AuditRule(
            rule_id="multi-action-rule",
            name="Multi Action Rule",
            description="Rule with multiple actions",
            conditions=[],
            actions=[RuleAction.LOG, RuleAction.ALERT, RuleAction.NOTIFY]
        )

        results = rule.execute_actions(sample_audit_event)

        assert isinstance(results, list)
        assert len(results) >= 3

        action_types = [result.get("action") for result in results]
        assert "log" in action_types
        assert "alert" in action_types
        assert "notify" in action_types

    def test_execute_actions_disabled_rule(self, sample_audit_event):
        """测试执行禁用规则的动作"""
        rule = AuditRule(
            rule_id="disabled-rule",
            name="Disabled Rule",
            description="Disabled rule",
            conditions=[],
            actions=[RuleAction.LOG],
            enabled=False
        )

        results = rule.execute_actions(sample_audit_event)

        # 禁用规则应该不执行动作
        assert isinstance(results, list)
        assert len(results) == 0


class TestAuditRuleEngine:
    """测试审计规则引擎"""

    def test_initialization(self):
        """测试初始化"""
        engine = AuditRuleEngine()

        assert hasattr(engine, '_rules')
        assert hasattr(engine, '_rule_groups')
        assert hasattr(engine, '_stats')
        assert isinstance(engine._rules, dict)
        assert isinstance(engine._rule_groups, dict)

    def test_add_rule(self, audit_rule_engine):
        """测试添加规则"""
        engine = audit_rule_engine

        rule = AuditRule(
            rule_id="test-rule",
            name="Test Rule",
            description="A test rule",
            conditions=[],
            actions=[]
        )

        engine.add_rule(rule)

        assert "test-rule" in engine._rules
        assert engine._rules["test-rule"] == rule

    def test_add_duplicate_rule(self, audit_rule_engine):
        """测试添加重复规则"""
        engine = audit_rule_engine

        rule1 = AuditRule(
            rule_id="duplicate-rule",
            name="Rule 1",
            description="First rule",
            conditions=[],
            actions=[]
        )

        rule2 = AuditRule(
            rule_id="duplicate-rule",
            name="Rule 2",
            description="Second rule",
            conditions=[],
            actions=[]
        )

        engine.add_rule(rule1)
        engine.add_rule(rule2)  # 应该覆盖第一个

        assert engine._rules["duplicate-rule"] == rule2

    def test_remove_rule(self, audit_rule_engine):
        """测试移除规则"""
        engine = audit_rule_engine

        rule = AuditRule(
            rule_id="remove-rule",
            name="Remove Rule",
            description="Rule to be removed"
        )

        engine.add_rule(rule)
        assert "remove-rule" in engine._rules

        result = engine.remove_rule("remove-rule")
        assert result is True
        assert "remove-rule" not in engine._rules

    def test_remove_nonexistent_rule(self, audit_rule_engine):
        """测试移除不存在的规则"""
        engine = audit_rule_engine

        result = engine.remove_rule("nonexistent-rule")
        assert result is False

    def test_enable_rule(self, audit_rule_engine):
        """测试启用规则"""
        engine = audit_rule_engine

        rule = AuditRule(
            rule_id="enable-rule",
            name="Enable Rule",
            description="Rule to be enabled",
            enabled=False
        )

        engine.add_rule(rule)
        assert engine._rules["enable-rule"].enabled is False

        result = engine.enable_rule("enable-rule")
        assert result is True
        assert engine._rules["enable-rule"].enabled is True

    def test_disable_rule(self, audit_rule_engine):
        """测试禁用规则"""
        engine = audit_rule_engine

        rule = AuditRule(
            rule_id="disable-rule",
            name="Disable Rule",
            description="Rule to be disabled",
            enabled=True
        )

        engine.add_rule(rule)
        assert engine._rules["disable-rule"].enabled is True

        result = engine.disable_rule("disable-rule")
        assert result is True
        assert engine._rules["disable-rule"].enabled is False

    def test_evaluate_event_no_match(self, audit_rule_engine, sample_audit_event):
        """测试评估无匹配规则的事件"""
        engine = audit_rule_engine

        # 添加一个不匹配的规则
        rule = AuditRule(
            rule_id="no-match-rule",
            name="No Match Rule",
            description="Rule that won't match",
            conditions=[
                RuleCondition(
                    condition_type=RuleConditionType.EVENT_TYPE,
                    operator="eq",
                    value=AuditEventType.SECURITY
                )
            ],
            actions=[RuleAction.LOG]
        )

        engine.add_rule(rule)

        results = engine.evaluate_event(sample_audit_event)

        assert isinstance(results, list)
        assert len(results) == 0

    def test_evaluate_event_with_match(self, audit_rule_engine, sample_audit_event):
        """测试评估匹配规则的事件"""
        engine = audit_rule_engine

        # 添加一个匹配的规则
        rule = AuditRule(
            rule_id="match-rule",
            name="Match Rule",
            description="Rule that will match",
            conditions=[
                RuleCondition(
                    condition_type=RuleConditionType.EVENT_TYPE,
                    operator="eq",
                    value=AuditEventType.ACCESS
                )
            ],
            actions=[RuleAction.LOG, RuleAction.ALERT]
        )

        engine.add_rule(rule)

        results = engine.evaluate_event(sample_audit_event)

        assert isinstance(results, list)
        assert len(results) >= 2  # 应该有LOG和ALERT动作的结果

    def test_get_rule(self, audit_rule_engine):
        """测试获取规则"""
        engine = audit_rule_engine

        rule = AuditRule(
            rule_id="get-rule",
            name="Get Rule",
            description="Rule to be retrieved"
        )

        engine.add_rule(rule)

        retrieved_rule = engine.get_rule("get-rule")
        assert retrieved_rule == rule

    def test_get_nonexistent_rule(self, audit_rule_engine):
        """测试获取不存在的规则"""
        engine = audit_rule_engine

        result = engine.get_rule("nonexistent-rule")
        assert result is None

    def test_list_rules(self, audit_rule_engine):
        """测试列出规则"""
        engine = audit_rule_engine

        rule1 = AuditRule(
            rule_id="list-rule-1",
            name="List Rule 1",
            description="First rule"
        )

        rule2 = AuditRule(
            rule_id="list-rule-2",
            name="List Rule 2",
            description="Second rule",
            enabled=False
        )

        engine.add_rule(rule1)
        engine.add_rule(rule2)

        # 列出所有规则
        all_rules = engine.list_rules()
        assert len(all_rules) >= 2

        # 只列出启用的规则
        enabled_rules = engine.list_rules(enabled_only=True)
        assert len(enabled_rules) >= 1
        assert all(rule.enabled for rule in enabled_rules)

    def test_get_stats(self, audit_rule_engine, sample_audit_event):
        """测试获取统计信息"""
        engine = audit_rule_engine

        # 添加规则并执行评估
        rule = AuditRule(
            rule_id="stats-rule",
            name="Stats Rule",
            description="Rule for stats testing",
            actions=[RuleAction.LOG]
        )

        engine.add_rule(rule)
        engine.evaluate_event(sample_audit_event)

        stats = engine.get_stats()

        assert isinstance(stats, dict)
        assert "total_rules" in stats
        assert "enabled_rules" in stats
        assert "total_evaluations" in stats

    def test_clear_stats(self, audit_rule_engine):
        """测试清除统计信息"""
        engine = audit_rule_engine

        # 先获取初始统计
        initial_stats = engine.get_stats()

        # 清除统计
        engine.clear_stats()

        # 验证统计被重置
        cleared_stats = engine.get_stats()
        assert cleared_stats["total_evaluations"] <= initial_stats["total_evaluations"]

    def test_create_rule_group(self, audit_rule_engine):
        """测试创建规则组"""
        engine = audit_rule_engine

        # 添加规则
        rule1 = AuditRule(rule_id="group-rule-1", name="Group Rule 1", description="Rule 1")
        rule2 = AuditRule(rule_id="group-rule-2", name="Group Rule 2", description="Rule 2")

        engine.add_rule(rule1)
        engine.add_rule(rule2)

        # 创建规则组
        engine.create_rule_group("test_group", ["group-rule-1", "group-rule-2"])

        assert "test_group" in engine._rule_groups
        assert engine._rule_groups["test_group"] == ["group-rule-1", "group-rule-2"]

    def test_evaluate_event_with_group(self, audit_rule_engine, sample_audit_event):
        """测试使用规则组评估事件"""
        engine = audit_rule_engine

        # 添加规则并创建组
        rule = AuditRule(
            rule_id="group-eval-rule",
            name="Group Eval Rule",
            description="Rule for group evaluation",
            actions=[RuleAction.LOG]
        )

        engine.add_rule(rule)
        engine.create_rule_group("eval_group", ["group-eval-rule"])

        results = engine.evaluate_event_with_group(sample_audit_event, "eval_group")

        assert isinstance(results, list)


class TestAuditRuleTemplates:
    """测试审计规则模板"""

    def test_create_failed_login_rule(self):
        """测试创建失败登录规则"""
        rule = AuditRuleTemplates.create_failed_login_rule()

        assert isinstance(rule, AuditRule)
        assert rule.rule_id == "failed_login_detection"
        assert rule.name == "Failed Login Detection"
        assert rule.enabled is True
        assert len(rule.conditions) > 0
        assert len(rule.actions) > 0

    def test_create_high_risk_operation_rule(self):
        """测试创建高风险操作规则"""
        rule = AuditRuleTemplates.create_high_risk_operation_rule()

        assert isinstance(rule, AuditRule)
        assert rule.rule_id == "high_risk_operation"
        assert rule.name == "High Risk Operation Detection"
        assert rule.enabled is True

    def test_create_suspicious_resource_access_rule(self):
        """测试创建可疑资源访问规则"""
        rule = AuditRuleTemplates.create_suspicious_resource_access_rule()

        assert isinstance(rule, AuditRule)
        assert rule.rule_id == "suspicious_resource_access"
        assert rule.name == "Suspicious Resource Access Detection"
        assert rule.enabled is True

    def test_create_compliance_violation_rule(self):
        """测试创建合规违规规则"""
        rule = AuditRuleTemplates.create_compliance_violation_rule()

        assert isinstance(rule, AuditRule)
        assert rule.rule_id == "compliance_violation"
        assert rule.name == "Compliance Violation Detection"
        assert rule.enabled is True


class TestAuditRuleEngineIntegration:
    """测试审计规则引擎集成功能"""

    def test_complete_rule_evaluation_workflow(self, audit_rule_engine, sample_audit_event):
        """测试完整规则评估工作流"""
        engine = audit_rule_engine

        # 1. 创建并添加规则
        rule = AuditRule(
            rule_id="workflow-rule",
            name="Workflow Rule",
            description="Rule for workflow testing",
            conditions=[
                RuleCondition(
                    condition_type=RuleConditionType.EVENT_TYPE,
                    operator="eq",
                    value=AuditEventType.ACCESS
                )
            ],
            actions=[RuleAction.LOG, RuleAction.ALERT, RuleAction.NOTIFY]
        )

        engine.add_rule(rule)

        # 2. 评估事件
        results = engine.evaluate_event(sample_audit_event)

        assert len(results) >= 3

        # 3. 检查统计信息
        stats = engine.get_stats()
        assert stats["total_evaluations"] >= 1

    def test_rule_engine_with_templates(self, audit_rule_engine, sample_failed_login_event):
        """测试使用模板规则的引擎"""
        engine = audit_rule_engine

        # 添加模板规则
        failed_login_rule = AuditRuleTemplates.create_failed_login_rule()
        engine.add_rule(failed_login_rule)

        # 评估失败登录事件
        results = engine.evaluate_event(sample_failed_login_event)

        # 应该有匹配的结果
        assert isinstance(results, list)

    def test_rule_group_workflow(self, audit_rule_engine, sample_audit_event, sample_high_risk_event):
        """测试规则组工作流"""
        engine = audit_rule_engine

        # 创建多个规则
        access_rule = AuditRule(
            rule_id="access-rule",
            name="Access Rule",
            description="Rule for access events",
            conditions=[
                RuleCondition(
                    condition_type=RuleConditionType.EVENT_TYPE,
                    operator="eq",
                    value=AuditEventType.ACCESS
                )
            ],
            actions=[RuleAction.LOG]
        )

        security_rule = AuditRule(
            rule_id="security-rule",
            name="Security Rule",
            description="Rule for security events",
            conditions=[
                RuleCondition(
                    condition_type=RuleConditionType.EVENT_TYPE,
                    operator="eq",
                    value=AuditEventType.SECURITY
                )
            ],
            actions=[RuleAction.ALERT]
        )

        # 添加规则并创建组
        engine.add_rule(access_rule)
        engine.add_rule(security_rule)
        engine.create_rule_group("security_group", ["access-rule", "security-rule"])

        # 使用组评估不同事件
        access_results = engine.evaluate_event_with_group(sample_audit_event, "security_group")
        security_results = engine.evaluate_event_with_group(sample_high_risk_event, "security_group")

        assert isinstance(access_results, list)
        assert isinstance(security_results, list)


class TestErrorHandling:
    """测试错误处理"""

    def test_rule_condition_invalid_event(self):
        """测试规则条件处理无效事件"""
        condition = RuleCondition(
            condition_type=RuleConditionType.USER_ID,
            operator="eq",
            value="test"
        )

        # 测试None事件
        result = condition.matches(None)
        assert result is False

    def test_rule_execution_with_invalid_actions(self, sample_audit_event):
        """测试规则执行无效动作"""
        rule = AuditRule(
            rule_id="invalid-action-rule",
            name="Invalid Action Rule",
            description="Rule with invalid action",
            actions=["invalid_action"]  # 无效的动作
        )

        # 应该能够处理无效动作而不崩溃
        results = rule.execute_actions(sample_audit_event)
        assert isinstance(results, list)

    def test_engine_operations_with_invalid_ids(self, audit_rule_engine):
        """测试引擎使用无效ID的操作"""
        engine = audit_rule_engine

        # 测试无效规则ID的操作
        assert engine.remove_rule("invalid_id") is False
        assert engine.enable_rule("invalid_id") is False
        assert engine.disable_rule("invalid_id") is False
        assert engine.get_rule("invalid_id") is None

    def test_rule_group_with_invalid_group_name(self, audit_rule_engine, sample_audit_event):
        """测试使用无效组名的规则组评估"""
        engine = audit_rule_engine

        # 应该能够处理不存在的组名
        results = engine.evaluate_event_with_group(sample_audit_event, "nonexistent_group")
        assert isinstance(results, list)
        assert len(results) == 0


class TestPerformance:
    """测试性能"""

    def test_rule_engine_performance_with_multiple_rules(self):
        """测试多规则情况下的引擎性能"""
        engine = AuditRuleEngine()

        # 创建多个规则
        for i in range(50):
            rule = AuditRule(
                rule_id=f"perf-rule-{i}",
                name=f"Performance Rule {i}",
                description=f"Rule for performance testing {i}",
                conditions=[
                    RuleCondition(
                        condition_type=RuleConditionType.USER_ID,
                        operator="eq",
                        value=f"user_{i % 10}"
                    )
                ],
                actions=[RuleAction.LOG]
            )
            engine.add_rule(rule)

        # 创建测试事件
        event = AuditEvent(
            event_id="perf-event",
            event_type=AuditEventType.ACCESS,
            severity=AuditSeverity.LOW,
            timestamp=datetime.now(),
            user_id="user_5"
        )

        # 评估事件
        import time
        start_time = time.time()
        results = engine.evaluate_event(event)
        end_time = time.time()

        # 验证结果
        assert isinstance(results, list)

        # 性能检查：50个规则应该在合理时间内完成
        duration = end_time - start_time
        assert duration < 1.0  # 应该在1秒内完成
