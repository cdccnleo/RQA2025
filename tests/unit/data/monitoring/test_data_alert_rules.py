import asyncio
import pandas as pd
from unittest.mock import Mock

# Mock数据管理器模块以绕过复杂的导入问题
mock_data_manager = Mock()
mock_data_manager.DataManager = Mock()
mock_data_manager.DataLoaderError = Exception

# 配置DataManager实例方法
mock_instance = Mock()
mock_instance.validate_all_configs.return_value = True
mock_instance.health_check.return_value = {"status": "healthy"}
mock_instance.store_data.return_value = True
mock_instance.has_data.return_value = True
mock_instance.get_metadata.return_value = {"data_type": "test", "symbol": "X"}
mock_instance.retrieve_data.return_value = pd.DataFrame({"col": [1, 2, 3]})
mock_instance.get_stats.return_value = {"total_items": 1}
mock_instance.validate_data.return_value = {"valid": True}
mock_instance.shutdown.return_value = None

mock_data_manager.DataManager.return_value = mock_instance

# Mock整个模块
import sys
sys.modules["src.data.data_manager"] = mock_data_manager


import importlib
import json
import sys
from datetime import datetime, timedelta
from types import ModuleType, MethodType

import pytest
from enum import Enum


try:
    integration_module = importlib.import_module("src.core.integration")
except ImportError:
    integration_module = ModuleType("src.core.integration")
    sys.modules["src.core.integration"] = integration_module
    if "src.core" not in sys.modules:
        sys.modules["src.core"] = ModuleType("src.core")
    setattr(sys.modules["src.core"], "integration", integration_module)


if not hasattr(integration_module, "get_data_adapter"):
    def _default_adapter():
        class DummyAdapter:
            def get_monitoring(self):
                return None

            def get_logger(self):
                class _Logger:
                    def info(self, *args, **kwargs):
                        return None

                    def warning(self, *args, **kwargs):
                        return None

                    def error(self, *args, **kwargs):
                        return None

                return _Logger()

        return DummyAdapter()

    integration_module.get_data_adapter = _default_adapter


if "src.interfaces.standard_interfaces" not in sys.modules:
    interfaces_module = ModuleType("src.interfaces.standard_interfaces")

    class _DataSourceType(Enum):
        UNKNOWN = "unknown"
        CACHE = "cache"
        API = "api"

    interfaces_module.DataSourceType = _DataSourceType

    if "src.interfaces" not in sys.modules:
        sys.modules["src.interfaces"] = ModuleType("src.interfaces")
    setattr(sys.modules["src.interfaces"], "standard_interfaces", interfaces_module)
    sys.modules["src.interfaces.standard_interfaces"] = interfaces_module


from src.data.monitoring.data_alert_rules import (
    AlertCondition,
    AlertConditionType,
    AlertRule,
    AlertRuleType,
    AlertSeverity,
    AlertSuppression,
    DataAlertRulesEngine,
)
from src.interfaces.standard_interfaces import DataSourceType


@pytest.fixture
def engine(monkeypatch):
    class DummyAdapter:
        def get_monitoring(self):
            return None

        def get_logger(self):
            class _Logger:
                def info(self, *args, **kwargs):
                    return None

                def warning(self, *args, **kwargs):
                    return None

                def error(self, *args, **kwargs):
                    return None

            return _Logger()

    monkeypatch.setattr(
        "src.data.monitoring.data_alert_rules.get_data_adapter",
        lambda: DummyAdapter(),
    )

    eng = DataAlertRulesEngine()
    eng.rules.clear()
    eng.rule_performance.clear()
    eng.alert_history.clear()
    eng.suppressions.clear()
    return eng


def _build_threshold_rule(rule_id="test_rule"):
    return AlertRule(
        rule_id=rule_id,
        name="threshold rule",
        description="demo",
        rule_type=AlertRuleType.THRESHOLD,
        conditions=[
            AlertCondition(
                type=AlertConditionType.LESS_THAN,
                field="hit_rate",
                value=0.8,
            )
        ],
        severity=AlertSeverity.WARNING,
        message_template="hit rate {hit_rate:.2%}",
        cooldown_minutes=5,
    )


def test_alert_condition_evaluate_variations():
    data = {"metric": 10, "status": "failed"}

    assert AlertCondition(AlertConditionType.GREATER_THAN, "metric", 5).evaluate(data) is True
    assert AlertCondition(AlertConditionType.LESS_THAN, "metric", 20).evaluate(data) is True
    assert AlertCondition(AlertConditionType.EQUAL, "status", "failed").evaluate(data) is True
    assert AlertCondition(AlertConditionType.NOT_EQUAL, "status", "ok").evaluate(data) is True
    assert AlertCondition(AlertConditionType.GREATER_EQUAL, "metric", 10).evaluate(data) is True
    assert AlertCondition(AlertConditionType.LESS_EQUAL, "metric", 10).evaluate(data) is True

    assert AlertCondition(
        AlertConditionType.BETWEEN, "metric", [5, 15]
    ).evaluate(data) is True
    assert AlertCondition(
        AlertConditionType.OUTSIDE, "metric", [0, 5]
    ).evaluate(data) is True

    custom_called = {"count": 0}

    def _custom_func(payload):
        custom_called["count"] += 1
        return payload["metric"] == 10

    assert AlertCondition(
        AlertConditionType.CUSTOM, "metric", 0, custom_func=_custom_func
    ).evaluate(data) is True
    assert custom_called["count"] == 1

    change_rate_condition = AlertCondition(
        AlertConditionType.CHANGE_RATE, "metric", 0.1
    )
    assert change_rate_condition.evaluate(data) is False

    missing_field = AlertCondition(AlertConditionType.GREATER_THAN, "missing", 1)
    assert missing_field.evaluate(data) is False

    invalid_between = AlertCondition(AlertConditionType.BETWEEN, "metric", [5])
    assert invalid_between.evaluate(data) is False


def test_alert_rule_data_type_filter_and_condition_errors():
    threshold_rule = AlertRule(
        rule_id="threshold",
        name="threshold",
        description="demo",
        rule_type=AlertRuleType.THRESHOLD,
        conditions=[
            AlertCondition(
                type=AlertConditionType.GREATER_THAN,
                field="metric",
                value=5,
            )
        ],
        severity=AlertSeverity.WARNING,
        message_template="metric {metric}",
        data_types=[],
    )
    assert threshold_rule.evaluate({"metric": 10}) is not None

    filtered_rule = AlertRule(
        rule_id="typed",
        name="typed",
        description="demo",
        rule_type=AlertRuleType.THRESHOLD,
        conditions=[AlertCondition(AlertConditionType.LESS_THAN, "metric", 1)],
        severity=AlertSeverity.INFO,
        message_template="metric {metric}",
        data_types=[None],  # 仅需非空以触发字符串->枚举解析分支
    )
    assert filtered_rule.evaluate({"metric": 0, "data_type": "api"}) is None
    assert filtered_rule.evaluate({"metric": 0, "data_type": "unknown_enum"}) is None

    class RaisingCondition:
        def evaluate(self, payload):
            raise RuntimeError("boom")

    composite_rule = AlertRule(
        rule_id="composite",
        name="composite",
        description="demo",
        rule_type=AlertRuleType.COMPOSITE,
        conditions=[
            AlertCondition(AlertConditionType.LESS_THAN, "metric", 1),
            RaisingCondition(),
        ],
        severity=AlertSeverity.ERROR,
        message_template="metric {metric}",
    )
    alert = composite_rule.evaluate({"metric": 0})
    assert alert is not None


def test_alert_rule_from_dict_ignores_invalid_data_types():
    rule = _build_threshold_rule("from_dict_rule")
    data = rule.to_dict()
    data["data_types"].append("invalid_type")
    reconstructed = AlertRule.from_dict(data)
    assert len(reconstructed.data_types) == len(rule.data_types)


def test_engine_init_fallback_logger(monkeypatch):
    def _raise_adapter():
        raise RuntimeError("adapter init failed")

    monkeypatch.setattr(
        "src.data.monitoring.data_alert_rules.get_data_adapter",
        _raise_adapter,
    )

    engine = DataAlertRulesEngine()
    assert engine.monitoring is None
    assert engine.logger is not None


def test_add_update_remove_rule(engine):
    rule = _build_threshold_rule()
    assert engine.add_rule(rule) is True
    assert rule.rule_id in engine.rules

    assert engine.update_rule(rule.rule_id, {"enabled": False}) is True
    assert engine.rules[rule.rule_id].enabled is False

    assert engine.remove_rule(rule.rule_id) is True
    assert rule.rule_id not in engine.rules


def test_add_rule_duplicate_returns_false(engine):
    rule = _build_threshold_rule("dup_rule")
    assert engine.add_rule(rule) is True
    assert engine.add_rule(rule) is False


def test_update_remove_rule_missing(engine):
    assert engine.update_rule("missing_rule", {"enabled": True}) is False
    assert engine.remove_rule("missing_rule") is False


def test_evaluate_rules_respects_cooldown(engine):
    rule = _build_threshold_rule()
    engine.add_rule(rule)

    data = {"hit_rate": 0.5}
    first_alerts = engine.evaluate_rules(data)
    assert len(first_alerts) == 1
    assert engine.rule_performance[rule.rule_id]["alerts_triggered"] == 1

    second_alerts = engine.evaluate_rules(data)
    assert second_alerts == []


def test_suppression_prevents_alert(engine):
    rule = _build_threshold_rule()
    engine.add_rule(rule)

    suppression = AlertSuppression(
        suppression_id="sup1",
        rule_ids=[rule.rule_id],
        condition=AlertCondition(
            type=AlertConditionType.EQUAL,
            field="environment",
            value="prod",
        ),
        duration_minutes=10,
        reason="maintenance",
        created_at=datetime.now(),
        expires_at=datetime.now() + timedelta(minutes=10),
    )
    engine.add_suppression(suppression)

    data = {"hit_rate": 0.5, "environment": "prod"}
    alerts = engine.evaluate_rules(data)
    assert alerts == []


def test_export_import_roundtrip(engine, monkeypatch):
    rule = _build_threshold_rule("round_trip_rule")
    engine.add_rule(rule)

    suppression = AlertSuppression(
        suppression_id="sup2",
        rule_ids=[rule.rule_id],
        condition=AlertCondition(
            type=AlertConditionType.EQUAL,
            field="service",
            value="api",
        ),
        duration_minutes=5,
        reason="test",
        created_at=datetime.now(),
        expires_at=datetime.now() + timedelta(minutes=5),
    )
    engine.add_suppression(suppression)

    exported = engine.export_rules()

    # new engine for import
    class DummyAdapter:
        def get_monitoring(self):
            return None

        def get_logger(self):
            class _Logger:
                def info(self, *args, **kwargs):
                    return None

                def warning(self, *args, **kwargs):
                    return None

                def error(self, *args, **kwargs):
                    return None

            return _Logger()

    monkeypatch.setattr(
        "src.data.monitoring.data_alert_rules.get_data_adapter",
        lambda: DummyAdapter(),
    )

    new_engine = engine.__class__()
    new_engine.rules.clear()
    new_engine.rule_performance.clear()
    new_engine.suppressions.clear()

    result = new_engine.import_rules(exported)
    assert result["success"] is True
    assert result["imported_rules"] == 1
    assert result["imported_suppressions"] == 1
    assert "round_trip_rule" in new_engine.rules


def test_cleanup_expired_suppressions(engine):
    """测试清理过期抑制"""
    rule = _build_threshold_rule()
    engine.add_rule(rule)

    expired_sup = AlertSuppression(
        suppression_id="expired",
        rule_ids=[rule.rule_id],
        condition=AlertCondition(
            type=AlertConditionType.EQUAL,
            field="env",
            value="test",
        ),
        duration_minutes=1,
        reason="test",
        created_at=datetime.now() - timedelta(minutes=10),
        expires_at=datetime.now() - timedelta(minutes=5),
    )
    active_sup = AlertSuppression(
        suppression_id="active",
        rule_ids=[rule.rule_id],
        condition=AlertCondition(
            type=AlertConditionType.EQUAL,
            field="env",
            value="prod",
        ),
        duration_minutes=10,
        reason="test",
        created_at=datetime.now(),
        expires_at=datetime.now() + timedelta(minutes=10),
    )

    engine.add_suppression(expired_sup)
    engine.add_suppression(active_sup)
    assert len(engine.suppressions) == 2

    cleaned = engine.cleanup_expired_suppressions()
    assert cleaned == 1
    assert "expired" not in engine.suppressions
    assert "active" in engine.suppressions


def test_remove_suppression_missing_raises(engine):
    """测试移除不存在的抑制"""
    assert engine.remove_suppression("nonexistent") is False


def test_remove_suppression_success(engine):
    rule = _build_threshold_rule("sup_rule")
    engine.add_rule(rule)
    suppression = AlertSuppression(
        suppression_id="temp-sup",
        rule_ids=[rule.rule_id],
        condition=AlertCondition(
            type=AlertConditionType.EQUAL,
            field="env",
            value="test",
        ),
        duration_minutes=5,
        reason="cleanup",
        created_at=datetime.now(),
        expires_at=datetime.now() + timedelta(minutes=5),
    )
    assert engine.add_suppression(suppression) is True
    assert engine.remove_suppression(suppression.suppression_id) is True


def test_add_suppression_duplicate_returns_false(engine):
    """测试添加重复抑制ID"""
    sup = AlertSuppression(
        suppression_id="dup",
        rule_ids=["r1"],
        condition=AlertCondition(
            type=AlertConditionType.EQUAL,
            field="x",
            value=1,
        ),
        duration_minutes=5,
        reason="test",
        created_at=datetime.now(),
        expires_at=datetime.now() + timedelta(minutes=5),
    )
    assert engine.add_suppression(sup) is True
    assert engine.add_suppression(sup) is False


def test_get_engine_stats(engine):
    """测试获取引擎统计信息"""
    rule1 = _build_threshold_rule("rule1")
    rule1.severity = AlertSeverity.WARNING
    rule1.enabled = True
    engine.add_rule(rule1)

    rule2 = _build_threshold_rule("rule2")
    rule2.severity = AlertSeverity.CRITICAL
    rule2.enabled = False
    engine.add_rule(rule2)

    data = {"hit_rate": 0.5}
    engine.evaluate_rules(data)

    stats = engine.get_engine_stats()
    assert stats["total_rules"] == 2
    assert stats["enabled_rules"] == 1
    assert stats["disabled_rules"] == 1
    assert "warning" in stats["rules_by_severity"]
    assert "critical" in stats["rules_by_severity"]
    assert stats["total_alerts"] >= 1
    assert "rule1" in stats["alerts_by_rule"]


def test_get_rule_performance_report(engine):
    """测试获取规则性能报告"""
    rule1 = _build_threshold_rule("perf1")
    rule2 = _build_threshold_rule("perf2")
    engine.add_rule(rule1)
    engine.add_rule(rule2)

    for _ in range(3):
        engine.evaluate_rules({"hit_rate": 0.5})

    report = engine.get_rule_performance_report()
    assert "rule_performance" in report
    assert len(report["top_triggered_rules"]) >= 1
    assert report["rule_performance"]["perf1"]["alerts_triggered"] >= 1


def test_import_rules_invalid_json(engine):
    """测试导入无效JSON"""
    result = engine.import_rules("{invalid json")
    assert result["success"] is False
    assert len(result["errors"]) > 0


def test_import_rules_parse_error(engine):
    """测试导入规则解析错误"""
    invalid_rule_json = json.dumps({
        "rules": [{"rule_id": "bad", "invalid": "data"}],
        "suppressions": []
    })
    result = engine.import_rules(invalid_rule_json)
    assert result["imported_rules"] == 0
    assert len(result["errors"]) > 0


def test_import_rules_suppression_parse_error(engine):
    """测试导入抑制解析错误"""
    invalid_suppression_json = json.dumps({
        "rules": [],
        "suppressions": [{"suppression_id": "bad", "missing_fields": True}]
    })
    result = engine.import_rules(invalid_suppression_json)
    assert result["imported_suppressions"] == 0
    assert len(result["errors"]) > 0


def test_import_rules_add_rule_failure(engine, monkeypatch):
    payload = json.dumps({
        "rules": [_build_threshold_rule("fail_rule").to_dict()],
        "suppressions": []
    })

    def _fake_add_rule(self, rule):
        return False

    monkeypatch.setattr(
        engine,
        "add_rule",
        MethodType(_fake_add_rule, engine),
    )

    result = engine.import_rules(payload)
    assert result["imported_rules"] == 0
    assert any("Failed to add rule" in msg for msg in result["errors"])


def test_import_rules_add_suppression_failure(engine, monkeypatch):
    now = datetime.now()
    payload = json.dumps({
        "rules": [],
        "suppressions": [{
            "suppression_id": "sup_fail",
            "rule_ids": ["r1"],
            "condition": {
                "type": AlertConditionType.EQUAL.value,
                "field": "env",
                "value": "prod"
            },
            "duration_minutes": 5,
            "reason": "test",
            "created_at": now.isoformat(),
            "expires_at": (now + timedelta(minutes=5)).isoformat()
        }]
    })

    def _fake_add_suppression(self, suppression):
        return False

    monkeypatch.setattr(
        engine,
        "add_suppression",
        MethodType(_fake_add_suppression, engine),
    )

    result = engine.import_rules(payload)
    assert result["imported_suppressions"] == 0
    assert any("Failed to add suppression" in msg for msg in result["errors"])


def test_cooldown_expires_allows_alert(engine):
    """测试冷却时间过期后允许告警"""
    rule = _build_threshold_rule()
    rule.cooldown_minutes = 1  # 1分钟冷却时间
    engine.add_rule(rule)

    data = {"hit_rate": 0.5}
    first_alerts = engine.evaluate_rules(data)
    assert len(first_alerts) == 1

    # 手动设置last_triggered为过期时间（10分钟前，已超过1分钟冷却）
    engine.rule_performance[rule.rule_id]["last_triggered"] = (
        datetime.now() - timedelta(minutes=10)
    ).isoformat()

    # 应该可以再次触发，因为冷却时间已过期
    second_alerts = engine.evaluate_rules(data)
    assert len(second_alerts) == 1


def test_record_alert_history_limit(engine):
    """测试告警历史记录限制"""
    rule = _build_threshold_rule()
    engine.add_rule(rule)

    # 模拟超过10000条历史记录
    for i in range(10002):
        alert = {
            "rule_id": rule.rule_id,
            "severity": "WARNING",
            "message": f"Alert {i}",
            "cooldown_minutes": 0,
            "timestamp": datetime.now().isoformat(),
        }
        engine.alert_history.append({
            **alert,
            "recorded_at": datetime.now().isoformat(),
        })

    # 触发新告警，应该触发截断逻辑
    engine.evaluate_rules({"hit_rate": 0.5})
    assert len(engine.alert_history) <= 5000


# ---- Spots: data_types 过滤与复合规则异常分支覆盖 --------------------------------
def test_rule_data_types_filter_handles_unknown_string_type():
    rule = AlertRule(
        rule_id="typed_only_cache",
        name="typed",
        description="",
        rule_type=AlertRuleType.THRESHOLD,
        conditions=[AlertCondition(AlertConditionType.LESS_THAN, "metric", 1)],
        severity=AlertSeverity.INFO,
        message_template="m {metric}",
        data_types=[],
    )
    assert rule.evaluate({"metric": 0, "data_type": "api"}) is not None

    from src.interfaces.standard_interfaces import DataSourceType
    # 仅需非空以触发解析与异常分支，无需具体枚举成员
    rule.data_types = [None]
    assert rule.evaluate({"metric": 0, "data_type": "unknown"}) is None
    assert rule.evaluate({"metric": 0, "data_type": "unknown_enum"}) is None


def test_composite_rule_exception_in_condition_is_ignored():
    class RaisingCondition:
        def evaluate(self, payload):
            raise RuntimeError("boom")

    rule = AlertRule(
        rule_id="composite_with_exception",
        name="c",
        description="",
        rule_type=AlertRuleType.COMPOSITE,
        conditions=[
            AlertCondition(AlertConditionType.LESS_THAN, "metric", 1),
            RaisingCondition(),
        ],
        severity=AlertSeverity.WARNING,
        message_template="m {metric}",
    )
    alert = rule.evaluate({"metric": 0})
    assert alert is not None and alert["rule_id"] == "composite_with_exception"


def test_clear_history(engine):
    """测试清除历史记录"""
    rule = _build_threshold_rule()
    engine.add_rule(rule)
    engine.evaluate_rules({"hit_rate": 0.5})

    assert len(engine.alert_history) > 0
    engine.clear_history()
    assert len(engine.alert_history) == 0


def test_reset_rule_performance(engine):
    """测试重置规则性能统计"""
    rule = _build_threshold_rule()
    engine.add_rule(rule)

    engine.evaluate_rules({"hit_rate": 0.5})
    assert engine.rule_performance[rule.rule_id]["alerts_triggered"] > 0

    engine.reset_rule_performance()
    assert engine.rule_performance[rule.rule_id]["alerts_triggered"] == 0
    assert engine.rule_performance[rule.rule_id]["false_positives"] == 0
    assert engine.rule_performance[rule.rule_id]["last_triggered"] is None


def test_log_operation_with_monitoring(engine, monkeypatch):
    """测试日志操作使用监控服务"""
    recorded_metrics = []

    class MockMonitoring:
        def record_metric(self, name, value, tags=None, description=None):
            recorded_metrics.append({
                "name": name,
                "value": value,
                "tags": tags,
                "description": description,
            })

    class MockAdapter:
        def get_monitoring(self):
            return MockMonitoring()

        def get_logger(self):
            class _Logger:
                def warning(self, *args, **kwargs):
                    pass

            return _Logger()

    monkeypatch.setattr(
        "src.data.monitoring.data_alert_rules.get_data_adapter",
        lambda: MockAdapter(),
    )

    eng = DataAlertRulesEngine()
    eng.rules.clear()

    rule = _build_threshold_rule()
    eng.add_rule(rule)

    assert len(recorded_metrics) > 0
    assert any(m["name"] == "alert_engine_operation" for m in recorded_metrics)


def test_log_operation_monitoring_failure_warns(engine):
    warnings = []

    class FailingMonitoring:
        def record_metric(self, *args, **kwargs):
            raise RuntimeError("metric failed")

    class Logger:
        def warning(self, message):
            warnings.append(message)

    engine.monitoring = FailingMonitoring()
    engine.logger = Logger()

    engine._log_operation("test_op", "target", "status")
    assert warnings


def test_log_operation_logger_failure_triggers_outer_except(engine, monkeypatch):
    printed = []

    class FailingMonitoring:
        def record_metric(self, *args, **kwargs):
            raise RuntimeError("metric failed")

    class FailingLogger:
        def warning(self, message):
            raise RuntimeError("log failed")

    def fake_print(*args, **kwargs):
        printed.append(" ".join(str(a) for a in args))

    monkeypatch.setattr("builtins.print", fake_print)

    engine.monitoring = FailingMonitoring()
    engine.logger = FailingLogger()

    engine._log_operation("outer", "target", "status")
    assert any("outer" in msg for msg in printed)


def test_evaluate_rules_exception_handling(engine, monkeypatch):
    """测试评估规则时的异常处理"""
    rule = _build_threshold_rule()

    def failing_evaluate(data, context=None):
        raise RuntimeError("Evaluation failed")

    rule.evaluate = failing_evaluate
    engine.add_rule(rule)

    alerts = engine.evaluate_rules({"hit_rate": 0.5})
    assert alerts == []  # 异常被捕获，不影响其他规则


def test_is_in_cooldown_no_performance(engine):
    """测试冷却检查：无性能记录"""
    # _is_in_cooldown内部会检查rule_id是否在rule_performance中
    # 如果不在，应该返回False（不在冷却中）
    assert not engine._is_in_cooldown("nonexistent_rule", 5)


def test_enable_disable_rule(engine):
    """测试启用和禁用规则"""
    rule = _build_threshold_rule()
    rule.enabled = False
    engine.add_rule(rule)

    assert engine.enable_rule(rule.rule_id) is True
    assert engine.rules[rule.rule_id].enabled is True

    assert engine.disable_rule(rule.rule_id) is True
    assert engine.rules[rule.rule_id].enabled is False

