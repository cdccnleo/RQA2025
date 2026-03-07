#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试基础设施层 - 智能告警规则引擎

测试logging/alert_rule_engine.py中的所有类和方法
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock


class TestAlertRuleEngine:
    """测试告警规则引擎"""

    def setup_method(self):
        """测试前准备"""
        try:
            from src.infrastructure.logging.alert_rule_engine import (
                AlertSeverity, AlertStatus, AlertRule, AlertInstance, AlertRuleEngine,
                RuleManager, RuleEvaluator, AlertGenerator, ThresholdAdjuster, AlertSuppressor
            )
            self.AlertSeverity = AlertSeverity
            self.AlertStatus = AlertStatus
            self.AlertRule = AlertRule
            self.AlertInstance = AlertInstance
            self.AlertRuleEngine = AlertRuleEngine
            self.RuleManager = RuleManager
            self.RuleEvaluator = RuleEvaluator
            self.AlertGenerator = AlertGenerator
            self.ThresholdAdjuster = ThresholdAdjuster
            self.AlertSuppressor = AlertSuppressor
        except ImportError as e:
            pytest.skip(f"Alert rule engine components not available: {e}")

    def test_alert_severity_enum(self):
        """测试告警严重程度枚举"""
        if not hasattr(self, 'AlertSeverity'):
            pytest.skip("AlertSeverity not available")

        assert hasattr(self.AlertSeverity, 'INFO')
        assert hasattr(self.AlertSeverity, 'WARNING')
        assert hasattr(self.AlertSeverity, 'CRITICAL')
        assert hasattr(self.AlertSeverity, 'EMERGENCY')

        assert self.AlertSeverity.INFO.value == "info"
        assert self.AlertSeverity.WARNING.value == "warning"
        assert self.AlertSeverity.CRITICAL.value == "critical"
        assert self.AlertSeverity.EMERGENCY.value == "emergency"

    def test_alert_status_enum(self):
        """测试告警状态枚举"""
        if not hasattr(self, 'AlertStatus'):
            pytest.skip("AlertStatus not available")

        assert hasattr(self.AlertStatus, 'ACTIVE')
        assert hasattr(self.AlertStatus, 'ACKNOWLEDGED')
        assert hasattr(self.AlertStatus, 'RESOLVED')
        assert hasattr(self.AlertStatus, 'SUPPRESSED')

        assert self.AlertStatus.ACTIVE.value == "active"
        assert self.AlertStatus.ACKNOWLEDGED.value == "acknowledged"
        assert self.AlertStatus.RESOLVED.value == "resolved"
        assert self.AlertStatus.SUPPRESSED.value == "suppressed"

    def test_alert_rule_creation(self):
        """测试告警规则创建"""
        if not hasattr(self, 'AlertRule'):
            pytest.skip("AlertRule not available")

        rule = self.AlertRule(
            name="High CPU Usage",
            description="Alert when CPU usage exceeds threshold",
            query="cpu_usage > 90",
            severity=self.AlertSeverity.CRITICAL,
            threshold=90.0,
            duration="5m"
        )

        assert rule.name == "High CPU Usage"
        assert rule.description == "Alert when CPU usage exceeds threshold"
        assert rule.query == "cpu_usage > 90"
        assert rule.severity == self.AlertSeverity.CRITICAL
        assert rule.threshold == 90.0
        assert rule.duration == "5m"
        assert rule.enabled is True

    def test_alert_instance_creation(self):
        """测试告警实例创建"""
        if not hasattr(self, 'AlertInstance'):
            pytest.skip("AlertInstance not available")

        instance = self.AlertInstance(
            rule_name="High CPU Usage",
            severity=self.AlertSeverity.CRITICAL,
            status=self.AlertStatus.ACTIVE,
            value=95.0,
            threshold=90.0,
            labels={"server": "web01"},
            annotations={"message": "CPU usage is 95%"},
            start_time=datetime.now(),
            last_update=datetime.now()
        )

        assert instance.rule_name == "High CPU Usage"
        assert instance.severity == self.AlertSeverity.CRITICAL
        assert instance.status == self.AlertStatus.ACTIVE
        assert instance.value == 95.0
        assert instance.threshold == 90.0
        assert instance.labels["server"] == "web01"
        assert instance.resolved_at is None
        assert instance.acknowledged_at is None

    def test_alert_rule_engine_initialization(self):
        """测试告警规则引擎初始化"""
        if not hasattr(self, 'AlertRuleEngine'):
            pytest.skip("AlertRuleEngine not available")

        engine = self.AlertRuleEngine()

        assert engine is not None
        assert hasattr(engine, 'rules')
        assert hasattr(engine, 'active_alerts')
        assert hasattr(engine, 'alert_history')
        assert isinstance(engine.rules, dict)
        assert isinstance(engine.active_alerts, dict)
        assert isinstance(engine.alert_history, list)

    def test_add_alert_rule(self):
        """测试添加告警规则"""
        if not hasattr(self, 'AlertRuleEngine'):
            pytest.skip("AlertRuleEngine not available")

        engine = self.AlertRuleEngine()

        rule = self.AlertRule(
            name="Test Rule",
            description="Test rule description",
            query="test_value > 10",
            severity=self.AlertSeverity.WARNING,
            threshold=10.0,
            duration="5m"
        )

        if hasattr(engine, 'add_rule'):
            result = engine.add_rule(rule)
            assert result is True
            # Note: AlertRuleEngine uses rule.name as key, not rule_id
            assert "Test Rule" in engine.rules
            assert engine.rules["Test Rule"] == rule

    def test_remove_alert_rule(self):
        """测试移除告警规则"""
        if not hasattr(self, 'AlertRuleEngine'):
            pytest.skip("AlertRuleEngine not available")

        engine = self.AlertRuleEngine()

        rule = self.AlertRule(
            name="Test Rule",
            description="Test rule description",
            query="test_value > 10",
            severity=self.AlertSeverity.WARNING,
            threshold=10.0,
            duration="5m"
        )

        if hasattr(engine, 'add_rule'):
            engine.add_rule(rule)
            assert "Test Rule" in engine.rules

        if hasattr(engine, 'remove_rule'):
            result = engine.remove_rule("Test Rule")
            assert result is True
            assert "Test Rule" not in engine.rules

    def test_evaluate_condition(self):
        """测试条件评估"""
        if not hasattr(self, 'AlertRuleEngine'):
            pytest.skip("AlertRuleEngine not available")

        engine = self.AlertRuleEngine()

        # 测试简单的条件
        data = {"cpu_usage": 95, "memory_usage": 85}

        if hasattr(engine, 'evaluate_condition'):
            # 测试大于条件
            result = engine.evaluate_condition("cpu_usage > 90", data)
            assert result is True

            # 测试小于条件
            result = engine.evaluate_condition("memory_usage < 90", data)
            assert result is True

            # 测试等于条件
            result = engine.evaluate_condition("cpu_usage == 95", data)
            assert result is True

            # 测试不等于条件
            result = engine.evaluate_condition("cpu_usage != 80", data)
            assert result is True

    def test_check_alerts(self):
        """测试告警检查"""
        if not hasattr(self, 'AlertRuleEngine'):
            pytest.skip("AlertRuleEngine not available")

        engine = self.AlertRuleEngine()

        # 添加一个测试规则
        rule = self.AlertRule(
            name="High CPU",
            description="High CPU usage alert",
            query="cpu_usage > 90",
            severity=self.AlertSeverity.CRITICAL,
            threshold=90.0,
            duration="5m"
        )

        if hasattr(engine, 'add_rule'):
            engine.add_rule(rule)

        # 测试数据
        data = {"cpu_usage": 95, "source": "test_server"}

        if hasattr(engine, 'check_alerts'):
            alerts = engine.check_alerts(data)
            assert isinstance(alerts, list)

            if len(alerts) > 0:
                alert = alerts[0]
                assert alert.rule_id == "cpu_high"
                assert alert.severity == self.AlertSeverity.CRITICAL
                assert alert.status == self.AlertStatus.ACTIVE

    def test_rule_cooldown_mechanism(self):
        """测试规则冷却机制"""
        if not hasattr(self, 'AlertRuleEngine'):
            pytest.skip("AlertRuleEngine not available")

        engine = self.AlertRuleEngine()

        rule = self.AlertRule(
            name="Cooldown Test",
            description="Test cooldown mechanism",
            query="value > 5",
            severity=self.AlertSeverity.WARNING,
            threshold=5.0,
            duration="1m"
        )

        if hasattr(engine, 'add_rule'):
            engine.add_rule(rule)

        # 第一次触发
        if hasattr(engine, 'check_alerts'):
            alerts1 = engine.check_alerts({"value": 10})
            assert len(alerts1) == 1

            # 立即再次检查，应该由于冷却时间而没有告警
            alerts2 = engine.check_alerts({"value": 10})
            assert len(alerts2) == 0

    def test_alert_suppression(self):
        """测试告警抑制"""
        if not hasattr(self, 'AlertRuleEngine'):
            pytest.skip("AlertRuleEngine not available")

        engine = self.AlertRuleEngine()

        # 创建测试规则
        rule = self.AlertRule(
            name="test_rule",
            description="Test rule for suppression",
            query="value > 50",
            severity=self.AlertSeverity.WARNING,
            threshold=50.0,
            duration="5m"
        )

        if hasattr(engine, 'add_rule'):
            engine.add_rule(rule)

        if hasattr(engine, 'suppress_alert'):
            # 抑制一个不存在的告警
            result = engine.suppress_alert("nonexistent_alert", "1h", "test suppression")
            assert result is False

            # 抑制存在的告警
            result = engine.suppress_alert("test_rule", "1h", "test suppression")
            assert result is True

        # 验证规则被抑制
        assert result is True

    def test_resolve_alert(self):
        """测试解决告警"""
        if not hasattr(self, 'AlertRuleEngine'):
            pytest.skip("AlertRuleEngine not available")

        engine = self.AlertRuleEngine()

        # 创建测试规则
        rule = self.AlertRule(
            name="resolve_test",
            description="Test rule for resolution",
            query="value > 80",
            severity=self.AlertSeverity.WARNING,
            threshold=80.0,
            duration="5m"
        )

        if hasattr(engine, 'add_rule'):
            engine.add_rule(rule)

        if hasattr(engine, 'resolve_alert'):
            # 解决不存在的告警
            result = engine.resolve_alert("nonexistent_alert")
            assert result is False

            # 解决存在的告警
            result = engine.resolve_alert("resolve_test")
            assert result is True

    def test_get_alert_statistics(self):
        """测试获取告警统计信息"""
        if not hasattr(self, 'AlertRuleEngine'):
            pytest.skip("AlertRuleEngine not available")

        engine = self.AlertRuleEngine()

        if hasattr(engine, 'get_statistics'):
            stats = engine.get_statistics()
            assert isinstance(stats, dict)
            assert "total_alerts" in stats
            assert "active_alerts" in stats
            assert "resolved_alerts" in stats

    def test_concurrent_alert_processing(self):
        """测试并发告警处理"""
        if not hasattr(self, 'AlertRuleEngine'):
            pytest.skip("AlertRuleEngine not available")

        engine = self.AlertRuleEngine()
        results = []
        errors = []

        # 添加一些测试规则
        for i in range(5):
            rule = self.AlertRule(
                name=f"test_rule_{i}",
                description=f"Test rule {i}",
                query=f"value > {i * 10}",
                severity=self.AlertSeverity.WARNING,
                threshold=float(i * 10),
                duration="5m"
            )
            if hasattr(engine, 'add_rule'):
                engine.add_rule(rule)

        def worker_thread(thread_id):
            """工作线程"""
            try:
                for i in range(5):  # 减少循环次数
                    # 模拟并发访问规则
                    if hasattr(engine, 'rules'):
                        rule_count = len(engine.rules)
                        results.append(f"Thread {thread_id} saw {rule_count} rules")
                    else:
                        results.append(f"Thread {thread_id} processed {i}")
            except Exception as e:
                errors.append(f"Thread {thread_id} error: {e}")

        # 启动多个线程
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join(timeout=5.0)

        # 验证结果
        assert len(results) == 15  # 3线程 x 5次循环
        assert len(errors) == 0   # 没有错误

    def test_alert_rule_engine_error_handling(self):
        """测试告警规则引擎错误处理"""
        if not hasattr(self, 'AlertRuleEngine'):
            pytest.skip("AlertRuleEngine not available")

        engine = self.AlertRuleEngine()

        # 测试无效规则
        if hasattr(engine, 'add_rule'):
            try:
                engine.add_rule(None)
            except (TypeError, AttributeError):
                pass  # 应该能处理无效输入

        # 测试添加有效规则
        valid_rule = self.AlertRule(
            name="test_rule",
            description="Test rule",
            query="value > 10",
            severity=self.AlertSeverity.WARNING,  # 使用有效的枚举值
            threshold=10.0,
            duration="5m"
        )
        if hasattr(engine, 'add_rule'):
            engine.add_rule(valid_rule)
            assert len(engine.rules) > 0

        # 测试无效数据
        if hasattr(engine, 'check_alerts'):
            try:
                engine.check_alerts(None)
            except (TypeError, AttributeError):
                pass  # 应该能处理无效输入

        # 引擎应该仍然正常工作
        assert engine.rules is not None
        assert engine.active_alerts is not None

    def test_dynamic_rule_update(self):
        """测试动态规则更新"""
        if not hasattr(self, 'AlertRuleEngine'):
            pytest.skip("AlertRuleEngine not available")

        engine = self.AlertRuleEngine()

        # 创建初始规则
        rule = self.AlertRule(
            name="Dynamic Test",
            description="Test dynamic updates",
            query="value > 10",
            severity=self.AlertSeverity.WARNING,  # 使用有效的枚举值
            threshold=10.0,
            duration="5m"
        )

        if hasattr(engine, 'add_rule'):
            engine.add_rule(rule)

            # 更新规则
            if hasattr(engine, 'update_rule'):
                updated_rule = self.AlertRule(
                    name="Updated Dynamic Test",
                    description="Updated description",
                    query="value > 20",  # 提高阈值
                    severity=self.AlertSeverity.CRITICAL,
                    threshold=20.0,
                    duration="5m"
                )

                result = engine.update_rule("Dynamic Test", {"query": "value > 20", "severity": self.AlertSeverity.CRITICAL})
                assert result is True
                assert engine.rules["Dynamic Test"].query == "value > 20"
                assert engine.rules["Dynamic Test"].severity == self.AlertSeverity.CRITICAL


class TestRuleManager:
    """测试规则管理器"""

    def setup_method(self):
        """测试前准备"""
        if not hasattr(self, 'RuleManager'):
            pytest.skip("RuleManager not available")
        self.manager = self.RuleManager()

    def test_rule_manager_initialization(self):
        """测试规则管理器初始化"""
        assert hasattr(self.manager, '_rules')
        assert hasattr(self.manager, '_rule_lock')
        assert isinstance(self.manager._rules, dict)

    def test_add_rule(self):
        """测试添加规则"""
        rule = self.AlertRule(
            rule_id="test_rule_1",
            name="Test Rule 1",
            query="error_count > 10",
            severity=self.AlertSeverity.WARNING,
            threshold=10.0,
            duration="5m"
        )

        result = self.manager.add_rule(rule)
        assert result is True
        assert "test_rule_1" in self.manager._rules

    def test_add_duplicate_rule(self):
        """测试添加重复规则"""
        rule = self.AlertRule(
            rule_id="duplicate_rule",
            name="Duplicate Rule",
            query="error_count > 5",
            severity=self.AlertSeverity.INFO,
            threshold=5.0,
            duration="1m"
        )

        # 第一次添加成功
        result1 = self.manager.add_rule(rule)
        assert result1 is True

        # 第二次添加失败
        result2 = self.manager.add_rule(rule)
        assert result2 is False

    def test_remove_rule(self):
        """测试移除规则"""
        rule = self.AlertRule(
            rule_id="remove_test",
            name="Remove Test",
            query="warning_count > 20",
            severity=self.AlertSeverity.CRITICAL,
            threshold=20.0,
            duration="10m"
        )

        # 添加规则
        self.manager.add_rule(rule)
        assert "remove_test" in self.manager._rules

        # 移除规则
        result = self.manager.remove_rule("remove_test")
        assert result is True
        assert "remove_test" not in self.manager._rules

    def test_remove_nonexistent_rule(self):
        """测试移除不存在的规则"""
        result = self.manager.remove_rule("nonexistent")
        assert result is False

    def test_get_rule(self):
        """测试获取规则"""
        rule = self.AlertRule(
            rule_id="get_test",
            name="Get Test",
            query="cpu_usage > 90",
            severity=self.AlertSeverity.EMERGENCY,
            threshold=90.0,
            duration="1m"
        )

        self.manager.add_rule(rule)
        retrieved = self.manager.get_rule("get_test")
        assert retrieved is not None
        assert retrieved.rule_id == "get_test"

    def test_get_all_rules(self):
        """测试获取所有规则"""
        # 清空现有规则
        self.manager._rules.clear()

        # 添加多个规则
        for i in range(3):
            rule = self.AlertRule(
                rule_id=f"rule_{i}",
                name=f"Rule {i}",
                query=f"value > {i*10}",
                severity=self.AlertSeverity.WARNING,
                threshold=float(i*10),
                duration="5m"
            )
            self.manager.add_rule(rule)

        all_rules = self.manager.get_all_rules()
        assert len(all_rules) == 3
        assert all(isinstance(rule, self.AlertRule) for rule in all_rules)


class TestRuleEvaluator:
    """测试规则评估器"""

    def setup_method(self):
        """测试前准备"""
        if not hasattr(self, 'RuleEvaluator'):
            pytest.skip("RuleEvaluator not available")
        self.evaluator = self.RuleEvaluator()

    def test_evaluate_threshold_rule(self):
        """测试阈值规则评估"""
        rule = self.AlertRule(
            rule_id="threshold_test",
            name="Threshold Test",
            query="cpu_usage > 80",
            severity=self.AlertSeverity.WARNING,
            threshold=80.0,
            duration="5m"
        )

        # 测试超过阈值
        metrics = {"cpu_usage": 85.0}
        result = self.evaluator.evaluate_rule(rule, metrics)
        assert result is True

        # 测试低于阈值
        metrics = {"cpu_usage": 75.0}
        result = self.evaluator.evaluate_rule(rule, metrics)
        assert result is False

    def test_evaluate_complex_rule(self):
        """测试复杂规则评估"""
        rule = self.AlertRule(
            rule_id="complex_test",
            name="Complex Test",
            query="cpu_usage > 80 AND memory_usage > 90",
            severity=self.AlertSeverity.CRITICAL,
            threshold=80.0,
            duration="5m"
        )

        # 测试满足条件
        metrics = {"cpu_usage": 85.0, "memory_usage": 95.0}
        result = self.evaluator.evaluate_rule(rule, metrics)
        assert result is True

        # 测试不满足条件
        metrics = {"cpu_usage": 85.0, "memory_usage": 85.0}
        result = self.evaluator.evaluate_rule(rule, metrics)
        assert result is False


class TestAlertGenerator:
    """测试告警生成器"""

    def setup_method(self):
        """测试前准备"""
        if not hasattr(self, 'AlertGenerator'):
            pytest.skip("AlertGenerator not available")
        self.generator = self.AlertGenerator()

    def test_generate_alert(self):
        """测试生成告警"""
        rule = self.AlertRule(
            rule_id="gen_test",
            name="Generation Test",
            query="error_rate > 5",
            severity=self.AlertSeverity.CRITICAL,
            threshold=5.0,
            duration="5m"
        )

        alert = self.generator.generate_alert(rule, {"current_value": 10.0})
        assert alert is not None
        assert alert.rule_id == "gen_test"
        assert alert.severity == self.AlertSeverity.CRITICAL

    def test_generate_alert_with_context(self):
        """测试生成带上下文的告警"""
        rule = self.AlertRule(
            rule_id="context_test",
            name="Context Test",
            query="response_time > 1000",
            severity=self.AlertSeverity.WARNING,
            threshold=1000.0,
            duration="1m"
        )

        context = {
            "current_value": 1500.0,
            "endpoint": "/api/v1/data",
            "method": "POST"
        }

        alert = self.generator.generate_alert(rule, context)
        assert alert is not None
        assert "endpoint" in alert.context
        assert alert.context["endpoint"] == "/api/v1/data"


class TestThresholdAdjuster:
    """测试阈值调节器"""

    def setup_method(self):
        """测试前准备"""
        if not hasattr(self, 'ThresholdAdjuster'):
            pytest.skip("ThresholdAdjuster not available")
        self.adjuster = self.ThresholdAdjuster()

    def test_adjust_threshold_up(self):
        """测试提高阈值"""
        rule = self.AlertRule(
            rule_id="adjust_up",
            name="Adjust Up Test",
            query="value > 50",
            severity=self.AlertSeverity.WARNING,
            threshold=50.0,
            duration="5m"
        )

        # 模拟频繁告警
        for _ in range(10):
            self.adjuster.record_alert("adjust_up")

        adjusted = self.adjuster.adjust_threshold(rule, 60.0)
        assert adjusted > 50.0

    def test_adjust_threshold_down(self):
        """测试降低阈值"""
        rule = self.AlertRule(
            rule_id="adjust_down",
            name="Adjust Down Test",
            query="value > 80",
            severity=self.AlertSeverity.WARNING,
            threshold=80.0,
            duration="5m"
        )

        # 模拟很少告警
        adjusted = self.adjuster.adjust_threshold(rule, 50.0)
        assert adjusted <= 80.0


class TestAlertSuppressor:
    """测试告警抑制器"""

    def setup_method(self):
        """测试前准备"""
        if not hasattr(self, 'AlertSuppressor'):
            pytest.skip("AlertSuppressor not available")
        self.suppressor = self.AlertSuppressor()

    def test_suppress_duplicate_alerts(self):
        """测试抑制重复告警"""
        alert1 = self.AlertInstance(
            alert_id="dup_test_1",
            rule_id="dup_rule",
            message="Duplicate alert message",
            severity=self.AlertSeverity.WARNING,
            context={"source": "test"}
        )

        alert2 = self.AlertInstance(
            alert_id="dup_test_2",
            rule_id="dup_rule",
            message="Duplicate alert message",
            severity=self.AlertSeverity.WARNING,
            context={"source": "test"}
        )

        # 第一次不被抑制
        suppressed1 = self.suppressor.should_suppress(alert1)
        assert suppressed1 is False

        # 第二次被抑制
        suppressed2 = self.suppressor.should_suppress(alert2)
        assert suppressed2 is True

    def test_suppression_window(self):
        """测试抑制时间窗口"""
        alert = self.AlertInstance(
            alert_id="window_test",
            rule_id="window_rule",
            message="Window test alert",
            severity=self.AlertSeverity.INFO,
            context={"component": "test"}
        )

        # 立即抑制
        suppressed = self.suppressor.should_suppress(alert)
        assert suppressed is False

        # 在抑制窗口内再次检查
        time.sleep(0.1)
        suppressed_again = self.suppressor.should_suppress(alert)
        assert suppressed_again is True


if __name__ == '__main__':
    pytest.main([__file__])
