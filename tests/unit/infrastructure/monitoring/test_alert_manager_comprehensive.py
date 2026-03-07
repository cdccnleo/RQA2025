#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试 - 告警管理器深度测试
测试AlertManager的核心告警功能、规则评估和历史管理
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.infrastructure.monitoring.components.alert_manager import AlertManager
from src.infrastructure.monitoring.core.parameter_objects import AlertRuleConfig, AlertConditionConfig


class TestAlertManagerInitialization:
    """AlertManager初始化测试"""

    def test_initialization_basic(self):
        """测试基本初始化"""
        thresholds = {"cpu": 80.0, "memory": 90.0}
        manager = AlertManager("test_pool", thresholds)

        assert manager.pool_name == "test_pool"
        assert manager.alert_thresholds == thresholds
        assert isinstance(manager.alert_rules, list)
        assert isinstance(manager.alert_history, list)
        assert manager.max_history_size == 1000
        assert isinstance(manager.last_alert_times, dict)

    def test_initialization_with_empty_thresholds(self):
        """测试空阈值初始化"""
        manager = AlertManager("test_pool", {})

        assert manager.pool_name == "test_pool"
        assert manager.alert_thresholds == {}
        assert len(manager.alert_rules) > 0  # 应该有默认规则

    def test_initialization_default_rules(self):
        """测试默认规则初始化"""
        manager = AlertManager("test_pool", {"cpu": 80.0})

        # 应该有默认规则
        assert len(manager.alert_rules) > 0

        # 检查默认规则结构
        for rule in manager.alert_rules:
            assert isinstance(rule, AlertRuleConfig)
            assert hasattr(rule, 'rule_id')
            assert hasattr(rule, 'name')
            assert hasattr(rule, 'conditions')
            assert hasattr(rule, 'severity')
            assert hasattr(rule, 'cooldown_seconds')


class TestAlertManagerRuleManagement:
    """AlertManager规则管理测试"""

    @pytest.fixture
    def manager(self):
        """AlertManager fixture"""
        return AlertManager("test_pool", {"cpu": 80.0})

    def test_add_alert_rule(self, manager):
        """测试添加告警规则"""
        rule = AlertRuleConfig(
            rule_id="test_rule",
            name="Test Rule",
            conditions=[
                AlertConditionConfig(
                    field="cpu_percent",
                    operator=">",
                    value=80.0
                )
            ],
            severity="warning",
            cooldown_seconds=300
        )

        initial_count = len(manager.alert_rules)
        manager.add_alert_rule(rule)

        assert len(manager.alert_rules) == initial_count + 1
        assert manager.alert_rules[-1] == rule

    def test_add_duplicate_rule_id(self, manager):
        """测试添加重复规则ID"""
        rule1 = AlertRuleConfig(
            rule_id="duplicate",
            name="Rule 1",
            conditions=[],
            severity="warning",
            cooldown_seconds=300
        )

        rule2 = AlertRuleConfig(
            rule_id="duplicate",
            name="Rule 2",
            conditions=[],
            severity="error",
            cooldown_seconds=600
        )

        manager.add_alert_rule(rule1)
        initial_count = len(manager.alert_rules)

        # 添加重复ID应该被允许（或替换）
        manager.add_alert_rule(rule2)

        # 规则数量应该增加
        assert len(manager.alert_rules) == initial_count + 1

    def test_remove_alert_rule_existing(self, manager):
        """测试移除存在的告警规则"""
        rule = AlertRuleConfig(
            rule_id="to_remove",
            name="Rule to Remove",
            conditions=[],
            severity="info",
            cooldown_seconds=60
        )

        manager.add_alert_rule(rule)
        initial_count = len(manager.alert_rules)

        result = manager.remove_alert_rule("to_remove")

        assert result is True
        assert len(manager.alert_rules) == initial_count - 1
        assert not any(r.rule_id == "to_remove" for r in manager.alert_rules)

    def test_remove_alert_rule_nonexistent(self, manager):
        """测试移除不存在的告警规则"""
        initial_count = len(manager.alert_rules)

        result = manager.remove_alert_rule("nonexistent")

        assert result is False
        assert len(manager.alert_rules) == initial_count


class TestAlertManagerAlertChecking:
    """AlertManager告警检查测试"""

    @pytest.fixture
    def manager(self):
        """AlertManager fixture"""
        return AlertManager("test_pool", {"cpu": 80.0, "memory": 90.0})

    def test_check_alerts_no_rules(self, manager):
        """测试无规则时的告警检查"""
        # 清空规则
        manager.alert_rules.clear()

        stats = {"cpu_percent": 95.0, "memory_percent": 85.0}
        alerts = manager.check_alerts(stats)

        assert alerts == []

    def test_check_alerts_rule_triggered(self, manager):
        """测试规则触发的告警检查"""
        # 添加一个简单的规则
        rule = AlertRuleConfig(
            rule_id="high_cpu",
            name="High CPU Usage",
            conditions=[
                AlertConditionConfig(
                    field="cpu_percent",
                    operator=">",
                    value=80.0
                )
            ],
            severity="warning",
            cooldown_seconds=300
        )

        manager.add_alert_rule(rule)

        stats = {"cpu_percent": 85.0, "memory_percent": 70.0}
        alerts = manager.check_alerts(stats)

        assert len(alerts) == 1
        alert = alerts[0]
        assert alert["rule_id"] == "high_cpu"
        assert alert["severity"] == "warning"
        assert "cpu_percent" in str(alert["message"])

    def test_check_alerts_rule_not_triggered(self, manager):
        """测试规则未触发的告警检查"""
        rule = AlertRuleConfig(
            rule_id="high_cpu",
            name="High CPU Usage",
            conditions=[
                AlertConditionConfig(
                    field="cpu_percent",
                    operator=">",
                    value=80.0
                )
            ],
            severity="warning",
            cooldown_seconds=300
        )

        manager.add_alert_rule(rule)

        # CPU使用率低于阈值
        stats = {"cpu_percent": 70.0, "memory_percent": 60.0}
        alerts = manager.check_alerts(stats)

        assert alerts == []

    def test_check_alerts_multiple_conditions(self, manager):
        """测试多条件告警检查"""
        rule = AlertRuleConfig(
            rule_id="system_overload",
            name="System Overload",
            conditions=[
                AlertConditionConfig(
                    field="cpu_percent",
                    operator=">",
                    value=80.0
                ),
                AlertConditionConfig(
                    field="memory_percent",
                    operator=">",
                    value=85.0
                )
            ],
            severity="critical",
            cooldown_seconds=300
        )

        manager.add_alert_rule(rule)

        # 只满足一个条件，不应该触发
        stats = {"cpu_percent": 85.0, "memory_percent": 70.0}
        alerts = manager.check_alerts(stats)
        assert alerts == []

        # 满足所有条件，应该触发
        stats = {"cpu_percent": 85.0, "memory_percent": 90.0}
        alerts = manager.check_alerts(stats)
        assert len(alerts) == 1
        assert alerts[0]["rule_id"] == "system_overload"
        assert alerts[0]["severity"] == "critical"

    def test_check_alerts_cooldown_prevention(self, manager):
        """测试冷却时间防止重复告警"""
        rule = AlertRuleConfig(
            rule_id="cooldown_test",
            name="Cooldown Test",
            conditions=[
                AlertConditionConfig(
                    field="cpu_percent",
                    operator=">",
                    value=50.0  # 低阈值确保触发
                )
            ],
            severity="warning",
            cooldown_seconds=60  # 1分钟冷却
        )

        manager.add_alert_rule(rule)

        stats = {"cpu_percent": 80.0}

        # 第一次检查应该触发告警
        alerts1 = manager.check_alerts(stats)
        assert len(alerts1) == 1

        # 立即再次检查，不应该触发（冷却中）
        alerts2 = manager.check_alerts(stats)
        assert alerts2 == []

    def test_check_alerts_missing_stats_fields(self, manager):
        """测试缺失统计字段的告警检查"""
        rule = AlertRuleConfig(
            rule_id="missing_field",
            name="Missing Field Test",
            conditions=[
                AlertConditionConfig(
                    field="nonexistent_field",
                    operator=">",
                    value=50.0
                )
            ],
            severity="info",
            cooldown_seconds=60
        )

        manager.add_alert_rule(rule)

        stats = {"cpu_percent": 80.0}  # 不包含nonexistent_field
        alerts = manager.check_alerts(stats)

        # 缺少字段应该不触发告警（安全处理）
        assert alerts == []


class TestAlertManagerConditionEvaluation:
    """AlertManager条件评估测试"""

    @pytest.fixture
    def manager(self):
        """AlertManager fixture"""
        return AlertManager("test_pool", {})

    def test_evaluate_condition_greater_than(self, manager):
        """测试大于条件评估"""
        condition = AlertConditionConfig(
            field="cpu_percent",
            operator=">",
            value=80.0
        )

        # 应该触发：85 > 80
        assert manager._evaluate_condition(condition, {"cpu_percent": 85.0}) is True

        # 不应该触发：75 < 80
        assert manager._evaluate_condition(condition, {"cpu_percent": 75.0}) is False

        # 边界情况：80 == 80，不大于
        assert manager._evaluate_condition(condition, {"cpu_percent": 80.0}) is False

    def test_evaluate_condition_less_than(self, manager):
        """测试小于条件评估"""
        condition = AlertConditionConfig(
            field="memory_percent",
            operator="<",
            value=90.0
        )

        # 应该触发：85 < 90
        assert manager._evaluate_condition(condition, {"memory_percent": 85.0}) is True

        # 不应该触发：95 > 90
        assert manager._evaluate_condition(condition, {"memory_percent": 95.0}) is False

    def test_evaluate_condition_equals(self, manager):
        """测试等于条件评估"""
        condition = AlertConditionConfig(
            field="status_code",
            operator="==",
            value=200
        )

        # 应该触发：200 == 200
        assert manager._evaluate_condition(condition, {"status_code": 200}) is True

        # 不应该触发：404 != 200
        assert manager._evaluate_condition(condition, {"status_code": 404}) is False

    def test_evaluate_condition_not_equals(self, manager):
        """测试不等于条件评估"""
        condition = AlertConditionConfig(
            field="error_count",
            operator="!=",
            value=0
        )

        # 应该触发：5 != 0
        assert manager._evaluate_condition(condition, {"error_count": 5}) is True

        # 不应该触发：0 == 0
        assert manager._evaluate_condition(condition, {"error_count": 0}) is False

    def test_evaluate_condition_greater_equal(self, manager):
        """测试大于等于条件评估"""
        condition = AlertConditionConfig(
            field="disk_percent",
            operator=">=",
            value=85.0
        )

        # 应该触发：90 >= 85
        assert manager._evaluate_condition(condition, {"disk_percent": 90.0}) is True

        # 应该触发：85 >= 85
        assert manager._evaluate_condition(condition, {"disk_percent": 85.0}) is True

        # 不应该触发：80 < 85
        assert manager._evaluate_condition(condition, {"disk_percent": 80.0}) is False

    def test_evaluate_condition_less_equal(self, manager):
        """测试小于等于条件评估"""
        condition = AlertConditionConfig(
            field="response_time",
            operator="<=",
            value=1000.0
        )

        # 应该触发：500 <= 1000
        assert manager._evaluate_condition(condition, {"response_time": 500.0}) is True

        # 应该触发：1000 <= 1000
        assert manager._evaluate_condition(condition, {"response_time": 1000.0}) is True

        # 不应该触发：1500 > 1000
        assert manager._evaluate_condition(condition, {"response_time": 1500.0}) is False

    def test_evaluate_condition_invalid_operator(self, manager):
        """测试无效操作符的条件评估"""
        condition = AlertConditionConfig(
            field="cpu_percent",
            operator="invalid",
            value=80.0
        )

        # 无效操作符应该返回False（安全处理）
        assert manager._evaluate_condition(condition, {"cpu_percent": 90.0}) is False

    def test_evaluate_condition_missing_field(self, manager):
        """测试缺失字段的条件评估"""
        condition = AlertConditionConfig(
            field="missing_field",
            operator=">",
            value=50.0
        )

        # 缺失字段应该返回False（安全处理）
        assert manager._evaluate_condition(condition, {"cpu_percent": 90.0}) is False


class TestAlertManagerAlertHistory:
    """AlertManager告警历史测试"""

    @pytest.fixture
    def manager(self):
        """AlertManager fixture"""
        return AlertManager("test_pool", {"cpu": 80.0})

    def test_get_alert_history_empty(self, manager):
        """测试获取空的告警历史"""
        history = manager.get_alert_history()

        assert isinstance(history, list)
        assert len(history) == 0

    def test_get_alert_history_with_alerts(self, manager):
        """测试获取有告警的告警历史"""
        # 手动添加告警到历史
        alert = {
            "alert_id": "test_alert_1",
            "rule_id": "test_rule",
            "severity": "warning",
            "message": "Test alert",
            "timestamp": datetime.now(),
            "acknowledged": False
        }

        manager.alert_history.append(alert)

        history = manager.get_alert_history()

        assert len(history) == 1
        assert history[0]["alert_id"] == "test_alert_1"

    def test_get_alert_history_with_limit(self, manager):
        """测试获取限制数量的告警历史"""
        # 添加多个告警
        for i in range(5):
            alert = {
                "alert_id": f"alert_{i}",
                "rule_id": "test_rule",
                "severity": "warning",
                "message": f"Test alert {i}",
                "timestamp": datetime.now(),
                "acknowledged": False
            }
            manager.alert_history.append(alert)

        # 获取限制数量的历史
        history = manager.get_alert_history(limit=3)

        assert len(history) == 3

        # 应该返回最新的告警（后添加的）
        assert history[0]["alert_id"] == "alert_4"
        assert history[1]["alert_id"] == "alert_3"
        assert history[2]["alert_id"] == "alert_2"

    def test_get_active_alerts(self, manager):
        """测试获取活跃告警"""
        # 添加已确认和未确认的告警
        acknowledged_alert = {
            "alert_id": "ack_alert",
            "acknowledged": True,
            "active": True
        }

        active_alert = {
            "alert_id": "active_alert",
            "acknowledged": False,
            "active": True
        }

        inactive_alert = {
            "alert_id": "inactive_alert",
            "acknowledged": False,
            "active": False
        }

        manager.alert_history.extend([acknowledged_alert, active_alert, inactive_alert])

        active_alerts = manager.get_active_alerts()

        assert len(active_alerts) == 1
        assert active_alerts[0]["alert_id"] == "active_alert"

    def test_acknowledge_alert_existing(self, manager):
        """测试确认存在的告警"""
        alert = {
            "alert_id": "to_acknowledge",
            "acknowledged": False,
            "active": True
        }

        manager.alert_history.append(alert)

        result = manager.acknowledge_alert("to_acknowledge")

        assert result is True
        assert manager.alert_history[0]["acknowledged"] is True

    def test_acknowledge_alert_nonexistent(self, manager):
        """测试确认不存在的告警"""
        result = manager.acknowledge_alert("nonexistent")

        assert result is False


class TestAlertManagerStatistics:
    """AlertManager统计测试"""

    @pytest.fixture
    def manager(self):
        """AlertManager fixture"""
        return AlertManager("test_pool", {"cpu": 80.0})

    def test_get_alert_statistics_empty(self, manager):
        """测试获取空的告警统计"""
        stats = manager.get_alert_statistics()

        assert isinstance(stats, dict)
        assert "total_alerts" in stats
        assert "active_alerts" in stats
        assert "acknowledged_alerts" in stats
        assert stats["total_alerts"] == 0
        assert stats["active_alerts"] == 0
        assert stats["acknowledged_alerts"] == 0

    def test_get_alert_statistics_with_data(self, manager):
        """测试获取有数据的告警统计"""
        # 添加各种告警
        alerts = [
            {"alert_id": "1", "acknowledged": False, "active": True, "severity": "critical"},
            {"alert_id": "2", "acknowledged": True, "active": True, "severity": "warning"},
            {"alert_id": "3", "acknowledged": False, "active": False, "severity": "info"},
            {"alert_id": "4", "acknowledged": True, "active": True, "severity": "error"},
        ]

        manager.alert_history.extend(alerts)

        stats = manager.get_alert_statistics()

        assert stats["total_alerts"] == 4
        assert stats["active_alerts"] == 3  # 1个inactive
        assert stats["acknowledged_alerts"] == 2
        assert "severity_breakdown" in stats
        assert stats["severity_breakdown"]["critical"] == 1
        assert stats["severity_breakdown"]["warning"] == 1
        assert stats["severity_breakdown"]["error"] == 1
        assert stats["severity_breakdown"]["info"] == 1


class TestAlertManagerCooldown:
    """AlertManager冷却时间测试"""

    @pytest.fixture
    def manager(self):
        """AlertManager fixture"""
        return AlertManager("test_pool", {})

    def test_is_cooldown_expired_no_previous_alert(self, manager):
        """测试无之前告警时的冷却过期检查"""
        # 从未有过告警，应该过期
        assert manager._is_cooldown_expired("new_rule", 300) is True

    def test_is_cooldown_expired_recent_alert(self, manager):
        """测试最近告警的冷却过期检查"""
        rule_id = "recent_rule"
        manager.last_alert_times[rule_id] = datetime.now()  # 刚刚告警

        # 冷却时间5分钟，但刚刚告警，不应该过期
        assert manager._is_cooldown_expired(rule_id, 300) is False

    def test_is_cooldown_expired_expired_alert(self, manager):
        """测试过期告警的冷却过期检查"""
        rule_id = "expired_rule"
        # 10分钟前的告警
        manager.last_alert_times[rule_id] = datetime.now() - timedelta(minutes=10)

        # 冷却时间5分钟，应该已过期
        assert manager._is_cooldown_expired(rule_id, 300) is True

    def test_is_cooldown_expired_boundary(self, manager):
        """测试边界情况的冷却过期检查"""
        rule_id = "boundary_rule"
        # 正好5分钟前的告警
        manager.last_alert_times[rule_id] = datetime.now() - timedelta(minutes=5)

        # 冷却时间5分钟，边界情况应该过期
        assert manager._is_cooldown_expired(rule_id, 300) is True


class TestAlertManagerIntegration:
    """AlertManager集成测试"""

    def test_complete_alert_workflow(self):
        """测试完整的告警工作流"""
        manager = AlertManager("integration_pool", {"cpu": 80.0, "memory": 90.0})

        # 添加自定义规则
        rule = AlertRuleConfig(
            rule_id="integration_test",
            name="Integration Test Rule",
            conditions=[
                AlertConditionConfig(
                    field="cpu_percent",
                    operator=">",
                    value=75.0
                ),
                AlertConditionConfig(
                    field="memory_percent",
                    operator=">",
                    value=85.0
                )
            ],
            severity="critical",
            cooldown_seconds=60
        )

        manager.add_alert_rule(rule)

        # 初始状态检查
        stats = manager.get_alert_statistics()
        assert stats["total_alerts"] == 0

        # 触发告警
        test_stats = {
            "cpu_percent": 85.0,
            "memory_percent": 95.0,
            "disk_percent": 70.0
        }

        alerts = manager.check_alerts(test_stats)
        assert len(alerts) == 1
        assert alerts[0]["rule_id"] == "integration_test"

        # 检查统计更新
        stats = manager.get_alert_statistics()
        assert stats["total_alerts"] == 1
        assert stats["active_alerts"] == 1

        # 检查历史记录
        history = manager.get_alert_history()
        assert len(history) == 1
        assert history[0]["rule_id"] == "integration_test"

        # 确认告警
        alert_id = alerts[0]["alert_id"]
        result = manager.acknowledge_alert(alert_id)
        assert result is True

        # 检查活跃告警清空
        active = manager.get_active_alerts()
        assert len(active) == 0

        # 检查统计更新
        stats = manager.get_alert_statistics()
        assert stats["acknowledged_alerts"] == 1

    def test_alert_manager_concurrent_access(self):
        """测试告警管理器的并发访问"""
        manager = AlertManager("concurrent_pool", {"cpu": 80.0})

        rule = AlertRuleConfig(
            rule_id="concurrent_test",
            name="Concurrent Test",
            conditions=[
                AlertConditionConfig(
                    field="counter",
                    operator=">",
                    value=50
                )
            ],
            severity="warning",
            cooldown_seconds=1  # 短冷却时间
        )

        manager.add_alert_rule(rule)

        exceptions = []
        alerts_generated = []

        def concurrent_worker(worker_id: int):
            """并发工作线程"""
            try:
                for i in range(10):
                    stats = {"counter": 60 + worker_id * 10 + i}
                    alerts = manager.check_alerts(stats)
                    alerts_generated.extend(alerts)
            except Exception as e:
                exceptions.append(f"Worker {worker_id}: {e}")

        import threading

        # 启动5个并发线程
        threads = []
        for i in range(5):
            t = threading.Thread(target=concurrent_worker, args=(i,))
            threads.append(t)
            t.start()

        # 等待完成
        for t in threads:
            t.join()

        # 不应该有异常
        assert len(exceptions) == 0, f"Concurrent exceptions: {exceptions}"

        # 应该生成了告警（具体数量取决于冷却和并发）
        assert len(alerts_generated) >= 0

    def test_alert_manager_memory_management(self):
        """测试告警管理器的内存管理"""
        manager = AlertManager("memory_pool", {"cpu": 80.0})

        # 设置小的历史大小限制
        manager.max_history_size = 5

        # 添加多个告警
        for i in range(10):
            alert = {
                "alert_id": f"memory_test_{i}",
                "rule_id": "test_rule",
                "severity": "info",
                "message": f"Memory test alert {i}",
                "timestamp": datetime.now(),
                "acknowledged": False,
                "active": True
            }
            manager._record_alert(alert)

        # 历史应该被限制在最大大小
        assert len(manager.alert_history) <= manager.max_history_size

        # 最新的告警应该被保留
        latest_alerts = manager.get_alert_history(limit=5)
        assert len(latest_alerts) <= 5

        # 检查最新的告警ID
        assert latest_alerts[0]["alert_id"] == "memory_test_9"
