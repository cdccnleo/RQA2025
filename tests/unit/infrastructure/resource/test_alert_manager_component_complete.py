"""
测试告警管理组件 - 完整版本

验证AlertManager类的完整功能，包括规则管理、条件评估、告警触发等
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
import threading
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from src.infrastructure.resource.core.alert_manager_component import AlertManager
from src.infrastructure.resource.models.alert_dataclasses import AlertRule, Alert, PerformanceMetrics, TestExecutionInfo
from src.infrastructure.resource.models.alert_enums import AlertType, AlertLevel


class TestAlertManagerComplete:
    """完整测试AlertManager类"""

    def test_alert_manager_initialization(self):
        """测试告警管理器初始化"""
        manager = AlertManager()

        assert len(manager.alert_rules) == 0
        assert len(manager.active_alerts) == 0
        assert isinstance(manager.alert_handlers, dict)
        assert isinstance(manager.logger, object)  # StandardLogger
        assert isinstance(manager.error_handler, object)  # BaseErrorHandler

    def test_alert_manager_with_config(self):
        """测试带配置的告警管理器初始化"""
        config = {'test_setting': 'value'}
        manager = AlertManager(config=config)

        assert len(manager.alert_rules) == 0

    def test_add_alert_rule_new(self):
        """测试添加新告警规则"""
        manager = AlertManager()
        rule = AlertRule(
            name="CPU高负载",
            alert_type=AlertType.SYSTEM_ERROR,
            alert_level=AlertLevel.WARNING,
            condition="cpu_usage > threshold",
            threshold=80.0
        )

        with patch.object(manager.logger, 'log_info') as mock_log:
            manager.add_alert_rule(rule)

            assert len(manager.alert_rules) == 1
            assert manager.alert_rules[0] == rule
            mock_log.assert_called_with("添加告警规则: CPU高负载")

    def test_add_alert_rule_replace_existing(self):
        """测试替换现有告警规则"""
        manager = AlertManager()

        # 添加第一个规则
        rule1 = AlertRule(
            name="测试规则",
            alert_type=AlertType.SYSTEM_ERROR,
            alert_level=AlertLevel.WARNING,
            condition="cpu_usage > threshold",
            threshold=80.0
        )
        manager.add_alert_rule(rule1)

        # 添加同名规则，应该替换
        rule2 = AlertRule(
            name="测试规则",
            alert_type=AlertType.SYSTEM_ERROR,
            alert_level=AlertLevel.ERROR,
            condition="cpu_usage > threshold",
            threshold=90.0
        )

        with patch.object(manager.logger, 'warning') as mock_warning, \
             patch.object(manager.logger, 'log_info') as mock_log:
            manager.add_alert_rule(rule2)

            assert len(manager.alert_rules) == 1
            assert manager.alert_rules[0].alert_level == AlertLevel.ERROR
            assert manager.alert_rules[0].threshold == 90.0
            mock_warning.assert_called_with("告警规则 '测试规则' 已存在，将被替换")
            mock_log.assert_called_with("添加告警规则: 测试规则")

    def test_remove_alert_rule_existing(self):
        """测试移除现有告警规则"""
        manager = AlertManager()
        rule = AlertRule(
            name="测试规则",
            alert_type=AlertType.SYSTEM_ERROR,
            alert_level=AlertLevel.WARNING,
            condition="cpu_usage > threshold",
            threshold=80.0
        )
        manager.add_alert_rule(rule)

        with patch.object(manager.logger, 'log_info') as mock_log:
            manager.remove_alert_rule("测试规则")

            assert len(manager.alert_rules) == 0
            mock_log.assert_called_with("移除告警规则: 测试规则")

    def test_remove_alert_rule_nonexistent(self):
        """测试移除不存在的告警规则"""
        manager = AlertManager()

        # 不应该抛出异常
        manager.remove_alert_rule("不存在的规则")
        assert len(manager.alert_rules) == 0

    def test_register_alert_handler(self):
        """测试注册告警处理器"""
        manager = AlertManager()
        handler = MagicMock()

        with patch.object(manager.logger, 'log_info') as mock_log:
            manager.register_alert_handler(AlertType.SYSTEM_ERROR, handler)

            assert AlertType.SYSTEM_ERROR in manager.alert_handlers
            assert handler in manager.alert_handlers[AlertType.SYSTEM_ERROR]
            mock_log.assert_called_with(f"注册告警处理器: {AlertType.SYSTEM_ERROR.value}")

    def test_evaluate_condition_cpu_usage(self):
        """测试CPU使用率条件评估"""
        manager = AlertManager()
        rule = AlertRule(
            name="CPU告警",
            alert_type=AlertType.SYSTEM_ERROR,
            alert_level=AlertLevel.WARNING,
            condition="cpu_usage > threshold",
            threshold=80.0
        )

        # 超过阈值
        metrics_high = PerformanceMetrics(cpu_usage=85.0)
        result = manager._evaluate_condition(rule, metrics_high, None)
        assert result is True

        # 未超过阈值
        metrics_low = PerformanceMetrics(cpu_usage=70.0)
        result = manager._evaluate_condition(rule, metrics_low, None)
        assert result is False

    def test_evaluate_condition_memory_usage(self):
        """测试内存使用率条件评估"""
        manager = AlertManager()
        rule = AlertRule(
            name="内存告警",
            alert_type=AlertType.RESOURCE_EXHAUSTION,
            alert_level=AlertLevel.WARNING,
            condition="memory_usage > threshold",
            threshold=85.0
        )

        metrics = PerformanceMetrics(memory_usage=90.0)
        result = manager._evaluate_condition(rule, metrics, None)
        assert result is True

    def test_evaluate_condition_disk_usage(self):
        """测试磁盘使用率条件评估"""
        manager = AlertManager()
        rule = AlertRule(
            name="磁盘告警",
            alert_type=AlertType.RESOURCE_EXHAUSTION,
            alert_level=AlertLevel.WARNING,
            condition="disk_usage > threshold",
            threshold=90.0
        )

        metrics = PerformanceMetrics(disk_usage=95.0)
        result = manager._evaluate_condition(rule, metrics, None)
        assert result is True

    def test_evaluate_condition_network_latency(self):
        """测试网络延迟条件评估"""
        manager = AlertManager()
        rule = AlertRule(
            name="网络告警",
            alert_type=AlertType.NETWORK_ISSUE,
            alert_level=AlertLevel.WARNING,
            condition="network_latency > threshold",
            threshold=100.0
        )

        metrics = PerformanceMetrics(network_latency=150.0)
        result = manager._evaluate_condition(rule, metrics, None)
        assert result is True

    def test_evaluate_condition_test_execution_time(self):
        """测试测试执行时间条件评估"""
        manager = AlertManager()
        rule = AlertRule(
            name="超时告警",
            alert_type=AlertType.TEST_TIMEOUT,
            alert_level=AlertLevel.ERROR,
            condition="test_execution_time > threshold",
            threshold=300.0
        )

        test_info = TestExecutionInfo(
            test_id="test_001",
            test_name="慢测试",
            start_time=datetime.now(),
            execution_time=350.0
        )

        result = manager._evaluate_condition(rule, PerformanceMetrics(), test_info)
        assert result is True

    def test_evaluate_condition_test_success_rate(self):
        """测试测试成功率条件评估"""
        manager = AlertManager()
        rule = AlertRule(
            name="成功率告警",
            alert_type=AlertType.TEST_FAILURE,
            alert_level=AlertLevel.ERROR,
            condition="test_success_rate < threshold",
            threshold=90.0
        )

        metrics = PerformanceMetrics(test_success_rate=85.0)
        result = manager._evaluate_condition(rule, metrics, None)
        assert result is True

    def test_evaluate_condition_invalid(self):
        """测试无效条件评估"""
        manager = AlertManager()
        rule = AlertRule(
            name="无效规则",
            alert_type=AlertType.SYSTEM_ERROR,
            alert_level=AlertLevel.WARNING,
            condition="invalid_condition",
            threshold=50.0
        )

        result = manager._evaluate_condition(rule, PerformanceMetrics(), None)
        assert result is False

    def test_evaluate_condition_exception_handling(self):
        """测试条件评估异常处理"""
        manager = AlertManager()
        rule = AlertRule(
            name="异常规则",
            alert_type=AlertType.SYSTEM_ERROR,
            alert_level=AlertLevel.WARNING,
            condition="cpu_usage > threshold",
            threshold=80.0
        )

        # 模拟异常
        with patch.object(manager.error_handler, 'handle_error') as mock_error:
            # 让metrics访问抛出异常
            faulty_metrics = MagicMock()
            faulty_metrics.cpu_usage = property(lambda self: (_ for _ in ()).throw(Exception("Access error")))

            result = manager._evaluate_condition(rule, faulty_metrics, None)
            assert result is False
            mock_error.assert_called_once()

    def test_check_alerts_no_rules(self):
        """测试检查告警（无规则）"""
        manager = AlertManager()
        metrics = PerformanceMetrics(cpu_usage=90.0)

        # 不应该抛出异常
        manager.check_alerts(metrics)

    def test_check_alerts_disabled_rule(self):
        """测试检查告警（禁用规则）"""
        manager = AlertManager()
        rule = AlertRule(
            name="禁用规则",
            alert_type=AlertType.SYSTEM_ERROR,
            alert_level=AlertLevel.WARNING,
            condition="cpu_usage > threshold",
            threshold=80.0,
            enabled=False
        )
        manager.add_alert_rule(rule)

        metrics = PerformanceMetrics(cpu_usage=90.0)
        manager.check_alerts(metrics)

        # 不应该触发告警
        assert len(manager.active_alerts) == 0

    def test_check_alerts_cooldown_active(self):
        """测试检查告警（冷却期内）"""
        manager = AlertManager()
        rule = AlertRule(
            name="冷却测试",
            alert_type=AlertType.SYSTEM_ERROR,
            alert_level=AlertLevel.WARNING,
            condition="cpu_usage > threshold",
            threshold=80.0,
            cooldown=300  # 5分钟冷却
        )
        rule.last_triggered = datetime.now()  # 刚刚触发
        manager.add_alert_rule(rule)

        metrics = PerformanceMetrics(cpu_usage=90.0)
        manager.check_alerts(metrics)

        # 不应该触发告警（冷却期内）
        assert len(manager.active_alerts) == 0

    def test_check_alerts_trigger_success(self):
        """测试检查告警（成功触发）"""
        manager = AlertManager()
        rule = AlertRule(
            name="CPU告警",
            alert_type=AlertType.SYSTEM_ERROR,
            alert_level=AlertLevel.WARNING,
            condition="cpu_usage > threshold",
            threshold=80.0
        )
        manager.add_alert_rule(rule)

        metrics = PerformanceMetrics(cpu_usage=85.0)

        with patch.object(manager, '_trigger_alert_with_params_unsafe') as mock_trigger:
            manager.check_alerts(metrics)

            # 验证方法被调用，但参数结构可能不同（使用了AlertCheckParameters）
            assert mock_trigger.called

    def test_trigger_alert_complete_flow(self):
        """测试告警触发完整流程"""
        manager = AlertManager()
        rule = AlertRule(
            name="测试告警",
            alert_type=AlertType.SYSTEM_ERROR,
            alert_level=AlertLevel.WARNING,
            condition="cpu_usage > threshold",
            threshold=80.0
        )

        metrics = PerformanceMetrics(
            cpu_usage=85.0,
            memory_usage=70.0,
            disk_usage=60.0,
            network_latency=25.0
        )

        # 注册处理器
        handler = MagicMock()
        manager.register_alert_handler(AlertType.SYSTEM_ERROR, handler)

        with patch('src.infrastructure.resource.core.alert_manager_component.time') as mock_time, \
             patch.object(manager.logger, 'warning') as mock_warning:

            mock_time.time.return_value = 1234567890

            manager._trigger_alert(rule, metrics, None)

            # 验证告警已创建
            assert len(manager.active_alerts) == 1
            alert_id = "system_error_1234567890"
            assert alert_id in manager.active_alerts

            alert = manager.active_alerts[alert_id]
            assert alert.alert_type == AlertType.SYSTEM_ERROR
            assert alert.alert_level == AlertLevel.WARNING
            assert alert.message == "触发告警规则: 测试告警"
            assert alert.source == "performance_monitor"
            assert alert.details['rule_name'] == '测试告警'
            assert alert.details['current_value'] == 85.0

            # 验证规则的last_triggered已更新
            assert rule.last_triggered is not None

            # 验证日志记录
            mock_warning.assert_called_once()

            # 等待处理器线程执行（带超时）
            import time
            timeout = 0.5  # 最多等待0.5秒
            start_time = time.time()
            while time.time() - start_time < timeout:
                if handler.called:
                    break
                time.sleep(0.01)  # 短暂等待

            # 验证处理器被调用
            handler.assert_called_once_with(alert)

    def test_trigger_alert_handler_exception(self):
        """测试告警触发处理器异常处理"""
        manager = AlertManager()
        rule = AlertRule(
            name="异常测试",
            alert_type=AlertType.SYSTEM_ERROR,
            alert_level=AlertLevel.WARNING,
            condition="cpu_usage > threshold",
            threshold=80.0
        )

        # 注册抛出异常的处理器
        handler = MagicMock(side_effect=Exception("Handler failed"))
        manager.register_alert_handler(AlertType.SYSTEM_ERROR, handler)

        metrics = PerformanceMetrics(cpu_usage=85.0)

        with patch.object(manager.error_handler, 'handle_error') as mock_error:
            manager._trigger_alert(rule, metrics, None)

            # 等待处理器线程执行
            time.sleep(0.1)

            # 验证错误被处理
            mock_error.assert_called_once()

    def test_get_current_value_cpu_usage(self):
        """测试获取当前值（CPU使用率）"""
        manager = AlertManager()
        rule = AlertRule(
            name="CPU规则",
            alert_type=AlertType.SYSTEM_ERROR,
            alert_level=AlertLevel.WARNING,
            condition="cpu_usage > threshold",
            threshold=80.0
        )

        metrics = PerformanceMetrics(cpu_usage=85.5)
        value = manager._get_current_value(rule, metrics, None)

        assert value == 85.5

    def test_get_current_value_memory_usage(self):
        """测试获取当前值（内存使用率）"""
        manager = AlertManager()
        rule = AlertRule(
            name="内存规则",
            alert_type=AlertType.RESOURCE_EXHAUSTION,
            alert_level=AlertLevel.WARNING,
            condition="memory_usage > threshold",
            threshold=85.0
        )

        metrics = PerformanceMetrics(memory_usage=90.2)
        value = manager._get_current_value(rule, metrics, None)

        assert value == 90.2

    def test_get_current_value_test_execution_time(self):
        """测试获取当前值（测试执行时间）"""
        manager = AlertManager()
        rule = AlertRule(
            name="超时规则",
            alert_type=AlertType.TEST_TIMEOUT,
            alert_level=AlertLevel.ERROR,
            condition="test_execution_time > threshold",
            threshold=300.0
        )

        test_info = TestExecutionInfo(
            test_id="test_001",
            test_name="慢测试",
            start_time=datetime.now(),
            execution_time=350.5
        )

        metrics = PerformanceMetrics()
        value = manager._get_current_value(rule, metrics, test_info)

        assert value == 350.5

    def test_get_current_value_unknown_condition(self):
        """测试获取当前值（未知条件）"""
        manager = AlertManager()
        rule = AlertRule(
            name="未知规则",
            alert_type=AlertType.SYSTEM_ERROR,
            alert_level=AlertLevel.WARNING,
            condition="unknown_condition",
            threshold=50.0
        )

        metrics = PerformanceMetrics()
        value = manager._get_current_value(rule, metrics, None)

        assert value == 0.0

    def test_concurrent_alert_operations(self):
        """测试并发告警操作"""
        manager = AlertManager()

        # 添加规则
        rule = AlertRule(
            name="并发测试",
            alert_type=AlertType.SYSTEM_ERROR,
            alert_level=AlertLevel.WARNING,
            condition="cpu_usage > threshold",
            threshold=80.0
        )
        manager.add_alert_rule(rule)

        results = []
        errors = []

        def concurrent_check(operation_id):
            try:
                metrics = PerformanceMetrics(cpu_usage=85.0 + operation_id)
                manager.check_alerts(metrics)

                results.append(f"operation_{operation_id}")
            except Exception as e:
                errors.append(e)

        # 并发执行多个告警检查
        threads = []
        for i in range(10):
            thread = threading.Thread(target=concurrent_check, args=(i,))
            threads.append(thread)

        for thread in threads:
            thread.start()

        # 使用超时机制等待线程完成，避免无限等待
        for thread in threads:
            thread.join(timeout=2.0)  # 最多等待2秒
            if thread.is_alive():
                # 如果线程仍然存活，记录错误但不无限等待
                errors.append(TimeoutError(f"Thread {thread.ident} did not complete within timeout"))

        # 验证没有出现异常
        assert len(errors) == 0
        assert len(results) == 10

        # 应该有多个告警被触发（具体数量可能因线程执行顺序而异）
        assert len(manager.active_alerts) >= 1

    def test_alert_manager_cleanup(self):
        """测试告警管理器清理"""
        manager = AlertManager()

        # 添加规则和告警
        rule = AlertRule(
            name="清理测试",
            alert_type=AlertType.SYSTEM_ERROR,
            alert_level=AlertLevel.WARNING,
            condition="cpu_usage > threshold",
            threshold=80.0
        )
        manager.add_alert_rule(rule)

        # 手动添加活跃告警
        alert = Alert(
            id="test_alert",
            alert_type=AlertType.SYSTEM_ERROR,
            alert_level=AlertLevel.WARNING,
            message="测试告警",
            details={},
            timestamp=datetime.now(),
            source="test"
        )
        manager.active_alerts["test_alert"] = alert

        # 清理
        manager.cleanup()

        assert len(manager.alert_rules) == 0
        assert len(manager.active_alerts) == 0
        assert len(manager.alert_handlers) == 0  # defaultdict会重置为空
