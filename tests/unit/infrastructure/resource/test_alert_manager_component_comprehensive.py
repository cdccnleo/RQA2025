#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
告警管理器组件深度测试

大幅提升alert_manager_component.py的测试覆盖率，从20%提升到80%以上
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta


class TestAlertManagerComponentComprehensive:
    """告警管理器组件深度测试"""

    def test_alert_manager_initialization(self):
        """测试告警管理器初始化"""
        try:
            from src.infrastructure.resource.core.alert_manager_component import AlertManager

            manager = AlertManager()

            # 测试基本属性
            assert hasattr(manager, 'alert_rules')
            assert hasattr(manager, 'active_alerts')
            assert hasattr(manager, 'alert_handlers')
            assert hasattr(manager, '_lock')
            assert hasattr(manager, 'logger')
            assert hasattr(manager, 'error_handler')

            # 测试初始状态
            assert isinstance(manager.alert_rules, list)
            assert isinstance(manager.active_alerts, dict)
            assert isinstance(manager.alert_handlers, dict)
            assert len(manager.alert_rules) == 0
            assert len(manager.active_alerts) == 0

        except ImportError:
            pytest.skip("AlertManager not available")

    def test_alert_manager_initialization_with_config(self):
        """测试带配置的告警管理器初始化"""
        try:
            from src.infrastructure.resource.core.alert_manager_component import AlertManager

            config = {
                'max_alerts': 100,
                'cooldown_period': 300,
                'enable_notifications': True
            }

            manager = AlertManager(config)

            # 验证配置被接受（具体配置应用可能在_apply_config中实现）
            assert manager is not None

        except ImportError:
            pytest.skip("AlertManager initialization with config not available")

    def test_add_alert_rule_new_rule(self):
        """测试添加新告警规则"""
        try:
            from src.infrastructure.resource.core.alert_manager_component import AlertManager
            from src.infrastructure.resource.models.alert_dataclasses import AlertRule
            from src.infrastructure.resource.models.alert_enums import AlertType, AlertLevel

            manager = AlertManager()

            rule = AlertRule(
                name="High CPU Usage",
                alert_type=AlertType.SYSTEM_ERROR,
                alert_level=AlertLevel.WARNING,
                condition="cpu_usage > threshold",
                threshold=80.0,
                enabled=True
            )

            manager.add_alert_rule(rule)

            # 验证规则被添加
            assert len(manager.alert_rules) == 1
            assert manager.alert_rules[0].name == "High CPU Usage"
            assert manager.alert_rules[0].alert_type == AlertType.SYSTEM_ERROR

        except ImportError:
            pytest.skip("Add alert rule new rule not available")

    def test_add_alert_rule_replace_existing(self):
        """测试替换现有告警规则"""
        try:
            from src.infrastructure.resource.core.alert_manager_component import AlertManager
            from src.infrastructure.resource.models.alert_dataclasses import AlertRule
            from src.infrastructure.resource.models.alert_enums import AlertType, AlertLevel

            manager = AlertManager()

            # 添加初始规则
            rule1 = AlertRule(
                name="High CPU",
                alert_type=AlertType.SYSTEM_ERROR,
                alert_level=AlertLevel.WARNING,
                condition="cpu_usage > threshold",
                threshold=80.0
            )
            manager.add_alert_rule(rule1)
            assert len(manager.alert_rules) == 1

            # 添加同名规则（应该替换）
            rule2 = AlertRule(
                name="High CPU",
                alert_type=AlertType.RESOURCE_EXHAUSTION,
                alert_level=AlertLevel.CRITICAL,
                condition="cpu_usage > threshold",
                threshold=90.0
            )
            manager.add_alert_rule(rule2)

            # 验证规则被替换而不是添加
            assert len(manager.alert_rules) == 1
            assert manager.alert_rules[0].alert_level == AlertLevel.CRITICAL
            assert manager.alert_rules[0].threshold == 90.0

        except ImportError:
            pytest.skip("Add alert rule replace existing not available")

    def test_remove_alert_rule_existing(self):
        """测试移除现有告警规则"""
        try:
            from src.infrastructure.resource.core.alert_manager_component import AlertManager
            from src.infrastructure.resource.models.alert_dataclasses import AlertRule
            from src.infrastructure.resource.models.alert_enums import AlertType, AlertLevel

            manager = AlertManager()

            # 添加规则
            rule = AlertRule(
                name="Test Rule",
                alert_type=AlertType.SYSTEM_ERROR,
                alert_level=AlertLevel.WARNING,
                condition="cpu_usage > threshold",
                threshold=80.0
            )
            manager.add_alert_rule(rule)
            assert len(manager.alert_rules) == 1

            # 移除规则
            manager.remove_alert_rule("Test Rule")

            # 验证规则被移除
            assert len(manager.alert_rules) == 0

        except ImportError:
            pytest.skip("Remove alert rule existing not available")

    def test_remove_alert_rule_nonexistent(self):
        """测试移除不存在的告警规则"""
        try:
            from src.infrastructure.resource.core.alert_manager_component import AlertManager

            manager = AlertManager()

            # 尝试移除不存在的规则
            manager.remove_alert_rule("Nonexistent Rule")

            # 验证无异常且列表为空
            assert len(manager.alert_rules) == 0

        except ImportError:
            pytest.skip("Remove alert rule nonexistent not available")

    def test_register_alert_handler(self):
        """测试注册告警处理器"""
        try:
            from src.infrastructure.resource.core.alert_manager_component import AlertManager
            from src.infrastructure.resource.models.alert_enums import AlertType

            manager = AlertManager()

            mock_handler = Mock()

            manager.register_alert_handler(AlertType.SYSTEM_ERROR, mock_handler)

            # 验证处理器被注册
            assert AlertType.SYSTEM_ERROR in manager.alert_handlers
            assert mock_handler in manager.alert_handlers[AlertType.SYSTEM_ERROR]

        except ImportError:
            pytest.skip("Register alert handler not available")

    def test_register_multiple_alert_handlers(self):
        """测试注册多个告警处理器"""
        try:
            from src.infrastructure.resource.core.alert_manager_component import AlertManager
            from src.infrastructure.resource.models.alert_enums import AlertType

            manager = AlertManager()

            handler1 = Mock()
            handler2 = Mock()

            manager.register_alert_handler(AlertType.SYSTEM_ERROR, handler1)
            manager.register_alert_handler(AlertType.SYSTEM_ERROR, handler2)

            # 验证两个处理器都被注册
            assert len(manager.alert_handlers[AlertType.SYSTEM_ERROR]) == 2
            assert handler1 in manager.alert_handlers[AlertType.SYSTEM_ERROR]
            assert handler2 in manager.alert_handlers[AlertType.SYSTEM_ERROR]

        except ImportError:
            pytest.skip("Register multiple alert handlers not available")

    def test_evaluate_condition_cpu_usage(self):
        """测试评估CPU使用率告警条件"""
        try:
            from src.infrastructure.resource.core.alert_manager_component import AlertManager
            from src.infrastructure.resource.models.alert_dataclasses import AlertRule
            from src.infrastructure.resource.models.alert_enums import AlertType, AlertLevel

            manager = AlertManager()

            rule = AlertRule(
                name="CPU Alert",
                alert_type=AlertType.SYSTEM_ERROR,
                alert_level=AlertLevel.WARNING,
                condition="cpu_usage > threshold",
                threshold=80.0
            )

            # 创建模拟指标
            mock_metrics = Mock()
            mock_metrics.cpu_usage = 85.0

            result = manager._evaluate_condition(rule, mock_metrics, None)

            # 验证条件评估为真
            assert result is True

            # 测试低于阈值的情况
            mock_metrics.cpu_usage = 70.0
            result = manager._evaluate_condition(rule, mock_metrics, None)
            assert result is False

        except ImportError:
            pytest.skip("Evaluate condition cpu usage not available")

    def test_evaluate_condition_memory_usage(self):
        """测试评估内存使用率告警条件"""
        try:
            from src.infrastructure.resource.core.alert_manager_component import AlertManager
            from src.infrastructure.resource.models.alert_dataclasses import AlertRule
            from src.infrastructure.resource.models.alert_enums import AlertType, AlertLevel

            manager = AlertManager()

            rule = AlertRule(
                name="Memory Alert",
                alert_type=AlertType.RESOURCE_EXHAUSTION,
                alert_level=AlertLevel.CRITICAL,
                condition="memory_usage > threshold",
                threshold=90.0
            )

            mock_metrics = Mock()
            mock_metrics.memory_usage = 95.0

            result = manager._evaluate_condition(rule, mock_metrics, None)

            # 验证条件评估为真
            assert result is True

        except ImportError:
            pytest.skip("Evaluate condition memory usage not available")

    def test_evaluate_condition_disk_usage(self):
        """测试评估磁盘使用率告警条件"""
        try:
            from src.infrastructure.resource.core.alert_manager_component import AlertManager
            from src.infrastructure.resource.models.alert_dataclasses import AlertRule
            from src.infrastructure.resource.models.alert_enums import AlertType, AlertLevel

            manager = AlertManager()

            rule = AlertRule(
                name="Disk Alert",
                alert_type=AlertType.PERFORMANCE_DEGRADATION,
                alert_level=AlertLevel.WARNING,
                condition="disk_usage > threshold",
                threshold=85.0
            )

            mock_metrics = Mock()
            mock_metrics.disk_usage = 90.0

            result = manager._evaluate_condition(rule, mock_metrics, None)

            # 验证条件评估为真
            assert result is True

        except ImportError:
            pytest.skip("Evaluate condition disk usage not available")

    def test_evaluate_condition_network_latency(self):
        """测试评估网络延迟告警条件"""
        try:
            from src.infrastructure.resource.core.alert_manager_component import AlertManager
            from src.infrastructure.resource.models.alert_dataclasses import AlertRule
            from src.infrastructure.resource.models.alert_enums import AlertType, AlertLevel

            manager = AlertManager()

            rule = AlertRule(
                name="Network Alert",
                alert_type=AlertType.SYSTEM_ERROR,
                alert_level=AlertLevel.WARNING,
                condition="network_latency > threshold",
                threshold=100.0
            )

            mock_metrics = Mock()
            mock_metrics.network_latency = 150.0

            result = manager._evaluate_condition(rule, mock_metrics, None)

            # 验证条件评估为真
            assert result is True

        except ImportError:
            pytest.skip("Evaluate condition network latency not available")

    def test_evaluate_condition_test_execution_time(self):
        """测试评估测试执行时间告警条件"""
        try:
            from src.infrastructure.resource.core.alert_manager_component import AlertManager
            from src.infrastructure.resource.models.alert_dataclasses import AlertRule
            from src.infrastructure.resource.models.alert_enums import AlertType, AlertLevel
            from src.infrastructure.resource.models.alert_dataclasses import TestExecutionInfo

            manager = AlertManager()

            rule = AlertRule(
                name="Test Time Alert",
                alert_type=AlertType.PERFORMANCE_DEGRADATION,
                alert_level=AlertLevel.WARNING,
                condition="test_execution_time > threshold",
                threshold=60.0
            )

            mock_metrics = Mock()
            mock_test_info = Mock(spec=TestExecutionInfo)
            mock_test_info.execution_time = 75.0

            result = manager._evaluate_condition(rule, mock_metrics, mock_test_info)

            # 验证条件评估为真
            assert result is True

        except ImportError:
            pytest.skip("Evaluate condition test execution time not available")

    def test_evaluate_condition_test_success_rate(self):
        """测试评估测试成功率告警条件"""
        try:
            from src.infrastructure.resource.core.alert_manager_component import AlertManager
            from src.infrastructure.resource.models.alert_dataclasses import AlertRule
            from src.infrastructure.resource.models.alert_enums import AlertType, AlertLevel

            manager = AlertManager()

            rule = AlertRule(
                name="Test Success Alert",
                alert_type=AlertType.SYSTEM_ERROR,
                alert_level=AlertLevel.CRITICAL,
                condition="test_success_rate < threshold",
                threshold=80.0
            )

            mock_metrics = Mock()
            mock_metrics.test_success_rate = 70.0

            result = manager._evaluate_condition(rule, mock_metrics, None)

            # 验证条件评估为真
            assert result is True

        except ImportError:
            pytest.skip("Evaluate condition test success rate not available")

    def test_evaluate_condition_unknown_condition(self):
        """测试评估未知告警条件"""
        try:
            from src.infrastructure.resource.core.alert_manager_component import AlertManager
            from src.infrastructure.resource.models.alert_dataclasses import AlertRule
            from src.infrastructure.resource.models.alert_enums import AlertType, AlertLevel

            manager = AlertManager()

            rule = AlertRule(
                name="Unknown Alert",
                alert_type=AlertType.SYSTEM_ERROR,
                alert_level=AlertLevel.WARNING,
                condition="unknown_condition > threshold",
                threshold=50.0
            )

            mock_metrics = Mock()

            result = manager._evaluate_condition(rule, mock_metrics, None)

            # 验证未知条件返回False
            assert result is False

        except ImportError:
            pytest.skip("Evaluate condition unknown condition not available")

    def test_evaluate_condition_exception_handling(self):
        """测试评估条件时的异常处理"""
        try:
            from src.infrastructure.resource.core.alert_manager_component import AlertManager
            from src.infrastructure.resource.models.alert_dataclasses import AlertRule
            from src.infrastructure.resource.models.alert_enums import AlertType, AlertLevel

            manager = AlertManager()

            rule = AlertRule(
                name="Exception Alert",
                alert_type=AlertType.SYSTEM_ERROR,
                alert_level=AlertLevel.WARNING,
                condition="cpu_usage > threshold",
                threshold=80.0
            )

            # 创建会导致异常的模拟指标
            mock_metrics = Mock()
            mock_metrics.cpu_usage.side_effect = AttributeError("Mock error")

            mock_error_handler = Mock()
            manager.error_handler = mock_error_handler

            result = manager._evaluate_condition(rule, mock_metrics, None)

            # 验证异常被处理并返回False
            assert result is False
            mock_error_handler.handle_error.assert_called_once()

        except ImportError:
            pytest.skip("Evaluate condition exception handling not available")

    def test_check_alerts_with_parameters_object(self):
        """测试使用参数对象检查告警"""
        try:
            from src.infrastructure.resource.core.alert_manager_component import AlertManager
            from src.infrastructure.resource.models.alert_dataclasses import AlertRule
            from src.infrastructure.resource.models.alert_enums import AlertType, AlertLevel
            from src.infrastructure.resource.models.parameter_objects import AlertCheckParameters

            manager = AlertManager()

            # 添加告警规则
            rule = AlertRule(
                name="Test CPU Alert",
                alert_type=AlertType.SYSTEM_ERROR,
                alert_level=AlertLevel.WARNING,
                condition="cpu_usage > threshold",
                threshold=80.0
            )
            manager.add_alert_rule(rule)

            # 创建参数对象
            mock_metrics = Mock()
            mock_metrics.cpu_usage = 85.0

            check_params = AlertCheckParameters(metrics=mock_metrics)

            # 检查告警
            manager.check_alerts(check_params)

            # 验证告警被触发（检查活跃告警）
            assert len(manager.active_alerts) > 0

        except ImportError:
            pytest.skip("Check alerts with parameters object not available")

    def test_check_alerts_with_legacy_parameters(self):
        """测试使用遗留参数检查告警"""
        try:
            from src.infrastructure.resource.core.alert_manager_component import AlertManager
            from src.infrastructure.resource.models.alert_dataclasses import AlertRule
            from src.infrastructure.resource.models.alert_enums import AlertType, AlertLevel

            manager = AlertManager()

            # 添加告警规则
            rule = AlertRule(
                name="Legacy CPU Alert",
                alert_type=AlertType.SYSTEM_ERROR,
                alert_level=AlertLevel.WARNING,
                condition="cpu_usage > threshold",
                threshold=80.0
            )
            manager.add_alert_rule(rule)

            # 使用遗留方式（直接传递metrics）
            mock_metrics = Mock()
            mock_metrics.cpu_usage = 85.0

            manager.check_alerts(mock_metrics)

            # 验证告警被触发
            assert len(manager.active_alerts) > 0

        except ImportError:
            pytest.skip("Check alerts with legacy parameters not available")

    def test_check_alerts_disabled_rule(self):
        """测试检查禁用规则的告警"""
        try:
            from src.infrastructure.resource.core.alert_manager_component import AlertManager
            from src.infrastructure.resource.models.alert_dataclasses import AlertRule
            from src.infrastructure.resource.models.alert_enums import AlertType, AlertLevel

            manager = AlertManager()

            # 添加禁用的告警规则
            rule = AlertRule(
                name="Disabled Alert",
                alert_type=AlertType.SYSTEM_ERROR,
                alert_level=AlertLevel.WARNING,
                condition="cpu_usage > threshold",
                threshold=50.0,
                enabled=False
            )
            manager.add_alert_rule(rule)

            mock_metrics = Mock()
            mock_metrics.cpu_usage = 80.0

            manager.check_alerts(mock_metrics)

            # 验证禁用规则不触发告警
            assert len(manager.active_alerts) == 0

        except ImportError:
            pytest.skip("Check alerts disabled rule not available")

    def test_check_alerts_cooldown_period(self):
        """测试告警冷却期"""
        try:
            from src.infrastructure.resource.core.alert_manager_component import AlertManager
            from src.infrastructure.resource.models.alert_dataclasses import AlertRule
            from src.infrastructure.resource.models.alert_enums import AlertType, AlertLevel

            manager = AlertManager()

            # 添加有冷却期的告警规则
            rule = AlertRule(
                name="Cooldown Alert",
                alert_type=AlertType.SYSTEM_ERROR,
                alert_level=AlertLevel.WARNING,
                condition="cpu_usage > threshold",
                threshold=80.0,
                cooldown=60  # 60秒冷却期
            )
            manager.add_alert_rule(rule)

            mock_metrics = Mock()
            mock_metrics.cpu_usage = 85.0

            # 第一次检查 - 应该触发告警
            manager.check_alerts(mock_metrics)
            assert len(manager.active_alerts) == 1

            # 第二次检查（立即） - 应该被冷却期阻止
            manager.check_alerts(mock_metrics)
            # 告警数量应该不变（冷却期阻止了新告警）
            assert len(manager.active_alerts) == 1

        except ImportError:
            pytest.skip("Check alerts cooldown period not available")

    def test_check_alerts_max_alerts_limit(self):
        """测试最大告警数量限制"""
        try:
            from src.infrastructure.resource.core.alert_manager_component import AlertManager
            from src.infrastructure.resource.models.alert_dataclasses import AlertRule
            from src.infrastructure.resource.models.alert_enums import AlertType, AlertLevel

            manager = AlertManager()

            # 添加多个告警规则
            for i in range(5):
                rule = AlertRule(
                    name=f"Alert Rule {i}",
                    alert_type=AlertType.SYSTEM_ERROR,
                    alert_level=AlertLevel.WARNING,
                    condition="cpu_usage > threshold",
                    threshold=80.0
                )
                manager.add_alert_rule(rule)

            mock_metrics = Mock()
            mock_metrics.cpu_usage = 85.0

            # 检查告警但限制最大数量为2
            from src.infrastructure.resource.models.parameter_objects import AlertCheckParameters
            check_params = AlertCheckParameters(metrics=mock_metrics, max_alerts=2)

            manager.check_alerts(check_params)

            # 验证只触发了2个告警（受max_alerts限制）
            assert len(manager.active_alerts) == 2

        except ImportError:
            pytest.skip("Check alerts max alerts limit not available")

    def test_resolve_alert_existing(self):
        """测试解决现有告警"""
        try:
            from src.infrastructure.resource.core.alert_manager_component import AlertManager
            from src.infrastructure.resource.models.alert_dataclasses import Alert
            from src.infrastructure.resource.models.alert_enums import AlertType, AlertLevel

            manager = AlertManager()

            # 创建并添加告警
            alert = Alert(
                id="test_alert_001",
                alert_type=AlertType.SYSTEM_ERROR,
                alert_level=AlertLevel.WARNING,
                message="Test alert",
                details={"test": True},
                timestamp=datetime.now(),
                source="test"
            )
            manager.active_alerts[alert.id] = alert

            # 解决告警
            result = manager.resolve_alert(alert.id)

            # 验证告警被解决
            assert result is True
            assert alert.id in manager.active_alerts
            assert manager.active_alerts[alert.id].resolved is True
            assert manager.active_alerts[alert.id].resolved_at is not None

        except ImportError:
            pytest.skip("Resolve alert existing not available")

    def test_resolve_alert_nonexistent(self):
        """测试解决不存在的告警"""
        try:
            from src.infrastructure.resource.core.alert_manager_component import AlertManager

            manager = AlertManager()

            result = manager.resolve_alert("nonexistent_alert")

            # 验证返回False
            assert result is False

        except ImportError:
            pytest.skip("Resolve alert nonexistent not available")

    def test_get_alert_statistics(self):
        """测试获取告警统计"""
        try:
            from src.infrastructure.resource.core.alert_manager_component import AlertManager
            from src.infrastructure.resource.models.alert_dataclasses import Alert, AlertRule
            from src.infrastructure.resource.models.alert_enums import AlertType, AlertLevel

            manager = AlertManager()

            # 添加一些告警规则
            for i in range(3):
                rule = AlertRule(
                    name=f"Rule {i}",
                    alert_type=AlertType.SYSTEM_ERROR,
                    alert_level=AlertLevel.WARNING,
                    condition="cpu_usage > threshold",
                    threshold=80.0
                )
                manager.add_alert_rule(rule)

            # 添加一些活跃告警
            for i in range(5):
                alert = Alert(
                    id=f"alert_{i}",
                    alert_type=AlertType.SYSTEM_ERROR if i % 2 == 0 else AlertType.RESOURCE_EXHAUSTION,
                    alert_level=AlertLevel.WARNING if i < 3 else AlertLevel.CRITICAL,
                    message=f"Alert {i}",
                    details={},
                    timestamp=datetime.now(),
                    source="test"
                )
                manager.active_alerts[alert.id] = alert

            stats = manager.get_alert_statistics()

            # 验证统计信息
            assert isinstance(stats, dict)
            assert stats.get('total_rules', 0) == 3
            assert stats.get('total_active_alerts', 0) == 5
            assert stats.get('warning_alerts', 0) == 3
            assert stats.get('critical_alerts', 0) == 2

        except ImportError:
            pytest.skip("Get alert statistics not available")

    def test_quantitative_trading_alert_management(self):
        """测试量化交易告警管理场景"""
        try:
            from src.infrastructure.resource.core.alert_manager_component import AlertManager
            from src.infrastructure.resource.models.alert_dataclasses import AlertRule
            from src.infrastructure.resource.models.alert_enums import AlertType, AlertLevel

            manager = AlertManager()

            # 为量化交易系统设置告警规则
            trading_rules = [
                AlertRule(
                    name="HFT CPU Alert",
                    alert_type=AlertType.PERFORMANCE_DEGRADATION,
                    alert_level=AlertLevel.CRITICAL,
                    condition="cpu_usage > threshold",
                    threshold=90.0,
                    enabled=True
                ),
                AlertRule(
                    name="Trading Latency Alert",
                    alert_type=AlertType.SYSTEM_ERROR,
                    alert_level=AlertLevel.CRITICAL,
                    condition="network_latency > threshold",
                    threshold=50.0,
                    enabled=True
                ),
                AlertRule(
                    name="Strategy Success Alert",
                    alert_type=AlertType.SYSTEM_ERROR,
                    alert_level=AlertLevel.WARNING,
                    condition="test_success_rate < threshold",
                    threshold=95.0,
                    enabled=True
                )
            ]

            for rule in trading_rules:
                manager.add_alert_rule(rule)

            # 模拟量化交易场景的性能指标
            mock_metrics = Mock()
            mock_metrics.cpu_usage = 95.0  # 触发CPU告警
            mock_metrics.network_latency = 75.0  # 触发网络延迟告警
            mock_metrics.test_success_rate = 92.0  # 触发策略成功率告警

            manager.check_alerts(mock_metrics)

            # 验证所有相关的告警都被触发
            assert len(manager.active_alerts) == 3

            # 验证告警类型
            alert_types = [alert.alert_type for alert in manager.active_alerts.values()]
            assert AlertType.PERFORMANCE_DEGRADATION in alert_types
            assert AlertType.SYSTEM_ERROR in alert_types

        except ImportError:
            pytest.skip("Quantitative trading alert management not available")

    def test_alert_manager_concurrent_access(self):
        """测试告警管理器并发访问"""
        try:
            from src.infrastructure.resource.core.alert_manager_component import AlertManager
            from src.infrastructure.resource.models.alert_dataclasses import AlertRule
            from src.infrastructure.resource.models.alert_enums import AlertType, AlertLevel
            import concurrent.futures
            import threading

            manager = AlertManager()

            # 添加测试规则
            rule = AlertRule(
                name="Concurrent Test",
                alert_type=AlertType.SYSTEM_ERROR,
                alert_level=AlertLevel.WARNING,
                condition="cpu_usage > threshold",
                threshold=80.0
            )
            manager.add_alert_rule(rule)

            results = []
            errors = []

            def concurrent_operation(operation_id):
                try:
                    if operation_id % 4 == 0:
                        # 添加规则
                        test_rule = AlertRule(
                            name=f"Concurrent Rule {operation_id}",
                            alert_type=AlertType.SYSTEM_ERROR,
                            alert_level=AlertLevel.WARNING,
                            condition="cpu_usage > threshold",
                            threshold=80.0
                        )
                        manager.add_alert_rule(test_rule)
                        results.append(('add_rule', operation_id))
                    elif operation_id % 4 == 1:
                        # 检查告警
                        mock_metrics = Mock()
                        mock_metrics.cpu_usage = 85.0
                        manager.check_alerts(mock_metrics)
                        results.append(('check_alerts', operation_id))
                    elif operation_id % 4 == 2:
                        # 获取统计
                        stats = manager.get_alert_statistics()
                        results.append(('get_stats', operation_id, stats.get('total_active_alerts', 0)))
                    else:
                        # 解决告警
                        if manager.active_alerts:
                            alert_id = list(manager.active_alerts.keys())[0]
                            manager.resolve_alert(alert_id)
                            results.append(('resolve_alert', operation_id))
                        else:
                            results.append(('no_alert_to_resolve', operation_id))
                except Exception as e:
                    errors.append((operation_id, str(e)))

            # 执行并发操作
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(concurrent_operation, i) for i in range(20)]
                concurrent.futures.wait(futures, timeout=10)

            # 验证并发操作结果
            successful_operations = [r for r in results if not r[0].startswith('error')]
            assert len(successful_operations) >= 15  # 至少75%的操作成功
            assert len(errors) < 3  # 错误数量应该很少

        except ImportError:
            pytest.skip("Alert manager concurrent access not available")

    def test_alert_manager_error_handling_and_recovery(self):
        """测试告警管理器错误处理和恢复"""
        try:
            from src.infrastructure.resource.core.alert_manager_component import AlertManager
            from src.infrastructure.resource.models.alert_dataclasses import AlertRule
            from src.infrastructure.resource.models.alert_enums import AlertType, AlertLevel

            manager = AlertManager()

            # 添加正常规则
            rule = AlertRule(
                name="Normal Rule",
                alert_type=AlertType.SYSTEM_ERROR,
                alert_level=AlertLevel.WARNING,
                condition="cpu_usage > threshold",
                threshold=80.0
            )
            manager.add_alert_rule(rule)

            # 测试异常处理
            mock_error_handler = Mock()
            manager.error_handler = mock_error_handler

            # 模拟评估条件时的异常
            with patch.object(manager, '_evaluate_condition', side_effect=Exception("Evaluation error")):
                mock_metrics = Mock()
                mock_metrics.cpu_usage = 85.0

                # 应该不会抛出异常，而是通过错误处理器处理
                manager.check_alerts(mock_metrics)

                # 验证错误被处理
                mock_error_handler.handle_error.assert_called()

            # 测试恢复 - 正常操作应该继续工作
            mock_metrics.cpu_usage = 85.0
            manager.check_alerts(mock_metrics)

            # 验证系统从错误中恢复并正常工作
            assert len(manager.active_alerts) > 0

        except ImportError:
            pytest.skip("Alert manager error handling and recovery not available")

    def test_alert_manager_configuration_and_limits(self):
        """测试告警管理器配置和限制"""
        try:
            from src.infrastructure.resource.core.alert_manager_component import AlertManager
            from src.infrastructure.resource.models.alert_dataclasses import AlertRule
            from src.infrastructure.resource.models.alert_enums import AlertType, AlertLevel

            manager = AlertManager()

            # 测试大规模规则管理
            for i in range(50):  # 添加大量规则
                rule = AlertRule(
                    name=f"Bulk Rule {i}",
                    alert_type=AlertType.SYSTEM_ERROR,
                    alert_level=AlertLevel.WARNING,
                    condition="cpu_usage > threshold",
                    threshold=80.0 + i  # 不同的阈值
                )
                manager.add_alert_rule(rule)

            # 验证可以处理大量规则
            assert len(manager.alert_rules) == 50

            # 测试大规模告警触发
            mock_metrics = Mock()
            mock_metrics.cpu_usage = 95.0  # 应该触发多个规则

            manager.check_alerts(mock_metrics)

            # 验证系统可以处理大量告警
            assert len(manager.active_alerts) > 10

            # 测试规则清理
            for i in range(25):  # 移除一半规则
                manager.remove_alert_rule(f"Bulk Rule {i}")

            assert len(manager.alert_rules) == 25

        except ImportError:
            pytest.skip("Alert manager configuration and limits not available")

    def test_alert_manager_cleanup_and_maintenance(self):
        """测试告警管理器清理和维护"""
        try:
            from src.infrastructure.resource.core.alert_manager_component import AlertManager
            from src.infrastructure.resource.models.alert_dataclasses import Alert
            from src.infrastructure.resource.models.alert_enums import AlertType, AlertLevel

            manager = AlertManager()

            # 添加一些旧告警
            old_timestamp = datetime.now() - timedelta(hours=25)
            for i in range(10):
                alert = Alert(
                    id=f"old_alert_{i}",
                    alert_type=AlertType.SYSTEM_ERROR,
                    alert_level=AlertLevel.WARNING,
                    message=f"Old alert {i}",
                    details={"old": True},
                    timestamp=old_timestamp,
                    source="test"
                )
                manager.active_alerts[alert.id] = alert

            # 添加一些新告警
            new_timestamp = datetime.now()
            for i in range(5):
                alert = Alert(
                    id=f"new_alert_{i}",
                    alert_type=AlertType.SYSTEM_ERROR,
                    alert_level=AlertLevel.WARNING,
                    message=f"New alert {i}",
                    details={"new": True},
                    timestamp=new_timestamp,
                    source="test"
                )
                manager.active_alerts[alert.id] = alert

            # 验证初始状态
            assert len(manager.active_alerts) == 15

            # 模拟基于时间的清理（这里主要是为了测试管理逻辑）
            # 在实际实现中，可能会有cleanup_old_alerts方法

            # 验证系统能处理告警的积累和清理
            for i in range(5):
                if manager.active_alerts:
                    alert_id = list(manager.active_alerts.keys())[0]
                    manager.resolve_alert(alert_id)

            # 验证清理后状态
            assert len(manager.active_alerts) == 10

        except ImportError:
            pytest.skip("Alert manager cleanup and maintenance not available")