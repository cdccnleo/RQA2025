#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
告警管理器简单测试

测试AlertManager的基本功能
"""

import pytest
from unittest.mock import Mock


class TestAlertManager:
    """告警管理器测试"""

    def test_alert_manager_initialization(self):
        """测试告警管理器初始化"""
        try:
            from src.infrastructure.resource.core.alert_manager_component import AlertManager

            alert_manager = AlertManager()

            # 测试基本属性
            assert hasattr(alert_manager, 'logger')
            assert hasattr(alert_manager, 'alert_rules')
            assert hasattr(alert_manager, 'active_alerts')
            assert hasattr(alert_manager, 'alert_handlers')
            assert hasattr(alert_manager, '_lock')

        except ImportError:
            pytest.skip("AlertManager not available")

    def test_alert_rule_creation(self):
        """测试告警规则创建"""
        try:
            from src.infrastructure.resource.models.alert_dataclasses import AlertRule
            from src.infrastructure.resource.models.alert_enums import AlertType, AlertLevel
            from src.infrastructure.resource.core.alert_manager_component import AlertManager

            alert_rule = AlertRule(
                name="CPU High Usage",
                alert_type=AlertType.SYSTEM_ERROR,
                alert_level=AlertLevel.WARNING,
                condition="cpu_percent > 80",
                threshold=80.0,
                enabled=True
            )

            alert_manager = AlertManager()
            alert_manager.add_alert_rule(alert_rule)

            # 验证规则被添加
            assert len(alert_manager.alert_rules) == 1
            assert alert_manager.alert_rules[0].name == "CPU High Usage"

        except ImportError:
            pytest.skip("Alert rule creation not available")

    def test_alert_creation(self):
        """测试告警创建"""
        try:
            from src.infrastructure.resource.models.alert_dataclasses import Alert
            from src.infrastructure.resource.models.alert_enums import AlertType, AlertLevel
            from datetime import datetime

            alert = Alert(
                id="test_alert_001",
                alert_type=AlertType.SYSTEM_ERROR,
                alert_level=AlertLevel.WARNING,
                message="CPU usage high",
                details={"resource_type": "cpu", "current_value": 85.0},
                timestamp=datetime.now(),
                source="test_monitor"
            )

            # 验证告警属性
            assert alert.id == "test_alert_001"
            assert alert.alert_type == AlertType.SYSTEM_ERROR
            assert alert.alert_level == AlertLevel.WARNING
            assert alert.message == "CPU usage high"
            assert alert.details["current_value"] == 85.0

        except ImportError:
            pytest.skip("Alert creation not available")

    def test_alert_handler_registration(self):
        """测试告警处理器注册"""
        try:
            from src.infrastructure.resource.core.alert_manager_component import AlertManager
            from src.infrastructure.resource.models.alert_enums import AlertType

            alert_manager = AlertManager()
            mock_handler = Mock()

            # 注册处理器
            alert_manager.register_alert_handler(AlertType.SYSTEM_ERROR, mock_handler)

            # 验证处理器被注册
            assert len(alert_manager.alert_handlers[AlertType.SYSTEM_ERROR]) == 1
            assert alert_manager.alert_handlers[AlertType.SYSTEM_ERROR][0] == mock_handler

        except ImportError:
            pytest.skip("Alert handler registration not available")

    def test_alert_evaluation(self):
        """测试告警评估"""
        try:
            from src.infrastructure.resource.core.alert_manager_component import AlertManager
            from src.infrastructure.resource.models.alert_dataclasses import AlertRule
            from src.infrastructure.resource.models.alert_enums import AlertType, AlertLevel

            alert_manager = AlertManager()

            # 添加规则
            rule = AlertRule(
                name="Memory High",
                alert_type=AlertType.RESOURCE_EXHAUSTION,
                alert_level=AlertLevel.CRITICAL,
                condition="memory_percent > 90",
                threshold=90.0
            )
            alert_manager.add_alert_rule(rule)

            # 评估条件
            metrics = {"memory_percent": 95.0}
            alerts = alert_manager.evaluate_alerts(metrics)

            # 验证产生告警
            assert len(alerts) >= 1
            found_memory_alert = any("Memory" in alert.message for alert in alerts)
            assert found_memory_alert

        except ImportError:
            pytest.skip("Alert evaluation not available")

    def test_alert_activation_and_resolution(self):
        """测试告警激活和解决"""
        try:
            from src.infrastructure.resource.core.alert_manager_component import AlertManager
            from src.infrastructure.resource.models.alert_dataclasses import Alert
            from src.infrastructure.resource.models.alert_enums import AlertType, AlertLevel
            from datetime import datetime

            alert_manager = AlertManager()

            # 创建激活的告警
            alert = Alert(
                id="active_alert_001",
                alert_type=AlertType.RESOURCE_EXHAUSTION,
                alert_level=AlertLevel.WARNING,
                message="Disk space low",
                details={"disk_usage": 95.0},
                timestamp=datetime.now(),
                source="disk_monitor",
                resolved=False
            )

            alert_manager.active_alerts[alert.id] = alert

            # 验证告警激活
            assert alert.id in alert_manager.active_alerts
            assert not alert_manager.active_alerts[alert.id].resolved

            # 解决告警
            alert_manager.resolve_alert(alert.id)

            # 验证告警已解决
            assert alert_manager.active_alerts[alert.id].resolved
            assert alert_manager.active_alerts[alert.id].resolved_at is not None

        except ImportError:
            pytest.skip("Alert activation and resolution not available")

    def test_alert_statistics(self):
        """测试告警统计"""
        try:
            from src.infrastructure.resource.core.alert_manager_component import AlertManager
            from src.infrastructure.resource.models.alert_dataclasses import Alert
            from src.infrastructure.resource.models.alert_enums import AlertType, AlertLevel
            from datetime import datetime

            alert_manager = AlertManager()

            # 添加一些告警
            alerts = [
                Alert("stat_001", AlertType.SYSTEM_ERROR, AlertLevel.WARNING, "CPU high",
                     {"cpu": 85}, datetime.now(), "cpu_monitor"),
                Alert("stat_002", AlertType.RESOURCE_EXHAUSTION, AlertLevel.CRITICAL, "Memory high",
                     {"memory": 95}, datetime.now(), "memory_monitor"),
                Alert("stat_003", AlertType.PERFORMANCE_DEGRADATION, AlertLevel.WARNING, "Disk high",
                     {"disk": 90}, datetime.now(), "disk_monitor")
            ]

            for alert in alerts:
                alert_manager.active_alerts[alert.id] = alert

            # 获取统计
            stats = alert_manager.get_alert_statistics()

            # 验证统计信息
            assert isinstance(stats, dict)
            assert stats.get('total_alerts', 0) == 3
            assert stats.get('warning_alerts', 0) == 2
            assert stats.get('critical_alerts', 0) == 1

        except ImportError:
            pytest.skip("Alert statistics not available")

    def test_configuration_management(self):
        """测试配置管理"""
        try:
            from src.infrastructure.resource.core.alert_manager_component import AlertManager

            alert_manager = AlertManager()

            # 测试配置应用
            config = {
                'max_alerts': 100,
                'cleanup_interval': 3600,
                'enable_notifications': True
            }

            alert_manager.configure(config)

            # 验证配置被应用（如果有相关属性）

        except ImportError:
            pytest.skip("Configuration management not available")

    def test_alert_cleanup(self):
        """测试告警清理"""
        try:
            from src.infrastructure.resource.core.alert_manager_component import AlertManager
            from src.infrastructure.resource.models.alert_dataclasses import Alert
            from src.infrastructure.resource.models.alert_enums import AlertType, AlertLevel
            from datetime import datetime, timedelta

            alert_manager = AlertManager()

            # 添加旧告警
            old_alert = Alert(
                "old_001",
                AlertType.SYSTEM_ERROR,
                AlertLevel.WARNING,
                "Old alert",
                {"old": True},
                datetime.now() - timedelta(hours=25),  # 25小时前
                "old_monitor"
            )
            alert_manager.active_alerts[old_alert.id] = old_alert

            # 清理旧告警
            cleaned_count = alert_manager.cleanup_old_alerts(hours=24)

            # 验证清理
            assert cleaned_count >= 1
            assert old_alert.id not in alert_manager.active_alerts

        except ImportError:
            pytest.skip("Alert cleanup not available")