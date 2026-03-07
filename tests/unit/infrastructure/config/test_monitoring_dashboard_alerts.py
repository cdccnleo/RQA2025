#!/usr/bin/env python3
"""
测试监控面板告警管理

测试覆盖：
- AlertManager基类的基础功能
- 具体实现的告警创建和管理
- 告警监听器机制
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
from unittest.mock import MagicMock, patch
from typing import Optional, Dict

from src.infrastructure.config.monitoring.dashboard_alerts import AlertManager, InMemoryAlertManager
from src.infrastructure.config.monitoring.dashboard_models import (
    Alert, AlertSeverity, AlertStatus
)


class ConcreteAlertManager(AlertManager):
    """具体的AlertManager实现，用于测试"""

    def __init__(self):
        super().__init__()
        self._next_id = 1

    def create_alert(self, name: str, description: str, severity: AlertSeverity,
                     labels: Optional[Dict[str, str]] = None,
                     annotations: Optional[Dict[str, str]] = None,
                     value: Optional[float] = None,
                     threshold: Optional[float] = None) -> str:
        """创建告警的具体实现"""
        alert_id = f"alert_{self._next_id}"
        self._next_id += 1

        alert = Alert(
            id=alert_id,
            name=name,
            description=description,
            severity=severity,
            status=AlertStatus.ACTIVE,
            timestamp=time.time(),
            labels=labels or {},
            annotations=annotations or {},
            value=value,
            threshold=threshold
        )

        with self._lock:
            self._alerts[alert_id] = alert

        # 触发监听器
        for listener in self._listeners:
            try:
                listener(alert)
            except Exception as e:
                # 在测试中我们忽略监听器异常
                pass

        return alert_id

    def resolve_alert(self, alert_id: str) -> bool:
        """解决告警的具体实现"""
        with self._lock:
            if alert_id in self._alerts:
                alert = self._alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.updated_at = time.time()
                return True
        return False

    def acknowledge_alert(self, alert_id: str) -> bool:
        """确认告警的具体实现"""
        with self._lock:
            if alert_id in self._alerts:
                alert = self._alerts[alert_id]
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.updated_at = time.time()
                return True
        return False


class TestAlertManager:
    """测试告警管理器"""

    def setup_method(self):
        """测试前准备"""
        self.manager = InMemoryAlertManager()

    def test_initialization(self):
        """测试初始化"""
        assert self.manager is not None
        assert hasattr(self.manager, '_alerts')
        assert hasattr(self.manager, '_listeners')
        assert hasattr(self.manager, '_lock')

    def test_create_alert_basic(self):
        """测试创建基本告警"""
        alert_id = self.manager.create_alert(
            name="Test Alert",
            description="This is a test alert",
            severity=AlertSeverity.WARNING
        )

        assert alert_id.startswith("alert_")
        assert alert_id in self.manager._alerts

        alert = self.manager._alerts[alert_id]
        assert alert.name == "Test Alert"
        assert alert.description == "This is a test alert"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.status == AlertStatus.ACTIVE

    def test_create_alert_with_labels_and_annotations(self):
        """测试创建带有标签和注释的告警"""
        labels = {"service": "config", "component": "manager"}
        annotations = {"runbook": "check_config_service", "summary": "Config service is down"}

        alert_id = self.manager.create_alert(
            name="Service Down",
            description="Configuration service is not responding",
            severity=AlertSeverity.CRITICAL,
            labels=labels,
            annotations=annotations,
            value=0.0,
            threshold=1.0
        )

        alert = self.manager._alerts[alert_id]
        assert alert.labels == labels
        assert alert.annotations == annotations
        assert alert.value == 0.0
        assert alert.threshold == 1.0

    def test_resolve_alert(self):
        """测试解决告警"""
        alert_id = self.manager.create_alert(
            name="Test Alert",
            description="Test description",
            severity=AlertSeverity.INFO
        )

        # 确认告警存在且状态为ACTIVE
        assert self.manager._alerts[alert_id].status == AlertStatus.ACTIVE

        # 解决告警
        result = self.manager.resolve_alert(alert_id)
        assert result is True

        # 确认状态已更新
        assert self.manager._alerts[alert_id].status == AlertStatus.RESOLVED

    def test_resolve_nonexistent_alert(self):
        """测试解决不存在的告警"""
        result = self.manager.resolve_alert("nonexistent_alert")
        assert result is False

    def test_acknowledge_alert(self):
        """测试确认告警"""
        alert_id = self.manager.create_alert(
            name="Test Alert",
            description="Test description",
            severity=AlertSeverity.WARNING
        )

        # 确认告警
        result = self.manager.acknowledge_alert(alert_id)
        assert result is True

        # 确认状态已更新
        assert self.manager._alerts[alert_id].status == AlertStatus.ACKNOWLEDGED

    def test_acknowledge_nonexistent_alert(self):
        """测试确认不存在的告警"""
        result = self.manager.acknowledge_alert("nonexistent_alert")
        assert result is False

    def test_add_listener(self):
        """测试添加监听器"""
        listener = MagicMock()
        self.manager.add_listener(listener)

        assert listener in self.manager._listeners

    def test_remove_listener(self):
        """测试移除监听器"""
        listener = MagicMock()
        self.manager.add_listener(listener)
        assert listener in self.manager._listeners

        self.manager.remove_listener(listener)
        assert listener not in self.manager._listeners

    def test_remove_nonexistent_listener(self):
        """测试移除不存在的监听器"""
        listener1 = MagicMock()
        listener2 = MagicMock()

        self.manager.add_listener(listener1)
        self.manager.remove_listener(listener2)  # 尝试移除未添加的监听器

        assert listener1 in self.manager._listeners

    def test_listener_notification(self):
        """测试监听器通知"""
        listener = MagicMock()

        self.manager.add_listener(listener)

        # 创建告警，应该触发监听器
        alert_id = self.manager.create_alert(
            name="Test Alert",
            description="Test description",
            severity=AlertSeverity.ERROR
        )

        # 验证监听器被调用
        listener.assert_called_once()
        called_alert = listener.call_args[0][0]
        assert called_alert.id == alert_id
        assert called_alert.name == "Test Alert"

    def test_get_alert(self):
        """测试获取告警"""
        alert_id = self.manager.create_alert(
            name="Test Alert",
            description="Test description",
            severity=AlertSeverity.WARNING
        )

        alert = self.manager.get_alert(alert_id)
        assert alert is not None
        assert alert.id == alert_id
        assert alert.name == "Test Alert"

    def test_get_alert_nonexistent(self):
        """测试获取不存在的告警"""
        alert = self.manager.get_alert("nonexistent_alert")
        assert alert is None

    def test_get_alerts_by_status(self):
        """测试按状态获取告警"""
        # 创建不同状态的告警
        alert1_id = self.manager.create_alert("Alert 1", "Desc 1", AlertSeverity.WARNING)
        alert2_id = self.manager.create_alert("Alert 2", "Desc 2", AlertSeverity.ERROR)

        self.manager.resolve_alert(alert1_id)

        firing_alerts = self.manager.get_alerts_by_status(AlertStatus.ACTIVE)
        resolved_alerts = self.manager.get_alerts_by_status(AlertStatus.RESOLVED)

        assert len(firing_alerts) == 1
        assert len(resolved_alerts) == 1
        assert firing_alerts[0].id == alert2_id
        assert resolved_alerts[0].id == alert1_id

    def test_get_alerts_by_severity(self):
        """测试按严重程度获取告警"""
        self.manager.create_alert("Warning Alert", "Warning", AlertSeverity.WARNING)
        self.manager.create_alert("Error Alert", "Error", AlertSeverity.ERROR)
        self.manager.create_alert("Critical Alert", "Critical", AlertSeverity.CRITICAL)

        warning_alerts = self.manager.get_alerts_by_severity(AlertSeverity.WARNING)
        error_alerts = self.manager.get_alerts_by_severity(AlertSeverity.ERROR)
        critical_alerts = self.manager.get_alerts_by_severity(AlertSeverity.CRITICAL)

        assert len(warning_alerts) == 1
        assert len(error_alerts) == 1
        assert len(critical_alerts) == 1

    def test_clear_resolved_alerts(self):
        """测试清除已解决的告警"""
        alert1_id = self.manager.create_alert("Alert 1", "Desc 1", AlertSeverity.WARNING)
        alert2_id = self.manager.create_alert("Alert 2", "Desc 2", AlertSeverity.ERROR)

        self.manager.resolve_alert(alert1_id)

        # 清除已解决的告警（设置max_age_days=0以清除所有已解决告警）
        cleared_count = self.manager.clear_resolved_alerts(max_age_days=0)
        assert cleared_count == 1

        # 验证已解决的告警被清除
        assert alert1_id not in self.manager._alerts
        assert alert2_id in self.manager._alerts

    def test_get_alerts_summary(self):
        """测试获取告警摘要"""
        self.manager.create_alert("Alert 1", "Desc 1", AlertSeverity.WARNING)
        self.manager.create_alert("Alert 2", "Desc 2", AlertSeverity.ERROR)
        alert3_id = self.manager.create_alert("Alert 3", "Desc 3", AlertSeverity.CRITICAL)

        self.manager.acknowledge_alert(alert3_id)

        summary = self.manager.get_alerts_summary()

        assert summary["total"] == 3
        assert summary["firing"] == 2
        assert summary["acknowledged"] == 1
        assert summary["resolved"] == 0
        assert summary["by_severity"]["warning"] == 1
        assert summary["by_severity"]["error"] == 1
        assert summary["by_severity"]["critical"] == 1

    def test_thread_safety(self):
        """测试线程安全性"""
        import threading
        import concurrent.futures

        results = []
        errors = []

        def create_alerts(worker_id):
            try:
                for i in range(10):
                    alert_id = self.manager.create_alert(
                        f"Thread {worker_id} Alert {i}",
                        f"Description {i}",
                        AlertSeverity.INFO
                    )
                    results.append(alert_id)
            except Exception as e:
                errors.append(str(e))

        # 创建多个线程并发创建告警
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(create_alerts, i) for i in range(5)]
            concurrent.futures.wait(futures)

        # 验证结果
        assert len(results) == 50  # 5个线程 * 10个告警
        assert len(errors) == 0
        assert len(self.manager._alerts) == 50
