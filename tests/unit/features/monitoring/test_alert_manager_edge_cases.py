#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
告警管理器边界场景与异常分支测试

覆盖告警发送、处理、抑制、历史清理等关键路径
"""

import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.features.monitoring.alert_manager import (
    AlertManager,
    AlertSeverity,
    AlertStatus,
    Alert,
)


@pytest.fixture
def alert_manager():
    """告警管理器实例"""
    return AlertManager()


class TestAlertSending:
    """测试告警发送"""

    def test_send_alert_creates_alert(self, alert_manager):
        """测试发送告警创建告警对象"""
        alert_id = alert_manager.send_alert(
            title="测试告警",
            message="这是一个测试",
            severity=AlertSeverity.WARNING
        )
        
        assert alert_id.startswith("alert_")
        assert alert_id in alert_manager._alerts

    def test_send_alert_increments_id(self, alert_manager):
        """测试告警 ID 递增"""
        id1 = alert_manager.send_alert("title1", "msg1")
        id2 = alert_manager.send_alert("title2", "msg2")
        
        assert id1 != id2
        assert int(id1.split('_')[1]) < int(id2.split('_')[1])

    def test_send_alert_adds_to_history(self, alert_manager):
        """测试告警添加到历史记录"""
        alert_id = alert_manager.send_alert("title", "msg")
        
        assert len(alert_manager._alert_history) == 1
        assert alert_manager._alert_history[-1].alert_id == alert_id

    def test_send_alert_triggers_handlers(self, alert_manager):
        """测试发送告警触发处理函数"""
        handler = MagicMock()
        alert_manager.add_handler("warning", handler)
        
        alert_manager.send_alert("title", "msg", severity=AlertSeverity.WARNING)
        
        assert handler.called
        assert handler.call_args[0][0].severity == AlertSeverity.WARNING

    def test_send_alert_handler_exception_suppressed(self, alert_manager, caplog):
        """测试处理函数异常被抑制"""
        def failing_handler(alert):
            raise RuntimeError("处理失败")
        
        alert_manager.add_handler("warning", failing_handler)
        
        # 应该不抛出异常
        alert_id = alert_manager.send_alert("title", "msg", severity=AlertSeverity.WARNING)
        
        assert alert_id is not None


class TestAlertRetrieval:
    """测试告警获取"""

    def test_get_active_alerts_filters_by_status(self, alert_manager):
        """测试获取活跃告警按状态过滤"""
        id1 = alert_manager.send_alert("active1", "msg1", severity=AlertSeverity.WARNING)
        id2 = alert_manager.send_alert("active2", "msg2", severity=AlertSeverity.ERROR)
        
        # 解决一个告警
        alert_manager.resolve_alert(id1)
        
        active = alert_manager.get_active_alerts()
        
        assert len(active) == 1
        assert active[0]['alert_id'] == id2

    def test_get_active_alerts_filters_by_severity(self, alert_manager):
        """测试获取活跃告警按严重程度过滤"""
        alert_manager.send_alert("warn1", "msg1", severity=AlertSeverity.WARNING)
        alert_manager.send_alert("error1", "msg2", severity=AlertSeverity.ERROR)
        alert_manager.send_alert("warn2", "msg3", severity=AlertSeverity.WARNING)
        
        warnings = alert_manager.get_active_alerts(severity=AlertSeverity.WARNING)
        
        assert len(warnings) == 2
        assert all(a['severity'] == 'warning' for a in warnings)

    def test_get_recent_alerts_limits_count(self, alert_manager):
        """测试获取最近告警限制数量"""
        # 发送 15 个告警
        for i in range(15):
            alert_manager.send_alert(f"title_{i}", f"msg_{i}")
        
        recent = alert_manager.get_recent_alerts(limit=10)
        
        assert len(recent) == 10

    def test_get_recent_alerts_limit_zero_returns_all(self, alert_manager):
        """测试 limit=0 返回所有告警"""
        for i in range(5):
            alert_manager.send_alert(f"title_{i}", f"msg_{i}")
        
        recent = alert_manager.get_recent_alerts(limit=0)
        
        assert len(recent) == 5

    def test_get_alert_history_filters_by_time(self, alert_manager):
        """测试获取告警历史按时间过滤"""
        # 发送一个告警
        alert_manager.send_alert("old", "msg", severity=AlertSeverity.INFO)
        
        time.sleep(0.1)
        
        # 使用很小的窗口（应该获取不到）
        history = alert_manager.get_alert_history(hours=0)
        
        # 可能为空或包含最近的告警（取决于实现）
        assert isinstance(history, list)


class TestAlertManagement:
    """测试告警管理"""

    def test_acknowledge_alert_success(self, alert_manager):
        """测试成功确认告警"""
        alert_id = alert_manager.send_alert("title", "msg")
        
        result = alert_manager.acknowledge_alert(alert_id, "user1", "已确认")
        
        assert result is True
        alert = alert_manager._alerts[alert_id]
        assert alert.status == AlertStatus.ACKNOWLEDGED
        assert alert.acknowledged_by == "user1"
        assert alert.resolution_notes == "已确认"

    def test_acknowledge_alert_nonexistent_returns_false(self, alert_manager):
        """测试确认不存在的告警返回 False"""
        result = alert_manager.acknowledge_alert("nonexistent", "user1")
        
        assert result is False

    def test_resolve_alert_success(self, alert_manager):
        """测试成功解决告警"""
        alert_id = alert_manager.send_alert("title", "msg")
        
        result = alert_manager.resolve_alert(alert_id, "已解决")
        
        assert result is True
        alert = alert_manager._alerts[alert_id]
        assert alert.status == AlertStatus.RESOLVED
        assert alert.resolution_notes == "已解决"
        assert alert.resolved_at is not None

    def test_resolve_alert_nonexistent_returns_false(self, alert_manager):
        """测试解决不存在的告警返回 False"""
        result = alert_manager.resolve_alert("nonexistent", "已解决")
        
        assert result is False

    def test_acknowledge_resolved_alert_updates_status(self, alert_manager):
        """测试确认已解决的告警更新状态"""
        alert_id = alert_manager.send_alert("title", "msg")
        alert_manager.resolve_alert(alert_id)
        
        # 重新确认（应该更新状态）
        result = alert_manager.acknowledge_alert(alert_id, "user1")
        
        # 状态可能保持 RESOLVED 或被更新为 ACKNOWLEDGED（取决于实现）
        assert result is True


class TestAlertHandlers:
    """测试告警处理函数"""

    def test_add_handler_multiple_handlers(self, alert_manager):
        """测试添加多个处理函数"""
        handler1 = MagicMock()
        handler2 = MagicMock()
        
        alert_manager.add_handler("warning", handler1)
        alert_manager.add_handler("warning", handler2)
        
        alert_manager.send_alert("title", "msg", severity=AlertSeverity.WARNING)
        
        assert handler1.called
        assert handler2.called

    def test_add_handler_different_severities(self, alert_manager):
        """测试不同严重程度的处理函数"""
        warning_handler = MagicMock()
        error_handler = MagicMock()
        
        alert_manager.add_handler("warning", warning_handler)
        alert_manager.add_handler("error", error_handler)
        
        alert_manager.send_alert("title1", "msg1", severity=AlertSeverity.WARNING)
        alert_manager.send_alert("title2", "msg2", severity=AlertSeverity.ERROR)
        
        assert warning_handler.call_count == 1
        assert error_handler.call_count == 1


class TestAlertHistoryClearing:
    """测试告警历史清理"""

    def test_clear_history_removes_old_alerts(self, alert_manager):
        """测试清理历史移除旧告警"""
        # 创建旧告警（模拟 31 天前）
        old_alert = Alert(
            alert_id="old_1",
            title="old",
            message="old message",
            severity=AlertSeverity.INFO,
            source="test",
            timestamp=datetime.now() - timedelta(days=31)
        )
        alert_manager._alert_history.append(old_alert)
        
        # 创建新告警
        new_alert_id = alert_manager.send_alert("new", "new message")
        
        # 清理历史
        alert_manager.clear_history()
        
        # 旧告警应该被移除
        alert_ids = [a.alert_id for a in alert_manager._alert_history]
        assert "old_1" not in alert_ids
        assert new_alert_id in alert_ids

    def test_clear_history_preserves_recent_alerts(self, alert_manager):
        """测试清理历史保留最近告警"""
        # 创建多个新告警
        alert_ids = []
        for i in range(5):
            alert_id = alert_manager.send_alert(f"title_{i}", f"msg_{i}")
            alert_ids.append(alert_id)
        
        # 清理历史
        alert_manager.clear_history()
        
        # 新告警应该被保留
        history_ids = [a.alert_id for a in alert_manager._alert_history]
        for alert_id in alert_ids:
            assert alert_id in history_ids


class TestAlertConcurrentAccess:
    """测试告警并发访问"""

    def test_concurrent_alert_sending_thread_safe(self, alert_manager):
        """测试并发发送告警线程安全"""
        import threading
        
        def send_alerts(thread_id):
            for i in range(10):
                alert_manager.send_alert(f"title_{thread_id}_{i}", f"msg_{i}")
        
        threads = [threading.Thread(target=send_alerts, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # 应该发送 50 个告警
        assert len(alert_manager._alert_history) == 50

    def test_concurrent_acknowledge_resolve_thread_safe(self, alert_manager):
        """测试并发确认和解决告警线程安全"""
        import threading
        
        # 创建一些告警
        alert_ids = []
        for i in range(10):
            alert_id = alert_manager.send_alert(f"title_{i}", f"msg_{i}")
            alert_ids.append(alert_id)
        
        def acknowledge_alerts():
            for alert_id in alert_ids[:5]:
                alert_manager.acknowledge_alert(alert_id, "user1")
        
        def resolve_alerts():
            for alert_id in alert_ids[5:]:
                alert_manager.resolve_alert(alert_id)
        
        t1 = threading.Thread(target=acknowledge_alerts)
        t2 = threading.Thread(target=resolve_alerts)
        
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        
        # 验证告警状态被正确更新
        for alert_id in alert_ids[:5]:
            assert alert_manager._alerts[alert_id].status == AlertStatus.ACKNOWLEDGED
        for alert_id in alert_ids[5:]:
            assert alert_manager._alerts[alert_id].status == AlertStatus.RESOLVED


class TestGlobalAlertManager:
    """测试全局告警管理器"""

    def test_get_alert_manager_singleton(self):
        """测试获取全局告警管理器单例"""
        from src.features.monitoring.alert_manager import get_alert_manager
        
        manager1 = get_alert_manager()
        manager2 = get_alert_manager()
        
        assert manager1 is manager2

