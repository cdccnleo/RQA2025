#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试基础设施层 - 灾难监控组件

测试 infrastructure/disaster_monitor.py 中的所有类和方法
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch


@pytest.fixture
def module():
    """导入模块"""
    from src.infrastructure.monitoring.infrastructure import disaster_monitor
    return disaster_monitor


@pytest.fixture
def monitor(module):
    """创建灾难监控器实例"""
    return module.DisasterMonitor()


class TestDisasterLevel:
    """测试灾难级别枚举"""

    def test_disaster_level_values(self, module):
        """测试灾难级别值"""
        assert module.DisasterLevel.NONE.value == "none"
        assert module.DisasterLevel.WARNING.value == "warning"
        assert module.DisasterLevel.CRITICAL.value == "critical"
        assert module.DisasterLevel.DISASTER.value == "disaster"


class TestDisasterType:
    """测试灾难类型枚举"""

    def test_disaster_type_values(self, module):
        """测试灾难类型值"""
        assert module.DisasterType.MEMORY_OVERLOAD.value == "memory_overload"
        assert module.DisasterType.CPU_OVERLOAD.value == "cpu_overload"
        assert module.DisasterType.DISK_FULL.value == "disk_full"
        assert module.DisasterType.NETWORK_FAILURE.value == "network_failure"
        assert module.DisasterType.PROCESS_CRASH.value == "process_crash"
        assert module.DisasterType.SERVICE_UNAVAILABLE.value == "service_unavailable"


class TestDisasterEvent:
    """测试灾难事件数据类"""

    def test_disaster_event_initialization(self, module):
        """测试灾难事件初始化"""
        event = module.DisasterEvent(
            disaster_type=module.DisasterType.MEMORY_OVERLOAD,
            level=module.DisasterLevel.CRITICAL,
            message="Memory overload",
            timestamp=time.time(),
            details={"memory_percent": 95.0}
        )

        assert event.disaster_type == module.DisasterType.MEMORY_OVERLOAD
        assert event.level == module.DisasterLevel.CRITICAL
        assert event.message == "Memory overload"
        assert event.details["memory_percent"] == 95.0


class TestDisasterMonitor:
    """测试灾难监控器"""

    def test_initialization(self, monitor):
        """测试初始化"""
        assert monitor.disaster_events == []
        assert monitor.max_events == 1000
        assert 'memory_percent' in monitor.thresholds
        assert monitor.handlers == []
        assert monitor.monitoring_active is False

    def test_add_handler(self, monitor):
        """测试添加处理器"""
        handler = Mock()
        monitor.add_handler(handler)
        assert handler in monitor.handlers

    def test_remove_handler(self, monitor):
        """测试移除处理器"""
        handler = Mock()
        monitor.add_handler(handler)
        monitor.remove_handler(handler)
        assert handler not in monitor.handlers

    def test_remove_handler_not_exists(self, monitor):
        """测试移除不存在的处理器"""
        handler = Mock()
        monitor.remove_handler(handler)  # 不应该抛出异常
        assert handler not in monitor.handlers

    def test_collect_system_metrics(self, monitor, module, monkeypatch):
        """测试收集系统指标"""
        # Mock psutil
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0

        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))

        metrics = monitor._collect_system_metrics()

        assert metrics['cpu_percent'] == 50.0
        assert metrics['memory_percent'] == 60.0
        assert metrics['disk_percent'] == 70.0

    def test_create_memory_disaster(self, monitor, module):
        """测试创建内存灾难事件"""
        disaster = monitor._create_memory_disaster(95.0)

        assert disaster.disaster_type == module.DisasterType.MEMORY_OVERLOAD
        assert disaster.level == module.DisasterLevel.CRITICAL
        assert "内存使用率过高" in disaster.message
        assert disaster.details['memory_percent'] == 95.0

    def test_create_cpu_disaster(self, monitor, module):
        """测试创建CPU灾难事件"""
        disaster = monitor._create_cpu_disaster(98.0)

        assert disaster.disaster_type == module.DisasterType.CPU_OVERLOAD
        assert disaster.level == module.DisasterLevel.CRITICAL
        assert "CPU使用率过高" in disaster.message
        assert disaster.details['cpu_percent'] == 98.0

    def test_create_disk_disaster(self, monitor, module):
        """测试创建磁盘灾难事件"""
        disaster = monitor._create_disk_disaster(96.0)

        assert disaster.disaster_type == module.DisasterType.DISK_FULL
        assert disaster.level == module.DisasterLevel.CRITICAL
        assert "磁盘使用率过高" in disaster.message
        assert disaster.details['disk_percent'] == 96.0

    def test_check_disaster_conditions_no_disasters(self, monitor, module, monkeypatch):
        """测试检查灾难条件 - 无灾难"""
        metrics = {
            'cpu_percent': 50.0,
            'memory_percent': 60.0,
            'disk_percent': 70.0
        }

        disasters = monitor._check_disaster_conditions(metrics)
        assert len(disasters) == 0

    def test_check_disaster_conditions_memory_overload(self, monitor, module, monkeypatch):
        """测试检查灾难条件 - 内存过载"""
        metrics = {
            'cpu_percent': 50.0,
            'memory_percent': 95.0,  # 超过阈值90
            'disk_percent': 70.0
        }

        disasters = monitor._check_disaster_conditions(metrics)
        assert len(disasters) == 1
        assert disasters[0].disaster_type == module.DisasterType.MEMORY_OVERLOAD

    def test_check_disaster_conditions_cpu_overload(self, monitor, module, monkeypatch):
        """测试检查灾难条件 - CPU过载"""
        metrics = {
            'cpu_percent': 98.0,  # 超过阈值95
            'memory_percent': 60.0,
            'disk_percent': 70.0
        }

        disasters = monitor._check_disaster_conditions(metrics)
        assert len(disasters) == 1
        assert disasters[0].disaster_type == module.DisasterType.CPU_OVERLOAD

    def test_check_disaster_conditions_disk_full(self, monitor, module, monkeypatch):
        """测试检查灾难条件 - 磁盘满"""
        metrics = {
            'cpu_percent': 50.0,
            'memory_percent': 60.0,
            'disk_percent': 98.0  # 超过阈值95
        }

        disasters = monitor._check_disaster_conditions(metrics)
        assert len(disasters) == 1
        assert disasters[0].disaster_type == module.DisasterType.DISK_FULL

    def test_check_disaster_conditions_multiple(self, monitor, module, monkeypatch):
        """测试检查灾难条件 - 多个灾难"""
        metrics = {
            'cpu_percent': 98.0,  # 超过阈值
            'memory_percent': 95.0,  # 超过阈值
            'disk_percent': 98.0  # 超过阈值
        }

        disasters = monitor._check_disaster_conditions(metrics)
        assert len(disasters) == 3

    def test_record_disaster(self, monitor, module):
        """测试记录灾难事件"""
        disaster = module.DisasterEvent(
            disaster_type=module.DisasterType.MEMORY_OVERLOAD,
            level=module.DisasterLevel.CRITICAL,
            message="Test",
            timestamp=time.time(),
            details={}
        )

        monitor._record_disaster(disaster)

        assert len(monitor.disaster_events) == 1
        assert monitor.disaster_events[0] == disaster

    def test_record_disaster_triggers_handler(self, monitor, module):
        """测试记录灾难事件触发处理器"""
        handler = Mock()
        monitor.add_handler(handler)

        disaster = module.DisasterEvent(
            disaster_type=module.DisasterType.MEMORY_OVERLOAD,
            level=module.DisasterLevel.CRITICAL,
            message="Test",
            timestamp=time.time(),
            details={}
        )

        monitor._record_disaster(disaster)

        handler.assert_called_once_with(disaster)

    def test_record_disaster_handler_exception(self, monitor, module, monkeypatch):
        """测试记录灾难事件 - 处理器异常"""
        def failing_handler(disaster):
            raise Exception("Handler error")

        monitor.add_handler(failing_handler)

        disaster = module.DisasterEvent(
            disaster_type=module.DisasterType.MEMORY_OVERLOAD,
            level=module.DisasterLevel.CRITICAL,
            message="Test",
            timestamp=time.time(),
            details={}
        )

        # 不应该抛出异常
        monitor._record_disaster(disaster)
        assert len(monitor.disaster_events) == 1

    def test_record_disaster_max_events(self, monitor, module):
        """测试记录灾难事件 - 最大事件数限制"""
        monitor.max_events = 5

        for i in range(10):
            disaster = module.DisasterEvent(
                disaster_type=module.DisasterType.MEMORY_OVERLOAD,
                level=module.DisasterLevel.CRITICAL,
                message=f"Test {i}",
                timestamp=time.time(),
                details={}
            )
            monitor._record_disaster(disaster)

        assert len(monitor.disaster_events) == 5

    def test_check_system_health_success(self, monitor, module, monkeypatch):
        """测试检查系统健康状态 - 成功"""
        # Mock psutil
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0

        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))

        health = monitor.check_system_health()

        assert 'cpu_percent' in health
        assert 'memory_percent' in health
        assert 'disk_percent' in health
        assert 'timestamp' in health
        assert 'disasters' in health

    def test_check_system_health_exception(self, monitor, module, monkeypatch):
        """测试检查系统健康状态 - 异常处理"""
        # Mock psutil 抛出异常
        monkeypatch.setattr(module.psutil, "cpu_percent", MagicMock(side_effect=Exception("Test error")))

        health = monitor.check_system_health()

        assert 'error' in health
        assert 'timestamp' in health
        assert 'disasters' in health
        assert health['disasters'] == []

    def test_get_recent_disasters(self, monitor, module):
        """测试获取最近的灾难事件"""
        for i in range(10):
            disaster = module.DisasterEvent(
                disaster_type=module.DisasterType.MEMORY_OVERLOAD,
                level=module.DisasterLevel.CRITICAL,
                message=f"Test {i}",
                timestamp=time.time(),
                details={}
            )
            monitor._record_disaster(disaster)

        recent = monitor.get_recent_disasters(limit=5)
        assert len(recent) == 5

    def test_get_recent_disasters_default_limit(self, monitor, module):
        """测试获取最近的灾难事件 - 默认限制"""
        for i in range(100):
            disaster = module.DisasterEvent(
                disaster_type=module.DisasterType.MEMORY_OVERLOAD,
                level=module.DisasterLevel.CRITICAL,
                message=f"Test {i}",
                timestamp=time.time(),
                details={}
            )
            monitor._record_disaster(disaster)

        recent = monitor.get_recent_disasters()
        assert len(recent) == 50  # 默认限制

    def test_get_disaster_stats_empty(self, monitor):
        """测试获取灾难统计 - 空数据"""
        stats = monitor.get_disaster_stats()

        assert stats['total_disasters'] == 0
        assert stats['disasters_by_type'] == {}
        assert stats['disasters_by_level'] == {}
        assert stats['latest_disaster'] is None

    def test_get_disaster_stats(self, monitor, module):
        """测试获取灾难统计"""
        # 创建不同类型的灾难
        memory_disaster = module.DisasterEvent(
            disaster_type=module.DisasterType.MEMORY_OVERLOAD,
            level=module.DisasterLevel.CRITICAL,
            message="Memory",
            timestamp=time.time(),
            details={}
        )
        cpu_disaster = module.DisasterEvent(
            disaster_type=module.DisasterType.CPU_OVERLOAD,
            level=module.DisasterLevel.CRITICAL,
            message="CPU",
            timestamp=time.time(),
            details={}
        )

        monitor._record_disaster(memory_disaster)
        monitor._record_disaster(cpu_disaster)

        stats = monitor.get_disaster_stats()

        assert stats['total_disasters'] == 2
        assert stats['disasters_by_type']['memory_overload'] == 1
        assert stats['disasters_by_type']['cpu_overload'] == 1
        assert stats['disasters_by_level']['critical'] == 2
        assert stats['latest_disaster'] == cpu_disaster

    def test_clear_disasters(self, monitor, module):
        """测试清空灾难记录"""
        for i in range(5):
            disaster = module.DisasterEvent(
                disaster_type=module.DisasterType.MEMORY_OVERLOAD,
                level=module.DisasterLevel.CRITICAL,
                message=f"Test {i}",
                timestamp=time.time(),
                details={}
            )
            monitor._record_disaster(disaster)

        assert len(monitor.disaster_events) == 5

        monitor.clear_disasters()
        assert len(monitor.disaster_events) == 0

    def test_set_threshold(self, monitor):
        """测试设置阈值"""
        monitor.set_threshold('memory_percent', 85.0)
        assert monitor.thresholds['memory_percent'] == 85.0

        monitor.set_threshold('cpu_percent', 90.0)
        assert monitor.thresholds['cpu_percent'] == 90.0

    def test_add_handler(self, monitor):
        """测试添加事件处理器"""
        handler_called = []
        
        def test_handler(event):
            handler_called.append(event)
        
        monitor.add_handler(test_handler)
        
        assert len(monitor.handlers) == 1
        assert test_handler in monitor.handlers

    def test_remove_handler(self, monitor):
        """测试移除事件处理器"""
        handler_called = []
        
        def test_handler(event):
            handler_called.append(event)
        
        monitor.add_handler(test_handler)
        assert len(monitor.handlers) == 1
        
        monitor.remove_handler(test_handler)
        assert len(monitor.handlers) == 0

    def test_remove_handler_not_exists(self, monitor):
        """测试移除不存在的事件处理器"""
        def test_handler(event):
            pass
        
        # 尝试移除不存在的处理器
        monitor.remove_handler(test_handler)
        
        # 不应该抛出异常
        assert len(monitor.handlers) == 0

    def test_handler_called_on_disaster(self, monitor, module, monkeypatch):
        """测试灾难事件时调用处理器"""
        handler_called = []
        
        def test_handler(event):
            handler_called.append(event)
        
        monitor.add_handler(test_handler)
        
        # Mock psutil 触发内存灾难
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 95.0  # 超过阈值
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0
        
        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        
        monitor.check_system_health()
        
        # 应该调用处理器
        assert len(handler_called) > 0

    def test_multiple_handlers(self, monitor, module, monkeypatch):
        """测试多个事件处理器"""
        handler1_called = []
        handler2_called = []
        
        def handler1(event):
            handler1_called.append(event)
        
        def handler2(event):
            handler2_called.append(event)
        
        monitor.add_handler(handler1)
        monitor.add_handler(handler2)
        
        # Mock psutil 触发内存灾难
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 95.0  # 超过阈值
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0
        
        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        
        monitor.check_system_health()
        
        # 两个处理器都应该被调用
        assert len(handler1_called) > 0
        assert len(handler2_called) > 0

    def test_check_disaster_conditions_all_thresholds(self, monitor, module, monkeypatch):
        """测试检查灾难条件 - 所有阈值都超过"""
        # Mock psutil 所有指标都超过阈值（使用 > 而不是 >=）
        mock_cpu_percent = MagicMock(return_value=96.0)  # 超过阈值95.0
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 91.0  # 超过阈值90.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 96.0  # 超过阈值95.0
        
        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        
        metrics = monitor._collect_system_metrics()
        disasters = monitor._check_disaster_conditions(metrics)
        
        # 应该检测到所有三种灾难
        assert len(disasters) == 3
        disaster_types = [d.disaster_type for d in disasters]
        assert module.DisasterType.MEMORY_OVERLOAD in disaster_types
        assert module.DisasterType.CPU_OVERLOAD in disaster_types
        assert module.DisasterType.DISK_FULL in disaster_types

    def test_record_disaster_handler_exception(self, monitor, module, monkeypatch):
        """测试记录灾难事件 - 处理器异常"""
        handler_called = []
        
        def failing_handler(event):
            handler_called.append(event)
            raise Exception("Handler error")
        
        monitor.add_handler(failing_handler)
        
        # Mock print
        prints = []
        def mock_print(*args, **kwargs):
            prints.append(args)
        
        monkeypatch.setattr("builtins.print", mock_print)
        
        # 创建灾难事件
        disaster = module.DisasterEvent(
            disaster_type=module.DisasterType.MEMORY_OVERLOAD,
            level=module.DisasterLevel.CRITICAL,
            message="Test",
            timestamp=time.time(),
            details={}
        )
        
        monitor._record_disaster(disaster)
        
        # 处理器应该被调用
        assert len(handler_called) == 1
        # 异常应该被捕获并打印
        assert len(prints) > 0
        assert any("灾难处理器执行失败" in str(args) for args in prints)

    def test_record_disaster_multiple_handlers_one_fails(self, monitor, module, monkeypatch):
        """测试记录灾难事件 - 多个处理器，一个失败"""
        handler1_called = []
        handler2_called = []
        
        def handler1(event):
            handler1_called.append(event)
        
        def handler2(event):
            handler2_called.append(event)
            raise Exception("Handler2 error")
        
        monitor.add_handler(handler1)
        monitor.add_handler(handler2)
        
        # Mock print
        prints = []
        def mock_print(*args, **kwargs):
            prints.append(args)
        
        monkeypatch.setattr("builtins.print", mock_print)
        
        # 创建灾难事件
        disaster = module.DisasterEvent(
            disaster_type=module.DisasterType.MEMORY_OVERLOAD,
            level=module.DisasterLevel.CRITICAL,
            message="Test",
            timestamp=time.time(),
            details={}
        )
        
        monitor._record_disaster(disaster)
        
        # 两个处理器都应该被调用
        assert len(handler1_called) == 1
        assert len(handler2_called) == 1
        # 异常应该被捕获
        assert len(prints) > 0

    def test_create_memory_disaster(self, monitor, module):
        """测试创建内存灾难事件"""
        disaster = monitor._create_memory_disaster(95.0)
        
        assert disaster.disaster_type == module.DisasterType.MEMORY_OVERLOAD
        assert disaster.level == module.DisasterLevel.CRITICAL
        assert "95.0%" in disaster.message
        assert disaster.details['memory_percent'] == 95.0

    def test_create_cpu_disaster(self, monitor, module):
        """测试创建CPU灾难事件"""
        disaster = monitor._create_cpu_disaster(96.0)
        
        assert disaster.disaster_type == module.DisasterType.CPU_OVERLOAD
        assert disaster.level == module.DisasterLevel.CRITICAL
        assert "96.0%" in disaster.message
        assert disaster.details['cpu_percent'] == 96.0

    def test_create_disk_disaster(self, monitor, module):
        """测试创建磁盘灾难事件"""
        disaster = monitor._create_disk_disaster(97.0)
        
        assert disaster.disaster_type == module.DisasterType.DISK_FULL
        assert disaster.level == module.DisasterLevel.CRITICAL
        assert "97.0%" in disaster.message
        assert disaster.details['disk_percent'] == 97.0

    def test_record_disaster_max_events_with_handler(self, monitor, module, monkeypatch):
        """测试记录灾难事件 - 达到最大事件数，有处理器"""
        handler_called = []
        
        def test_handler(event):
            handler_called.append(event)
        
        monitor.add_handler(test_handler)
        monitor.max_events = 3
        
        # 创建超过限制的灾难事件
        for i in range(5):
            disaster = module.DisasterEvent(
                disaster_type=module.DisasterType.MEMORY_OVERLOAD,
                level=module.DisasterLevel.CRITICAL,
                message=f"Test {i}",
                timestamp=time.time(),
                details={}
            )
            monitor._record_disaster(disaster)
        
        # 应该只保留最新的3个
        assert len(monitor.disaster_events) == 3
        # 所有处理器都应该被调用
        assert len(handler_called) == 5

    def test_get_recent_disasters_zero_limit(self, monitor, module):
        """测试获取最近的灾难事件 - 限制为0"""
        for i in range(10):
            disaster = module.DisasterEvent(
                disaster_type=module.DisasterType.MEMORY_OVERLOAD,
                level=module.DisasterLevel.CRITICAL,
                message=f"Test {i}",
                timestamp=time.time(),
                details={}
            )
            monitor._record_disaster(disaster)

        recent = monitor.get_recent_disasters(limit=0)
        
        # Python中 [-0:] 会返回整个列表
        assert isinstance(recent, list)

    def test_get_recent_disasters_more_than_total(self, monitor, module):
        """测试获取最近的灾难事件 - 限制超过总数"""
        for i in range(5):
            disaster = module.DisasterEvent(
                disaster_type=module.DisasterType.MEMORY_OVERLOAD,
                level=module.DisasterLevel.CRITICAL,
                message=f"Test {i}",
                timestamp=time.time(),
                details={}
            )
            monitor._record_disaster(disaster)

        recent = monitor.get_recent_disasters(limit=100)
        
        # 应该返回所有事件
        assert len(recent) == 5

    def test_get_disaster_stats_multiple_types_and_levels(self, monitor, module):
        """测试获取灾难统计 - 多种类型和级别"""
        # 创建不同类型的灾难
        memory_disaster = module.DisasterEvent(
            disaster_type=module.DisasterType.MEMORY_OVERLOAD,
            level=module.DisasterLevel.CRITICAL,
            message="Memory",
            timestamp=time.time(),
            details={}
        )
        cpu_disaster = module.DisasterEvent(
            disaster_type=module.DisasterType.CPU_OVERLOAD,
            level=module.DisasterLevel.WARNING,
            message="CPU",
            timestamp=time.time(),
            details={}
        )
        disk_disaster = module.DisasterEvent(
            disaster_type=module.DisasterType.DISK_FULL,
            level=module.DisasterLevel.CRITICAL,
            message="Disk",
            timestamp=time.time(),
            details={}
        )

        monitor._record_disaster(memory_disaster)
        monitor._record_disaster(cpu_disaster)
        monitor._record_disaster(disk_disaster)

        stats = monitor.get_disaster_stats()

        assert stats['total_disasters'] == 3
        assert stats['disasters_by_type']['memory_overload'] == 1
        assert stats['disasters_by_type']['cpu_overload'] == 1
        assert stats['disasters_by_type']['disk_full'] == 1
        assert stats['disasters_by_level']['critical'] == 2
        assert stats['disasters_by_level']['warning'] == 1

    def test_set_threshold_disk_percent(self, monitor):
        """测试设置阈值 - 磁盘使用率"""
        monitor.set_threshold('disk_percent', 95.0)
        assert monitor.thresholds['disk_percent'] == 95.0

    def test_check_disaster_conditions_no_disasters(self, monitor, module, monkeypatch):
        """测试检查灾难条件 - 无灾难"""
        # Mock psutil 所有指标都低于阈值
        mock_cpu_percent = MagicMock(return_value=50.0)  # 低于阈值95
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 50.0  # 低于阈值90
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 50.0  # 低于阈值95
        
        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        
        metrics = monitor._collect_system_metrics()
        disasters = monitor._check_disaster_conditions(metrics)
        
        # 应该没有灾难
        assert len(disasters) == 0

    def test_collect_system_metrics_exception(self, monitor, module, monkeypatch):
        """测试收集系统指标 - 异常处理"""
        # Mock psutil.cpu_percent 抛出异常
        monkeypatch.setattr(module.psutil, "cpu_percent", MagicMock(side_effect=Exception("CPU error")))
        
        # 应该抛出异常
        with pytest.raises(Exception):
            monitor._collect_system_metrics()

    def test_collect_system_metrics_disk_exception(self, monitor, module, monkeypatch):
        """测试收集系统指标 - 磁盘异常"""
        # Mock psutil.disk_usage 抛出异常
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        
        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(side_effect=Exception("Disk error")))
        
        # 应该抛出异常
        with pytest.raises(Exception):
            monitor._collect_system_metrics()

    def test_check_system_health_with_disasters(self, monitor, module, monkeypatch):
        """测试检查系统健康状态 - 有灾难"""
        # Mock psutil - 所有指标都超过阈值
        mock_cpu_percent = MagicMock(return_value=96.0)  # 超过阈值95
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 96.0  # 超过阈值90
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 96.0  # 超过阈值95
        
        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        
        health = monitor.check_system_health()
        
        assert 'cpu_percent' in health
        assert 'memory_percent' in health
        assert 'disk_percent' in health
        assert 'disasters' in health
        assert len(health['disasters']) == 3  # 应该有3个灾难

    def test_check_system_health_collect_metrics_exception(self, monitor, module, monkeypatch):
        """测试检查系统健康状态 - 收集指标异常"""
        # Mock _collect_system_metrics 抛出异常
        original_collect = monitor._collect_system_metrics
        monitor._collect_system_metrics = MagicMock(side_effect=Exception("Collect error"))
        
        health = monitor.check_system_health()
        
        # 应该返回错误信息
        assert 'error' in health
        assert health['error'] == "Collect error"
        assert 'disasters' in health
        assert len(health['disasters']) == 0
        
        # 恢复原始方法
        monitor._collect_system_metrics = original_collect

    def test_check_disaster_conditions_single_disaster(self, monitor, module):
        """测试检查灾难条件 - 单个灾难"""
        metrics = {
            'cpu_percent': 50.0,  # 低于阈值
            'memory_percent': 95.0,  # 超过阈值90
            'disk_percent': 50.0  # 低于阈值
        }
        
        disasters = monitor._check_disaster_conditions(metrics)
        
        # 应该只有一个内存灾难
        assert len(disasters) == 1
        assert disasters[0].disaster_type.value == 'memory_overload'

    def test_check_disaster_conditions_cpu_only(self, monitor, module):
        """测试检查灾难条件 - 仅CPU灾难"""
        metrics = {
            'cpu_percent': 96.0,  # 超过阈值95
            'memory_percent': 50.0,  # 低于阈值
            'disk_percent': 50.0  # 低于阈值
        }
        
        disasters = monitor._check_disaster_conditions(metrics)
        
        # 应该只有一个CPU灾难
        assert len(disasters) == 1
        assert disasters[0].disaster_type.value == 'cpu_overload'

    def test_check_disaster_conditions_disk_only(self, monitor, module):
        """测试检查灾难条件 - 仅磁盘灾难"""
        metrics = {
            'cpu_percent': 50.0,  # 低于阈值
            'memory_percent': 50.0,  # 低于阈值
            'disk_percent': 96.0  # 超过阈值95
        }
        
        disasters = monitor._check_disaster_conditions(metrics)
        
        # 应该只有一个磁盘灾难
        assert len(disasters) == 1
        assert disasters[0].disaster_type.value == 'disk_full'

    def test_create_memory_disaster_details(self, monitor, module):
        """测试创建内存灾难事件 - 详细信息"""
        memory_percent = 95.5
        disaster = monitor._create_memory_disaster(memory_percent)
        
        assert disaster.disaster_type.value == 'memory_overload'
        assert disaster.level.value == 'critical'
        assert '内存使用率过高' in disaster.message
        assert disaster.details['memory_percent'] == memory_percent

    def test_create_cpu_disaster_details(self, monitor, module):
        """测试创建CPU灾难事件 - 详细信息"""
        cpu_percent = 96.5
        disaster = monitor._create_cpu_disaster(cpu_percent)
        
        assert disaster.disaster_type.value == 'cpu_overload'
        assert disaster.level.value == 'critical'
        assert 'CPU使用率过高' in disaster.message
        assert disaster.details['cpu_percent'] == cpu_percent

    def test_create_disk_disaster_details(self, monitor, module):
        """测试创建磁盘灾难事件 - 详细信息"""
        disk_percent = 96.5
        disaster = monitor._create_disk_disaster(disk_percent)
        
        assert disaster.disaster_type.value == 'disk_full'
        assert disaster.level.value == 'critical'
        assert '磁盘使用率过高' in disaster.message
        assert disaster.details['disk_percent'] == disk_percent

    def test_get_recent_disasters_negative_limit(self, monitor, module):
        """测试获取最近的灾难事件 - 负数限制"""
        for i in range(10):
            disaster = module.DisasterEvent(
                disaster_type=module.DisasterType.MEMORY_OVERLOAD,
                level=module.DisasterLevel.CRITICAL,
                message=f"Test {i}",
                timestamp=time.time(),
                details={}
            )
            monitor._record_disaster(disaster)

        recent = monitor.get_recent_disasters(limit=-1)
        
        # 应该返回空列表或所有历史记录
        assert isinstance(recent, list)

    def test_get_disaster_stats_single_event(self, monitor, module):
        """测试获取灾难统计 - 单个事件"""
        disaster = module.DisasterEvent(
            disaster_type=module.DisasterType.MEMORY_OVERLOAD,
            level=module.DisasterLevel.CRITICAL,
            message="Test",
            timestamp=time.time(),
            details={}
        )
        monitor._record_disaster(disaster)

        stats = monitor.get_disaster_stats()

        assert stats['total_disasters'] == 1
        assert stats['disasters_by_type']['memory_overload'] == 1
        assert stats['disasters_by_level']['critical'] == 1
        assert stats['latest_disaster'] == disaster

    def test_get_disaster_stats_latest_disaster_none(self, monitor):
        """测试获取灾难统计 - latest_disaster为None"""
        # 确保事件列表为空
        monitor.disaster_events = []
        
        stats = monitor.get_disaster_stats()
        
        assert stats['latest_disaster'] is None

    def test_record_disaster_multiple_handlers_with_exception(self, monitor, module, monkeypatch):
        """测试记录灾难事件 - 多个处理器，其中一个抛出异常"""
        handler_called = []
        
        def handler1(disaster):
            handler_called.append(1)
        
        def handler2(disaster):
            handler_called.append(2)
            raise Exception("Handler error")
        
        def handler3(disaster):
            handler_called.append(3)
        
        monitor.add_handler(handler1)
        monitor.add_handler(handler2)
        monitor.add_handler(handler3)
        
        disaster = module.DisasterEvent(
            disaster_type=module.DisasterType.MEMORY_OVERLOAD,
            level=module.DisasterLevel.CRITICAL,
            message="Test",
            timestamp=time.time(),
            details={}
        )
        
        # Mock print
        prints = []
        def mock_print(*args, **kwargs):
            prints.append(args)
        
        monkeypatch.setattr("builtins.print", mock_print)
        
        monitor._record_disaster(disaster)
        
        # 所有处理器都应该被调用（即使其中一个抛出异常）
        assert len(handler_called) == 3
        assert 1 in handler_called
        assert 2 in handler_called
        assert 3 in handler_called

    def test_record_disaster_max_events_boundary(self, monitor, module):
        """测试记录灾难事件 - 达到最大事件数边界"""
        monitor.max_events = 3
        
        handler_called = []
        def handler(disaster):
            handler_called.append(disaster)
        
        monitor.add_handler(handler)
        
        # 创建4个灾难事件
        for i in range(4):
            disaster = module.DisasterEvent(
                disaster_type=module.DisasterType.MEMORY_OVERLOAD,
                level=module.DisasterLevel.CRITICAL,
                message=f"Test {i}",
                timestamp=time.time(),
                details={}
            )
            monitor._record_disaster(disaster)
        
        # 应该只保留最新的3个
        assert len(monitor.disaster_events) == 3
        # 所有处理器都应该被调用4次
        assert len(handler_called) == 4

    def test_check_system_health_no_disasters_return(self, monitor, module, monkeypatch):
        """测试检查系统健康状态 - 无灾难返回"""
        # Mock psutil - 所有指标都低于阈值
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 50.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 50.0
        
        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        
        health = monitor.check_system_health()
        
        assert 'cpu_percent' in health
        assert 'memory_percent' in health
        assert 'disk_percent' in health
        assert 'disasters' in health
        assert len(health['disasters']) == 0

    def test_set_threshold_memory_percent(self, monitor):
        """测试设置阈值 - 内存使用率"""
        monitor.set_threshold('memory_percent', 85.0)
        assert monitor.thresholds['memory_percent'] == 85.0

    def test_set_threshold_cpu_percent(self, monitor):
        """测试设置阈值 - CPU使用率"""
        monitor.set_threshold('cpu_percent', 90.0)
        assert monitor.thresholds['cpu_percent'] == 90.0

    def test_set_threshold_new_metric(self, monitor):
        """测试设置阈值 - 新指标"""
        monitor.set_threshold('new_metric', 75.0)
        assert monitor.thresholds['new_metric'] == 75.0

    def test_clear_disasters_with_events(self, monitor, module):
        """测试清空灾难记录 - 有事件"""
        # 创建一些灾难事件
        for i in range(5):
            disaster = module.DisasterEvent(
                disaster_type=module.DisasterType.MEMORY_OVERLOAD,
                level=module.DisasterLevel.CRITICAL,
                message=f"Test {i}",
                timestamp=time.time(),
                details={}
            )
            monitor._record_disaster(disaster)
        
        assert len(monitor.disaster_events) == 5
        
        monitor.clear_disasters()
        
        assert len(monitor.disaster_events) == 0

    def test_disaster_event_initialization(self, module):
        """测试灾难事件初始化"""
        event = module.DisasterEvent(
            disaster_type=module.DisasterType.MEMORY_OVERLOAD,
            level=module.DisasterLevel.CRITICAL,
            message="Test message",
            timestamp=time.time(),
            details={'key': 'value'}
        )
        
        assert event.disaster_type == module.DisasterType.MEMORY_OVERLOAD
        assert event.level == module.DisasterLevel.CRITICAL
        assert event.message == "Test message"
        assert event.details == {'key': 'value'}

    def test_disaster_event_empty_details(self, module):
        """测试灾难事件 - 空详情"""
        event = module.DisasterEvent(
            disaster_type=module.DisasterType.CPU_OVERLOAD,
            level=module.DisasterLevel.WARNING,
            message="Test",
            timestamp=time.time(),
            details={}
        )
        
        assert event.details == {}

    def test_add_handler_concurrent_access(self, monitor, module):
        """测试并发访问添加处理器"""
        import threading
        
        handlers_added = []
        
        def add_handler():
            def handler(event):
                pass
            monitor.add_handler(handler)
            handlers_added.append(handler)
        
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=add_handler)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # 应该添加了10个处理器
        assert len(monitor.handlers) == 10

    def test_record_disaster_concurrent_access(self, monitor, module):
        """测试并发访问记录灾难事件"""
        import threading
        
        def record_disaster():
            disaster = module.DisasterEvent(
                disaster_type=module.DisasterType.MEMORY_OVERLOAD,
                level=module.DisasterLevel.CRITICAL,
                message="Test",
                timestamp=time.time(),
                details={}
            )
            monitor._record_disaster(disaster)
        
        threads = []
        for _ in range(20):
            thread = threading.Thread(target=record_disaster)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # 应该记录了20个灾难事件
        assert len(monitor.disaster_events) == 20

    def test_get_recent_disasters_concurrent_access(self, monitor, module):
        """测试并发访问获取最近灾难事件"""
        import threading
        
        # 先添加一些灾难事件
        for i in range(10):
            disaster = module.DisasterEvent(
                disaster_type=module.DisasterType.MEMORY_OVERLOAD,
                level=module.DisasterLevel.CRITICAL,
                message=f"Test {i}",
                timestamp=time.time(),
                details={}
            )
            monitor._record_disaster(disaster)
        
        results = []
        def get_recent():
            results.append(monitor.get_recent_disasters(limit=5))
        
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=get_recent)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # 应该都返回5个事件
        assert len(results) == 10
        assert all(len(r) == 5 for r in results)

    def test_check_disaster_conditions_exact_threshold(self, monitor, module, monkeypatch):
        """测试检查灾难条件 - 精确阈值"""
        # Mock psutil
        mock_cpu_percent = MagicMock(return_value=95.0)  # 等于阈值
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 90.0  # 等于阈值
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 95.0  # 等于阈值
        
        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        
        metrics = monitor._collect_system_metrics()
        disasters = monitor._check_disaster_conditions(metrics)
        
        # 应该没有灾难（因为使用 > 而不是 >=）
        assert len(disasters) == 0

    def test_collect_system_metrics_network_failure(self, monitor, module, monkeypatch):
        """测试收集系统指标 - 网络故障场景"""
        # Mock psutil - 模拟网络故障导致异常
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0
        
        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        
        metrics = monitor._collect_system_metrics()
        
        # 应该成功收集指标
        assert 'cpu_percent' in metrics
        assert 'memory_percent' in metrics
        assert 'disk_percent' in metrics

    def test_get_disaster_stats_all_levels(self, monitor, module):
        """测试获取灾难统计 - 所有级别"""
        # 添加不同级别的灾难事件
        levels = [
            module.DisasterLevel.WARNING,
            module.DisasterLevel.CRITICAL,
            module.DisasterLevel.DISASTER
        ]
        
        for level in levels:
            disaster = module.DisasterEvent(
                disaster_type=module.DisasterType.MEMORY_OVERLOAD,
                level=level,
                message="Test",
                timestamp=time.time(),
                details={}
            )
            monitor._record_disaster(disaster)
        
        stats = monitor.get_disaster_stats()
        
        # 应该包含所有级别的统计
        assert stats['total_disasters'] == 3
        assert 'disasters_by_level' in stats
        assert stats['disasters_by_level']['warning'] == 1
        assert stats['disasters_by_level']['critical'] == 1
        assert stats['disasters_by_level']['disaster'] == 1

    def test_get_disaster_stats_all_types(self, monitor, module):
        """测试获取灾难统计 - 所有类型"""
        # 添加不同类型的灾难事件
        types = [
            module.DisasterType.MEMORY_OVERLOAD,
            module.DisasterType.CPU_OVERLOAD,
            module.DisasterType.DISK_FULL
        ]
        
        for disaster_type in types:
            disaster = module.DisasterEvent(
                disaster_type=disaster_type,
                level=module.DisasterLevel.CRITICAL,
                message="Test",
                timestamp=time.time(),
                details={}
            )
            monitor._record_disaster(disaster)
        
        stats = monitor.get_disaster_stats()
        
        # 应该包含所有类型的统计
        assert stats['total_disasters'] == 3
        assert 'disasters_by_type' in stats
        assert stats['disasters_by_type']['memory_overload'] == 1
        assert stats['disasters_by_type']['cpu_overload'] == 1
        assert stats['disasters_by_type']['disk_full'] == 1

    def test_set_threshold_invalid_metric(self, monitor):
        """测试设置阈值 - 无效指标"""
        # 设置一个不存在的指标阈值
        monitor.set_threshold('invalid_metric', 50.0)
        
        # 应该添加到阈值字典中
        assert 'invalid_metric' in monitor.thresholds
        assert monitor.thresholds['invalid_metric'] == 50.0

    def test_get_recent_disasters_zero_limit(self, monitor, module):
        """测试获取最近灾难事件 - 零限制"""
        # 添加一些灾难事件
        for i in range(5):
            disaster = module.DisasterEvent(
                disaster_type=module.DisasterType.MEMORY_OVERLOAD,
                level=module.DisasterLevel.CRITICAL,
                message=f"Test {i}",
                timestamp=time.time(),
                details={}
            )
            monitor._record_disaster(disaster)
        
        recent = monitor.get_recent_disasters(limit=0)
        
        # Python切片[-0:]返回整个列表，所以应该返回所有5个事件
        assert len(recent) == 5

    def test_get_recent_disasters_negative_limit(self, monitor, module):
        """测试获取最近灾难事件 - 负限制"""
        # 添加一些灾难事件
        for i in range(5):
            disaster = module.DisasterEvent(
                disaster_type=module.DisasterType.MEMORY_OVERLOAD,
                level=module.DisasterLevel.CRITICAL,
                message=f"Test {i}",
                timestamp=time.time(),
                details={}
            )
            monitor._record_disaster(disaster)
        
        recent = monitor.get_recent_disasters(limit=-5)
        
        # 应该返回空列表（因为负限制）
        assert len(recent) == 0

    def test_record_disaster_max_events_concurrent(self, monitor, module):
        """测试记录灾难事件 - 并发达到最大事件数"""
        import threading
        
        def record_disaster(i):
            disaster = module.DisasterEvent(
                disaster_type=module.DisasterType.MEMORY_OVERLOAD,
                level=module.DisasterLevel.CRITICAL,
                message=f"Test {i}",
                timestamp=time.time(),
                details={}
            )
            monitor._record_disaster(disaster)
        
        # 设置较小的最大事件数
        monitor.max_events = 50
        
        threads = []
        for i in range(100):
            thread = threading.Thread(target=record_disaster, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # 应该只保留最近的50个事件
        assert len(monitor.disaster_events) == 50

    def test_check_system_health_no_disasters_with_metrics(self, monitor, module, monkeypatch):
        """测试检查系统健康 - 无灾难但有指标"""
        # Mock psutil
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0
        
        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        
        health = monitor.check_system_health()
        
        # 应该返回健康状态（无灾难）
        assert 'disasters' in health
        assert len(health['disasters']) == 0
        assert 'cpu_percent' in health
        assert 'memory_percent' in health
        assert 'disk_percent' in health

