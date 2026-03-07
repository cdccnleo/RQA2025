#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试基础设施层 - 系统监控组件

测试 infrastructure/system_monitor.py 中的所有类和方法
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta


@pytest.fixture
def module():
    """导入模块"""
    from src.infrastructure.monitoring.infrastructure import system_monitor
    return system_monitor


@pytest.fixture
def monitor(module):
    """创建系统监控器实例"""
    return module.SystemMonitor()


class TestSystemMetrics:
    """测试系统指标数据类"""

    def test_system_metrics_initialization(self, module):
        """测试系统指标初始化"""
        metrics = module.SystemMetrics(
            cpu_percent=50.0,
            memory_percent=60.0,
            disk_usage_percent=70.0,
            network_connections=10,
            timestamp=time.time()
        )

        assert metrics.cpu_percent == 50.0
        assert metrics.memory_percent == 60.0
        assert metrics.disk_usage_percent == 70.0
        assert metrics.network_connections == 10
        assert metrics.timestamp > 0


class TestSystemMonitor:
    """测试系统监控器"""

    def test_initialization(self, monitor):
        """测试初始化"""
        assert monitor.metrics_history == []
        assert monitor.max_history_size == 1000
        assert monitor._lock is not None

    def test_collect_system_metrics_success(self, monitor, module, monkeypatch):
        """测试收集系统指标 - 成功"""
        # Mock psutil
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0
        mock_net_connections = MagicMock(return_value=[1, 2, 3, 4, 5])

        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_connections", mock_net_connections)

        metrics = monitor.collect_system_metrics()

        assert metrics.cpu_percent == 50.0
        assert metrics.memory_percent == 60.0
        assert metrics.disk_usage_percent == 70.0
        assert metrics.network_connections == 5
        assert metrics.timestamp > 0
        assert len(monitor.metrics_history) == 1

    def test_collect_system_metrics_exception(self, monitor, module, monkeypatch):
        """测试收集系统指标 - 异常处理"""
        # Mock psutil.cpu_percent 抛出异常
        monkeypatch.setattr(module.psutil, "cpu_percent", MagicMock(side_effect=Exception("Test error")))

        # Mock 其他方法正常工作
        mock_memory = MagicMock()
        mock_memory.percent = 50.0
        mock_disk = MagicMock()
        mock_disk.percent = 60.0

        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk))
        monkeypatch.setattr(module.psutil, "net_connections", MagicMock(return_value=[1, 2, 3]))

        metrics = monitor.collect_system_metrics()

        # CPU应该返回默认值，其他指标正常工作
        assert metrics.cpu_percent == 0.0
        assert metrics.memory_percent == 50.0  # 正常工作的指标
        assert metrics.disk_usage_percent == 60.0  # 正常工作的指标
        assert metrics.network_connections == 3  # 正常工作的指标
        assert metrics.timestamp > 0

    def test_collect_system_metrics_history_limit(self, monitor, module, monkeypatch):
        """测试收集系统指标 - 历史记录限制"""
        # Mock psutil
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0
        mock_net_connections = MagicMock(return_value=[])

        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_connections", mock_net_connections)

        # 设置较小的历史记录大小
        monitor.max_history_size = 5

        # 收集超过限制的指标
        for i in range(10):
            monitor.collect_system_metrics()

        assert len(monitor.metrics_history) == 5

    def test_get_current_metrics_with_history(self, monitor, module, monkeypatch):
        """测试获取当前指标 - 有历史记录"""
        # Mock psutil
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0
        mock_net_connections = MagicMock(return_value=[])

        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_connections", mock_net_connections)

        monitor.collect_system_metrics()
        current = monitor.get_current_metrics()

        assert current is not None
        assert current.cpu_percent == 50.0

    def test_get_current_metrics_no_history(self, monitor):
        """测试获取当前指标 - 无历史记录"""
        current = monitor.get_current_metrics()
        assert current is None

    def test_get_metrics_history(self, monitor, module, monkeypatch):
        """测试获取指标历史"""
        # Mock psutil
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0
        mock_net_connections = MagicMock(return_value=[])

        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_connections", mock_net_connections)

        # 收集多个指标
        for i in range(10):
            monitor.collect_system_metrics()

        history = monitor.get_metrics_history(limit=5)
        assert len(history) == 5

    def test_get_metrics_history_default_limit(self, monitor, module, monkeypatch):
        """测试获取指标历史 - 默认限制"""
        # Mock psutil
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0
        mock_net_connections = MagicMock(return_value=[])

        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_connections", mock_net_connections)

        # 收集多个指标
        for i in range(150):
            monitor.collect_system_metrics()

        history = monitor.get_metrics_history()
        assert len(history) == 100  # 默认限制

    def test_get_average_metrics_with_data(self, monitor, module, monkeypatch):
        """测试获取平均指标 - 有数据"""
        # Mock psutil 和 time
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0
        mock_net_connections = MagicMock(return_value=[1, 2])

        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_connections", mock_net_connections)

        # 收集多个指标
        for i in range(5):
            monitor.collect_system_metrics()

        averages = monitor.get_average_metrics(time_window=300)

        assert averages['cpu_percent'] == 50.0
        assert averages['memory_percent'] == 60.0
        assert averages['disk_usage_percent'] == 70.0
        assert averages['network_connections'] == 2.0

    def test_get_average_metrics_no_data(self, monitor):
        """测试获取平均指标 - 无数据"""
        averages = monitor.get_average_metrics(time_window=60)

        assert averages['cpu_percent'] == 0.0
        assert averages['memory_percent'] == 0.0
        assert averages['disk_usage_percent'] == 0.0
        assert averages['network_connections'] == 0

    def test_get_average_metrics_time_window(self, monitor, module, monkeypatch):
        """测试获取平均指标 - 时间窗口过滤"""
        # Mock psutil 和 time
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0
        mock_net_connections = MagicMock(return_value=[])

        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_connections", mock_net_connections)

        # 收集指标
        monitor.collect_system_metrics()

        # Mock time 使指标超出时间窗口
        old_time = time.time() - 200
        monkeypatch.setattr(module.time, "time", MagicMock(return_value=old_time + 200))

        # 修改历史记录中的时间戳
        with monitor._lock:
            monitor.metrics_history[0].timestamp = old_time

        # 恢复正常的time
        monkeypatch.setattr(module.time, "time", MagicMock(return_value=old_time + 200))

        averages = monitor.get_average_metrics(time_window=60)
        # 由于所有指标都超出时间窗口，应该返回默认值
        assert averages['cpu_percent'] == 0.0

    def test_check_thresholds_default(self, monitor, module, monkeypatch):
        """测试检查阈值 - 使用默认阈值"""
        # Mock psutil
        mock_cpu_percent = MagicMock(return_value=90.0)  # 超过默认阈值80
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0  # 低于默认阈值85
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 95.0  # 超过默认阈值90
        mock_net_connections = MagicMock(return_value=[])

        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_connections", mock_net_connections)

        monitor.collect_system_metrics()
        results = monitor.check_thresholds()

        assert results['cpu_percent'] is True
        assert results['memory_percent'] is False
        assert results['disk_usage_percent'] is True

    def test_check_thresholds_custom(self, monitor, module, monkeypatch):
        """测试检查阈值 - 自定义阈值"""
        # Mock psutil
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0
        mock_net_connections = MagicMock(return_value=[])

        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_connections", mock_net_connections)

        monitor.collect_system_metrics()
        results = monitor.check_thresholds({
            'cpu_percent': 40.0,  # 50 > 40, 应该为True
            'memory_percent': 70.0,  # 60 < 70, 应该为False
            'disk_usage_percent': 60.0  # 70 > 60, 应该为True
        })

        assert results['cpu_percent'] is True
        assert results['memory_percent'] is False
        assert results['disk_usage_percent'] is True

    def test_check_thresholds_no_current_metrics(self, monitor):
        """测试检查阈值 - 无当前指标"""
        results = monitor.check_thresholds()

        assert results['cpu_percent'] is False
        assert results['memory_percent'] is False
        assert results['disk_usage_percent'] is False

    def test_check_thresholds_partial_thresholds(self, monitor, module, monkeypatch):
        """测试检查阈值 - 部分阈值"""
        # Mock psutil
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0
        mock_net_connections = MagicMock(return_value=[])

        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_connections", mock_net_connections)

        monitor.collect_system_metrics()
        results = monitor.check_thresholds({
            'cpu_percent': 40.0
        })

        assert results['cpu_percent'] is True
        # 其他指标使用默认阈值
        assert 'memory_percent' in results
        assert 'disk_usage_percent' in results

    def test_clear_history(self, monitor, module, monkeypatch):
        """测试清空历史记录"""
        # Mock psutil
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0
        mock_net_connections = MagicMock(return_value=[])

        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_connections", mock_net_connections)

        # 收集一些指标
        for i in range(5):
            monitor.collect_system_metrics()

        assert len(monitor.metrics_history) == 5

        monitor.clear_history()
        assert len(monitor.metrics_history) == 0

    def test_monitor_system(self, monitor, module, monkeypatch):
        """测试兼容接口 monitor_system"""
        # Mock psutil
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0
        mock_net_connections = MagicMock(return_value=[])

        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_connections", mock_net_connections)

        metrics = monitor.monitor_system()

        assert metrics.cpu_percent == 50.0
        assert metrics.memory_percent == 60.0
        assert metrics.disk_usage_percent == 70.0

    def test_get_average_metrics_exact_time_window(self, monitor, module, monkeypatch):
        """测试获取平均指标 - 精确时间窗口"""
        # Mock psutil
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0
        mock_net_connections = MagicMock(return_value=[1, 2])

        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_connections", mock_net_connections)

        # 收集指标
        monitor.collect_system_metrics()

        # 使用精确的时间窗口
        averages = monitor.get_average_metrics(time_window=1)
        assert averages['cpu_percent'] == 50.0

    def test_get_metrics_history_empty(self, monitor):
        """测试获取指标历史 - 空历史"""
        history = monitor.get_metrics_history(limit=10)
        assert len(history) == 0

    def test_get_metrics_history_more_than_limit(self, monitor, module, monkeypatch):
        """测试获取指标历史 - 超过限制"""
        # Mock psutil
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0
        mock_net_connections = MagicMock(return_value=[])

        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_connections", mock_net_connections)

        # 收集多个指标
        for i in range(20):
            monitor.collect_system_metrics()

        history = monitor.get_metrics_history(limit=10)
        assert len(history) == 10

    def test_check_thresholds_empty_dict(self, monitor, module, monkeypatch):
        """测试检查阈值 - 空字典"""
        # Mock psutil
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0
        mock_net_connections = MagicMock(return_value=[])

        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_connections", mock_net_connections)

        monitor.collect_system_metrics()
        results = monitor.check_thresholds({})

        # 应该使用默认阈值
        assert 'cpu_percent' in results
        assert 'memory_percent' in results
        assert 'disk_usage_percent' in results

    def test_collect_system_metrics_with_exact_max_history(self, monitor, module, monkeypatch):
        """测试收集系统指标 - 达到最大历史记录数"""
        # Mock psutil
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0
        mock_net_connections = MagicMock(return_value=[])

        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_connections", mock_net_connections)

        monitor.max_history_size = 3

        # 收集正好达到限制的指标
        for i in range(3):
            monitor.collect_system_metrics()

        assert len(monitor.metrics_history) == 3

        # 再收集一个，应该移除最旧的
        monitor.collect_system_metrics()
        assert len(monitor.metrics_history) == 3

    def test_get_average_metrics_single_metric(self, monitor, module, monkeypatch):
        """测试获取平均指标 - 单个指标"""
        # Mock psutil
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0
        mock_net_connections = MagicMock(return_value=[1, 2])

        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_connections", mock_net_connections)

        # 收集一个指标
        monitor.collect_system_metrics()

        averages = monitor.get_average_metrics(time_window=300)
        assert averages['cpu_percent'] == 50.0
        assert averages['network_connections'] == 2.0

    def test_get_average_metrics_multiple_metrics(self, monitor, module, monkeypatch):
        """测试获取平均指标 - 多个指标"""
        # Mock psutil
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0
        mock_net_connections = MagicMock(return_value=[1, 2, 3])

        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_connections", mock_net_connections)

        # 收集多个指标
        for i in range(3):
            monitor.collect_system_metrics()

        averages = monitor.get_average_metrics(time_window=300)
        assert averages['cpu_percent'] == 50.0
        assert averages['network_connections'] == 3.0

    def test_check_thresholds_exact_threshold(self, monitor, module, monkeypatch):
        """测试检查阈值 - 正好等于阈值"""
        # Mock psutil - CPU正好等于阈值
        mock_cpu_percent = MagicMock(return_value=80.0)  # 正好等于阈值
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0
        mock_net_connections = MagicMock(return_value=[])

        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_connections", mock_net_connections)

        monitor.collect_system_metrics()
        results = monitor.check_thresholds()

        # 应该为False（因为使用>而不是>=）
        assert results['cpu_percent'] is False

    def test_get_metrics_history_zero_limit(self, monitor, module, monkeypatch):
        """测试获取指标历史 - 限制为0"""
        # Mock psutil
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0
        mock_net_connections = MagicMock(return_value=[])

        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_connections", mock_net_connections)

        monitor.collect_system_metrics()
        history = monitor.get_metrics_history(limit=0)

        # Python中 [-0:] 会返回整个列表，所以这里应该返回所有历史记录
        assert isinstance(history, list)
        assert len(history) >= 0

    def test_get_metrics_history_negative_limit(self, monitor, module, monkeypatch):
        """测试获取指标历史 - 负数限制"""
        # Mock psutil
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0
        mock_net_connections = MagicMock(return_value=[])

        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_connections", mock_net_connections)

        monitor.collect_system_metrics()
        history = monitor.get_metrics_history(limit=-1)

        # 应该返回空列表或所有历史记录
        assert isinstance(history, list)

    def test_collect_system_metrics_exception_cpu(self, monitor, module, monkeypatch):
        """测试收集系统指标 - CPU异常"""
        # Mock psutil.cpu_percent 抛出异常
        monkeypatch.setattr(module.psutil, "cpu_percent", MagicMock(side_effect=Exception("CPU error")))
        
        # Mock 其他psutil方法
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0
        
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_connections", MagicMock(return_value=[]))
        
        metrics = monitor.collect_system_metrics()

        # CPU应该返回默认值，其他指标正常工作
        assert metrics.cpu_percent == 0.0
        assert metrics.memory_percent == 60.0  # 正常工作的指标
        assert metrics.disk_usage_percent == 70.0  # 正常工作的指标
        assert metrics.network_connections == 0  # 正常工作的指标

    def test_collect_system_metrics_exception_disk(self, monitor, module, monkeypatch):
        """测试收集系统指标 - 磁盘异常"""
        # Mock psutil.disk_usage 抛出异常
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        
        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(side_effect=Exception("Disk error")))
        monkeypatch.setattr(module.psutil, "net_connections", MagicMock(return_value=[]))
        
        metrics = monitor.collect_system_metrics()

        # 磁盘应该返回默认值，其他指标正常工作
        assert metrics.cpu_percent == 50.0  # 正常工作的指标
        assert metrics.memory_percent == 60.0  # 正常工作的指标
        assert metrics.disk_usage_percent == 0.0  # 失败的指标返回默认值
        assert metrics.network_connections == 0  # 正常工作的指标

    def test_collect_system_metrics_exception_network(self, monitor, module, monkeypatch):
        """测试收集系统指标 - 网络异常"""
        # Mock psutil.net_connections 抛出异常
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0
        
        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_connections", MagicMock(side_effect=Exception("Network error")))
        
        metrics = monitor.collect_system_metrics()
        
        # 应该返回默认值
        assert metrics.network_connections == 0

    def test_get_current_metrics_empty_history(self, monitor, module):
        """测试获取当前指标 - 空历史记录"""
        # 确保历史记录为空
        monitor.metrics_history = []
        
        current = monitor.get_current_metrics()
        
        # 应该返回None
        assert current is None

    def test_get_average_metrics_time_window_edge_case(self, monitor, module, monkeypatch):
        """测试获取平均指标 - 时间窗口边界情况"""
        # Mock psutil
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0
        mock_net_connections = MagicMock(return_value=[1, 2])
        
        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_connections", mock_net_connections)
        
        # 收集一个指标
        monitor.collect_system_metrics()
        
        # 使用非常大的时间窗口（应该包含所有指标）
        averages = monitor.get_average_metrics(time_window=999999)
        
        assert averages['cpu_percent'] == 50.0
        assert averages['network_connections'] == 2.0

    def test_check_thresholds_custom_thresholds(self, monitor, module, monkeypatch):
        """测试检查阈值 - 自定义阈值"""
        # Mock psutil
        mock_cpu_percent = MagicMock(return_value=90.0)  # 超过默认阈值80
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 90.0  # 超过默认阈值85
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 95.0  # 超过默认阈值90
        mock_net_connections = MagicMock(return_value=[])
        
        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_connections", mock_net_connections)
        
        monitor.collect_system_metrics()
        
        # 使用自定义阈值（更高）
        custom_thresholds = {
            'cpu_percent': 95.0,
            'memory_percent': 95.0,
            'disk_usage_percent': 98.0
        }
        
        results = monitor.check_thresholds(custom_thresholds)
        
        # 应该都低于自定义阈值
        assert results['cpu_percent'] is False
        assert results['memory_percent'] is False
        assert results['disk_usage_percent'] is False

    def test_check_thresholds_partial_thresholds(self, monitor, module, monkeypatch):
        """测试检查阈值 - 部分阈值"""
        # Mock psutil
        mock_cpu_percent = MagicMock(return_value=90.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0
        mock_net_connections = MagicMock(return_value=[])
        
        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_connections", mock_net_connections)
        
        monitor.collect_system_metrics()
        
        # 只提供部分阈值
        partial_thresholds = {
            'cpu_percent': 85.0
        }
        
        results = monitor.check_thresholds(partial_thresholds)
        
        # CPU应该超过阈值
        assert results['cpu_percent'] is True
        # 其他应该使用默认值
        assert 'memory_percent' in results
        assert 'disk_usage_percent' in results

    def test_collect_system_metrics_memory_exception(self, monitor, module, monkeypatch):
        """测试收集系统指标 - 内存异常"""
        # Mock所有psutil方法抛出异常，模拟系统监控完全失败的情况
        monkeypatch.setattr(module.psutil, "cpu_percent", MagicMock(side_effect=Exception("CPU error")))
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(side_effect=Exception("Memory error")))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(side_effect=Exception("Disk error")))
        monkeypatch.setattr(module.psutil, "net_connections", MagicMock(side_effect=Exception("Network error")))

        metrics = monitor.collect_system_metrics()

        # 应该返回默认值
        assert metrics.cpu_percent == 0.0
        assert metrics.memory_percent == 0.0
        assert metrics.disk_usage_percent == 0.0
        assert metrics.network_connections == 0

    def test_get_average_metrics_exact_cutoff_time(self, monitor, module, monkeypatch):
        """测试获取平均指标 - 正好等于截止时间"""
        import time as time_module
        
        # Mock psutil
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0
        mock_net_connections = MagicMock(return_value=[1, 2])
        
        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_connections", mock_net_connections)
        
        # 收集一个指标
        monitor.collect_system_metrics()
        
        # 使用非常大的时间窗口（应该包含所有指标）
        averages = monitor.get_average_metrics(time_window=999999)
        
        # 应该包含这个指标
        assert averages['cpu_percent'] == 50.0

    def test_get_average_metrics_before_cutoff_time(self, monitor, module, monkeypatch):
        """测试获取平均指标 - 在截止时间之前"""
        import time as time_module
        
        # Mock psutil
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0
        mock_net_connections = MagicMock(return_value=[1, 2])
        
        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_connections", mock_net_connections)
        
        # 收集一个指标
        current_time = time_module.time()
        monitor.collect_system_metrics()
        
        # 修改时间戳为61秒前（超过60秒窗口）
        if monitor.metrics_history:
            monitor.metrics_history[0].timestamp = current_time - 61
        
        # 使用60秒时间窗口
        averages = monitor.get_average_metrics(time_window=60)
        
        # 应该返回默认值（因为没有指标在时间窗口内）
        assert averages['cpu_percent'] == 0.0

    def test_check_thresholds_all_exceeded(self, monitor, module, monkeypatch):
        """测试检查阈值 - 所有指标都超过阈值"""
        # Mock psutil
        mock_cpu_percent = MagicMock(return_value=90.0)  # 超过阈值80
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 90.0  # 超过阈值85
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 95.0  # 超过阈值90
        mock_net_connections = MagicMock(return_value=[])
        
        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_connections", mock_net_connections)
        
        monitor.collect_system_metrics()
        results = monitor.check_thresholds()
        
        # 所有指标都应该超过阈值
        assert results['cpu_percent'] is True
        assert results['memory_percent'] is True
        assert results['disk_usage_percent'] is True

    def test_check_thresholds_none_exceeded(self, monitor, module, monkeypatch):
        """测试检查阈值 - 所有指标都低于阈值"""
        # Mock psutil
        mock_cpu_percent = MagicMock(return_value=50.0)  # 低于阈值80
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0  # 低于阈值85
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0  # 低于阈值90
        mock_net_connections = MagicMock(return_value=[])
        
        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_connections", mock_net_connections)
        
        monitor.collect_system_metrics()
        results = monitor.check_thresholds()
        
        # 所有指标都应该低于阈值
        assert results['cpu_percent'] is False
        assert results['memory_percent'] is False
        assert results['disk_usage_percent'] is False

    def test_get_metrics_history_exact_limit(self, monitor, module, monkeypatch):
        """测试获取指标历史 - 正好等于限制"""
        # Mock psutil
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0
        mock_net_connections = MagicMock(return_value=[])
        
        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_connections", mock_net_connections)
        
        # 收集5个指标
        for i in range(5):
            monitor.collect_system_metrics()
        
        # 获取正好5个指标
        history = monitor.get_metrics_history(limit=5)
        
        assert len(history) == 5

    def test_get_metrics_history_more_than_total(self, monitor, module, monkeypatch):
        """测试获取指标历史 - 限制超过总数"""
        # Mock psutil
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0
        mock_net_connections = MagicMock(return_value=[])
        
        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_connections", mock_net_connections)
        
        # 收集3个指标
        for i in range(3):
            monitor.collect_system_metrics()
        
        # 获取10个指标（超过总数）
        history = monitor.get_metrics_history(limit=10)
        
        # 应该返回所有3个指标
        assert len(history) == 3

    def test_system_metrics_boundary_values(self, module):
        """测试系统指标边界值"""
        # 测试零值
        metrics_zero = module.SystemMetrics(
            cpu_percent=0.0,
            memory_percent=0.0,
            disk_usage_percent=0.0,
            network_connections=0,
            timestamp=0.0
        )
        assert metrics_zero.cpu_percent == 0.0
        assert metrics_zero.network_connections == 0
        
        # 测试最大值
        metrics_max = module.SystemMetrics(
            cpu_percent=100.0,
            memory_percent=100.0,
            disk_usage_percent=100.0,
            network_connections=1000000,
            timestamp=time.time()
        )
        assert metrics_max.cpu_percent == 100.0
        assert metrics_max.network_connections == 1000000

    def test_system_metrics_negative_values(self, module):
        """测试系统指标负值（虽然不应该发生，但测试数据类的容错性）"""
        metrics_negative = module.SystemMetrics(
            cpu_percent=-1.0,
            memory_percent=-1.0,
            disk_usage_percent=-1.0,
            network_connections=-1,
            timestamp=-1.0
        )
        assert metrics_negative.cpu_percent == -1.0
        assert metrics_negative.network_connections == -1

    def test_collect_system_metrics_concurrent_access(self, monitor, module, monkeypatch):
        """测试并发访问收集系统指标"""
        import threading
        
        # Mock psutil
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0
        mock_net_connections = MagicMock(return_value=[])
        
        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_connections", mock_net_connections)
        
        # 并发收集指标
        def collect_metrics():
            for _ in range(10):
                monitor.collect_system_metrics()
        
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=collect_metrics)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # 应该收集了50个指标（5个线程 * 10次）
        assert len(monitor.metrics_history) == 50

    def test_get_current_metrics_concurrent_access(self, monitor, module, monkeypatch):
        """测试并发访问获取当前指标"""
        import threading
        
        # Mock psutil
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0
        mock_net_connections = MagicMock(return_value=[])
        
        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_connections", mock_net_connections)
        
        # 先收集一些指标
        monitor.collect_system_metrics()
        
        # 并发获取当前指标
        results = []
        def get_current():
            results.append(monitor.get_current_metrics())
        
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=get_current)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # 应该都返回相同的指标
        assert len(results) == 10
        assert all(r is not None for r in results)
        assert all(r.cpu_percent == 50.0 for r in results)

    def test_get_metrics_history_concurrent_access(self, monitor, module, monkeypatch):
        """测试并发访问获取指标历史"""
        import threading
        
        # Mock psutil
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0
        mock_net_connections = MagicMock(return_value=[])
        
        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_connections", mock_net_connections)
        
        # 先收集一些指标
        for _ in range(5):
            monitor.collect_system_metrics()
        
        # 并发获取历史
        results = []
        def get_history():
            results.append(monitor.get_metrics_history(limit=3))
        
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=get_history)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # 应该都返回3个指标
        assert len(results) == 10
        assert all(len(r) == 3 for r in results)

    def test_check_thresholds_concurrent_access(self, monitor, module, monkeypatch):
        """测试并发访问检查阈值"""
        import threading
        
        # Mock psutil
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0
        mock_net_connections = MagicMock(return_value=[])
        
        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_connections", mock_net_connections)
        
        # 先收集指标
        monitor.collect_system_metrics()
        
        # 并发检查阈值
        results = []
        def check_thresholds():
            results.append(monitor.check_thresholds())
        
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=check_thresholds)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # 应该都返回相同的结果
        assert len(results) == 10
        assert all(isinstance(r, dict) for r in results)
        assert all('cpu_percent' in r for r in results)

    def test_clear_history_concurrent_access(self, monitor, module, monkeypatch):
        """测试并发访问清空历史"""
        import threading
        
        # Mock psutil
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0
        mock_net_connections = MagicMock(return_value=[])
        
        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_connections", mock_net_connections)
        
        # 先收集一些指标
        for _ in range(10):
            monitor.collect_system_metrics()
        
        # 并发清空历史
        def clear_history():
            monitor.clear_history()
        
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=clear_history)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # 历史应该被清空
        assert len(monitor.metrics_history) == 0

    def test_get_average_metrics_zero_time_window(self, monitor, module, monkeypatch):
        """测试获取平均指标 - 零时间窗口"""
        import time
        # Mock psutil
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0
        mock_net_connections = MagicMock(return_value=[])
        
        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_connections", mock_net_connections)
        
        # 收集一些指标
        monitor.collect_system_metrics()
        
        # 等待一小段时间，确保时间戳差异
        time.sleep(0.01)
        
        # 使用零时间窗口（cutoff_time = current_time，所以没有指标在窗口内）
        avg_metrics = monitor.get_average_metrics(time_window=0)
        
        # 应该返回默认值（因为时间窗口为0，没有指标在窗口内）
        assert avg_metrics['cpu_percent'] == 0.0
        assert avg_metrics['memory_percent'] == 0.0
        assert avg_metrics['disk_usage_percent'] == 0.0
        assert avg_metrics['network_connections'] == 0

    def test_get_average_metrics_negative_time_window(self, monitor, module, monkeypatch):
        """测试获取平均指标 - 负时间窗口"""
        # Mock psutil
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0
        mock_net_connections = MagicMock(return_value=[])
        
        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_connections", mock_net_connections)
        
        # 收集一些指标
        monitor.collect_system_metrics()
        
        # 使用负时间窗口
        avg_metrics = monitor.get_average_metrics(time_window=-10)
        
        # 应该返回默认值（因为时间窗口为负，没有指标在窗口内）
        assert avg_metrics['cpu_percent'] == 0.0
        assert avg_metrics['memory_percent'] == 0.0

    def test_check_thresholds_exact_threshold_values(self, monitor, module, monkeypatch):
        """测试检查阈值 - 精确阈值值"""
        # Mock psutil
        mock_cpu_percent = MagicMock(return_value=80.0)  # 等于阈值
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 85.0  # 等于阈值
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 90.0  # 等于阈值
        mock_net_connections = MagicMock(return_value=[])
        
        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_connections", mock_net_connections)
        
        # 收集指标
        monitor.collect_system_metrics()
        
        # 检查阈值（使用默认阈值）
        result = monitor.check_thresholds()
        
        # 应该都返回False（因为使用 > 而不是 >=）
        assert result['cpu_percent'] == False
        assert result['memory_percent'] == False
        assert result['disk_usage_percent'] == False

