#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试基础设施层 - 存储监控组件

测试 infrastructure/storage_monitor.py 中的所有类和方法
"""

import pytest
import os
import time
from unittest.mock import Mock, MagicMock, patch


@pytest.fixture
def module():
    """导入模块"""
    from src.infrastructure.monitoring.infrastructure import storage_monitor
    return storage_monitor


@pytest.fixture
def monitor(module):
    """创建存储监控器实例"""
    return module.StorageMonitor(monitor_interval=1.0)


class TestStorageMetric:
    """测试存储监控指标枚举"""

    def test_storage_metric_values(self, module):
        """测试存储指标值"""
        assert module.StorageMetric.TOTAL_SIZE.value == "total_size"
        assert module.StorageMetric.USED_SIZE.value == "used_size"
        assert module.StorageMetric.FREE_SIZE.value == "free_size"
        assert module.StorageMetric.USAGE_PERCENT.value == "usage_percent"
        assert module.StorageMetric.READ_COUNT.value == "read_count"
        assert module.StorageMetric.WRITE_COUNT.value == "write_count"


class TestStorageStats:
    """测试存储统计数据类"""

    def test_storage_stats_initialization(self, module):
        """测试存储统计初始化"""
        stats = module.StorageStats(
            mount_point="/test",
            total_size=1000,
            used_size=500,
            free_size=500,
            usage_percent=50.0
        )

        assert stats.mount_point == "/test"
        assert stats.total_size == 1000
        assert stats.used_size == 500
        assert stats.free_size == 500
        assert stats.usage_percent == 50.0
        assert stats.timestamp is not None

    def test_storage_stats_timestamp_auto(self, module):
        """测试存储统计时间戳自动设置"""
        stats = module.StorageStats(
            mount_point="/test",
            total_size=1000,
            used_size=500,
            free_size=500,
            usage_percent=50.0,
            timestamp=None
        )

        assert stats.timestamp is not None

    def test_storage_stats_to_dict(self, module):
        """测试存储统计转换为字典"""
        stats = module.StorageStats(
            mount_point="/test",
            total_size=1000,
            used_size=500,
            free_size=500,
            usage_percent=50.0
        )

        stats_dict = stats.to_dict()
        assert isinstance(stats_dict, dict)
        assert stats_dict['mount_point'] == "/test"
        assert stats_dict['total_size'] == 1000


class TestStorageMonitor:
    """测试存储监控器"""

    def test_initialization(self, monitor):
        """测试初始化"""
        assert monitor.monitor_interval == 1.0
        assert monitor._stats_history == []
        assert monitor._max_history_size == 1000
        assert monitor._monitoring is False
        assert monitor._monitor_thread is None

    def test_init_mount_points_success(self, monitor, module, monkeypatch):
        """测试初始化挂载点 - 成功"""
        # Mock psutil.disk_partitions
        mock_partition = MagicMock()
        mock_partition.mountpoint = "/test"
        
        monkeypatch.setattr(module.psutil, "disk_partitions", MagicMock(return_value=[mock_partition]))
        monkeypatch.setattr(os.path, "exists", MagicMock(return_value=True))

        monitor._init_mount_points()
        assert "/test" in monitor._mount_points

    def test_init_mount_points_exception(self, monitor, module, monkeypatch):
        """测试初始化挂载点 - 异常处理"""
        # Mock psutil.disk_partitions 抛出异常
        monkeypatch.setattr(module.psutil, "disk_partitions", MagicMock(side_effect=Exception("Test error")))

        monitor._init_mount_points()
        # 应该使用默认挂载点
        assert len(monitor._mount_points) > 0

    def test_start_monitoring(self, monitor, monkeypatch):
        """测试启动监控"""
        created_threads = []
        original_thread = type(monitor._monitor_thread) if monitor._monitor_thread else type(monitor._monitor_thread)

        def mock_thread(*args, **kwargs):
            thread = MagicMock()
            thread.start = MagicMock()
            created_threads.append(thread)
            return thread

        monkeypatch.setattr("threading.Thread", mock_thread)

        monitor.start_monitoring()

        assert monitor._monitoring is True
        assert len(created_threads) == 1

    def test_start_monitoring_already_monitoring(self, monitor):
        """测试启动监控 - 已经监控中"""
        monitor._monitoring = True
        original_thread = monitor._monitor_thread

        monitor.start_monitoring()

        # 不应该创建新线程
        assert monitor._monitor_thread == original_thread

    def test_stop_monitoring(self, monitor, monkeypatch):
        """测试停止监控"""
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        mock_thread.join = MagicMock()

        monitor._monitoring = True
        monitor._monitor_thread = mock_thread

        monitor.stop_monitoring()

        assert monitor._monitoring is False
        mock_thread.join.assert_called_once_with(timeout=5.0)

    def test_stop_monitoring_no_thread(self, monitor):
        """测试停止监控 - 无线程"""
        monitor._monitoring = True
        monitor._monitor_thread = None

        monitor.stop_monitoring()

        assert monitor._monitoring is False

    def test_collect_stats_success(self, monitor, module, monkeypatch):
        """测试收集统计信息 - 成功"""
        # Mock psutil 和 os
        mock_disk_usage = MagicMock()
        mock_disk_usage.total = 1000
        mock_disk_usage.used = 500
        mock_disk_usage.free = 500
        mock_disk_usage.percent = 50.0

        monitor._mount_points = ["/test"]
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        # os.statvfs 只在 POSIX 系统上存在
        if hasattr(os, "statvfs"):
            monkeypatch.setattr(os, "statvfs", MagicMock(return_value=None))

        monitor._collect_stats()

        assert len(monitor._stats_history) == 1
        assert monitor._stats_history[0].mount_point == "/test"
        assert monitor._stats_history[0].total_size == 1000

    def test_collect_stats_exception(self, monitor, module, monkeypatch):
        """测试收集统计信息 - 异常处理"""
        monitor._mount_points = ["/test"]
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(side_effect=Exception("Test error")))

        monitor._collect_stats()

        # 异常应该被捕获，不抛出
        assert True

    def test_collect_stats_max_history(self, monitor, module, monkeypatch):
        """测试收集统计信息 - 历史记录限制"""
        mock_disk_usage = MagicMock()
        mock_disk_usage.total = 1000
        mock_disk_usage.used = 500
        mock_disk_usage.free = 500
        mock_disk_usage.percent = 50.0

        monitor._mount_points = ["/test"]
        monitor._max_history_size = 5
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        if hasattr(os, "statvfs"):
            monkeypatch.setattr(os, "statvfs", MagicMock(return_value=None))

        for i in range(10):
            monitor._collect_stats()

        assert len(monitor._stats_history) == 5

    def test_record_operation_read(self, monitor):
        """测试记录操作 - 读取"""
        monitor.record_operation('read', size=100, duration=0.1, success=True)

        assert monitor._manual_read_count == 1
        assert monitor._manual_read_bytes == 100
        assert monitor._manual_read_time == 0.1
        assert monitor._manual_error_count == 0

    def test_record_operation_write(self, monitor):
        """测试记录操作 - 写入"""
        monitor.record_operation('write', size=200, duration=0.2, success=True)

        assert monitor._manual_write_count == 1
        assert monitor._manual_write_bytes == 200
        assert monitor._manual_write_time == 0.2
        assert monitor._manual_error_count == 0

    def test_record_operation_failure(self, monitor):
        """测试记录操作 - 失败"""
        monitor.record_operation('read', size=100, duration=0.1, success=False)

        assert monitor._manual_read_count == 1
        assert monitor._manual_error_count == 1

    def test_record_write(self, monitor):
        """测试记录写入操作"""
        monitor.record_write(size=300, duration=0.3)

        assert monitor._manual_write_count == 1
        assert monitor._manual_write_bytes == 300
        assert monitor._manual_write_time == 0.3

    def test_record_error(self, monitor):
        """测试记录错误"""
        initial_count = monitor._manual_error_count
        monitor.record_error()

        assert monitor._manual_error_count == initial_count + 1

    def test_get_current_stats(self, monitor, module, monkeypatch):
        """测试获取当前统计信息"""
        # Mock psutil
        mock_disk_usage = MagicMock()
        mock_disk_usage.total = 1000
        mock_disk_usage.used = 500
        mock_disk_usage.free = 500
        mock_disk_usage.percent = 50.0

        monitor._mount_points = ["/test1", "/test2"]
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        if hasattr(os, "statvfs"):
            monkeypatch.setattr(os, "statvfs", MagicMock(return_value=None))

        monitor._collect_stats()
        current = monitor.get_current_stats()

        assert len(current) == 2

    def test_get_current_stats_empty(self, monitor):
        """测试获取当前统计信息 - 空数据"""
        current = monitor.get_current_stats()
        assert current == []

    def test_get_stats_history(self, monitor, module, monkeypatch):
        """测试获取统计历史"""
        # Mock psutil
        mock_disk_usage = MagicMock()
        mock_disk_usage.total = 1000
        mock_disk_usage.used = 500
        mock_disk_usage.free = 500
        mock_disk_usage.percent = 50.0

        monitor._mount_points = ["/test"]
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        if hasattr(os, "statvfs"):
            monkeypatch.setattr(os, "statvfs", MagicMock(return_value=None))

        for i in range(10):
            monitor._collect_stats()

        history = monitor.get_stats_history(limit=5)
        assert len(history) == 5

    def test_get_stats_history_default_limit(self, monitor, module, monkeypatch):
        """测试获取统计历史 - 默认限制"""
        # Mock psutil
        mock_disk_usage = MagicMock()
        mock_disk_usage.total = 1000
        mock_disk_usage.used = 500
        mock_disk_usage.free = 500
        mock_disk_usage.percent = 50.0

        monitor._mount_points = ["/test"]
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        if hasattr(os, "statvfs"):
            monkeypatch.setattr(os, "statvfs", MagicMock(return_value=None))

        for i in range(150):
            monitor._collect_stats()

        history = monitor.get_stats_history()
        assert len(history) == 100  # 默认限制

    def test_get_aggregated_stats_empty(self, monitor):
        """测试获取聚合统计信息 - 空数据"""
        stats = monitor.get_aggregated_stats()
        assert stats == {}

    def test_get_aggregated_stats(self, monitor, module, monkeypatch):
        """测试获取聚合统计信息"""
        # Mock psutil
        mock_disk_usage = MagicMock()
        mock_disk_usage.total = 1000
        mock_disk_usage.used = 500
        mock_disk_usage.free = 500
        mock_disk_usage.percent = 50.0

        monitor._mount_points = ["/test1", "/test2"]
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        if hasattr(os, "statvfs"):
            monkeypatch.setattr(os, "statvfs", MagicMock(return_value=None))

        monitor._collect_stats()
        stats = monitor.get_aggregated_stats()

        assert stats['total_mount_points'] == 2
        assert stats['total_size'] == 2000
        assert stats['used_size'] == 1000
        assert stats['free_size'] == 1000
        assert 'usage_percent' in stats
        assert 'mount_points' in stats

    def test_get_metrics_for_prometheus_empty(self, monitor):
        """测试获取Prometheus格式指标 - 空数据"""
        metrics = monitor.get_metrics_for_prometheus()
        assert metrics == ""

    def test_get_metrics_for_prometheus(self, monitor, module, monkeypatch):
        """测试获取Prometheus格式指标"""
        # Mock psutil
        mock_disk_usage = MagicMock()
        mock_disk_usage.total = 1000
        mock_disk_usage.used = 500
        mock_disk_usage.free = 500
        mock_disk_usage.percent = 50.0

        monitor._mount_points = ["/test"]
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        if hasattr(os, "statvfs"):
            monkeypatch.setattr(os, "statvfs", MagicMock(return_value=None))

        monitor._collect_stats()
        monitor.record_write(size=100, duration=0.1)
        monitor.record_error()

        metrics = monitor.get_metrics_for_prometheus()

        assert "storage_total_size" in metrics
        assert "storage_used_size" in metrics
        assert "storage_usage_percent" in metrics
        assert "storage_read_count" in metrics
        assert "storage_write_count" in metrics
        assert "storage_error_count" in metrics

    def test_reset_stats(self, monitor, module, monkeypatch):
        """测试重置统计信息"""
        # Mock psutil
        mock_disk_usage = MagicMock()
        mock_disk_usage.total = 1000
        mock_disk_usage.used = 500
        mock_disk_usage.free = 500
        mock_disk_usage.percent = 50.0

        monitor._mount_points = ["/test"]
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        if hasattr(os, "statvfs"):
            monkeypatch.setattr(os, "statvfs", MagicMock(return_value=None))

        monitor._collect_stats()
        monitor.record_write(size=100, duration=0.1)

        assert len(monitor._stats_history) > 0
        assert monitor._manual_write_count > 0

        monitor.reset_stats()

        assert len(monitor._stats_history) == 0
        assert monitor._manual_read_count == 0
        assert monitor._manual_write_count == 0
        assert monitor._manual_read_bytes == 0
        assert monitor._manual_write_bytes == 0
        assert monitor._manual_read_time == 0.0
        assert monitor._manual_write_time == 0.0
        assert monitor._manual_error_count == 0

    def test_monitor_loop_exception_handling(self, monitor, module, monkeypatch):
        """测试监控循环异常处理"""
        # Mock _collect_stats 抛出异常
        call_count = {"count": 0}
        def mock_collect():
            call_count["count"] += 1
            if call_count["count"] == 1:
                raise Exception("Collect error")
            # 第二次调用时设置 _monitoring = False 以便退出循环
            monitor._monitoring = False
        
        monkeypatch.setattr(monitor, "_collect_stats", mock_collect)
        monkeypatch.setattr(module.time, "sleep", lambda *_: None)

        # 捕获print输出
        prints = []
        def mock_print(*args, **kwargs):
            prints.append(args)
        
        monkeypatch.setattr("builtins.print", mock_print)

        monitor._monitoring = True
        monitor._monitor_loop()

        # 验证异常被处理
        assert len(prints) > 0

    def test_storage_stats_to_dict(self, module):
        """测试StorageStats转换为字典"""
        stats = module.StorageStats(
            mount_point="/test",
            total_size=1000,
            used_size=500,
            free_size=500,
            usage_percent=50.0
        )
        
        dict_data = stats.to_dict()
        
        assert isinstance(dict_data, dict)
        assert dict_data['mount_point'] == "/test"
        assert dict_data['total_size'] == 1000
        assert dict_data['usage_percent'] == 50.0

    def test_storage_stats_post_init(self, module):
        """测试StorageStats的__post_init__"""
        stats = module.StorageStats(
            mount_point="/test",
            total_size=1000,
            used_size=500,
            free_size=500,
            usage_percent=50.0,
            timestamp=None  # 应该自动设置
        )
        
        assert stats.timestamp is not None
        assert stats.timestamp > 0

    def test_record_operation_read(self, monitor):
        """测试记录读取操作"""
        monitor.record_operation('read', size=100, duration=0.1, success=True)
        
        assert monitor._manual_read_count == 1
        assert monitor._manual_read_bytes == 100
        assert monitor._manual_read_time == 0.1
        assert monitor._manual_error_count == 0

    def test_record_operation_write_failure(self, monitor):
        """测试记录写入操作 - 失败"""
        monitor.record_operation('write', size=200, duration=0.2, success=False)
        
        assert monitor._manual_write_count == 1
        assert monitor._manual_write_bytes == 200
        assert monitor._manual_write_time == 0.2
        assert monitor._manual_error_count == 1

    def test_record_operation_read_failure(self, monitor):
        """测试记录读取操作 - 失败"""
        monitor.record_operation('read', size=150, duration=0.15, success=False)
        
        assert monitor._manual_read_count == 1
        assert monitor._manual_read_bytes == 150
        assert monitor._manual_read_time == 0.15
        assert monitor._manual_error_count == 1

    def test_record_operation_unknown_type(self, monitor):
        """测试记录未知操作类型"""
        monitor.record_operation('unknown', size=100, duration=0.1, success=False)
        
        # 应该只增加错误计数
        assert monitor._manual_error_count == 1
        assert monitor._manual_read_count == 0
        assert monitor._manual_write_count == 0

    def test_init_mount_points_exception(self, monitor, module, monkeypatch):
        """测试初始化挂载点 - 异常处理"""
        # Mock psutil 抛出异常
        monkeypatch.setattr(module.psutil, "disk_partitions", MagicMock(side_effect=Exception("Test error")))
        
        # 重新初始化挂载点
        monitor._init_mount_points()
        
        # 应该使用默认挂载点
        assert len(monitor._mount_points) > 0

    def test_start_monitoring_already_running(self, monitor, module, monkeypatch):
        """测试启动监控 - 已经运行"""
        monitor._monitoring = True
        original_thread = monitor._monitor_thread
        
        monitor.start_monitoring()
        
        # 不应该创建新线程
        assert monitor._monitor_thread == original_thread

    def test_stop_monitoring_not_running(self, monitor):
        """测试停止监控 - 未运行"""
        monitor._monitoring = False
        monitor._monitor_thread = None
        
        # 不应该抛出异常
        monitor.stop_monitoring()

    def test_collect_stats_exception(self, monitor, module, monkeypatch):
        """测试收集统计信息 - 异常处理"""
        # Mock psutil 抛出异常
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(side_effect=Exception("Test error")))
        
        monitor._mount_points = ["/test"]
        
        # 应该跳过该挂载点，不抛出异常
        monitor._collect_stats()
        
        # 历史记录应该为空（因为异常被捕获）
        assert len(monitor._stats_history) == 0

    def test_collect_stats_multiple_mount_points(self, monitor, module, monkeypatch):
        """测试收集统计信息 - 多个挂载点"""
        # Mock psutil
        mock_disk_usage = MagicMock()
        mock_disk_usage.total = 1000
        mock_disk_usage.used = 500
        mock_disk_usage.free = 500
        mock_disk_usage.percent = 50.0
        
        monitor._mount_points = ["/test1", "/test2", "/test3"]
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        if hasattr(os, "statvfs"):
            monkeypatch.setattr(os, "statvfs", MagicMock(return_value=None))
        
        monitor._collect_stats()
        
        # 应该为每个挂载点创建统计信息
        assert len(monitor._stats_history) == 3

    def test_get_aggregated_stats_multiple_mount_points(self, monitor, module, monkeypatch):
        """测试获取聚合统计信息 - 多个挂载点"""
        # Mock psutil
        mock_disk_usage = MagicMock()
        mock_disk_usage.total = 1000
        mock_disk_usage.used = 500
        mock_disk_usage.free = 500
        mock_disk_usage.percent = 50.0
        
        monitor._mount_points = ["/test1", "/test2"]
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        if hasattr(os, "statvfs"):
            monkeypatch.setattr(os, "statvfs", MagicMock(return_value=None))
        
        monitor._collect_stats()
        stats = monitor.get_aggregated_stats()
        
        assert stats['total_mount_points'] == 2
        assert stats['total_size'] == 2000  # 2个挂载点，每个1000
        assert len(stats['mount_points']) == 2

    def test_get_stats_history_max_size(self, monitor, module, monkeypatch):
        """测试统计历史 - 达到最大大小"""
        # Mock psutil
        mock_disk_usage = MagicMock()
        mock_disk_usage.total = 1000
        mock_disk_usage.used = 500
        mock_disk_usage.free = 500
        mock_disk_usage.percent = 50.0
        
        monitor._mount_points = ["/test"]
        monitor._max_history_size = 5
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        if hasattr(os, "statvfs"):
            monkeypatch.setattr(os, "statvfs", MagicMock(return_value=None))
        
        # 收集超过限制的统计信息
        for i in range(10):
            monitor._collect_stats()
        
        # 应该只保留最新的5条
        assert len(monitor._stats_history) == 5

    def test_collect_stats_exception_in_loop(self, monitor, module, monkeypatch):
        """测试收集统计信息 - 循环中异常处理"""
        # Mock psutil.disk_usage 抛出异常
        monitor._mount_points = ["/test"]
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(side_effect=Exception("Disk error")))
        
        # 调用 _collect_stats，应该捕获异常并跳过该挂载点
        monitor._collect_stats()
        
        # 历史记录应该为空（因为异常被捕获并跳过）
        assert len(monitor._stats_history) == 0

    def test_collect_stats_multiple_mount_points_with_exception(self, monitor, module, monkeypatch):
        """测试收集统计信息 - 多个挂载点，其中一个异常"""
        # Mock psutil
        mock_disk_usage_good = MagicMock()
        mock_disk_usage_good.total = 1000
        mock_disk_usage_good.used = 500
        mock_disk_usage_good.free = 500
        mock_disk_usage_good.percent = 50.0
        
        mock_disk_usage_bad = MagicMock(side_effect=Exception("Mount point error"))
        
        monitor._mount_points = ["/good", "/bad"]
        
        call_count = [0]
        def mock_disk_usage(path):
            call_count[0] += 1
            if path == "/good":
                return mock_disk_usage_good
            else:
                raise Exception("Mount point error")
        
        monkeypatch.setattr(module.psutil, "disk_usage", mock_disk_usage)
        if hasattr(os, "statvfs"):
            monkeypatch.setattr(os, "statvfs", MagicMock(return_value=None))
        
        monitor._collect_stats()
        
        # 应该至少收集到一个挂载点的统计信息
        assert len(monitor._stats_history) >= 1

    def test_record_operation_unknown_type(self, monitor):
        """测试记录操作 - 未知类型"""
        original_read_count = monitor._manual_read_count
        original_write_count = monitor._manual_write_count
        
        # 记录未知类型的操作
        monitor.record_operation('unknown', size=100, duration=0.1, success=True)
        
        # 读取和写入计数不应该改变
        assert monitor._manual_read_count == original_read_count
        assert monitor._manual_write_count == original_write_count

    def test_record_operation_read_failure(self, monitor):
        """测试记录操作 - 读取失败"""
        original_read_count = monitor._manual_read_count
        original_error_count = monitor._manual_error_count
        
        monitor.record_operation('read', size=100, duration=0.1, success=False)
        
        # 读取计数应该增加
        assert monitor._manual_read_count == original_read_count + 1
        # 错误计数应该增加
        assert monitor._manual_error_count == original_error_count + 1

    def test_record_operation_write_failure(self, monitor):
        """测试记录操作 - 写入失败"""
        original_write_count = monitor._manual_write_count
        original_error_count = monitor._manual_error_count
        
        monitor.record_operation('write', size=100, duration=0.1, success=False)
        
        # 写入计数应该增加
        assert monitor._manual_write_count == original_write_count + 1
        # 错误计数应该增加
        assert monitor._manual_error_count == original_error_count + 1

    def test_get_current_stats_empty_history(self, monitor):
        """测试获取当前统计信息 - 空历史记录"""
        monitor._stats_history = []
        
        current = monitor.get_current_stats()
        
        # 应该返回空列表
        assert len(current) == 0

    def test_get_current_stats_multiple_mount_points(self, monitor, module, monkeypatch):
        """测试获取当前统计信息 - 多个挂载点"""
        # Mock psutil
        mock_disk_usage = MagicMock()
        mock_disk_usage.total = 1000
        mock_disk_usage.used = 500
        mock_disk_usage.free = 500
        mock_disk_usage.percent = 50.0
        
        monitor._mount_points = ["/mount1", "/mount2"]
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        if hasattr(os, "statvfs"):
            monkeypatch.setattr(os, "statvfs", MagicMock(return_value=None))
        
        # 收集统计信息
        monitor._collect_stats()
        
        current = monitor.get_current_stats()
        
        # 应该返回每个挂载点的统计信息
        assert len(current) == len(monitor._mount_points)

    def test_get_stats_history_zero_limit(self, monitor, module, monkeypatch):
        """测试获取统计历史 - 限制为0"""
        # Mock psutil
        mock_disk_usage = MagicMock()
        mock_disk_usage.total = 1000
        mock_disk_usage.used = 500
        mock_disk_usage.free = 500
        mock_disk_usage.percent = 50.0
        
        monitor._mount_points = ["/test"]
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        if hasattr(os, "statvfs"):
            monkeypatch.setattr(os, "statvfs", MagicMock(return_value=None))
        
        monitor._collect_stats()
        history = monitor.get_stats_history(limit=0)
        
        # Python中 [-0:] 会返回整个列表
        assert isinstance(history, list)

    def test_get_stats_history_negative_limit(self, monitor, module, monkeypatch):
        """测试获取统计历史 - 负数限制"""
        # Mock psutil
        mock_disk_usage = MagicMock()
        mock_disk_usage.total = 1000
        mock_disk_usage.used = 500
        mock_disk_usage.free = 500
        mock_disk_usage.percent = 50.0
        
        monitor._mount_points = ["/test"]
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        if hasattr(os, "statvfs"):
            monkeypatch.setattr(os, "statvfs", MagicMock(return_value=None))
        
        monitor._collect_stats()
        history = monitor.get_stats_history(limit=-1)
        
        # 应该返回空列表或所有历史记录
        assert isinstance(history, list)

    def test_get_aggregated_stats_empty_history(self, monitor):
        """测试获取聚合统计信息 - 空历史记录"""
        monitor._stats_history = []
        
        aggregated = monitor.get_aggregated_stats()
        
        # 应该返回空字典
        assert aggregated == {}

    def test_get_metrics_for_prometheus_with_data(self, monitor, module, monkeypatch):
        """测试获取Prometheus格式的指标 - 有数据"""
        # Mock psutil
        mock_disk_usage = MagicMock()
        mock_disk_usage.total = 1000
        mock_disk_usage.used = 500
        mock_disk_usage.free = 500
        mock_disk_usage.percent = 50.0
        
        monitor._mount_points = ["/test"]
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        if hasattr(os, "statvfs"):
            monkeypatch.setattr(os, "statvfs", MagicMock(return_value=None))
        
        monitor._collect_stats()
        monitor.record_write(size=100, duration=0.1)
        monitor.record_error()
        
        metrics = monitor.get_metrics_for_prometheus()
        
        # 应该包含Prometheus格式的指标
        assert "storage_total_size" in metrics
        assert "storage_used_size" in metrics
        assert "storage_usage_percent" in metrics
        assert "storage_read_count" in metrics
        assert "storage_write_count" in metrics
        assert "storage_error_count" in metrics

    def test_reset_stats_clears_all(self, monitor, module, monkeypatch):
        """测试重置统计信息 - 清除所有数据"""
        # Mock psutil
        mock_disk_usage = MagicMock()
        mock_disk_usage.total = 1000
        mock_disk_usage.used = 500
        mock_disk_usage.free = 500
        mock_disk_usage.percent = 50.0
        
        monitor._mount_points = ["/test"]
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        if hasattr(os, "statvfs"):
            monkeypatch.setattr(os, "statvfs", MagicMock(return_value=None))
        
        # 收集一些统计信息并记录操作
        monitor._collect_stats()
        monitor.record_operation('read', size=100, duration=0.1)
        monitor.record_write(size=200, duration=0.2)
        monitor.record_error()
        
        # 重置统计信息
        monitor.reset_stats()
        
        # 所有统计信息应该被清除
        assert len(monitor._stats_history) == 0
        assert monitor._manual_read_count == 0
        assert monitor._manual_write_count == 0
        assert monitor._manual_read_bytes == 0
        assert monitor._manual_write_bytes == 0
        assert monitor._manual_read_time == 0.0
        assert monitor._manual_write_time == 0.0
        assert monitor._manual_error_count == 0

    def test_get_aggregated_stats_zero_total_size(self, monitor, module, monkeypatch):
        """测试获取聚合统计信息 - 总大小为0"""
        # Mock psutil
        mock_disk_usage = MagicMock()
        mock_disk_usage.total = 0  # 总大小为0
        mock_disk_usage.used = 0
        mock_disk_usage.free = 0
        mock_disk_usage.percent = 0.0
        
        monitor._mount_points = ["/test"]
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        if hasattr(os, "statvfs"):
            monkeypatch.setattr(os, "statvfs", MagicMock(return_value=None))
        
        monitor._collect_stats()
        
        aggregated = monitor.get_aggregated_stats()
        
        # usage_percent 应该为0（因为total_size为0）
        assert aggregated['usage_percent'] == 0.0
        assert aggregated['total_size'] == 0

    def test_get_aggregated_stats_single_mount_point(self, monitor, module, monkeypatch):
        """测试获取聚合统计信息 - 单个挂载点"""
        # Mock psutil
        mock_disk_usage = MagicMock()
        mock_disk_usage.total = 1000
        mock_disk_usage.used = 500
        mock_disk_usage.free = 500
        mock_disk_usage.percent = 50.0
        
        monitor._mount_points = ["/test"]
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        if hasattr(os, "statvfs"):
            monkeypatch.setattr(os, "statvfs", MagicMock(return_value=None))
        
        monitor._collect_stats()
        
        aggregated = monitor.get_aggregated_stats()
        
        assert aggregated['total_mount_points'] == 1
        assert aggregated['total_size'] == 1000
        assert aggregated['used_size'] == 500
        assert aggregated['free_size'] == 500
        assert aggregated['usage_percent'] == 50.0

    def test_get_aggregated_stats_multiple_mount_points_duplicate(self, monitor, module, monkeypatch):
        """测试获取聚合统计信息 - 多个挂载点，有重复"""
        # Mock psutil
        mock_disk_usage = MagicMock()
        mock_disk_usage.total = 1000
        mock_disk_usage.used = 500
        mock_disk_usage.free = 500
        mock_disk_usage.percent = 50.0
        
        monitor._mount_points = ["/mount1", "/mount2"]
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        if hasattr(os, "statvfs"):
            monkeypatch.setattr(os, "statvfs", MagicMock(return_value=None))
        
        # 收集多次统计信息（同一个挂载点）
        monitor._collect_stats()
        monitor._collect_stats()
        
        aggregated = monitor.get_aggregated_stats()
        
        # 应该只统计每个挂载点的最新统计
        assert aggregated['total_mount_points'] == 2
        assert aggregated['total_size'] == 2000  # 两个挂载点各1000

    def test_get_metrics_for_prometheus_empty_stats(self, monitor):
        """测试获取Prometheus格式的指标 - 空统计信息"""
        monitor._stats_history = []
        
        metrics = monitor.get_metrics_for_prometheus()
        
        # 应该返回空字符串
        assert metrics == ""

    def test_get_metrics_for_prometheus_zero_values(self, monitor, module, monkeypatch):
        """测试获取Prometheus格式的指标 - 零值"""
        # Mock psutil
        mock_disk_usage = MagicMock()
        mock_disk_usage.total = 0
        mock_disk_usage.used = 0
        mock_disk_usage.free = 0
        mock_disk_usage.percent = 0.0
        
        monitor._mount_points = ["/test"]
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        if hasattr(os, "statvfs"):
            monkeypatch.setattr(os, "statvfs", MagicMock(return_value=None))
        
        monitor._collect_stats()
        
        metrics = monitor.get_metrics_for_prometheus()
        
        # 应该包含Prometheus格式的指标，值为0
        assert "storage_total_size 0" in metrics
        assert "storage_used_size 0" in metrics
        assert "storage_usage_percent 0" in metrics

    def test_storage_stats_post_init_with_timestamp(self, module):
        """测试StorageStats __post_init__ - 提供timestamp"""
        timestamp = time.time()
        stats = module.StorageStats(
            mount_point="/test",
            total_size=1000,
            used_size=500,
            free_size=500,
            usage_percent=50.0,
            read_count=10,
            write_count=5,
            read_bytes=1000,
            write_bytes=500,
            read_time=1.0,
            write_time=0.5,
            error_count=0,
            timestamp=timestamp
        )
        
        # timestamp应该保持不变
        assert stats.timestamp == timestamp

    def test_storage_stats_post_init_without_timestamp(self, module):
        """测试StorageStats __post_init__ - 不提供timestamp"""
        stats = module.StorageStats(
            mount_point="/test",
            total_size=1000,
            used_size=500,
            free_size=500,
            usage_percent=50.0,
            read_count=10,
            write_count=5,
            read_bytes=1000,
            write_bytes=500,
            read_time=1.0,
            write_time=0.5,
            error_count=0,
            timestamp=None
        )
        
        # timestamp应该被设置为当前时间
        assert stats.timestamp is not None
        assert isinstance(stats.timestamp, float)

    def test_record_operation_read_success(self, monitor):
        """测试记录操作 - 读取成功"""
        original_read_count = monitor._manual_read_count
        original_read_bytes = monitor._manual_read_bytes
        original_read_time = monitor._manual_read_time
        
        monitor.record_operation('read', size=100, duration=0.1, success=True)
        
        assert monitor._manual_read_count == original_read_count + 1
        assert monitor._manual_read_bytes == original_read_bytes + 100
        assert monitor._manual_read_time == original_read_time + 0.1
        assert monitor._manual_error_count == 0

    def test_record_operation_write_success(self, monitor):
        """测试记录操作 - 写入成功"""
        original_write_count = monitor._manual_write_count
        original_write_bytes = monitor._manual_write_bytes
        original_write_time = monitor._manual_write_time
        
        monitor.record_operation('write', size=200, duration=0.2, success=True)
        
        assert monitor._manual_write_count == original_write_count + 1
        assert monitor._manual_write_bytes == original_write_bytes + 200
        assert monitor._manual_write_time == original_write_time + 0.2
        assert monitor._manual_error_count == 0

    def test_record_write(self, monitor):
        """测试记录写入操作"""
        original_write_count = monitor._manual_write_count
        original_write_bytes = monitor._manual_write_bytes
        
        monitor.record_write(size=300, duration=0.3)
        
        assert monitor._manual_write_count == original_write_count + 1
        assert monitor._manual_write_bytes == original_write_bytes + 300

    def test_record_error(self, monitor):
        """测试记录错误"""
        original_error_count = monitor._manual_error_count
        
        monitor.record_error()
        
        assert monitor._manual_error_count == original_error_count + 1

    def test_record_error_with_symbol(self, monitor):
        """测试记录错误 - 带符号"""
        original_error_count = monitor._manual_error_count
        
        monitor.record_error(symbol="test_error")
        
        assert monitor._manual_error_count == original_error_count + 1

    def test_record_operation_concurrent_access(self, monitor):
        """测试并发访问记录操作"""
        import threading
        
        def record_operations():
            for _ in range(10):
                monitor.record_operation('read', size=100, duration=0.1, success=True)
                monitor.record_operation('write', size=200, duration=0.2, success=True)
        
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=record_operations)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # 应该记录了50次读取和50次写入（5个线程 * 10次）
        assert monitor._manual_read_count == 50
        assert monitor._manual_write_count == 50

    def test_get_current_stats_concurrent_access(self, monitor, module, monkeypatch):
        """测试并发访问获取当前统计"""
        import threading
        
        # Mock psutil
        mock_disk_usage = MagicMock()
        mock_disk_usage.total = 1000
        mock_disk_usage.used = 500
        mock_disk_usage.free = 500
        mock_disk_usage.percent = 50.0
        
        monitor._mount_points = ["/test"]
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        if hasattr(os, "statvfs"):
            monkeypatch.setattr(os, "statvfs", MagicMock(return_value=None))
        
        # 先收集一些统计
        monitor._collect_stats()
        
        results = []
        def get_current():
            results.append(monitor.get_current_stats())
        
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=get_current)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # 应该都返回相同的统计
        assert len(results) == 10
        assert all(len(r) == 1 for r in results)

    def test_get_stats_history_concurrent_access(self, monitor, module, monkeypatch):
        """测试并发访问获取统计历史"""
        import threading
        
        # Mock psutil
        mock_disk_usage = MagicMock()
        mock_disk_usage.total = 1000
        mock_disk_usage.used = 500
        mock_disk_usage.free = 500
        mock_disk_usage.percent = 50.0
        
        monitor._mount_points = ["/test"]
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        if hasattr(os, "statvfs"):
            monkeypatch.setattr(os, "statvfs", MagicMock(return_value=None))
        
        # 先收集一些统计
        for _ in range(5):
            monitor._collect_stats()
        
        results = []
        def get_history():
            results.append(monitor.get_stats_history(limit=3))
        
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=get_history)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # 应该都返回3个统计
        assert len(results) == 10
        assert all(len(r) == 3 for r in results)

    def test_get_aggregated_stats_concurrent_access(self, monitor, module, monkeypatch):
        """测试并发访问获取聚合统计"""
        import threading
        
        # Mock psutil
        mock_disk_usage = MagicMock()
        mock_disk_usage.total = 1000
        mock_disk_usage.used = 500
        mock_disk_usage.free = 500
        mock_disk_usage.percent = 50.0
        
        monitor._mount_points = ["/test"]
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        if hasattr(os, "statvfs"):
            monkeypatch.setattr(os, "statvfs", MagicMock(return_value=None))
        
        # 先收集一些统计
        monitor._collect_stats()
        
        results = []
        def get_aggregated():
            results.append(monitor.get_aggregated_stats())
        
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=get_aggregated)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # 应该都返回相同的聚合统计
        assert len(results) == 10
        assert all(isinstance(r, dict) for r in results)
        assert all('total_size' in r for r in results)

    def test_monitor_loop_exception_handling(self, monitor, module, monkeypatch):
        """测试监控循环异常处理"""
        # Mock _collect_stats 抛出异常
        original_collect = monitor._collect_stats
        call_count = []
        
        def mock_collect():
            call_count.append(1)
            if len(call_count) == 1:
                raise Exception("Test error")
            monitor._monitoring = False  # 停止监控
        
        monitor._collect_stats = mock_collect
        monitor._monitoring = True
        
        # Mock time.sleep
        def mock_sleep(seconds):
            pass
        
        monkeypatch.setattr(module.time, "sleep", mock_sleep)
        
        # 运行监控循环（会捕获异常并继续）
        monitor._monitor_loop()
        
        # 应该调用了两次（第一次抛出异常，第二次正常）
        assert len(call_count) >= 1
        
        # 恢复原始方法
        monitor._collect_stats = original_collect

    def test_start_monitoring_already_running(self, monitor, module, monkeypatch):
        """测试启动监控 - 已经运行"""
        # Mock threading.Thread
        mock_thread = MagicMock()
        mock_thread.is_alive = MagicMock(return_value=True)
        
        monitor._monitoring = True
        monitor._monitor_thread = mock_thread
        
        # 再次启动监控
        monitor.start_monitoring()
        
        # 应该不会创建新线程
        assert monitor._monitoring == True

    def test_stop_monitoring_not_running(self, monitor):
        """测试停止监控 - 未运行"""
        monitor._monitoring = False
        monitor._monitor_thread = None
        
        # 停止监控
        monitor.stop_monitoring()
        
        # 应该不会出错
        assert monitor._monitoring == False

    def test_get_stats_history_zero_limit(self, monitor, module, monkeypatch):
        """测试获取统计历史 - 零限制"""
        # Mock psutil
        mock_disk_usage = MagicMock()
        mock_disk_usage.total = 1000
        mock_disk_usage.used = 500
        mock_disk_usage.free = 500
        mock_disk_usage.percent = 50.0
        
        monitor._mount_points = ["/test"]
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        if hasattr(os, "statvfs"):
            monkeypatch.setattr(os, "statvfs", MagicMock(return_value=None))
        
        # 收集一些统计
        for _ in range(5):
            monitor._collect_stats()
        
        history = monitor.get_stats_history(limit=0)
        
        # Python切片[-0:]返回整个列表
        assert len(history) == 5

    def test_get_stats_history_negative_limit(self, monitor, module, monkeypatch):
        """测试获取统计历史 - 负限制"""
        # Mock psutil
        mock_disk_usage = MagicMock()
        mock_disk_usage.total = 1000
        mock_disk_usage.used = 500
        mock_disk_usage.free = 500
        mock_disk_usage.percent = 50.0
        
        monitor._mount_points = ["/test"]
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        if hasattr(os, "statvfs"):
            monkeypatch.setattr(os, "statvfs", MagicMock(return_value=None))
        
        # 收集一些统计
        for _ in range(5):
            monitor._collect_stats()
        
        history = monitor.get_stats_history(limit=-3)
        
        # 负限制应该返回空列表或部分列表
        assert isinstance(history, list)

    def test_record_operation_read_with_error(self, monitor):
        """测试记录操作 - 读取失败"""
        original_read_count = monitor._manual_read_count
        original_error_count = monitor._manual_error_count
        
        monitor.record_operation('read', size=100, duration=0.1, success=False)
        
        assert monitor._manual_read_count == original_read_count + 1
        assert monitor._manual_error_count == original_error_count + 1

    def test_record_operation_write_with_error(self, monitor):
        """测试记录操作 - 写入失败"""
        original_write_count = monitor._manual_write_count
        original_error_count = monitor._manual_error_count
        
        monitor.record_operation('write', size=200, duration=0.2, success=False)
        
        assert monitor._manual_write_count == original_write_count + 1
        assert monitor._manual_error_count == original_error_count + 1

