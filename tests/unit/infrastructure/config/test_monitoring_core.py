#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试监控核心模块

测试 src/infrastructure/config/monitoring/core.py 文件的功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../../'))

try:
    from src.infrastructure.config.monitoring.core import PerformanceMonitorDashboardCore
    MODULE_AVAILABLE = True
    IMPORT_ERROR = None
except ImportError as e:
    MODULE_AVAILABLE = False
    IMPORT_ERROR = e


@pytest.mark.skipif(not MODULE_AVAILABLE, reason=f"模块导入失败: {IMPORT_ERROR if IMPORT_ERROR else 'Unknown error'}")
class TestPerformanceMonitorDashboardCore:
    """测试性能监控面板核心类"""

    def setup_method(self):
        """测试前准备"""
        self.monitor = PerformanceMonitorDashboardCore(
            storage_path="test_storage",
            retention_days=7
        )

    def test_initialization_default(self):
        """测试默认初始化"""
        monitor = PerformanceMonitorDashboardCore()
        assert monitor.storage_path == "config/performance"
        assert monitor.retention_days == 30
        assert monitor._storage_initialized is False
        assert hasattr(monitor, '_lock')

    def test_initialization_custom(self):
        """测试自定义参数初始化"""
        monitor = PerformanceMonitorDashboardCore(
            storage_path="custom_path",
            retention_days=14
        )
        assert monitor.storage_path == "custom_path"
        assert monitor.retention_days == 14

    @patch('src.infrastructure.config.monitoring.core.logger')
    def test_storage_initialization(self, mock_logger):
        """测试存储初始化"""
        assert self.monitor._storage_initialized is False
        self.monitor._initialize_storage()
        assert self.monitor._storage_initialized is True

    @patch('src.infrastructure.config.monitoring.core.logger')
    def test_start_and_stop(self, mock_logger):
        """测试启动和停止"""
        # 测试启动
        self.monitor.start()
        assert self.monitor._storage_initialized is True
        mock_logger.info.assert_called_with("性能监控面板核心已启动")

        # 测试停止
        mock_logger.reset_mock()
        self.monitor.stop()
        mock_logger.info.assert_called_with("性能监控面板核心已停止")

    def test_record_operation_basic(self):
        """测试记录基本操作"""
        # Mock record_metric方法
        self.monitor.record_metric = Mock()
        
        self.monitor.record_operation("test_operation", 1.5, success=True)
        
        # 验证调用了record_metric
        assert self.monitor.record_metric.call_count >= 2
        self.monitor.record_metric.assert_any_call("operation.test_operation", 1.5)
        self.monitor.record_metric.assert_any_call("operation.test_operation.status", 1)

    def test_record_operation_with_metadata(self):
        """测试记录带元数据的操作"""
        self.monitor.record_metric = Mock()
        
        metadata = {"user_id": "123", "request_id": "abc"}
        self.monitor.record_operation("test_operation", 2.0, success=False, metadata=metadata)
        
        # 验证基本记录
        self.monitor.record_metric.assert_any_call("operation.test_operation", 2.0)
        self.monitor.record_metric.assert_any_call("operation.test_operation.status", 0)
        
        # 验证元数据记录
        self.monitor.record_metric.assert_any_call("operation.test_operation.metadata.user_id", "123")
        self.monitor.record_metric.assert_any_call("operation.test_operation.metadata.request_id", "abc")

    def test_record_operation_auto_initializes_storage(self):
        """测试记录操作时自动初始化存储"""
        assert self.monitor._storage_initialized is False
        
        self.monitor.record_metric = Mock()
        self.monitor.record_operation("test", 1.0)
        
        assert self.monitor._storage_initialized is True

    def test_get_operation_stats(self):
        """测试获取操作统计"""
        stats = self.monitor.get_operation_stats()
        
        assert isinstance(stats, dict)
        assert "total_operations" in stats
        assert "success_rate" in stats
        assert "avg_duration" in stats
        assert stats["total_operations"] == 0
        assert stats["success_rate"] == 1.0
        assert stats["avg_duration"] == 0.0

    @patch('src.infrastructure.config.monitoring.core.psutil')
    def test_get_system_health_status_with_psutil(self, mock_psutil):
        """测试获取系统健康状态（有psutil）"""
        # Mock psutil数据 - 正确设置Mock对象
        mock_cpu = 45.0
        mock_memory = Mock()
        mock_memory.percent = 60.0
        mock_memory.used = 1024**3  # 1GB
        mock_memory.total = 2 * 1024**3  # 2GB
        
        mock_disk = Mock()
        mock_disk.percent = 40.0
        mock_disk.free = 5 * 1024**3  # 5GB
        
        mock_psutil.cpu_percent.return_value = mock_cpu
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.disk_usage.return_value = mock_disk
        mock_psutil.pids.return_value = [1, 2, 3, 4, 5]  # 模拟5个进程
        
        status = self.monitor.get_system_health_status()
        
        assert isinstance(status, dict)
        assert "status" in status
        if "health_score" in status:
            assert "cpu_usage" in status
            assert "memory_usage" in status
            assert "disk_usage" in status

    def test_get_system_health_status_without_psutil(self):
        """测试获取系统健康状态（无psutil）"""
        with patch('src.infrastructure.config.monitoring.core.psutil', None):
            status = self.monitor.get_system_health_status()
            
            assert isinstance(status, dict)
            assert status["status"] == "unknown"
            assert "psutil not available" in status["error"]
            assert "timestamp" in status

    def test_calculate_health_score(self):
        """测试健康评分计算"""
        # 测试正常情况
        score = self.monitor._calculate_health_score(30.0, 50.0, 40.0)
        assert isinstance(score, (int, float))
        assert 0 <= score <= 100

        # 测试高负载情况
        score_high = self.monitor._calculate_health_score(90.0, 85.0, 80.0)
        assert score_high < score  # 高负载应该获得更低分数

    def test_get_memory_leak_detection(self):
        """测试内存泄漏检测"""
        result = self.monitor.get_memory_leak_detection()
        
        assert isinstance(result, dict)
        assert "timestamp" in result

    def test_get_connection_pool_metrics(self):
        """测试连接池监控指标"""
        result = self.monitor.get_connection_pool_metrics()
        
        assert isinstance(result, dict)
        # 检查是否有预期的键
        if "database_connections" in result:
            assert isinstance(result["database_connections"], dict)


class TestPerformanceMonitorDashboardCoreIntegration:
    """测试性能监控面板核心集成功能"""

    def setup_method(self):
        """测试前准备"""
        if MODULE_AVAILABLE:
            self.monitor = PerformanceMonitorDashboardCore()

    def test_full_lifecycle(self):
        """测试完整生命周期"""
        if not MODULE_AVAILABLE:
            pytest.skip("模块不可用")
        
        # Mock record_metric以避免实际调用
        self.monitor.record_metric = Mock()
        
        # 启动
        self.monitor.start()
        # 检查启动状态，但不依赖于is_running方法
        assert self.monitor._storage_initialized is True
        
        # 记录一些操作
        self.monitor.record_operation("test_op", 1.0, True)
        self.monitor.record_operation("test_op", 2.0, False)
        
        # 获取统计信息
        stats = self.monitor.get_operation_stats()
        assert isinstance(stats, dict)
        
        # 停止
        self.monitor.stop()

    def test_concurrent_operations(self):
        """测试并发操作"""
        if not MODULE_AVAILABLE:
            pytest.skip("模块不可用")
        
        import threading
        
        self.monitor.record_metric = Mock()
        results = []
        
        def record_operation(operation_id):
            self.monitor.record_operation(f"op_{operation_id}", 1.0, True)
            results.append(operation_id)
        
        # 创建多个线程同时记录操作
        threads = []
        for i in range(5):
            thread = threading.Thread(target=record_operation, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证所有操作都被记录
        assert len(results) == 5
        assert set(results) == {0, 1, 2, 3, 4}


def mock_open():
    """创建mock的open函数"""
    from unittest.mock import mock_open as original_mock_open
    return original_mock_open
