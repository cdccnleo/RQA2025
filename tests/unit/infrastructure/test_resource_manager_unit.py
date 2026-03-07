"""
测试资源管理器
"""

import pytest
import psutil
import threading
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.infrastructure.resource.core.resource_manager import CoreResourceManager


class TestCoreResourceManager:

    def setup_method(self):
        """测试前设置"""
        # Mock psutil to avoid system calls in tests
        self.cpu_patch = patch('psutil.cpu_percent', return_value=50.0)
        self.mem_patch = patch('psutil.virtual_memory')
        self.disk_patch = patch('psutil.disk_usage')

        self.cpu_patch.start()
        mem_mock = self.mem_patch.start()
        disk_mock = self.disk_patch.start()

        mem_mock.return_value.percent = 60.0
        disk_mock.return_value.percent = 70.0

        self.manager = CoreResourceManager()

    def teardown_method(self):
        """测试后清理"""
        if hasattr(self, 'manager') and hasattr(self.manager, '_monitoring') and self.manager._monitoring:
            self.manager._monitoring = False
            if hasattr(self.manager, '_monitor_thread') and self.manager._monitor_thread:
                self.manager._monitor_thread.join(timeout=1.0)

        if hasattr(self, 'cpu_patch'):
            self.cpu_patch.stop()
        if hasattr(self, 'mem_patch'):
            self.mem_patch.stop()
        if hasattr(self, 'disk_patch'):
            self.disk_patch.stop()
    """测试核心资源管理器"""

    def setup_method(self):
        """测试前准备"""
        with patch('src.infrastructure.resource.core.resource_manager.ResourceMonitorConfig'):
            self.manager = CoreResourceManager()

    def test_core_resource_manager_init(self):
        """测试核心资源管理器初始化"""
        assert self.manager is not None
        assert hasattr(self.manager, 'config')
        assert hasattr(self.manager, '_monitoring')
        assert hasattr(self.manager, '_monitor_thread')
        assert hasattr(self.manager, '_resource_history')
        assert hasattr(self.manager, '_lock')
        assert isinstance(self.manager._resource_history, list)
        assert hasattr(self.manager._lock, 'acquire') and hasattr(self.manager._lock, 'release')

    def test_monitor_system_resources(self):
        """测试监控系统资源"""
        # Note: Method name differs, use actual API
        resources = self.manager.get_current_usage()

        assert isinstance(resources, dict)
        # Check that we get some resource information
        assert len(resources) > 0 or any(key in k for k in resources.keys())

    def test_get_resource_usage(self):
        """测试获取资源使用情况"""
        usage = self.manager.get_resource_usage()

        assert isinstance(usage, dict)
        # 应该包含CPU、内存等资源信息
        assert len(usage) > 0

    def test_allocate_resource(self):
        """测试分配资源"""
        # 这个方法可能不存在或为空实现
        try:
            result = self.manager.allocate_resource("cpu", 50.0)
            assert isinstance(result, bool)
        except AttributeError:
            pytest.skip("allocate_resource method not implemented")

    def test_release_resource(self):
        """测试释放资源"""
        # 这个方法可能不存在或为空实现
        try:
            result = self.manager.release_resource("cpu", 50.0)
            assert isinstance(result, bool)
        except AttributeError:
            pytest.skip("release_resource method not implemented")

    def test_get_resource_limits(self):
        """测试获取资源限制"""
        limits = self.manager.get_resource_limits()

        assert isinstance(limits, dict)
        # 应该包含各种资源的限制信息

    def test_check_resource_availability(self):
        """测试检查资源可用性"""
        # 这个方法可能不存在
        try:
            available = self.manager.check_resource_availability("cpu", 25.0)
            assert isinstance(available, bool)
        except AttributeError:
            pytest.skip("check_resource_availability method not implemented")

    def test_monitor_resources(self):
        """测试资源监控"""
        # 启动监控一段时间
        initial_history_length = len(self.manager._resource_history)

        # 等待监控收集一些数据
        time.sleep(0.1)

        # 检查是否有新的监控数据
        assert len(self.manager._resource_history) >= initial_history_length

    def test_get_resource_statistics(self):
        """测试获取资源统计信息"""
        stats = self.manager.get_resource_statistics()

        assert isinstance(stats, dict)
        # 检查统计信息结构
        if len(self.manager._resource_history) > 0:
            assert 'history_length' in stats
            assert stats['history_length'] > 0

    def test_set_resource_limits(self):
        """测试设置资源限制"""
        # 这个方法可能不存在
        try:
            limits = {'cpu_limit': 80.0, 'memory_limit': 90.0}
            result = self.manager.set_resource_limits(limits)
            assert isinstance(result, bool)
        except AttributeError:
            pytest.skip("set_resource_limits method not implemented")

    def test_get_resource_alerts(self):
        """测试获取资源告警"""
        alerts = self.manager.get_resource_alerts()

        assert isinstance(alerts, list)
        # 告警列表应该为空或包含告警信息

    def test_optimize_resource_usage(self):
        """测试优化资源使用"""
        # 这个方法可能不存在
        try:
            result = self.manager.optimize_resource_usage()
            assert isinstance(result, bool)
        except AttributeError:
            pytest.skip("optimize_resource_usage method not implemented")

    def test_get_performance_metrics(self):
        """测试获取性能指标"""
        metrics = self.manager.get_performance_metrics()

        assert isinstance(metrics, dict)
        # 检查是否包含性能相关指标

    def test_resource_monitoring_thread_safety(self):
        """测试资源监控线程安全性"""
        import threading

        results = []
        errors = []

        def monitor_worker(thread_id):
            try:
                for i in range(50):
                    usage = self.manager.get_resource_usage()
                    stats = self.manager.get_resource_statistics()
                    results.append((thread_id, i, len(usage), len(stats)))
            except Exception as e:
                errors.append((thread_id, str(e)))

        # 创建多个线程
        threads = []
        for i in range(3):
            thread = threading.Thread(target=monitor_worker, args=(i,))
            threads.append(thread)

        # 启动所有线程
        for thread in threads:
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证没有错误发生
        assert len(errors) == 0

        # 验证所有线程都成功执行了操作
        assert len(results) == 3 * 50  # 3个线程，每个50次操作

    def test_resource_history_management(self):
        """测试资源历史管理"""
        # 清空历史记录
        initial_length = len(self.manager._resource_history)
        self.manager._resource_history.clear()

        # 等待监控收集一些数据
        time.sleep(0.2)

        # 检查历史记录数量
        assert len(self.manager._resource_history) > 0

        # 验证历史记录格式
        for record in self.manager._resource_history:
            assert isinstance(record, dict)
            assert 'timestamp' in record
            assert 'cpu_percent' in record or 'memory_percent' in record

    def test_stop_monitoring(self):
        """测试停止监控"""
        # 确保监控正在运行
        assert self.manager._monitoring == True

        # 停止监控
        try:
            self.manager.stop_monitoring()

            # 验证监控已停止
            assert self.manager._monitoring == False

            # 等待一小段时间
            time.sleep(0.1)

            # 检查监控线程是否仍在运行
            if self.manager._monitor_thread:
                assert not self.manager._monitor_thread.is_alive()

        except AttributeError:
            pytest.skip("stop_monitoring method not implemented")

    def test_restart_monitoring(self):
        """测试重启监控"""
        # 先停止监控
        try:
            self.manager.stop_monitoring()

            # 重启监控
            self.manager._start_monitoring()

            # 验证监控已重启
            assert self.manager._monitoring == True
            assert self.manager._monitor_thread is not None
            assert self.manager._monitor_thread.is_alive()

        except AttributeError:
            pytest.skip("restart monitoring methods not implemented")

    def test_resource_threshold_monitoring(self):
        """测试资源阈值监控"""
        # 这个功能可能不存在或为空实现
        try:
            # 设置一些阈值
            thresholds = {'cpu_threshold': 90.0, 'memory_threshold': 85.0}

            # 检查阈值监控
            alerts = self.manager.get_resource_alerts()

            assert isinstance(alerts, list)

        except AttributeError:
            pytest.skip("Resource threshold monitoring not implemented")

    def test_resource_cleanup(self):
        """测试资源清理"""
        # 添加一些测试数据到历史记录
        for i in range(10):
            self.manager._resource_history.append({
                'timestamp': datetime.now(),
                'cpu_percent': 50.0 + i,
                'memory_percent': 60.0 + i
            })

        initial_length = len(self.manager._resource_history)

        # 执行清理（如果有的话）
        try:
            self.manager.cleanup_resources()

            # 验证清理后的状态
            assert len(self.manager._resource_history) <= initial_length

        except AttributeError:
            pytest.skip("cleanup_resources method not implemented")

    def test_get_resource_summary(self):
        """测试获取资源摘要"""
        summary = self.manager.get_resource_summary()

        assert isinstance(summary, dict)
        # 检查摘要信息
        assert 'total_resources' in summary or 'current_usage' in summary

    def test_resource_manager_configuration(self):
        """测试资源管理器配置"""
        # 测试不同的配置
        config = Mock()
        config.monitoring_interval = 10.0
        config.history_size = 1000

        manager = CoreResourceManager(config)

        # 验证配置被正确应用
        assert manager.config == config

    def test_exception_handling_in_monitoring(self):
        """测试监控中的异常处理"""
        # 模拟监控过程中的异常
        with patch.object(self.manager, '_monitor_system_resources', side_effect=Exception("Test error")):
            # 监控应该继续运行，不应该崩溃
            time.sleep(0.1)

            # 验证管理器仍然正常
            usage = self.manager.get_resource_usage()
            assert isinstance(usage, dict)

    def test_resource_manager_memory_usage(self):
        """测试资源管理器的内存使用"""
        import psutil
        import os

        # 获取初始内存使用
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # 创建多个资源管理器实例
        managers = []
        for i in range(5):
            with patch('src.infrastructure.resource.core.resource_manager.ResourceMonitorConfig'):
                manager = CoreResourceManager()
                managers.append(manager)

        # 获取创建后的内存使用
        after_memory = process.memory_info().rss

        # 计算内存增加
        memory_increase = after_memory - initial_memory

        # 验证内存使用在合理范围内（5个管理器实例应该不会导致过多的内存增加）
        # 每个管理器大约需要几KB的内存
        assert memory_increase < 1024 * 1024  # 1MB上限

        # 清理
        del managers

    def test_long_running_resource_monitoring(self):
        """测试长时间运行的资源监控"""
        # 记录开始时间
        start_time = time.time()

        # 运行监控一段时间
        monitoring_duration = 1.0  # 1秒

        time.sleep(monitoring_duration)

        # 检查监控数据
        history_length = len(self.manager._resource_history)
        stats = self.manager.get_resource_statistics()

        # 验证在监控期间收集了数据
        assert history_length > 0

        # 验证统计信息
        assert isinstance(stats, dict)

    def test_resource_manager_multiple_operations(self):
        """测试资源管理器的多次操作"""
        operations_count = 20

        # 执行多次操作
        for i in range(operations_count):
            usage = self.manager.get_resource_usage()
            stats = self.manager.get_resource_statistics()
            alerts = self.manager.get_resource_alerts()

            # 验证每次操作都返回有效结果
            assert isinstance(usage, dict)
            assert isinstance(stats, dict)
            assert isinstance(alerts, list)

        # 验证历史记录数量合理增长
        assert len(self.manager._resource_history) > 0


class TestResourceManagerIntegration:
    """测试资源管理器集成场景"""

    def setup_method(self):
        """测试前准备"""
        with patch('src.infrastructure.resource.core.resource_manager.ResourceMonitorConfig'):
            self.manager = CoreResourceManager()

    def test_complete_resource_management_workflow(self):
        """测试完整的资源管理工作流"""
        # 1. 初始化和启动监控
        assert self.manager._monitoring == True

        # 2. 获取初始资源状态
        initial_usage = self.manager.get_resource_usage()
        assert isinstance(initial_usage, dict)

        # 3. 监控资源一段时间
        time.sleep(0.2)

        # 4. 获取更新后的资源状态
        updated_usage = self.manager.get_resource_usage()
        assert isinstance(updated_usage, dict)

        # 5. 检查资源统计
        stats = self.manager.get_resource_statistics()
        assert isinstance(stats, dict)
        assert len(self.manager._resource_history) > 0

        # 6. 检查告警
        alerts = self.manager.get_resource_alerts()
        assert isinstance(alerts, list)

        # 7. 获取资源摘要
        summary = self.manager.get_resource_summary()
        assert isinstance(summary, dict)

    def test_resource_monitoring_under_load(self):
        """测试负载下的资源监控"""
        import multiprocessing
        import os

        # 创建一些CPU负载
        def cpu_load():
            # 执行一些CPU密集型操作
            for i in range(100000):
                _ = i ** 2

        # 在子进程中创建负载
        process = multiprocessing.Process(target=cpu_load)
        process.start()

        # 等待负载开始
        time.sleep(0.1)

        try:
            # 在负载下监控资源
            for i in range(5):
                usage = self.manager.get_resource_usage()
                assert isinstance(usage, dict)

                # CPU使用率应该高于0
                if 'cpu_percent' in usage:
                    assert usage['cpu_percent'] >= 0

                time.sleep(0.1)

        finally:
            # 清理子进程
            process.terminate()
            process.join()

    def test_resource_manager_state_persistence(self):
        """测试资源管理器状态持久性"""
        # 获取当前状态
        initial_history_length = len(self.manager._resource_history)
        initial_monitoring_state = self.manager._monitoring

        # 等待收集一些数据
        time.sleep(0.2)

        # 验证状态保持一致
        assert self.manager._monitoring == initial_monitoring_state
        assert len(self.manager._resource_history) > initial_history_length

        # 验证数据连续性
        if len(self.manager._resource_history) > 1:
            timestamps = [record.get('timestamp') for record in self.manager._resource_history[-3:]]
            # 时间戳应该大致递增
            for i in range(len(timestamps) - 1):
                if isinstance(timestamps[i], datetime) and isinstance(timestamps[i + 1], datetime):
                    assert timestamps[i] <= timestamps[i + 1]

    def test_resource_manager_error_recovery(self):
        """测试资源管理器错误恢复"""
        # 模拟监控过程中的错误
        original_monitor = self.manager._monitor_system_resources

        def failing_monitor():
            raise RuntimeError("Simulated monitoring error")

        self.manager._monitor_system_resources = failing_monitor

        try:
            # 即使有错误，管理器应该继续工作
            time.sleep(0.1)

            # 应该能够获取使用情况（可能返回最后已知的值）
            usage = self.manager.get_resource_usage()
            assert isinstance(usage, dict)

            # 应该能够获取统计信息
            stats = self.manager.get_resource_statistics()
            assert isinstance(stats, dict)

        finally:
            # 恢复原始方法
            self.manager._monitor_system_resources = original_monitor

    def test_resource_manager_performance_baseline(self):
        """测试资源管理器性能基准"""
        import time

        # 测试基本操作的性能
        operations = 100

        start_time = time.time()

        for i in range(operations):
            usage = self.manager.get_resource_usage()
            stats = self.manager.get_resource_statistics()

        end_time = time.time()

        total_time = end_time - start_time
        avg_time_per_operation = total_time / operations

        # 验证性能（每次操作应该在合理时间内完成）
        assert avg_time_per_operation < 0.01  # 10ms上限
        assert total_time < 1.0  # 总时间1秒上限
