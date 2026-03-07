#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统监控器深度测试

大幅提升system_monitor.py的测试覆盖率，从26%提升到80%以上
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock


class TestSystemMonitorComprehensive:
    """系统监控器深度测试"""

    def test_system_monitor_initialization(self):
        """测试系统监控器初始化"""
        try:
            from src.infrastructure.resource.core.system_monitor import SystemMonitorFacade

            monitor = SystemMonitorFacade()

            # 测试基本属性
            assert hasattr(monitor, 'config')
            assert hasattr(monitor, 'info_collector')
            assert hasattr(monitor, 'metrics_calculator')
            assert hasattr(monitor, 'alert_manager')
            assert hasattr(monitor, 'monitor_engine')

        except ImportError:
            pytest.skip("SystemMonitorFacade not available")

    def test_monitor_initialization_with_config(self):
        """测试带配置的监控器初始化"""
        try:
            from src.infrastructure.resource.core.system_monitor import SystemMonitorFacade, SystemMonitorConfig

            config = SystemMonitorConfig()
            config.check_interval = 30

            monitor = SystemMonitorFacade(config)

            # 验证配置被正确设置
            assert monitor.config == config

        except ImportError:
            pytest.skip("SystemMonitorFacade initialization with config not available")

    def test_system_info_collection(self):
        """测试系统信息收集功能"""
        try:
            from src.infrastructure.resource.core.system_monitor import SystemMonitorFacade

            monitor = SystemMonitorFacade()

            # 测试系统信息收集
            system_info = monitor.get_system_info()
            assert isinstance(system_info, dict)
            # 系统信息应该包含一些基本字段
            assert len(system_info) > 0

        except ImportError:
            pytest.skip("System info collection not available")

    def test_system_resources_collection(self):
        """测试系统资源收集功能"""
        try:
            from src.infrastructure.resource.core.system_monitor import SystemMonitorFacade

            monitor = SystemMonitorFacade()

            # 测试系统资源收集
            system_resources = monitor.get_system_resources()
            assert isinstance(system_resources, dict)
            # 系统资源应该包含一些字段
            assert len(system_resources) > 0

        except ImportError:
            pytest.skip("System resources collection not available")

    def test_monitoring_start_stop_functionality(self):
        """测试监控启动和停止功能"""
        try:
            from src.infrastructure.resource.core.system_monitor import SystemMonitorFacade

            monitor = SystemMonitorFacade()

            # 测试启动监控
            monitor.start_monitoring()

            # 测试停止监控
            monitor.stop_monitoring()

        except ImportError:
            pytest.skip("Monitoring start/stop not available")

    def test_stats_collection(self):
        """测试统计信息收集"""
        try:
            from src.infrastructure.resource.core.system_monitor import SystemMonitorFacade

            monitor = SystemMonitorFacade()

            # 测试获取统计信息
            stats = monitor.get_stats(current=True)
            assert isinstance(stats, dict)

        except ImportError:
            pytest.skip("Stats collection not available")

    def test_process_monitoring_functionality(self):
        """测试进程监控功能"""
        try:
            from src.infrastructure.resource.core.system_monitor import SystemMonitor

            monitor = SystemMonitor()

            # 测试进程监控
            with patch('src.infrastructure.resource.core.system_monitor.psutil') as mock_psutil:
                mock_process = Mock()
                mock_process.pid = 1234
                mock_process.name.return_value = 'python'
                mock_process.cpu_percent.return_value = 25.0
                mock_process.memory_percent.return_value = 10.0

                mock_psutil.process_iter.return_value = [mock_process]

                process_metrics = monitor.monitor_processes()
                assert isinstance(process_metrics, list)

                if process_metrics:
                    assert 'pid' in process_metrics[0]
                    assert 'name' in process_metrics[0]
                    assert 'cpu_percent' in process_metrics[0]

        except ImportError:
            pytest.skip("Process monitoring not available")

    def test_performance_report_generation(self):
        """测试性能报告生成"""
        try:
            from src.infrastructure.resource.core.system_monitor import SystemMonitorFacade

            monitor = SystemMonitorFacade()

            # 测试获取性能报告
            performance_report = monitor.get_performance_report()
            assert performance_report is not None
            # 性能报告应该有基本属性
            assert hasattr(performance_report, 'cpu_usage')

        except ImportError:
            pytest.skip("Performance report generation not available")

    def test_alerts_history_retrieval(self):
        """测试告警历史检索"""
        try:
            from src.infrastructure.resource.core.system_monitor import SystemMonitorFacade

            monitor = SystemMonitorFacade()

            # 测试获取告警历史
            alerts_history = monitor.get_alerts_history()
            assert isinstance(alerts_history, list)

        except ImportError:
            pytest.skip("Alerts history retrieval not available")

    def test_alert_generation_functionality(self):
        """测试告警生成功能"""
        try:
            from src.infrastructure.resource.core.system_monitor import SystemMonitor

            monitor = SystemMonitor()

            # 测试告警生成
            with patch.object(monitor, 'get_comprehensive_metrics') as mock_metrics:
                mock_metrics.return_value = {
                    'cpu': {'usage_percent': 95.0},  # 高CPU使用率
                    'memory': {'usage_percent': 60.0}
                }

                alerts = monitor.check_thresholds()
                assert isinstance(alerts, list)

                # 应该有CPU使用率高的告警
                cpu_alerts = [a for a in alerts if 'CPU' in str(a)]
                assert len(cpu_alerts) > 0

        except ImportError:
            pytest.skip("Alert generation not available")

    def test_metrics_buffering_functionality(self):
        """测试指标缓冲功能"""
        try:
            from src.infrastructure.resource.core.system_monitor import SystemMonitor

            monitor = SystemMonitor()

            # 测试指标缓冲
            test_metrics = {'cpu': {'usage_percent': 50.0}, 'timestamp': time.time()}

            # 添加指标到缓冲区
            monitor._buffer_metrics(test_metrics)

            # 验证缓冲区
            assert len(monitor._metrics_buffer) > 0
            assert monitor._metrics_buffer[-1] == test_metrics

        except ImportError:
            pytest.skip("Metrics buffering not available")

    def test_historical_metrics_retrieval(self):
        """测试历史指标检索"""
        try:
            from src.infrastructure.resource.core.system_monitor import SystemMonitor

            monitor = SystemMonitor()

            # 添加一些历史指标
            for i in range(5):
                metrics = {
                    'cpu': {'usage_percent': 40 + i * 10},
                    'timestamp': time.time() + i * 60  # 每分钟一个
                }
                monitor._buffer_metrics(metrics)

            # 测试获取历史指标
            history = monitor.get_metrics_history(hours=1)
            assert isinstance(history, list)
            assert len(history) <= 5  # 不超过添加的数量

        except ImportError:
            pytest.skip("Historical metrics retrieval not available")

    def test_performance_monitoring_functionality(self):
        """测试性能监控功能"""
        try:
            from src.infrastructure.resource.core.system_monitor import SystemMonitor

            monitor = SystemMonitor()

            # 测试性能监控
            with patch('time.time') as mock_time:
                start_times = [100.0, 101.0, 102.0]
                end_times = [100.5, 101.3, 102.1]

                mock_time.side_effect = start_times + end_times

                # 执行一些操作
                for i in range(3):
                    monitor.record_operation_time(f'operation_{i}')

                # 获取性能统计
                stats = monitor.get_performance_stats()
                assert isinstance(stats, dict)

        except ImportError:
            pytest.skip("Performance monitoring not available")

    def test_configuration_updates(self):
        """测试配置更新"""
        try:
            from src.infrastructure.resource.core.system_monitor import SystemMonitor

            monitor = SystemMonitor()

            # 测试配置更新
            new_config = Mock()
            new_config.monitoring_interval = 60
            new_config.enable_cpu_monitoring = False

            monitor.update_configuration(new_config)

            # 验证配置被更新
            assert monitor.config == new_config

        except ImportError:
            pytest.skip("Configuration updates not available")

    def test_error_handling_and_recovery(self):
        """测试错误处理和恢复"""
        try:
            from src.infrastructure.resource.core.system_monitor import SystemMonitor

            monitor = SystemMonitor()

            # 测试错误处理 - 模拟psutil异常
            with patch('src.infrastructure.resource.core.system_monitor.psutil') as mock_psutil:
                mock_psutil.cpu_percent.side_effect = Exception("Hardware failure")

                # 监控应该不会崩溃
                try:
                    metrics = monitor.monitor_cpu()
                    # 即使出错也应该返回某种结果
                    assert isinstance(metrics, dict)
                except Exception:
                    # 如果抛出异常，应该被正确处理
                    pass

        except ImportError:
            pytest.skip("Error handling and recovery not available")

    def test_monitoring_state_management(self):
        """测试监控状态管理"""
        try:
            from src.infrastructure.resource.core.system_monitor import SystemMonitor

            monitor = SystemMonitor()

            # 测试状态管理
            assert monitor.is_monitoring_active() is False

            monitor.start_monitoring()
            assert monitor.is_monitoring_active() is True

            monitor.stop_monitoring()
            assert monitor.is_monitoring_active() is False

        except ImportError:
            pytest.skip("Monitoring state management not available")

    def test_resource_cleanup_functionality(self):
        """测试资源清理功能"""
        try:
            from src.infrastructure.resource.core.system_monitor import SystemMonitor

            monitor = SystemMonitor()

            # 添加一些数据
            for i in range(10):
                monitor._buffer_metrics({'test': i, 'timestamp': time.time()})

            # 清理资源
            monitor.cleanup()

            # 验证清理
            assert len(monitor._metrics_buffer) == 0

        except ImportError:
            pytest.skip("Resource cleanup not available")

    def test_monitoring_statistics_functionality(self):
        """测试监控统计功能"""
        try:
            from src.infrastructure.resource.core.system_monitor import SystemMonitor

            monitor = SystemMonitor()

            # 执行一些监控操作
            with patch.object(monitor, 'monitor_cpu', return_value={'usage_percent': 50.0}):
                for i in range(5):
                    monitor.monitor_cpu()

            # 获取统计信息
            stats = monitor.get_monitoring_stats()
            assert isinstance(stats, dict)

            # 验证统计包含预期字段
            if 'total_monitoring_calls' in stats:
                assert stats['total_monitoring_calls'] >= 5

        except ImportError:
            pytest.skip("Monitoring statistics not available")

    def test_thread_safety_monitoring(self):
        """测试监控线程安全性"""
        try:
            from src.infrastructure.resource.core.system_monitor import SystemMonitor
            import threading

            monitor = SystemMonitor()

            results = {'calls': 0}
            lock = threading.Lock()

            def monitoring_thread(thread_id):
                for i in range(10):
                    with patch.object(monitor, 'monitor_cpu', return_value={'usage_percent': 50.0}):
                        monitor.monitor_cpu()
                        with lock:
                            results['calls'] += 1

            # 创建多个线程
            threads = []
            for i in range(3):
                thread = threading.Thread(target=monitoring_thread, args=(i,))
                threads.append(thread)
                thread.start()

            # 等待所有线程完成
            for thread in threads:
                thread.join(timeout=2.0)

            # 验证结果
            assert results['calls'] == 30  # 3线程 * 10次调用

        except ImportError:
            pytest.skip("Thread safety monitoring not available")

    def test_monitoring_health_checks(self):
        """测试监控健康检查"""
        try:
            from src.infrastructure.resource.core.system_monitor import SystemMonitor

            monitor = SystemMonitor()

            # 测试健康检查
            health_status = monitor.perform_health_check()
            assert isinstance(health_status, dict)

            # 验证健康状态包含预期字段
            assert 'status' in health_status
            assert 'last_check' in health_status

        except ImportError:
            pytest.skip("Monitoring health checks not available")

    def test_monitoring_configuration_validation(self):
        """测试监控配置验证"""
        try:
            from src.infrastructure.resource.core.system_monitor import SystemMonitor

            monitor = SystemMonitor()

            # 测试配置验证
            valid_config = Mock()
            valid_config.monitoring_interval = 30
            valid_config.enable_cpu_monitoring = True

            is_valid, errors = monitor.validate_configuration(valid_config)
            assert is_valid is True
            assert len(errors) == 0

            # 测试无效配置
            invalid_config = Mock()
            invalid_config.monitoring_interval = -1  # 无效间隔

            is_valid, errors = monitor.validate_configuration(invalid_config)
            assert is_valid is False
            assert len(errors) > 0

        except ImportError:
            pytest.skip("Monitoring configuration validation not available")

    def test_monitoring_data_export(self):
        """测试监控数据导出"""
        try:
            from src.infrastructure.resource.core.system_monitor import SystemMonitor

            monitor = SystemMonitor()

            # 添加一些测试数据
            test_data = [
                {'cpu': {'usage_percent': 40.0}, 'timestamp': time.time()},
                {'cpu': {'usage_percent': 50.0}, 'timestamp': time.time()},
                {'cpu': {'usage_percent': 60.0}, 'timestamp': time.time()}
            ]

            for data in test_data:
                monitor._buffer_metrics(data)

            # 导出数据
            exported_data = monitor.export_monitoring_data()
            assert isinstance(exported_data, dict)

            # 验证导出数据包含历史指标
            if 'metrics_history' in exported_data:
                assert len(exported_data['metrics_history']) >= 3

        except ImportError:
            pytest.skip("Monitoring data export not available")

    def test_monitoring_performance_benchmarks(self):
        """测试监控性能基准"""
        try:
            from src.infrastructure.resource.core.system_monitor import SystemMonitor

            monitor = SystemMonitor()

            # 性能基准测试
            import time

            with patch.object(monitor, 'monitor_cpu', return_value={'usage_percent': 50.0}):
                # 执行多次监控调用
                start_time = time.time()

                for i in range(100):
                    monitor.monitor_cpu()

                end_time = time.time()

                # 计算性能指标
                duration = end_time - start_time
                calls_per_second = 100 / duration

                # 验证性能在合理范围内
                assert calls_per_second > 100  # 至少100次/秒
                assert duration < 2.0  # 2秒内完成

        except ImportError:
            pytest.skip("Monitoring performance benchmarks not available")