#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试业务流程监控

测试目标：提升business_process/monitor/monitor.py的覆盖率到100%
"""

import pytest

# 尝试导入所需模块
try:
    from src.core.business_process.monitor.monitor import ProcessMonitor
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
import time
import threading
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from src.core.business_process.monitor.monitor import BusinessMonitor
from src.core.business_process.config.enums import BusinessProcessState
from src.core.business_process.monitor.business_process_models import ProcessInstance


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestBusinessMonitor:
    """测试业务监控器"""

    @pytest.fixture
    def business_monitor(self):
        """创建业务监控器实例"""
        return BusinessMonitor()

    @pytest.fixture
    def sample_process_info(self):
        """创建示例流程信息"""
        return {
            'process_type': 'data_processing',
            'priority': 'high',
            'description': 'Sample data processing workflow'
        }

    def test_business_monitor_initialization(self, business_monitor):
        """测试业务监控器初始化"""
        assert hasattr(business_monitor, '_active_processes')
        assert hasattr(business_monitor, '_performance_metrics')
        assert hasattr(business_monitor, '_alerts')
        assert isinstance(business_monitor._active_processes, dict)
        assert isinstance(business_monitor._performance_metrics, dict)
        assert isinstance(business_monitor._alerts, list)

    def test_register_process(self, business_monitor, sample_process_info):
        """测试注册业务流程"""
        process_id = "test_process_001"

        business_monitor.register_process(process_id, sample_process_info)

        assert process_id in business_monitor._active_processes
        process_data = business_monitor._active_processes[process_id]

        assert process_data['process_id'] == process_id
        assert process_data['status'] == 'running'
        assert 'start_time' in process_data
        assert isinstance(process_data['start_time'], datetime)

    def test_update_process_status(self, business_monitor, sample_process_info):
        """测试更新流程状态"""
        process_id = "test_process_002"
        business_monitor.register_process(process_id, sample_process_info)

        new_status = "completed"
        metrics = {"throughput": 100, "latency": 50}

        business_monitor.update_process_status(process_id, new_status, metrics)

        process_data = business_monitor._active_processes[process_id]
        assert process_data['status'] == new_status
        assert process_data['metrics'] == metrics
        assert 'last_update' in process_data

    def test_update_process_status_not_registered(self, business_monitor):
        """测试更新未注册的流程状态"""
        process_id = "nonexistent_process"

        # 应该不会引发异常，但也不会更新任何内容
        business_monitor.update_process_status(process_id, "completed")

        assert process_id not in business_monitor._active_processes

    def test_unregister_process(self, business_monitor, sample_process_info):
        """测试注销业务流程"""
        process_id = "test_process_003"
        business_monitor.register_process(process_id, sample_process_info)

        assert process_id in business_monitor._active_processes

        business_monitor.unregister_process(process_id)

        process_data = business_monitor._active_processes[process_id]
        assert process_data['status'] == 'completed'
        assert 'end_time' in process_data

    def test_unregister_process_not_registered(self, business_monitor):
        """测试注销未注册的流程"""
        process_id = "nonexistent_process"

        # 应该不会引发异常
        business_monitor.unregister_process(process_id)

        assert process_id not in business_monitor._active_processes

    def test_get_process_status(self, business_monitor, sample_process_info):
        """测试获取流程状态"""
        process_id = "test_process_004"
        business_monitor.register_process(process_id, sample_process_info)

        status = business_monitor.get_process_status(process_id)

        assert isinstance(status, dict)
        assert status['process_id'] == process_id
        assert status['status'] == 'running'

    def test_get_process_status_not_found(self, business_monitor):
        """测试获取不存在的流程状态"""
        status = business_monitor.get_process_status("nonexistent")

        assert status is None

    def test_get_active_processes(self, business_monitor, sample_process_info):
        """测试获取活跃流程"""
        # 注册多个流程
        for i in range(3):
            process_id = f"active_process_{i}"
            business_monitor.register_process(process_id, sample_process_info)

        active_processes = business_monitor.get_active_processes()

        assert isinstance(active_processes, list)
        assert len(active_processes) == 3

        # 验证所有流程都是活跃的
        for process in active_processes:
            assert process['status'] == 'running'

    def test_get_completed_processes(self, business_monitor, sample_process_info):
        """测试获取已完成流程"""
        # 注册并完成一些流程
        for i in range(2):
            process_id = f"completed_process_{i}"
            business_monitor.register_process(process_id, sample_process_info)
            business_monitor.unregister_process(process_id)

        completed_processes = business_monitor.get_completed_processes()

        assert isinstance(completed_processes, list)
        assert len(completed_processes) == 2

        for process in completed_processes:
            assert process['status'] == 'completed'

    def test_collect_performance_metrics(self, business_monitor, sample_process_info):
        """测试收集性能指标"""
        # 注册一些流程
        for i in range(2):
            process_id = f"perf_process_{i}"
            business_monitor.register_process(process_id, sample_process_info)

            # 更新性能指标
            metrics = {
                "cpu_usage": 45.5 + i * 10,
                "memory_usage": 256 + i * 50,
                "throughput": 100 + i * 20
            }
            business_monitor.update_process_status(process_id, "running", metrics)

        # 收集性能指标
        performance_data = business_monitor.collect_performance_metrics()

        assert isinstance(performance_data, dict)
        assert "timestamp" in performance_data
        assert "active_processes" in performance_data
        assert "average_cpu" in performance_data
        assert "average_memory" in performance_data
        assert "total_throughput" in performance_data

    def test_collect_performance_metrics_no_processes(self, business_monitor):
        """测试收集性能指标 - 无活跃流程"""
        performance_data = business_monitor.collect_performance_metrics()

        assert isinstance(performance_data, dict)
        assert performance_data["active_processes"] == 0
        assert performance_data["average_cpu"] == 0.0

    def test_generate_alert(self, business_monitor, sample_process_info):
        """测试生成告警"""
        process_id = "alert_process_001"
        business_monitor.register_process(process_id, sample_process_info)

        alert_type = "high_cpu"
        message = "CPU usage exceeded threshold"
        severity = "warning"

        business_monitor.generate_alert(process_id, alert_type, message, severity)

        assert len(business_monitor._alerts) == 1

        alert = business_monitor._alerts[0]
        assert alert['process_id'] == process_id
        assert alert['alert_type'] == alert_type
        assert alert['message'] == message
        assert alert['severity'] == severity
        assert 'timestamp' in alert

    def test_get_alerts(self, business_monitor, sample_process_info):
        """测试获取告警"""
        # 生成一些告警
        for i in range(3):
            process_id = f"alert_process_{i}"
            business_monitor.register_process(process_id, sample_process_info)
            business_monitor.generate_alert(
                process_id,
                f"alert_type_{i}",
                f"Alert message {i}",
                "warning" if i % 2 == 0 else "error"
            )

        alerts = business_monitor.get_alerts()

        assert isinstance(alerts, list)
        assert len(alerts) == 3

    def test_get_alerts_by_severity(self, business_monitor, sample_process_info):
        """测试按严重程度获取告警"""
        # 生成不同严重程度的告警
        severities = ["info", "warning", "error", "critical"]

        for i, severity in enumerate(severities):
            process_id = f"severity_process_{i}"
            business_monitor.register_process(process_id, sample_process_info)
            business_monitor.generate_alert(
                process_id,
                "test_alert",
                f"Test alert {i}",
                severity
            )

        # 测试获取特定严重程度的告警
        warning_alerts = business_monitor.get_alerts_by_severity("warning")
        assert len(warning_alerts) == 1

        error_alerts = business_monitor.get_alerts_by_severity("error")
        assert len(error_alerts) == 1

    def test_clear_alerts(self, business_monitor, sample_process_info):
        """测试清除告警"""
        # 生成一些告警
        for i in range(2):
            process_id = f"clear_process_{i}"
            business_monitor.register_process(process_id, sample_process_info)
            business_monitor.generate_alert(
                process_id,
                "test_alert",
                f"Test alert {i}",
                "warning"
            )

        assert len(business_monitor._alerts) == 2

        business_monitor.clear_alerts()

        assert len(business_monitor._alerts) == 0

    def test_health_check(self, business_monitor, sample_process_info):
        """测试健康检查"""
        # 注册一些流程
        for i in range(2):
            process_id = f"health_process_{i}"
            business_monitor.register_process(process_id, sample_process_info)

        health_status = business_monitor.health_check()

        assert isinstance(health_status, dict)
        assert "status" in health_status
        assert "active_processes" in health_status
        assert "total_alerts" in health_status
        assert health_status["active_processes"] == 2

    def test_health_check_unhealthy(self, business_monitor, sample_process_info):
        """测试健康检查 - 不健康状态"""
        # 注册流程并生成严重告警
        process_id = "unhealthy_process"
        business_monitor.register_process(process_id, sample_process_info)

        # 生成多个严重告警
        for i in range(5):
            business_monitor.generate_alert(
                process_id,
                "critical_error",
                f"Critical error {i}",
                "critical"
            )

        health_status = business_monitor.health_check()

        assert health_status["status"] == "unhealthy"
        assert health_status["total_alerts"] == 5

    def test_get_system_metrics(self, business_monitor):
        """测试获取系统指标"""
        with patch('psutil.cpu_percent', return_value=45.5), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk:

            mock_memory.return_value.percent = 67.8
            mock_disk.return_value.percent = 23.4

            system_metrics = business_monitor.get_system_metrics()

            assert isinstance(system_metrics, dict)
            assert "cpu_percent" in system_metrics
            assert "memory_percent" in system_metrics
            assert "disk_percent" in system_metrics
            assert system_metrics["cpu_percent"] == 45.5
            assert system_metrics["memory_percent"] == 67.8

    def test_get_system_metrics_psutil_error(self, business_monitor):
        """测试获取系统指标 - psutil错误"""
        with patch('psutil.cpu_percent', side_effect=Exception("psutil error")):
            system_metrics = business_monitor.get_system_metrics()

            assert isinstance(system_metrics, dict)
            assert system_metrics["cpu_percent"] == 0.0
            assert "error" in system_metrics

    def test_monitor_process_lifecycle(self, business_monitor, sample_process_info):
        """测试监控流程生命周期"""
        process_id = "lifecycle_process"

        # 1. 注册流程
        business_monitor.register_process(process_id, sample_process_info)
        status = business_monitor.get_process_status(process_id)
        assert status['status'] == 'running'

        # 2. 更新状态
        business_monitor.update_process_status(process_id, "processing", {"progress": 50})
        status = business_monitor.get_process_status(process_id)
        assert status['status'] == 'processing'
        assert status['metrics']['progress'] == 50

        # 3. 完成流程
        business_monitor.unregister_process(process_id)
        status = business_monitor.get_process_status(process_id)
        assert status['status'] == 'completed'
        assert 'end_time' in status

    def test_performance_monitoring(self, business_monitor, sample_process_info):
        """测试性能监控"""
        # 注册多个流程并设置不同的性能指标
        processes_data = [
            {"id": "fast_process", "metrics": {"throughput": 200, "latency": 10}},
            {"id": "slow_process", "metrics": {"throughput": 50, "latency": 100}},
            {"id": "normal_process", "metrics": {"throughput": 120, "latency": 30}}
        ]

        for process_data in processes_data:
            business_monitor.register_process(process_data["id"], sample_process_info)
            business_monitor.update_process_status(
                process_data["id"],
                "running",
                process_data["metrics"]
            )

        # 收集性能指标
        perf_data = business_monitor.collect_performance_metrics()

        assert perf_data["active_processes"] == 3
        assert perf_data["total_throughput"] == 370  # 200 + 50 + 120
        assert perf_data["average_latency"] == 46.67  # (10 + 100 + 30) / 3

    def test_alert_management(self, business_monitor, sample_process_info):
        """测试告警管理"""
        # 生成不同类型的告警
        alert_types = ["cpu_high", "memory_low", "disk_full", "network_error"]

        for i, alert_type in enumerate(alert_types):
            process_id = f"alert_test_{i}"
            business_monitor.register_process(process_id, sample_process_info)
            business_monitor.generate_alert(
                process_id,
                alert_type,
                f"Alert for {alert_type}",
                "warning" if i % 2 == 0 else "error"
            )

        # 验证告警数量
        assert len(business_monitor._alerts) == 4

        # 按类型过滤告警
        cpu_alerts = [a for a in business_monitor._alerts if a['alert_type'] == 'cpu_high']
        assert len(cpu_alerts) == 1

        # 按严重程度过滤
        warning_alerts = business_monitor.get_alerts_by_severity("warning")
        error_alerts = business_monitor.get_alerts_by_severity("error")

        assert len(warning_alerts) == 2  # 索引0和2
        assert len(error_alerts) == 2    # 索引1和3

    def test_concurrent_monitoring(self, business_monitor, sample_process_info):
        """测试并发监控"""
        import threading

        results = []
        errors = []

        def monitor_operations(thread_id):
            try:
                # 每个线程注册自己的流程
                process_id = f"thread_{thread_id}_process"
                business_monitor.register_process(process_id, sample_process_info)

                # 更新状态
                business_monitor.update_process_status(
                    process_id,
                    "running",
                    {"thread_id": thread_id, "operation": "test"}
                )

                # 获取状态
                status = business_monitor.get_process_status(process_id)

                results.append({
                    "thread_id": thread_id,
                    "process_id": process_id,
                    "status": status['status']
                })

            except Exception as e:
                errors.append(f"thread_{thread_id}_error: {str(e)}")

        # 创建多个线程并发操作
        threads = []
        for i in range(5):
            thread = threading.Thread(target=monitor_operations, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证没有错误
        assert len(errors) == 0

        # 验证所有线程都成功完成了操作
        assert len(results) == 5

        for result in results:
            assert result['status'] == 'running'

    def test_monitoring_under_load(self, business_monitor, sample_process_info):
        """测试负载下的监控性能"""
        import time

        start_time = time.time()

        # 注册大量流程并执行监控操作
        for i in range(20):
            process_id = f"load_test_{i}"
            business_monitor.register_process(process_id, sample_process_info)

            # 更新状态和指标
            business_monitor.update_process_status(
                process_id,
                "running",
                {"cpu": 40 + i, "memory": 200 + i * 10}
            )

            # 生成告警
            if i % 5 == 0:  # 每5个流程生成一个告警
                business_monitor.generate_alert(
                    process_id,
                    "performance_warning",
                    f"High resource usage for process {i}",
                    "warning"
                )

        # 执行监控操作
        active_processes = business_monitor.get_active_processes()
        performance_data = business_monitor.collect_performance_metrics()
        health_status = business_monitor.health_check()
        alerts = business_monitor.get_alerts()

        end_time = time.time()
        duration = end_time - start_time

        # 验证操作完成且性能合理
        assert len(active_processes) == 20
        assert performance_data["active_processes"] == 20
        assert health_status["active_processes"] == 20
        assert len(alerts) == 4  # 20/5 = 4

        # 性能应该在合理范围内（2秒内完成所有操作）
        assert duration < 2.0

    def test_monitoring_data_persistence_simulation(self, business_monitor, sample_process_info):
        """测试监控数据持久化模拟"""
        # 注册流程并执行各种操作
        process_id = "persistence_test"
        business_monitor.register_process(process_id, sample_process_info)

        # 执行一系列状态更新
        operations = [
            ("running", {"phase": "initialization"}),
            ("processing", {"phase": "data_loading", "progress": 25}),
            ("processing", {"phase": "computation", "progress": 60}),
            ("processing", {"phase": "finalization", "progress": 90}),
        ]

        for status, metrics in operations:
            business_monitor.update_process_status(process_id, status, metrics)
            time.sleep(0.01)  # 模拟时间间隔

        # 完成流程
        business_monitor.unregister_process(process_id)

        # 验证最终状态
        final_status = business_monitor.get_process_status(process_id)
        assert final_status['status'] == 'completed'
        assert 'end_time' in final_status
        assert final_status['metrics']['progress'] == 90

    def test_monitoring_cleanup_and_reset(self, business_monitor, sample_process_info):
        """测试监控清理和重置"""
        # 注册一些流程并生成告警
        for i in range(3):
            process_id = f"cleanup_test_{i}"
            business_monitor.register_process(process_id, sample_process_info)
            business_monitor.generate_alert(
                process_id,
                "test_alert",
                f"Test alert {i}",
                "info"
            )

        # 验证初始状态
        assert len(business_monitor._active_processes) == 3
        assert len(business_monitor._alerts) == 3

        # 完成所有流程
        for i in range(3):
            business_monitor.unregister_process(f"cleanup_test_{i}")

        # 验证完成状态
        completed_processes = business_monitor.get_completed_processes()
        assert len(completed_processes) == 3

        # 清除告警
        business_monitor.clear_alerts()
        assert len(business_monitor._alerts) == 0

    def test_monitoring_resource_tracking(self, business_monitor, sample_process_info):
        """测试监控资源跟踪"""
        # 注册流程并跟踪资源使用
        process_id = "resource_test"
        business_monitor.register_process(process_id, sample_process_info)

        # 模拟不同的资源使用模式
        resource_usage = [
            {"cpu": 20, "memory": 150, "disk_io": 10},
            {"cpu": 45, "memory": 280, "disk_io": 25},
            {"cpu": 80, "memory": 450, "disk_io": 60},  # 高负载
            {"cpu": 35, "memory": 320, "disk_io": 15},
        ]

        for i, resources in enumerate(resource_usage):
            business_monitor.update_process_status(
                process_id,
                "running",
                {"resources": resources, "step": i + 1}
            )

            # 在高负载时生成告警
            if resources["cpu"] > 70:
                business_monitor.generate_alert(
                    process_id,
                    "high_cpu_usage",
                    f"CPU usage spiked to {resources['cpu']}%",
                    "warning"
                )

        # 验证资源跟踪
        status = business_monitor.get_process_status(process_id)
        assert status['metrics']['resources']['cpu'] == 35  # 最后一次更新

        # 验证告警生成
        alerts = business_monitor.get_alerts()
        high_cpu_alerts = [a for a in alerts if a['alert_type'] == 'high_cpu_usage']
        assert len(high_cpu_alerts) == 1

    def test_monitoring_threshold_based_alerting(self, business_monitor, sample_process_info):
        """测试基于阈值的告警"""
        process_id = "threshold_test"
        business_monitor.register_process(process_id, sample_process_info)

        # 定义阈值
        thresholds = {
            "cpu_threshold": 70,
            "memory_threshold": 400,
            "error_rate_threshold": 0.05
        }

        # 测试不同场景
        scenarios = [
            {"cpu": 50, "memory": 300, "errors": 2, "total": 100},  # 正常
            {"cpu": 75, "memory": 350, "errors": 3, "total": 100},  # CPU超阈值
            {"cpu": 60, "memory": 450, "errors": 4, "total": 100},  # 内存超阈值
            {"cpu": 55, "memory": 380, "errors": 8, "total": 100},  # 错误率超阈值
        ]

        expected_alerts = 0

        for metrics in scenarios:
            business_monitor.update_process_status(process_id, "running", metrics)

            # 检查并生成基于阈值的告警
            if metrics["cpu"] > thresholds["cpu_threshold"]:
                business_monitor.generate_alert(
                    process_id,
                    "cpu_threshold_exceeded",
                    f"CPU usage {metrics['cpu']}% exceeds threshold {thresholds['cpu_threshold']}%",
                    "warning"
                )
                expected_alerts += 1

            if metrics["memory"] > thresholds["memory_threshold"]:
                business_monitor.generate_alert(
                    process_id,
                    "memory_threshold_exceeded",
                    f"Memory usage {metrics['memory']}MB exceeds threshold {thresholds['memory_threshold']}MB",
                    "warning"
                )
                expected_alerts += 1

            error_rate = metrics["errors"] / metrics["total"]
            if error_rate > thresholds["error_rate_threshold"]:
                business_monitor.generate_alert(
                    process_id,
                    "error_rate_threshold_exceeded",
                    f"Error rate {error_rate:.1%} exceeds threshold {thresholds['error_rate_threshold']:.1%}",
                    "error"
                )
                expected_alerts += 1

        # 验证告警数量
        alerts = business_monitor.get_alerts()
        assert len(alerts) == expected_alerts

        # 验证不同类型的告警
        cpu_alerts = [a for a in alerts if "cpu" in a['alert_type']]
        memory_alerts = [a for a in alerts if "memory" in a['alert_type']]
        error_alerts = [a for a in alerts if "error_rate" in a['alert_type']]

        assert len(cpu_alerts) == 1
        assert len(memory_alerts) == 1
        assert len(error_alerts) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
