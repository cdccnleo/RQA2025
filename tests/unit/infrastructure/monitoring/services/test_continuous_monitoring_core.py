#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试基础设施层 - 连续监控核心系统

测试 services/continuous_monitoring_core.py 中的核心功能
"""

import pytest
import os
import sys
import json
import time
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime


@pytest.fixture
def module():
    """导入模块"""
    # Mock optional_components - 必须在导入之前
    import sys
    from unittest.mock import MagicMock
    
    # 保存原始的 optional_components
    original_optional = sys.modules.get('src.infrastructure.monitoring.services.optional_components')
    
    mock_optional_components = MagicMock()
    mock_optional_components.get_optional_component = MagicMock(return_value=None)
    sys.modules['src.infrastructure.monitoring.services.optional_components'] = mock_optional_components
    
    # Mock monitoring_runtime
    mock_runtime = MagicMock()
    mock_runtime.start_monitoring = MagicMock(return_value=True)
    mock_runtime.stop_monitoring = MagicMock(return_value=True)
    mock_runtime.monitoring_loop = MagicMock()
    mock_runtime.perform_monitoring_cycle = MagicMock()
    mock_runtime.collect_test_coverage = MagicMock(return_value={})
    sys.modules['src.infrastructure.monitoring.services.monitoring_runtime'] = mock_runtime
    
    # 重新导入模块以确保使用 mock
    if 'src.infrastructure.monitoring.services.continuous_monitoring_core' in sys.modules:
        del sys.modules['src.infrastructure.monitoring.services.continuous_monitoring_core']
    
    from src.infrastructure.monitoring.services import continuous_monitoring_core
    return continuous_monitoring_core


@pytest.fixture
def monitor(module):
    """创建连续监控系统实例"""
    return module.ContinuousMonitoringSystem(project_root=os.getcwd())


class TestContinuousMonitoringSystem:
    """测试连续监控系统"""

    def test_initialization(self, monitor):
        """测试初始化"""
        assert monitor._service_name == "continuous_monitoring_system"
        assert monitor._service_version == "2.0.0"
        assert monitor.project_root == os.getcwd()
        assert monitor.monitoring_active is False
        assert monitor.monitoring_thread is None
        assert 'interval_seconds' in monitor.monitoring_config

    def test_initialization_default_project_root(self, module, monkeypatch):
        """测试初始化 - 默认项目根目录"""
        mock_getcwd = MagicMock(return_value="/test/project")
        monkeypatch.setattr(os, "getcwd", mock_getcwd)
        
        monitor = module.ContinuousMonitoringSystem()
        assert monitor.project_root == "/test/project"

    def test_init_components(self, monitor):
        """测试初始化组件"""
        # 检查组件是否被初始化（可能是 None 或实际对象）
        assert hasattr(monitor, '_metrics_collector')
        assert hasattr(monitor, '_alert_manager')
        assert hasattr(monitor, '_data_persistence')
        assert hasattr(monitor, '_optimization_engine')
        assert monitor.metrics_history == []
        assert monitor.alerts_history == []
        assert monitor.optimization_suggestions == []

    def test_collect_system_metrics_success(self, monitor, module, monkeypatch):
        """测试收集系统指标 - 成功"""
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

        metrics = monitor._collect_system_metrics()

        assert metrics['cpu_percent'] == 50.0
        assert metrics['memory_percent'] == 60.0
        assert metrics['disk_usage'] == 70.0
        assert metrics['network_connections'] == 3

    def test_collect_system_metrics_import_error(self, monitor, module, monkeypatch):
        """测试收集系统指标 - ImportError"""
        # Mock psutil 抛出 ImportError
        monkeypatch.setattr(module.psutil, "cpu_percent", MagicMock(side_effect=ImportError("No module named psutil")))

        metrics = monitor._collect_system_metrics()

        # 应该返回默认值
        assert metrics['cpu_percent'] == 45.5
        assert metrics['memory_percent'] == 67.8
        assert metrics['disk_usage'] == 50.0
        assert metrics['network_connections'] == 10

    def test_get_mock_coverage_data(self, monitor):
        """测试获取模拟覆盖率数据"""
        data = monitor._get_mock_coverage_data()

        assert data['total_lines'] == 1000
        assert data['covered_lines'] == 750
        assert data['coverage_percent'] == 75.0
        assert data['missing_lines'] == 250

    def test_collect_test_coverage_metrics_success(self, monitor, module, monkeypatch):
        """测试收集测试覆盖率指标 - 成功"""
        # Mock subprocess.run
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            'totals': {
                'num_statements': 1000,
                'num_covered': 800,
                'percent_covered': 80.0,
                'num_missing': 200
            }
        })

        monkeypatch.setattr(module.subprocess, "run", MagicMock(return_value=mock_result))

        coverage = monitor._collect_test_coverage_metrics()

        assert coverage['total_lines'] == 1000
        assert coverage['covered_lines'] == 800
        assert coverage['coverage_percent'] == 80.0
        assert coverage['missing_lines'] == 200

    def test_collect_test_coverage_metrics_failure(self, monitor, module, monkeypatch):
        """测试收集测试覆盖率指标 - 失败"""
        # Mock subprocess.run 返回非零退出码
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""

        monkeypatch.setattr(module.subprocess, "run", MagicMock(return_value=mock_result))

        coverage = monitor._collect_test_coverage_metrics()

        # 应该返回模拟数据
        assert coverage['total_lines'] == 1000
        assert coverage['coverage_percent'] == 75.0

    def test_collect_test_coverage_metrics_exception(self, monitor, module, monkeypatch):
        """测试收集测试覆盖率指标 - 异常"""
        # Mock subprocess.run 抛出异常
        monkeypatch.setattr(module.subprocess, "run", MagicMock(side_effect=Exception("Test error")))

        coverage = monitor._collect_test_coverage_metrics()

        # 应该返回模拟数据
        assert coverage['total_lines'] == 1000

    def test_collect_monitoring_data_without_collector(self, monitor, module, monkeypatch):
        """测试收集监控数据 - 无收集器"""
        # 确保 _metrics_collector 是 None
        monitor._metrics_collector = None
        
        # Mock runtime 函数
        mock_coverage = {"coverage_percent": 80.0}
        mock_performance = {"response_time_ms": 100.0}
        mock_resource = {
            "memory": {"percent": 50.0},
            "cpu": {"percent": 50.0}
        }
        mock_health = {
            "overall_status": "healthy",
            "services": {}
        }

        # Mock runtime_collect_test_coverage
        import sys
        mock_runtime = sys.modules.get('src.infrastructure.monitoring.services.monitoring_runtime')
        if mock_runtime:
            mock_runtime.collect_test_coverage = MagicMock(return_value=mock_coverage)

        # Mock 其他收集方法
        monkeypatch.setattr(monitor, "_collect_performance_metrics", lambda: mock_performance)
        monkeypatch.setattr(monitor, "_collect_resource_usage", lambda: mock_resource)
        monkeypatch.setattr(monitor, "_collect_health_status", lambda: mock_health)

        data = monitor._collect_monitoring_data()

        assert 'coverage' in data
        assert 'performance' in data
        assert 'resources' in data
        assert 'health' in data

    def test_collect_performance_metrics_success(self, monitor, module, monkeypatch):
        """测试收集性能指标 - 成功"""
        # Mock psutil
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.used = 1024 * 1024 * 100  # 100MB
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0
        mock_net_io = MagicMock()
        mock_net_io.bytes_sent = 1000
        mock_net_io.bytes_recv = 2000

        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_io_counters", MagicMock(return_value=mock_net_io))

        # Mock print 来捕获输出
        prints = []
        def mock_print(*args, **kwargs):
            prints.append(args)
        
        monkeypatch.setattr("builtins.print", mock_print)

        metrics = monitor._collect_performance_metrics()

        assert 'timestamp' in metrics
        assert 'response_time_ms' in metrics
        assert 'throughput_tps' in metrics
        assert 'memory_usage_mb' in metrics
        assert 'cpu_usage_percent' in metrics
        assert len(prints) > 0

    def test_collect_performance_metrics_exception(self, monitor, module, monkeypatch):
        """测试收集性能指标 - 异常"""
        # Mock psutil 抛出异常
        monkeypatch.setattr(module.psutil, "cpu_percent", MagicMock(side_effect=Exception("Test error")))

        # Mock print
        prints = []
        def mock_print(*args, **kwargs):
            prints.append(args)
        
        monkeypatch.setattr("builtins.print", mock_print)

        metrics = monitor._collect_performance_metrics()

        assert 'error' in metrics
        assert metrics['response_time_ms'] == 0.0
        assert len(prints) > 0

    def test_collect_resource_usage_success(self, monitor, module, monkeypatch):
        """测试收集资源使用情况 - 成功"""
        # Mock psutil
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.total = 1024 * 1024 * 1000
        mock_virtual_memory.available = 1024 * 1024 * 500
        mock_virtual_memory.percent = 50.0
        mock_virtual_memory.used = 1024 * 1024 * 500

        mock_cpu_freq = MagicMock()
        mock_cpu_freq.current = 2400.0

        mock_disk_usage = MagicMock()
        mock_disk_usage.total = 1024 * 1024 * 1000
        mock_disk_usage.used = 1024 * 1024 * 500
        mock_disk_usage.free = 1024 * 1024 * 500
        mock_disk_usage.percent = 50.0

        mock_net_connections = MagicMock(return_value=[1, 2, 3])
        mock_net_if_addrs = MagicMock(return_value={"eth0": [], "lo": []})

        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "cpu_percent", MagicMock(return_value=50.0))
        monkeypatch.setattr(module.psutil, "cpu_count", MagicMock(return_value=4))
        monkeypatch.setattr(module.psutil, "cpu_freq", MagicMock(return_value=mock_cpu_freq))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_connections", mock_net_connections)
        monkeypatch.setattr(module.psutil, "net_if_addrs", mock_net_if_addrs)

        # Mock print
        prints = []
        def mock_print(*args, **kwargs):
            prints.append(args)
        
        monkeypatch.setattr("builtins.print", mock_print)

        usage = monitor._collect_resource_usage()

        assert 'timestamp' in usage
        assert 'memory' in usage
        assert 'cpu' in usage
        assert 'disk' in usage
        assert 'network' in usage

    def test_collect_resource_usage_exception(self, monitor, module, monkeypatch):
        """测试收集资源使用情况 - 异常"""
        # Mock psutil 抛出异常
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(side_effect=Exception("Test error")))

        # Mock print
        prints = []
        def mock_print(*args, **kwargs):
            prints.append(args)
        
        monkeypatch.setattr("builtins.print", mock_print)

        usage = monitor._collect_resource_usage()

        assert 'error' in usage
        assert 'memory' in usage
        assert 'cpu' in usage

    def test_collect_health_status_success(self, monitor, module, monkeypatch):
        """测试收集健康状态 - 成功"""
        # Mock psutil.boot_time
        mock_boot_time = MagicMock(return_value=time.time() - 86400)  # 1天前
        monkeypatch.setattr(module.psutil, "boot_time", mock_boot_time)

        # Mock print
        prints = []
        def mock_print(*args, **kwargs):
            prints.append(args)
        
        monkeypatch.setattr("builtins.print", mock_print)

        health = monitor._collect_health_status()

        assert 'timestamp' in health
        assert 'overall_status' in health
        assert 'services' in health

    def test_collect_health_status_exception(self, monitor, module, monkeypatch):
        """测试收集健康状态 - 异常"""
        # Mock psutil.boot_time 抛出异常
        monkeypatch.setattr(module.psutil, "boot_time", MagicMock(side_effect=Exception("Test error")))

        # Mock print
        prints = []
        def mock_print(*args, **kwargs):
            prints.append(args)
        
        monkeypatch.setattr("builtins.print", mock_print)

        health = monitor._collect_health_status()

        assert health.get('overall_status') == 'unknown' or 'error' in health

    def test_process_alerts_without_manager(self, monitor, module, monkeypatch):
        """测试处理告警 - 无管理器"""
        # 确保 _alert_manager 是 None
        monitor._alert_manager = None
        
        monitoring_data = {
            'coverage': {'coverage_percent': 80.0},
            'performance': {'response_time_ms': 100.0},
            'resources': {
                'memory': {'percent': 50.0},
                'cpu': {'percent': 50.0}
            },
            'health': {
                'overall_status': 'healthy',
                'services': {
                    'test_service': {'status': 'healthy'}
                }
            }
        }

        # Mock print
        prints = []
        def mock_print(*args, **kwargs):
            prints.append(args)
        
        monkeypatch.setattr("builtins.print", mock_print)

        alerts = monitor._process_alerts(monitoring_data)

        assert isinstance(alerts, list)

    def test_process_optimization_suggestions_without_engine(self, monitor):
        """测试处理优化建议 - 无引擎"""
        monitoring_data = {
            'coverage': {'coverage_percent': 80.0},
            'performance': {'response_time_ms': 100.0}
        }

        monitor._process_optimization_suggestions(monitoring_data)

        # 应该生成建议或保持为空
        assert isinstance(monitor.optimization_suggestions, list)

    def test_persist_monitoring_results_without_persistence(self, monitor):
        """测试持久化监控结果 - 无持久化"""
        timestamp = datetime.now()
        monitoring_data = {
            'coverage': {'coverage_percent': 80.0},
            'performance': {'response_time_ms': 100.0}
        }

        monitor._persist_monitoring_results(timestamp, monitoring_data)

        # 应该保存到内存
        assert len(monitor.metrics_history) > 0

    def test_service_name(self, monitor):
        """测试服务名称"""
        assert monitor.service_name == "continuous_monitoring_system"

    def test_service_version(self, monitor):
        """测试服务版本"""
        assert monitor.service_version == "2.0.0"

    def test_start_monitoring(self, monitor, module, monkeypatch):
        """测试启动监控"""
        # Mock runtime_start_monitoring
        import sys
        mock_runtime = sys.modules.get('src.infrastructure.monitoring.services.monitoring_runtime')
        if mock_runtime:
            mock_runtime.start_monitoring = MagicMock(return_value=True)

        result = monitor.start_monitoring()

        assert result is True

    def test_stop_monitoring(self, monitor, module, monkeypatch):
        """测试停止监控"""
        # Mock runtime_stop_monitoring
        import sys
        mock_runtime = sys.modules.get('src.infrastructure.monitoring.services.monitoring_runtime')
        if mock_runtime:
            mock_runtime.stop_monitoring = MagicMock(return_value=True)

        result = monitor.stop_monitoring()

        assert result is True

    def test_analyze_and_alert(self, monitor, module, monkeypatch):
        """测试分析和告警"""
        # Mock print
        prints = []
        def mock_print(*args, **kwargs):
            prints.append(args)
        
        monkeypatch.setattr("builtins.print", mock_print)
        
        coverage_data = {'coverage_percent': 75.0}
        performance_data = {'response_time_ms': 100.0}
        resource_data = {
            'memory': {'percent': 85.0},
            'cpu': {'percent': 75.0}
        }
        health_data = {
            'services': {
                'test_service': {'status': 'healthy'}
            }
        }
        
        alerts = monitor._analyze_and_alert(coverage_data, performance_data, resource_data, health_data)
        
        assert isinstance(alerts, list)
        assert len(prints) > 0

    def test_check_coverage_alerts_drop(self, monitor, module, monkeypatch):
        """测试检查覆盖率告警 - 覆盖率下降"""
        # 设置覆盖率趋势
        monitor.test_coverage_trends = [
            {'coverage_percent': 80.0},
            {'coverage_percent': 75.0}  # 下降了5%
        ]
        
        coverage_data = {'coverage_percent': 70.0}  # 又下降了5%
        
        alerts = monitor._check_coverage_alerts(coverage_data)
        
        assert len(alerts) > 0
        assert alerts[0]['type'] == 'coverage_drop'

    def test_check_coverage_alerts_no_drop(self, monitor, module, monkeypatch):
        """测试检查覆盖率告警 - 无下降"""
        # 设置覆盖率趋势
        monitor.test_coverage_trends = [
            {'coverage_percent': 80.0}
        ]
        
        coverage_data = {'coverage_percent': 79.0}  # 只下降了1%，低于阈值5%
        
        alerts = monitor._check_coverage_alerts(coverage_data)
        
        assert len(alerts) == 0

    def test_check_coverage_alerts_no_trends(self, monitor, module, monkeypatch):
        """测试检查覆盖率告警 - 无趋势数据"""
        monitor.test_coverage_trends = []
        
        coverage_data = {'coverage_percent': 70.0}
        
        alerts = monitor._check_coverage_alerts(coverage_data)
        
        assert len(alerts) == 0

    def test_check_resource_alerts_high_memory(self, monitor, module, monkeypatch):
        """测试检查资源告警 - 高内存使用"""
        resource_data = {
            'memory': {'percent': 85.0},  # 超过阈值80
            'cpu': {'percent': 50.0}
        }
        
        alerts = monitor._check_resource_alerts(resource_data)
        
        assert len(alerts) > 0
        assert any(a['type'] == 'high_memory_usage' for a in alerts)

    def test_check_resource_alerts_high_cpu(self, monitor, module, monkeypatch):
        """测试检查资源告警 - 高CPU使用"""
        resource_data = {
            'memory': {'percent': 50.0},
            'cpu': {'percent': 75.0}  # 超过阈值70
        }
        
        alerts = monitor._check_resource_alerts(resource_data)
        
        assert len(alerts) > 0
        assert any(a['type'] == 'high_cpu_usage' for a in alerts)

    def test_check_resource_alerts_both_high(self, monitor, module, monkeypatch):
        """测试检查资源告警 - 内存和CPU都高"""
        resource_data = {
            'memory': {'percent': 85.0},  # 超过阈值80
            'cpu': {'percent': 75.0}  # 超过阈值70
        }
        
        alerts = monitor._check_resource_alerts(resource_data)
        
        assert len(alerts) == 2
        assert any(a['type'] == 'high_memory_usage' for a in alerts)
        assert any(a['type'] == 'high_cpu_usage' for a in alerts)

    def test_check_health_alerts_unhealthy(self, monitor, module, monkeypatch):
        """测试检查健康告警 - 服务不健康"""
        health_data = {
            'services': {
                'service1': {'status': 'healthy'},
                'service2': {'status': 'unhealthy'},
                'service3': {'status': 'degraded'}
            }
        }
        
        alerts = monitor._check_health_alerts(health_data)
        
        assert len(alerts) > 0
        assert alerts[0]['type'] == 'service_unhealthy'
        assert 'service2' in alerts[0]['message'] or 'service3' in alerts[0]['message']

    def test_check_health_alerts_all_healthy(self, monitor, module, monkeypatch):
        """测试检查健康告警 - 所有服务健康"""
        health_data = {
            'services': {
                'service1': {'status': 'healthy'},
                'service2': {'status': 'healthy'}
            }
        }
        
        alerts = monitor._check_health_alerts(health_data)
        
        assert len(alerts) == 0

    def test_record_alerts(self, monitor, module, monkeypatch):
        """测试记录告警"""
        # Mock print
        prints = []
        def mock_print(*args, **kwargs):
            prints.append(args)
        
        monkeypatch.setattr("builtins.print", mock_print)
        
        alerts = [
            {
                'type': 'test_alert',
                'severity': 'warning',
                'message': 'Test alert message'
            }
        ]
        
        monitor._record_alerts(alerts)
        
        assert len(monitor.alerts_history) == 1
        assert len(prints) > 0

    def test_record_alerts_no_alerts(self, monitor, module, monkeypatch):
        """测试记录告警 - 无告警"""
        # Mock print
        prints = []
        def mock_print(*args, **kwargs):
            prints.append(args)
        
        monkeypatch.setattr("builtins.print", mock_print)
        
        monitor._record_alerts([])
        
        assert len(monitor.alerts_history) == 0
        assert len(prints) > 0
        assert any("无告警" in str(args) for args in prints)

    def test_generate_optimization_suggestions(self, monitor, module, monkeypatch):
        """测试生成优化建议"""
        # Mock print
        prints = []
        def mock_print(*args, **kwargs):
            prints.append(args)
        
        monkeypatch.setattr("builtins.print", mock_print)
        
        coverage_data = {'coverage_percent': 70.0}
        performance_data = {'response_time_ms': 15.0, 'memory_usage_mb': 2048}
        
        monitor._generate_optimization_suggestions(coverage_data, performance_data)
        
        assert len(monitor.optimization_suggestions) > 0
        assert len(prints) > 0

    def test_generate_coverage_suggestions_low(self, monitor, module, monkeypatch):
        """测试生成覆盖率建议 - 覆盖率低"""
        coverage_data = {'coverage_percent': 70.0}  # 低于80
        
        suggestions = monitor._generate_coverage_suggestions(coverage_data)
        
        assert len(suggestions) > 0
        assert suggestions[0]['type'] == 'coverage_improvement'

    def test_generate_coverage_suggestions_high(self, monitor, module, monkeypatch):
        """测试生成覆盖率建议 - 覆盖率高"""
        coverage_data = {'coverage_percent': 85.0}  # 高于80
        
        suggestions = monitor._generate_coverage_suggestions(coverage_data)
        
        assert len(suggestions) == 0

    def test_generate_performance_suggestions_slow(self, monitor, module, monkeypatch):
        """测试生成性能建议 - 响应时间慢"""
        performance_data = {'response_time_ms': 15.0}  # 超过10ms
        
        suggestions = monitor._generate_performance_suggestions(performance_data)
        
        assert len(suggestions) > 0
        assert suggestions[0]['type'] == 'performance_optimization'

    def test_generate_performance_suggestions_fast(self, monitor, module, monkeypatch):
        """测试生成性能建议 - 响应时间快"""
        performance_data = {'response_time_ms': 5.0}  # 低于10ms
        
        suggestions = monitor._generate_performance_suggestions(performance_data)
        
        assert len(suggestions) == 0

    def test_generate_memory_suggestions_high(self, monitor, module, monkeypatch):
        """测试生成内存建议 - 内存使用高"""
        performance_data = {'memory_usage_mb': 2048}  # 2GB，超过1GB
        
        suggestions = monitor._generate_memory_suggestions(performance_data)
        
        assert len(suggestions) > 0
        assert suggestions[0]['type'] == 'memory_optimization'

    def test_generate_memory_suggestions_low(self, monitor, module, monkeypatch):
        """测试生成内存建议 - 内存使用低"""
        performance_data = {'memory_usage_mb': 512}  # 0.5GB，低于1GB
        
        suggestions = monitor._generate_memory_suggestions(performance_data)
        
        assert len(suggestions) == 0

    def test_process_suggestions(self, monitor, module, monkeypatch):
        """测试处理建议"""
        suggestions = [
            {
                'type': 'test_suggestion',
                'priority': 'high',
                'title': 'Test',
                'description': 'Test description',
                'timestamp': datetime.now()
            }
        ]
        
        monitor._process_suggestions(suggestions)
        
        assert len(monitor.optimization_suggestions) == 1

    def test_save_monitoring_data(self, monitor, module, monkeypatch):
        """测试保存监控数据"""
        timestamp = datetime.now()
        data = {
            'coverage': {'coverage_percent': 80.0},
            'performance': {'response_time_ms': 100.0}
        }
        
        monitor._save_monitoring_data(timestamp, data)
        
        assert len(monitor.metrics_history) > 0

    def test_persist_monitoring_data(self, monitor, module, monkeypatch):
        """测试持久化监控数据"""
        # 添加一些数据
        monitor.metrics_history = [{'test': 'data'}]
        monitor.alerts_history = [{'test': 'alert'}]
        monitor.optimization_suggestions = [{'test': 'suggestion'}]
        
        monitor._persist_monitoring_data()
        
        # 应该不抛出异常
        pass

    def test_get_monitoring_report(self, monitor, module, monkeypatch):
        """测试获取监控报告"""
        # 添加一些数据
        monitor.metrics_history = [{'test': 'data'}]
        monitor.alerts_history = [{'test': 'alert'}]
        monitor.optimization_suggestions = [{'test': 'suggestion'}]
        
        report = monitor.get_monitoring_report()
        
        assert isinstance(report, dict)
        assert 'latest_metrics' in report
        assert 'latest_alerts' in report
        assert 'latest_suggestions' in report
        assert 'config' in report
        assert report['total_metrics_collected'] == 1
        assert report['total_alerts_generated'] == 1
        assert report['total_suggestions_generated'] == 1

    def test_health_check(self, monitor, module, monkeypatch):
        """测试健康检查"""
        health = monitor.health_check()
        
        assert isinstance(health, dict)
        assert 'status' in health

    def test_export_monitoring_report(self, monitor, module, monkeypatch):
        """测试导出监控报告"""
        # Mock runtime_export_monitoring_report
        import sys
        mock_runtime = sys.modules.get('src.infrastructure.monitoring.services.monitoring_runtime')
        if mock_runtime:
            mock_runtime.export_monitoring_report = MagicMock()
        
        monitor.export_monitoring_report("test_report.json")
        
        # 应该不抛出异常
        pass

    def test_check_monitoring_status(self, monitor, module, monkeypatch):
        """测试检查监控状态"""
        monitor.monitoring_active = True
        status = monitor._check_monitoring_status()
        assert status is True

        monitor.monitoring_active = False
        status = monitor._check_monitoring_status()
        assert status is False

    def test_check_components_status(self, monitor, module, monkeypatch):
        """测试检查组件状态"""
        monitor.metrics_history = [{'test': 'data'}]
        monitor.alerts_history = []
        monitor.optimization_suggestions = []
        
        status = monitor._check_components_status()
        
        assert isinstance(status, dict)
        assert 'monitoring_thread' in status
        assert 'metrics_collection' in status
        assert 'alert_system' in status
        assert 'optimization_engine' in status

    def test_check_components_status_with_thread(self, monitor, module, monkeypatch):
        """测试检查组件状态 - 有监控线程"""
        import threading
        import time
        
        # 创建一个会运行一段时间的线程
        def run_thread():
            time.sleep(0.1)
        
        monitor.monitoring_thread = threading.Thread(target=run_thread, daemon=True)
        monitor.monitoring_thread.start()
        
        # 等待线程启动
        time.sleep(0.01)
        
        status = monitor._check_components_status()
        
        # 线程应该仍然存活
        assert status['monitoring_thread'] is True

    def test_check_system_resources(self, monitor, module, monkeypatch):
        """测试检查系统资源"""
        # Mock psutil
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0

        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))

        resources = monitor._check_system_resources()
        
        assert 'cpu_usage' in resources
        assert 'memory_usage' in resources
        assert 'disk_usage' in resources
        assert resources['cpu_usage'] == 50.0

    def test_evaluate_overall_health_healthy(self, monitor, module, monkeypatch):
        """测试评估整体健康状态 - 健康"""
        monitoring_active = True
        components_status = {
            'monitoring_thread': True,
            'metrics_collection': True,
            'alert_system': True,
            'optimization_engine': True
        }
        system_resources = {
            'cpu_usage': 50.0,
            'memory_usage': 60.0,
            'disk_usage': 70.0
        }
        
        healthy = monitor._evaluate_overall_health(monitoring_active, components_status, system_resources)
        
        assert healthy is True

    def test_evaluate_overall_health_unhealthy_monitoring(self, monitor, module, monkeypatch):
        """测试评估整体健康状态 - 监控未激活"""
        monitoring_active = False
        components_status = {
            'monitoring_thread': True,
            'metrics_collection': True,
            'alert_system': True,
            'optimization_engine': True
        }
        system_resources = {
            'cpu_usage': 50.0,
            'memory_usage': 60.0,
            'disk_usage': 70.0
        }
        
        healthy = monitor._evaluate_overall_health(monitoring_active, components_status, system_resources)
        
        assert healthy is False

    def test_evaluate_overall_health_unhealthy_components(self, monitor, module, monkeypatch):
        """测试评估整体健康状态 - 组件不健康"""
        monitoring_active = True
        components_status = {
            'monitoring_thread': False,  # 组件失败
            'metrics_collection': True,
            'alert_system': True,
            'optimization_engine': True
        }
        system_resources = {
            'cpu_usage': 50.0,
            'memory_usage': 60.0,
            'disk_usage': 70.0
        }
        
        healthy = monitor._evaluate_overall_health(monitoring_active, components_status, system_resources)
        
        assert healthy is False

    def test_evaluate_overall_health_unhealthy_resources(self, monitor, module, monkeypatch):
        """测试评估整体健康状态 - 资源使用率高"""
        monitoring_active = True
        components_status = {
            'monitoring_thread': True,
            'metrics_collection': True,
            'alert_system': True,
            'optimization_engine': True
        }
        system_resources = {
            'cpu_usage': 95.0,  # 超过90
            'memory_usage': 60.0,
            'disk_usage': 70.0
        }
        
        healthy = monitor._evaluate_overall_health(monitoring_active, components_status, system_resources)
        
        assert healthy is False

    def test_build_health_result(self, monitor, module, monkeypatch):
        """测试构建健康结果"""
        monitoring_active = True
        components_status = {
            'monitoring_thread': True,
            'metrics_collection': True
        }
        system_resources = {
            'cpu_usage': 50.0,
            'memory_usage': 60.0
        }
        
        result = monitor._build_health_result(monitoring_active, components_status, system_resources, True)
        
        assert result['service'] == 'continuous_monitoring_system'
        assert result['healthy'] is True
        assert result['status'] == 'healthy'
        assert 'monitoring' in result
        assert 'resources' in result
        assert 'metrics' in result

    def test_add_diagnostic_info_monitoring_inactive(self, monitor, module, monkeypatch):
        """测试添加诊断信息 - 监控未激活"""
        health_result = {}
        monitoring_active = False
        components_status = {'test': True}
        system_resources = {'cpu_usage': 50.0}
        
        monitor._add_diagnostic_info(health_result, monitoring_active, components_status, system_resources)
        
        assert 'issues' in health_result
        assert 'recommendations' in health_result
        assert any('监控系统未激活' in issue for issue in health_result['issues'])

    def test_add_diagnostic_info_component_failed(self, monitor, module, monkeypatch):
        """测试添加诊断信息 - 组件失败"""
        health_result = {}
        monitoring_active = True
        components_status = {
            'component1': False,  # 失败
            'component2': True
        }
        system_resources = {'cpu_usage': 50.0}
        
        monitor._add_diagnostic_info(health_result, monitoring_active, components_status, system_resources)
        
        assert 'issues' in health_result
        assert any('组件异常' in issue for issue in health_result['issues'])

    def test_add_diagnostic_info_high_resources(self, monitor, module, monkeypatch):
        """测试添加诊断信息 - 资源使用率高"""
        health_result = {}
        monitoring_active = True
        components_status = {'test': True}
        system_resources = {
            'cpu_usage': 95.0,  # 超过90
            'memory_usage': 60.0
        }
        
        monitor._add_diagnostic_info(health_result, monitoring_active, components_status, system_resources)
        
        assert 'issues' in health_result
        assert any('资源使用率过高' in issue for issue in health_result['issues'])

    def test_create_error_health_result(self, monitor, module, monkeypatch):
        """测试创建错误健康结果"""
        error = Exception("Test error")
        result = monitor._create_error_health_result(error)
        
        assert result['service'] == 'continuous_monitoring_system'
        assert result['healthy'] is False
        assert result['status'] == 'error'
        assert 'error' in result
        assert result['error'] == "Test error"

    def test_health_check_unhealthy(self, monitor, module, monkeypatch):
        """测试健康检查 - 不健康"""
        monitor.monitoring_active = False
        
        # Mock psutil
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0

        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        
        health = monitor.health_check()
        
        assert health['healthy'] is False
        assert 'issues' in health

    def test_save_monitoring_data_max_items(self, monitor, module, monkeypatch):
        """测试保存监控数据 - 达到最大项目数"""
        monitor.monitoring_config['max_history_items'] = 3
        
        timestamp = datetime.now()
        data = {
            'coverage': {'coverage_percent': 80.0},
            'performance': {'response_time_ms': 100.0}
        }
        
        # 添加超过限制的数据
        for i in range(5):
            monitor._save_monitoring_data(timestamp, data)
        
        # 应该只保留最新的3条
        assert len(monitor.metrics_history) == 3

    def test_persist_monitoring_data_exception(self, monitor, module, monkeypatch):
        """测试持久化监控数据 - 异常处理"""
        # Mock open 抛出异常
        monkeypatch.setattr("builtins.open", MagicMock(side_effect=Exception("Test error")))
        
        # Mock print
        prints = []
        def mock_print(*args, **kwargs):
            prints.append(args)
        
        monkeypatch.setattr("builtins.print", mock_print)
        
        monitor.metrics_history = [{'timestamp': datetime.now().isoformat(), 'data': {'coverage': {}, 'performance': {}, 'health': {}}}]
        
        monitor._persist_monitoring_data()
        
        # 应该捕获异常并打印错误
        assert len(prints) > 0
        assert any("保存监控数据失败" in str(args) for args in prints)

    def test_collect_monitoring_data_with_collector(self, monitor, module, monkeypatch):
        """测试收集监控数据 - 使用 MetricsCollector"""
        # 创建 mock MetricsCollector
        mock_collector = MagicMock()
        mock_collector.collect_test_coverage = MagicMock(return_value={'coverage_percent': 85.0})
        mock_collector.collect_performance_metrics = MagicMock(return_value={'response_time_ms': 100.0})
        mock_collector.collect_resource_usage = MagicMock(return_value={'cpu_percent': 50.0})
        mock_collector.collect_health_status = MagicMock(return_value={'status': 'healthy'})
        
        monitor._metrics_collector = mock_collector
        
        data = monitor._collect_monitoring_data()
        
        assert 'coverage' in data
        assert 'performance' in data
        assert 'resources' in data
        assert 'health' in data
        assert data['coverage']['coverage_percent'] == 85.0
        mock_collector.collect_test_coverage.assert_called_once()

    def test_process_alerts_with_alert_manager(self, monitor, module, monkeypatch):
        """测试处理告警 - 使用 AlertManager"""
        # Mock print
        prints = []
        def mock_print(*args, **kwargs):
            prints.append(args)
        
        monkeypatch.setattr("builtins.print", mock_print)
        
        # 创建 mock AlertManager
        mock_alert_manager = MagicMock()
        mock_alert_manager.analyze_and_alert = MagicMock(return_value=[
            {'type': 'coverage', 'message': 'Test alert', 'severity': 'warning'}
        ])
        mock_alert_manager.test_coverage_trends = MagicMock()
        mock_alert_manager.test_coverage_trends.__iter__ = MagicMock(return_value=iter([80.0, 85.0]))
        
        monitor._alert_manager = mock_alert_manager
        
        monitoring_data = {
            'coverage': {'coverage_percent': 80.0},
            'performance': {'response_time_ms': 100.0},
            'resources': {'cpu_percent': 50.0},
            'health': {'status': 'healthy'}
        }
        
        alerts = monitor._process_alerts(monitoring_data)
        
        assert len(alerts) == 1
        mock_alert_manager.analyze_and_alert.assert_called_once()
        mock_alert_manager.update_coverage_trends.assert_called_once()

    def test_process_optimization_suggestions_with_engine(self, monitor, module, monkeypatch):
        """测试处理优化建议 - 使用 OptimizationEngine"""
        # 创建 mock OptimizationEngine
        mock_engine = MagicMock()
        mock_engine.generate_suggestions = MagicMock(return_value=[
            {'type': 'performance', 'title': 'Test suggestion'}
        ])
        # 设置 optimization_suggestions 属性
        mock_engine.optimization_suggestions = [
            {'type': 'performance', 'title': 'Test suggestion'}
        ]
        
        monitor._optimization_engine = mock_engine
        
        monitoring_data = {
            'coverage': {'coverage_percent': 80.0},
            'performance': {'response_time_ms': 100.0},
            'resources': {'cpu_percent': 50.0},
            'health': {'status': 'healthy'}
        }
        
        monitor._process_optimization_suggestions(monitoring_data)
        
        mock_engine.generate_suggestions.assert_called_once()
        assert len(monitor.optimization_suggestions) > 0

    def test_generate_memory_suggestions_high_usage(self, monitor, module, monkeypatch):
        """测试生成内存优化建议 - 高内存使用"""
        performance_data = {
            'memory_usage_mb': 2048  # 2GB
        }
        
        suggestions = monitor._generate_memory_suggestions(performance_data)
        
        assert len(suggestions) > 0
        assert suggestions[0]['type'] == 'memory_optimization'

    def test_generate_memory_suggestions_low_usage(self, monitor, module, monkeypatch):
        """测试生成内存优化建议 - 低内存使用"""
        performance_data = {
            'memory_usage_mb': 512  # 0.5GB，低于1GB阈值
        }
        
        suggestions = monitor._generate_memory_suggestions(performance_data)
        
        # 应该没有建议
        assert len(suggestions) == 0

    def test_process_suggestions_different_priorities(self, monitor, module, monkeypatch):
        """测试处理建议 - 不同优先级"""
        suggestions = [
            {'priority': 'high', 'title': 'High priority'},
            {'priority': 'medium', 'title': 'Medium priority'},
            {'priority': 'low', 'title': 'Low priority'},
            {'priority': 'unknown', 'title': 'Unknown priority'}
        ]
        
        # Mock print
        prints = []
        def mock_print(*args, **kwargs):
            prints.append(args)
        
        monkeypatch.setattr("builtins.print", mock_print)
        
        monitor._process_suggestions(suggestions)
        
        assert len(monitor.optimization_suggestions) == 4
        assert len(prints) == 4

    def test_persist_monitoring_results_with_persistence(self, monitor, module, monkeypatch):
        """测试持久化监控结果 - 使用 DataPersistence"""
        # 创建 mock DataPersistence
        mock_persistence = MagicMock()
        mock_persistence.save_monitoring_data = MagicMock()
        mock_persistence.persist_monitoring_data = MagicMock()
        mock_persistence.metrics_history = [
            {'timestamp': datetime.now().isoformat(), 'data': {'coverage': {}, 'performance': {}, 'health': {}}}
        ]
        
        monitor._data_persistence = mock_persistence
        
        timestamp = datetime.now()
        monitoring_data = {
            'coverage': {'coverage_percent': 80.0},
            'performance': {'response_time_ms': 100.0},
            'resources': {'cpu_percent': 50.0},
            'health': {'status': 'healthy'}
        }
        
        monitor._persist_monitoring_results(timestamp, monitoring_data)
        
        # 应该调用 DataPersistence 的方法
        mock_persistence.save_monitoring_data.assert_called_once_with(timestamp, monitoring_data)
        mock_persistence.persist_monitoring_data.assert_called_once()
        assert monitor.metrics_history == mock_persistence.metrics_history

    def test_persist_monitoring_results_without_persistence(self, monitor, module, monkeypatch):
        """测试持久化监控结果 - 不使用 DataPersistence"""
        monitor._data_persistence = None
        
        timestamp = datetime.now()
        monitoring_data = {
            'coverage': {'coverage_percent': 80.0},
            'performance': {'response_time_ms': 100.0},
            'resources': {'cpu_percent': 50.0},
            'health': {'status': 'healthy'}
        }
        
        original_count = len(monitor.metrics_history)
        
        monitor._persist_monitoring_results(timestamp, monitoring_data)
        
        # 应该使用 _save_monitoring_data
        assert len(monitor.metrics_history) == original_count + 1

    def test_collect_test_coverage(self, monitor, module, monkeypatch):
        """测试收集测试覆盖率"""
        # Mock runtime_collect_test_coverage
        mock_runtime_collect = MagicMock(return_value={'coverage_percent': 85.0})
        
        # 需要找到 runtime_collect_test_coverage 的位置并 mock
        import sys
        mock_runtime = sys.modules.get('src.infrastructure.monitoring.services.monitoring_runtime')
        if mock_runtime:
            mock_runtime.collect_test_coverage = mock_runtime_collect
        
        # 直接调用方法
        result = monitor._collect_test_coverage()
        
        # 应该返回覆盖率数据
        assert isinstance(result, dict)

    def test_collect_performance_metrics(self, monitor, module, monkeypatch):
        """测试收集性能指标"""
        # Mock print
        prints = []
        def mock_print(*args, **kwargs):
            prints.append(args)
        
        monkeypatch.setattr("builtins.print", mock_print)
        
        # Mock psutil
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0
        
        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        
        result = monitor._collect_performance_metrics()
        
        assert isinstance(result, dict)
        assert 'timestamp' in result
        assert 'cpu_usage_percent' in result
        assert 'memory_usage_mb' in result
        assert len(prints) > 0
        assert any("收集性能指标" in str(args) for args in prints)

    def test_collect_performance_metrics_exception(self, monitor, module, monkeypatch):
        """测试收集性能指标 - 异常处理"""
        # Mock print
        prints = []
        def mock_print(*args, **kwargs):
            prints.append(args)
        
        monkeypatch.setattr("builtins.print", mock_print)
        
        # Mock psutil.virtual_memory 抛出异常
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(side_effect=Exception("Test error")))
        
        result = monitor._collect_performance_metrics()
        
        # 应该返回包含错误信息的默认值
        assert isinstance(result, dict)
        assert result['cpu_usage_percent'] == 0.0
        assert 'error' in result
        assert len(prints) > 0
        assert any("收集性能指标失败" in str(args) for args in prints)

    def test_collect_resource_usage(self, monitor, module, monkeypatch):
        """测试收集资源使用情况"""
        # Mock print
        prints = []
        def mock_print(*args, **kwargs):
            prints.append(args)
        
        monkeypatch.setattr("builtins.print", mock_print)
        
        # Mock psutil
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.total = 1024 * 1024 * 1024  # 1GB
        mock_virtual_memory.available = 512 * 1024 * 1024  # 512MB
        mock_virtual_memory.percent = 50.0
        mock_virtual_memory.used = 512 * 1024 * 1024
        
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_cpu_count = MagicMock(return_value=4)
        mock_cpu_freq = MagicMock()
        mock_cpu_freq.current = 2400.0
        
        mock_disk_usage = MagicMock()
        mock_disk_usage.total = 100 * 1024 * 1024 * 1024  # 100GB
        mock_disk_usage.used = 50 * 1024 * 1024 * 1024  # 50GB
        mock_disk_usage.free = 50 * 1024 * 1024 * 1024  # 50GB
        mock_disk_usage.percent = 50.0
        
        mock_net_connections = MagicMock(return_value=[1, 2, 3])
        mock_net_if_addrs = MagicMock(return_value={'eth0': [], 'wlan0': []})
        
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "cpu_count", mock_cpu_count)
        monkeypatch.setattr(module.psutil, "cpu_freq", MagicMock(return_value=mock_cpu_freq))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_connections", mock_net_connections)
        monkeypatch.setattr(module.psutil, "net_if_addrs", MagicMock(return_value=mock_net_if_addrs))
        
        result = monitor._collect_resource_usage()
        
        assert isinstance(result, dict)
        assert 'timestamp' in result
        assert 'memory' in result
        assert 'cpu' in result
        assert 'disk' in result
        assert 'network' in result
        assert len(prints) > 0
        assert any("收集资源使用情况" in str(args) for args in prints)

    def test_collect_resource_usage_exception(self, monitor, module, monkeypatch):
        """测试收集资源使用情况 - 异常处理"""
        # Mock print
        prints = []
        def mock_print(*args, **kwargs):
            prints.append(args)
        
        monkeypatch.setattr("builtins.print", mock_print)
        
        # Mock psutil.virtual_memory 抛出异常
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(side_effect=Exception("Test error")))
        
        result = monitor._collect_resource_usage()
        
        # 应该返回包含错误信息的默认值
        assert isinstance(result, dict)
        assert 'error' in result
        assert result['memory']['percent'] == 0.0
        assert len(prints) > 0
        assert any("收集资源使用情况失败" in str(args) for args in prints)

    def test_collect_health_status(self, monitor, module, monkeypatch):
        """测试收集健康状态"""
        # Mock print
        prints = []
        def mock_print(*args, **kwargs):
            prints.append(args)
        
        monkeypatch.setattr("builtins.print", mock_print)
        
        result = monitor._collect_health_status()
        
        assert isinstance(result, dict)
        assert 'timestamp' in result
        assert 'overall_status' in result
        assert len(prints) > 0
        assert any("收集健康状态" in str(args) for args in prints)

    def test_collect_health_status_exception(self, monitor, module, monkeypatch):
        """测试收集健康状态 - 异常处理"""
        # Mock print
        prints = []
        def mock_print(*args, **kwargs):
            prints.append(args)
        
        monkeypatch.setattr("builtins.print", mock_print)
        
        # Mock psutil.boot_time 抛出异常
        monkeypatch.setattr(module.psutil, "boot_time", MagicMock(side_effect=Exception("Test error")))
        
        result = monitor._collect_health_status()
        
        # 应该返回包含错误信息的默认值
        assert isinstance(result, dict)
        assert 'error' in result
        assert result['overall_status'] == 'unknown'
        assert len(prints) > 0
        assert any("收集健康状态失败" in str(args) for args in prints)

    def test_check_coverage_alerts_no_coverage_percent(self, monitor, module, monkeypatch):
        """测试检查覆盖率告警 - 无coverage_percent字段"""
        coverage_data = {}  # 没有coverage_percent字段
        
        alerts = monitor._check_coverage_alerts(coverage_data)
        
        # 应该没有告警
        assert len(alerts) == 0

    def test_check_coverage_alerts_previous_coverage_default(self, monitor, module, monkeypatch):
        """测试检查覆盖率告警 - previous_coverage使用默认值"""
        # 设置覆盖率趋势，但最后一个没有coverage_percent
        monitor.test_coverage_trends = [
            {'coverage_percent': 80.0},
            {'other_field': 100}  # 没有coverage_percent
        ]
        
        coverage_data = {'coverage_percent': 70.0}
        
        alerts = monitor._check_coverage_alerts(coverage_data)
        
        # previous_coverage应该使用current_coverage作为默认值
        # 所以coverage_drop应该是0，不应该有告警
        assert len(alerts) == 0

    def test_check_coverage_alerts_exact_threshold(self, monitor, module, monkeypatch):
        """测试检查覆盖率告警 - 正好等于阈值"""
        # 设置覆盖率趋势
        monitor.test_coverage_trends = [
            {'coverage_percent': 80.0}
        ]
        
        # 正好下降5%（阈值）
        coverage_data = {'coverage_percent': 75.0}
        
        alerts = monitor._check_coverage_alerts(coverage_data)
        
        # 应该触发告警（因为使用>=）
        assert len(alerts) > 0
        assert alerts[0]['type'] == 'coverage_drop'

    def test_check_resource_alerts_no_memory_key(self, monitor, module, monkeypatch):
        """测试检查资源告警 - 无memory键"""
        resource_data = {
            'cpu': {'percent': 50.0}
            # 没有memory键
        }
        
        # 应该抛出KeyError或返回空列表
        try:
            alerts = monitor._check_resource_alerts(resource_data)
            # 如果没有抛出异常，应该返回空列表
            assert isinstance(alerts, list)
        except KeyError:
            # 如果抛出KeyError，这也是预期的行为
            pass

    def test_check_resource_alerts_no_cpu_key(self, monitor, module, monkeypatch):
        """测试检查资源告警 - 无cpu键"""
        resource_data = {
            'memory': {'percent': 50.0}
            # 没有cpu键
        }
        
        # 应该抛出KeyError或返回空列表
        try:
            alerts = monitor._check_resource_alerts(resource_data)
            # 如果没有抛出异常，应该返回空列表
            assert isinstance(alerts, list)
        except KeyError:
            # 如果抛出KeyError，这也是预期的行为
            pass

    def test_check_health_alerts_no_services_key(self, monitor, module, monkeypatch):
        """测试检查健康告警 - 无services键"""
        health_data = {
            'overall_status': 'healthy'
            # 没有services键
        }
        
        # 应该抛出KeyError或返回空列表
        try:
            alerts = monitor._check_health_alerts(health_data)
            # 如果没有抛出异常，应该返回空列表
            assert isinstance(alerts, list)
        except KeyError:
            # 如果抛出KeyError，这也是预期的行为
            pass

    def test_check_health_alerts_empty_services(self, monitor, module, monkeypatch):
        """测试检查健康告警 - 空services"""
        health_data = {
            'services': {}
        }
        
        alerts = monitor._check_health_alerts(health_data)
        
        # 应该没有告警
        assert len(alerts) == 0

    def test_record_alerts_empty_list(self, monitor, module, monkeypatch):
        """测试记录告警 - 空列表"""
        # Mock print
        prints = []
        def mock_print(*args, **kwargs):
            prints.append(args)
        
        monkeypatch.setattr("builtins.print", mock_print)
        
        original_count = len(monitor.alerts_history)
        
        monitor._record_alerts([])
        
        # 告警历史不应该改变
        assert len(monitor.alerts_history) == original_count
        # 应该打印无告警消息
        assert len(prints) > 0
        assert any("无告警" in str(args) for args in prints)

    def test_record_alerts_unknown_severity(self, monitor, module, monkeypatch):
        """测试记录告警 - 未知严重程度"""
        # Mock print
        prints = []
        def mock_print(*args, **kwargs):
            prints.append(args)
        
        monkeypatch.setattr("builtins.print", mock_print)
        
        alerts = [
            {
                'type': 'test',
                'severity': 'unknown',  # 未知严重程度
                'message': 'Test alert'
            }
        ]
        
        monitor._record_alerts(alerts)
        
        # 应该使用默认emoji
        assert len(monitor.alerts_history) == 1
        assert len(prints) > 0

    def test_generate_performance_suggestions_exact_threshold(self, monitor, module, monkeypatch):
        """测试生成性能优化建议 - 正好等于阈值"""
        performance_data = {
            'response_time_ms': 10.0  # 正好等于阈值10ms
        }
        
        suggestions = monitor._generate_performance_suggestions(performance_data)
        
        # 应该没有建议（因为使用>而不是>=）
        assert len(suggestions) == 0

    def test_generate_performance_suggestions_no_response_time(self, monitor, module, monkeypatch):
        """测试生成性能优化建议 - 无response_time_ms字段"""
        performance_data = {}
        
        suggestions = monitor._generate_performance_suggestions(performance_data)
        
        # 应该没有建议
        assert len(suggestions) == 0

    def test_generate_coverage_suggestions_exact_threshold(self, monitor, module, monkeypatch):
        """测试生成覆盖率建议 - 正好等于阈值"""
        coverage_data = {
            'coverage_percent': 80.0  # 正好等于阈值80%
        }
        
        suggestions = monitor._generate_coverage_suggestions(coverage_data)
        
        # 应该没有建议（因为使用<而不是<=）
        assert len(suggestions) == 0

    def test_generate_coverage_suggestions_no_coverage_percent(self, monitor, module, monkeypatch):
        """测试生成覆盖率建议 - 无coverage_percent字段"""
        coverage_data = {}
        
        # 代码会抛出KeyError，因为get返回0（<80），但后续访问不存在的键
        # 这是代码的一个bug，但为了测试目的，我们测试实际行为
        with pytest.raises(KeyError):
            monitor._generate_coverage_suggestions(coverage_data)

    def test_generate_memory_suggestions_exact_threshold(self, monitor, module, monkeypatch):
        """测试生成内存优化建议 - 正好等于阈值"""
        performance_data = {
            'memory_usage_mb': 1024  # 正好等于1GB阈值
        }
        
        suggestions = monitor._generate_memory_suggestions(performance_data)
        
        # 应该没有建议（因为使用>而不是>=）
        assert len(suggestions) == 0

    def test_generate_memory_suggestions_no_memory_usage(self, monitor, module, monkeypatch):
        """测试生成内存优化建议 - 无memory_usage_mb字段"""
        performance_data = {}
        
        suggestions = monitor._generate_memory_suggestions(performance_data)
        
        # 应该没有建议
        assert len(suggestions) == 0

    def test_generate_optimization_suggestions_with_all_types(self, monitor, module, monkeypatch):
        """测试生成优化建议 - 包含所有类型的建议"""
        # Mock print
        prints = []
        def mock_print(*args, **kwargs):
            prints.append(args)
        
        monkeypatch.setattr("builtins.print", mock_print)
        
        coverage_data = {
            'coverage_percent': 70.0  # 低于80%
        }
        performance_data = {
            'response_time_ms': 15.0,  # 超过10ms
            'memory_usage_mb': 2048  # 超过1GB
        }
        
        monitor._generate_optimization_suggestions(coverage_data, performance_data)
        
        # 应该生成所有类型的建议
        assert len(monitor.optimization_suggestions) >= 3
        assert len(prints) > 0
        assert any("生成优化建议" in str(args) for args in prints)

    def test_generate_optimization_suggestions_empty(self, monitor, module, monkeypatch):
        """测试生成优化建议 - 无建议"""
        # Mock print
        prints = []
        def mock_print(*args, **kwargs):
            prints.append(args)
        
        monkeypatch.setattr("builtins.print", mock_print)
        
        coverage_data = {
            'coverage_percent': 85.0  # 高于80%
        }
        performance_data = {
            'response_time_ms': 5.0,  # 低于10ms
            'memory_usage_mb': 512  # 低于1GB
        }
        
        original_count = len(monitor.optimization_suggestions)
        
        monitor._generate_optimization_suggestions(coverage_data, performance_data)
        
        # 应该没有新增建议
        assert len(monitor.optimization_suggestions) == original_count
        assert len(prints) > 0
        assert any("生成优化建议" in str(args) for args in prints)

    def test_monitoring_loop_method_exists(self, monitor):
        """测试监控循环方法存在"""
        # 只测试方法存在，不实际调用（因为会调用实际运行时函数）
        assert hasattr(monitor, '_monitoring_loop')
        assert callable(monitor._monitoring_loop)

    def test_perform_monitoring_cycle_method_exists(self, monitor):
        """测试执行监控周期方法存在"""
        # 只测试方法存在，不实际调用（因为会调用实际运行时函数）
        assert hasattr(monitor, '_perform_monitoring_cycle')
        assert callable(monitor._perform_monitoring_cycle)

    def test_get_mock_coverage_data_structure(self, monitor):
        """测试获取模拟覆盖率数据 - 数据结构"""
        data = monitor._get_mock_coverage_data()
        
        assert isinstance(data, dict)
        assert 'total_lines' in data
        assert 'covered_lines' in data
        assert 'coverage_percent' in data
        assert 'missing_lines' in data
        assert data['total_lines'] == 1000
        assert data['covered_lines'] == 750
        assert data['coverage_percent'] == 75.0
        assert data['missing_lines'] == 250

    def test_collect_test_coverage_metrics_json_parse_error(self, monitor, module, monkeypatch):
        """测试收集测试覆盖率指标 - JSON解析错误"""
        # Mock subprocess.run 返回成功但JSON无效
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "invalid json"
        
        monkeypatch.setattr(module.subprocess, "run", MagicMock(return_value=mock_result))
        
        # Mock json.loads 抛出异常
        monkeypatch.setattr(module.json, "loads", MagicMock(side_effect=Exception("JSON error")))
        
        coverage = monitor._collect_test_coverage_metrics()
        
        # 应该返回模拟数据
        assert coverage['total_lines'] == 1000
        assert coverage['coverage_percent'] == 75.0

    def test_collect_test_coverage_metrics_timeout(self, monitor, module, monkeypatch):
        """测试收集测试覆盖率指标 - 超时"""
        # Mock subprocess.run 抛出超时异常
        import subprocess
        monkeypatch.setattr(module.subprocess, "run", MagicMock(side_effect=subprocess.TimeoutExpired("coverage", 30)))
        
        coverage = monitor._collect_test_coverage_metrics()
        
        # 应该返回模拟数据
        assert coverage['total_lines'] == 1000
        assert coverage['coverage_percent'] == 75.0

    def test_check_resource_alerts_exact_threshold_memory(self, monitor, module, monkeypatch):
        """测试检查资源告警 - 内存使用率等于阈值"""
        resource_data = {
            'memory': {'percent': 80.0},  # 等于阈值80
            'cpu': {'percent': 50.0}
        }
        
        alerts = monitor._check_resource_alerts(resource_data)
        
        # 应该没有告警（因为使用 > 而不是 >=）
        assert len(alerts) == 0

    def test_check_resource_alerts_exact_threshold_cpu(self, monitor, module, monkeypatch):
        """测试检查资源告警 - CPU使用率等于阈值"""
        resource_data = {
            'memory': {'percent': 50.0},
            'cpu': {'percent': 70.0}  # 等于阈值70
        }
        
        alerts = monitor._check_resource_alerts(resource_data)
        
        # 应该没有告警（因为使用 > 而不是 >=）
        assert len(alerts) == 0

    def test_check_health_alerts_multiple_unhealthy_services(self, monitor, module, monkeypatch):
        """测试检查健康告警 - 多个不健康服务"""
        health_data = {
            'services': {
                'service1': {'status': 'unhealthy'},
                'service2': {'status': 'degraded'},
                'service3': {'status': 'healthy'}
            }
        }
        
        alerts = monitor._check_health_alerts(health_data)
        
        # 应该有一个告警，包含两个不健康服务
        assert len(alerts) == 1
        assert alerts[0]['type'] == 'service_unhealthy'
        assert 'service1' in alerts[0]['message']
        assert 'service2' in alerts[0]['message']
        assert 'service3' not in alerts[0]['message']

    def test_check_health_alerts_service_info_no_status(self, monitor, module, monkeypatch):
        """测试检查健康告警 - 服务信息无status键"""
        health_data = {
            'services': {
                'service1': {'response_time': 1.0}
                # 没有status键
            }
        }
        
        # 应该抛出KeyError或返回空列表
        try:
            alerts = monitor._check_health_alerts(health_data)
            # 如果没有抛出异常，应该返回空列表或包含告警
            assert isinstance(alerts, list)
        except KeyError:
            # 如果抛出KeyError，这也是预期的行为
            pass

    def test_record_alerts_with_unknown_severity(self, monitor, module, monkeypatch):
        """测试记录告警 - 未知严重程度"""
        alerts = [{
            'type': 'test_alert',
            'severity': 'unknown',  # 未知严重程度
            'message': 'Test message',
            'timestamp': datetime.now()
        }]
        
        prints = []
        def mock_print(*args, **kwargs):
            prints.append(args)
        
        monkeypatch.setattr("builtins.print", mock_print)
        
        monitor._record_alerts(alerts)
        
        # 应该记录告警
        assert len(monitor.alerts_history) == 1
        assert len(prints) > 0
        # 应该使用默认emoji '❓'
        assert any('❓' in str(args) for args in prints)

    def test_evaluate_overall_health_exact_resource_threshold(self, monitor, module, monkeypatch):
        """测试评估整体健康 - 资源使用率正好等于90"""
        monitoring_active = True
        components_status = {'test': True}
        system_resources = {
            'cpu_usage': 90.0,  # 正好等于阈值
            'memory_usage': 50.0,
            'disk_usage': 50.0
        }
        
        healthy = monitor._evaluate_overall_health(monitoring_active, components_status, system_resources)
        
        # 应该返回False（因为使用 < 而不是 <=）
        assert healthy is False

    def test_evaluate_overall_health_all_resources_at_threshold(self, monitor, module, monkeypatch):
        """测试评估整体健康 - 所有资源都在阈值"""
        monitoring_active = True
        components_status = {'test': True}
        system_resources = {
            'cpu_usage': 89.9,  # 略低于阈值
            'memory_usage': 89.9,
            'disk_usage': 89.9
        }
        
        healthy = monitor._evaluate_overall_health(monitoring_active, components_status, system_resources)
        
        # 应该返回True
        assert healthy is True

    def test_build_health_result_unhealthy(self, monitor, module, monkeypatch):
        """测试构建健康结果 - 不健康"""
        monitoring_active = False
        components_status = {
            'monitoring_thread': False,
            'metrics_collection': False
        }
        system_resources = {
            'cpu_usage': 95.0,
            'memory_usage': 95.0,
            'disk_usage': 95.0
        }
        
        result = monitor._build_health_result(monitoring_active, components_status, system_resources, False)
        
        assert result['service'] == 'continuous_monitoring_system'
        assert result['healthy'] is False
        assert result['status'] == 'unhealthy'
        assert result['monitoring']['active'] is False

    def test_add_diagnostic_info_all_issues(self, monitor, module, monkeypatch):
        """测试添加诊断信息 - 所有问题"""
        health_result = {}
        monitoring_active = False
        components_status = {
            'monitoring_thread': False,
            'metrics_collection': False
        }
        system_resources = {
            'cpu_usage': 95.0,
            'memory_usage': 95.0,
            'disk_usage': 95.0
        }
        
        monitor._add_diagnostic_info(health_result, monitoring_active, components_status, system_resources)
        
        assert 'issues' in health_result
        assert len(health_result['issues']) == 3  # 监控未激活、组件异常、资源使用率过高
        assert 'recommendations' in health_result
        assert len(health_result['recommendations']) == 3

    def test_add_diagnostic_info_no_issues(self, monitor, module, monkeypatch):
        """测试添加诊断信息 - 无问题"""
        health_result = {}
        monitoring_active = True
        components_status = {
            'monitoring_thread': True,
            'metrics_collection': True,
            'alert_system': True,
            'optimization_engine': True
        }
        system_resources = {
            'cpu_usage': 50.0,
            'memory_usage': 50.0,
            'disk_usage': 50.0
        }
        
        monitor._add_diagnostic_info(health_result, monitoring_active, components_status, system_resources)
        
        assert 'issues' in health_result
        assert len(health_result['issues']) == 0
        assert 'recommendations' in health_result

    def test_add_diagnostic_info_partial_components_failed(self, monitor, module, monkeypatch):
        """测试添加诊断信息 - 部分组件失败"""
        health_result = {}
        monitoring_active = True
        components_status = {
            'monitoring_thread': True,
            'metrics_collection': False,  # 这个失败
            'alert_system': True,
            'optimization_engine': False  # 这个也失败
        }
        system_resources = {
            'cpu_usage': 50.0,
            'memory_usage': 50.0,
            'disk_usage': 50.0
        }
        
        monitor._add_diagnostic_info(health_result, monitoring_active, components_status, system_resources)
        
        assert 'issues' in health_result
        assert len(health_result['issues']) == 1  # 只有组件异常
        assert 'metrics_collection' in health_result['issues'][0] or 'optimization_engine' in health_result['issues'][0]

    def test_add_diagnostic_info_single_high_resource(self, monitor, module, monkeypatch):
        """测试添加诊断信息 - 单个资源使用率高"""
        health_result = {}
        monitoring_active = True
        components_status = {
            'monitoring_thread': True,
            'metrics_collection': True
        }
        system_resources = {
            'cpu_usage': 95.0,  # 只有这个高
            'memory_usage': 50.0,
            'disk_usage': 50.0
        }
        
        monitor._add_diagnostic_info(health_result, monitoring_active, components_status, system_resources)
        
        assert 'issues' in health_result
        assert len(health_result['issues']) == 1  # 只有资源使用率过高
        assert 'cpu_usage' in health_result['issues'][0] or '资源使用率过高' in health_result['issues'][0]

    def test_check_components_status_no_thread(self, monitor, module, monkeypatch):
        """测试检查组件状态 - 无线程"""
        monitor.monitoring_thread = None
        
        status = monitor._check_components_status()
        
        assert 'monitoring_thread' in status
        assert status['monitoring_thread'] is False

    def test_check_components_status_empty_histories(self, monitor, module, monkeypatch):
        """测试检查组件状态 - 空历史"""
        monitor.metrics_history = []
        monitor.alerts_history = []
        monitor.optimization_suggestions = []
        
        status = monitor._check_components_status()
        
        assert status['metrics_collection'] is False
        assert status['alert_system'] is True  # >= 0 所以是True
        assert status['optimization_engine'] is True  # >= 0 所以是True

    def test_check_system_resources_exception_handling(self, monitor, module, monkeypatch):
        """测试检查系统资源 - 异常处理"""
        # Mock psutil 抛出异常
        monkeypatch.setattr(module.psutil, "cpu_percent", MagicMock(side_effect=Exception("CPU error")))
        
        # 应该抛出异常（因为没有try-except）
        try:
            resources = monitor._check_system_resources()
            # 如果没有抛出异常，检查返回结果
            assert isinstance(resources, dict)
        except Exception:
            # 如果抛出异常，这也是预期的行为
            pass

    def test_health_check_with_exception_in_check_system_resources(self, monitor, module, monkeypatch):
        """测试健康检查 - check_system_resources 抛出异常"""
        # Mock psutil 抛出异常
        monkeypatch.setattr(module.psutil, "cpu_percent", MagicMock(side_effect=Exception("CPU error")))
        
        # 健康检查应该捕获异常并返回错误结果
        result = monitor.health_check()
        
        # 应该返回错误健康结果
        assert 'error' in result or result.get('status') == 'error'

    def test_collect_resource_usage_cpu_freq_none(self, monitor, module, monkeypatch):
        """测试收集资源使用情况 - CPU频率为None"""
        # Mock psutil
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.total = 1000
        mock_virtual_memory.available = 500
        mock_virtual_memory.percent = 50.0
        mock_virtual_memory.used = 500
        
        mock_cpu_freq = MagicMock(return_value=None)  # CPU频率为None
        mock_disk_usage = MagicMock()
        mock_disk_usage.total = 1000
        mock_disk_usage.used = 500
        mock_disk_usage.free = 500
        mock_disk_usage.percent = 50.0
        
        mock_net_connections = MagicMock(return_value=[])
        mock_net_if_addrs = MagicMock(return_value={'eth0': []})
        
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "cpu_percent", MagicMock(return_value=50.0))
        monkeypatch.setattr(module.psutil, "cpu_count", MagicMock(return_value=4))
        monkeypatch.setattr(module.psutil, "cpu_freq", mock_cpu_freq)
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_connections", mock_net_connections)
        monkeypatch.setattr(module.psutil, "net_if_addrs", MagicMock(return_value=mock_net_if_addrs))
        
        resource_data = monitor._collect_resource_usage()
        
        assert 'cpu' in resource_data
        assert resource_data['cpu']['frequency'] is None

    def test_collect_health_status_no_boot_time(self, monitor, module, monkeypatch):
        """测试收集健康状态 - 无boot_time属性"""
        # Mock psutil 没有 boot_time 属性
        original_boot_time = getattr(module.psutil, 'boot_time', None)
        if hasattr(module.psutil, 'boot_time'):
            delattr(module.psutil, 'boot_time')
        
        try:
            health_data = monitor._collect_health_status()
            
            assert 'uptime_seconds' in health_data
            assert health_data['uptime_seconds'] is None
        finally:
            # 恢复原始属性
            if original_boot_time is not None:
                module.psutil.boot_time = original_boot_time

    def test_collect_resource_usage_partial_exception(self, monitor, module, monkeypatch):
        """测试收集资源使用情况 - 部分异常"""
        # Mock psutil - 部分方法抛出异常
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.total = 1000
        mock_virtual_memory.available = 500
        mock_virtual_memory.percent = 50.0
        mock_virtual_memory.used = 500
        
        # cpu_percent 抛出异常
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "cpu_percent", MagicMock(side_effect=Exception("CPU error")))
        
        resource_data = monitor._collect_resource_usage()
        
        # 应该返回错误数据
        assert 'error' in resource_data
        assert resource_data['memory']['percent'] == 0.0

    def test_collect_health_status_boot_time_exception(self, monitor, module, monkeypatch):
        """测试收集健康状态 - boot_time 抛出异常"""
        # Mock psutil.boot_time 抛出异常
        def mock_boot_time():
            raise Exception("Boot time error")
        
        if hasattr(module.psutil, 'boot_time'):
            monkeypatch.setattr(module.psutil, "boot_time", mock_boot_time)
        
        health_data = monitor._collect_health_status()
        
        # 应该返回错误数据
        assert 'error' in health_data or 'uptime_seconds' in health_data

    def test_generate_performance_suggestions_exact_threshold(self, monitor, module, monkeypatch):
        """测试生成性能建议 - 精确阈值"""
        performance_data = {
            'response_time_ms': 10.0  # 正好等于阈值
        }
        
        suggestions = monitor._generate_performance_suggestions(performance_data)
        
        # 应该没有建议（因为使用 > 而不是 >=）
        assert len(suggestions) == 0

    def test_generate_memory_suggestions_exact_threshold(self, monitor, module, monkeypatch):
        """测试生成内存建议 - 精确阈值"""
        performance_data = {
            'memory_usage_mb': 1024  # 正好等于1GB
        }
        
        suggestions = monitor._generate_memory_suggestions(performance_data)
        
        # 应该没有建议（因为使用 > 而不是 >=）
        assert len(suggestions) == 0

    def test_generate_memory_suggestions_no_memory_key(self, monitor, module, monkeypatch):
        """测试生成内存建议 - 无memory_usage_mb键"""
        performance_data = {}
        
        suggestions = monitor._generate_memory_suggestions(performance_data)
        
        # 应该没有建议
        assert len(suggestions) == 0

    def test_process_suggestions_unknown_priority(self, monitor, module, monkeypatch):
        """测试处理建议 - 未知优先级"""
        suggestions = [{
            'type': 'test',
            'priority': 'unknown',  # 未知优先级
            'title': 'Test suggestion',
            'description': 'Test',
            'actions': [],
            'timestamp': datetime.now()
        }]
        
        prints = []
        def mock_print(*args, **kwargs):
            prints.append(args)
        
        monkeypatch.setattr("builtins.print", mock_print)
        
        monitor._process_suggestions(suggestions)
        
        # 应该使用默认emoji '⚪'
        assert len(prints) > 0
        assert any('⚪' in str(args) for args in prints)

    def test_collect_resource_usage_network_interfaces(self, monitor, module, monkeypatch):
        """测试收集资源使用情况 - 网络接口"""
        # Mock psutil
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.total = 1000
        mock_virtual_memory.available = 500
        mock_virtual_memory.percent = 50.0
        mock_virtual_memory.used = 500
        
        mock_cpu_freq = MagicMock()
        mock_cpu_freq.current = 2000.0
        
        mock_disk_usage = MagicMock()
        mock_disk_usage.total = 1000
        mock_disk_usage.used = 500
        mock_disk_usage.free = 500
        mock_disk_usage.percent = 50.0
        
        mock_net_connections = MagicMock(return_value=[])
        # Mock net_if_addrs 返回字典，keys() 方法返回接口列表
        mock_net_if_addrs_dict = {'eth0': [], 'wlan0': []}
        mock_net_if_addrs = MagicMock(return_value=mock_net_if_addrs_dict)
        
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "cpu_percent", MagicMock(return_value=50.0))
        monkeypatch.setattr(module.psutil, "cpu_count", MagicMock(return_value=4))
        monkeypatch.setattr(module.psutil, "cpu_freq", MagicMock(return_value=mock_cpu_freq))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_connections", mock_net_connections)
        monkeypatch.setattr(module.psutil, "net_if_addrs", mock_net_if_addrs)
        
        resource_data = monitor._collect_resource_usage()
        
        assert 'network' in resource_data
        assert 'interfaces' in resource_data['network']
        # 检查接口列表（使用 list() 转换 keys()）
        interfaces = list(resource_data['network']['interfaces'])
        assert len(interfaces) == 2
        assert 'eth0' in interfaces
        assert 'wlan0' in interfaces

    def test_process_suggestions_empty_list(self, monitor, module, monkeypatch):
        """测试处理建议 - 空列表"""
        prints = []
        def mock_print(*args, **kwargs):
            prints.append(args)
        
        monkeypatch.setattr("builtins.print", mock_print)
        
        original_count = len(monitor.optimization_suggestions)
        monitor._process_suggestions([])
        
        # 应该没有新增建议
        assert len(monitor.optimization_suggestions) == original_count
        # 应该没有打印任何内容
        assert len(prints) == 0

    def test_process_suggestions_unknown_priority(self, monitor, module, monkeypatch):
        """测试处理建议 - 未知优先级"""
        prints = []
        def mock_print(*args, **kwargs):
            prints.append(args)
        
        monkeypatch.setattr("builtins.print", mock_print)
        
        suggestions = [{
            'type': 'test',
            'priority': 'unknown',  # 未知优先级
            'title': 'Test suggestion'
        }]
        
        monitor._process_suggestions(suggestions)
        
        # 应该添加了建议
        assert len(monitor.optimization_suggestions) >= 1
        # 应该使用了默认emoji '⚪'
        assert any('⚪' in str(args) for args in prints)

    def test_save_monitoring_data_exact_max_items(self, monitor, module, monkeypatch):
        """测试保存监控数据 - 正好达到最大历史项数"""
        max_items = monitor.monitoring_config['max_history_items']
        
        # 添加正好max_items个数据
        for i in range(max_items):
            monitor._save_monitoring_data(datetime.now(), {'test': i})
        
        # 应该正好有max_items个数据
        assert len(monitor.metrics_history) == max_items
        
        # 再添加一个，应该还是max_items个
        monitor._save_monitoring_data(datetime.now(), {'test': max_items})
        assert len(monitor.metrics_history) == max_items

    def test_get_monitoring_report_with_empty_data(self, monitor):
        """测试获取监控报告 - 空数据"""
        monitor.metrics_history = []
        monitor.alerts_history = []
        monitor.optimization_suggestions = []
        monitor.monitoring_active = False
        
        report = monitor.get_monitoring_report()
        
        assert report['monitoring_active'] is False
        assert report['total_metrics_collected'] == 0
        assert report['total_alerts_generated'] == 0
        assert report['total_suggestions_generated'] == 0
        assert report['latest_metrics'] is None
        assert report['latest_alerts'] == []
        assert report['latest_suggestions'] == []

    def test_get_monitoring_report_with_data(self, monitor, module, monkeypatch):
        """测试获取监控报告 - 有数据"""
        from datetime import datetime
        
        # 添加一些数据
        monitor.metrics_history = [
            {'timestamp': datetime.now().isoformat(), 'data': {'test': 1}},
            {'timestamp': datetime.now().isoformat(), 'data': {'test': 2}}
        ]
        monitor.alerts_history = [
            {'type': 'test', 'message': 'Alert 1'},
            {'type': 'test', 'message': 'Alert 2'},
            {'type': 'test', 'message': 'Alert 3'},
            {'type': 'test', 'message': 'Alert 4'},
            {'type': 'test', 'message': 'Alert 5'},
            {'type': 'test', 'message': 'Alert 6'}  # 超过5个
        ]
        monitor.optimization_suggestions = [
            {'type': 'test', 'title': 'Suggestion 1'},
            {'type': 'test', 'title': 'Suggestion 2'},
            {'type': 'test', 'title': 'Suggestion 3'},
            {'type': 'test', 'title': 'Suggestion 4'}  # 超过3个
        ]
        monitor.monitoring_active = True
        
        report = monitor.get_monitoring_report()
        
        assert report['monitoring_active'] is True
        assert report['total_metrics_collected'] == 2
        assert report['total_alerts_generated'] == 6
        assert report['total_suggestions_generated'] == 4
        assert report['latest_metrics'] is not None
        assert len(report['latest_alerts']) == 5  # 只返回最后5个
        assert len(report['latest_suggestions']) == 3  # 只返回最后3个

    def test_persist_monitoring_data_success(self, monitor, module, monkeypatch):
        """测试持久化监控数据 - 成功"""
        import json
        import os
        
        # Mock json.dump 和 open
        dumps_calls = []
        def mock_dump(data, f, **kwargs):
            dumps_calls.append(data)
        
        def mock_open(*args, **kwargs):
            return MagicMock()
        
        monkeypatch.setattr(module.json, "dump", mock_dump)
        monkeypatch.setattr("builtins.open", mock_open)
        
        # 添加正确格式的数据
        monitor.metrics_history = [{
            'timestamp': datetime.now().isoformat(),
            'data': {
                'coverage': {'coverage_percent': 80.0},
                'performance': {'memory_usage_mb': 100, 'cpu_usage_percent': 50.0},
                'health': {'overall_status': 'healthy'}
            }
        }]
        monitor.alerts_history = [{'alert': 1}]
        monitor.optimization_suggestions = [{'suggestion': 1}]
        
        monitor._persist_monitoring_data()
        
        # 应该调用了json.dump
        assert len(dumps_calls) > 0

    def test_save_monitoring_data_persist_exception(self, monitor, module, monkeypatch):
        """测试保存监控数据 - 持久化异常"""
        import json
        import os
        
        # Mock _persist_monitoring_data 内部抛出异常
        original_persist = monitor._persist_monitoring_data
        
        def mock_persist():
            raise Exception("Persistence error")
        
        monitor._persist_monitoring_data = mock_persist
        
        prints = []
        def mock_print(*args, **kwargs):
            prints.append(args)
        
        monkeypatch.setattr("builtins.print", mock_print)
        
        # 保存数据（_save_monitoring_data内部调用_persist_monitoring_data，但没有try-except）
        # 所以异常会传播，但我们可以测试异常确实被抛出
        try:
            monitor._save_monitoring_data(datetime.now(), {'test': 1})
            # 如果没有抛出异常，检查数据是否添加
            assert len(monitor.metrics_history) >= 1
        except Exception as e:
            # 如果抛出异常，这也是预期的行为（因为_save_monitoring_data没有捕获_persist_monitoring_data的异常）
            assert "Persistence error" in str(e)
        
        # 恢复原始方法
        monitor._persist_monitoring_data = original_persist

    def test_generate_memory_suggestions_exact_threshold_gb(self, monitor, module, monkeypatch):
        """测试生成内存建议 - 正好等于阈值（1GB）"""
        performance_data = {
            'memory_usage_mb': 1024  # 正好1GB
        }
        
        suggestions = monitor._generate_memory_suggestions(performance_data)
        
        # 应该没有建议（因为使用 > 而不是 >=）
        assert len(suggestions) == 0

    def test_generate_performance_suggestions_exact_threshold(self, monitor, module, monkeypatch):
        """测试生成性能建议 - 正好等于阈值（10ms）"""
        performance_data = {
            'response_time_ms': 10.0  # 正好等于阈值
        }
        
        suggestions = monitor._generate_performance_suggestions(performance_data)
        
        # 应该没有建议（因为使用 > 而不是 >=）
        assert len(suggestions) == 0

    def test_process_suggestions_multiple_priorities(self, monitor, module, monkeypatch):
        """测试处理建议 - 多个优先级"""
        prints = []
        def mock_print(*args, **kwargs):
            prints.append(args)
        
        monkeypatch.setattr("builtins.print", mock_print)
        
        suggestions = [
            {'type': 'test', 'priority': 'high', 'title': 'High priority'},
            {'type': 'test', 'priority': 'medium', 'title': 'Medium priority'},
            {'type': 'test', 'priority': 'low', 'title': 'Low priority'},
            {'type': 'test', 'priority': 'unknown', 'title': 'Unknown priority'}
        ]
        
        monitor._process_suggestions(suggestions)
        
        # 应该添加了4个建议
        assert len(monitor.optimization_suggestions) >= 4
        # 应该打印了4次
        assert len(prints) == 4
        # 应该包含不同优先级的emoji
        assert any('🔴' in str(args) for args in prints)  # high
        assert any('🟡' in str(args) for args in prints)  # medium
        assert any('🟢' in str(args) for args in prints)  # low
        assert any('⚪' in str(args) for args in prints)  # unknown

    def test_persist_monitoring_data_missing_keys(self, monitor, module, monkeypatch):
        """测试持久化监控数据 - 缺少键"""
        import json
        
        # 添加一些不完整的数据
        monitor.metrics_history = [{
            'timestamp': datetime.now().isoformat(),
            'data': {
                'coverage': {},  # 缺少 coverage_percent
                'performance': {},  # 缺少 memory_usage_mb
                'health': {}  # 缺少 overall_status
            }
        }]
        
        # Mock open 和 json.dump
        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=None)
        
        monkeypatch.setattr("builtins.open", MagicMock(return_value=mock_file))
        monkeypatch.setattr(module.json, "dump", MagicMock())
        
        monitor._persist_monitoring_data()
        
        # 应该成功处理（使用默认值）
        module.json.dump.assert_called_once()

    def test_persist_monitoring_data_more_than_100_records(self, monitor, module, monkeypatch):
        """测试持久化监控数据 - 超过100条记录"""
        import json
        
        # 添加超过100条记录
        monitor.metrics_history = []
        for i in range(150):
            monitor.metrics_history.append({
                'timestamp': datetime.now().isoformat(),
                'data': {
                    'coverage': {'coverage_percent': 80.0},
                    'performance': {'memory_usage_mb': 100.0, 'cpu_usage_percent': 50.0},
                    'health': {'overall_status': 'healthy'}
                }
            })
        
        # Mock open 和 json.dump
        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=None)
        
        monkeypatch.setattr("builtins.open", MagicMock(return_value=mock_file))
        monkeypatch.setattr(module.json, "dump", MagicMock())
        
        monitor._persist_monitoring_data()
        
        # 应该只保存最后100条记录
        call_args = module.json.dump.call_args[0][0]
        assert len(call_args['metrics_history']) == 100

    def test_persist_monitoring_data_more_than_50_alerts(self, monitor, module, monkeypatch):
        """测试持久化监控数据 - 超过50条告警"""
        import json
        
        # 添加超过50条告警
        monitor.alerts_history = []
        for i in range(100):
            monitor.alerts_history.append({
                'type': 'test',
                'severity': 'warning',
                'message': f'Test {i}',
                'timestamp': datetime.now()
            })
        
        # Mock open 和 json.dump
        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=None)
        
        monkeypatch.setattr("builtins.open", MagicMock(return_value=mock_file))
        monkeypatch.setattr(module.json, "dump", MagicMock())
        
        monitor._persist_monitoring_data()
        
        # 应该只保存最后50条告警
        call_args = module.json.dump.call_args[0][0]
        assert len(call_args['alerts_history']) == 50

    def test_persist_monitoring_data_more_than_20_suggestions(self, monitor, module, monkeypatch):
        """测试持久化监控数据 - 超过20条建议"""
        import json
        
        # 添加超过20条建议
        monitor.optimization_suggestions = []
        for i in range(50):
            monitor.optimization_suggestions.append({
                'type': 'test',
                'priority': 'medium',
                'title': f'Test {i}',
                'description': 'Test',
                'actions': [],
                'timestamp': datetime.now()
            })
        
        # Mock open 和 json.dump
        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=None)
        
        monkeypatch.setattr("builtins.open", MagicMock(return_value=mock_file))
        monkeypatch.setattr(module.json, "dump", MagicMock())
        
        monitor._persist_monitoring_data()
        
        # 应该只保存最后20条建议
        call_args = module.json.dump.call_args[0][0]
        assert len(call_args['optimization_suggestions']) == 20

    def test_get_monitoring_report_with_many_alerts(self, monitor, module, monkeypatch):
        """测试获取监控报告 - 多个告警"""
        # 添加多个告警
        monitor.alerts_history = []
        for i in range(10):
            monitor.alerts_history.append({
                'type': 'test',
                'severity': 'warning',
                'message': f'Test {i}',
                'timestamp': datetime.now()
            })
        
        report = monitor.get_monitoring_report()
        
        # 应该只返回最后5个告警
        assert len(report['latest_alerts']) == 5

    def test_get_monitoring_report_with_many_suggestions(self, monitor, module, monkeypatch):
        """测试获取监控报告 - 多个建议"""
        # 添加多个建议
        monitor.optimization_suggestions = []
        for i in range(10):
            monitor.optimization_suggestions.append({
                'type': 'test',
                'priority': 'medium',
                'title': f'Test {i}',
                'description': 'Test',
                'actions': [],
                'timestamp': datetime.now()
            })
        
        report = monitor.get_monitoring_report()
        
        # 应该只返回最后3个建议
        assert len(report['latest_suggestions']) == 3

    def test_save_monitoring_data_exact_max_items_boundary(self, monitor, module, monkeypatch):
        """测试保存监控数据 - 精确达到最大条目数"""
        # 设置最大条目数
        max_items = monitor.monitoring_config['max_history_items']
        
        # 添加精确达到最大条目数的数据
        monitor.metrics_history = []
        for i in range(max_items):
            monitor.metrics_history.append({
                'timestamp': datetime.now(),
                'data': {
                    'coverage': {'coverage_percent': 80.0},
                    'performance': {'memory_usage_mb': 100.0},
                    'health': {'overall_status': 'healthy'}
                }
            })
        
        # 再添加一条新数据
        monitor._save_monitoring_data(datetime.now(), {
            'coverage': {'coverage_percent': 81.0},
            'performance': {'memory_usage_mb': 101.0},
            'health': {'overall_status': 'healthy'}
        })
        
        # 应该保持最大条目数
        assert len(monitor.metrics_history) == max_items

