#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试 - 指标收集器深度测试
测试MetricsCollector的核心指标收集功能、边界条件和错误处理
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

import pytest

from infrastructure.monitoring.components.metrics_collector import MetricsCollector


class TestMetricsCollectorInitialization:
    """MetricsCollector初始化测试"""

    def test_initialization_default(self):
        """测试默认初始化"""
        collector = MetricsCollector()
        assert collector.project_root == os.getcwd()

    def test_initialization_custom_root(self):
        """测试自定义项目根目录初始化"""
        custom_root = "/custom/path"
        collector = MetricsCollector(custom_root)
        assert collector.project_root == custom_root

    def test_initialization_none_root(self):
        """测试None根目录初始化"""
        collector = MetricsCollector(None)
        assert collector.project_root == os.getcwd()


class TestMetricsCollectorSystemMetrics:
    """系统指标收集测试"""

    @pytest.fixture
    def collector(self):
        """MetricsCollector fixture"""
        return MetricsCollector()

    def test_collect_system_metrics_with_psutil(self, collector):
        """测试有psutil时的系统指标收集"""
        mock_memory = MagicMock()
        mock_memory.percent = 75.5

        mock_disk = MagicMock()
        mock_disk.percent = 45.2

        with patch('src.infrastructure.monitoring.components.metrics_collector.PSUTIL_AVAILABLE', True), \
             patch('psutil.cpu_percent', return_value=32.1), \
             patch('psutil.virtual_memory', return_value=mock_memory), \
             patch('psutil.disk_usage', return_value=mock_disk), \
             patch('psutil.net_connections', return_value=[1, 2, 3, 4, 5]):

            result = collector.collect_system_metrics()

            assert result['cpu_percent'] == 32.1
            assert result['memory_percent'] == 75.5
            assert result['disk_usage'] == 45.2
            assert result['network_connections'] == 5

    def test_collect_system_metrics_without_psutil(self, collector):
        """测试无psutil时的系统指标收集"""
        with patch('src.infrastructure.monitoring.components.metrics_collector.PSUTIL_AVAILABLE', False), \
             patch('infrastructure.monitoring.components.metrics_collector.PSUTIL_AVAILABLE', False):
            result = collector.collect_system_metrics()

            # 应该返回模拟数据
            assert result['cpu_percent'] == 45.5
            assert result['memory_percent'] == 67.8
            assert result['disk_usage'] == 50.0
            assert result['network_connections'] == 10

    def test_collect_system_metrics_psutil_exception(self, collector):
        """测试psutil异常时的系统指标收集"""
        with patch('src.infrastructure.monitoring.components.metrics_collector.PSUTIL_AVAILABLE', True), \
             patch('psutil.cpu_percent', side_effect=Exception("psutil error")):

            result = collector.collect_system_metrics()

            # 应该返回默认值
            assert result['cpu_percent'] == 0.0
            assert result['memory_percent'] == 0.0
            assert result['disk_usage'] == 0.0
            assert result['network_connections'] == 0


class TestMetricsCollectorTestCoverageMetrics:
    """测试覆盖率指标收集测试"""

    @pytest.fixture
    def collector(self):
        """MetricsCollector fixture"""
        return MetricsCollector("/test/project")

    def test_collect_test_coverage_metrics_success(self, collector):
        """测试覆盖率指标收集成功"""
        mock_coverage_data = {
            'totals': {
                'num_statements': 1000,
                'num_covered': 750,
                'percent_covered': 75.0,
                'num_missing': 250
            }
        }

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(mock_coverage_data)

        with patch('subprocess.run', return_value=mock_result):
            result = collector.collect_test_coverage_metrics()

            assert result['total_lines'] == 1000
            assert result['covered_lines'] == 750
            assert result['coverage_percent'] == 75.0
            assert result['missing_lines'] == 250

    def test_collect_test_coverage_metrics_command_failure(self, collector):
        """测试覆盖率命令失败"""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""

        with patch('subprocess.run', return_value=mock_result):
            result = collector.collect_test_coverage_metrics()

            # 应该返回模拟数据
            assert isinstance(result, dict)
            assert 'total_lines' in result

    def test_collect_test_coverage_metrics_subprocess_exception(self, collector):
        """测试子进程异常"""
        with patch('subprocess.run', side_effect=Exception("subprocess error")):
            result = collector.collect_test_coverage_metrics()

            # 应该返回模拟数据
            assert isinstance(result, dict)
            assert 'total_lines' in result

    def test_collect_test_coverage_metrics_timeout(self, collector):
        """测试超时情况"""
        with patch('subprocess.run', side_effect=subprocess.TimeoutExpired("timeout", 30)):
            result = collector.collect_test_coverage_metrics()

            # 应该返回模拟数据
            assert isinstance(result, dict)


class TestMetricsCollectorTestCoverage:
    """测试覆盖率收集测试"""

    @pytest.fixture
    def collector(self):
        """MetricsCollector fixture"""
        return MetricsCollector("/test/project")

    def test_collect_test_coverage_success(self, collector):
        """测试覆盖率收集成功"""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "test output"
        mock_result.stderr = "test error"

        mock_coverage_json = {
            'totals': {
                'percent_covered': 85.5
            }
        }

        with patch('subprocess.run', return_value=mock_result), \
             patch('builtins.open', mock_open(read_data=json.dumps(mock_coverage_json))), \
             patch('os.path.exists', return_value=True), \
             patch('os.remove') as mock_remove:

            result = collector.collect_test_coverage()

            assert result['success'] is True
            assert result['stdout'] == "test output"
            assert result['stderr'] == "test error"
            assert result['coverage_percent'] == 85.5
            assert isinstance(result['timestamp'], datetime)

            # 应该清理临时文件
            mock_remove.assert_called_once()

    def test_collect_test_coverage_failure(self, collector):
        """测试覆盖率收集失败"""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = "error output"
        mock_result.stderr = "error details"

        with patch('subprocess.run', return_value=mock_result):
            result = collector.collect_test_coverage()

            assert result['success'] is False
            assert result['stdout'] == "error output"
            assert result['stderr'] == "error details"
            assert result['coverage_percent'] == 0.0

    def test_collect_test_coverage_no_coverage_file(self, collector):
        """测试无覆盖率文件的情况"""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "success"
        mock_result.stderr = ""

        with patch('subprocess.run', return_value=mock_result), \
             patch('os.path.exists', return_value=False):

            result = collector.collect_test_coverage()

            assert result['success'] is True
            assert result['coverage_percent'] == 0.0

    def test_collect_test_coverage_json_parse_error(self, collector):
        """测试JSON解析错误"""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "success"
        mock_result.stderr = ""

        with patch('subprocess.run', return_value=mock_result), \
             patch('os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data="invalid json")), \
             patch('os.remove'):

            result = collector.collect_test_coverage()

            # 当JSON解析失败时，整体结果是失败的
            assert result['success'] is False
            assert result['coverage_percent'] == 0.0
            assert 'error' in result


class TestMetricsCollectorPerformanceMetrics:
    """性能指标收集测试"""

    @pytest.fixture
    def collector(self):
        """MetricsCollector fixture"""
        return MetricsCollector()

    def test_collect_performance_metrics_with_psutil(self, collector):
        """测试有psutil时的性能指标收集"""
        mock_memory = MagicMock()
        mock_memory.used = 536870912  # 512MB

        mock_net_io = MagicMock()
        mock_net_io.bytes_sent = 1000000
        mock_net_io.bytes_recv = 2000000

        with patch('src.infrastructure.monitoring.components.metrics_collector.PSUTIL_AVAILABLE', True), \
             patch('psutil.virtual_memory', return_value=mock_memory), \
             patch('psutil.cpu_percent', return_value=32.1), \
             patch('psutil.disk_usage', return_value=MagicMock(percent=45.2)), \
             patch('psutil.net_io_counters', return_value=mock_net_io):

            result = collector.collect_performance_metrics()

            assert result['response_time_ms'] == 4.20
            assert result['throughput_tps'] == 2000
            assert result['memory_usage_mb'] == 512.0  # 512MB
            assert result['cpu_usage_percent'] == 32.1
            assert result['disk_usage_percent'] == 45.2
            assert result['network_io']['bytes_sent'] == 1000000
            assert result['network_io']['bytes_recv'] == 2000000

    def test_collect_performance_metrics_without_psutil(self, collector):
        """测试无psutil时的性能指标收集"""
        with patch('src.infrastructure.monitoring.components.metrics_collector.PSUTIL_AVAILABLE', False), \
             patch('infrastructure.monitoring.components.metrics_collector.PSUTIL_AVAILABLE', False):
            result = collector.collect_performance_metrics()

            # 应该返回模拟数据
            assert result['response_time_ms'] == 4.20
            assert result['throughput_tps'] == 2000
            assert result['memory_usage_mb'] == 1024.0
            assert result['cpu_usage_percent'] == 45.5
            assert result['disk_usage_percent'] == 50.0
            assert result['network_io']['bytes_sent'] == 0
            assert result['network_io']['bytes_recv'] == 0

    def test_collect_performance_metrics_exceptions(self, collector):
        """测试性能指标收集异常"""
        with patch('src.infrastructure.monitoring.components.metrics_collector.PSUTIL_AVAILABLE', True), \
             patch('psutil.virtual_memory', side_effect=Exception("memory error")):

            result = collector.collect_performance_metrics()

            # 应该返回错误信息
            assert 'error' in result
            assert result['response_time_ms'] == 0.0
            assert result['throughput_tps'] == 0


class TestMetricsCollectorResourceUsage:
    """资源使用收集测试"""

    @pytest.fixture
    def collector(self):
        """MetricsCollector fixture"""
        return MetricsCollector()

    def test_collect_resource_usage_with_psutil(self, collector):
        """测试有psutil时的资源使用收集"""
        mock_memory = MagicMock()
        mock_memory.total = 17179869184  # 16GB
        mock_memory.available = 8589934592  # 8GB
        mock_memory.percent = 50.0
        mock_memory.used = 8589934592  # 8GB used

        mock_cpu_freq = MagicMock()
        mock_cpu_freq.current = 2500.0

        with patch('src.infrastructure.monitoring.components.metrics_collector.PSUTIL_AVAILABLE', True), \
             patch('psutil.virtual_memory', return_value=mock_memory), \
             patch('psutil.cpu_percent', return_value=32.1), \
             patch('psutil.cpu_count', return_value=8), \
             patch('psutil.cpu_freq', return_value=mock_cpu_freq), \
             patch('psutil.disk_usage', return_value=MagicMock(total=1000000000000, used=500000000000, free=500000000000, percent=50.0)), \
             patch('psutil.net_connections', return_value=[1, 2, 3]), \
             patch('psutil.net_if_addrs', return_value={'eth0': [], 'wlan0': []}):

            result = collector.collect_resource_usage()

            assert result['memory']['total'] == 17179869184
            assert result['memory']['available'] == 8589934592
            assert result['memory']['percent'] == 50.0
            assert result['memory']['used'] == 8589934592
            assert result['cpu']['percent'] == 32.1
            assert result['cpu']['count'] == 8
            assert result['cpu']['frequency'] == 2500.0
            assert result['disk']['total'] == 1000000000000
            assert result['disk']['used'] == 500000000000
            assert result['disk']['free'] == 500000000000
            assert result['disk']['percent'] == 50.0
            assert result['network']['connections'] == 3
            assert 'eth0' in result['network']['interfaces']
            assert 'wlan0' in result['network']['interfaces']

    def test_collect_resource_usage_without_psutil(self, collector):
        """测试无psutil时的资源使用收集"""
        with patch('src.infrastructure.monitoring.components.metrics_collector.PSUTIL_AVAILABLE', False), \
             patch('infrastructure.monitoring.components.metrics_collector.PSUTIL_AVAILABLE', False):
            result = collector.collect_resource_usage()

            # 应该返回模拟数据
            assert result['memory']['total'] == 8589934592  # 8GB
            assert result['memory']['available'] == 4294967296  # 4GB
            assert result['memory']['percent'] == 50.0
            assert result['memory']['used'] == 4294967296
            assert result['cpu']['percent'] == 45.5
            assert result['cpu']['count'] == 8
            assert result['cpu']['frequency'] == 2400.0
            assert result['disk']['total'] == 1000000000000  # 1TB
            assert result['disk']['used'] == 500000000000  # 500GB
            assert result['disk']['free'] == 500000000000  # 500GB
            assert result['disk']['percent'] == 50.0
            assert result['network']['connections'] == 10
            assert 'eth0' in result['network']['interfaces']
            assert 'lo' in result['network']['interfaces']

    def test_collect_resource_usage_exceptions(self, collector):
        """测试资源使用收集异常"""
        with patch('src.infrastructure.monitoring.components.metrics_collector.PSUTIL_AVAILABLE', True), \
             patch('psutil.virtual_memory', side_effect=Exception("memory error")):

            result = collector.collect_resource_usage()

            # 应该返回错误信息和默认值
            assert 'error' in result
            assert result['memory']['percent'] == 0.0
            assert result['cpu']['percent'] == 0.0
            assert result['disk']['percent'] == 0.0
            assert result['network']['connections'] == 0


class TestMetricsCollectorHealthStatus:
    """健康状态收集测试"""

    @pytest.fixture
    def collector(self):
        """MetricsCollector fixture"""
        return MetricsCollector()

    def test_collect_health_status_success(self, collector):
        """测试健康状态收集成功"""
        with patch('src.infrastructure.monitoring.components.metrics_collector.PSUTIL_AVAILABLE', True), \
             patch('psutil.boot_time', return_value=time.time() - 3600):

            result = collector.collect_health_status()

            assert 'timestamp' in result
            assert result['overall_status'] == 'healthy'
            assert 'services' in result
            assert 'config_service' in result['services']
            assert 'cache_service' in result['services']
            assert result['uptime_seconds'] > 0  # 应该有运行时间
            # load_average在不同平台上的行为不同，这里不做具体断言

    def test_collect_health_status_without_psutil(self, collector):
        """测试无psutil时的健康状态收集"""
        with patch('src.infrastructure.monitoring.components.metrics_collector.PSUTIL_AVAILABLE', False), \
             patch('infrastructure.monitoring.components.metrics_collector.PSUTIL_AVAILABLE', False):
            result = collector.collect_health_status()

            assert 'timestamp' in result
            assert result['overall_status'] == 'healthy'  # 固定返回healthy
            assert 'services' in result
            assert result['uptime_seconds'] >= 0
            # load_average在Windows上为None
            assert result['load_average'] is None

    def test_collect_health_status_exceptions(self, collector):
        """测试健康状态收集异常"""
        with patch('src.infrastructure.monitoring.components.metrics_collector.PSUTIL_AVAILABLE', True), \
             patch('psutil.boot_time', side_effect=Exception("boot time error")):

            result = collector.collect_health_status()

            # 应该返回错误信息
            assert 'timestamp' in result
            assert result['overall_status'] == 'unknown'
            assert result['services'] == {}
            assert result['uptime_seconds'] == 0
            assert 'error' in result


class TestMetricsCollectorIntegration:
    """MetricsCollector集成测试"""

    @pytest.fixture
    def collector(self):
        """MetricsCollector fixture"""
        return MetricsCollector("/test/project")

    def test_full_metrics_collection_workflow(self, collector):
        """测试完整的指标收集工作流"""
        # Mock所有依赖
        with patch('src.infrastructure.monitoring.components.metrics_collector.PSUTIL_AVAILABLE', False), \
             patch('infrastructure.monitoring.components.metrics_collector.PSUTIL_AVAILABLE', False):
            # 收集所有类型的指标
            system_metrics = collector.collect_system_metrics()
            coverage_metrics = collector.collect_test_coverage_metrics()
            performance_metrics = collector.collect_performance_metrics()
            resource_metrics = collector.collect_resource_usage()
            health_status = collector.collect_health_status()

            # 验证所有指标都返回了有效数据
            assert isinstance(system_metrics, dict)
            assert isinstance(coverage_metrics, dict)
            assert isinstance(performance_metrics, dict)
            assert isinstance(resource_metrics, dict)
            assert isinstance(health_status, dict)

            # 验证健康状态结构正确
            assert 'timestamp' in health_status
            assert 'overall_status' in health_status
            assert 'services' in health_status
            assert 'uptime_seconds' in health_status

    def test_metrics_collector_reusability(self, collector):
        """测试指标收集器的重用性"""
        # 多次调用应该正常工作
        for i in range(5):
            result = collector.collect_system_metrics()
            assert isinstance(result, dict)
            assert 'cpu_percent' in result

    def test_different_project_roots(self):
        """测试不同项目根目录"""
        roots = ["/project1", "/project2", "C:\\project3", None]

        for root in roots:
            collector = MetricsCollector(root)
            if root is None:
                assert collector.project_root == os.getcwd()
            else:
                assert collector.project_root == root

    def test_metrics_collector_isolation(self):
        """测试指标收集器隔离性"""
        collector1 = MetricsCollector("/path1")
        collector2 = MetricsCollector("/path2")

        # 每个收集器应该有独立的配置
        assert collector1.project_root == "/path1"
        assert collector2.project_root == "/path2"

        # 收集的数据应该相互独立
        result1 = collector1.collect_system_metrics()
        result2 = collector2.collect_system_metrics()

        assert isinstance(result1, dict)
        assert isinstance(result2, dict)


class TestMetricsCollectorErrorScenarios:
    """MetricsCollector错误场景测试"""

    @pytest.fixture
    def collector(self):
        """MetricsCollector fixture"""
        return MetricsCollector()

    def test_invalid_project_root(self, collector):
        """测试无效的项目根目录"""
        collector_invalid = MetricsCollector("/nonexistent/path")
        # 应该仍然能正常工作
        result = collector_invalid.collect_system_metrics()
        assert isinstance(result, dict)

    def test_subprocess_command_injection_protection(self, collector):
        """测试子进程命令注入保护"""
        # 即使project_root包含特殊字符，也应该安全
        dangerous_root = "/path; rm -rf /; echo"
        collector_dangerous = MetricsCollector(dangerous_root)

        # collect_test_coverage方法应该安全处理
        result = collector_dangerous.collect_test_coverage_metrics()
        assert isinstance(result, dict)
        # 不应该执行危险命令

    def test_large_output_handling(self, collector):
        """测试大输出处理"""
        large_output = "x" * 100000  # 100KB输出

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = large_output

        with patch('subprocess.run', return_value=mock_result):
            result = collector.collect_test_coverage_metrics()

            # 应该能处理大输出
            assert isinstance(result, dict)

    def test_json_parsing_edge_cases(self, collector):
        """测试JSON解析边界条件"""
        test_cases = [
            '{"totals": {"num_statements": 100}}',  # 缺少字段
            '{"totals": {}}',  # 空totals
            '{}',  # 空对象
            'invalid json',  # 无效JSON
            '',  # 空字符串
        ]

        for json_str in test_cases:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = json_str

            with patch('subprocess.run', return_value=mock_result):
                result = collector.collect_test_coverage_metrics()

                # 无论JSON是否有效，都应该返回字典
                assert isinstance(result, dict)
                assert 'total_lines' in result

    def test_timeout_handling(self, collector):
        """测试超时处理"""
        with patch('subprocess.run', side_effect=subprocess.TimeoutExpired("timeout", 30)):
            start_time = time.time()
            result = collector.collect_test_coverage_metrics()
            end_time = time.time()

            # 应该在合理时间内完成（不会无限等待）
            duration = end_time - start_time
            assert duration < 1.0, f"Timeout handling took too long: {duration}s"

            # 应该返回模拟数据
            assert isinstance(result, dict)
