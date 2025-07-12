#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RQA2025 混沌引擎集成测试
测试混沌引擎与监控系统的集成
"""

import unittest
from unittest.mock import patch, MagicMock
import time
from src.infrastructure.testing.chaos_engine import ChaosEngine
from src.infrastructure.monitoring.prometheus_monitor import PrometheusMonitor

class TestChaosIntegration(unittest.TestCase):
    """混沌引擎集成测试"""

    def setUp(self):
        """测试初始化"""
        self.monitor = PrometheusMonitor()
        self.engine = ChaosEngine(enable_production=False)

        # Mock Docker客户端和容器
        self.mock_container = MagicMock()
        self.mock_container.id = "test123"
        self.mock_container.name = "test_service"
        self.mock_container.status = "running"

        self.mock_docker_client = MagicMock()
        self.mock_docker_client.containers.list.return_value = [self.mock_container]
        self.engine.docker_client = self.mock_docker_client

        # Mock Prometheus客户端
        self.mock_prometheus = MagicMock()
        self.monitor.client = self.mock_prometheus

    @patch('subprocess.run')
    @patch('src.infrastructure.monitoring.prometheus_monitor.PrometheusMonitor.alert')
    def test_network_partition_with_monitoring(self, mock_alert, mock_subprocess):
        """测试网络分区与监控告警集成"""
        # 准备模拟数据
        mock_subprocess.return_value = MagicMock(returncode=0)

        # 执行混沌实验
        report = self.engine.simulate_network_partition(duration=5)

        # 验证监控告警被触发
        mock_alert.assert_called_with(
            "ChaosEngine",
            f"Network partition simulated on {report.affected_components}",
            severity="warning"
        )

        # 验证恢复后被清除
        self.assertEqual(len(self.engine.active_faults), 0)

    @patch('subprocess.run')
    @patch('src.infrastructure.monitoring.prometheus_monitor.PrometheusMonitor.alert')
    def test_fpga_failure_with_monitoring(self, mock_alert, mock_subprocess):
        """测试FPGA故障与监控告警集成"""
        # 准备模拟数据
        mock_subprocess.return_value = MagicMock(returncode=0)

        # 执行混沌实验
        report = self.engine.simulate_fpga_failure(duration=5)

        # 验证监控告警被触发
        mock_alert.assert_called_with(
            "ChaosEngine",
            "FPGA failure simulated (mode: complete)",
            severity="critical"
        )

        # 验证指标被记录
        self.mock_prometheus.send_metric.assert_called_with(
            "chaos_experiment_duration_seconds",
            report.recovery_time,
            labels={"experiment": "fpga_failure"}
        )

    @patch('subprocess.run')
    @patch('src.infrastructure.monitoring.prometheus_monitor.PrometheusMonitor.alert')
    def test_emergency_recovery_alert(self, mock_alert, mock_subprocess):
        """测试紧急恢复触发告警"""
        # 模拟命令执行失败
        mock_subprocess.side_effect = Exception("Command failed")

        # 执行会失败的实验
        with self.assertRaises(ChaosError):
            self.engine.simulate_network_partition(duration=5)

        # 验证紧急恢复告警
        mock_alert.assert_called_with(
            "ChaosEngine",
            "Emergency recovery activated",
            severity="critical"
        )


if __name__ == '__main__':
    unittest.main()
