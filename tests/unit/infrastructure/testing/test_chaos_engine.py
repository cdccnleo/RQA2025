#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RQA2025 混沌引擎测试用例
"""

import unittest
from unittest.mock import patch, MagicMock
import docker
import psutil
from src.infrastructure.testing.chaos_engine import (
    ChaosEngine,
    ChaosError,
    FaultType,
    ChaosReport
)

class TestChaosEngine(unittest.TestCase):
    """混沌引擎单元测试"""

    def setUp(self):
        """测试初始化"""
        try:
            self.engine = ChaosEngine(enable_production=False)
            # 尝试获取真实Docker客户端
            self.engine.docker_client = docker.from_env()
            # 测试连接是否有效
            self.engine.docker_client.ping()
        except Exception as e:
            # 如果Docker不可用，使用mock
            self.mock_container = MagicMock()
            self.mock_container.id = "test123"
            self.mock_container.name = "test_service"
            self.mock_container.status = "running"

            self.mock_docker_client = MagicMock()
            self.mock_docker_client.containers.list.return_value = [self.mock_container]
            self.mock_docker_client.ping.return_value = True
            self.engine.docker_client = self.mock_docker_client

    @patch('subprocess.run')
    @patch('psutil.getloadavg')
    @patch('psutil.cpu_count')
    def test_simulate_network_partition_success(self, mock_cpu_count, mock_loadavg, mock_subprocess):
        """测试成功模拟网络分区"""
        # 准备模拟数据
        mock_cpu_count.return_value = 4
        mock_loadavg.return_value = (1.0, 1.0, 1.0)  # 低负载
        mock_subprocess.return_value = MagicMock(returncode=0)

        # 执行测试
        report = self.engine.simulate_network_partition(duration=5, target_services=["test_service"])

        # 验证结果
        self.assertTrue(report.is_success)
        self.assertEqual(report.fault_type, FaultType.NETWORK_PARTITION)
        self.assertEqual(len(report.affected_components), 1)

        # 验证Docker API调用
        self.mock_docker_client.containers.list.assert_called()

    @patch('subprocess.run')
    def test_simulate_network_partition_failure(self, mock_subprocess):
        """测试网络分区模拟失败"""
        # 模拟命令执行失败
        mock_subprocess.side_effect = Exception("Command failed")

        # 验证抛出异常
        with self.assertRaises(ChaosError):
            self.engine.simulate_network_partition(duration=5)

    @patch('subprocess.run')
    def test_simulate_fpga_failure_complete(self, mock_subprocess):
        """测试模拟FPGA完全故障"""
        # 模拟命令执行成功
        mock_subprocess.return_value = MagicMock(returncode=0)

        # 执行测试
        report = self.engine.simulate_fpga_failure(duration=5, failure_mode="complete")

        # 验证结果
        self.assertTrue(report.is_success)
        self.assertEqual(report.fault_type, FaultType.FPGA_FAILURE)
        self.assertEqual(report.affected_components, ["fpga_accelerator"])

    @patch('subprocess.run')
    def test_simulate_fpga_failure_invalid_mode(self, mock_subprocess):
        """测试无效的FPGA故障模式"""
        # 验证抛出异常
        with self.assertRaises(ChaosError):
            self.engine.simulate_fpga_failure(duration=5, failure_mode="invalid_mode")

    def test_emergency_recovery(self):
        """测试紧急恢复功能"""
        # 添加模拟故障
        self.engine.active_faults.add((FaultType.NETWORK_PARTITION, "test123"))

        # 执行恢复
        self.engine._emergency_recovery()

        # 验证故障已清除
        self.assertEqual(len(self.engine.active_faults), 0)

    @patch('builtins.open')
    def test_safeguard_production_check(self, mock_open):
        """测试生产环境安全检查"""
        # 模拟生产环境
        mock_open.return_value.__enter__.return_value.read.return_value = "ENV=production"

        # 验证返回False
        self.assertFalse(self.engine.safeguard.check_environment())

    @patch('psutil.getloadavg')
    @patch('psutil.cpu_count')
    def test_safeguard_high_load_check(self, mock_cpu_count, mock_loadavg):
        """测试高负载安全检查"""
        # 模拟高负载
        mock_cpu_count.return_value = 4
        mock_loadavg.return_value = (3.5, 3.5, 3.5)  # 高负载

        # 验证返回False
        self.assertFalse(self.engine.safeguard.check_environment())

    @patch('docker.DockerClient')
    def test_safeguard_critical_services_check(self, mock_docker):
        """测试关键服务检查"""
        # 模拟关键服务异常
        mock_client = MagicMock()
        mock_client.containers.list.return_value = []
        self.engine.docker_client = mock_client

        # 验证返回False
        self.assertFalse(self.engine.safeguard._critical_services_ok())


class TestChaosReport(unittest.TestCase):
    """混沌测试报告测试"""

    def test_report_creation(self):
        """测试报告创建"""
        report = ChaosReport(
            fault_type=FaultType.NETWORK_PARTITION,
            start_time=1000,
            end_time=1010,
            affected_components=["service1", "service2"],
            recovery_time=10,
            is_success=True
        )

        self.assertEqual(report.fault_type, FaultType.NETWORK_PARTITION)
        self.assertEqual(report.recovery_time, 10)
        self.assertTrue(report.is_success)


if __name__ == '__main__':
    unittest.main()
