#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RQA2025 混沌编排器测试用例
"""

import unittest
from unittest.mock import patch, MagicMock
import yaml
from datetime import datetime
from src.infrastructure.testing.chaos_orchestrator import ChaosOrchestrator
from src.infrastructure.testing.chaos_engine import ChaosError

class TestChaosOrchestrator(unittest.TestCase):
    """混沌编排器测试"""

    def setUp(self):
        """测试初始化"""
        # 模拟配置文件内容
        self.mock_config = {
            'experiments': [
                {
                    'name': "网络分区测试",
                    'type': "network_partition",
                    'enabled': True,
                    'duration': 60,
                    'target_services': ["service1"]
                },
                {
                    'name': "FPGA故障测试",
                    'type': "fpga_failure",
                    'enabled': True,
                    'duration': 30,
                    'mode': "complete"
                },
                {
                    'name': "禁用实验",
                    'type': "network_partition",
                    'enabled': False,
                    'duration': 10
                }
            ]
        }

        # 模拟YAML加载
        self.patcher = patch('yaml.safe_load', return_value=self.mock_config)
        self.mock_yaml = self.patcher.start()

        # 初始化编排器(使用模拟配置路径)
        self.orchestrator = ChaosOrchestrator("mock_path.yaml")

        # 模拟混沌引擎
        self.mock_engine = MagicMock()
        self.orchestrator.engine = self.mock_engine

    def tearDown(self):
        """测试清理"""
        self.patcher.stop()

    def test_load_config(self):
        """测试配置加载"""
        # 验证YAML被正确加载
        self.mock_yaml.assert_called_once()
        self.assertEqual(len(self.orchestrator.experiments), 3)

    def test_run_enabled_experiment(self):
        """测试执行启用的实验"""
        # 设置模拟返回值
        self.mock_engine.simulate_network_partition.return_value = MagicMock(
            recovery_time=60,
            is_success=True,
            affected_components=["service1"]
        )

        # 执行测试
        result = self.orchestrator.run_experiment_by_name("网络分区测试")

        # 验证结果
        self.assertTrue(result['success'])
        self.assertEqual(result['duration'], 60)
        self.mock_engine.simulate_network_partition.assert_called_once_with(
            duration=60,
            target_services=["service1"]
        )

    def test_run_disabled_experiment(self):
        """测试执行禁用的实验"""
        result = self.orchestrator.run_experiment_by_name("禁用实验")
        self.assertIsNone(result)

    def test_run_nonexistent_experiment(self):
        """测试执行不存在的实验"""
        result = self.orchestrator.run_experiment_by_name("不存在的实验")
        self.assertIsNone(result)

    def test_run_fpga_experiment(self):
        """测试执行FPGA故障实验"""
        # 设置模拟返回值
        self.mock_engine.simulate_fpga_failure.return_value = MagicMock(
            recovery_time=30,
            is_success=True,
            affected_components=["fpga_accelerator"]
        )

        # 执行测试
        result = self.orchestrator.run_experiment_by_name("FPGA故障测试")

        # 验证结果
        self.assertTrue(result['success'])
        self.assertEqual(result['duration'], 30)
        self.mock_engine.simulate_fpga_failure.assert_called_once_with(
            duration=30,
            failure_mode="complete"
        )

    @patch('apscheduler.schedulers.background.BackgroundScheduler.add_job')
    def test_schedule_experiments(self, mock_add_job):
        """测试实验调度"""
        # 执行调度
        self.orchestrator.schedule_experiments()

        # 验证只有启用的实验被调度
        self.assertEqual(mock_add_job.call_count, 2)

        # 验证调度参数
        calls = mock_add_job.call_args_list
        self.assertEqual(calls[0][0][1], '网络分区测试')
        self.assertEqual(calls[1][0][1], 'FPGA故障测试')

    def test_list_available_experiments(self):
        """测试获取可用实验列表"""
        experiments = self.orchestrator.list_available_experiments()

        # 验证返回格式和数量
        self.assertEqual(len(experiments), 3)
        self.assertEqual(experiments[0]['name'], "网络分区测试")
        self.assertTrue(experiments[0]['enabled'])
        self.assertFalse(experiments[2]['enabled'])


if __name__ == '__main__':
    unittest.main()
