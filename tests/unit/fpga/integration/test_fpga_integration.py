#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FPGA模块集成测试
测试FPGA各组件间的集成和协同工作
"""

import unittest
from unittest.mock import patch, MagicMock
from src.fpga.fpga_manager import FPGAManager
from src.fpga.fpga_risk_engine import FPGARiskEngine
from src.fpga.fpga_fallback_manager import FPGAFallbackManager
from src.fpga.fpga_performance_monitor import FPGAPerformanceMonitor

class TestFPGAIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 初始化所有FPGA组件
        cls.fpga_manager = FPGAManager()
        cls.risk_engine = FPGARiskEngine(cls.fpga_manager)
        cls.fallback_manager = FPGAFallbackManager(cls.fpga_manager)
        cls.performance_monitor = FPGAPerformanceMonitor(cls.fpga_manager)

        # Mock FPGA管理器初始化
        cls.fpga_manager.initialize = MagicMock(return_value=True)
        cls.fpga_manager.get_device_status = MagicMock(return_value={
            'status': 'ready',
            'utilization': 0.6,
            'last_heartbeat': time.time()
        })

    def test_risk_engine_integration(self):
        """测试风险引擎与FPGA管理器的集成"""
        # 设置FPGA风险检查结果
        self.fpga_manager.execute_command = MagicMock(return_value={
            'circuit_breaker': False,
            'price_limit': False,
            't1_restriction': False
        })

        # 执行风险检查
        result = self.risk_engine.check_risks(order={})
        self.assertFalse(result['has_risk'])
        self.fpga_manager.execute_command.assert_called_once()

    def test_fallback_mechanism(self):
        """测试降级管理器与FPGA的协同工作"""
        # 第一次调用使用FPGA
        with patch.object(self.fpga_manager, 'execute_command',
                         return_value="fpga_result") as mock_exec:
            result = self.fallback_manager.execute_with_fallback(
                self.fpga_manager.execute_command,
                lambda: "software_result",
                "test_command"
            )
            self.assertEqual(result, "fpga_result")
            mock_exec.assert_called_once_with("test_command")

        # 模拟FPGA失败，应自动降级
        with patch.object(self.fpga_manager, 'execute_command',
                         side_effect=Exception("FPGA error")) as mock_exec:
            result = self.fallback_manager.execute_with_fallback(
                self.fpga_manager.execute_command,
                lambda: "software_result",
                "test_command"
            )
            self.assertEqual(result, "software_result")
            self.assertTrue(self.fallback_manager.fallback_mode)

    def test_performance_monitoring(self):
        """测试性能监控器集成"""
        # 记录延迟
        self.performance_monitor.record_latency("risk_check", 0.05)

        # 更新资源利用率
        self.performance_monitor.update_utilization()

        # 获取报告
        report = self.performance_monitor.generate_report()
        self.assertEqual(report['latency_stats']['count'], 1)
        self.assertEqual(report['utilization_stats']['count'], 1)

    def test_end_to_end_workflow(self):
        """测试端到端交易流程"""
        # 设置mock
        self.fpga_manager.execute_command = MagicMock(side_effect=[
            {"circuit_breaker": False},  # 风险检查结果
            {"vwap": 10.5, "twap": 10.6}  # 订单优化结果
        ])

        # 执行完整流程
        risk_result = self.risk_engine.check_risks(order={})
        self.assertFalse(risk_result['has_risk'])

        # 记录性能
        self.performance_monitor.record_latency("risk_check", 0.03)

        # 检查降级状态
        if not self.fallback_manager.fallback_mode:
            # 使用FPGA优化订单
            optimized = self.fallback_manager.execute_with_fallback(
                lambda: self.fpga_manager.execute_command("optimize_order"),
                lambda: {"vwap": 10.5, "twap": 10.5},  # 软件实现
            )
            self.assertEqual(optimized['vwap'], 10.5)

        # 更新性能数据
        self.performance_monitor.update_utilization()

if __name__ == '__main__':
    unittest.main()
