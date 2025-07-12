#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FPGA压力测试
测试FPGA模块在高负载下的表现
"""

import unittest
import time
import threading
from unittest.mock import MagicMock
from src.fpga.fpga_manager import FPGAManager
from src.fpga.fpga_risk_engine import FPGARiskEngine
from src.fpga.fpga_fallback_manager import FPGAFallbackManager
from src.fpga.fpga_performance_monitor import FPGAPerformanceMonitor

class TestFPGAStress(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 初始化所有FPGA组件
        cls.fpga_manager = FPGAManager()
        cls.risk_engine = FPGARiskEngine(cls.fpga_manager)
        cls.fallback_manager = FPGAFallbackManager(cls.fpga_manager)
        cls.performance_monitor = FPGAPerformanceMonitor(cls.fpga_manager)

        # Mock FPGA管理器
        cls.fpga_manager.initialize = MagicMock(return_value=True)
        cls.fpga_manager.get_device_status = MagicMock(return_value={
            'status': 'ready',
            'utilization': 0.0,
            'last_heartbeat': time.time()
        })
        cls.fpga_manager.execute_command = MagicMock(return_value={
            'circuit_breaker': False,
            'price_limit': False,
            't1_restriction': False
        })

    def test_high_concurrency(self):
        """测试高并发风险检查"""
        # 模拟100个并发请求
        threads = []
        results = []

        def run_risk_check():
            result = self.risk_engine.check_risks(order={})
            results.append(result)

        for _ in range(100):
            t = threading.Thread(target=run_risk_check)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # 验证所有请求都成功处理
        self.assertEqual(len(results), 100)
        self.assertTrue(all(not r['has_risk'] for r in results))

        # 检查资源利用率
        utilization = self.performance_monitor.generate_report()['utilization_stats']['max']
        self.assertLess(utilization, 1.0, "资源利用率不应超过100%")

    def test_resource_exhaustion(self):
        """测试FPGA资源耗尽情况"""
        # 模拟资源耗尽
        self.fpga_manager.get_device_status = MagicMock(return_value={
            'status': 'overloaded',
            'utilization': 0.99,
            'last_heartbeat': time.time()
        })

        # 执行风险检查，应触发降级
        result = self.fallback_manager.execute_with_fallback(
            lambda: self.risk_engine.check_risks(order={}),
            lambda: {'has_risk': False, 'reason': 'software fallback'},
            "risk_check"
        )

        self.assertTrue(self.fallback_manager.fallback_mode)
        self.assertEqual(result['reason'], 'software fallback')

    def test_long_running_stability(self):
        """测试长时间运行的稳定性"""
        # 模拟长时间运行(1000次连续调用)
        for i in range(1000):
            # 每100次模拟一次设备状态更新
            if i % 100 == 0:
                self.fpga_manager.get_device_status = MagicMock(return_value={
                    'status': 'ready',
                    'utilization': i/1500.0,  # 模拟逐渐增加的负载
                    'last_heartbeat': time.time()
                })

            self.risk_engine.check_risks(order={})
            self.performance_monitor.update_utilization()

        # 检查性能监控数据
        report = self.performance_monitor.generate_report()
        self.assertEqual(report['latency_stats']['count'], 1000)
        self.assertLess(report['utilization_stats']['max'], 0.8, "资源利用率应保持稳定")

    def test_fallback_performance(self):
        """测试降级模式下的性能"""
        # 强制进入降级模式
        self.fallback_manager.fallback_mode = True

        # 测试100次降级调用
        start_time = time.time()
        for _ in range(100):
            self.risk_engine.check_risks(order={})
        elapsed = time.time() - start_time

        # 检查性能(软件实现应比FPGA慢)
        self.assertGreater(elapsed, 0.01, "软件实现应有合理延迟")
        self.assertLess(elapsed, 1.0, "软件实现不应过慢")

if __name__ == '__main__':
    unittest.main()
