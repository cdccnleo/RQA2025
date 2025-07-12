#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock
from src.fpga.fpga_manager import FPGAManager
from src.fpga.fpga_performance_monitor import FPGAPerformanceMonitor

class TestFPGAPerformanceMonitor(unittest.TestCase):
    def setUp(self):
        self.fpga_manager = FPGAManager()
        self.monitor = FPGAPerformanceMonitor(self.fpga_manager)

        # Mock FPGA管理器
        self.fpga_manager.get_device_status = MagicMock(return_value={
            'utilization': 0.75
        })

    def test_record_latency(self):
        # 测试记录延迟
        self.monitor.record_latency('risk_check', 0.05)
        self.monitor.record_latency('order_optimize', 0.12)  # 超过阈值

        self.assertEqual(len(self.monitor.metrics['latency']), 2)
        self.assertEqual(self.monitor.metrics['latency'][0]['operation'], 'risk_check')
        self.assertEqual(self.monitor.metrics['latency'][1]['value'], 0.12)

    def test_record_throughput(self):
        # 测试记录吞吐量
        self.monitor.record_throughput('risk_check', 1000)
        self.monitor.record_throughput('order_optimize', 500)

        self.assertEqual(len(self.monitor.metrics['throughput']), 2)
        self.assertEqual(self.monitor.metrics['throughput'][0]['value'], 1000)

    def test_update_utilization(self):
        # 测试更新资源利用率
        self.monitor.update_utilization()
        self.assertEqual(len(self.monitor.metrics['utilization']), 1)
        self.assertEqual(self.monitor.metrics['utilization'][0]['value'], 0.75)

        # 测试超过阈值情况
        self.fpga_manager.get_device_status.return_value = {'utilization': 0.95}
        self.monitor.update_utilization()
        self.assertEqual(len(self.monitor.metrics['utilization']), 2)

    def test_get_recent_metrics(self):
        # 测试获取最近指标
        now = datetime.now()
        test_data = [
            {'timestamp': now - timedelta(minutes=6), 'value': 0.04},
            {'timestamp': now - timedelta(minutes=4), 'value': 0.06},
            {'timestamp': now - timedelta(minutes=2), 'value': 0.05}
        ]
        self.monitor.metrics['latency'] = [
            {'timestamp': ts, 'operation': 'test', 'value': val['value']}
            for ts, val in zip([d['timestamp'] for d in test_data], test_data)
        ]

        recent = self.monitor.get_recent_metrics('latency', minutes=5)
        self.assertEqual(len(recent), 2)  # 应该返回最近5分钟的2条记录

    def test_generate_report(self):
        # 测试生成报告
        self.monitor.record_latency('risk_check', 0.05)
        self.monitor.record_latency('order_optimize', 0.12)  # 超过阈值
        self.monitor.record_throughput('risk_check', 1000)
        self.monitor.update_utilization()

        report = self.monitor.generate_report()
        self.assertIn('latency_stats', report)
        self.assertIn('throughput_stats', report)
        self.assertIn('utilization_stats', report)
        self.assertIn('warning_count', report)

        # 检查统计信息
        self.assertEqual(report['latency_stats']['max'], 0.12)
        self.assertEqual(report['warning_count']['latency'], 1)

if __name__ == '__main__':
    unittest.main()
