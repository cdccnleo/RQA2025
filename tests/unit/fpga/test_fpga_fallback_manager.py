#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import time
from unittest.mock import MagicMock
from src.fpga.fpga_manager import FPGAManager
from src.fpga.fpga_fallback_manager import FPGAFallbackManager

class TestFPGAFallbackManager(unittest.TestCase):
    def setUp(self):
        self.fpga_manager = FPGAManager()
        self.fallback_manager = FPGAFallbackManager(self.fpga_manager)

        # Mock functions
        self.fpga_func = MagicMock(return_value="fpga_result")
        self.software_func = MagicMock(return_value="software_result")

    def test_initialization(self):
        # 测试正常初始化
        self.fpga_manager.initialize = MagicMock(return_value=True)
        self.assertTrue(self.fallback_manager.initialize())
        self.assertFalse(self.fallback_manager.fallback_mode)

        # 测试初始化失败进入降级模式
        self.fpga_manager.initialize = MagicMock(return_value=False)
        self.assertFalse(self.fallback_manager.initialize())
        self.assertTrue(self.fallback_manager.fallback_mode)

    def test_execute_with_fallback(self):
        # 测试正常FPGA执行
        self.fallback_manager.initialize()
        result = self.fallback_manager.execute_with_fallback(
            self.fpga_func, self.software_func)
        self.assertEqual(result, "fpga_result")
        self.fpga_func.assert_called_once()
        self.software_func.assert_not_called()

        # 测试降级模式执行
        self.fallback_manager.fallback_mode = True
        result = self.fallback_manager.execute_with_fallback(
            self.fpga_func, self.software_func)
        self.assertEqual(result, "software_result")
        self.software_func.assert_called_once()

        # 测试FPGA失败自动降级
        self.fallback_manager.fallback_mode = False
        self.fpga_func.side_effect = Exception("FPGA error")
        result = self.fallback_manager.execute_with_fallback(
            self.fpga_func, self.software_func)
        self.assertEqual(result, "software_result")
        self.assertTrue(self.fallback_manager.fallback_mode)

    def test_auto_recovery(self):
        # 设置降级模式
        self.fallback_manager.fallback_mode = True
        self.fallback_manager.last_failure_time = time.time() - 61  # 超过冷却期

        # Mock健康检查
        self.fpga_manager.get_device_status = MagicMock(return_value={
            'status': 'ready',
            'last_heartbeat': time.time()
        })

        # 测试自动恢复
        self.assertTrue(self.fallback_manager.auto_recovery())
        self.assertFalse(self.fallback_manager.fallback_mode)

        # 测试未恢复情况
        self.fallback_manager.fallback_mode = True
        self.fpga_manager.get_device_status = MagicMock(return_value=None)
        self.assertFalse(self.fallback_manager.auto_recovery())
        self.assertTrue(self.fallback_manager.fallback_mode)

if __name__ == '__main__':
    unittest.main()
