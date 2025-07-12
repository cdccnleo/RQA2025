"""智能订单执行集成测试"""
import unittest
import numpy as np
from unittest.mock import MagicMock, patch
from src.trading.order_executor import OrderExecutor, ChinaOrderExecutor
from src.features.feature_engine import FeatureEngine
from src.fpga.fpga_accelerator import FpgaManager
from src.signal.signal_generator import Signal

class TestOrderExecutorIntegration(unittest.TestCase):
    """订单执行器集成测试用例"""

    def setUp(self):
        """测试初始化"""
        self.engine = FeatureEngine()

        # 创建执行器
        self.executor = OrderExecutor(self.engine)
        self.china_executor = ChinaOrderExecutor(self.engine)

        # 测试数据
        self.test_signal = Signal(
            symbol="600519.SH",
            signal="BUY",
            confidence=0.8,
            target_price=100.0,
            position=0.1
        )

        self.test_market_data = {
            "bid": 99.9,
            "ask": 100.1,
            "volume": 10000,
            "spread": 0.2
        }

        # 批量测试数据
        self.batch_signals = [self.test_signal] * 3
        self.batch_market_data = [self.test_market_data] * 3

    def test_basic_order_execution(self):
        """测试基本订单执行"""
        execution = self.executor.execute(self.test_signal, self.test_market_data)

        # 检查执行结果结构
        self.assertIn("status", execution)
        self.assertIn("avg_price", execution)
        self.assertIn("filled_quantity", execution)
        self.assertIn("fees", execution)

        # 检查执行状态
        self.assertEqual(execution["status"], "SUCCESS")

    def test_a_share_specific_rules(self):
        """测试A股特定规则"""
        # 测试ST股票
        st_signal = Signal(
            symbol="ST600519",
            signal="BUY",
            confidence=0.8,
            target_price=5.0,
            position=0.1
        )
        execution = self.china_executor.execute(st_signal, self.test_market_data)

        # 检查ST股票执行结果
        self.assertEqual(execution["status"], "SUCCESS")
        self.assertEqual(execution["filled_quantity"] % 100, 0)  # 确保是100的整数倍

        # 测试科创板
        star_signal = Signal(
            symbol="688981.SH",
            signal="BUY",
            confidence=0.8,
            target_price=50.0,
            position=0.1
        )
        execution = self.china_executor.execute(star_signal, self.test_market_data)
        self.assertEqual(execution["filled_quantity"] % 200, 0)  # 科创板200股整数倍

    def test_fpga_acceleration(self):
        """测试FPGA加速路径"""
        # 模拟FPGA加速器
        fpga_mock = MagicMock()
        fpga_mock.health_monitor.is_healthy.return_value = True
        fpga_mock._optimize_with_fpga.return_value = {
            "price": 99.9,
            "quantity": 100,
            "strategy": "TWAP"
        }

        with patch.object(FpgaManager, 'get_accelerator', return_value=fpga_mock):
            execution = self.executor.execute(self.test_signal, self.test_market_data)
            self.assertEqual(execution["status"], "SUCCESS")
            fpga_mock._optimize_with_fpga.assert_called_once()

    def test_software_fallback(self):
        """测试软件降级路径"""
        # 禁用FPGA
        self.executor.config.use_fpga = False

        execution = self.executor.execute(self.test_signal, self.test_market_data)
        self.assertEqual(execution["status"], "SUCCESS")

    def test_a_share_fees_calculation(self):
        """测试A股费用计算"""
        # 买入订单
        buy_execution = self.china_executor.execute(self.test_signal, self.test_market_data)
        self.assertGreater(buy_execution["fees"], 0)

        # 卖出订单
        sell_signal = Signal(
            symbol="600519.SH",
            signal="SELL",
            confidence=0.8,
            target_price=100.0,
            position=0.1
        )
        sell_execution = self.china_executor.execute(sell_signal, self.test_market_data)
        self.assertGreater(sell_execution["fees"], buy_execution["fees"])  # 卖出有印花税

    def test_batch_execution(self):
        """测试批量订单执行"""
        executions = self.executor.batch_execute(self.batch_signals, self.batch_market_data)

        # 检查批量结果
        self.assertEqual(len(executions), len(self.batch_signals))
        for execution in executions:
            self.assertEqual(execution["status"], "SUCCESS")

    def test_integration_with_signal_generator(self):
        """测试与信号生成模块的集成"""
        # 生成测试信号
        signal = self.test_signal

        # 执行订单
        execution = self.executor.execute(signal, self.test_market_data)
        self.assertEqual(execution["status"], "SUCCESS")

    def test_performance_metrics(self):
        """测试性能指标"""
        import time

        # 大批量测试
        large_batch_signals = [self.test_signal] * 1000
        large_batch_market_data = [self.test_market_data] * 1000

        start = time.time()
        self.executor.batch_execute(large_batch_signals, large_batch_market_data)
        elapsed = time.time() - start

        # 检查性能
        self.assertLess(elapsed, 1.0)  # 1000次执行应在1秒内完成
        print(f"订单执行性能: {1000/elapsed:.2f} 次/秒")

if __name__ == '__main__':
    unittest.main()
