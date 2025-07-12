"""风控系统集成测试"""
import unittest
import numpy as np
from unittest.mock import MagicMock, patch
from src.risk.risk_controller import RiskController, ChinaRiskController
from src.features.feature_engine import FeatureEngine
from src.fpga.fpga_accelerator import FpgaManager
from src.trading.order_executor import Order

class TestRiskControllerIntegration(unittest.TestCase):
    """风控控制器集成测试用例"""

    def setUp(self):
        """测试初始化"""
        self.engine = FeatureEngine()

        # 创建风控控制器
        self.controller = RiskController(self.engine)
        self.china_controller = ChinaRiskController(self.engine)

        # 测试数据
        self.test_order = Order(
            symbol="600519.SH",
            price=100.0,
            quantity=100,
            side="BUY"
        )

        self.test_market_data = {
            "volatility": 0.02,
            "liquidity": 10000,
            "market_drop": 0.03
        }

        # 批量测试数据
        self.batch_orders = [self.test_order] * 3
        self.batch_market_data = [self.test_market_data] * 3

    def test_basic_risk_check(self):
        """测试基础风控检查"""
        result = self.controller.check_order(self.test_order, self.test_market_data)
        self.assertTrue(result)

        # 测试无效订单
        invalid_order = Order(
            symbol="600519.SH",
            price=100.0,
            quantity=-100,  # 无效数量
            side="BUY"
        )
        result = self.controller.check_order(invalid_order, self.test_market_data)
        self.assertFalse(result)

    def test_a_share_specific_rules(self):
        """测试A股特定风控规则"""
        # 测试ST股票
        st_order = Order(
            symbol="ST600519",
            price=5.0,
            quantity=100,
            side="BUY"
        )
        result = self.china_controller.check_order(st_order, self.test_market_data)
        self.assertTrue(result)

        # 测试科创板
        star_order = Order(
            symbol="688981.SH",
            price=50.0,
            quantity=200,
            side="BUY"
        )
        result = self.china_controller.check_order(star_order, self.test_market_data)
        self.assertTrue(result)

    def test_fpga_acceleration(self):
        """测试FPGA加速路径"""
        # 模拟FPGA加速器
        fpga_mock = MagicMock()
        fpga_mock.health_monitor.is_healthy.return_value = True
        fpga_mock._check_with_fpga.return_value = True

        with patch.object(FpgaManager, 'get_accelerator', return_value=fpga_mock):
            result = self.controller.check_order(self.test_order, self.test_market_data)
            self.assertTrue(result)
            fpga_mock._check_with_fpga.assert_called_once()

    def test_software_fallback(self):
        """测试软件降级路径"""
        # 禁用FPGA
        self.controller.config.use_fpga = False

        result = self.controller.check_order(self.test_order, self.test_market_data)
        self.assertTrue(result)

    def test_circuit_breaker(self):
        """测试熔断机制"""
        # 触发5%熔断
        market_data = {**self.test_market_data, "market_drop": 0.05}
        result = self.controller.check_order(self.test_order, market_data)
        self.assertFalse(result)

        # 触发7%熔断
        market_data = {**self.test_market_data, "market_drop": 0.07}
        result = self.controller.check_order(self.test_order, market_data)
        self.assertFalse(result)

    def test_batch_check(self):
        """测试批量风控检查"""
        results = self.controller.batch_check(self.batch_orders, self.batch_market_data)

        # 检查批量结果
        self.assertEqual(len(results), len(self.batch_orders))
        for result in results:
            self.assertTrue(result)

    def test_integration_with_order_executor(self):
        """测试与订单执行模块的集成"""
        # 模拟订单执行
        order = self.test_order

        # 风控检查
        result = self.controller.check_order(order, self.test_market_data)
        self.assertTrue(result)

    def test_performance_metrics(self):
        """测试性能指标"""
        import time

        # 大批量测试
        large_batch_orders = [self.test_order] * 1000
        large_batch_market_data = [self.test_market_data] * 1000

        start = time.time()
        self.controller.batch_check(large_batch_orders, large_batch_market_data)
        elapsed = time.time() - start

        # 检查性能
        self.assertLess(elapsed, 1.0)  # 1000次检查应在1秒内完成
        print(f"风控检查性能: {1000/elapsed:.2f} 次/秒")

if __name__ == '__main__':
    unittest.main()
