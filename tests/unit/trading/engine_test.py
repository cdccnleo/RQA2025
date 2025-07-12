import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.trading.engine import TradingEngine
from src.trading.models import Order, Execution

class TestTradingEngine(unittest.TestCase):
    """统一后的交易引擎测试"""

    def setUp(self):
        self.engine = TradingEngine()
        self.sample_signal = {
            'symbol': '600000.SH',
            'price': 42.50,
            'quantity': 100,
            'side': 'BUY'
        }

        # 模拟市场数据
        self.market_data = pd.DataFrame({
            'symbol': ['600000.SH'],
            'bid': [42.49],
            'ask': [42.51],
            'volume': [100000]
        })

    def test_order_generation(self):
        """测试订单生成"""
        orders = self.engine.generate_orders([self.sample_signal], self.market_data)
        self.assertEqual(len(orders), 1)
        self.assertIsInstance(orders[0], Order)
        self.assertEqual(orders[0].symbol, '600000.SH')

    # 从test_trading_engine.py合并的测试
    @patch('src.trading.engine.RiskController.validate_order')
    def test_risk_validation(self, mock_validate):
        """测试风险控制验证"""
        mock_validate.return_value = True
        order = Order(symbol='600000.SH', quantity=100, side='BUY')
        result = self.engine.validate_order(order)
        self.assertTrue(result)

    # 从test_order_manager.py合并的测试
    def test_order_execution(self):
        """测试订单执行"""
        order = Order(symbol='600000.SH', quantity=100, side='BUY')
        with patch('src.trading.engine.BrokerAPI.execute') as mock_execute:
            mock_execute.return_value = Execution(
                order_id=order.id,
                filled_quantity=100,
                avg_price=42.50
            )
            execution = self.engine.execute_order(order)
            self.assertEqual(execution.filled_quantity, 100)

    # 从test_execution_algorithm.py合并的测试
    def test_smart_execution(self):
        """测试智能执行算法"""
        large_order = Order(symbol='600000.SH', quantity=10000, side='BUY')
        executions = self.engine.smart_execute(large_order, self.market_data)
        self.assertTrue(len(executions) > 1)  # 应拆分执行
        self.assertEqual(sum(e.filled_quantity for e in executions), 10000)

    # 新增性能测试
    def test_high_frequency_execution(self):
        """测试高频交易性能"""
        orders = [Order(symbol='600000.SH', quantity=100, side='BUY')] * 100
        import time
        start = time.time()
        results = self.engine.batch_execute(orders, self.market_data)
        elapsed = time.time() - start
        self.assertEqual(len(results), 100)
        self.assertLess(elapsed, 0.5)  # 应在0.5秒内完成

if __name__ == '__main__':
    unittest.main()
