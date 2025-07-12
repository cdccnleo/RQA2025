"""结算引擎集成测试"""
import unittest
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
from src.settlement.settlement_engine import SettlementEngine, ChinaSettlementEngine
from src.features.feature_engine import FeatureEngine
from src.trading.order_executor import Trade

class TestSettlementEngineIntegration(unittest.TestCase):
    """结算引擎集成测试用例"""

    def setUp(self):
        """测试初始化"""
        self.engine = FeatureEngine()

        # 创建结算引擎
        self.settlement = SettlementEngine(self.engine)
        self.china_settlement = ChinaSettlementEngine(self.engine)

        # 测试数据
        self.test_trade = Trade(
            symbol="600519.SH",
            price=100.0,
            quantity=100,
            side="BUY",
            prev_close=95.0,
            close_price=98.0
        )

        # 批量测试数据
        self.batch_trades = [self.test_trade] * 3

    def test_basic_settlement(self):
        """测试基础结算功能"""
        # 处理T+1结算
        frozen_amount = self.settlement.process_t1_settlement([self.test_trade])
        self.assertGreater(frozen_amount, 0)

        # 检查冻结资金
        self.assertEqual(self.settlement.frozen_cash, frozen_amount)

        # 释放结算资金
        released_amount = self.settlement.release_settlement()
        self.assertEqual(released_amount, frozen_amount)
        self.assertEqual(self.settlement.frozen_cash, 0)

    def test_a_share_t1_settlement(self):
        """测试A股T+1结算"""
        # 处理T+1结算
        frozen_amount = self.china_settlement.process_t1_settlement([self.test_trade])
        self.assertGreater(frozen_amount, 0)

        # 检查冻结资金
        self.assertEqual(self.china_settlement.frozen_cash, frozen_amount)

        # 检查持仓
        self.assertEqual(self.china_settlement.settled_positions.get("600519.SH", 0), 100)

    def test_a_share_fees_calculation(self):
        """测试A股费用计算"""
        # 买入订单
        buy_trade = Trade(
            symbol="600519.SH",
            price=100.0,
            quantity=100,
            side="BUY",
            prev_close=95.0
        )
        buy_fees = self.china_settlement._calculate_a_share_fees(buy_trade)
        self.assertGreater(buy_fees, 0)

        # 卖出订单
        sell_trade = Trade(
            symbol="600519.SH",
            price=100.0,
            quantity=100,
            side="SELL",
            prev_close=95.0
        )
        sell_fees = self.china_settlement._calculate_a_share_fees(sell_trade)
        self.assertGreater(sell_fees, buy_fees)  # 卖出有印花税

    def test_star_market_settlement(self):
        """测试科创板结算"""
        star_trade = Trade(
            symbol="688981.SH",
            price=50.0,
            quantity=200,  # 科创板200股整数倍
            side="BUY",
            prev_close=45.0
        )

        # 处理T+1结算
        frozen_amount = self.china_settlement.process_t1_settlement([star_trade])
        self.assertGreater(frozen_amount, 0)

        # 检查持仓
        self.assertEqual(self.china_settlement.settled_positions.get("688981.SH", 0), 200)

    def test_after_hours_trading(self):
        """测试科创板盘后交易"""
        star_trade = Trade(
            symbol="688981.SH",
            price=50.0,
            quantity=200,
            side="BUY",
            prev_close=45.0,
            close_price=48.0
        )

        # 处理盘后交易
        frozen_amount = self.china_settlement.process_after_hours_trading([star_trade])
        self.assertGreater(frozen_amount, 0)

        # 检查持仓
        self.assertEqual(self.china_settlement.settled_positions.get("688981.SH", 0), 200)

    def test_margin_settlement(self):
        """测试融资融券结算"""
        positions = {
            "600519.SH": 100000,  # 10万元
            "000001.SZ": 50000    # 5万元
        }

        # 处理融资融券结算
        adjustments = self.china_settlement.process_margin_settlement(positions)
        self.assertEqual(len(adjustments), 0)  # 默认情况下应无调整

    def test_reconciliation(self):
        """测试对账功能"""
        # 处理T+1结算
        self.china_settlement.process_t1_settlement([self.test_trade])

        # 模拟券商数据
        broker_data = {
            "600519.SH": 100,
            "frozen_cash": 10000.0
        }

        # 对账
        discrepancies = self.china_settlement.reconcile_with_broker(broker_data)
        self.assertEqual(len(discrepancies), 0)  # 数据一致时应无差异

    def test_performance_metrics(self):
        """测试性能指标"""
        import time

        # 大批量测试
        large_batch_trades = [self.test_trade] * 1000

        start = time.time()
        self.settlement.process_t1_settlement(large_batch_trades)
        elapsed = time.time() - start

        # 检查性能
        self.assertLess(elapsed, 1.0)  # 1000次结算应在1秒内完成
        print(f"结算处理性能: {1000/elapsed:.2f} 次/秒")

if __name__ == '__main__':
    unittest.main()
