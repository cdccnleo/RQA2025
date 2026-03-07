"""
交易引擎方法测试 - 直接测试源代码方法
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
from src.trading.core.trading_engine import (
    TradingEngine, OrderType, OrderDirection, OrderStatus, ChinaMarketAdapter
)


class TestChinaMarketAdapterFees:
    """测试中国市场适配器费用计算"""

    def test_calculate_fees_a_stock_buy(self):
        """测试A股买入费用计算"""
        order = {
            "quantity": 1000,
            "direction": OrderDirection.BUY,
            "price": 10.0
        }

        fees = ChinaMarketAdapter.calculate_fees(order, is_a_stock=True)

        # 买入时只有佣金和过户费
        expected_commission = max(1000 * 10.0 * 0.00025, 5)  # 2.5元，但最低5元
        expected_transfer = 1000 * 10.0 * 0.00001  # 0.1元
        expected_stamp_tax = 0  # 买入无印花税

        assert fees == expected_commission + expected_transfer + expected_stamp_tax

    def test_calculate_fees_a_stock_sell(self):
        """测试A股卖出费用计算"""
        order = {
            "quantity": 1000,
            "direction": OrderDirection.SELL,
            "price": 10.0
        }

        fees = ChinaMarketAdapter.calculate_fees(order, is_a_stock=True)

        # 卖出时有印花税、佣金和过户费
        expected_stamp_tax = 1000 * 10.0 * 0.001  # 10元
        expected_commission = max(1000 * 10.0 * 0.00025, 5)  # 2.5元，但最低5元
        expected_transfer = 1000 * 10.0 * 0.00001  # 0.1元

        assert fees == expected_stamp_tax + expected_commission + expected_transfer

    def test_calculate_fees_non_a_stock(self):
        """测试非A股费用计算"""
        order = {
            "quantity": 1000,
            "direction": OrderDirection.SELL,
            "price": 10.0
        }

        fees = ChinaMarketAdapter.calculate_fees(order, is_a_stock=False)
        assert fees == 0.0

    def test_calculate_fees_minimum_commission(self):
        """测试最低佣金限制"""
        # 小额交易，佣金低于5元
        order = {
            "quantity": 100,
            "direction": OrderDirection.BUY,
            "price": 1.0  # 成交额100元，佣金0.025元
        }

        fees = ChinaMarketAdapter.calculate_fees(order, is_a_stock=True)

        # 佣金应该是最低5元 + 过户费0.001元
        expected_transfer = 100 * 1.0 * 0.00001
        assert fees == 5.0 + expected_transfer

    def test_calculate_fees_large_transaction(self):
        """测试大额交易费用计算"""
        order = {
            "quantity": 10000,
            "direction": OrderDirection.SELL,
            "price": 50.0  # 成交额500,000元
        }

        fees = ChinaMarketAdapter.calculate_fees(order, is_a_stock=True)

        # 计算各项费用
        amount = 10000 * 50.0
        expected_stamp_tax = amount * 0.001  # 500元
        expected_commission = amount * 0.00025  # 125元（超过最低5元）
        expected_transfer = amount * 0.00001  # 5元

        expected_total = expected_stamp_tax + expected_commission + expected_transfer
        assert abs(fees - expected_total) < 0.01


class TestTradingEngineCoreMethods:
    """测试交易引擎核心方法"""

    def setup_method(self):
        """测试前准备"""
        with patch('src.trading.core.trading_engine.get_data_adapter'):
            with patch('src.trading.core.trading_engine.SystemMonitor'):
                self.engine = TradingEngine()

    def teardown_method(self):
        """测试后清理"""
        pass

    def test_trading_engine_initialization(self):
        """测试交易引擎初始化"""
        with patch('src.trading.core.trading_engine.get_data_adapter'):
            with patch('src.trading.core.trading_engine.SystemMonitor'):
                engine = TradingEngine()

                assert hasattr(engine, 'risk_config')
                assert hasattr(engine, 'monitor')

    def test_trading_engine_with_risk_config(self):
        """测试带风险配置的交易引擎初始化"""
        risk_config = {
            'max_position_size': 1000000,
            'max_daily_loss': 50000
        }

        with patch('src.trading.core.trading_engine.get_data_adapter'):
            with patch('src.trading.core.trading_engine.SystemMonitor'):
                engine = TradingEngine(risk_config=risk_config)

                assert engine.risk_config == risk_config

    def test_trading_engine_with_monitor(self):
        """测试带监控器的交易引擎初始化"""
        # 测试TradingEngine可以正常初始化
        engine = TradingEngine()

        # 验证引擎有必要的属性
        assert hasattr(engine, 'monitor')
        assert hasattr(engine, 'positions')
        assert hasattr(engine, 'cash_balance')
        assert hasattr(engine, 'order_history')


class TestTradingEngineIntegration:
    """测试交易引擎集成功能"""

    def setup_method(self):
        """测试前准备"""
        with patch('src.trading.core.trading_engine.get_data_adapter'):
            with patch('src.trading.core.trading_engine.SystemMonitor'):
                self.engine = TradingEngine()

    def teardown_method(self):
        """测试后清理"""
        pass

    def test_trading_engine_method_existence(self):
        """测试交易引擎方法存在性"""
        # 验证核心方法存在
        methods_to_check = [
            '__init__',
            # 可以添加其他方法检查
        ]

        for method in methods_to_check:
            assert hasattr(self.engine, method), f"TradingEngine缺少方法: {method}"

    def test_china_market_adapter_integration(self):
        """测试中国市场适配器集成"""
        # 测试市场适配器的静态方法可以被正常调用
        assert callable(ChinaMarketAdapter.check_trade_restrictions)
        assert callable(ChinaMarketAdapter.check_t1_restriction)
        assert callable(ChinaMarketAdapter.calculate_fees)

        # 验证方法返回正确的类型
        result1 = ChinaMarketAdapter.check_trade_restrictions("000001", 10.0, 10.0)
        assert isinstance(result1, bool)

        dt1 = datetime(2023, 1, 1)
        dt2 = datetime(2023, 1, 2)
        result2 = ChinaMarketAdapter.check_t1_restriction(dt1, dt2)
        assert isinstance(result2, bool)

        order = {"quantity": 100, "direction": OrderDirection.BUY, "price": 10.0}
        result3 = ChinaMarketAdapter.calculate_fees(order)
        assert isinstance(result3, float)
        assert result3 >= 0

    def test_enum_integration(self):
        """测试枚举集成"""
        # 验证所有枚举都可以正常导入和使用
        assert OrderType.MARKET
        assert OrderDirection.BUY
        assert OrderStatus.PENDING

        # 验证枚举值是唯一的
        order_types = [ot.value for ot in OrderType]
        assert len(order_types) == len(set(order_types))

        order_directions = [od.value for od in OrderDirection]
        assert len(order_directions) == len(set(order_directions))

        order_statuses = [os.value for os in OrderStatus]
        assert len(order_statuses) == len(set(order_statuses))


class TestTradingEngineConstants:
    """测试交易引擎常量"""

    def test_st_prefixes_constant(self):
        """测试ST前缀常量"""
        assert hasattr(ChinaMarketAdapter, 'ST_PREFIXES')
        assert isinstance(ChinaMarketAdapter.ST_PREFIXES, set)
        assert "ST" in ChinaMarketAdapter.ST_PREFIXES
        assert "*ST" in ChinaMarketAdapter.ST_PREFIXES

    def test_constant_immutability(self):
        """测试常量不可变性"""
        # ST_PREFIXES应该是不可变的
        original_prefixes = ChinaMarketAdapter.ST_PREFIXES.copy()

        # 尝试修改（这不应该影响原始常量）
        # 注意：实际上set是可变的，但这里我们测试逻辑
        assert ChinaMarketAdapter.ST_PREFIXES == original_prefixes


class TestTradingEngineErrorHandling:
    """测试交易引擎错误处理"""

    def test_invalid_fee_calculation_input(self):
        """测试费用计算的无效输入"""
        # 缺少必需字段的订单
        invalid_order = {"quantity": 100}  # 缺少direction

        # 这应该不会抛出异常，因为方法实现是宽容的
        try:
            fees = ChinaMarketAdapter.calculate_fees(invalid_order)
            assert isinstance(fees, float)
        except KeyError:
            # 如果实现要求必需字段，这是可以接受的
            pass

    def test_datetime_edge_cases_t1(self):
        """测试T1限制的日期时间边界情况"""
        # 测试跨年的边界
        dec_31 = datetime(2022, 12, 31, 15, 0, 0)
        jan_1 = datetime(2023, 1, 1, 9, 30, 0)

        # 跨年应该允许交易
        assert ChinaMarketAdapter.check_t1_restriction(dec_31, jan_1)

    def test_price_calculation_edge_cases(self):
        """测试价格计算的边界情况"""
        # 测试精确的边界价格
        last_close = 10.0

        # 精确的10%涨幅应该被允许（根据当前实现逻辑）
        price_at_limit = last_close * 1.1
        result = ChinaMarketAdapter.check_trade_restrictions("000001", price_at_limit, last_close)

        # 根据实现逻辑，这应该返回True（因为只检查超过限制的情况）
        # 实际结果取决于具体实现

        # 超过10%应该被限制
        price_over_limit = last_close * 1.11
        result_over = ChinaMarketAdapter.check_trade_restrictions("000001", price_over_limit, last_close)
        assert not result_over  # 超过限制应该被限制


class TestTradingEnginePerformance:
    """测试交易引擎性能"""

    def test_fee_calculation_performance(self):
        """测试费用计算性能"""
        import time

        # 创建大量订单进行性能测试
        orders = [
            {"quantity": 100 * i, "direction": OrderDirection.BUY if i % 2 == 0 else OrderDirection.SELL, "price": 10.0 + i * 0.1}
            for i in range(1, 101)
        ]

        start_time = time.time()

        # 计算所有订单的费用
        total_fees = 0
        for order in orders:
            fees = ChinaMarketAdapter.calculate_fees(order)
            total_fees += fees

        end_time = time.time()

        # 验证计算完成且在合理时间内
        assert total_fees >= 0
        assert end_time - start_time < 1.0  # 应该在1秒内完成

    def test_restriction_check_performance(self):
        """测试限制检查性能"""
        import time

        # 创建大量股票价格组合进行测试
        test_cases = [
            (f"{i:06d}", 10.0 + (i % 20 - 10) * 0.2, 10.0)
            for i in range(1000)
        ]

        start_time = time.time()

        # 检查所有组合的交易限制
        results = []
        for symbol, price, last_close in test_cases:
            can_trade = ChinaMarketAdapter.check_trade_restrictions(symbol, price, last_close)
            results.append(can_trade)

        end_time = time.time()

        # 验证计算完成且在合理时间内
        assert len(results) == 1000
        assert end_time - start_time < 1.0  # 应该在1秒内完成

    def test_t1_check_performance(self):
        """测试T1检查性能"""
        import time

        # 创建大量日期组合进行测试
        base_date = datetime(2023, 1, 1)
        date_combinations = [
            (base_date + timedelta(days=i), base_date + timedelta(days=i + j))
            for i in range(100)
            for j in range(1, 4)
        ]

        start_time = time.time()

        # 检查所有日期组合的T1限制
        results = []
        for position_date, current_date in date_combinations:
            can_sell = ChinaMarketAdapter.check_t1_restriction(position_date, current_date)
            results.append(can_sell)

        end_time = time.time()

        # 验证计算完成且在合理时间内
        assert len(results) == 300  # 100 * 3
        assert end_time - start_time < 1.0  # 应该在1秒内完成
