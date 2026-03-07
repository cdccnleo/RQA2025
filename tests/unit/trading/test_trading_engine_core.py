"""
交易引擎核心功能测试 - 直接测试源代码
"""

import pytest
from datetime import datetime, timedelta
from src.trading.core.trading_engine import (
    OrderType, OrderDirection, OrderStatus, ChinaMarketAdapter
)


class TestOrderEnums:
    """测试订单枚举"""

    def test_order_type_enum_values(self):
        """测试订单类型枚举值"""
        assert OrderType.MARKET.value == 1
        assert OrderType.LIMIT.value == 2
        assert OrderType.STOP.value == 3

    def test_order_direction_enum_values(self):
        """测试订单方向枚举值"""
        assert OrderDirection.BUY.value == 1
        assert OrderDirection.SELL.value == -1

    def test_order_status_enum_values(self):
        """测试订单状态枚举值"""
        assert OrderStatus.PENDING.value == 1
        assert OrderStatus.PARTIAL.value == 2
        assert OrderStatus.FILLED.value == 3
        assert OrderStatus.CANCELLED.value == 4
        assert OrderStatus.REJECTED.value == 5

    def test_enum_uniqueness(self):
        """测试枚举值唯一性"""
        order_type_values = [ot.value for ot in OrderType]
        assert len(order_type_values) == len(set(order_type_values))

        order_direction_values = [od.value for od in OrderDirection]
        assert len(order_direction_values) == len(set(order_direction_values))

        order_status_values = [os.value for os in OrderStatus]
        assert len(order_status_values) == len(set(order_status_values))


class TestChinaMarketAdapter:
    """测试中国市场适配器"""

    def test_st_stock_restriction(self):
        """测试ST股票交易限制"""
        # ST股票应该被限制
        assert not ChinaMarketAdapter.check_trade_restrictions("ST000001", 10.0, 10.0)
        assert not ChinaMarketAdapter.check_trade_restrictions("*ST000002", 10.0, 10.0)

        # 普通股票应该可以交易
        assert ChinaMarketAdapter.check_trade_restrictions("000001", 10.0, 10.0)

    def test_price_limit_restriction(self):
        """测试涨跌停价格限制"""
        last_close = 10.0

        # 测试涨停（超过10%涨幅）
        high_price = 11.1  # 超过10%涨幅
        assert not ChinaMarketAdapter.check_trade_restrictions("000001", high_price, last_close)

        # 测试跌停（超过10%跌幅）
        low_price = 8.9   # 超过10%跌幅
        assert not ChinaMarketAdapter.check_trade_restrictions("000001", low_price, last_close)

        # 测试正常价格范围
        normal_high = 10.9  # 9%涨幅，正常
        normal_low = 9.1    # 9%跌幅，正常
        assert ChinaMarketAdapter.check_trade_restrictions("000001", normal_high, last_close)
        assert ChinaMarketAdapter.check_trade_restrictions("000001", normal_low, last_close)

    def test_st_with_price_limit(self):
        """测试ST股票同时满足价格限制的情况"""
        # ST股票即使价格正常也应该被限制
        assert not ChinaMarketAdapter.check_trade_restrictions("ST000001", 10.0, 10.0)
        assert not ChinaMarketAdapter.check_trade_restrictions("*ST000002", 10.5, 10.0)

    def test_edge_cases_price_limits(self):
        """测试价格限制的边界情况"""
        last_close = 10.0

        # 超过涨停价格应该限制（>10%不允许）
        over_limit_high = 11.1  # 11%涨幅
        over_limit_low = 8.9    # 11%跌幅

        # 这些应该被限制
        assert not ChinaMarketAdapter.check_trade_restrictions("000001", over_limit_high, last_close)
        assert not ChinaMarketAdapter.check_trade_restrictions("000001", over_limit_low, last_close)

    def test_t1_restriction_same_day(self):
        """测试T+1限制 - 同一天"""
        position_date = datetime(2023, 1, 1, 10, 0, 0)
        current_date = datetime(2023, 1, 1, 15, 0, 0)  # 同一天

        # 同一天应该限制交易（T+1）
        assert not ChinaMarketAdapter.check_t1_restriction(position_date, current_date)

    def test_t1_restriction_next_day(self):
        """测试T+1限制 - 下一天"""
        position_date = datetime(2023, 1, 1, 15, 0, 0)
        current_date = datetime(2023, 1, 2, 9, 30, 0)  # 下一天早上

        # 下一天应该允许交易
        assert ChinaMarketAdapter.check_t1_restriction(position_date, current_date)

    def test_t1_restriction_two_days_later(self):
        """测试T+1限制 - 两天后"""
        position_date = datetime(2023, 1, 1, 15, 0, 0)
        current_date = datetime(2023, 1, 3, 9, 30, 0)  # 两天后

        # T+2应该允许交易
        assert ChinaMarketAdapter.check_t1_restriction(position_date, current_date)

    def test_t1_restriction_weekend_boundary(self):
        """测试T+1限制 - 周末边界情况"""
        # 周五买入，周一卖出
        friday_position = datetime(2023, 1, 6, 15, 0, 0)  # 周五
        monday_current = datetime(2023, 1, 9, 9, 30, 0)   # 周一

        # 周一应该允许交易（虽然是T+3，但跳过了周末）
        assert ChinaMarketAdapter.check_t1_restriction(friday_position, monday_current)

    def test_t1_restriction_time_precision(self):
        """测试T+1限制的时间精度"""
        # 测试精确到秒的比较
        position_time = datetime(2023, 1, 1, 14, 59, 59)
        same_day_later = datetime(2023, 1, 1, 15, 0, 0)

        # 同一天应该限制
        assert not ChinaMarketAdapter.check_t1_restriction(position_time, same_day_later)

        next_day = datetime(2023, 1, 2, 9, 30, 0)
        # 下一天应该允许
        assert ChinaMarketAdapter.check_t1_restriction(position_time, next_day)


class TestTradingEngineCoreLogic:
    """测试交易引擎核心逻辑"""

    def test_enum_combinations(self):
        """测试枚举组合的有效性"""
        # 测试所有订单类型和方向的组合都是有效的
        for order_type in OrderType:
            for direction in OrderDirection:
                for status in OrderStatus:
                    # 这些组合在逻辑上都是有效的
                    assert order_type is not None
                    assert direction is not None
                    assert status is not None

    def test_market_adapter_static_methods(self):
        """测试市场适配器的静态方法"""
        # 验证方法是静态的（可以不实例化调用）
        assert hasattr(ChinaMarketAdapter, 'check_trade_restrictions')
        assert hasattr(ChinaMarketAdapter, 'check_t1_restriction')

        # 验证可以直接调用
        result1 = ChinaMarketAdapter.check_trade_restrictions("000001", 10.0, 10.0)
        assert isinstance(result1, bool)

        dt1 = datetime(2023, 1, 1)
        dt2 = datetime(2023, 1, 2)
        result2 = ChinaMarketAdapter.check_t1_restriction(dt1, dt2)
        assert isinstance(result2, bool)

    def test_market_restriction_comprehensive_scenarios(self):
        """测试市场限制的综合场景"""
        test_cases = [
            # (symbol, price, last_close, expected_can_trade)
            ("000001", 10.0, 10.0, True),     # 正常股票，价格不变
            ("000002", 10.5, 10.0, True),     # 正常股票，小幅上涨
            ("000003", 9.5, 10.0, True),      # 正常股票，小幅下跌
            ("ST000004", 10.0, 10.0, False),  # ST股票
            ("*ST000005", 10.0, 10.0, False), # *ST股票
            ("000006", 11.1, 10.0, False),    # 涨停（超过10%）
            ("000007", 8.9, 10.0, False),     # 跌停（超过10%）
            ("600000", 10.5, 10.0, True),     # 上海股票，正常价格
            ("000001", 10.99, 10.0, True),    # 接近但未到涨停
            ("000001", 9.01, 10.0, True),     # 接近但未到跌停
        ]

        for symbol, price, last_close, expected in test_cases:
            result = ChinaMarketAdapter.check_trade_restrictions(symbol, price, last_close)
            assert result == expected, f"Failed for {symbol}: price={price}, last_close={last_close}"

    def test_t1_restriction_comprehensive_scenarios(self):
        """测试T+1限制的综合场景"""
        base_date = datetime(2023, 1, 2, 15, 0, 0)  # 周一收盘时间

        test_cases = [
            # (position_date, current_date, expected_can_trade)
            (base_date, base_date, False),                                   # 同一天 (T+1限制)
            (base_date, base_date + timedelta(hours=1), False),              # 当天稍晚 (T+1限制)
            (base_date, base_date + timedelta(days=1, hours=1), True),       # 次日（可以交易）
            (base_date, base_date + timedelta(days=2), True),                # T+2
            (base_date, base_date + timedelta(days=3), True),                # T+3
            # 周末测试
            (datetime(2023, 1, 6, 15, 0, 0), datetime(2023, 1, 9, 9, 30, 0), True),  # 周五到周一
            # 月末边界
            (datetime(2023, 1, 31, 15, 0, 0), datetime(2023, 2, 1, 9, 30, 0), True), # 月末到月初 (可以交易)
            (datetime(2023, 1, 31, 15, 0, 0), datetime(2023, 2, 2, 9, 30, 0), True),  # 月末到月初T+2
        ]

        for position_date, current_date, expected in test_cases:
            result = ChinaMarketAdapter.check_t1_restriction(position_date, current_date)
            assert result == expected, f"Failed for position={position_date}, current={current_date}"

    def test_trading_restrictions_interaction(self):
        """测试交易限制的交互效应"""
        # ST股票即使价格正常也不可交易
        st_cases = [
            ("ST000001", 10.0, 10.0, False),
            ("ST000002", 11.0, 10.0, False),  # ST + 涨停都不行
            ("ST000003", 9.0, 10.0, False),   # ST + 跌停都不行
        ]

        for symbol, price, last_close, expected in st_cases:
            result = ChinaMarketAdapter.check_trade_restrictions(symbol, price, last_close)
            assert result == expected, f"ST restriction failed for {symbol}"

    def test_enum_string_representations(self):
        """测试枚举的字符串表示"""
        # 测试枚举的字符串表示是否合理
        assert str(OrderType.MARKET) == "OrderType.MARKET"
        assert str(OrderDirection.BUY) == "OrderDirection.BUY"
        assert str(OrderStatus.PENDING) == "OrderStatus.PENDING"

        # 测试枚举值的字符串表示
        assert OrderType.MARKET.name == "MARKET"
        assert OrderDirection.BUY.name == "BUY"
        assert OrderStatus.FILLED.name == "FILLED"

    def test_market_adapter_constants(self):
        """测试市场适配器常量"""
        # 验证ST前缀常量
        assert hasattr(ChinaMarketAdapter, 'ST_PREFIXES')
        assert isinstance(ChinaMarketAdapter.ST_PREFIXES, set)
        assert "ST" in ChinaMarketAdapter.ST_PREFIXES
        assert "*ST" in ChinaMarketAdapter.ST_PREFIXES

        # 验证常量内容
        expected_prefixes = {"ST", "*ST"}
        assert ChinaMarketAdapter.ST_PREFIXES == expected_prefixes


class TestTradingEngineEdgeCases:
    """测试交易引擎边界情况"""

    def test_extreme_price_values(self):
        """测试极端价格值"""
        # 测试非常高的价格
        assert not ChinaMarketAdapter.check_trade_restrictions("000001", 1000.0, 10.0)
        assert not ChinaMarketAdapter.check_trade_restrictions("000001", 0.01, 10.0)

        # 测试零价格和负价格
        assert not ChinaMarketAdapter.check_trade_restrictions("000001", 0.0, 0.0)  # 零对零 (不允许)
        assert not ChinaMarketAdapter.check_trade_restrictions("000001", -1.0, 10.0)  # 负价格 (不允许)

    def test_datetime_edge_cases(self):
        """测试日期时间边界情况"""
        # 测试跨年的情况
        year_end = datetime(2022, 12, 31, 15, 0, 0)
        new_year = datetime(2023, 1, 1, 9, 30, 0)

        assert ChinaMarketAdapter.check_t1_restriction(year_end, new_year)  # 应该可以交易

        # 测试闰年情况
        leap_year_before = datetime(2020, 2, 28, 15, 0, 0)
        leap_year_after = datetime(2020, 2, 29, 9, 30, 0)

        assert ChinaMarketAdapter.check_t1_restriction(leap_year_before, leap_year_after)

    def test_symbol_format_variations(self):
        """测试股票代码格式变体"""
        # 测试不同格式的股票代码
        valid_symbols = ["000001", "600000", "000002", "300001"]
        st_symbols = ["ST000001", "*ST000002", "ST600000", "*ST300001"]

        for symbol in valid_symbols:
            # 正常股票应该可以通过基本检查
            basic_check = not any(symbol.startswith(prefix) for prefix in ChinaMarketAdapter.ST_PREFIXES)
            assert basic_check, f"Valid symbol {symbol} incorrectly identified as ST"

        for symbol in st_symbols:
            # ST股票应该被识别
            is_st = any(symbol.startswith(prefix) for prefix in ChinaMarketAdapter.ST_PREFIXES)
            assert is_st, f"ST symbol {symbol} not correctly identified"

    def test_enum_iteration_and_membership(self):
        """测试枚举迭代和成员关系"""
        # 测试可以迭代所有枚举值
        order_types = list(OrderType)
        assert len(order_types) == 3
        assert OrderType.MARKET in order_types
        assert OrderType.LIMIT in order_types
        assert OrderType.STOP in order_types

        order_directions = list(OrderDirection)
        assert len(order_directions) == 2
        assert OrderDirection.BUY in order_directions
        assert OrderDirection.SELL in order_directions

        order_statuses = list(OrderStatus)
        assert len(order_statuses) == 5
        assert OrderStatus.PENDING in order_statuses
        assert OrderStatus.FILLED in order_statuses
        assert OrderStatus.REJECTED in order_statuses
