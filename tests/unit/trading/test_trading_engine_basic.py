#!/usr/bin/env python3
"""
交易引擎基础测试用例

测试TradingEngine类的基本功能
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
from src.trading.trading_engine import TradingEngine, OrderDirection


class TestTradingEngineBasic:
    """交易引擎基础测试类"""

    @pytest.fixture
    def trading_engine(self):
        """交易引擎实例"""
        config = {
            "initial_capital": 1000000.0,
            "market_type": "A",
            "max_position_size": 0.1,
            "risk_per_trade": 0.02
        }
        engine = TradingEngine(risk_config=config)
        return engine

    @pytest.fixture
    def sample_prices(self):
        """样本价格数据"""
        return {
            '000001.SZ': 10.0,
            '000002.SZ': 15.0,
            '600000.SH': 20.0
        }

    def test_initialization(self, trading_engine):
        """测试初始化"""
        assert trading_engine.cash_balance == 1000000.0
        assert trading_engine.is_a_stock is True
        assert isinstance(trading_engine.positions, dict)
        assert isinstance(trading_engine.order_history, list)
        assert trading_engine._is_running is False

    def test_portfolio_value_calculation(self, trading_engine, sample_prices):
        """测试投资组合价值计算"""
        # 设置持仓
        trading_engine.positions = {
            '000001.SZ': 1000,
            '000002.SZ': 500
        }

        portfolio_value = trading_engine.get_portfolio_value(sample_prices)

        expected_value = (
            trading_engine.cash_balance +
            1000 * sample_prices['000001.SZ'] +
            500 * sample_prices['000002.SZ']
        )

        assert portfolio_value == expected_value

    def test_position_size_calculation(self, trading_engine, sample_prices):
        """测试仓位大小计算"""
        symbol = '000001.SZ'
        signal = 1
        strength = 0.8
        price = sample_prices[symbol]

        position_size = trading_engine._calculate_position_size(
            symbol, signal, strength, price
        )

        assert position_size > 0
        # 验证不超过最大仓位限制
        max_position_value = trading_engine.cash_balance * trading_engine.risk_config.get("max_position_size", 0.1)
        expected_max_quantity = max_position_value / price
        assert position_size <= expected_max_quantity

    def test_start_stop_engine(self, trading_engine):
        """测试引擎启动和停止"""
        # 测试启动
        trading_engine.start()
        assert trading_engine._is_running is True
        assert trading_engine.start_time is not None

        # 测试停止
        trading_engine.stop()
        assert trading_engine._is_running is False
        assert trading_engine.end_time is not None

    def test_risk_metrics(self, trading_engine):
        """测试风险指标"""
        metrics = trading_engine.get_risk_metrics()

        assert isinstance(metrics, dict)
        # 验证包含基本的风险指标
        assert 'total_exposure' in metrics or len(metrics) > 0

    def test_execution_stats(self, trading_engine):
        """测试执行统计"""
        stats = trading_engine.get_execution_stats()

        assert isinstance(stats, dict)
        # 验证包含基本的执行统计
        assert 'total_orders' in stats or len(stats) > 0
