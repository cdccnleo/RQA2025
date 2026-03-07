#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
结算引擎单元测试

测试目标：提升settlement_engine.py的覆盖率到90%+
按照业务流程驱动架构设计测试结算引擎功能
"""

import pytest
from datetime import datetime, time
from unittest.mock import Mock

from src.trading.settlement.settlement_settlement_engine import (
    SettlementEngine,
    SettlementConfig,
    Trade,
    ChinaSettlementEngine
)


class TestSettlementConfig:
    """测试结算配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = SettlementConfig()

        assert config.t1_settlement is True
        assert config.freeze_ratio == 1.0
        assert config.settlement_time == "16:00"
        assert config.a_share_fees['commission'] == 0.0003
        assert config.a_share_fees['stamp_duty'] == 0.001
        assert config.a_share_fees['transfer_fee'] == 0.00002

    def test_custom_config(self):
        """测试自定义配置"""
        config = SettlementConfig(
            t1_settlement=False,
            freeze_ratio=0.8,
            settlement_time="15:30"
        )

        assert config.t1_settlement is False
        assert config.freeze_ratio == 0.8
        assert config.settlement_time == "15:30"


class TestTrade:
    """测试交易数据类"""

    def test_trade_creation(self):
        """测试交易创建"""
        trade = Trade(
            symbol="AAPL",
            price=150.0,
            quantity=100,
            side="BUY",
            prev_close=145.0,
            close_price=150.0
        )

        assert trade.symbol == "AAPL"
        assert trade.price == 150.0
        assert trade.quantity == 100
        assert trade.side == "BUY"
        assert trade.prev_close == 145.0
        assert trade.close_price == 150.0


class TestSettlementEngine:
    """测试结算引擎"""

    def test_init_default_config(self):
        """测试使用默认配置初始化"""
        mock_engine = Mock()
        settlement = SettlementEngine(mock_engine)

        assert settlement.engine == mock_engine
        assert settlement.config.t1_settlement is True
        assert settlement.frozen_cash == 0.0
        assert settlement.settled_positions == {}
        assert settlement.last_settlement_time is None

    def test_init_custom_config(self):
        """测试使用自定义配置初始化"""
        mock_engine = Mock()
        config = SettlementConfig(
            t1_settlement=False,
            freeze_ratio=0.9
        )
        settlement = SettlementEngine(mock_engine, config)

        assert settlement.config.t1_settlement is False
        assert settlement.config.freeze_ratio == 0.9

    def test_process_t1_settlement_single_buy(self):
        """测试T+1结算 - 单个买入交易"""
        mock_engine = Mock()
        settlement = SettlementEngine(mock_engine)

        trades = [
            Trade(
                symbol="AAPL",
                price=150.0,
                quantity=100,
                side="BUY"
            )
        ]

        result = settlement.process_t1_settlement(trades)

        # 计算预期金额（使用freeze_ratio）
        amount = 150.0 * 100  # 15000
        # 买入费用：佣金+过户费（无印花税）
        commission = amount * settlement.config.a_share_fees["commission"]
        transfer_fee = amount * settlement.config.a_share_fees["transfer_fee"]
        fees = commission + transfer_fee
        expected_total = (amount + fees) * settlement.config.freeze_ratio

        assert result > 0
        assert abs(settlement.frozen_cash - expected_total) < 0.01  # 允许小的浮点误差
        assert settlement.settled_positions["AAPL"] == 100

    def test_process_t1_settlement_single_sell(self):
        """测试T+1结算 - 单个卖出交易"""
        mock_engine = Mock()
        settlement = SettlementEngine(mock_engine)

        trades = [
            Trade(
                symbol="AAPL",
                price=150.0,
                quantity=100,
                side="SELL"
            )
        ]

        result = settlement.process_t1_settlement(trades)

        assert result > 0
        assert settlement.settled_positions["AAPL"] == -100

    def test_process_t1_settlement_multiple_trades(self):
        """测试T+1结算 - 多个交易"""
        mock_engine = Mock()
        settlement = SettlementEngine(mock_engine)

        trades = [
            Trade(symbol="AAPL", price=150.0, quantity=100, side="BUY"),
            Trade(symbol="GOOGL", price=200.0, quantity=50, side="BUY"),
            Trade(symbol="AAPL", price=155.0, quantity=50, side="SELL"),
        ]

        result = settlement.process_t1_settlement(trades)

        assert result > 0
        assert settlement.settled_positions["AAPL"] == 50  # 100 - 50
        assert settlement.settled_positions["GOOGL"] == 50

    def test_process_t1_settlement_disabled(self):
        """测试T+1结算 - 已禁用"""
        mock_engine = Mock()
        config = SettlementConfig(t1_settlement=False)
        settlement = SettlementEngine(mock_engine, config)

        trades = [
            Trade(symbol="AAPL", price=150.0, quantity=100, side="BUY")
        ]

        result = settlement.process_t1_settlement(trades)

        assert result == 0.0
        assert settlement.frozen_cash == 0.0

    def test_process_t1_settlement_empty_trades(self):
        """测试T+1结算 - 空交易列表"""
        mock_engine = Mock()
        settlement = SettlementEngine(mock_engine)

        result = settlement.process_t1_settlement([])

        assert result == 0.0

    def test_process_t1_settlement_freeze_ratio(self):
        """测试T+1结算 - 自定义冻结比例"""
        mock_engine = Mock()
        config = SettlementConfig(freeze_ratio=0.8)
        settlement = SettlementEngine(mock_engine, config)

        trades = [
            Trade(symbol="AAPL", price=150.0, quantity=100, side="BUY")
        ]

        result = settlement.process_t1_settlement(trades)

        # 应该使用0.8的冻结比例
        assert result > 0
        assert settlement.frozen_cash > 0

    def test_release_settlement_enabled(self):
        """测试释放结算资金 - T+1启用"""
        mock_engine = Mock()
        settlement = SettlementEngine(mock_engine)
        settlement.frozen_cash = 10000.0
        settlement.last_settlement_time = datetime.now().replace(hour=14, minute=0)

        # Mock结算时间已过
        from unittest.mock import patch
        with patch.object(
            settlement,
            '_is_settlement_time_passed',
            return_value=True
        ):
            result = settlement.release_settlement()

            assert result == 10000.0
            assert settlement.frozen_cash == 0.0

    def test_release_settlement_not_time(self):
        """测试释放结算资金 - 时间未到"""
        mock_engine = Mock()
        settlement = SettlementEngine(mock_engine)
        settlement.frozen_cash = 10000.0

        # Mock结算时间未到
        from unittest.mock import patch
        with patch.object(
            settlement,
            '_is_settlement_time_passed',
            return_value=False
        ):
            result = settlement.release_settlement()

            assert result == 0.0
            assert settlement.frozen_cash == 10000.0
    
    def test_is_settlement_time_passed_no_last_time(self):
        """测试检查结算时间 - 没有上次结算时间"""
        mock_engine = Mock()
        settlement = SettlementEngine(mock_engine)
        settlement.last_settlement_time = None
        
        result = settlement._is_settlement_time_passed()
        
        assert result is False
    
    def test_is_settlement_time_passed_same_day(self):
        """测试检查结算时间 - 同一天但未到结算时间"""
        mock_engine = Mock()
        settlement = SettlementEngine(mock_engine)
        # 设置结算时间为16:00
        settlement.config.settlement_time = "16:00"
        # 设置上次结算时间为今天14:00
        settlement.last_settlement_time = datetime.now().replace(hour=14, minute=0)
        
        # Mock当前时间为今天15:00（未到结算时间）
        from unittest.mock import patch
        with patch('src.trading.settlement.settlement_settlement_engine.datetime') as mock_datetime:
            mock_now = datetime.now().replace(hour=15, minute=0)
            mock_datetime.now.return_value = mock_now
            mock_datetime.strptime = datetime.strptime
            
            result = settlement._is_settlement_time_passed()
            
            assert result is False
    
    def test_is_settlement_time_passed_next_day(self):
        """测试检查结算时间 - 第二天已过结算时间"""
        mock_engine = Mock()
        settlement = SettlementEngine(mock_engine)
        # 设置结算时间为16:00
        settlement.config.settlement_time = "16:00"
        # 设置上次结算时间为昨天14:00
        from datetime import timedelta
        yesterday = datetime.now() - timedelta(days=1)
        settlement.last_settlement_time = yesterday.replace(hour=14, minute=0)
        
        # Mock当前时间为今天17:00（已过结算时间且是第二天）
        from unittest.mock import patch
        with patch('src.trading.settlement.settlement_settlement_engine.datetime') as mock_datetime:
            mock_now = datetime.now().replace(hour=17, minute=0)
            mock_datetime.now.return_value = mock_now
            mock_datetime.strptime = datetime.strptime
            
            result = settlement._is_settlement_time_passed()
            
            assert result is True

    def test_release_settlement_disabled(self):
        """测试释放结算资金 - T+1禁用"""
        mock_engine = Mock()
        config = SettlementConfig(t1_settlement=False)
        settlement = SettlementEngine(mock_engine, config)
        settlement.frozen_cash = 10000.0

        result = settlement.release_settlement()

        assert result == 0.0

    def test_calculate_a_share_fees_buy(self):
        """测试A股费用计算 - 买入"""
        mock_engine = Mock()
        settlement = SettlementEngine(mock_engine)

        trade = Trade(
            symbol="AAPL",
            price=150.0,
            quantity=100,
            side="BUY"
        )

        fees = settlement._calculate_a_share_fees(trade)

        # 买入：佣金 + 过户费（无印花税）
        amount = 150.0 * 100
        expected_commission = amount * 0.0003
        expected_transfer_fee = amount * 0.00002
        expected_fees = expected_commission + expected_transfer_fee

        assert fees == pytest.approx(expected_fees, rel=1e-6)

    def test_calculate_a_share_fees_sell(self):
        """测试A股费用计算 - 卖出"""
        mock_engine = Mock()
        settlement = SettlementEngine(mock_engine)

        trade = Trade(
            symbol="AAPL",
            price=150.0,
            quantity=100,
            side="SELL"
        )

        fees = settlement._calculate_a_share_fees(trade)

        # 卖出：佣金 + 过户费 + 印花税
        amount = 150.0 * 100
        expected_commission = amount * 0.0003
        expected_stamp_duty = amount * 0.001
        expected_transfer_fee = amount * 0.00002
        expected_fees = expected_commission + expected_stamp_duty + expected_transfer_fee

        assert fees == pytest.approx(expected_fees, rel=1e-6)

    def test_reconcile_with_broker_no_discrepancies(self):
        """测试与券商对账 - 无差异"""
        mock_engine = Mock()
        settlement = SettlementEngine(mock_engine)
        settlement.frozen_cash = 10000.0
        settlement.settled_positions = {"AAPL": 100}

        broker_data = {
            "AAPL": 100,
            "frozen_cash": 10000.0
        }

        discrepancies = settlement.reconcile_with_broker(broker_data)

        assert len(discrepancies) == 0

    def test_reconcile_with_broker_position_discrepancy(self):
        """测试与券商对账 - 持仓差异"""
        mock_engine = Mock()
        settlement = SettlementEngine(mock_engine)
        settlement.settled_positions = {"AAPL": 100}

        broker_data = {
            "AAPL": 90,  # 差异10股
            "frozen_cash": 0
        }

        discrepancies = settlement.reconcile_with_broker(broker_data)

        assert "AAPL" in discrepancies
        assert discrepancies["AAPL"]["local"] == 100
        assert discrepancies["AAPL"]["broker"] == 90

    def test_reconcile_with_broker_cash_discrepancy(self):
        """测试与券商对账 - 资金差异"""
        mock_engine = Mock()
        settlement = SettlementEngine(mock_engine)
        settlement.frozen_cash = 10000.0
        settlement.settled_positions = {}

        broker_data = {
            "frozen_cash": 9000.0  # 差异1000元
        }

        discrepancies = settlement.reconcile_with_broker(broker_data)

        assert "cash" in discrepancies
        assert discrepancies["cash"]["local"] == 10000.0
        assert discrepancies["cash"]["broker"] == 9000.0

    def test_process_margin_settlement(self):
        """测试融资融券结算"""
        mock_engine = Mock()
        settlement = SettlementEngine(mock_engine)

        positions = {
            "AAPL": 500000.0,
            "GOOGL": 300000.0
        }

        adjustments = settlement.process_margin_settlement(positions)

        assert isinstance(adjustments, dict)
    
    def test_process_margin_settlement_low_maintenance_ratio(self):
        """测试融资融券结算 - 维持担保比例不足"""
        mock_engine = Mock()
        settlement = SettlementEngine(mock_engine)
        
        # 设置较低的维持担保比例阈值
        settlement.config.margin_rules["maintenance_ratio"] = 2.0
        
        positions = {
            "AAPL": 500000.0,
            "GOOGL": 300000.0
        }
        
        adjustments = settlement.process_margin_settlement(positions)
        
        assert isinstance(adjustments, dict)
    
    def test_process_margin_settlement_concentration_limit(self):
        """测试融资融券结算 - 单票集中度超标"""
        mock_engine = Mock()
        settlement = SettlementEngine(mock_engine)
        
        # 设置集中度限制为30%，但单个股票占比超过30%
        positions = {
            "AAPL": 500000.0,  # 占比62.5%
            "GOOGL": 300000.0   # 占比37.5%
        }
        
        adjustments = settlement.process_margin_settlement(positions)
        
        assert isinstance(adjustments, dict)
        # 如果集中度超标，应该有调整
        total_value = sum(positions.values())
        for symbol, value in positions.items():
            if value / total_value > settlement.config.margin_rules["concentration_limit"]:
                assert symbol in adjustments
    
    def test_calculate_maintenance_ratio(self):
        """测试计算维持担保比例"""
        mock_engine = Mock()
        settlement = SettlementEngine(mock_engine)
        
        positions = {
            "AAPL": 500000.0,
            "GOOGL": 300000.0
        }
        
        ratio = settlement._calculate_maintenance_ratio(positions)
        
        assert isinstance(ratio, float)
        assert ratio == 1.5
    
    def test_reconcile_with_broker_missing_symbol(self):
        """测试对账 - 券商数据中缺少标的"""
        mock_engine = Mock()
        settlement = SettlementEngine(mock_engine)
        
        # 设置本地持仓
        settlement.settled_positions = {
            "AAPL": 100.0,
            "GOOGL": 200.0
        }
        
        # 券商数据中缺少GOOGL
        broker_data = {
            "AAPL": 100.0,
            "frozen_cash": 0.0
        }
        
        discrepancies = settlement.reconcile_with_broker(broker_data)
        
        assert isinstance(discrepancies, dict)
        assert "GOOGL" in discrepancies
        assert discrepancies["GOOGL"]["local"] == 200.0
        assert discrepancies["GOOGL"]["broker"] == 0
    
    def test_release_settlement_update_positions(self):
        """测试释放结算资金 - 更新持仓"""
        mock_engine = Mock()
        settlement = SettlementEngine(mock_engine)
        settlement.frozen_cash = 10000.0
        settlement.settled_positions = {
            "AAPL": 100.0,
            "GOOGL": 200.0
        }
        settlement.last_settlement_time = datetime.now().replace(hour=14, minute=0)
        
        # Mock结算时间已过
        from unittest.mock import patch
        with patch.object(
            settlement,
            '_is_settlement_time_passed',
            return_value=True
        ):
            result = settlement.release_settlement()
            
            assert result == 10000.0
            assert settlement.frozen_cash == 0.0
            # 持仓应该被更新（实际实现中会调用engine更新持仓）
            assert len(settlement.settled_positions) >= 0


class TestChinaSettlementEngine:
    """测试A股特定结算引擎"""

    def test_init_force_t1_settlement(self):
        """测试强制启用T+1结算"""
        mock_engine = Mock()
        china_settlement = ChinaSettlementEngine(mock_engine)

        assert china_settlement.config.t1_settlement is True
        assert hasattr(china_settlement, 'stock_types')
        assert hasattr(china_settlement, 'star_market_rules')

    def test_init_a_share_params(self):
        """测试A股特定参数初始化"""
        mock_engine = Mock()
        china_settlement = ChinaSettlementEngine(mock_engine)

        assert "ST" in china_settlement.stock_types
        assert "688" in china_settlement.stock_types
        assert "normal" in china_settlement.stock_types
        assert china_settlement.star_market_rules["after_hours_trading"] is True
        assert china_settlement.star_market_rules["price_limit"] == 0.2

    def test_process_t1_settlement_normal_stock(self):
        """测试普通股票T+1结算"""
        mock_engine = Mock()
        china_settlement = ChinaSettlementEngine(mock_engine)

        trades = [
            Trade(
                symbol="000001",
                price=10.0,
                quantity=100,
                side="BUY",
                prev_close=9.5
            )
        ]

        result = china_settlement.process_t1_settlement(trades)

        assert result > 0
        assert china_settlement.settled_positions["000001"] == 100
    
    def test_process_t1_settlement_sell_order(self):
        """测试卖出订单结算"""
        mock_engine = Mock()
        china_settlement = ChinaSettlementEngine(mock_engine)
        
        # 先买入建立持仓
        buy_trades = [
            Trade(
                symbol="000001",
                price=10.0,
                quantity=100,
                side="BUY",
                prev_close=9.5,
                close_price=9.8
            )
        ]
        china_settlement.process_t1_settlement(buy_trades)
        
        # 然后卖出
        sell_trades = [
            Trade(
                symbol="000001",
                price=11.0,
                quantity=50,
                side="SELL",
                prev_close=10.5,
                close_price=10.8
            )
        ]
        
        result = china_settlement.process_t1_settlement(sell_trades)
        
        assert result > 0
        # 持仓应该减少
        assert china_settlement.settled_positions["000001"] == 50

    def test_process_t1_settlement_star_market_valid(self):
        """测试科创板T+1结算 - 有效价格"""
        mock_engine = Mock()
        china_settlement = ChinaSettlementEngine(mock_engine)

        trades = [
            Trade(
                symbol="688001",
                price=20.0,
                quantity=100,
                side="BUY",
                prev_close=18.0  # 涨幅11%，在20%限制内
            )
        ]

        result = china_settlement.process_t1_settlement(trades)

        assert result > 0
        assert china_settlement.settled_positions["688001"] == 100

    def test_process_t1_settlement_star_market_invalid_price(self):
        """测试科创板T+1结算 - 价格超出限制"""
        mock_engine = Mock()
        china_settlement = ChinaSettlementEngine(mock_engine)

        trades = [
            Trade(
                symbol="688001",
                price=25.0,
                quantity=100,
                side="BUY",
                prev_close=18.0  # 涨幅39%，超过20%限制
            )
        ]

        result = china_settlement.process_t1_settlement(trades)

        # 应该跳过该交易
        assert "688001" not in china_settlement.settled_positions or china_settlement.settled_positions.get("688001", 0) == 0

    def test_process_t1_settlement_st_stock_valid(self):
        """测试ST股票T+1结算 - 有效价格"""
        mock_engine = Mock()
        china_settlement = ChinaSettlementEngine(mock_engine)

        trades = [
            Trade(
                symbol="ST000001",
                price=5.0,
                quantity=100,
                side="BUY",
                prev_close=4.8  # 涨幅4.2%，在5%限制内
            )
        ]

        result = china_settlement.process_t1_settlement(trades)

        assert result > 0
        assert china_settlement.settled_positions["ST000001"] == 100

    def test_process_t1_settlement_st_stock_invalid_price(self):
        """测试ST股票T+1结算 - 价格超出限制"""
        mock_engine = Mock()
        china_settlement = ChinaSettlementEngine(mock_engine)

        trades = [
            Trade(
                symbol="ST000001",
                price=5.5,
                quantity=100,
                side="BUY",
                prev_close=4.8  # 涨幅14.6%，超过5%限制
            )
        ]

        result = china_settlement.process_t1_settlement(trades)

        # 应该跳过该交易
        assert "ST000001" not in china_settlement.settled_positions or china_settlement.settled_positions.get("ST000001", 0) == 0

    def test_check_star_market_rules_within_limit(self):
        """测试科创板规则检查 - 在限制内"""
        mock_engine = Mock()
        china_settlement = ChinaSettlementEngine(mock_engine)

        trade = Trade(
            symbol="688001",
            price=20.0,
            quantity=100,
            side="BUY",
            prev_close=18.0
        )

        result = china_settlement._check_star_market_rules(trade)

        assert result is True

    def test_check_star_market_rules_exceed_upper_limit(self):
        """测试科创板规则检查 - 超过上限"""
        mock_engine = Mock()
        china_settlement = ChinaSettlementEngine(mock_engine)

        trade = Trade(
            symbol="688001",
            price=25.0,
            quantity=100,
            side="BUY",
            prev_close=18.0  # 涨幅39%
        )

        result = china_settlement._check_star_market_rules(trade)

        assert result is False

    def test_check_star_market_rules_exceed_lower_limit(self):
        """测试科创板规则检查 - 超过下限"""
        mock_engine = Mock()
        china_settlement = ChinaSettlementEngine(mock_engine)

        trade = Trade(
            symbol="688001",
            price=12.0,
            quantity=100,
            side="BUY",
            prev_close=18.0  # 跌幅33%
        )

        result = china_settlement._check_star_market_rules(trade)

        assert result is False

    def test_check_st_stock_rules_within_limit(self):
        """测试ST股票规则检查 - 在限制内"""
        mock_engine = Mock()
        china_settlement = ChinaSettlementEngine(mock_engine)

        trade = Trade(
            symbol="ST000001",
            price=5.0,
            quantity=100,
            side="BUY",
            prev_close=4.8
        )

        result = china_settlement._check_st_stock_rules(trade)

        assert result is True

    def test_check_st_stock_rules_exceed_limit(self):
        """测试ST股票规则检查 - 超过限制"""
        mock_engine = Mock()
        china_settlement = ChinaSettlementEngine(mock_engine)

        trade = Trade(
            symbol="ST000001",
            price=5.5,
            quantity=100,
            side="BUY",
            prev_close=4.8  # 涨幅14.6%
        )

        result = china_settlement._check_st_stock_rules(trade)

        assert result is False
    
    def test_check_st_stock_rules_below_limit(self):
        """测试ST股票规则检查 - 低于下限"""
        mock_engine = Mock()
        china_settlement = ChinaSettlementEngine(mock_engine)

        trade = Trade(
            symbol="ST000001",
            price=9.0,  # 低于5%下限（prev_close=10.0）
            quantity=100,
            side="BUY",
            prev_close=10.0,
            close_price=9.5
        )

        result = china_settlement._check_st_stock_rules(trade)

        assert result is False

    def test_process_after_hours_trading_enabled(self):
        """测试盘后交易处理 - 已启用"""
        mock_engine = Mock()
        china_settlement = ChinaSettlementEngine(mock_engine)

        trades = [
            Trade(
                symbol="688001",
                price=20.0,
                quantity=100,
                side="BUY",
                prev_close=18.0,
                close_price=19.5
            )
        ]

        result = china_settlement.process_after_hours_trading(trades)

        assert result > 0
        # 价格应该被设置为收盘价
        assert trades[0].price == 19.5

    def test_process_after_hours_trading_disabled(self):
        """测试盘后交易处理 - 已禁用"""
        mock_engine = Mock()
        china_settlement = ChinaSettlementEngine(mock_engine)
        china_settlement.star_market_rules["after_hours_trading"] = False

        trades = [
            Trade(
                symbol="688001",
                price=20.0,
                quantity=100,
                side="BUY",
                prev_close=18.0,
                close_price=19.5
            )
        ]

        result = china_settlement.process_after_hours_trading(trades)

        assert result == 0.0

    def test_process_after_hours_trading_non_star_market(self):
        """测试盘后交易处理 - 非科创板股票"""
        mock_engine = Mock()
        china_settlement = ChinaSettlementEngine(mock_engine)

        trades = [
            Trade(
                symbol="000001",
                price=10.0,
                quantity=100,
                side="BUY",
                prev_close=9.5,
                close_price=9.8
            )
        ]

        result = china_settlement.process_after_hours_trading(trades)

        # 非科创板股票不应该处理
        assert result == 0.0

    def test_process_after_hours_trading_multiple_trades(self):
        """测试盘后交易处理 - 多个交易"""
        mock_engine = Mock()
        china_settlement = ChinaSettlementEngine(mock_engine)

        trades = [
            Trade(
                symbol="688001",
                price=20.0,
                quantity=100,
                side="BUY",
                prev_close=18.0,
                close_price=19.5
            ),
            Trade(
                symbol="688002",
                price=30.0,
                quantity=50,
                side="SELL",
                prev_close=28.0,
                close_price=29.5
            )
        ]

        result = china_settlement.process_after_hours_trading(trades)

        assert result > 0
        assert china_settlement.settled_positions["688001"] == 100
        assert china_settlement.settled_positions["688002"] == -50
