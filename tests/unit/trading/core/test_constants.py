#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交易层常量测试

测试目标：提升constants.py的覆盖率
"""

import pytest

from src.trading.core import constants


class TestTradingConstants:
    """测试交易层常量"""
    
    def test_order_parameters(self):
        """测试订单参数常量"""
        assert constants.DEFAULT_ORDER_TIMEOUT == 300
        assert constants.DEFAULT_SLIPPAGE_TOLERANCE == 0.001
        assert constants.MAX_ORDER_RETRIES == 3
    
    def test_trading_limits(self):
        """测试交易限制常量"""
        assert constants.MAX_ORDERS_PER_SECOND == 100
        assert constants.MAX_POSITION_SIZE == 1000000
        assert constants.MIN_ORDER_SIZE == 1
    
    def test_risk_control(self):
        """测试风险控制常量"""
        assert constants.DEFAULT_STOP_LOSS_PCT == 0.05
        assert constants.DEFAULT_TAKE_PROFIT_PCT == 0.10
        assert constants.MAX_DAILY_LOSS_PCT == 0.02
    
    def test_commission_rates(self):
        """测试手续费率常量"""
        assert constants.DEFAULT_COMMISSION_RATE == 0.003
        assert constants.DEFAULT_MARKET_IMPACT_COST == 0.001
    
    def test_execution_parameters(self):
        """测试执行参数常量"""
        assert constants.DEFAULT_EXECUTION_TIMEOUT == 60
        assert constants.EXECUTION_CHECK_INTERVAL == 1
    
    def test_connection_parameters(self):
        """测试连接参数常量"""
        assert constants.CONNECTION_TIMEOUT == 30
        assert constants.RECONNECT_ATTEMPTS == 5
        assert constants.HEARTBEAT_INTERVAL == 30
    
    def test_cache_settings(self):
        """测试缓存设置常量"""
        assert constants.ORDER_CACHE_SIZE == 10000
        assert constants.POSITION_CACHE_SIZE == 1000
        assert constants.CACHE_TTL_SECONDS == 3600
    
    def test_batch_processing(self):
        """测试批量处理常量"""
        assert constants.DEFAULT_BATCH_SIZE == 100
        assert constants.MAX_BATCH_SIZE == 1000
    
    def test_monitoring_thresholds(self):
        """测试监控阈值常量"""
        assert constants.ORDER_PROCESSING_TIME_THRESHOLD == 5
        assert constants.EXECUTION_LATENCY_THRESHOLD == 100
    
    def test_capital_parameters(self):
        """测试资金参数常量"""
        assert constants.DEFAULT_LEVERAGE == 1.0
        assert constants.MAX_LEVERAGE == 10.0
    
    def test_market_data(self):
        """测试市场数据常量"""
        assert constants.MARKET_DATA_TIMEOUT == 10
        assert constants.PRICE_PRECISION == 4
        assert constants.VOLUME_PRECISION == 0
    
    def test_report_parameters(self):
        """测试报告参数常量"""
        assert constants.REPORT_UPDATE_INTERVAL == 60
        assert constants.PERFORMANCE_CHECK_INTERVAL == 300
    
    def test_alert_thresholds(self):
        """测试告警阈值常量"""
        assert constants.ALERT_THRESHOLD_HIGH == 0.8
        assert constants.ALERT_THRESHOLD_MEDIUM == 0.6
    
    def test_system_limits(self):
        """测试系统限制常量"""
        assert constants.MAX_ACTIVE_ORDERS == 1000
        assert constants.MAX_OPEN_POSITIONS == 100

