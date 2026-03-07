#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交易层异常测试

测试目标：提升exceptions.py的覆盖率
"""

import pytest
from datetime import datetime, timedelta

from src.trading.core.exceptions import (
    TradingException,
    OrderException,
    ExecutionException,
    ConnectionException,
    InsufficientFundsException,
    InvalidOrderException,
    MarketDataException,
    RiskControlException,
    TimeoutException,
    BrokerException,
    handle_trading_exception,
    validate_order_params,
    validate_connection_status,
    validate_sufficient_funds,
    check_order_timeout
)


class TestTradingExceptions:
    """测试交易异常类"""
    
    def test_trading_exception(self):
        """测试基础交易异常"""
        exc = TradingException("测试错误", error_code=1001)
        assert str(exc) == "测试错误"
        assert exc.error_code == 1001
        assert exc.message == "测试错误"
    
    def test_order_exception(self):
        """测试订单异常"""
        exc = OrderException("订单错误", order_id="ORD001")
        assert "ORD001" in str(exc)
        assert exc.order_id == "ORD001"
        assert exc.error_code == -1
    
    def test_execution_exception(self):
        """测试执行异常"""
        exc = ExecutionException("执行错误", execution_id="EXE001")
        assert "EXE001" in str(exc)
        assert exc.execution_id == "EXE001"
    
    def test_connection_exception(self):
        """测试连接异常"""
        exc = ConnectionException("连接错误", broker_name="TestBroker")
        assert "TestBroker" in str(exc)
        assert exc.broker_name == "TestBroker"
    
    def test_insufficient_funds_exception(self):
        """测试资金不足异常"""
        exc = InsufficientFundsException("资金不足", required_amount=10000.0)
        assert "10000.0" in str(exc)
        assert exc.required_amount == 10000.0
    
    def test_invalid_order_exception(self):
        """测试无效订单异常"""
        order_details = {"symbol": "AAPL", "quantity": 100}
        exc = InvalidOrderException("无效订单", order_details=order_details)
        assert str(order_details) in str(exc)
        assert exc.order_details == order_details
    
    def test_market_data_exception(self):
        """测试市场数据异常"""
        exc = MarketDataException("市场数据错误", symbol="AAPL")
        assert "AAPL" in str(exc)
        assert exc.symbol == "AAPL"
    
    def test_risk_control_exception(self):
        """测试风险控制异常"""
        exc = RiskControlException("风险控制错误", risk_type="position_limit")
        assert "position_limit" in str(exc)
        assert exc.risk_type == "position_limit"
    
    def test_timeout_exception(self):
        """测试超时异常"""
        exc = TimeoutException("操作超时", timeout_seconds=30)
        assert "30" in str(exc)
        assert exc.timeout_seconds == 30
    
    def test_broker_exception(self):
        """测试券商异常"""
        exc = BrokerException("券商错误", broker_code="BROKER001")
        assert "BROKER001" in str(exc)
        assert exc.broker_code == "BROKER001"


class TestExceptionDecorators:
    """测试异常装饰器"""
    
    def test_handle_trading_exception_success(self):
        """测试异常处理装饰器 - 成功"""
        @handle_trading_exception
        def test_func():
            return "success"
        
        assert test_func() == "success"
    
    def test_handle_trading_exception_trading_error(self):
        """测试异常处理装饰器 - 交易异常"""
        @handle_trading_exception
        def test_func():
            raise TradingException("交易错误")
        
        with pytest.raises(TradingException) as exc_info:
            test_func()
        assert "交易错误" in str(exc_info.value)
    
    def test_handle_trading_exception_general_error(self):
        """测试异常处理装饰器 - 通用异常"""
        @handle_trading_exception
        def test_func():
            raise ValueError("通用错误")
        
        with pytest.raises(TradingException) as exc_info:
            test_func()
        assert "意外交易错误" in str(exc_info.value)


class TestValidationFunctions:
    """测试验证函数"""
    
    def test_validate_order_params_success(self):
        """测试订单参数验证 - 成功"""
        order = {
            'symbol': 'AAPL',
            'side': 'BUY',
            'quantity': 100
        }
        # 不应该抛出异常
        validate_order_params(order)
    
    def test_validate_order_params_missing_fields(self):
        """测试订单参数验证 - 缺少字段"""
        order = {
            'symbol': 'AAPL'
        }
        with pytest.raises(InvalidOrderException) as exc_info:
            validate_order_params(order)
        assert "缺少必需字段" in str(exc_info.value)
    
    def test_validate_order_params_zero_quantity(self):
        """测试订单参数验证 - 数量为0"""
        order = {
            'symbol': 'AAPL',
            'side': 'BUY',
            'quantity': 0
        }
        with pytest.raises(InvalidOrderException) as exc_info:
            validate_order_params(order)
        assert "订单数量必须大于0" in str(exc_info.value)
    
    def test_validate_order_params_invalid_side(self):
        """测试订单参数验证 - 无效方向"""
        order = {
            'symbol': 'AAPL',
            'side': 'INVALID',
            'quantity': 100
        }
        with pytest.raises(InvalidOrderException) as exc_info:
            validate_order_params(order)
        assert "无效订单方向" in str(exc_info.value)
    
    def test_validate_connection_status_connected(self):
        """测试连接状态验证 - 已连接"""
        # 不应该抛出异常
        validate_connection_status(True, "TestBroker")
    
    def test_validate_connection_status_disconnected(self):
        """测试连接状态验证 - 未连接"""
        with pytest.raises(ConnectionException) as exc_info:
            validate_connection_status(False, "TestBroker")
        assert "TestBroker" in str(exc_info.value)
    
    def test_validate_sufficient_funds_sufficient(self):
        """测试资金充足性验证 - 充足"""
        # 不应该抛出异常
        validate_sufficient_funds(10000.0, 5000.0)
    
    def test_validate_sufficient_funds_insufficient(self):
        """测试资金充足性验证 - 不足"""
        with pytest.raises(InsufficientFundsException) as exc_info:
            validate_sufficient_funds(1000.0, 5000.0)
        assert "不足" in str(exc_info.value)
    
    def test_check_order_timeout_not_timeout(self):
        """测试订单超时检查 - 未超时"""
        order_timestamp = datetime.now() - timedelta(seconds=10)
        result = check_order_timeout(order_timestamp, 30)
        assert result is False
    
    def test_check_order_timeout_timeout(self):
        """测试订单超时检查 - 超时"""
        order_timestamp = datetime.now() - timedelta(seconds=40)
        with pytest.raises(TimeoutException) as exc_info:
            check_order_timeout(order_timestamp, 30)
        assert "超时" in str(exc_info.value)
    
    def test_check_order_timeout_string_timestamp(self):
        """测试订单超时检查 - 字符串时间戳"""
        order_timestamp = (datetime.now() - timedelta(seconds=10)).isoformat()
        result = check_order_timeout(order_timestamp, 30)
        assert result is False
    
    def test_check_order_timeout_string_timestamp_timeout(self):
        """测试订单超时检查 - 字符串时间戳超时"""
        order_timestamp = (datetime.now() - timedelta(seconds=40)).isoformat()
        with pytest.raises(TimeoutException):
            check_order_timeout(order_timestamp, 30)

