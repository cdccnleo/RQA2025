"""
交易服务层核心异常测试
测试交易相关的异常类和错误处理机制
"""

import pytest
from pathlib import Path
import sys

# 确保Python路径正确配置
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
src_path_str = str(project_root / "src")
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

# 导入异常类
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
    BrokerException
)


class TestTradingExceptions:
    """交易异常测试"""

    def test_trading_exception_basic(self):
        """测试基础交易异常"""
        message = "Trading operation failed"
        error_code = 500

        exception = TradingException(message, error_code)

        assert str(exception) == message
        assert exception.error_code == error_code
        assert exception.message == message

    def test_order_exception(self):
        """测试订单异常"""
        message = "Order submission failed"
        order_id = "ORD_001"

        exception = OrderException(message, order_id)

        assert "订单异常" in str(exception)
        assert order_id in str(exception)
        assert exception.order_id == order_id

    def test_execution_exception(self):
        """测试执行异常"""
        message = "Order execution failed"
        execution_id = "EXEC_001"

        exception = ExecutionException(message, execution_id)

        assert "执行异常" in str(exception)
        assert execution_id in str(exception)
        assert exception.execution_id == execution_id

    def test_connection_exception(self):
        """测试连接异常"""
        message = "Connection lost"
        broker_name = "SimulatedBroker"

        exception = ConnectionException(message, broker_name)

        assert "连接异常" in str(exception)
        assert broker_name in str(exception)
        assert exception.broker_name == broker_name

    def test_insufficient_funds_exception(self):
        """测试资金不足异常"""
        message = "Not enough balance"
        required_amount = 10000.0

        exception = InsufficientFundsException(message, required_amount)

        assert "资金不足" in str(exception)
        assert str(required_amount) in str(exception)
        assert exception.required_amount == required_amount

    def test_invalid_order_exception(self):
        """测试无效订单异常"""
        message = "Order quantity must be positive"
        order_details = {"type": "LIMIT", "quantity": -100}

        exception = InvalidOrderException(message, order_details)

        assert "无效订单" in str(exception)
        assert str(order_details) in str(exception)
        assert exception.order_details == order_details

    def test_market_data_exception(self):
        """测试市场数据异常"""
        message = "Market data unavailable"
        symbol = "AAPL"

        exception = MarketDataException(message, symbol)

        assert "市场数据异常" in str(exception)
        assert symbol in str(exception)
        assert exception.symbol == symbol

    def test_risk_control_exception(self):
        """测试风险控制异常"""
        message = "Risk limit exceeded"
        risk_type = "max_position"

        exception = RiskControlException(message, risk_type)

        assert "风险控制异常" in str(exception)
        assert risk_type in str(exception)
        assert exception.risk_type == risk_type

    def test_timeout_exception(self):
        """测试超时异常"""
        message = "Operation timed out"
        timeout_seconds = 30

        exception = TimeoutException(message, timeout_seconds)

        assert "操作超时" in str(exception)
        assert str(timeout_seconds) in str(exception)
        assert exception.timeout_seconds == timeout_seconds

    def test_broker_exception(self):
        """测试券商异常"""
        message = "Broker API error"
        broker_code = "SIM001"

        exception = BrokerException(message, broker_code)

        assert "券商异常" in str(exception)
        assert broker_code in str(exception)
        assert exception.broker_code == broker_code

    def test_exception_inheritance(self):
        """测试异常继承关系"""
        base_exception = TradingException("test")
        assert isinstance(base_exception, Exception)

        order_exception = OrderException("test", "ORD_001")
        assert isinstance(order_exception, TradingException)

        execution_exception = ExecutionException("test", "EXEC_001")
        assert isinstance(execution_exception, TradingException)

        assert issubclass(OrderException, TradingException)
        assert issubclass(ExecutionException, TradingException)

    def test_exception_with_default_values(self):
        """测试异常默认值"""
        # 测试没有额外参数的异常
        exception = TradingException("test")
        assert exception.error_code == -1

        # 测试有默认参数的异常
        order_exception = OrderException("test")
        assert order_exception.order_id is None

        execution_exception = ExecutionException("test")
        assert execution_exception.execution_id is None
