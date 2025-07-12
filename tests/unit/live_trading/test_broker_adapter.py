import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
from src.live_trading.broker_adapter import (
    BrokerAdapter,
    CTPSimulatorAdapter,
    BrokerAdapterFactory,
    OrderStatus
)

@pytest.fixture
def mock_config():
    """模拟券商配置"""
    return {
        "host": "127.0.0.1",
        "port": 12345,
        "account": "test_account",
        "password": "test_password"
    }

@pytest.fixture
def ctp_adapter(mock_config):
    """CTP模拟器适配器实例"""
    return CTPSimulatorAdapter(config=mock_config)

def test_adapter_connection(ctp_adapter):
    """测试连接和断开"""
    # 测试连接
    assert ctp_adapter.connect() is True
    assert ctp_adapter.connected is True

    # 测试断开
    assert ctp_adapter.disconnect() is True
    assert ctp_adapter.connected is False

def test_order_placement(ctp_adapter, mock_config):
    """测试下单功能"""
    # 先建立连接
    ctp_adapter.connect()

    # 测试市价单
    market_order = {
        "symbol": "600000",
        "direction": "buy",
        "type": "market",
        "quantity": 100,
        "account": mock_config["account"]
    }

    order_id = ctp_adapter.place_order(market_order)
    assert order_id.startswith("CTP_")

    # 测试限价单
    limit_order = {
        "symbol": "000001",
        "direction": "sell",
        "type": "limit",
        "quantity": 200,
        "price": 10.5,
        "account": mock_config["account"]
    }

    order_id = ctp_adapter.place_order(limit_order)
    assert order_id.startswith("CTP_")

    # 测试未连接状态
    ctp_adapter.disconnect()
    with pytest.raises(ConnectionError):
        ctp_adapter.place_order(market_order)

def test_order_cancellation(ctp_adapter):
    """测试撤单功能"""
    ctp_adapter.connect()

    # 模拟订单ID
    order_id = "CTP_20230801123000123"

    # 撤单测试
    assert ctp_adapter.cancel_order(order_id) is True

def test_order_status_check(ctp_adapter):
    """测试订单状态查询"""
    ctp_adapter.connect()

    # 模拟订单ID
    order_id = "CTP_20230801123000123"

    # 查询状态
    status = ctp_adapter.get_order_status(order_id)

    # 验证返回字段
    assert "order_id" in status
    assert "status" in status
    assert status["status"] == OrderStatus.FILLED.value
    assert "filled_quantity" in status
    assert "avg_price" in status
    assert "timestamp" in status

def test_position_query(ctp_adapter):
    """测试持仓查询"""
    ctp_adapter.connect()

    # 查询持仓
    positions = ctp_adapter.get_positions()

    # 验证返回结构
    assert isinstance(positions, list)
    if positions:
        position = positions[0]
        assert "symbol" in position
        assert "quantity" in position
        assert "cost_price" in position
        assert "market_value" in position

def test_account_balance(ctp_adapter, mock_config):
    """测试资金查询"""
    ctp_adapter.connect()

    # 查询资金
    balance = ctp_adapter.get_account_balance(mock_config["account"])

    # 验证返回字段
    assert "total_assets" in balance
    assert "available_cash" in balance
    assert "margin" in balance
    assert "frozen_cash" in balance

def test_market_data(ctp_adapter):
    """测试行情获取"""
    ctp_adapter.connect()

    # 获取行情
    symbols = ["600000", "000001"]
    market_data = ctp_adapter.get_market_data(symbols)

    # 验证返回结构
    assert isinstance(market_data, dict)
    for symbol in symbols:
        assert symbol in market_data
        assert "last_price" in market_data[symbol]
        assert "ask_price" in market_data[symbol]
        assert "bid_price" in market_data[symbol]
        assert "volume" in market_data[symbol]
        assert "timestamp" in market_data[symbol]

def test_adapter_factory(mock_config):
    """测试适配器工厂"""
    # 创建CTP适配器
    ctp_adapter = BrokerAdapterFactory.create_adapter("ctp", mock_config)
    assert isinstance(ctp_adapter, CTPSimulatorAdapter)

    # 创建模拟器适配器
    simulator_adapter = BrokerAdapterFactory.create_adapter("simulator", mock_config)
    assert isinstance(simulator_adapter, CTPSimulatorAdapter)

    # 测试不支持的券商类型
    with pytest.raises(ValueError):
        BrokerAdapterFactory.create_adapter("unknown", mock_config)

def test_abstract_adapter():
    """测试抽象基类"""
    # 不能直接实例化抽象类
    with pytest.raises(TypeError):
        BrokerAdapter(config={})
