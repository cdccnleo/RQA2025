import pytest
import time
from unittest.mock import MagicMock, patch
from src.trading.gateway import (
    BaseGateway,
    FIXGateway,
    RESTGateway,
    GatewayManager,
    GatewayStatus,
    AccountInfo
)
from src.trading.execution_engine import Order, OrderType, OrderSide

@pytest.fixture
def fix_gateway():
    """创建FIX网关实例"""
    return FIXGateway()

@pytest.fixture
def rest_gateway():
    """创建REST网关实例"""
    return RESTGateway()

@pytest.fixture
def gateway_manager():
    """创建网关管理器实例"""
    return GatewayManager()

def test_fix_gateway_connection(fix_gateway):
    """测试FIX网关连接"""
    assert fix_gateway.status == GatewayStatus.DISCONNECTED

    # 测试连接
    fix_gateway.connect(
        host="127.0.0.1",
        port=9876,
        sender_comp_id="CLIENT",
        target_comp_id="SERVER"
    )
    assert fix_gateway.status == GatewayStatus.CONNECTED
    assert fix_gateway.session_id == "CLIENT-SERVER"

    # 测试断开
    fix_gateway.disconnect()
    assert fix_gateway.status == GatewayStatus.DISCONNECTED
    assert fix_gateway.session_id is None

def test_rest_gateway_connection(rest_gateway):
    """测试REST网关配置"""
    assert rest_gateway.status == GatewayStatus.DISCONNECTED

    # 测试配置(连接)
    rest_gateway.connect(
        base_url="https://api.example.com",
        api_key="test_key",
        secret="test_secret"
    )
    assert rest_gateway.status == GatewayStatus.CONNECTED

    # 测试重置(断开)
    rest_gateway.disconnect()
    assert rest_gateway.status == GatewayStatus.DISCONNECTED

def test_order_sending(fix_gateway):
    """测试订单发送"""
    fix_gateway.connect(
        host="127.0.0.1",
        port=9876,
        sender_comp_id="CLIENT",
        target_comp_id="SERVER"
    )

    # 创建测试订单
    order = Order(
        symbol="AAPL",
        order_type=OrderType.LIMIT,
        side=OrderSide.BUY,
        quantity=100,
        price=150.0
    )

    # 发送订单
    order_id = fix_gateway.send_order(order)
    assert order_id.startswith("FIX_")

    # 验证订单状态
    active_orders = fix_gateway.get_active_orders()
    assert order_id in active_orders
    assert active_orders[order_id]['symbol'] == "AAPL"
    assert active_orders[order_id]['status'] == "NEW"

def test_order_cancellation(fix_gateway):
    """测试订单取消"""
    fix_gateway.connect(
        host="127.0.0.1",
        port=9876,
        sender_comp_id="CLIENT",
        target_comp_id="SERVER"
    )

    # 发送测试订单
    order = Order(
        symbol="AAPL",
        order_type=OrderType.LIMIT,
        side=OrderSide.BUY,
        quantity=100,
        price=150.0
    )
    order_id = fix_gateway.send_order(order)

    # 取消订单
    fix_gateway.cancel_order(order_id)

    # 验证订单仍在活跃列表中(实际实现中状态会更新)
    assert order_id in fix_gateway.get_active_orders()

def test_account_query(rest_gateway):
    """测试账户查询"""
    rest_gateway.connect(
        base_url="https://api.example.com",
        api_key="test_key",
        secret="test_secret"
    )

    # 查询账户
    account = rest_gateway.query_account()

    assert isinstance(account, AccountInfo)
    assert account.account_id == "REST_ACCOUNT"
    assert account.balance == 2000000

def test_position_query(fix_gateway):
    """测试持仓查询"""
    fix_gateway.connect(
        host="127.0.0.1",
        port=9876,
        sender_comp_id="CLIENT",
        target_comp_id="SERVER"
    )

    # 查询持仓
    positions = fix_gateway.query_position()
    assert "AAPL" in positions
    assert positions["AAPL"] == 1000

    # 查询指定品种持仓
    aapl_pos = fix_gateway.query_position("AAPL")
    assert aapl_pos["AAPL"] == 1000

def test_order_update_callback(fix_gateway):
    """测试订单状态更新回调"""
    fix_gateway.connect(
        host="127.0.0.1",
        port=9876,
        sender_comp_id="CLIENT",
        target_comp_id="SERVER"
    )

    # 发送测试订单
    order = Order(
        symbol="AAPL",
        order_type=OrderType.LIMIT,
        side=OrderSide.BUY,
        quantity=100,
        price=150.0
    )
    order_id = fix_gateway.send_order(order)

    # 模拟订单部分成交
    fix_gateway.on_order_update(
        order_id=order_id,
        status="PARTIALLY_FILLED",
        filled=50,
        remaining=50,
        avg_price=149.5
    )

    # 验证订单状态更新
    active_orders = fix_gateway.get_active_orders()
    assert active_orders[order_id]['status'] == "PARTIALLY_FILLED"
    assert active_orders[order_id]['filled'] == 50
    assert active_orders[order_id]['avg_price'] == 149.5

def test_gateway_management(gateway_manager, fix_gateway, rest_gateway):
    """测试网关管理器"""
    # 添加网关
    gateway_manager.add_gateway(fix_gateway)
    gateway_manager.add_gateway(rest_gateway)

    assert len(gateway_manager.get_all_gateways()) == 2
    assert gateway_manager.get_gateway("FIX Gateway") is not None

    # 发送订单
    order = Order(
        symbol="AAPL",
        order_type=OrderType.LIMIT,
        side=OrderSide.BUY,
        quantity=100,
        price=150.0
    )

    # 配置REST网关
    rest_gateway.connect(
        base_url="https://api.example.com",
        api_key="test_key",
        secret="test_secret"
    )

    # 通过REST网关发送订单
    order_id = gateway_manager.send_order("REST Gateway", order)
    assert order_id.startswith("REST_")

    # 移除网关
    gateway_manager.remove_gateway("FIX Gateway")
    assert len(gateway_manager.get_all_gateways()) == 1

def test_gateway_error_handling(fix_gateway):
    """测试网关错误处理"""
    # 未连接时发送订单
    order = Order(
        symbol="AAPL",
        order_type=OrderType.LIMIT,
        side=OrderSide.BUY,
        quantity=100,
        price=150.0
    )

    with pytest.raises(ConnectionError):
        fix_gateway.send_order(order)

    # 取消不存在的订单
    with pytest.raises(ValueError):
        fix_gateway.cancel_order("INVALID_ORDER_ID")
