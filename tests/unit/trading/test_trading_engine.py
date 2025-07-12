import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
import pandas as pd
from src.trading.trading_engine import TradingEngine, OrderDirection, OrderStatus
from src.infrastructure.monitoring import ApplicationMonitor

@pytest.fixture
def mock_monitor():
    """模拟监控系统"""
    return MagicMock(spec=ApplicationMonitor)

@pytest.fixture
def trading_engine(mock_monitor):
    """交易引擎测试实例"""
    risk_config = {
        "initial_capital": 1000000.0,
        "per_trade_risk": 0.02,  # 每笔交易风险2%
        "max_position": {
            "600000": 10000  # 最大持仓限制
        }
    }
    return TradingEngine(risk_config=risk_config, monitor=mock_monitor)

def test_order_generation(trading_engine):
    """测试订单生成逻辑"""
    # 准备测试数据
    signals = pd.DataFrame({
        "symbol": ["600000", "000001"],
        "signal": [1, -1],  # 买入600000, 卖出000001
        "strength": [0.5, 1.0]  # 信号强度
    })
    current_prices = {"600000": 10.0, "000001": 15.0}

    # 生成订单
    orders = trading_engine.generate_orders(signals, current_prices)

    # 验证结果
    assert len(orders) == 2

    # 验证买入订单
    buy_order = next(o for o in orders if o["symbol"] == "600000")
    assert buy_order["direction"] == OrderDirection.BUY
    assert buy_order["quantity"] == 1000  # 1000000 * 0.02 * 0.5 / 10

    # 验证卖出订单
    sell_order = next(o for o in orders if o["symbol"] == "000001")
    assert sell_order["direction"] == OrderDirection.SELL
    assert sell_order["quantity"] == 1333  # 1000000 * 0.02 * 1.0 / 15

def test_position_calculation(trading_engine):
    """测试仓位计算"""
    # 测试买入仓位计算
    buy_pos = trading_engine._calculate_position_size(
        symbol="600000",
        signal=1,
        strength=0.5,
        price=10.0
    )
    assert buy_pos == 1000  # 1000000 * 0.02 * 0.5 / 10

    # 测试卖出仓位计算(无持仓时应为0)
    sell_pos = trading_engine._calculate_position_size(
        symbol="000001",
        signal=-1,
        strength=1.0,
        price=15.0
    )
    assert sell_pos == 0

    # 测试最大持仓限制
    trading_engine.positions["600000"] = 5000
    buy_pos = trading_engine._calculate_position_size(
        symbol="600000",
        signal=1,
        strength=1.0,
        price=10.0
    )
    assert buy_pos == 5000  # 最大持仓10000 - 当前5000 = 可买5000

def test_order_status_update(trading_engine):
    """测试订单状态更新"""
    # 生成测试订单
    signals = pd.DataFrame({
        "symbol": ["600000"],
        "signal": [1],
        "strength": [1.0]
    })
    current_prices = {"600000": 10.0}
    orders = trading_engine.generate_orders(signals, current_prices)

    # 更新订单状态为完全成交
    order_id = orders[0]["order_id"]
    trading_engine.update_order_status(
        order_id=order_id,
        filled_quantity=orders[0]["quantity"],
        avg_price=10.0,
        status=OrderStatus.FILLED
    )

    # 验证持仓和资金更新
    assert trading_engine.positions["600000"] == 2000  # 1000000 * 0.02 * 1.0 / 10
    assert trading_engine.cash_balance == 1000000 - 2000 * 10

    # 验证监控记录
    trading_engine.monitor.record_metric.assert_any_call(
        "order_created",
        value=1,
        tags={
            "symbol": "600000",
            "direction": "BUY",
            "type": "MARKET"
        }
    )
    trading_engine.monitor.record_metric.assert_any_call(
        "order_updated",
        value=1,
        tags={
            "symbol": "600000",
            "status": "FILLED"
        }
    )

def test_portfolio_value(trading_engine):
    """测试组合价值计算"""
    # 设置持仓和现金
    trading_engine.positions = {"600000": 1000, "000001": 500}
    trading_engine.cash_balance = 500000

    # 当前价格
    current_prices = {"600000": 12.0, "000001": 16.0}

    # 计算组合价值
    portfolio_value = trading_engine.get_portfolio_value(current_prices)

    # 验证结果
    expected_value = 500000 + (1000 * 12.0) + (500 * 16.0)
    assert portfolio_value == expected_value

def test_risk_control(trading_engine):
    """测试风险控制"""
    # 设置高风险配置
    trading_engine.risk_config["per_trade_risk"] = 0.5  # 50%风险

    # 测试仓位计算
    pos = trading_engine._calculate_position_size(
        symbol="600000",
        signal=1,
        strength=1.0,
        price=10.0
    )

    # 验证仓位不超过可用资金
    assert pos == 50000  # 1000000 * 0.5 / 10

    # 测试资金不足情况
    trading_engine.cash_balance = 10000
    pos = trading_engine._calculate_position_size(
        symbol="600000",
        signal=1,
        strength=1.0,
        price=10.0
    )
    assert pos == 1000  # 10000 / 10

def test_error_handling(trading_engine):
    """测试错误处理"""
    # 生成无效信号(价格为0)
    signals = pd.DataFrame({
        "symbol": ["600000"],
        "signal": [1],
        "strength": [1.0]
    })
    current_prices = {"600000": 0.0}

    # 生成订单(应跳过无效信号)
    orders = trading_engine.generate_orders(signals, current_prices)
    assert len(orders) == 0

    # 验证错误处理被调用
    assert trading_engine.error_handler.handlers  # 确保有错误处理器
