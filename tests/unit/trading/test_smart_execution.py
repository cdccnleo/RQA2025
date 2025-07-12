import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from src.trading.smart_execution import (
    SmartExecutionEngine,
    ExecutionStrategy,
    MarketImpactModel,
    LiquidityAnalyzer,
    ExecutionOptimizer,
    TradingCostModel
)

@pytest.fixture
def sample_market_data():
    """生成测试市场数据"""
    return {
        'price': 100.0,
        'order_book': {
            'bids': [
                {'price': 99.9, 'volume': 1000},
                {'price': 99.8, 'volume': 2000},
                {'price': 99.7, 'volume': 3000}
            ],
            'asks': [
                {'price': 100.1, 'volume': 1000},
                {'price': 100.2, 'volume': 2000},
                {'price': 100.3, 'volume': 3000}
            ]
        },
        'volume_profile': {
            'open': 0.2,
            'midday': 0.3,
            'close': 0.5
        }
    }

@pytest.fixture
def historical_data():
    """生成历史数据"""
    dates = pd.date_range("2025-01-01", periods=100)
    data = pd.DataFrame({
        'close': np.cumsum(np.random.randn(100)) + 100,
        'volume': np.random.randint(10000, 50000, size=100)
    }, index=dates)
    return data

@pytest.fixture
def execution_engine(historical_data):
    """初始化执行引擎"""
    engine = SmartExecutionEngine(strategy=ExecutionStrategy.VWAP)
    engine.register_symbol("TEST", historical_data)
    return engine

def test_market_impact_model(historical_data):
    """测试市场冲击模型"""
    model = MarketImpactModel("TEST", historical_data)

    # 测试冲击估计
    impact = model.estimate_impact(order_size=10000, current_vol=50000)
    assert 0 <= impact <= 0.1  # 合理范围

    # 大订单应有更大冲击
    impact_large = model.estimate_impact(order_size=50000, current_vol=50000)
    impact_small = model.estimate_impact(order_size=1000, current_vol=50000)
    assert impact_large > impact_small

def test_liquidity_analyzer():
    """测试流动性分析器"""
    analyzer = LiquidityAnalyzer()
    order_book = {
        'bids': [
            {'price': 99.9, 'volume': 1000},
            {'price': 99.8, 'volume': 2000}
        ],
        'asks': [
            {'price': 100.1, 'volume': 1000},
            {'price': 100.2, 'volume': 2000}
        ]
    }

    # 分析流动性
    liquidity = analyzer.analyze_depth(order_book)
    assert liquidity['bid_volume'] == 3000
    assert liquidity['ask_volume'] == 3000
    assert liquidity['spread'] == 0.2

    # 测试趋势计算
    for _ in range(20):
        analyzer.analyze_depth(order_book)
    trend = analyzer.get_liquidity_trend()
    assert trend == 3000.0  # 20个相同数据点的平均值

def test_twap_strategy(execution_engine, sample_market_data):
    """测试TWAP执行策略"""
    orders = execution_engine.execute_order(
        symbol="TEST",
        target_quantity=5000,
        current_price=sample_market_data['price'],
        order_book=sample_market_data['order_book'],
        volume_profile=sample_market_data['volume_profile']
    )

    assert len(orders) == 5  # 默认分5片
    assert sum(o['quantity'] for o in orders) == 5000
    assert all(o['strategy'] == 'TWAP' for o in orders)
    assert all(o['limit_price'] >= 100.0 for o in orders)  # 买单限价应>=当前价

def test_vwap_strategy(execution_engine, sample_market_data):
    """测试VWAP执行策略"""
    engine = SmartExecutionEngine(strategy=ExecutionStrategy.VWAP)
    engine.register_symbol("TEST", pd.DataFrame())  # 不需要历史数据

    orders = engine.execute_order(
        symbol="TEST",
        target_quantity=5000,
        current_price=sample_market_data['price'],
        order_book=sample_market_data['order_book'],
        volume_profile=sample_market_data['volume_profile']
    )

    assert len(orders) == 3  # 按成交量分布分3个时段
    assert sum(o['quantity'] for o in orders) == 5000
    assert all(o['strategy'] == 'VWAP' for o in orders)

    # 检查各时段分配比例
    open_qty = next(o['quantity'] for o in orders if o['period'] == 'open')
    assert abs(open_qty - 1000) < 1e-6  # 5000 * 0.2

def test_execution_optimizer(historical_data, sample_market_data):
    """测试执行优化器"""
    optimizer = ExecutionOptimizer()
    results = optimizer.evaluate_strategies(
        symbol="TEST",
        quantity=10000,
        market_data=sample_market_data
    )

    assert 'TWAP' in results
    assert 'VWAP' in results
    assert 'IMPACT' in results

    # 检查成本结构
    for strategy in results.values():
        assert 'sub_orders' in strategy
        assert 'estimated_cost' in strategy
        assert isinstance(strategy['estimated_cost']['total_cost'], float)

def test_trading_cost_model(sample_market_data):
    """测试交易成本模型"""
    cost_model = TradingCostModel()
    orders = [
        {'quantity': 1000, 'limit_price': 100.1},
        {'quantity': 2000, 'limit_price': 100.2}
    ]

    costs = cost_model.estimate_execution_cost(orders, sample_market_data)

    assert 'commission' in costs
    assert 'slippage' in costs
    assert 'impact_cost' in costs
    assert 'total_cost' in costs

    # 佣金计算验证
    notional = 1000*100.1 + 2000*100.2
    expected_commission = notional * 0.0002
    assert abs(costs['commission'] - expected_commission) < 1e-6

def test_negative_quantity(execution_engine, sample_market_data):
    """测试卖单执行"""
    orders = execution_engine.execute_order(
        symbol="TEST",
        target_quantity=-3000,  # 卖单
        current_price=sample_market_data['price'],
        order_book=sample_market_data['order_book'],
        volume_profile=sample_market_data['volume_profile']
    )

    assert sum(o['quantity'] for o in orders) == -3000
    assert all(o['limit_price'] <= 100.0 for o in orders)  # 卖单限价应<=当前价

def test_small_quantity(execution_engine, sample_market_data):
    """测试小订单执行"""
    orders = execution_engine.execute_order(
        symbol="TEST",
        target_quantity=100,  # 小订单
        current_price=sample_market_data['price'],
        order_book=sample_market_data['order_book'],
        volume_profile=sample_market_data['volume_profile']
    )

    assert len(orders) == 1  # 小订单不应拆分
    assert orders[0]['quantity'] == 100

def test_zero_volume_profile(execution_engine, sample_market_data):
    """测试零成交量分布"""
    data = sample_market_data.copy()
    data['volume_profile'] = {'open': 0, 'midday': 0, 'close': 0}

    orders = execution_engine.execute_order(
        symbol="TEST",
        target_quantity=5000,
        current_price=data['price'],
        order_book=data['order_book'],
        volume_profile=data['volume_profile']
    )

    # 零成交量时应回退到TWAP
    assert len(orders) == 5
    assert all(o['strategy'] == 'TWAP' for o in orders)
