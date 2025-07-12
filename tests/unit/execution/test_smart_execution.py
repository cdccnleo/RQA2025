import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from src.execution.smart_execution import (
    ExchangeType,
    OrderType,
    RoutingDecision,
    ExecutionParameters,
    SmartRouter,
    AlgorithmicExecution,
    ExecutionAnalyzer,
    ExecutionRiskManager,
    SmartExecutionEngine
)

@pytest.fixture
def mock_adapters():
    """创建模拟交易所适配器"""
    return {
        ExchangeType.SHANGHAI: MagicMock(),
        ExchangeType.SHENZHEN: MagicMock(),
        ExchangeType.HONGKONG: MagicMock()
    }

@pytest.fixture
def sample_market_data():
    """生成测试市场数据"""
    return {
        'bid': 100.0,
        'ask': 100.1,
        'depth': 10000,
        'daily_volume': 1_000_000,
        'volatility': 0.15
    }

def test_smart_router_routing(mock_adapters, sample_market_data):
    """测试智能路由决策"""
    router = SmartRouter(mock_adapters)

    # 设置模拟数据
    for adapter in mock_adapters.values():
        adapter.get_market_data.return_value = sample_market_data

    # 测试直接路由
    router.routing_strategy = RoutingDecision.DIRECT
    exchange = router.determine_routing('600519.SH', OrderType.LIMIT, ExecutionParameters())
    assert exchange == ExchangeType.SHANGHAI

    # 测试智能路由
    router.routing_strategy = RoutingDecision.SMART
    exchange = router.determine_routing('600519.SH', OrderType.LIMIT, ExecutionParameters())
    assert exchange in mock_adapters.keys()

    # 测试延迟更新
    router.update_latency(ExchangeType.SHANGHAI, 0.05)
    assert router.latency_stats[ExchangeType.SHANGHAI] < 1.0

def test_algorithmic_execution(mock_adapters):
    """测试算法执行"""
    router = SmartRouter(mock_adapters)
    algo = AlgorithmicExecution(router)

    # 测试订单调度
    algo.schedule_order(
        symbol='600519.SH',
        quantity=10000,
        order_type=OrderType.LIMIT,
        params=ExecutionParameters(),
        priority=1
    )
    assert not algo.order_queue.empty()

    # 测试订单处理
    with patch.object(algo, '_execute_order') as mock_execute:
        algo.process_orders()
        mock_execute.assert_called_once()

def test_execution_analyzer():
    """测试执行分析"""
    analyzer = ExecutionAnalyzer()

    # 记录执行
    analyzer.log_execution(
        order_id='order1',
        exchange=ExchangeType.SHANGHAI,
        symbol='600519.SH',
        quantity=1000,
        price=100.05,
        timestamp=pd.Timestamp.now()
    )
    assert len(analyzer.execution_log) == 1

    # 测试VWAP计算
    vwap = analyzer.calculate_vwap(
        '600519.SH',
        pd.Timestamp.now() - pd.Timedelta('1d'),
        pd.Timestamp.now()
    )
    assert np.isclose(vwap, 100.05)

    # 测试滑点计算
    slippage = analyzer.calculate_slippage('600519.SH', 100.0)
    assert 'average' in slippage
    assert np.isclose(slippage['average'], 0.0005)

def test_execution_risk_manager():
    """测试执行风控"""
    risk_manager = ExecutionRiskManager(
        max_order_size=0.1,
        max_daily_volume=0.3,
        volatility_threshold=0.2
    )

    # 测试风险检查
    assert risk_manager.check_order_risk(
        '600519.SH', 100000, 1_000_000, 0.15
    )
    assert not risk_manager.check_order_risk(
        '600519.SH', 200000, 1_000_000, 0.15
    )

    # 测试交易量更新
    risk_manager.update_traded_volume('600519.SH', 100000)
    assert risk_manager.daily_traded['600519.SH'] == 100000

def test_smart_execution_engine(mock_adapters, sample_market_data):
    """测试智能执行引擎"""
    engine = SmartExecutionEngine()
    engine.adapters = mock_adapters

    # 设置模拟数据
    for adapter in mock_adapters.values():
        adapter.get_market_data.return_value = sample_market_data
        adapter.send_order.return_value = 'order123'

    # 测试订单执行
    with patch.object(engine.algorithm, 'process_orders'):
        engine.execute_order('600519.SH', 10000)
        engine.algorithm.schedule_order.assert_called()

    # 测试执行统计
    stats = engine.get_execution_stats('600519.SH')
    assert 'vwap' in stats
    assert 'slippage' in stats

@pytest.mark.parametrize("quantity,expected_chunks", [
    (10000, [2000, 2000, 2000, 2000, 2000]),  # 默认参与率20%
    (5000, [1000, 1000, 1000, 1000, 1000]),
    (3000, [600, 600, 600, 600, 600])
])
def test_order_splitting(quantity, expected_chunks):
    """测试订单拆分"""
    algo = AlgorithmicExecution(MagicMock())
    chunks = algo._split_order(quantity, 0.2)
    assert sum(chunks) == quantity
    assert all(c <= max(expected_chunks) for c in chunks)

@pytest.mark.parametrize("strategy,expected_type", [
    (RoutingDecision.DIRECT, ExchangeType.SHANGHAI),
    (RoutingDecision.SMART, ExchangeType.SHANGHAI),
    (RoutingDecision.DARK, ExchangeType.HONGKONG),
    (RoutingDecision.LIQUIDITY, ExchangeType.SHENZHEN)
])
def test_routing_strategies(mock_adapters, strategy, expected_type):
    """测试路由策略"""
    router = SmartRouter(mock_adapters)
    router.routing_strategy = strategy

    # 设置模拟数据
    for adapter in mock_adapters.values():
        adapter.get_market_data.return_value = {
            'bid': 100.0,
            'ask': 100.1,
            'depth': 10000
        }

    exchange = router.determine_routing(
        '600519.SH', OrderType.LIMIT, ExecutionParameters()
    )
    assert exchange == expected_type
