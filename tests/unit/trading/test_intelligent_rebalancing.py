import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from src.trading.intelligent_rebalancing import (
    MarketState,
    RebalancingSignal,
    PositionAdjustment,
    MarketStateClassifier,
    MultiFactorModel,
    AdaptiveRiskControl,
    IntelligentRebalancer
)

@pytest.fixture
def sample_market_data():
    """生成测试市场数据"""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=100)
    return pd.DataFrame({
        'close': np.cumprod(1 + np.random.normal(0.001, 0.02, 100)),
        'high': np.random.uniform(1.01, 1.05, 100),
        'low': np.random.uniform(0.95, 0.99, 100),
        'volume': np.random.randint(10000, 50000, 100)
    }, index=dates)

@pytest.fixture
def sample_factor_data():
    """生成测试因子数据"""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=100)
    return {
        'momentum': pd.DataFrame({
            'return': np.random.normal(0.001, 0.02, 100)
        }, index=dates),
        'value': pd.DataFrame({
            'return': np.random.normal(0.0005, 0.015, 100)
        }, index=dates),
        'quality': pd.DataFrame({
            'return': np.random.normal(0.0008, 0.01, 100)
        }, index=dates)
    }

@pytest.fixture
def sample_portfolio():
    """生成测试组合"""
    return {
        '600519.SH': 0.3,
        '000858.SZ': 0.2,
        '601318.SH': 0.15,
        '600036.SH': 0.1,
        '000333.SZ': 0.25
    }

def test_market_state_classifier(sample_market_data):
    """测试市场状态分类器"""
    classifier = MarketStateClassifier()

    # 模拟训练数据
    mock_features = np.random.rand(100, 4)
    mock_labels = np.random.randint(0, 4, 100)
    classifier.train(pd.DataFrame(mock_features), pd.Series(mock_labels))

    # 测试预测
    state = classifier.predict(sample_market_data)
    assert isinstance(state, MarketState)

def test_multi_factor_model():
    """测试多因子模型"""
    factors = ['momentum', 'value', 'quality']
    model = MultiFactorModel(factors)

    # 测试初始权重
    assert np.isclose(sum(model.factor_weights.values()), 1.0)

    # 测试权重更新
    model.update_weights({
        'momentum': 0.5,
        'value': 0.3,
        'quality': 0.2
    })
    assert np.isclose(model.factor_weights['momentum'], 0.5)

    # 测试信号生成
    scores = {
        'momentum': 0.8,
        'value': 0.6,
        'quality': 0.4
    }
    signal = model.generate_signal(scores)
    expected = 0.8*0.5 + 0.6*0.3 + 0.4*0.2
    assert np.isclose(signal, expected)

def test_adaptive_risk_control():
    """测试自适应风控"""
    risk_control = AdaptiveRiskControl(base_risk=0.1)

    # 测试波动率调整
    risk_control.adjust_for_volatility(0.25)
    assert np.isclose(risk_control.risk_adjustment, 0.7)

    # 测试回撤调整
    risk_control.adjust_for_drawdown(0.2)
    assert np.isclose(risk_control.risk_adjustment, 0.5)

    # 测试获取调整后风险
    assert np.isclose(risk_control.get_adjusted_risk(), 0.05)

def test_rebalancer_signals(sample_portfolio, sample_market_data, sample_factor_data):
    """测试调仓信号生成"""
    rebalancer = IntelligentRebalancer()

    # 模拟市场状态分类
    with patch.object(rebalancer.state_classifier, 'predict',
                     return_value=MarketState.TRENDING_UP):
        state = rebalancer.analyze_market(sample_market_data)
        assert state == MarketState.TRENDING_UP

    # 测试因子评估
    rebalancer.evaluate_factors(sample_factor_data)

    # 测试信号生成
    factor_scores = {
        'momentum': 0.8,
        'value': 0.6,
        'quality': 0.7,
        'volatility': 0.5
    }
    signals = rebalancer.generate_rebalancing_signals(
        sample_portfolio, state, factor_scores
    )

    assert len(signals) == len(sample_portfolio)
    assert all(isinstance(s, PositionAdjustment) for s in signals)

@pytest.mark.parametrize("state,score,expected_signal", [
    (MarketState.TRENDING_UP, 0.8, RebalancingSignal.INCREASE),
    (MarketState.TRENDING_DOWN, -0.6, RebalancingSignal.EXIT),
    (MarketState.VOLATILE, -0.4, RebalancingSignal.DECREASE),
    (MarketState.SIDEWAYS, 0.2, RebalancingSignal.HOLD)
])
def test_signal_determination(state, score, expected_signal):
    """测试信号决策逻辑"""
    rebalancer = IntelligentRebalancer()
    signal, _ = rebalancer._determine_signal(
        '600519.SH', 0.3, state, score
    )
    assert signal == expected_signal

def test_position_adjustment_calculation():
    """测试仓位调整计算"""
    rebalancer = IntelligentRebalancer()
    rebalancer.risk_control.base_risk = 0.2

    # 测试增加仓位
    amount = rebalancer._calculate_amount(
        RebalancingSignal.INCREASE, 0.1
    )
    assert np.isclose(amount, 0.1)

    # 测试减少仓位
    amount = rebalancer._calculate_amount(
        RebalancingSignal.DECREASE, 0.3
    )
    assert np.isclose(amount, 0.06)

    # 测试退出仓位
    amount = rebalancer._calculate_amount(
        RebalancingSignal.EXIT, 0.5
    )
    assert np.isclose(amount, 0.5)

    # 测试保持仓位
    amount = rebalancer._calculate_amount(
        RebalancingSignal.HOLD, 0.2
    )
    assert np.isclose(amount, 0.0)

def test_risk_adjustments():
    """测试风险调整应用"""
    rebalancer = IntelligentRebalancer()
    rebalancer.apply_risk_adjustments(volatility=0.25, drawdown=0.18)

    assert np.isclose(rebalancer.risk_control.risk_adjustment, 0.5)
    assert np.isclose(rebalancer.risk_control.get_adjusted_risk(), 0.05)
