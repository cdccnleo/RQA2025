import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from src.trading.strategy_optimizer import (
    StrategyWeight,
    PortfolioOptimizer,
    PositionManager,
    RiskController,
    SignalProcessor,
    BacktestAnalyzer
)

@pytest.fixture
def sample_strategies():
    """创建模拟策略"""
    strategy1 = MagicMock()
    strategy1.name = "strategy1"

    strategy2 = MagicMock()
    strategy2.name = "strategy2"

    return [
        StrategyWeight(name="strategy1", weight=0.6, strategy=strategy1),
        StrategyWeight(name="strategy2", weight=0.4, strategy=strategy2)
    ]

@pytest.fixture
def sample_returns():
    """生成测试收益率数据"""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=100)
    return pd.DataFrame({
        "strategy1": np.random.normal(0.001, 0.02, 100),
        "strategy2": np.random.normal(0.0005, 0.015, 100)
    }, index=dates)

def test_portfolio_optimizer(sample_strategies, sample_returns):
    """测试组合优化器"""
    optimizer = PortfolioOptimizer(sample_strategies)

    # 测试权重优化
    optimized = optimizer.optimize_weights(sample_returns)
    assert len(optimized) == 2
    assert np.isclose(sum(optimized.values()), 1.0)

    # 验证权重更新
    assert np.isclose(sample_strategies[0].weight, optimized["strategy1"])
    assert np.isclose(sample_strategies[1].weight, optimized["strategy2"])

def test_position_manager():
    """测试仓位管理器"""
    pm = PositionManager(max_position=0.3, risk_per_trade=0.02)

    # 测试仓位计算
    position_size = pm.calculate_position_size(
        signal_strength=0.8,
        volatility=0.15,
        account_size=1000000
    )
    assert 0 < position_size < 300000  # 不超过最大仓位限制

    # 测试持仓更新
    pm.update_positions("AAPL", 100000)
    assert pm.current_positions["AAPL"] == 100000
    assert pm.get_total_exposure() == 100000

def test_risk_controller(sample_returns):
    """测试风险控制器"""
    rc = RiskController(max_drawdown=0.15, volatility_threshold=0.2)

    # 测试回撤检查
    assert rc.check_drawdown(850000, 1000000)  # 15%回撤
    assert not rc.check_drawdown(800000, 1000000)  # 20%回撤

    # 测试波动率检查
    assert rc.check_volatility(sample_returns["strategy1"])

    # 测试策略风险评估
    stats = rc.evaluate_strategy_risk("strategy1", sample_returns["strategy1"])
    assert "volatility" in stats
    assert "max_drawdown" in stats
    assert "sharpe_ratio" in stats

def test_signal_processor():
    """测试信号处理器"""
    sp = SignalProcessor(confirmation_period=3, threshold=0.6)

    # 生成测试信号
    signals = pd.Series([0.8, 0.7, 0.9, 0.2, 0.65, 0.1, 0.7, 0.8, 0.9])

    # 测试信号过滤
    filtered = sp.filter_signals(signals)
    assert filtered.iloc[3] == 0  # 低于阈值的信号被过滤
    assert filtered.iloc[0] == 0.8  # 高于阈值的信号保留

    # 测试信号平滑
    smoothed = sp.smooth_signals(signals)
    assert len(smoothed) == len(signals)

def test_backtest_analyzer(sample_returns):
    """测试回测分析器"""
    ba = BacktestAnalyzer(initial_capital=1000000)

    # 测试绩效计算
    stats = ba.calculate_performance(sample_returns["strategy1"])
    assert "total_return" in stats
    assert "annualized_return" in stats
    assert "max_drawdown" in stats
    assert "sharpe_ratio" in stats

    # 测试报告生成
    results = {
        "strategy1": pd.DataFrame({"returns": sample_returns["strategy1"]}),
        "strategy2": pd.DataFrame({"returns": sample_returns["strategy2"]})
    }
    report = ba.generate_report(results)
    assert len(report) == 2
    assert "strategy1" in report
    assert "strategy2" in report

def test_weight_validation(sample_strategies):
    """测试权重验证"""
    # 测试有效权重
    PortfolioOptimizer(sample_strategies)

    # 测试无效权重
    invalid_strategies = [
        StrategyWeight(name="s1", weight=0.7, strategy=MagicMock()),
        StrategyWeight(name="s2", weight=0.4, strategy=MagicMock())
    ]
    with pytest.raises(ValueError):
        PortfolioOptimizer(invalid_strategies)

@pytest.mark.parametrize("max_pos,risk", [(0.1, 0.01), (0.3, 0.02), (0.5, 0.05)])
def test_position_manager_params(max_pos, risk):
    """测试仓位管理参数化"""
    pm = PositionManager(max_position=max_pos, risk_per_trade=risk)
    size = pm.calculate_position_size(0.8, 0.15, 1000000)
    assert size <= max_pos * 1000000
