import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from src.backtesting.backtest_engine import (
    BacktestEngine,
    CommissionModel,
    SlippageModel,
    BaseStrategy,
    PerformanceAnalyzer,
    CommissionType,
    SlippageType
)

@pytest.fixture
def sample_data():
    """生成测试数据"""
    dates = pd.date_range('2023-01-01', periods=10)
    return {
        '600000.SH': pd.DataFrame({
            'open': np.linspace(10, 11, 10),
            'high': np.linspace(11, 12, 10),
            'low': np.linspace(9.5, 10.5, 10),
            'close': np.linspace(10, 11, 10),
            'volume': np.linspace(1e6, 2e6, 10)
        }, index=dates),
        '000001.SZ': pd.DataFrame({
            'open': np.linspace(20, 21, 10),
            'high': np.linspace(21, 22, 10),
            'low': np.linspace(19.5, 20.5, 10),
            'close': np.linspace(20, 21, 10),
            'volume': np.linspace(2e6, 3e6, 10)
        }, index=dates)
    }

@pytest.fixture
def mock_strategy():
    """模拟策略"""
    class TestStrategy(BaseStrategy):
        def generate_signals(self, data):
            # 简单策略：价格高于10.5时买入
            if data['close'].iloc[-1] > 10.5:
                return 1
            return 0
    return TestStrategy()

def test_backtest_initialization(sample_data):
    """测试回测初始化"""
    engine = BacktestEngine(sample_data)
    assert engine.initial_capital == 1e6
    assert len(engine.data) == 2

def test_commission_calculation():
    """测试佣金计算"""
    # 按金额计算
    model = CommissionModel(CommissionType.PER_VALUE, 0.0003)
    assert abs(model.calculate(10000) - 3) < 1e-6

    # 按笔数计算
    model = CommissionModel(CommissionType.PER_TRADE, 5)
    assert model.calculate(10000) == 5

    # 最低佣金
    model = CommissionModel(CommissionType.PER_VALUE, 0.0001, min_commission=10)
    assert model.calculate(50000) == 10

def test_slippage_calculation():
    """测试滑点计算"""
    # 固定滑点
    model = SlippageModel(SlippageType.FIXED, 0.01)
    assert model.calculate(10, 100) == 0.01

    # 波动率滑点 (需要mock波动率数据)
    with patch('numpy.std', return_value=0.1):
        model = SlippageModel(SlippageType.VOLATILITY, 0.5)
        assert abs(model.calculate(10, 100) - 0.05) < 1e-6

def test_backtest_execution(sample_data, mock_strategy):
    """测试回测执行"""
    engine = BacktestEngine(sample_data)
    engine.run_backtest(mock_strategy)

    # 检查是否有交易记录
    assert len(engine.trade_records) > 0
    # 检查组合价值记录
    assert len(engine.portfolio_values) == 10

def test_performance_metrics(sample_data, mock_strategy):
    """测试绩效指标计算"""
    engine = BacktestEngine(sample_data)
    engine.run_backtest(mock_strategy)

    # 生成基准数据
    benchmark = pd.Series(np.random.normal(0.001, 0.01, 10),
                         index=pd.date_range('2023-01-01', periods=10))

    analyzer = PerformanceAnalyzer(engine.portfolio_values, benchmark)
    metrics = analyzer.calculate_metrics()

    # 检查关键指标
    assert 'total_return' in metrics
    assert 'annual_return' in metrics
    assert 'max_drawdown' in metrics
    assert 'sharpe_ratio' in metrics

def test_visualization(sample_data, mock_strategy):
    """测试可视化功能"""
    engine = BacktestEngine(sample_data)
    engine.run_backtest(mock_strategy)

    # 测试净值曲线
    fig = BacktestVisualizer.plot_equity_curve(engine.portfolio_values)
    assert fig is not None

    # 测试回撤曲线
    fig = BacktestVisualizer.plot_drawdown(engine.portfolio_values)
    assert fig is not None

    # 测试交易分布
    fig = BacktestVisualizer.plot_trade_distribution(engine.trade_records)
    assert fig is not None

def test_position_management(sample_data):
    """测试仓位管理"""
    class PositionTestStrategy(BaseStrategy):
        def generate_signals(self, data):
            # 固定信号测试仓位变化
            return 1 if data.index[-1].day % 2 == 0 else -1

    engine = BacktestEngine(sample_data)
    engine.run_backtest(PositionTestStrategy())

    # 检查仓位快照
    assert len(engine.position_snapshots) > 0
    # 检查交易方向
    directions = [t.direction for t in engine.trade_records]
    assert 1 in directions and -1 in directions

def test_commission_impact(sample_data, mock_strategy):
    """测试佣金影响"""
    # 无佣金回测
    engine1 = BacktestEngine(sample_data)
    engine1.set_commission(CommissionModel(CommissionType.PER_TRADE, 0))
    engine1.run_backtest(mock_strategy)

    # 有佣金回测
    engine2 = BacktestEngine(sample_data)
    engine2.set_commission(CommissionModel(CommissionType.PER_VALUE, 0.003))
    engine2.run_backtest(mock_strategy)

    # 检查佣金影响
    assert (engine1.portfolio_values[-1]['value'] >
            engine2.portfolio_values[-1]['value'])

def test_slippage_impact(sample_data, mock_strategy):
    """测试滑点影响"""
    # 无滑点回测
    engine1 = BacktestEngine(sample_data)
    engine1.set_slippage(SlippageModel(SlippageType.FIXED, 0))
    engine1.run_backtest(mock_strategy)

    # 有滑点回测
    engine2 = BacktestEngine(sample_data)
    engine2.set_slippage(SlippageModel(SlippageType.FIXED, 0.01))
    engine2.run_backtest(mock_strategy)

    # 检查滑点影响
    assert (engine1.portfolio_values[-1]['value'] >
            engine2.portfolio_values[-1]['value'])

def test_benchmark_analysis(sample_data, mock_strategy):
    """测试基准分析"""
    engine = BacktestEngine(sample_data)
    engine.run_backtest(mock_strategy)

    # 生成基准数据
    benchmark = pd.Series(np.random.normal(0.001, 0.01, 10),
                         index=pd.date_range('2023-01-01', periods=10))

    analyzer = PerformanceAnalyzer(engine.portfolio_values, benchmark)
    metrics = analyzer.calculate_metrics()

    # 检查基准相关指标
    assert not np.isnan(metrics['beta'])
    assert not np.isnan(metrics['alpha'])
