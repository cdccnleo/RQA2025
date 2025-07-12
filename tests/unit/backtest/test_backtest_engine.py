import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from src.backtest.backtest_engine import (
    BacktestEngine,
    BacktestConfig,
    BacktestMode,
    BacktestResult
)

@pytest.fixture
def mock_strategy():
    """模拟交易策略"""
    strategy = MagicMock()
    strategy.symbols = ['600000.SH', '000001.SZ']
    strategy.generate_signals.return_value = pd.DataFrame({
        '600000.SH': [1, 0, -1],
        '000001.SZ': [0, 1, 0]
    })
    strategy.clone.return_value = strategy
    return strategy

@pytest.fixture
def mock_data_provider():
    """模拟数据提供者"""
    data = MagicMock()

    # 模拟历史数据
    dates = pd.date_range('2023-01-01', periods=3)
    symbols = ['600000.SH', '000001.SZ']
    index = pd.MultiIndex.from_product(
        [dates, symbols],
        names=['datetime', 'symbol']
    )

    mock_hist_data = pd.DataFrame({
        'open': [10, 20, 10.5, 20.5, 11, 21],
        'high': [10.5, 21, 11, 21.5, 11.5, 22],
        'low': [9.5, 19, 10, 20, 10.5, 20.5],
        'close': [10, 20, 10.5, 20.5, 11, 21],
        'volume': [1e6, 2e6, 1.1e6, 2.1e6, 1.2e6, 2.2e6]
    }, index=index)

    data.load_hist_data.return_value = mock_hist_data
    return data

@pytest.fixture
def sample_config():
    """回测配置"""
    return BacktestConfig(
        start_date='2023-01-01',
        end_date='2023-01-03',
        initial_capital=1e6,
        commission=0.0005,
        slippage=0.001
    )

def test_single_backtest(mock_strategy, mock_data_provider, sample_config):
    """测试单策略回测"""
    engine = BacktestEngine(
        strategy=mock_strategy,
        data_provider=mock_data_provider,
        config=sample_config
    )

    results = engine.run(mode=BacktestMode.SINGLE)

    assert isinstance(results, dict)
    assert 'default' in results
    assert isinstance(results['default'], BacktestResult)
    assert len(results['default'].returns) == 3
    assert 'total_return' in results['default'].metrics

def test_multi_backtest(mock_strategy, mock_data_provider, sample_config):
    """测试多策略回测"""
    engine = BacktestEngine(
        strategy=mock_strategy,
        data_provider=mock_data_provider,
        config=sample_config
    )

    params_list = [
        {'param1': 0.1, 'name': 'strategy1'},
        {'param1': 0.2, 'name': 'strategy2'}
    ]

    with patch('concurrent.futures.ProcessPoolExecutor') as mock_executor:
        mock_executor.return_value.__enter__.return_value.submit.return_value.result.return_value = BacktestResult()

        results = engine.run(
            mode=BacktestMode.MULTI,
            params_list=params_list
        )

        assert len(results) == 2
        assert 'strategy1' in results
        assert 'strategy2' in results

def test_optimization_backtest(mock_strategy, mock_data_provider, sample_config):
    """测试参数优化回测"""
    engine = BacktestEngine(
        strategy=mock_strategy,
        data_provider=mock_data_provider,
        config=sample_config
    )

    params_grid = {
        'param1': [0.1, 0.2],
        'param2': [10, 20]
    }

    with patch('concurrent.futures.ProcessPoolExecutor') as mock_executor:
        mock_executor.return_value.__enter__.return_value.submit.return_value.result.return_value = BacktestResult()

        results = engine.run(
            mode=BacktestMode.OPTIMIZE,
            params_list=params_grid
        )

        assert len(results) == 4  # 2x2参数组合

def test_metric_calculation():
    """测试绩效指标计算"""
    result = BacktestResult()
    result.returns = pd.Series([1e6, 1.1e6, 1.2e6, 1.15e6],
                             index=pd.date_range('2023-01-01', periods=4))

    engine = BacktestEngine(None, None, None)
    engine._calculate_metrics(result)

    metrics = result.metrics
    assert 'total_return' in metrics
    assert metrics['total_return'] == pytest.approx(0.15, abs=0.01)
    assert 'annual_return' in metrics
    assert 'max_drawdown' in metrics
    assert metrics['max_drawdown'] == pytest.approx(0.05/1.2, abs=0.01)
    assert 'sharpe_ratio' in metrics

def test_trade_execution(mock_strategy, mock_data_provider, sample_config):
    """测试交易执行逻辑"""
    engine = BacktestEngine(
        strategy=mock_strategy,
        data_provider=mock_data_provider,
        config=sample_config
    )

    # 模拟信号
    signals = pd.DataFrame({
        '600000.SH': [1, 0, -1],
        '000001.SZ': [0, 1, 0]
    })

    positions = pd.DataFrame(columns=['symbol', 'amount', 'entry_price', 'entry_time'])
    prices = pd.Series([10, 20], index=['600000.SH', '000001.SZ'])
    capital = 1e6

    with patch.object(engine, '_update_positions') as mock_update:
        mock_update.return_value = (positions, capital)

        trades = engine._execute_trades(
            signals=signals,
            positions=positions,
            prices=prices,
            capital=capital
        )

        assert isinstance(trades, pd.DataFrame)

def test_position_update(mock_strategy, mock_data_provider, sample_config):
    """测试持仓更新逻辑"""
    engine = BacktestEngine(
        strategy=mock_strategy,
        data_provider=mock_data_provider,
        config=sample_config
    )

    positions = pd.DataFrame([{
        'symbol': '600000.SH',
        'amount': 1000,
        'entry_price': 10,
        'entry_time': '2023-01-01'
    }])

    trades = pd.DataFrame([{
        'symbol': '600000.SH',
        'amount': -500,
        'price': 11,
        'datetime': '2023-01-02'
    }])

    prices = pd.Series([11], index=['600000.SH'])
    capital = 1e6

    new_pos, new_capital = engine._update_positions(
        positions=positions,
        trades=trades,
        prices=prices
    )

    assert len(new_pos) == 1
    assert new_pos.iloc[0]['amount'] == 500
    assert new_capital > capital  # 应有利润

def test_plot_results(mock_strategy, mock_data_provider, sample_config):
    """测试结果绘图功能"""
    engine = BacktestEngine(
        strategy=mock_strategy,
        data_provider=mock_data_provider,
        config=sample_config
    )

    # 模拟结果
    result1 = BacktestResult()
    result1.returns = pd.Series([1e6, 1.1e6, 1.2e6],
                               index=pd.date_range('2023-01-01', periods=3))
    result1.metrics = {'total_return': 0.2}

    result2 = BacktestResult()
    result2.returns = pd.Series([1e6, 1.05e6, 1.15e6],
                               index=pd.date_range('2023-01-01', periods=3))
    result2.metrics = {'total_return': 0.15}

    with patch('matplotlib.pyplot.figure') as mock_fig:
        engine.plot_results({
            'strategy1': result1,
            'strategy2': result2
        })

        assert mock_fig.called
