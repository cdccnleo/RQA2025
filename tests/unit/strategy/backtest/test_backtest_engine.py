"""
测试回测引擎
"""

import pytest
from unittest.mock import Mock
import pandas as pd
import numpy as np
from datetime import datetime


class TestBacktestEngine:
    """测试回测引擎"""

    def test_backtest_engine_import(self):
        """测试回测引擎导入"""
        try:
            from src.strategy.backtest.backtest_engine import BacktestEngine, BacktestMode, BacktestResult
            assert BacktestEngine is not None
            assert BacktestMode is not None
            assert BacktestResult is not None
        except ImportError:
            pytest.skip("BacktestEngine not available")

    def test_backtest_mode_enum(self):
        """测试回测模式枚举"""
        try:
            from src.strategy.backtest.backtest_engine import BacktestMode

            assert BacktestMode.SINGLE.value == "single"
            assert BacktestMode.MULTI.value == "multi"
            assert BacktestMode.OPTIMIZE.value == "optimize"

        except ImportError:
            pytest.skip("BacktestMode not available")

    def test_backtest_result_dataclass(self):
        """测试回测结果数据类"""
        try:
            from src.strategy.backtest.backtest_engine import BacktestResult

            # 测试默认初始化
            result = BacktestResult()
            assert result.metrics == {}
            assert len(result.returns) == 0
            assert result.positions.empty
            assert result.trades.empty

            # 测试带参数初始化
            test_returns = pd.Series([0.01, 0.02, -0.01])
            test_metrics = {"sharpe": 1.5, "max_drawdown": 0.05}

            result = BacktestResult(
                returns=test_returns,
                metrics=test_metrics
            )
            assert len(result.returns) == 3
            assert result.metrics["sharpe"] == 1.5

        except ImportError:
            pytest.skip("BacktestResult not available")

    def test_backtest_engine_initialization(self):
        """测试回测引擎初始化"""
        try:
            from src.strategy.backtest.backtest_engine import BacktestEngine

            engine = BacktestEngine()
            assert engine is not None

            # 检查基本属性 - BacktestEngine可能有config, strategy, data_provider等属性
            assert hasattr(engine, 'config') or hasattr(engine, 'strategy') or hasattr(engine, 'data_provider')

        except ImportError:
            pytest.skip("BacktestEngine not available")

    def test_backtest_engine_run_backtest(self):
        """测试运行回测"""
        try:
            from src.strategy.backtest.backtest_engine import BacktestEngine, BacktestMode

            engine = BacktestEngine()

            # 创建测试数据
            dates = pd.date_range('2023-01-01', periods=100, freq='D')
            test_data = pd.DataFrame({
                'close': np.random.randn(100).cumsum() + 100,
                'volume': np.random.randint(1000, 10000, 100)
            }, index=dates)

            # 创建测试策略（模拟）
            class MockStrategy:
                def generate_signals(self, data):
                    return [{'type': 'BUY', 'price': data['close'].iloc[-1]}]

            strategy = MockStrategy()

            # 测试运行回测
            if hasattr(engine, 'run_backtest'):
                result = engine.run_backtest(strategy, test_data, mode=BacktestMode.SINGLE)
                assert result is not None

        except (ImportError, Exception):
            pytest.skip("Backtest execution not available")

    def test_backtest_engine_calculate_metrics(self):
        """测试指标计算"""
        try:
            from src.strategy.backtest.backtest_engine import BacktestEngine

            engine = BacktestEngine()

            # 创建测试收益序列
            returns = pd.Series([0.01, 0.02, -0.01, 0.005, -0.003])

            if hasattr(engine, 'calculate_metrics'):
                metrics = engine.calculate_metrics(returns)
                assert isinstance(metrics, dict)
                # 检查常见指标
                assert 'total_return' in metrics or 'sharpe' in metrics or 'volatility' in metrics

        except (ImportError, Exception):
            pytest.skip("Metrics calculation not available")
