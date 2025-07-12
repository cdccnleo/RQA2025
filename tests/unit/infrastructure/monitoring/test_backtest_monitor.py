import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from src.infrastructure.monitoring.backtest_monitor import BacktestMonitor

# 统一mock prometheus_client的指标对象
@pytest.fixture(autouse=True)
def mock_prometheus():
    with patch('prometheus_client.Counter', MagicMock()), \
         patch('prometheus_client.Gauge', MagicMock()), \
         patch('prometheus_client.Histogram', MagicMock()):
        yield

# Fixtures
@pytest.fixture
def backtest_monitor():
    """BacktestMonitor测试实例"""
    return BacktestMonitor()

# 测试用例
class TestBacktestMonitor:
    def test_record_trade(self, backtest_monitor):
        """测试记录交易事件"""
        test_time = datetime(2023, 1, 1, 10, 30)

        # 记录买入交易
        backtest_monitor.record_trade(
            symbol="600000.SH",
            action="BUY",
            price=15.2,
            quantity=1000,
            strategy="momentum",
            commission=5.0,
            timestamp=test_time
        )

        # 记录卖出交易
        backtest_monitor.record_trade(
            symbol="600000.SH",
            action="SELL",
            price=16.5,
            quantity=1000,
            strategy="momentum",
            timestamp=test_time + timedelta(days=1)
        )

        # 验证交易记录
        trades = backtest_monitor.get_trade_history()
        assert len(trades) == 2
        assert trades[0]['value'] == 15.2
        assert trades[0]['tags']['action'] == "BUY"
        assert trades[1]['value'] == 16.5
        assert trades[1]['tags']['action'] == "SELL"

    def test_record_portfolio(self, backtest_monitor):
        """测试记录组合状态"""
        test_time = datetime(2023, 1, 1)
        positions = {
            "600000.SH": 1000,
            "000001.SZ": 500
        }

        # 记录组合状态
        backtest_monitor.record_portfolio(
            value=1000000,
            cash=200000,
            positions=positions,
            strategy="momentum",
            timestamp=test_time
        )

        # 验证组合记录
        portfolio = backtest_monitor.get_portfolio_history()[0]
        assert portfolio['value'] == 1000000
        assert portfolio['tags']['positions'] == "2"

        # 验证持仓详情
        position_metrics = backtest_monitor.get_custom_metrics(name='position_detail')
        assert len(position_metrics) == 2
        assert any(m['tags']['symbol'] == "600000.SH" and m['value'] == 1000 for m in position_metrics)
        assert any(m['tags']['symbol'] == "000001.SZ" and m['value'] == 500 for m in position_metrics)

    def test_record_performance(self, backtest_monitor):
        """测试记录策略表现"""
        test_time = datetime(2023, 1, 1)

        # 记录表现指标
        backtest_monitor.record_performance(
            returns=0.15,
            volatility=0.2,
            sharpe=1.2,
            max_drawdown=0.1,
            strategy="momentum",
            timestamp=test_time
        )

        # 验证指标记录
        metrics = backtest_monitor.get_performance_metrics()
        assert len(metrics['returns']) == 1
        assert metrics['returns'][0]['value'] == 0.15
        assert metrics['volatility'][0]['value'] == 0.2
        assert metrics['sharpe'][0]['value'] == 1.2
        assert metrics['max_drawdown'][0]['value'] == 0.1

    def test_filter_trades(self, backtest_monitor):
        """测试交易记录过滤"""
        base_time = datetime(2023, 1, 1)

        # 记录不同策略的交易
        backtest_monitor.record_trade(
            symbol="600000.SH",
            action="BUY",
            price=15.0,
            quantity=1000,
            strategy="momentum",
            timestamp=base_time
        )

        backtest_monitor.record_trade(
            symbol="000001.SZ",
            action="BUY",
            price=20.0,
            quantity=500,
            strategy="mean_reversion",
            timestamp=base_time + timedelta(days=1)
        )

        # 按策略过滤
        momentum_trades = backtest_monitor.get_trade_history(strategy="momentum")
        assert len(momentum_trades) == 1
        assert momentum_trades[0]['tags']['strategy'] == "momentum"

        # 按标的过滤
        sz_trades = backtest_monitor.get_trade_history(symbol="000001.SZ")
        assert len(sz_trades) == 1
        assert sz_trades[0]['tags']['symbol'] == "000001.SZ"

        # 按时间过滤
        start_time = base_time + timedelta(hours=12)
        filtered_trades = backtest_monitor.get_trade_history(start_time=start_time)
        assert len(filtered_trades) == 1
        assert filtered_trades[0]['tags']['strategy'] == "mean_reversion"

    def test_empty_metrics(self, backtest_monitor):
        """测试空指标查询"""
        assert not backtest_monitor.get_trade_history()
        assert not backtest_monitor.get_portfolio_history()
        assert not backtest_monitor.get_performance_metrics()

    def test_partial_performance(self, backtest_monitor):
        """测试部分表现指标记录"""
        # 只记录收益率和夏普比率
        backtest_monitor.record_performance(
            returns=0.1,
            sharpe=1.0,
            strategy="momentum"
        )

        metrics = backtest_monitor.get_performance_metrics()
        assert metrics['returns']
        assert metrics['sharpe']
        assert 'volatility' not in metrics
        assert 'max_drawdown' not in metrics

    def test_multiple_strategies(self, backtest_monitor):
        """测试多策略数据隔离"""
        # 记录不同策略的数据
        backtest_monitor.record_trade(
            symbol="600000.SH",
            action="BUY",
            price=15.0,
            quantity=1000,
            strategy="strategy_a"
        )

        backtest_monitor.record_portfolio(
            value=1000000,
            cash=200000,
            positions={"600000.SH": 1000},
            strategy="strategy_b"
        )

        # 验证数据隔离
        assert len(backtest_monitor.get_trade_history(strategy="strategy_a")) == 1
        assert not backtest_monitor.get_trade_history(strategy="strategy_b")

        assert len(backtest_monitor.get_portfolio_history(strategy="strategy_b")) == 1
        assert not backtest_monitor.get_portfolio_history(strategy="strategy_a")
