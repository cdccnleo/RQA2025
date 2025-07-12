import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import logging
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class EnhancedBacktestEngine:
    """增强的回测引擎"""

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.results = {}
        self._prepare_data()

    def _prepare_data(self):
        """数据预处理"""
        # 添加波动率指标
        self.data['volatility'] = self.data['close'].rolling(20).std()

        # 添加交易量异常标记
        mean_volume = self.data['volume'].rolling(20).mean()
        std_volume = self.data['volume'].rolling(20).std()
        self.data['volume_anomaly'] = (
            (self.data['volume'] > mean_volume + 2*std_volume) |
            (self.data['volume'] < mean_volume - 2*std_volume)
        )

        # 添加市场状态标记
        self.data['market_state'] = self._detect_market_state()

    def _detect_market_state(self) -> pd.Series:
        """检测市场状态"""
        returns = self.data['close'].pct_change()

        # 简单市场状态分类
        conditions = [
            (returns.rolling(5).std() > 0.015),  # 高波动
            (returns.rolling(5).mean() > 0.002),  # 上涨
            (returns.rolling(5).mean() < -0.002), # 下跌
            (self.data['volume'] > self.data['volume'].rolling(20).mean() * 1.5) # 放量
        ]
        choices = ['high_vol', 'up_trend', 'down_trend', 'high_volume']

        state = np.select(conditions, choices, default='normal')
        return pd.Series(state, index=self.data.index)

    def run_strategy(self, strategy_class: type, **params) -> Dict:
        """运行策略回测"""
        bt = Backtest(
            self.data,
            strategy_class,
            commission=0.001,  # 0.1%佣金
            margin=0.1,        # 10%保证金
            trade_on_close=True
        )

        # 优化参数
        if params.get('optimize'):
            opt_results = self._optimize_params(bt, strategy_class, params)
            return opt_results

        # 普通回测
        results = bt.run(**params)
        self.results[strategy_class.__name__] = results
        return results

    def _optimize_params(self, bt: Backtest, strategy_class: type, params: Dict) -> Dict:
        """参数优化"""
        optimize_params = params.get('optimize_params', {})
        constraint = params.get('constraint', lambda p: True)

        logger.info(f"Optimizing {strategy_class.__name__} with params: {optimize_params}")

        stats = bt.optimize(
            **optimize_params,
            constraint=constraint,
            maximize='Sharpe Ratio',
            return_heatmap=True
        )

        return {
            'best_params': stats._strategy._params,
            'performance': stats,
            'heatmap': stats._heatmap
        }

    def monte_carlo_test(self, strategy_class: type, n_runs: int = 100) -> Dict:
        """蒙特卡洛测试"""
        logger.info(f"Running Monte Carlo test with {n_runs} iterations")

        results = []
        for i in range(n_runs):
            # 随机采样数据
            sample = self.data.sample(frac=0.8, replace=True).sort_index()

            bt = Backtest(
                sample,
                strategy_class,
                commission=0.001,
                margin=0.1
            )

            result = bt.run()
            results.append({
                'return': result['Return [%]'],
                'sharpe': result['Sharpe Ratio'],
                'max_dd': result['Max. Drawdown [%]']
            })

        # 计算统计指标
        df = pd.DataFrame(results)
        stats = {
            'avg_return': df['return'].mean(),
            'std_return': df['return'].std(),
            'win_rate': (df['return'] > 0).mean(),
            'avg_sharpe': df['sharpe'].mean(),
            'max_dd_dist': df['max_dd'].describe()
        }

        return stats

    def sensitivity_analysis(self, strategy_class: type, param_ranges: Dict) -> pd.DataFrame:
        """参数敏感性分析"""
        results = []

        for param, values in param_ranges.items():
            for value in values:
                bt = Backtest(
                    self.data,
                    strategy_class,
                    commission=0.001,
                    margin=0.1
                )

                result = bt.run(**{param: value})
                results.append({
                    'param': param,
                    'value': value,
                    'return': result['Return [%]'],
                    'sharpe': result['Sharpe Ratio'],
                    'max_dd': result['Max. Drawdown [%]']
                })

        return pd.DataFrame(results)

    def plot_results(self, strategy_name: str):
        """可视化回测结果"""
        if strategy_name not in self.results:
            raise ValueError(f"No results for strategy {strategy_name}")

        result = self.results[strategy_name]

        # 创建绘图
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))

        # 资金曲线
        axes[0].set_title('Equity Curve')
        axes[0].plot(result._equity_curve['Equity'])
        axes[0].set_ylabel('Equity')

        # 回撤曲线
        axes[1].set_title('Drawdown')
        axes[1].plot(result._equity_curve['DrawdownPct'])
        axes[1].set_ylabel('Drawdown %')

        # 每日收益率分布
        axes[2].set_title('Daily Returns Distribution')
        axes[2].hist(result._equity_curve['ReturnPct'], bins=50)
        axes[2].set_xlabel('Daily Return %')

        plt.tight_layout()
        plt.savefig(f'{strategy_name}_backtest.png')
        logger.info(f"Saved backtest plot to {strategy_name}_backtest.png")

class LiveTradingValidator:
    """实盘验证工具"""

    def __init__(self, strategy, broker):
        self.strategy = strategy
        self.broker = broker
        self.performance = []

    def run_simulation(self, test_data: pd.DataFrame):
        """运行模拟交易"""
        logger.info("Running live simulation")

        for i, row in test_data.iterrows():
            # 获取当前市场状态
            market_state = self._get_market_state(row)

            # 生成信号
            signal = self.strategy.generate_signal(row, market_state)

            # 执行交易
            if signal:
                self._execute_trade(signal, row)

            # 记录表现
            self._record_performance(row)

    def _get_market_state(self, data: pd.Series) -> Dict:
        """获取当前市场状态"""
        return {
            'volatility': data.get('volatility', 0),
            'trend': data.get('trend', 'neutral'),
            'liquidity': data.get('liquidity', 1)
        }

    def _execute_trade(self, signal: Dict, data: pd.Series):
        """执行交易"""
        try:
            order = self.broker.create_order(
                symbol=data['symbol'],
                price=data['close'],
                amount=signal['amount'],
                side=signal['side'],
                order_type='limit'
            )

            logger.info(f"Executed {signal['side']} order at {data['close']}")
            return order
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return None

    def _record_performance(self, data: pd.Series):
        """记录表现指标"""
        self.performance.append({
            'timestamp': data.name,
            'price': data['close'],
            'position': self.strategy.current_position,
            'equity': self._calculate_equity(data['close'])
        })

    def _calculate_equity(self, current_price: float) -> float:
        """计算当前权益"""
        if not hasattr(self.strategy, 'initial_capital'):
            return 0

        position_value = self.strategy.current_position * current_price
        return self.strategy.initial_capital + position_value

    def compare_with_backtest(self, backtest_results: Dict) -> Dict:
        """与回测结果对比"""
        live_returns = pd.Series([p['equity'] for p in self.performance]).pct_change().dropna()
        backtest_returns = backtest_results['Return']

        comparison = {
            'live_avg_return': live_returns.mean(),
            'backtest_avg_return': backtest_returns.mean(),
            'return_diff': live_returns.mean() - backtest_returns.mean(),
            'live_volatility': live_returns.std(),
            'backtest_volatility': backtest_returns.std(),
            'correlation': live_returns.corr(backtest_returns)
        }

        return comparison

def main():
    """主测试流程"""
    # 加载测试数据
    data = pd.read_csv('data/backtest_data.csv', index_col=0, parse_dates=True)

    # 初始化回测引擎
    engine = EnhancedBacktestEngine(data)

    # 定义测试策略
    class TestStrategy(Strategy):
        def init(self):
            pass

        def next(self):
            if crossover(self.data.Close, self.data.Close.rolling(20).mean()):
                self.buy()
            elif crossover(self.data.Close.rolling(20).mean(), self.data.Close):
                self.sell()

    # 运行回测
    results = engine.run_strategy(TestStrategy)
    logger.info(f"Backtest results: {results}")

    # 参数优化
    opt_params = {
        'optimize': True,
        'optimize_params': {
            'n1': range(5, 50, 5),
            'n2': range(10, 100, 10)
        },
        'constraint': lambda p: p.n1 < p.n2
    }
    opt_results = engine.run_strategy(TestStrategy, **opt_params)
    logger.info(f"Optimization results: {opt_results['best_params']}")

    # 蒙特卡洛测试
    mc_results = engine.monte_carlo_test(TestStrategy)
    logger.info(f"Monte Carlo results: {mc_results}")

    # 敏感性分析
    sensitivity_params = {
        'n1': [5, 10, 15, 20],
        'n2': [30, 40, 50, 60]
    }
    sensitivity_results = engine.sensitivity_analysis(TestStrategy, sensitivity_params)
    sensitivity_results.to_csv('sensitivity_results.csv')
    logger.info("Saved sensitivity results")

    # 可视化结果
    engine.plot_results('TestStrategy')

if __name__ == "__main__":
    main()
