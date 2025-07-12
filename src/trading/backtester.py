# trading/backtester.py
from src.trading.strategies.factory import StrategyFactory
from typing import Dict
import pandas as pd
from backtrader import Analyzer
import backtrader as bt
import matplotlib.pyplot as plt
import seaborn as sns
from src.infrastructure.utils.logger import get_logger

logger = get_logger(__name__)  # 自动继承全局配置


class BacktestEngine:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.analyzer = BacktestAnalyzer()

    def _load_config(self, path: str) -> Dict:
        # 加载配置文件
        pass

    def run(
            self,
            strategy: str,
            data: pd.DataFrame,
            portfolio_params: Dict
    ) -> Dict:
        """执行回测"""
        # 动态加载策略
        strategy_class = StrategyFactory.create(strategy)
        # 初始化回测分析器
        results = self.analyzer.run_backtest(
            data=data,
            strategy=strategy_class,
            params=portfolio_params
        )
        # 生成报告
        report = self.analyze(results)
        return report

    def analyze(self, results) -> dict:
        """多维度绩效分析"""
        return {
            "risk_metrics": self.analyzer.calculate_risk(results),
            "performance": self.analyzer.evaluate_performance(results),
            "visualization": self.analyzer.generate_report(results)
        }

    def generate_report(results, format='html'):
        pass

    def apply_price_limit(data, limit=0.1):
        data['price_limit_up'] = data['close'].shift(1) * (1 + limit)
        data['price_limit_down'] = data['close'].shift(1) * (1 - limit)
        data['close'] = np.where(data['close'] > data['price_limit_up'], data['price_limit_up'], data['close'])
        data['close'] = np.where(data['close'] < data['price_limit_down'], data['price_limit_down'], data['close'])
        return data

    def apply_t_plus_1_rule(trades):
        trades['execution_date'] = trades['entry_date'].shift(-1)
        return trades.dropna(subset=['execution_date'])

    def forward_backward_adjustment(data, split_date):
        # 前向调整
        data['forward_adjusted'] = data['close'] / data.loc[data.index <= split_date, 'close'].iloc[-1] * \
                                   data.loc[data.index <= split_date, 'close'].iloc[0]

        # 后向调整
        data['backward_adjusted'] = data['close'] * data.loc[data.index > split_date, 'close'].iloc[0] / \
                                    data.loc[data.index > split_date, 'close'].iloc[-1]

        return data

    def information_barrier(features, target, info_date):
        # 确保特征不包含未来信息
        clean_features = features[features.index <= info_date]
        clean_target = target[target.index <= info_date]
        return clean_features, clean_target


class BacktestAnalyzer(Analyzer):
    def __init__(self):
        super().__init__()
        self.reports = []
        self.state_labels = {1: "牛市", 0: "震荡市", -1: "熊市"}
        self.data = None

        # 添加分析器
        self.analyzers = {
            'sharpe': bt.analyzers.SharpeRatio(),
            'drawdown': bt.analyzers.DrawDown(),
            'returns': bt.analyzers.Returns(),
            'trade': bt.analyzers.TradeAnalyzer(),
            'timereturn': bt.analyzers.TimeReturn(),
            'sqn': bt.analyzers.SQN()  # 确保添加 SQNAnalyzer
        }

        # 显式设置单独的分析器属性
        self.sharpe_analyzer = self.analyzers['sharpe']
        self.drawdown_analyzer = self.analyzers['drawdown']
        self.returns_analyzer = self.analyzers['returns']
        self.trade_analyzer = self.analyzers['trade']
        self.timereturn_analyzer = self.analyzers['timereturn']
        self.sqn_analyzer = self.analyzers['sqn']  # 添加 SQNAnalyzer 属性

    def start(self):
        if not self.datas or len(self.datas[0]) == 0:
            logger.warning("策略数据为空，跳过初始化")
            return  # 跳过初始化而不是抛出异常
        self.data = self.datas[0]

        # 确保数据中包含 market_state 列
        if not hasattr(self.data, 'market_state'):
            logger.warning("Market state data not found, using default value.")
            self.data.market_state = 0

    def _get_sqn(self) -> float:
        analysis = self.sqn_analyzer.get_analysis()
        return analysis.get('sqn', 0)

    def get_analysis(self) -> Dict:
        """返回分析结果摘要
        Returns:
            Dict: 包含三大类分析指标的结果字典
        """
        return {
            'performance_metrics': self._calculate_performance_metrics(),
            'risk_metrics': self._calculate_risk_metrics(),
            'market_analysis': self._analyze_market_states()
        }

    def _parse_trade_analysis(self) -> pd.DataFrame:
        try:
            trades = self.trade_analyzer.get_analysis().get('trades', [])
            if not trades:
                return pd.DataFrame(columns=['pnl', 'return_pct', 'market_state', 'duration'])

            trade_list = []
            for trade in trades:
                trade_info = {
                    'pnl': trade.pnl,
                    'return_pct': trade.return_pct,
                    'market_state': getattr(trade, 'market_state', 0),
                    'duration': getattr(trade, 'duration', 0)
                }
                trade_list.append(trade_info)
            return pd.DataFrame(trade_list)
        except Exception as e:
            logger.error(f"解析交易记录失败: {str(e)}")
            return pd.DataFrame(columns=['pnl', 'return_pct', 'market_state', 'duration'])

    def _calculate_performance_metrics(self) -> Dict:
        """计算关键绩效指标
        Returns:
            Dict: 包含夏普比率、年化收益等指标的字典
        """
        return {
            'sharpe_ratio': self._get_sharpe(),
            'annual_return': self._get_annual_return(),
            'max_drawdown': self._get_max_drawdown(),
            'win_rate': self._get_win_rate(),
            'profit_factor': self._get_profit_factor(),
            'sqn': self._get_sqn()
        }

    def _calculate_risk_metrics(self) -> Dict:
        """计算风险指标
        Returns:
            Dict: 包含波动率、VaR等风险指标的字典
        """
        return {
            'volatility': self._get_volatility(),
            'value_at_risk': self._get_var(),
            'expected_shortfall': self._get_es(),
            'tail_risk': self._get_tail_risk()
        }

    def _analyze_market_states(self) -> Dict:
        """分析不同市场状态下的表现"""
        return {
            'bull_market': self._get_state_performance(1),
            'bear_market': self._get_state_performance(-1),
            'volatile_market': self._get_state_performance(0)
        }

    def generate_report(self, strategy_name: str) -> None:
        """生成完整分析报告"""
        report = {
            'strategy': strategy_name,
            **self.get_analysis(),
            'trades': self._get_trade_details(),
            'equity_curve': self._get_equity_data()
        }
        self.reports.append(report)
        self._save_report(report)
        self._visualize_results()

    def _save_report(self, report: Dict) -> None:
        """保存报告到文件"""
        pd.DataFrame([report]).to_csv(f"{report['strategy']}_report.csv", index=False)
        logger.info(f"{report['strategy']} 分析报告已保存")

    def _visualize_results(self) -> None:
        """可视化分析结果"""
        self._plot_correlation_heatmap()
        self._plot_equity_curve()
        self._plot_drawdown()
        plt.close('all')

    def _plot_correlation_heatmap(self) -> None:
        """绘制指标相关性热力图"""
        plt.figure(figsize=(12, 8))
        sns.heatmap(pd.DataFrame(self.reports).corr(), annot=True, cmap='coolwarm')
        plt.title('策略指标相关性分析')
        plt.savefig('metrics_correlation.png')

    def _plot_equity_curve(self) -> None:
        """绘制资金曲线"""
        plt.figure(figsize=(12, 6))
        for report in self.reports:
            plt.plot(report['equity_curve'], label=report['strategy'])
        plt.legend()
        plt.title('资金曲线对比')
        plt.savefig('equity_curve_comparison.png')

    def _plot_drawdown(self) -> None:
        """绘制回撤曲线"""
        plt.figure(figsize=(12, 6))
        for report in self.reports:
            plt.plot(report['max_drawdown'], label=report['strategy'])
        plt.legend()
        plt.title('最大回撤对比')
        plt.savefig('drawdown_comparison.png')

    def _get_trade_details(self) -> pd.DataFrame:
        return self._parse_trade_analysis()

    # 指标计算具体实现
    def _get_sharpe(self) -> float:
        analysis = self.sharpe_analyzer.get_analysis()
        return analysis.get('sharperatio', 0.0) if analysis else 0.0

    def _get_annual_return(self) -> float:
        return self.returns_analyzer.get_analysis().get('rnorm100', 0) if self.returns_analyzer.get_analysis() else 0

    def _get_max_drawdown(self) -> float:
        analysis = self.drawdown_analyzer.get_analysis()
        return analysis.get('max', {}).get('drawdown', 0.0) if analysis else 0.0

    def _get_win_rate(self) -> float:
        trades = self._get_valid_trades()
        return len(trades[trades['pnl'] > 0]) / len(trades) if len(trades) > 0 else 0.0

    def _get_profit_factor(self) -> float:
        trades = self._get_valid_trades()
        wins = trades[trades['pnl'] > 0]['pnl'].sum()
        losses = abs(trades[trades['pnl'] < 0]['pnl'].sum())
        return wins / losses if losses != 0 else 0.0

    def _get_volatility(self) -> float:
        returns = self._get_returns_series()
        return returns.std() * (252 ** 0.5) if not returns.empty else 0.0

    def _get_var(self, confidence_level: float = 0.95) -> float:
        returns = self._get_returns_series()
        return returns.quantile(1 - confidence_level)

    def _get_es(self, confidence_level: float = 0.95) -> float:
        returns = self._get_returns_series()
        var = self._get_var(confidence_level)
        return returns[returns <= var].mean()

    def _get_tail_risk(self) -> float:
        return self._get_es() - self._get_var()

    def _get_state_performance(self, state: int) -> Dict:
        # 确保数据是DataFrame
        if not isinstance(self.data, pd.DataFrame) or self.data.empty:
            return {
                'trade_count': 0,
                'win_rate': 0.0,
                'avg_return': 0.0,
                'max_drawdown': 0.0
            }

        trades = self._get_valid_trades()
        state_trades = trades[trades['market_state'] == state]

        # 添加调试日志
        logger.debug(f"State: {state}, State Trades: {state_trades}")

        return {
            'trade_count': len(state_trades),
            'win_rate': self._calc_win_rate(state_trades),
            'avg_return': state_trades['return_pct'].mean(),
            'max_drawdown': self._calc_state_drawdown(state)
        }

    def _get_valid_trades(self) -> pd.DataFrame:
        trades = self._parse_trade_analysis()
        if trades.empty:
            return pd.DataFrame(columns=['pnl', 'return_pct', 'market_state', 'duration'])
        return trades[trades['duration'] > 0]

    def _get_returns_series(self) -> pd.Series:
        trades = self._get_valid_trades()
        if trades.empty:
            return pd.Series(dtype=float)
        return trades['return_pct']

    def _calc_win_rate(self, trades: pd.DataFrame) -> float:
        return len(trades[trades['pnl'] > 0]) / len(trades) if len(trades) > 0 else 0

    def _calc_state_drawdown(self, state: int) -> float:
        # 确保 self.data 是 pandas.DataFrame
        if not isinstance(self.data, pd.DataFrame):
            return 0.0  # 或者抛出异常

        state_data = self.data[self.data['market_state'] == state]
        return (state_data['close'].max() - state_data['close'].min()) / state_data['close'].max()

    def _get_equity_data(self) -> pd.Series:
        if not hasattr(self.data, 'datetime') or len(self.data) == 0:
            return pd.Series()  # 返回空序列避免错误

        # 确保 datetime 数据存在
        dates = [self.data.datetime.date(i) for i in range(len(self.data))]
        equity = [self.strategy.broker.getvalue() for _ in dates]
        return pd.Series(equity, index=pd.to_datetime(dates))

    def assess_strategy_capacity(self, trade_data: pd.DataFrame) -> Dict:
        """评估策略容量

        Args:
            trade_data: 包含交易数据的DataFrame，需包含'size'和'price'列

        Returns:
            包含策略容量评估指标的字典
        """
        if 'size' not in trade_data.columns or 'price' not in trade_data.columns:
            raise ValueError("交易数据必须包含'size'和'price'列")

        # 计算交易规模分布
        size_stats = trade_data['size'].describe()

        # 计算交易对市场价格的影响
        trade_data['price_impact'] = trade_data.groupby('date')['size'].transform(
            lambda x: x / x.rolling(20, min_periods=1).mean()
        )

        # 计算市场深度指标
        market_depth = trade_data.groupby('date').apply(
            lambda x: self._calculate_market_depth(x['size'], x['price'])
        ).mean()

        # 计算大额订单比例
        large_orders = trade_data[trade_data['size'] > size_stats['75%']]
        large_order_ratio = len(large_orders) / len(trade_data)

        # 计算交易容量指标
        daily_volume = trade_data.groupby('date')['size'].sum().mean()
        strategy_capacity = daily_volume * 0.1  # 策略容量为日均交易量的10%

        return {
            'size_statistics': size_stats,
            'average_price_impact': trade_data['price_impact'].mean(),
            'market_depth': market_depth,
            'large_order_ratio': large_order_ratio,
            'estimated_strategy_capacity': strategy_capacity
        }

    def _calculate_market_depth(self, sizes, prices):
        """计算市场深度"""
        # 简化版市场深度计算：交易量与价格变化的比率
        price_changes = prices.diff().fillna(0)
        return sizes / (price_changes.abs() + 1e-8)  # 防止除零
