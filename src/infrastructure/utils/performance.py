"""
性能分析工具模块
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from decimal import Decimal, getcontext

class PerformanceUtils:
    """性能分析工具类"""

    # 设置Decimal精度
    getcontext().prec = 8

    # A股交易成本参数
    A_SHARE_FEES = {
        'commission': Decimal('0.00025'),  # 佣金0.025%
        'stamp_duty': Decimal('0.001'),    # 印花税0.1% (卖出收取)
        'transfer': Decimal('0.00002')     # 过户费0.002%
    }

    @staticmethod
    def calculate_returns(positions: pd.DataFrame,
                         initial_capital: float,
                         include_fees: bool = True) -> pd.DataFrame:
        """
        计算持仓收益率
        Args:
            positions: 包含['date','symbol','quantity','price','action']的DataFrame
            initial_capital: 初始资金
            include_fees: 是否包含交易费用
        Returns:
            包含收益率和累计净值等指标的DataFrame
        """
        # 复制数据避免修改原始数据
        df = positions.copy()

        # 计算每笔交易金额
        df['amount'] = df['quantity'] * df['price']

        # 计算交易费用
        if include_fees:
            df['fees'] = df.apply(PerformanceUtils._calculate_fees, axis=1)
        else:
            df['fees'] = 0.0

        # 计算净现金流
        df['net_cashflow'] = np.where(
            df['action'] == 'buy',
            -(df['amount'] + df['fees']),
            df['amount'] - df['fees']
        )

        # 按日期分组计算每日净现金流
        daily_cashflow = df.groupby('date')['net_cashflow'].sum()
        daily_cashflow = daily_cashflow.reindex(
            pd.date_range(
                start=df['date'].min(),
                end=df['date'].max()
            ),
            fill_value=0.0
        )

        # 计算每日持仓价值
        daily_value = PerformanceUtils._calculate_daily_value(df, initial_capital)

        # 合并现金流和持仓价值
        performance = pd.DataFrame({
            'cashflow': daily_cashflow,
            'value': daily_value
        })
        performance['net_value'] = performance['value'] + performance['cashflow'].cumsum()

        # 计算收益率
        performance['daily_return'] = performance['net_value'].pct_change()
        performance['cum_return'] = (1 + performance['daily_return']).cumprod() - 1

        return performance

    @staticmethod
    def _calculate_fees(row: pd.Series) -> Decimal:
        """计算A股交易费用"""
        amount = Decimal(str(row['price'])) * Decimal(str(row['quantity']))

        # 佣金(双向收取)
        commission = amount * PerformanceUtils.A_SHARE_FEES['commission']

        # 印花税(卖出收取)
        stamp_duty = Decimal(0)
        if row['action'] == 'sell':
            stamp_duty = amount * PerformanceUtils.A_SHARE_FEES['stamp_duty']

        # 过户费(双向收取)
        transfer = amount * PerformanceUtils.A_SHARE_FEES['transfer']

        return commission + stamp_duty + transfer

    @staticmethod
    def _calculate_daily_value(df: pd.DataFrame, initial_capital: float) -> pd.Series:
        """计算每日持仓价值"""
        # 获取持仓变化
        holdings = df.pivot_table(
            index='date',
            columns='symbol',
            values='quantity',
            aggfunc='sum'
        ).fillna(0).cumsum()

        # 获取每日价格
        prices = df.pivot_table(
            index='date',
            columns='symbol',
            values='price',
            aggfunc='last'
        ).ffill()

        # 计算每日持仓价值
        daily_value = (holdings * prices).sum(axis=1)
        daily_value = daily_value.reindex(
            pd.date_range(
                start=df['date'].min(),
                end=df['date'].max()
            ),
            method='ffill'
        ).fillna(0)
        daily_value.iloc[0] += initial_capital

        return daily_value

    @staticmethod
    def calculate_risk_metrics(returns: pd.Series,
                             risk_free_rate: float = 0.03,
                             periods: int = 252) -> Dict[str, float]:
        """
        计算风险调整后收益指标
        Args:
            returns: 日收益率序列
            risk_free_rate: 无风险利率(年化)
            periods: 年化周期数
        Returns:
            包含各项风险指标的字典
        """
        if returns.empty:
            return {}

        # 移除NA值
        returns = returns.dropna()

        # 计算基本指标
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (periods / len(returns)) - 1
        annualized_vol = returns.std() * np.sqrt(periods)

        # 计算最大回撤
        cum_returns = (1 + returns).cumprod()
        peak = cum_returns.cummax()
        drawdown = (cum_returns - peak) / peak
        max_drawdown = drawdown.min()

        # 计算夏普比率
        excess_return = annualized_return - risk_free_rate
        sharpe_ratio = excess_return / annualized_vol if annualized_vol != 0 else 0.0

        # 计算胜率和盈亏比
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0.0
        avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0.0
        avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0.0
        profit_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0

        return {
            'total_return': float(total_return),
            'annualized_return': float(annualized_return),
            'annualized_volatility': float(annualized_vol),
            'max_drawdown': float(max_drawdown),
            'sharpe_ratio': float(sharpe_ratio),
            'win_rate': float(win_rate),
            'profit_ratio': float(profit_ratio),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss)
        }

    @staticmethod
    def analyze_strategy(positions: pd.DataFrame,
                        initial_capital: float,
                        benchmark: Optional[pd.Series] = None) -> Dict:
        """
        全面分析策略表现
        Args:
            positions: 交易记录DataFrame
            initial_capital: 初始资金
            benchmark: 基准收益率序列
        Returns:
            包含完整分析结果的字典
        """
        # 计算收益率
        performance = PerformanceUtils.calculate_returns(positions, initial_capital)

        # 计算风险指标
        metrics = PerformanceUtils.calculate_risk_metrics(performance['daily_return'])

        # 分析交易行为
        trades = PerformanceUtils.analyze_trades(positions)

        # 如果有基准，计算超额收益
        alpha = None
        if benchmark is not None:
            alpha = PerformanceUtils.calculate_alpha(
                performance['daily_return'],
                benchmark
            )
            metrics.update(alpha)

        return {
            'performance': performance,
            'metrics': metrics,
            'trades': trades,
            'alpha': alpha
        }

    @staticmethod
    def analyze_trades(positions: pd.DataFrame) -> Dict:
        """
        分析交易行为
        Args:
            positions: 交易记录DataFrame
        Returns:
            交易行为分析结果
        """
        # 按标的分组分析
        symbol_stats = positions.groupby('symbol').apply(
            lambda x: pd.Series({
                'num_trades': len(x),
                'avg_holding_period': PerformanceUtils._calc_holding_period(x),
                'win_rate': PerformanceUtils._calc_win_rate(x),
                'avg_profit': PerformanceUtils._calc_avg_profit(x)
            })
        ).to_dict('index')

        # 总体交易统计
        total_stats = {
            'total_trades': len(positions),
            'long_trades': len(positions[positions['action'] == 'buy']),
            'short_trades': len(positions[positions['action'] == 'sell']),
            'avg_trade_duration': PerformanceUtils._calc_holding_period(positions)
        }

        return {
            'symbol_stats': symbol_stats,
            'total_stats': total_stats
        }

    @staticmethod
    def _calc_holding_period(trades: pd.DataFrame) -> float:
        """计算平均持有周期"""
        if len(trades) < 2:
            return 0.0

        # 假设交易是成对的买入和卖出
        buys = trades[trades['action'] == 'buy']
        sells = trades[trades['action'] == 'sell']

        if len(buys) == 0 or len(sells) == 0:
            return 0.0

        # 计算每笔买入到对应卖出的持有时间
        holding_days = []
        for symbol in buys['symbol'].unique():
            symbol_buys = buys[buys['symbol'] == symbol]
            symbol_sells = sells[sells['symbol'] == symbol]

            for _, buy in symbol_buys.iterrows():
                sell = symbol_sells[symbol_sells['date'] > buy['date']]
                if not sell.empty:
                    holding_days.append((sell.iloc[0]['date'] - buy['date']).days)

        return float(np.mean(holding_days)) if holding_days else 0.0

    @staticmethod
    def _calc_win_rate(trades: pd.DataFrame) -> float:
        """计算胜率"""
        if len(trades) < 2:
            return 0.0

        # 计算每笔交易的盈亏
        trades = trades.sort_values(['symbol', 'date'])
        trades['pnl'] = trades.groupby('symbol')['price'].diff()
        trades['return'] = trades['pnl'] / trades['price'].shift(1)

        # 只考虑卖出交易的盈亏
        sell_trades = trades[trades['action'] == 'sell']
        if len(sell_trades) == 0:
            return 0.0

        win_rate = len(sell_trades[sell_trades['return'] > 0]) / len(sell_trades)
        return float(win_rate)

    @staticmethod
    def _calc_avg_profit(trades: pd.DataFrame) -> float:
        """计算平均盈亏"""
        if len(trades) < 2:
            return 0.0

        trades = trades.sort_values(['symbol', 'date'])
        trades['pnl'] = trades.groupby('symbol')['price'].diff()

        sell_trades = trades[trades['action'] == 'sell']
        if len(sell_trades) == 0:
            return 0.0

        avg_profit = sell_trades['pnl'].mean()
        return float(avg_profit)

    @staticmethod
    def calculate_alpha(strategy_returns: pd.Series,
                       benchmark_returns: pd.Series) -> Dict[str, float]:
        """
        计算超额收益(Alpha)
        Args:
            strategy_returns: 策略收益率序列
            benchmark_returns: 基准收益率序列
        Returns:
            包含Alpha和Beta等指标的字典
        """
        # 对齐数据
        aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
        if len(aligned) < 2:
            return {}

        strategy = aligned.iloc[:, 0]
        benchmark = aligned.iloc[:, 1]

        # 计算Beta
        cov = np.cov(strategy, benchmark)
        beta = cov[0, 1] / cov[1, 1]

        # 计算Alpha
        alpha = strategy.mean() - beta * benchmark.mean()

        # 年化Alpha
        annualized_alpha = (1 + alpha) ** 252 - 1

        return {
            'alpha': float(alpha),
            'beta': float(beta),
            'annualized_alpha': float(annualized_alpha)
        }

    @staticmethod
    def analyze_margin_trading(margin_data: pd.DataFrame) -> Dict:
        """
        分析融资融券策略表现
        Args:
            margin_data: 包含融资融券交易记录的DataFrame
        Returns:
            融资融券策略分析结果
        """
        # 计算杠杆率
        margin_data['leverage'] = margin_data['margin_balance'] / margin_data['total_assets']

        # 计算担保比例
        margin_data['collateral_ratio'] = (
            margin_data['total_assets'] / margin_data['margin_balance']
        )

        # 计算风险指标
        risk_metrics = {
            'avg_leverage': float(margin_data['leverage'].mean()),
            'min_collateral_ratio': float(margin_data['collateral_ratio'].min()),
            'margin_call_times': int(
                (margin_data['collateral_ratio'] < 1.3).sum()
            )
        }

        return risk_metrics

    @staticmethod
    def evaluate_limit_impact(positions: pd.DataFrame,
                            limit_data: pd.DataFrame) -> Dict:
        """
        评估涨跌停板对策略的影响
        Args:
            positions: 交易记录DataFrame
            limit_data: 包含每日涨跌停价格的DataFrame
        Returns:
            涨跌停影响评估结果
        """
        # 合并交易数据和涨跌停数据
        merged = pd.merge(
            positions,
            limit_data,
            on=['date', 'symbol'],
            how='left'
        )

        # 计算涨跌停影响
        buy_impact = merged[
            (merged['action'] == 'buy') &
            (merged['price'] >= merged['upper_limit'])
        ]
        sell_impact = merged[
            (merged['action'] == 'sell') &
            (merged['price'] <= merged['lower_limit'])
        ]

        return {
            'missed_buys': len(buy_impact),
            'missed_sells': len(sell_impact),
            'total_impact': len(buy_impact) + len(sell_impact),
            'impact_ratio': (len(buy_impact) + len(sell_impact)) / len(positions)
        }
