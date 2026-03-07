#!/usr/bin/env python3
"""
回测集成框架
将量化模型测试与回测框架集成
验证模型在历史数据上的表现一致性
确保测试覆盖各种市场情景
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import logging
from dataclasses import dataclass
from decimal import getcontext

# 设置高精度计算
getcontext().prec = 28

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class BacktestResult:
    """回测结果数据类"""
    model_name: str
    start_date: str
    end_date: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_trade_return: float
    volatility: float
    calmar_ratio: float
    sortino_ratio: float
    benchmark_return: float
    excess_return: float
    information_ratio: float
    test_scenarios: List[str]
    consistency_score: float
    risk_metrics: Dict[str, float]


class BacktestIntegrationFramework:
    """回测集成框架"""

    def __init__(self, data_dir: str = "data/historical", output_dir: str = "reports/backtest"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()

        # 市场情景定义
        self.market_scenarios = {
            'bull_market': {
                'description': '牛市情景',
                'conditions': ['trending_up', 'low_volatility', 'high_volume'],
                'expected_performance': 'high_returns'
            },
            'bear_market': {
                'description': '熊市情景',
                'conditions': ['trending_down', 'high_volatility', 'low_volume'],
                'expected_performance': 'defensive_returns'
            },
            'sideways_market': {
                'description': '震荡市情景',
                'conditions': ['no_trend', 'medium_volatility', 'medium_volume'],
                'expected_performance': 'moderate_returns'
            },
            'high_volatility': {
                'description': '高波动情景',
                'conditions': ['high_volatility', 'trend_breaks', 'gaps'],
                'expected_performance': 'risk_managed_returns'
            },
            'low_volatility': {
                'description': '低波动情景',
                'conditions': ['low_volatility', 'stable_trend', 'predictable'],
                'expected_performance': 'consistent_returns'
            },
            'crisis_market': {
                'description': '危机市场情景',
                'conditions': ['extreme_volatility', 'panic_selling', 'liquidity_crisis'],
                'expected_performance': 'capital_preservation'
            },
            'recovery_market': {
                'description': '复苏市场情景',
                'conditions': ['bounce_back', 'increasing_volume', 'trend_reversal'],
                'expected_performance': 'opportunistic_returns'
            }
        }

    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def generate_historical_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """生成历史数据用于回测"""
        self.logger.info(f"生成历史数据: {symbols}")

        historical_data = {}

        for symbol in symbols:
            # 生成模拟历史数据
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            n_days = len(dates)

            # 基础价格序列
            base_price = 100.0
            returns = np.random.normal(0.001, 0.02, n_days)  # 日收益率

            # 添加趋势和季节性
            trend = np.linspace(0, 0.1, n_days)  # 轻微上升趋势
            seasonal = 0.01 * np.sin(2 * np.pi * np.arange(n_days) / 252)  # 年度季节性

            # 合成价格序列
            cumulative_returns = np.cumprod(1 + returns + trend + seasonal)
            prices = base_price * cumulative_returns

            # 生成OHLC数据
            high = prices * (1 + np.random.uniform(0, 0.02, n_days))
            low = prices * (1 - np.random.uniform(0, 0.02, n_days))
            close = prices
            open_price = np.roll(close, 1)
            open_price[0] = close[0]

            # 生成成交量
            volume = np.random.uniform(1000000, 10000000, n_days)

            # 创建DataFrame
            df = pd.DataFrame({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            }, index=dates)

            historical_data[symbol] = df

        return historical_data

    def create_market_scenario_data(self, scenario: str, base_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """根据市场情景调整数据"""
        self.logger.info(f"创建市场情景数据: {scenario}")

        scenario_data = {}
        scenario_config = self.market_scenarios[scenario]

        for symbol, df in base_data.items():
            # 复制基础数据
            scenario_df = df.copy()

            if scenario == 'bull_market':
                # 牛市：增加上升趋势和成交量
                trend_factor = 1.5
                volume_factor = 1.3
                scenario_df['close'] = df['close'] * (1 + np.linspace(0, 0.2, len(df)))
                scenario_df['volume'] = df['volume'] * volume_factor

            elif scenario == 'bear_market':
                # 熊市：下降趋势，高波动
                scenario_df['close'] = df['close'] * (1 - np.linspace(0, 0.3, len(df)))
                scenario_df['volume'] = df['volume'] * 0.8
                # 增加波动性
                volatility_factor = 1.5
                returns = scenario_df['close'].pct_change()
                scenario_df['close'] = scenario_df['close'].iloc[0] * \
                    (1 + returns * volatility_factor).cumprod()

            elif scenario == 'high_volatility':
                # 高波动：增加价格波动
                volatility_factor = 2.0
                returns = df['close'].pct_change()
                scenario_df['close'] = df['close'].iloc[0] * \
                    (1 + returns * volatility_factor).cumprod()
                scenario_df['volume'] = df['volume'] * 1.2

            elif scenario == 'crisis_market':
                # 危机市场：极端波动和恐慌性抛售
                crisis_returns = np.random.normal(-0.05, 0.08, len(df))
                scenario_df['close'] = df['close'].iloc[0] * (1 + crisis_returns).cumprod()
                scenario_df['volume'] = df['volume'] * 2.0  # 恐慌性交易量增加

            elif scenario == 'recovery_market':
                # 复苏市场：从低点反弹
                recovery_factor = np.linspace(0.8, 1.2, len(df))  # 从80%反弹到120%
                scenario_df['close'] = df['close'] * recovery_factor
                scenario_df['volume'] = df['volume'] * 1.1

            # 更新OHLC数据
            scenario_df['high'] = scenario_df['close'] * (1 + np.random.uniform(0, 0.02, len(df)))
            scenario_df['low'] = scenario_df['close'] * (1 - np.random.uniform(0, 0.02, len(df)))
            scenario_df['open'] = scenario_df['close'].shift(1)
            scenario_df['open'].iloc[0] = scenario_df['close'].iloc[0]

            scenario_data[symbol] = scenario_df

        return scenario_data

    def run_model_backtest(self, model_name: str, data: Dict[str, pd.DataFrame],
                           initial_capital: float = 100000.0) -> BacktestResult:
        """运行模型回测"""
        self.logger.info(f"运行模型回测: {model_name}")

        # 模拟模型预测和交易
        all_returns = []
        all_trades = []
        positions = {}
        cash = initial_capital
        portfolio_value = initial_capital

        symbols = list(data.keys())
        dates = list(data[symbols[0]].index)

        for i, date in enumerate(dates):
            daily_return = 0.0
            daily_trades = []

            for symbol in symbols:
                if symbol not in data:
                    continue

                current_price = data[symbol].loc[date, 'close']

                # 模拟模型预测
                if i > 0:  # 从第二天开始交易
                    # 简单的移动平均策略
                    ma_short = data[symbol]['close'].rolling(5).mean().iloc[i]
                    ma_long = data[symbol]['close'].rolling(20).mean().iloc[i]

                    # 交易信号
                    if ma_short > ma_long and symbol not in positions:
                        # 买入信号
                        position_size = cash * 0.1  # 使用10%资金
                        shares = int(position_size / current_price)
                        if shares > 0:
                            positions[symbol] = shares
                            cash -= shares * current_price
                            daily_trades.append({
                                'symbol': symbol,
                                'action': 'buy',
                                'shares': shares,
                                'price': current_price,
                                'value': shares * current_price
                            })

                    elif ma_short < ma_long and symbol in positions:
                        # 卖出信号
                        shares = positions[symbol]
                        cash += shares * current_price
                        daily_trades.append({
                            'symbol': symbol,
                            'action': 'sell',
                            'shares': shares,
                            'price': current_price,
                            'value': shares * current_price
                        })
                        del positions[symbol]

                # 计算持仓价值
                if symbol in positions:
                    portfolio_value += positions[symbol] * current_price

            # 计算当日收益率
            if i > 0:
                prev_value = portfolio_value - sum(trade['value'] for trade in daily_trades)
                if prev_value > 0:
                    daily_return = (portfolio_value - prev_value) / prev_value
                    all_returns.append(daily_return)

            all_trades.extend(daily_trades)

        # 计算回测指标
        returns_series = pd.Series(all_returns)

        # 基础指标
        total_return = (portfolio_value - initial_capital) / initial_capital
        sharpe_ratio = returns_series.mean() / returns_series.std() * \
            np.sqrt(252) if returns_series.std() > 0 else 0
        max_drawdown = self._calculate_max_drawdown(returns_series)
        win_rate = (returns_series > 0).mean() if len(returns_series) > 0 else 0
        total_trades = len(all_trades)
        avg_trade_return = returns_series.mean() if len(returns_series) > 0 else 0
        volatility = returns_series.std() * np.sqrt(252) if len(returns_series) > 0 else 0

        # 高级指标
        calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
        sortino_ratio = returns_series.mean() / returns_series[returns_series < 0].std(
        ) * np.sqrt(252) if len(returns_series[returns_series < 0]) > 0 else 0

        # 基准对比（假设基准为买入持有）
        benchmark_returns = []
        for symbol in symbols:
            if symbol in data:
                symbol_returns = data[symbol]['close'].pct_change().dropna()
                benchmark_returns.extend(symbol_returns.values)

        benchmark_return = np.mean(benchmark_returns) * 252 if benchmark_returns else 0
        excess_return = total_return - benchmark_return
        information_ratio = excess_return / volatility if volatility > 0 else 0

        # 一致性评分
        consistency_score = self._calculate_consistency_score(returns_series)

        # 风险指标
        risk_metrics = {
            'var_95': np.percentile(returns_series, 5) if len(returns_series) > 0 else 0,
            'cvar_95': returns_series[returns_series <= np.percentile(returns_series, 5)].mean() if len(returns_series) > 0 else 0,
            'skewness': returns_series.skew() if len(returns_series) > 0 else 0,
            'kurtosis': returns_series.kurtosis() if len(returns_series) > 0 else 0
        }

        return BacktestResult(
            model_name=model_name,
            start_date=str(dates[0].date()),
            end_date=str(dates[-1].date()),
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=total_trades,
            avg_trade_return=avg_trade_return,
            volatility=volatility,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            benchmark_return=benchmark_return,
            excess_return=excess_return,
            information_ratio=information_ratio,
            test_scenarios=[],
            consistency_score=consistency_score,
            risk_metrics=risk_metrics
        )

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """计算最大回撤"""
        if len(returns) == 0:
            return 0.0

        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())

    def _calculate_consistency_score(self, returns: pd.Series) -> float:
        """计算一致性评分"""
        if len(returns) == 0:
            return 0.0

        # 计算连续盈利/亏损的稳定性
        positive_runs = (returns > 0).astype(int).groupby((returns <= 0).astype(int).cumsum()).sum()
        negative_runs = (returns < 0).astype(int).groupby((returns >= 0).astype(int).cumsum()).sum()

        # 计算变异系数
        cv = returns.std() / abs(returns.mean()) if returns.mean() != 0 else float('inf')

        # 计算一致性评分 (0-1)
        consistency = 1 / (1 + cv) if cv != float('inf') else 0

        return min(consistency, 1.0)

    def run_scenario_tests(self, model_name: str, symbols: List[str],
                           start_date: str, end_date: str) -> Dict[str, BacktestResult]:
        """运行多情景测试"""
        self.logger.info(f"运行多情景测试: {model_name}")

        # 生成基础历史数据
        base_data = self.generate_historical_data(symbols, start_date, end_date)

        scenario_results = {}

        for scenario_name, scenario_config in self.market_scenarios.items():
            self.logger.info(f"测试情景: {scenario_name}")

            # 创建情景数据
            scenario_data = self.create_market_scenario_data(scenario_name, base_data)

            # 运行回测
            result = self.run_model_backtest(model_name, scenario_data)
            result.test_scenarios = [scenario_name]

            scenario_results[scenario_name] = result

        return scenario_results

    def generate_backtest_report(self, scenario_results: Dict[str, BacktestResult]) -> str:
        """生成回测报告"""
        self.logger.info("生成回测报告")

        report_content = []
        report_content.append("# 量化模型回测集成报告")
        report_content.append("")
        report_content.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append(
            f"**测试模型数**: {len(set(r.model_name for r in scenario_results.values()))}")
        report_content.append(f"**市场情景数**: {len(scenario_results)}")
        report_content.append("")

        # 情景对比表
        report_content.append("## 📊 情景测试结果对比")
        report_content.append("")
        report_content.append("| 情景 | 总收益率 | 夏普比率 | 最大回撤 | 胜率 | 交易次数 | 一致性评分 |")
        report_content.append(
            "|------|----------|----------|----------|------|----------|------------|")

        for scenario_name, result in scenario_results.items():
            report_content.append(
                f"| {scenario_name} | {result.total_return:.2%} | {result.sharpe_ratio:.2f} | "
                f"{result.max_drawdown:.2%} | {result.win_rate:.2%} | {result.total_trades} | "
                f"{result.consistency_score:.2f} |"
            )

        report_content.append("")

        # 详细分析
        report_content.append("## 📈 详细分析")
        report_content.append("")

        for scenario_name, result in scenario_results.items():
            report_content.append(f"### {scenario_name}")
            report_content.append("")
            report_content.append(
                f"- **描述**: {self.market_scenarios[scenario_name]['description']}")
            report_content.append(f"- **测试期间**: {result.start_date} 至 {result.end_date}")
            report_content.append(f"- **总收益率**: {result.total_return:.2%}")
            report_content.append(f"- **年化夏普比率**: {result.sharpe_ratio:.2f}")
            report_content.append(f"- **最大回撤**: {result.max_drawdown:.2%}")
            report_content.append(f"- **胜率**: {result.win_rate:.2%}")
            report_content.append(f"- **总交易次数**: {result.total_trades}")
            report_content.append(f"- **平均交易收益**: {result.avg_trade_return:.2%}")
            report_content.append(f"- **年化波动率**: {result.volatility:.2%}")
            report_content.append(f"- **卡玛比率**: {result.calmar_ratio:.2f}")
            report_content.append(f"- **索提诺比率**: {result.sortino_ratio:.2f}")
            report_content.append(f"- **超额收益**: {result.excess_return:.2%}")
            report_content.append(f"- **信息比率**: {result.information_ratio:.2f}")
            report_content.append(f"- **一致性评分**: {result.consistency_score:.2f}")
            report_content.append("")

            # 风险指标
            report_content.append("#### 风险指标")
            report_content.append(f"- **VaR(95%)**: {result.risk_metrics['var_95']:.2%}")
            report_content.append(f"- **CVaR(95%)**: {result.risk_metrics['cvar_95']:.2%}")
            report_content.append(f"- **偏度**: {result.risk_metrics['skewness']:.2f}")
            report_content.append(f"- **峰度**: {result.risk_metrics['kurtosis']:.2f}")
            report_content.append("")

        # 一致性分析
        report_content.append("## 🔍 模型一致性分析")
        report_content.append("")

        consistency_scores = [r.consistency_score for r in scenario_results.values()]
        avg_consistency = np.mean(consistency_scores)

        report_content.append(f"**平均一致性评分**: {avg_consistency:.2f}")
        report_content.append("")

        if avg_consistency >= 0.8:
            report_content.append("✅ **优秀**: 模型在不同市场情景下表现高度一致")
        elif avg_consistency >= 0.6:
            report_content.append("⚠️ **良好**: 模型表现基本一致，但仍有改进空间")
        else:
            report_content.append("❌ **需改进**: 模型在不同情景下表现差异较大")

        report_content.append("")

        # 建议
        report_content.append("## 💡 改进建议")
        report_content.append("")

        # 分析各情景的表现
        best_scenario = max(scenario_results.items(), key=lambda x: x[1].sharpe_ratio)
        worst_scenario = min(scenario_results.items(), key=lambda x: x[1].sharpe_ratio)

        report_content.append(
            f"**最佳表现情景**: {best_scenario[0]} (夏普比率: {best_scenario[1].sharpe_ratio:.2f})")
        report_content.append(
            f"**最差表现情景**: {worst_scenario[0]} (夏普比率: {worst_scenario[1].sharpe_ratio:.2f})")
        report_content.append("")

        # 具体建议
        if worst_scenario[1].max_drawdown > 0.2:
            report_content.append("- 🔧 **风险控制**: 建议加强风险控制机制，降低最大回撤")

        if worst_scenario[1].win_rate < 0.4:
            report_content.append("- 🎯 **信号质量**: 建议优化交易信号生成逻辑，提高胜率")

        if avg_consistency < 0.6:
            report_content.append("- 🔄 **模型稳定性**: 建议增强模型在不同市场环境下的适应性")

        if worst_scenario[1].information_ratio < 0.5:
            report_content.append("- 📊 **超额收益**: 建议优化模型以获取更好的超额收益")

        report_content.append("")
        report_content.append("## 📋 测试覆盖情况")
        report_content.append("")

        covered_scenarios = list(scenario_results.keys())
        report_content.append(f"**已测试情景**: {', '.join(covered_scenarios)}")
        report_content.append("")

        for scenario_name in covered_scenarios:
            config = self.market_scenarios[scenario_name]
            report_content.append(f"- **{scenario_name}**: {config['description']}")
            report_content.append(f"  - 条件: {', '.join(config['conditions'])}")
            report_content.append(f"  - 期望表现: {config['expected_performance']}")
            report_content.append("")

        return "\n".join(report_content)

    def run_integration_test(self, model_names: List[str], symbols: List[str],
                             start_date: str, end_date: str) -> Dict[str, Any]:
        """运行完整的回测集成测试"""
        self.logger.info("开始回测集成测试")

        all_results = {}

        for model_name in model_names:
            self.logger.info(f"测试模型: {model_name}")

            # 运行多情景测试
            scenario_results = self.run_scenario_tests(model_name, symbols, start_date, end_date)
            all_results[model_name] = scenario_results

        # 生成综合报告
        report_content = self.generate_comprehensive_report(all_results)
        report_file = self.output_dir / "backtest_integration_report.md"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

        self.logger.info(f"回测集成报告已生成: {report_file}")

        return {
            'results': all_results,
            'report_file': str(report_file)
        }

    def generate_comprehensive_report(self, all_results: Dict[str, Dict[str, BacktestResult]]) -> str:
        """生成综合回测报告"""
        report_content = []
        report_content.append("# 量化模型回测集成综合报告")
        report_content.append("")
        report_content.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append(f"**测试模型数**: {len(all_results)}")
        report_content.append(f"**市场情景数**: {len(self.market_scenarios)}")
        report_content.append("")

        # 模型对比
        report_content.append("## 📊 模型表现对比")
        report_content.append("")

        # 计算每个模型的平均表现
        model_summaries = {}
        for model_name, scenario_results in all_results.items():
            avg_sharpe = np.mean([r.sharpe_ratio for r in scenario_results.values()])
            avg_return = np.mean([r.total_return for r in scenario_results.values()])
            avg_drawdown = np.mean([r.max_drawdown for r in scenario_results.values()])
            avg_consistency = np.mean([r.consistency_score for r in scenario_results.values()])

            model_summaries[model_name] = {
                'avg_sharpe': avg_sharpe,
                'avg_return': avg_return,
                'avg_drawdown': avg_drawdown,
                'avg_consistency': avg_consistency
            }

        # 排序并显示
        sorted_models = sorted(model_summaries.items(),
                               key=lambda x: x[1]['avg_sharpe'], reverse=True)

        report_content.append("| 模型 | 平均夏普比率 | 平均收益率 | 平均最大回撤 | 平均一致性 |")
        report_content.append("|------|-------------|------------|--------------|------------|")

        for model_name, summary in sorted_models:
            report_content.append(
                f"| {model_name} | {summary['avg_sharpe']:.2f} | {summary['avg_return']:.2%} | "
                f"{summary['avg_drawdown']:.2%} | {summary['avg_consistency']:.2f} |"
            )

        report_content.append("")

        # 详细分析
        for model_name, scenario_results in all_results.items():
            report_content.append(f"## 📈 {model_name} 详细分析")
            report_content.append("")

            for scenario_name, result in scenario_results.items():
                report_content.append(f"### {scenario_name}")
                report_content.append(f"- 总收益率: {result.total_return:.2%}")
                report_content.append(f"- 夏普比率: {result.sharpe_ratio:.2f}")
                report_content.append(f"- 最大回撤: {result.max_drawdown:.2%}")
                report_content.append(f"- 一致性评分: {result.consistency_score:.2f}")
                report_content.append("")

        return "\n".join(report_content)


def main():
    """主函数"""
    print("🚀 开始回测集成测试")
    print("="*60)

    # 创建回测集成框架
    framework = BacktestIntegrationFramework()

    # 测试参数
    model_names = ["MovingAverageModel", "RSIModel", "MACDModel"]
    symbols = ["600519.SH", "000858.SZ", "002415.SZ"]
    start_date = "2023-01-01"
    end_date = "2023-12-31"

    # 运行集成测试
    results = framework.run_integration_test(model_names, symbols, start_date, end_date)

    print("="*60)
    print("回测集成测试完成")
    print("="*60)
    print(f"测试模型: {len(results['results'])}个")
    print(f"市场情景: {len(framework.market_scenarios)}个")
    print(f"报告文件: {results['report_file']}")
    print("="*60)


if __name__ == "__main__":
    main()
