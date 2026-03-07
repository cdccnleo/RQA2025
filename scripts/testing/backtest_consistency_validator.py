#!/usr/bin/env python3
"""
回测一致性验证器
专门验证模型在历史数据上的表现一致性
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
import warnings
warnings.filterwarnings('ignore')

# 设置高精度计算
getcontext().prec = 28

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class ConsistencyMetrics:
    """一致性指标数据类"""
    model_name: str
    scenario_name: str
    return_consistency: float
    volatility_consistency: float
    drawdown_consistency: float
    sharpe_consistency: float
    overall_consistency: float
    stability_score: float
    robustness_score: float


class BacktestConsistencyValidator:
    """回测一致性验证器"""

    def __init__(self, output_dir: str = "reports/backtest_consistency"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()

        # 历史数据时间段
        self.historical_periods = {
            'recent_1y': ('2023-01-01', '2023-12-31'),
            'recent_2y': ('2022-01-01', '2023-12-31'),
            'bull_period': ('2019-01-01', '2021-12-31'),
            'bear_period': ('2018-01-01', '2018-12-31'),
            'crisis_period': ('2020-03-01', '2020-12-31'),
            'recovery_period': ('2021-01-01', '2021-12-31'),
            'sideways_period': ('2017-01-01', '2017-12-31')
        }

        # 市场情景定义
        self.market_scenarios = {
            'bull_market': {
                'description': '牛市情景',
                'expected_metrics': {
                    'min_return': 0.05,
                    'max_volatility': 0.25,
                    'min_sharpe': 0.5
                }
            },
            'bear_market': {
                'description': '熊市情景',
                'expected_metrics': {
                    'max_return': 0.10,
                    'max_volatility': 0.35,
                    'min_sharpe': -0.5
                }
            },
            'sideways_market': {
                'description': '震荡市情景',
                'expected_metrics': {
                    'min_return': -0.05,
                    'max_return': 0.15,
                    'max_volatility': 0.20,
                    'min_sharpe': 0.0
                }
            },
            'high_volatility': {
                'description': '高波动情景',
                'expected_metrics': {
                    'min_volatility': 0.25,
                    'max_drawdown': 0.30
                }
            },
            'low_volatility': {
                'description': '低波动情景',
                'expected_metrics': {
                    'max_volatility': 0.15,
                    'min_sharpe': 0.3
                }
            },
            'crisis_market': {
                'description': '危机市场情景',
                'expected_metrics': {
                    'max_return': 0.20,
                    'min_volatility': 0.30,
                    'max_drawdown': 0.40
                }
            },
            'recovery_market': {
                'description': '复苏市场情景',
                'expected_metrics': {
                    'min_return': 0.10,
                    'max_volatility': 0.25,
                    'min_sharpe': 0.8
                }
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

    def generate_realistic_historical_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """生成更真实的历史数据"""
        self.logger.info(f"生成真实历史数据: {symbols}")

        historical_data = {}

        for symbol in symbols:
            # 生成更真实的价格序列
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            n_days = len(dates)

            # 基础参数
            base_price = 100.0
            daily_return_mean = 0.0005  # 年化约12%
            daily_return_std = 0.02     # 年化约32%

            # 生成收益率序列
            np.random.seed(hash(symbol) % 1000)  # 确保不同股票有不同的随机性
            returns = np.random.normal(daily_return_mean, daily_return_std, n_days)

            # 添加自相关性
            for i in range(1, len(returns)):
                returns[i] = 0.1 * returns[i-1] + 0.9 * returns[i]

            # 添加波动率聚集效应
            volatility = np.ones(n_days) * daily_return_std
            for i in range(1, len(volatility)):
                volatility[i] = 0.95 * volatility[i-1] + 0.05 * abs(returns[i-1])
                returns[i] = returns[i] * (volatility[i] / daily_return_std)

            # 生成价格序列
            prices = base_price * np.cumprod(1 + returns)

            # 生成OHLC数据
            high = prices * (1 + np.random.uniform(0, 0.03, n_days))
            low = prices * (1 - np.random.uniform(0, 0.03, n_days))
            close = prices
            open_price = np.roll(close, 1)
            open_price[0] = close[0]

            # 生成成交量（与价格变动相关）
            volume_base = 1000000
            volume = volume_base * (1 + abs(returns) * 10 + np.random.uniform(-0.2, 0.2, n_days))
            volume = np.maximum(volume, volume_base * 0.1)  # 最小成交量

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

    def create_scenario_specific_data(self, scenario: str, base_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """根据特定情景调整数据"""
        self.logger.info(f"创建情景特定数据: {scenario}")

        scenario_data = {}
        scenario_config = self.market_scenarios[scenario]

        for symbol, df in base_data.items():
            # 复制基础数据
            scenario_df = df.copy()

            if scenario == 'bull_market':
                # 牛市：增加上升趋势，降低波动率
                trend_factor = 1.5
                volatility_factor = 0.8
                returns = df['close'].pct_change()
                adjusted_returns = returns * volatility_factor + \
                    np.linspace(0, 0.001, len(df)) * trend_factor
                scenario_df['close'] = df['close'].iloc[0] * (1 + adjusted_returns).cumprod()

            elif scenario == 'bear_market':
                # 熊市：下降趋势，高波动
                trend_factor = -1.2
                volatility_factor = 1.3
                returns = df['close'].pct_change()
                adjusted_returns = returns * volatility_factor + \
                    np.linspace(0, -0.001, len(df)) * trend_factor
                scenario_df['close'] = df['close'].iloc[0] * (1 + adjusted_returns).cumprod()

            elif scenario == 'high_volatility':
                # 高波动：增加波动率
                volatility_factor = 2.0
                returns = df['close'].pct_change()
                adjusted_returns = returns * volatility_factor
                scenario_df['close'] = df['close'].iloc[0] * (1 + adjusted_returns).cumprod()

            elif scenario == 'low_volatility':
                # 低波动：降低波动率
                volatility_factor = 0.6
                returns = df['close'].pct_change()
                adjusted_returns = returns * volatility_factor
                scenario_df['close'] = df['close'].iloc[0] * (1 + adjusted_returns).cumprod()

            elif scenario == 'crisis_market':
                # 危机市场：极端波动和恐慌性抛售
                crisis_returns = np.random.normal(-0.02, 0.05, len(df))
                # 添加恐慌性抛售效应
                panic_days = np.random.choice(len(df), size=len(df)//10, replace=False)
                crisis_returns[panic_days] = np.random.normal(-0.08, 0.03, len(panic_days))
                scenario_df['close'] = df['close'].iloc[0] * (1 + crisis_returns).cumprod()

            elif scenario == 'recovery_market':
                # 复苏市场：从低点反弹
                recovery_factor = np.linspace(0.8, 1.3, len(df))  # 从80%反弹到130%
                scenario_df['close'] = df['close'] * recovery_factor

            # 更新OHLC数据
            scenario_df['high'] = scenario_df['close'] * (1 + np.random.uniform(0, 0.02, len(df)))
            scenario_df['low'] = scenario_df['close'] * (1 - np.random.uniform(0, 0.02, len(df)))
            scenario_df['open'] = scenario_df['close'].shift(1)
            scenario_df['open'].iloc[0] = scenario_df['close'].iloc[0]

            # 调整成交量
            if scenario in ['bull_market', 'recovery_market']:
                scenario_df['volume'] = df['volume'] * 1.2
            elif scenario in ['bear_market', 'crisis_market']:
                scenario_df['volume'] = df['volume'] * 1.5  # 恐慌性交易量增加
            elif scenario == 'low_volatility':
                scenario_df['volume'] = df['volume'] * 0.8

            scenario_data[symbol] = scenario_df

        return scenario_data

    def run_consistency_backtest(self, model_name: str, data: Dict[str, pd.DataFrame],
                                 initial_capital: float = 100000.0) -> Dict[str, float]:
        """运行一致性回测"""
        self.logger.info(f"运行一致性回测: {model_name}")

        # 模拟模型预测和交易
        all_returns = []
        positions = {}
        cash = initial_capital
        portfolio_value = initial_capital

        symbols = list(data.keys())
        dates = list(data[symbols[0]].index)

        for i, date in enumerate(dates):
            daily_return = 0.0

            for symbol in symbols:
                if symbol not in data:
                    continue

                current_price = data[symbol].loc[date, 'close']

                # 模拟模型预测
                if i > 0:  # 从第二天开始交易
                    # 基于技术指标的交易策略
                    if model_name == "MovingAverageModel":
                        # 移动平均策略
                        ma_short = data[symbol]['close'].rolling(5).mean().iloc[i]
                        ma_long = data[symbol]['close'].rolling(20).mean().iloc[i]
                        signal = 1 if ma_short > ma_long else -1

                    elif model_name == "RSIModel":
                        # RSI策略
                        returns = data[symbol]['close'].pct_change()
                        gains = returns.where(returns > 0, 0)
                        losses = -returns.where(returns < 0, 0)
                        avg_gain = gains.rolling(14).mean().iloc[i]
                        avg_loss = losses.rolling(14).mean().iloc[i]
                        rs = avg_gain / avg_loss if avg_loss > 0 else 0
                        rsi = 100 - (100 / (1 + rs))
                        signal = 1 if rsi < 30 else (-1 if rsi > 70 else 0)

                    elif model_name == "MACDModel":
                        # MACD策略
                        ema_12 = data[symbol]['close'].ewm(span=12).mean().iloc[i]
                        ema_26 = data[symbol]['close'].ewm(span=26).mean().iloc[i]
                        macd = ema_12 - ema_26
                        signal = 1 if macd > 0 else -1

                    else:
                        # 默认策略
                        signal = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])

                    # 执行交易
                    if signal == 1 and symbol not in positions:
                        # 买入信号
                        position_size = cash * 0.1  # 使用10%资金
                        shares = int(position_size / current_price)
                        if shares > 0:
                            positions[symbol] = shares
                            cash -= shares * current_price

                    elif signal == -1 and symbol in positions:
                        # 卖出信号
                        shares = positions[symbol]
                        cash += shares * current_price
                        del positions[symbol]

                # 计算持仓价值
                if symbol in positions:
                    portfolio_value += positions[symbol] * current_price

            # 计算当日收益率
            if i > 0:
                prev_value = portfolio_value - \
                    sum(positions.get(s, 0) * data[s].loc[date, 'close']
                        for s in symbols if s in positions)
                if prev_value > 0:
                    daily_return = (portfolio_value - prev_value) / prev_value
                    all_returns.append(daily_return)

        # 计算一致性指标
        if len(all_returns) == 0:
            return {
                'total_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'consistency_score': 0.0
            }

        returns_series = pd.Series(all_returns)

        # 基础指标
        total_return = (portfolio_value - initial_capital) / initial_capital
        volatility = returns_series.std() * np.sqrt(252)
        sharpe_ratio = returns_series.mean() / returns_series.std() * \
            np.sqrt(252) if returns_series.std() > 0 else 0
        max_drawdown = self._calculate_max_drawdown(returns_series)
        win_rate = (returns_series > 0).mean()

        # 一致性评分
        consistency_score = self._calculate_consistency_score(returns_series)

        return {
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'consistency_score': consistency_score
        }

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

        # 计算多个一致性指标
        # 1. 收益率稳定性
        return_stability = 1 / (1 + returns.std() / abs(returns.mean())
                                ) if returns.mean() != 0 else 0

        # 2. 连续盈利/亏损的稳定性
        positive_runs = (returns > 0).astype(int).groupby((returns <= 0).astype(int).cumsum()).sum()
        negative_runs = (returns < 0).astype(int).groupby((returns >= 0).astype(int).cumsum()).sum()

        run_stability = 1 / (1 + positive_runs.std() + negative_runs.std()
                             ) if len(positive_runs) > 0 else 0

        # 3. 波动率稳定性
        volatility_stability = 1 / (1 + returns.rolling(20).std().std()
                                    ) if len(returns) >= 20 else 0

        # 综合一致性评分
        consistency = (return_stability + run_stability + volatility_stability) / 3
        return min(consistency, 1.0)

    def validate_scenario_consistency(self, model_name: str, scenario: str,
                                      results: Dict[str, float]) -> ConsistencyMetrics:
        """验证情景一致性"""
        scenario_config = self.market_scenarios[scenario]
        expected_metrics = scenario_config['expected_metrics']

        # 计算各项指标的一致性
        return_consistency = 1.0
        volatility_consistency = 1.0
        drawdown_consistency = 1.0
        sharpe_consistency = 1.0

        # 检查收益率一致性
        if 'min_return' in expected_metrics:
            if results['total_return'] < expected_metrics['min_return']:
                return_consistency = 0.5
        if 'max_return' in expected_metrics:
            if results['total_return'] > expected_metrics['max_return']:
                return_consistency = 0.5

        # 检查波动率一致性
        if 'min_volatility' in expected_metrics:
            if results['volatility'] < expected_metrics['min_volatility']:
                volatility_consistency = 0.5
        if 'max_volatility' in expected_metrics:
            if results['volatility'] > expected_metrics['max_volatility']:
                volatility_consistency = 0.5

        # 检查夏普比率一致性
        if 'min_sharpe' in expected_metrics:
            if results['sharpe_ratio'] < expected_metrics['min_sharpe']:
                sharpe_consistency = 0.5

        # 检查最大回撤一致性
        if 'max_drawdown' in expected_metrics:
            if results['max_drawdown'] > expected_metrics['max_drawdown']:
                drawdown_consistency = 0.5

        # 计算整体一致性
        overall_consistency = (return_consistency + volatility_consistency +
                               drawdown_consistency + sharpe_consistency) / 4

        # 计算稳定性评分
        stability_score = results['consistency_score']

        # 计算鲁棒性评分
        robustness_score = min(overall_consistency * stability_score, 1.0)

        return ConsistencyMetrics(
            model_name=model_name,
            scenario_name=scenario,
            return_consistency=return_consistency,
            volatility_consistency=volatility_consistency,
            drawdown_consistency=drawdown_consistency,
            sharpe_consistency=sharpe_consistency,
            overall_consistency=overall_consistency,
            stability_score=stability_score,
            robustness_score=robustness_score
        )

    def run_comprehensive_validation(self, model_names: List[str], symbols: List[str]) -> Dict[str, Any]:
        """运行综合验证"""
        self.logger.info("开始综合一致性验证")

        all_results = {}
        all_consistency_metrics = {}

        for model_name in model_names:
            self.logger.info(f"验证模型: {model_name}")
            model_results = {}
            model_consistency = {}

            for scenario_name in self.market_scenarios.keys():
                self.logger.info(f"  测试情景: {scenario_name}")

                # 生成历史数据
                base_data = self.generate_realistic_historical_data(
                    symbols, '2023-01-01', '2023-12-31')

                # 创建情景特定数据
                scenario_data = self.create_scenario_specific_data(scenario_name, base_data)

                # 运行回测
                backtest_results = self.run_consistency_backtest(model_name, scenario_data)

                # 验证一致性
                consistency_metrics = self.validate_scenario_consistency(
                    model_name, scenario_name, backtest_results)

                model_results[scenario_name] = backtest_results
                model_consistency[scenario_name] = consistency_metrics

            all_results[model_name] = model_results
            all_consistency_metrics[model_name] = model_consistency

        # 生成验证报告
        report_content = self.generate_validation_report(all_results, all_consistency_metrics)
        report_file = self.output_dir / "consistency_validation_report.md"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

        self.logger.info(f"一致性验证报告已生成: {report_file}")

        return {
            'results': all_results,
            'consistency_metrics': all_consistency_metrics,
            'report_file': str(report_file)
        }

    def generate_validation_report(self, results: Dict[str, Dict[str, Dict]],
                                   consistency_metrics: Dict[str, Dict[str, ConsistencyMetrics]]) -> str:
        """生成验证报告"""
        report_content = []
        report_content.append("# 量化模型回测一致性验证报告")
        report_content.append("")
        report_content.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append(f"**验证模型数**: {len(results)}")
        report_content.append(f"**市场情景数**: {len(self.market_scenarios)}")
        report_content.append("")

        # 模型一致性对比
        report_content.append("## 📊 模型一致性对比")
        report_content.append("")
        report_content.append("| 模型 | 平均一致性 | 平均稳定性 | 平均鲁棒性 | 最佳情景 | 最差情景 |")
        report_content.append(
            "|------|------------|------------|------------|----------|----------|")

        for model_name in results.keys():
            model_consistency = consistency_metrics[model_name]

            avg_consistency = np.mean([m.overall_consistency for m in model_consistency.values()])
            avg_stability = np.mean([m.stability_score for m in model_consistency.values()])
            avg_robustness = np.mean([m.robustness_score for m in model_consistency.values()])

            # 找出最佳和最差情景
            best_scenario = max(model_consistency.items(), key=lambda x: x[1].overall_consistency)
            worst_scenario = min(model_consistency.items(), key=lambda x: x[1].overall_consistency)

            report_content.append(
                f"| {model_name} | {avg_consistency:.2f} | {avg_stability:.2f} | {avg_robustness:.2f} | "
                f"{best_scenario[0]} | {worst_scenario[0]} |"
            )

        report_content.append("")

        # 详细分析
        for model_name in results.keys():
            report_content.append(f"## 📈 {model_name} 详细分析")
            report_content.append("")

            for scenario_name in results[model_name].keys():
                backtest_result = results[model_name][scenario_name]
                consistency_metric = consistency_metrics[model_name][scenario_name]

                report_content.append(f"### {scenario_name}")
                report_content.append(
                    f"- **描述**: {self.market_scenarios[scenario_name]['description']}")
                report_content.append(f"- **总收益率**: {backtest_result['total_return']:.2%}")
                report_content.append(f"- **年化波动率**: {backtest_result['volatility']:.2%}")
                report_content.append(f"- **夏普比率**: {backtest_result['sharpe_ratio']:.2f}")
                report_content.append(f"- **最大回撤**: {backtest_result['max_drawdown']:.2%}")
                report_content.append(f"- **胜率**: {backtest_result['win_rate']:.2%}")
                report_content.append(f"- **一致性评分**: {backtest_result['consistency_score']:.2f}")
                report_content.append("")
                report_content.append("#### 一致性指标")
                report_content.append(f"- **收益率一致性**: {consistency_metric.return_consistency:.2f}")
                report_content.append(
                    f"- **波动率一致性**: {consistency_metric.volatility_consistency:.2f}")
                report_content.append(f"- **回撤一致性**: {consistency_metric.drawdown_consistency:.2f}")
                report_content.append(f"- **夏普比率一致性**: {consistency_metric.sharpe_consistency:.2f}")
                report_content.append(f"- **整体一致性**: {consistency_metric.overall_consistency:.2f}")
                report_content.append(f"- **稳定性评分**: {consistency_metric.stability_score:.2f}")
                report_content.append(f"- **鲁棒性评分**: {consistency_metric.robustness_score:.2f}")
                report_content.append("")

        # 改进建议
        report_content.append("## 💡 改进建议")
        report_content.append("")

        for model_name in results.keys():
            model_consistency = consistency_metrics[model_name]
            avg_consistency = np.mean([m.overall_consistency for m in model_consistency.values()])
            avg_stability = np.mean([m.stability_score for m in model_consistency.values()])

            report_content.append(f"### {model_name}")

            if avg_consistency < 0.6:
                report_content.append("- 🔧 **一致性不足**: 建议优化模型在不同市场情景下的适应性")

            if avg_stability < 0.5:
                report_content.append("- 📊 **稳定性不足**: 建议增强模型的稳定性机制")

            # 分析具体问题情景
            worst_scenarios = sorted(model_consistency.items(),
                                     key=lambda x: x[1].overall_consistency)[:2]
            for scenario_name, metric in worst_scenarios:
                if metric.overall_consistency < 0.5:
                    report_content.append(f"- ⚠️ **{scenario_name}情景表现不佳**: 需要针对性优化")

            report_content.append("")

        return "\n".join(report_content)


def main():
    """主函数"""
    print("🔍 开始回测一致性验证")
    print("="*60)

    # 创建一致性验证器
    validator = BacktestConsistencyValidator()

    # 测试参数
    model_names = ["MovingAverageModel", "RSIModel", "MACDModel"]
    symbols = ["600519.SH", "000858.SZ", "002415.SZ"]

    # 运行综合验证
    results = validator.run_comprehensive_validation(model_names, symbols)

    print("="*60)
    print("回测一致性验证完成")
    print("="*60)
    print(f"验证模型: {len(results['results'])}个")
    print(f"市场情景: {len(validator.market_scenarios)}个")
    print(f"报告文件: {results['report_file']}")
    print("="*60)


if __name__ == "__main__":
    main()
