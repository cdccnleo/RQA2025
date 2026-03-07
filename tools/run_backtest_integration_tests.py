#!/usr/bin/env python3
"""
综合回测测试运行器
整合量化模型测试与回测框架集成
验证模型在历史数据上的表现一致性
确保测试覆盖各种市场情景
"""

import sys
import time
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
class BacktestTestResult:
    """回测测试结果"""
    test_name: str
    model_name: str
    scenario: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    consistency_score: float
    stability_score: float
    robustness_score: float
    test_duration: float
    status: str  # 'passed', 'failed', 'warning'


class BacktestIntegrationTestRunner:
    """回测集成测试运行器"""

    def __init__(self, output_dir: str = "reports/backtest_integration"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()

        # 测试配置
        self.test_config = {
            'models': ['MovingAverageModel', 'RSIModel', 'MACDModel'],
            'symbols': ['600519.SH', '000858.SZ', '002415.SZ'],
            'periods': {
                'short': ('2023-01-01', '2023-06-30'),
                'medium': ('2023-01-01', '2023-12-31'),
                'long': ('2022-01-01', '2023-12-31')
            },
            'scenarios': [
                'bull_market', 'bear_market', 'sideways_market',
                'high_volatility', 'low_volatility', 'crisis_market', 'recovery_market'
            ],
            'initial_capital': 100000.0
        }

        # 测试标准
        self.test_standards = {
            'min_sharpe_ratio': 0.5,
            'max_drawdown_threshold': 0.3,
            'min_consistency_score': 0.6,
            'min_stability_score': 0.5,
            'min_robustness_score': 0.4
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

    def generate_test_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """生成测试数据"""
        self.logger.info(f"生成测试数据: {symbols}")

        historical_data = {}

        for symbol in symbols:
            # 生成真实的价格序列
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            n_days = len(dates)

            # 基础参数
            base_price = 100.0
            daily_return_mean = 0.0005
            daily_return_std = 0.02

            # 生成收益率序列
            np.random.seed(hash(symbol) % 1000)
            returns = np.random.normal(daily_return_mean, daily_return_std, n_days)

            # 添加自相关性
            for i in range(1, len(returns)):
                returns[i] = 0.1 * returns[i-1] + 0.9 * returns[i]

            # 生成价格序列
            prices = base_price * np.cumprod(1 + returns)

            # 生成OHLC数据
            high = prices * (1 + np.random.uniform(0, 0.03, n_days))
            low = prices * (1 - np.random.uniform(0, 0.03, n_days))
            close = prices
            open_price = np.roll(close, 1)
            open_price[0] = close[0]

            # 生成成交量
            volume_base = 1000000
            volume = volume_base * (1 + abs(returns) * 10 + np.random.uniform(-0.2, 0.2, n_days))
            volume = np.maximum(volume, volume_base * 0.1)

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

    def create_scenario_data(self, scenario: str, base_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """创建情景特定数据"""
        self.logger.info(f"创建情景数据: {scenario}")

        scenario_data = {}

        for symbol, df in base_data.items():
            scenario_df = df.copy()

            if scenario == 'bull_market':
                # 牛市：上升趋势，低波动
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
                # 高波动
                volatility_factor = 2.0
                returns = df['close'].pct_change()
                adjusted_returns = returns * volatility_factor
                scenario_df['close'] = df['close'].iloc[0] * (1 + adjusted_returns).cumprod()

            elif scenario == 'low_volatility':
                # 低波动
                volatility_factor = 0.6
                returns = df['close'].pct_change()
                adjusted_returns = returns * volatility_factor
                scenario_df['close'] = df['close'].iloc[0] * (1 + adjusted_returns).cumprod()

            elif scenario == 'crisis_market':
                # 危机市场
                crisis_returns = np.random.normal(-0.02, 0.05, len(df))
                panic_days = np.random.choice(len(df), size=len(df)//10, replace=False)
                crisis_returns[panic_days] = np.random.normal(-0.08, 0.03, len(panic_days))
                scenario_df['close'] = df['close'].iloc[0] * (1 + crisis_returns).cumprod()

            elif scenario == 'recovery_market':
                # 复苏市场
                recovery_factor = np.linspace(0.8, 1.3, len(df))
                scenario_df['close'] = df['close'] * recovery_factor

            # 更新OHLC数据
            scenario_df['high'] = scenario_df['close'] * (1 + np.random.uniform(0, 0.02, len(df)))
            scenario_df['low'] = scenario_df['close'] * (1 - np.random.uniform(0, 0.02, len(df)))
            scenario_df['open'] = scenario_df['close'].shift(1)
            scenario_df['open'].iloc[0] = scenario_df['close'].iloc[0]

            scenario_data[symbol] = scenario_df

        return scenario_data

    def run_model_backtest(self, model_name: str, data: Dict[str, pd.DataFrame],
                           initial_capital: float = 100000.0) -> Dict[str, float]:
        """运行模型回测"""
        self.logger.info(f"运行模型回测: {model_name}")

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
                if i > 0:
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
                        signal = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])

                    # 执行交易
                    if signal == 1 and symbol not in positions:
                        position_size = cash * 0.1
                        shares = int(position_size / current_price)
                        if shares > 0:
                            positions[symbol] = shares
                            cash -= shares * current_price

                    elif signal == -1 and symbol in positions:
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

        # 计算指标
        if len(all_returns) == 0:
            return {
                'total_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'consistency_score': 0.0,
                'stability_score': 0.0,
                'robustness_score': 0.0
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
        stability_score = self._calculate_stability_score(returns_series)
        robustness_score = min(consistency_score * stability_score, 1.0)

        return {
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'consistency_score': consistency_score,
            'stability_score': stability_score,
            'robustness_score': robustness_score
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

        # 计算变异系数
        cv = returns.std() / abs(returns.mean()) if returns.mean() != 0 else float('inf')
        consistency = 1 / (1 + cv) if cv != float('inf') else 0
        return min(consistency, 1.0)

    def _calculate_stability_score(self, returns: pd.Series) -> float:
        """计算稳定性评分"""
        if len(returns) == 0:
            return 0.0

        # 计算连续盈利/亏损的稳定性
        positive_runs = (returns > 0).astype(int).groupby((returns <= 0).astype(int).cumsum()).sum()
        negative_runs = (returns < 0).astype(int).groupby((returns >= 0).astype(int).cumsum()).sum()

        run_stability = 1 / (1 + positive_runs.std() + negative_runs.std()
                             ) if len(positive_runs) > 0 else 0
        return min(run_stability, 1.0)

    def evaluate_test_result(self, result: Dict[str, float], standards: Dict[str, float]) -> str:
        """评估测试结果"""
        if result['sharpe_ratio'] < standards['min_sharpe_ratio']:
            return 'failed'
        elif result['max_drawdown'] > standards['max_drawdown_threshold']:
            return 'failed'
        elif result['consistency_score'] < standards['min_consistency_score']:
            return 'warning'
        elif result['stability_score'] < standards['min_stability_score']:
            return 'warning'
        elif result['robustness_score'] < standards['min_robustness_score']:
            return 'warning'
        else:
            return 'passed'

    def run_comprehensive_tests(self) -> List[BacktestTestResult]:
        """运行综合测试"""
        self.logger.info("开始运行综合回测测试")

        test_results = []

        for model_name in self.test_config['models']:
            self.logger.info(f"测试模型: {model_name}")

            for period_name, (start_date, end_date) in self.test_config['periods'].items():
                self.logger.info(f"  测试期间: {period_name}")

                # 生成基础数据
                base_data = self.generate_test_data(
                    self.test_config['symbols'], start_date, end_date
                )

                for scenario in self.test_config['scenarios']:
                    self.logger.info(f"    测试情景: {scenario}")

                    # 创建情景数据
                    scenario_data = self.create_scenario_data(scenario, base_data)

                    # 运行回测
                    start_time = time.time()
                    backtest_result = self.run_model_backtest(
                        model_name, scenario_data, self.test_config['initial_capital']
                    )
                    test_duration = time.time() - start_time

                    # 评估结果
                    status = self.evaluate_test_result(backtest_result, self.test_standards)

                    # 创建测试结果
                    test_result = BacktestTestResult(
                        test_name=f"{model_name}_{scenario}_{period_name}",
                        model_name=model_name,
                        scenario=scenario,
                        total_return=backtest_result['total_return'],
                        sharpe_ratio=backtest_result['sharpe_ratio'],
                        max_drawdown=backtest_result['max_drawdown'],
                        win_rate=backtest_result['win_rate'],
                        consistency_score=backtest_result['consistency_score'],
                        stability_score=backtest_result['stability_score'],
                        robustness_score=backtest_result['robustness_score'],
                        test_duration=test_duration,
                        status=status
                    )

                    test_results.append(test_result)

        return test_results

    def generate_test_report(self, test_results: List[BacktestTestResult]) -> str:
        """生成测试报告"""
        self.logger.info("生成测试报告")

        # 统计结果
        total_tests = len(test_results)
        passed_tests = len([r for r in test_results if r.status == 'passed'])
        failed_tests = len([r for r in test_results if r.status == 'failed'])
        warning_tests = len([r for r in test_results if r.status == 'warning'])

        # 按模型统计
        model_stats = {}
        for result in test_results:
            if result.model_name not in model_stats:
                model_stats[result.model_name] = {'passed': 0, 'failed': 0, 'warning': 0}
            model_stats[result.model_name][result.status] += 1

        # 按情景统计
        scenario_stats = {}
        for result in test_results:
            if result.scenario not in scenario_stats:
                scenario_stats[result.scenario] = {'passed': 0, 'failed': 0, 'warning': 0}
            scenario_stats[result.scenario][result.status] += 1

        # 生成报告内容
        report_content = []
        report_content.append("# 量化模型回测集成测试报告")
        report_content.append("")
        report_content.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append(f"**总测试数**: {total_tests}")
        report_content.append(f"**通过**: {passed_tests}")
        report_content.append(f"**失败**: {failed_tests}")
        report_content.append(f"**警告**: {warning_tests}")
        report_content.append(f"**通过率**: {passed_tests/total_tests*100:.1f}%")
        report_content.append("")

        # 总体统计
        report_content.append("## 📊 总体统计")
        report_content.append("")
        report_content.append("| 指标 | 平均值 | 最小值 | 最大值 |")
        report_content.append("|------|--------|--------|--------|")

        avg_return = np.mean([r.total_return for r in test_results])
        min_return = np.min([r.total_return for r in test_results])
        max_return = np.max([r.total_return for r in test_results])
        report_content.append(f"| 总收益率 | {avg_return:.2%} | {min_return:.2%} | {max_return:.2%} |")

        avg_sharpe = np.mean([r.sharpe_ratio for r in test_results])
        min_sharpe = np.min([r.sharpe_ratio for r in test_results])
        max_sharpe = np.max([r.sharpe_ratio for r in test_results])
        report_content.append(f"| 夏普比率 | {avg_sharpe:.2f} | {min_sharpe:.2f} | {max_sharpe:.2f} |")

        avg_drawdown = np.mean([r.max_drawdown for r in test_results])
        min_drawdown = np.min([r.max_drawdown for r in test_results])
        max_drawdown = np.max([r.max_drawdown for r in test_results])
        report_content.append(
            f"| 最大回撤 | {avg_drawdown:.2%} | {min_drawdown:.2%} | {max_drawdown:.2%} |")

        avg_consistency = np.mean([r.consistency_score for r in test_results])
        min_consistency = np.min([r.consistency_score for r in test_results])
        max_consistency = np.max([r.consistency_score for r in test_results])
        report_content.append(
            f"| 一致性评分 | {avg_consistency:.2f} | {min_consistency:.2f} | {max_consistency:.2f} |")

        report_content.append("")

        # 模型表现对比
        report_content.append("## 📈 模型表现对比")
        report_content.append("")
        report_content.append("| 模型 | 通过率 | 平均夏普比率 | 平均一致性 | 平均稳定性 |")
        report_content.append("|------|--------|-------------|------------|------------|")

        for model_name, stats in model_stats.items():
            model_results = [r for r in test_results if r.model_name == model_name]
            pass_rate = stats['passed'] / len(model_results) * 100
            avg_sharpe = np.mean([r.sharpe_ratio for r in model_results])
            avg_consistency = np.mean([r.consistency_score for r in model_results])
            avg_stability = np.mean([r.stability_score for r in model_results])

            report_content.append(
                f"| {model_name} | {pass_rate:.1f}% | {avg_sharpe:.2f} | {avg_consistency:.2f} | {avg_stability:.2f} |"
            )

        report_content.append("")

        # 情景表现对比
        report_content.append("## 🎯 情景表现对比")
        report_content.append("")
        report_content.append("| 情景 | 通过率 | 平均收益率 | 平均波动率 |")
        report_content.append("|------|--------|------------|------------|")

        for scenario_name, stats in scenario_stats.items():
            scenario_results = [r for r in test_results if r.scenario == scenario_name]
            pass_rate = stats['passed'] / len(scenario_results) * 100
            avg_return = np.mean([r.total_return for r in scenario_results])
            avg_volatility = np.mean(
                [r.volatility for r in scenario_results if hasattr(r, 'volatility')])

            report_content.append(
                f"| {scenario_name} | {pass_rate:.1f}% | {avg_return:.2%} | {avg_volatility:.2%} |"
            )

        report_content.append("")

        # 详细结果
        report_content.append("## 📋 详细测试结果")
        report_content.append("")

        for result in test_results:
            status_icon = "✅" if result.status == 'passed' else "❌" if result.status == 'failed' else "⚠️"
            report_content.append(f"### {status_icon} {result.test_name}")
            report_content.append(f"- **模型**: {result.model_name}")
            report_content.append(f"- **情景**: {result.scenario}")
            report_content.append(f"- **总收益率**: {result.total_return:.2%}")
            report_content.append(f"- **夏普比率**: {result.sharpe_ratio:.2f}")
            report_content.append(f"- **最大回撤**: {result.max_drawdown:.2%}")
            report_content.append(f"- **胜率**: {result.win_rate:.2%}")
            report_content.append(f"- **一致性评分**: {result.consistency_score:.2f}")
            report_content.append(f"- **稳定性评分**: {result.stability_score:.2f}")
            report_content.append(f"- **鲁棒性评分**: {result.robustness_score:.2f}")
            report_content.append(f"- **测试耗时**: {result.test_duration:.2f}秒")
            report_content.append(f"- **状态**: {result.status}")
            report_content.append("")

        # 改进建议
        report_content.append("## 💡 改进建议")
        report_content.append("")

        if failed_tests > 0:
            report_content.append("- 🔧 **修复失败的测试**: 检查模型在特定情景下的表现")

        if warning_tests > 0:
            report_content.append("- ⚠️ **优化警告测试**: 提升模型的一致性和稳定性")

        # 分析具体问题
        failed_results = [r for r in test_results if r.status == 'failed']
        if failed_results:
            report_content.append("### 失败测试分析")
            for result in failed_results:
                report_content.append(f"- **{result.test_name}**: 需要优化在{result.scenario}情景下的表现")

        report_content.append("")
        report_content.append("## 🎯 测试覆盖情况")
        report_content.append("")
        report_content.append(f"- **模型覆盖**: {len(self.test_config['models'])}个模型")
        report_content.append(f"- **情景覆盖**: {len(self.test_config['scenarios'])}个市场情景")
        report_content.append(f"- **期间覆盖**: {len(self.test_config['periods'])}个时间段")
        report_content.append(f"- **总测试用例**: {total_tests}个")

        return "\n".join(report_content)

    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        self.logger.info("开始运行所有回测集成测试")

        # 运行综合测试
        test_results = self.run_comprehensive_tests()

        # 生成报告
        report_content = self.generate_test_report(test_results)
        report_file = self.output_dir / "backtest_integration_test_report.md"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

        self.logger.info(f"测试报告已生成: {report_file}")

        # 统计结果
        total_tests = len(test_results)
        passed_tests = len([r for r in test_results if r.status == 'passed'])
        failed_tests = len([r for r in test_results if r.status == 'failed'])
        warning_tests = len([r for r in test_results if r.status == 'warning'])

        return {
            'test_results': test_results,
            'report_file': str(report_file),
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'warning_tests': warning_tests,
                'pass_rate': passed_tests / total_tests * 100 if total_tests > 0 else 0
            }
        }


def main():
    """主函数"""
    print("🚀 开始回测集成测试")
    print("="*60)

    # 创建测试运行器
    runner = BacktestIntegrationTestRunner()

    # 运行所有测试
    results = runner.run_all_tests()

    print("="*60)
    print("回测集成测试完成")
    print("="*60)
    print(f"总测试数: {results['summary']['total_tests']}")
    print(f"通过: {results['summary']['passed_tests']}")
    print(f"失败: {results['summary']['failed_tests']}")
    print(f"警告: {results['summary']['warning_tests']}")
    print(f"通过率: {results['summary']['pass_rate']:.1f}%")
    print(f"报告文件: {results['report_file']}")
    print("="*60)


if __name__ == "__main__":
    main()
