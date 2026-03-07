#!/usr/bin/env python3
"""
简化性能测试基准框架
建立性能测试基准，确保测试不影响模型执行效率
监控测试执行时间变化
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
import logging
from dataclasses import dataclass
from decimal import getcontext
import psutil
import gc

# 设置高精度计算
getcontext().prec = 28

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class SimplePerformanceMetrics:
    """简化性能指标数据类"""
    test_name: str
    model_name: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    throughput: float
    latency: float
    baseline_time: float
    performance_ratio: float
    efficiency_score: float


class SimplePerformanceBenchmark:
    """简化性能测试基准"""

    def __init__(self, output_dir: str = "reports/performance_benchmark"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()

        # 性能基准配置
        self.benchmark_config = {
            'models': ['MovingAverageModel', 'RSIModel', 'MACDModel'],
            'data_sizes': [1000, 5000, 10000],
            'iterations': 3,
            'baseline_threshold': 1.2,
            'improvement_threshold': 0.8,
            'memory_threshold': 1024,  # MB
            'cpu_threshold': 80.0  # %
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

    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            'cpu_count': os.cpu_count(),
            'memory_total': psutil.virtual_memory().total / (1024**3),  # GB
            'memory_available': psutil.virtual_memory().available / (1024**3),  # GB
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent
        }

    def generate_test_data(self, size: int) -> pd.DataFrame:
        """生成测试数据"""
        self.logger.info(f"生成测试数据，大小: {size}")

        # 生成时间序列数据
        dates = pd.date_range(start='2023-01-01', periods=size, freq='D')

        # 生成价格数据
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, size)
        prices = 100 * np.cumprod(1 + returns)

        # 生成OHLC数据
        high = prices * (1 + np.random.uniform(0, 0.03, size))
        low = prices * (1 - np.random.uniform(0, 0.03, size))
        close = prices
        open_price = np.roll(close, 1)
        open_price[0] = close[0]

        # 生成成交量
        volume = np.random.uniform(1000000, 10000000, size)

        df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=dates)

        return df

    def measure_execution_time(self, func, *args, **kwargs) -> Tuple[float, float, Any]:
        """测量函数执行时间"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024**2)  # MB

        # 执行函数
        result = func(*args, **kwargs)

        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / (1024**2)  # MB

        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory

        return execution_time, memory_usage, result

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """计算MACD指标"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd

    def _calculate_efficiency_score(self, execution_time: float, memory_usage: float, cpu_usage: float) -> float:
        """计算效率评分"""
        time_score = 1.0 / (1.0 + execution_time)
        memory_score = 1.0 / (1.0 + memory_usage / 100)
        cpu_score = 1.0 / (1.0 + cpu_usage / 100)

        efficiency_score = (time_score + memory_score + cpu_score) / 3
        return min(efficiency_score, 1.0)

    def run_data_processing_benchmark(self, data_size: int) -> SimplePerformanceMetrics:
        """运行数据处理性能基准测试"""
        self.logger.info(f"运行数据处理基准测试，数据大小: {data_size}")

        def data_processing_operations():
            """数据处理操作"""
            # 生成数据
            df = self.generate_test_data(data_size)

            # 数据清洗
            df = df.dropna()

            # 数据转换
            df['returns'] = df['close'].pct_change()
            df['ma_5'] = df['close'].rolling(5).mean()
            df['ma_20'] = df['close'].rolling(20).mean()
            df['volatility'] = df['returns'].rolling(20).std()

            return df

        # 测量执行时间
        execution_time, memory_usage, result = self.measure_execution_time(
            data_processing_operations)

        # 计算性能指标
        throughput = data_size / execution_time if execution_time > 0 else 0
        latency = execution_time / data_size if data_size > 0 else 0

        # 获取CPU使用率
        cpu_usage = psutil.cpu_percent(interval=1)

        # 计算效率评分
        efficiency_score = self._calculate_efficiency_score(execution_time, memory_usage, cpu_usage)

        return SimplePerformanceMetrics(
            test_name=f"data_processing_{data_size}",
            model_name="DataProcessor",
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            throughput=throughput,
            latency=latency,
            baseline_time=execution_time,
            performance_ratio=1.0,
            efficiency_score=efficiency_score
        )

    def run_model_training_benchmark(self, model_name: str, data_size: int) -> SimplePerformanceMetrics:
        """运行模型训练性能基准测试"""
        self.logger.info(f"运行模型训练基准测试，模型: {model_name}，数据大小: {data_size}")

        def model_training_operations():
            """模型训练操作"""
            # 生成数据
            df = self.generate_test_data(data_size)

            # 特征工程
            df['returns'] = df['close'].pct_change()
            df['ma_5'] = df['close'].rolling(5).mean()
            df['ma_20'] = df['close'].rolling(20).mean()
            df['rsi'] = self._calculate_rsi(df['close'])
            df['macd'] = self._calculate_macd(df['close'])

            # 模拟模型训练（不使用sklearn）
            features = df[['ma_5', 'ma_20', 'rsi', 'macd']].dropna()
            target = (df['close'].shift(-1) > df['close']).astype(int)
            target = target[features.index]

            # 简单的线性回归模拟
            if len(features) > 0 and len(target) > 0:
                # 使用简单的移动平均作为预测
                predictions = features['ma_5'] > features['ma_20']
                accuracy = (predictions == target).mean() if len(target) > 0 else 0
            else:
                accuracy = 0

            return {
                'accuracy': accuracy,
                'features_count': len(features.columns),
                'data_points': len(features)
            }

        # 测量执行时间
        execution_time, memory_usage, result = self.measure_execution_time(
            model_training_operations)

        # 计算性能指标
        throughput = data_size / execution_time if execution_time > 0 else 0
        latency = execution_time / data_size if data_size > 0 else 0

        # 获取CPU使用率
        cpu_usage = psutil.cpu_percent(interval=1)

        # 计算效率评分
        efficiency_score = self._calculate_efficiency_score(execution_time, memory_usage, cpu_usage)

        return SimplePerformanceMetrics(
            test_name=f"model_training_{model_name}_{data_size}",
            model_name=model_name,
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            throughput=throughput,
            latency=latency,
            baseline_time=execution_time,
            performance_ratio=1.0,
            efficiency_score=efficiency_score
        )

    def run_backtest_benchmark(self, model_name: str, data_size: int) -> SimplePerformanceMetrics:
        """运行回测性能基准测试"""
        self.logger.info(f"运行回测基准测试，模型: {model_name}，数据大小: {data_size}")

        def backtest_operations():
            """回测操作"""
            # 生成数据
            df = self.generate_test_data(data_size)

            # 初始化
            initial_capital = 100000.0
            cash = initial_capital
            positions = {}
            portfolio_values = []

            # 计算技术指标
            df['ma_5'] = df['close'].rolling(5).mean()
            df['ma_20'] = df['close'].rolling(20).mean()
            df['rsi'] = self._calculate_rsi(df['close'])
            df['macd'] = self._calculate_macd(df['close'])

            # 回测循环
            for i in range(len(df)):
                if i < 20:  # 跳过前20个数据点
                    continue

                current_price = df['close'].iloc[i]

                # 交易信号
                if model_name == "MovingAverageModel":
                    signal = 1 if df['ma_5'].iloc[i] > df['ma_20'].iloc[i] else -1
                elif model_name == "RSIModel":
                    rsi = df['rsi'].iloc[i]
                    signal = 1 if rsi < 30 else (-1 if rsi > 70 else 0)
                elif model_name == "MACDModel":
                    macd = df['macd'].iloc[i]
                    signal = 1 if macd > 0 else -1
                else:
                    signal = 0

                # 执行交易
                if signal == 1 and cash > 0:
                    shares = int(cash * 0.1 / current_price)
                    if shares > 0:
                        positions['stock'] = positions.get('stock', 0) + shares
                        cash -= shares * current_price

                elif signal == -1 and positions.get('stock', 0) > 0:
                    shares = positions['stock']
                    cash += shares * current_price
                    positions['stock'] = 0

                # 计算组合价值
                portfolio_value = cash + sum(positions.get(symbol, 0)
                                             * current_price for symbol in positions)
                portfolio_values.append(portfolio_value)

            return {
                'final_value': portfolio_values[-1] if portfolio_values else initial_capital,
                'total_return': (portfolio_values[-1] - initial_capital) / initial_capital if portfolio_values else 0,
                'trades_count': len([v for v in portfolio_values if v != initial_capital])
            }

        # 测量执行时间
        execution_time, memory_usage, result = self.measure_execution_time(backtest_operations)

        # 计算性能指标
        throughput = data_size / execution_time if execution_time > 0 else 0
        latency = execution_time / data_size if data_size > 0 else 0

        # 获取CPU使用率
        cpu_usage = psutil.cpu_percent(interval=1)

        # 计算效率评分
        efficiency_score = self._calculate_efficiency_score(execution_time, memory_usage, cpu_usage)

        return SimplePerformanceMetrics(
            test_name=f"backtest_{model_name}_{data_size}",
            model_name=model_name,
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            throughput=throughput,
            latency=latency,
            baseline_time=execution_time,
            performance_ratio=1.0,
            efficiency_score=efficiency_score
        )

    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """运行综合性能基准测试"""
        self.logger.info("开始运行综合性能基准测试")

        benchmark_results = []
        baseline_results = {}

        # 获取系统信息
        system_info = self.get_system_info()
        self.logger.info(f"系统信息: {system_info}")

        # 运行数据处理基准测试
        for data_size in self.benchmark_config['data_sizes']:
            self.logger.info(f"测试数据处理性能，数据大小: {data_size}")

            # 运行多次取平均值
            execution_times = []
            memory_usages = []
            cpu_usages = []

            for i in range(self.benchmark_config['iterations']):
                metrics = self.run_data_processing_benchmark(data_size)
                execution_times.append(metrics.execution_time)
                memory_usages.append(metrics.memory_usage)
                cpu_usages.append(metrics.cpu_usage)

                # 清理内存
                gc.collect()

            # 计算平均值
            avg_metrics = SimplePerformanceMetrics(
                test_name=f"data_processing_{data_size}",
                model_name="DataProcessor",
                execution_time=np.mean(execution_times),
                memory_usage=np.mean(memory_usages),
                cpu_usage=np.mean(cpu_usages),
                throughput=data_size / np.mean(execution_times),
                latency=np.mean(execution_times) / data_size,
                baseline_time=np.mean(execution_times),
                performance_ratio=1.0,
                efficiency_score=np.mean([self._calculate_efficiency_score(t, m, c)
                                         for t, m, c in zip(execution_times, memory_usages, cpu_usages)])
            )

            baseline_results[f"data_processing_{data_size}"] = avg_metrics
            benchmark_results.append(avg_metrics)

        # 运行模型训练基准测试
        for model_name in self.benchmark_config['models']:
            for data_size in self.benchmark_config['data_sizes']:
                self.logger.info(f"测试模型训练性能，模型: {model_name}，数据大小: {data_size}")

                execution_times = []
                memory_usages = []
                cpu_usages = []

                for i in range(self.benchmark_config['iterations']):
                    metrics = self.run_model_training_benchmark(model_name, data_size)
                    execution_times.append(metrics.execution_time)
                    memory_usages.append(metrics.memory_usage)
                    cpu_usages.append(metrics.cpu_usage)

                    gc.collect()

                avg_metrics = SimplePerformanceMetrics(
                    test_name=f"model_training_{model_name}_{data_size}",
                    model_name=model_name,
                    execution_time=np.mean(execution_times),
                    memory_usage=np.mean(memory_usages),
                    cpu_usage=np.mean(cpu_usages),
                    throughput=data_size / np.mean(execution_times),
                    latency=np.mean(execution_times) / data_size,
                    baseline_time=np.mean(execution_times),
                    performance_ratio=1.0,
                    efficiency_score=np.mean([self._calculate_efficiency_score(
                        t, m, c) for t, m, c in zip(execution_times, memory_usages, cpu_usages)])
                )

                baseline_results[f"model_training_{model_name}_{data_size}"] = avg_metrics
                benchmark_results.append(avg_metrics)

        # 运行回测基准测试
        for model_name in self.benchmark_config['models']:
            for data_size in self.benchmark_config['data_sizes']:
                self.logger.info(f"测试回测性能，模型: {model_name}，数据大小: {data_size}")

                execution_times = []
                memory_usages = []
                cpu_usages = []

                for i in range(self.benchmark_config['iterations']):
                    metrics = self.run_backtest_benchmark(model_name, data_size)
                    execution_times.append(metrics.execution_time)
                    memory_usages.append(metrics.memory_usage)
                    cpu_usages.append(metrics.cpu_usage)

                    gc.collect()

                avg_metrics = SimplePerformanceMetrics(
                    test_name=f"backtest_{model_name}_{data_size}",
                    model_name=model_name,
                    execution_time=np.mean(execution_times),
                    memory_usage=np.mean(memory_usages),
                    cpu_usage=np.mean(cpu_usages),
                    throughput=data_size / np.mean(execution_times),
                    latency=np.mean(execution_times) / data_size,
                    baseline_time=np.mean(execution_times),
                    performance_ratio=1.0,
                    efficiency_score=np.mean([self._calculate_efficiency_score(
                        t, m, c) for t, m, c in zip(execution_times, memory_usages, cpu_usages)])
                )

                baseline_results[f"backtest_{model_name}_{data_size}"] = avg_metrics
                benchmark_results.append(avg_metrics)

        # 生成基准报告
        report_content = self.generate_benchmark_report(benchmark_results, system_info)
        report_file = self.output_dir / "simple_performance_benchmark_report.md"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

        # 保存基准数据
        baseline_file = self.output_dir / "simple_performance_baseline.json"
        baseline_data = {
            'timestamp': datetime.now().isoformat(),
            'system_info': system_info,
            'baseline_metrics': {k: {
                'execution_time': v.execution_time,
                'memory_usage': v.memory_usage,
                'cpu_usage': v.cpu_usage,
                'throughput': v.throughput,
                'latency': v.latency,
                'efficiency_score': v.efficiency_score
            } for k, v in baseline_results.items()}
        }

        with open(baseline_file, 'w', encoding='utf-8') as f:
            json.dump(baseline_data, f, indent=2)

        self.logger.info(f"性能基准报告已生成: {report_file}")
        self.logger.info(f"基准数据已保存: {baseline_file}")

        return {
            'benchmark_results': benchmark_results,
            'baseline_results': baseline_results,
            'system_info': system_info,
            'report_file': str(report_file),
            'baseline_file': str(baseline_file)
        }

    def generate_benchmark_report(self, benchmark_results: List[SimplePerformanceMetrics],
                                  system_info: Dict[str, Any]) -> str:
        """生成基准测试报告"""
        report_content = []
        report_content.append("# 简化性能测试基准报告")
        report_content.append("")
        report_content.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append(f"**测试数量**: {len(benchmark_results)}")
        report_content.append("")

        # 系统信息
        report_content.append("## 💻 系统信息")
        report_content.append("")
        report_content.append(f"- **CPU核心数**: {system_info['cpu_count']}")
        report_content.append(f"- **总内存**: {system_info['memory_total']:.1f}GB")
        report_content.append(f"- **可用内存**: {system_info['memory_available']:.1f}GB")
        report_content.append(f"- **CPU使用率**: {system_info['cpu_percent']:.1f}%")
        report_content.append(f"- **内存使用率**: {system_info['memory_percent']:.1f}%")
        report_content.append("")

        # 性能统计
        report_content.append("## 📊 性能统计")
        report_content.append("")

        execution_times = [r.execution_time for r in benchmark_results]
        memory_usages = [r.memory_usage for r in benchmark_results]
        cpu_usages = [r.cpu_usage for r in benchmark_results]
        efficiency_scores = [r.efficiency_score for r in benchmark_results]

        report_content.append("| 指标 | 平均值 | 最小值 | 最大值 |")
        report_content.append("|------|--------|--------|--------|")
        report_content.append(
            f"| 执行时间(秒) | {np.mean(execution_times):.3f} | {np.min(execution_times):.3f} | {np.max(execution_times):.3f} |")
        report_content.append(
            f"| 内存使用(MB) | {np.mean(memory_usages):.1f} | {np.min(memory_usages):.1f} | {np.max(memory_usages):.1f} |")
        report_content.append(
            f"| CPU使用率(%) | {np.mean(cpu_usages):.1f} | {np.min(cpu_usages):.1f} | {np.max(cpu_usages):.1f} |")
        report_content.append(
            f"| 效率评分 | {np.mean(efficiency_scores):.3f} | {np.min(efficiency_scores):.3f} | {np.max(efficiency_scores):.3f} |")
        report_content.append("")

        # 按模型分类统计
        report_content.append("## 📈 模型性能对比")
        report_content.append("")
        report_content.append("| 模型 | 平均执行时间(秒) | 平均内存使用(MB) | 平均CPU使用率(%) | 平均效率评分 |")
        report_content.append(
            "|------|-----------------|------------------|------------------|--------------|")

        model_stats = {}
        for result in benchmark_results:
            model_name = result.model_name
            if model_name not in model_stats:
                model_stats[model_name] = {'times': [], 'memory': [], 'cpu': [], 'efficiency': []}

            model_stats[model_name]['times'].append(result.execution_time)
            model_stats[model_name]['memory'].append(result.memory_usage)
            model_stats[model_name]['cpu'].append(result.cpu_usage)
            model_stats[model_name]['efficiency'].append(result.efficiency_score)

        for model_name, stats in model_stats.items():
            avg_time = np.mean(stats['times'])
            avg_memory = np.mean(stats['memory'])
            avg_cpu = np.mean(stats['cpu'])
            avg_efficiency = np.mean(stats['efficiency'])

            report_content.append(
                f"| {model_name} | {avg_time:.3f} | {avg_memory:.1f} | {avg_cpu:.1f} | {avg_efficiency:.3f} |")

        report_content.append("")

        # 详细结果
        report_content.append("## 📋 详细测试结果")
        report_content.append("")

        for result in benchmark_results:
            report_content.append(f"### {result.test_name}")
            report_content.append(f"- **模型**: {result.model_name}")
            report_content.append(f"- **执行时间**: {result.execution_time:.3f}秒")
            report_content.append(f"- **内存使用**: {result.memory_usage:.1f}MB")
            report_content.append(f"- **CPU使用率**: {result.cpu_usage:.1f}%")
            report_content.append(f"- **吞吐量**: {result.throughput:.0f} 数据点/秒")
            report_content.append(f"- **延迟**: {result.latency:.6f} 秒/数据点")
            report_content.append(f"- **效率评分**: {result.efficiency_score:.3f}")
            report_content.append("")

        # 性能建议
        report_content.append("## 💡 性能优化建议")
        report_content.append("")

        # 找出性能最差的测试
        worst_performance = min(benchmark_results, key=lambda x: x.efficiency_score)
        best_performance = max(benchmark_results, key=lambda x: x.efficiency_score)

        report_content.append(
            f"**性能最差**: {worst_performance.test_name} (效率评分: {worst_performance.efficiency_score:.3f})")
        report_content.append(
            f"**性能最佳**: {best_performance.test_name} (效率评分: {best_performance.efficiency_score:.3f})")
        report_content.append("")

        if worst_performance.memory_usage > 100:
            report_content.append("- 🔧 **内存优化**: 建议优化内存使用，减少内存分配")

        if worst_performance.cpu_usage > 70:
            report_content.append("- ⚡ **CPU优化**: 建议优化算法复杂度，减少CPU使用")

        if worst_performance.execution_time > np.mean([r.execution_time for r in benchmark_results]) * 2:
            report_content.append("- 🚀 **执行时间优化**: 建议优化算法实现，减少执行时间")

        report_content.append("")
        report_content.append("## 📈 监控建议")
        report_content.append("")
        report_content.append("- 📊 **定期基准测试**: 建议每周运行一次基准测试")
        report_content.append("- 🔍 **性能监控**: 实时监控系统资源使用情况")
        report_content.append("- ⚠️ **告警设置**: 设置内存和CPU使用率告警")
        report_content.append("- 📈 **趋势分析**: 跟踪性能变化趋势")

        return "\n".join(report_content)


def main():
    """主函数"""
    print("🚀 开始简化性能测试基准")
    print("="*60)

    # 创建简化性能基准框架
    framework = SimplePerformanceBenchmark()

    # 运行综合基准测试
    results = framework.run_comprehensive_benchmark()

    print("="*60)
    print("简化性能测试基准完成")
    print("="*60)
    print(f"测试数量: {len(results['benchmark_results'])}")
    print(f"基准报告: {results['report_file']}")
    print(f"基准数据: {results['baseline_file']}")
    print("="*60)


if __name__ == "__main__":
    main()
