#!/usr/bin/env python3
"""
性能测试基准运行脚本
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
from dataclasses import dataclass, asdict
import psutil
import gc

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    test_name: str
    model_name: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    throughput: float
    latency: float
    efficiency_score: float
    timestamp: datetime


class PerformanceBenchmarkRunner:
    """性能测试基准运行器"""

    def __init__(self, output_dir: str = "reports/performance_benchmark"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()

        # 测试配置
        self.test_config = {
            'models': ['MovingAverageModel', 'RSIModel', 'MACDModel'],
            'data_sizes': [1000, 5000, 10000],
            'iterations': 3
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
        memory = psutil.virtual_memory()
        return {
            'cpu_count': os.cpu_count(),
            'memory_total': memory.total / (1024**3),  # GB
            'memory_available': memory.available / (1024**3),  # GB
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': memory.percent
        }

    def measure_execution_time(self, func, *args, **kwargs) -> Tuple[float, float, Any]:
        """测量函数执行时间和内存使用"""
        # 获取初始内存使用
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB

        # 记录开始时间
        start_time = time.time()

        # 执行函数
        result = func(*args, **kwargs)

        # 记录结束时间
        end_time = time.time()

        # 获取最终内存使用
        final_memory = process.memory_info().rss / (1024 * 1024)  # MB
        memory_usage = final_memory - initial_memory

        execution_time = end_time - start_time

        return execution_time, memory_usage, result

    def _calculate_efficiency_score(self, execution_time: float, memory_usage: float, cpu_usage: float) -> float:
        """计算效率评分"""
        # 基于执行时间、内存使用和CPU使用的综合评分
        time_score = 1.0 / (1.0 + execution_time)  # 时间越短越好
        memory_score = 1.0 / (1.0 + abs(memory_usage) / 100)  # 内存使用越少越好
        cpu_score = 1.0 / (1.0 + cpu_usage / 100)  # CPU使用越少越好

        # 加权平均
        efficiency_score = (0.4 * time_score + 0.3 * memory_score + 0.3 * cpu_score)
        return min(1.0, max(0.0, efficiency_score))

    def generate_test_data(self, size: int) -> pd.DataFrame:
        """生成测试数据"""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=size, freq='D')

        data = {
            'date': dates,
            'open': np.random.uniform(100, 200, size),
            'high': np.random.uniform(100, 200, size),
            'low': np.random.uniform(100, 200, size),
            'close': np.random.uniform(100, 200, size),
            'volume': np.random.uniform(1000000, 10000000, size)
        }

        return pd.DataFrame(data)

    def run_data_processing_benchmark(self, data_size: int) -> PerformanceMetrics:
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

        return PerformanceMetrics(
            test_name=f"data_processing_{data_size}",
            model_name="DataProcessor",
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            throughput=throughput,
            latency=latency,
            efficiency_score=efficiency_score,
            timestamp=datetime.now()
        )

    def run_model_training_benchmark(self, model_name: str, data_size: int) -> PerformanceMetrics:
        """运行模型训练性能基准测试"""
        self.logger.info(f"运行模型训练基准测试，模型: {model_name}，数据大小: {data_size}")

        def model_training_operations():
            """模型训练操作"""
            # 生成数据
            df = self.generate_test_data(data_size)

            # 特征计算
            df['returns'] = df['close'].pct_change()
            df['ma_5'] = df['close'].rolling(5).mean()
            df['ma_20'] = df['close'].rolling(20).mean()
            df['volatility'] = df['returns'].rolling(20).std()

            # 根据模型类型进行训练
            if model_name == 'MovingAverageModel':
                # 移动平均模型
                df['signal'] = np.where(df['ma_5'] > df['ma_20'], 1, -1)
            elif model_name == 'RSIModel':
                # RSI模型
                df['rsi'] = self._calculate_rsi(df['close'])
                df['signal'] = np.where(df['rsi'] > 70, -1, np.where(df['rsi'] < 30, 1, 0))
            elif model_name == 'MACDModel':
                # MACD模型
                df['macd'] = self._calculate_macd(df['close'])
                df['signal'] = np.where(df['macd'] > 0, 1, -1)

            # 计算性能指标
            returns = df['signal'].shift(1) * df['returns']
            total_return = returns.sum()
            sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0

            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'signal_count': len(df[df['signal'] != 0])
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

        return PerformanceMetrics(
            test_name=f"model_training_{model_name}_{data_size}",
            model_name=model_name,
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            throughput=throughput,
            latency=latency,
            efficiency_score=efficiency_score,
            timestamp=datetime.now()
        )

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """计算MACD指标"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return macd - signal_line

    def run_backtest_benchmark(self, model_name: str, data_size: int) -> PerformanceMetrics:
        """运行回测性能基准测试"""
        self.logger.info(f"运行回测基准测试，模型: {model_name}，数据大小: {data_size}")

        def backtest_operations():
            """回测操作"""
            # 生成数据
            df = self.generate_test_data(data_size)

            # 初始化回测参数
            initial_capital = 100000
            cash = initial_capital
            positions = {}
            portfolio_values = [initial_capital]

            # 计算信号
            df['returns'] = df['close'].pct_change()
            df['ma_5'] = df['close'].rolling(5).mean()
            df['ma_20'] = df['close'].rolling(20).mean()
            df['volatility'] = df['returns'].rolling(20).std()

            if model_name == 'MovingAverageModel':
                df['signal'] = np.where(df['ma_5'] > df['ma_20'], 1, -1)
            elif model_name == 'RSIModel':
                df['rsi'] = self._calculate_rsi(df['close'])
                df['signal'] = np.where(df['rsi'] > 70, -1, np.where(df['rsi'] < 30, 1, 0))
            elif model_name == 'MACDModel':
                df['macd'] = self._calculate_macd(df['close'])
                df['signal'] = np.where(df['macd'] > 0, 1, -1)

            # 执行回测
            for i in range(1, len(df)):
                current_price = df.iloc[i]['close']
                signal = df.iloc[i]['signal']

                # 交易逻辑
                if signal == 1 and cash > 0:  # 买入信号
                    shares_to_buy = cash // current_price
                    if shares_to_buy > 0:
                        cash -= shares_to_buy * current_price
                        positions['STOCK'] = positions.get('STOCK', 0) + shares_to_buy

                elif signal == -1 and positions.get('STOCK', 0) > 0:  # 卖出信号
                    shares_to_sell = positions['STOCK']
                    cash += shares_to_sell * current_price
                    positions['STOCK'] = 0

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

        return PerformanceMetrics(
            test_name=f"backtest_{model_name}_{data_size}",
            model_name=model_name,
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            throughput=throughput,
            latency=latency,
            efficiency_score=efficiency_score,
            timestamp=datetime.now()
        )

    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """运行综合性能基准测试"""
        self.logger.info("开始运行综合性能基准测试")

        benchmark_results = []

        # 获取系统信息
        system_info = self.get_system_info()
        self.logger.info(f"系统信息: {system_info}")

        # 运行数据处理基准测试
        for data_size in self.test_config['data_sizes']:
            self.logger.info(f"测试数据处理性能，数据大小: {data_size}")

            # 运行多次取平均值
            execution_times = []
            memory_usages = []
            cpu_usages = []
            efficiency_scores = []

            for i in range(self.test_config['iterations']):
                metrics = self.run_data_processing_benchmark(data_size)
                execution_times.append(metrics.execution_time)
                memory_usages.append(metrics.memory_usage)
                cpu_usages.append(metrics.cpu_usage)
                efficiency_scores.append(metrics.efficiency_score)

                # 清理内存
                gc.collect()

            # 计算平均值
            avg_metrics = PerformanceMetrics(
                test_name=f"data_processing_{data_size}",
                model_name="DataProcessor",
                execution_time=np.mean(execution_times),
                memory_usage=np.mean(memory_usages),
                cpu_usage=np.mean(cpu_usages),
                throughput=data_size / np.mean(execution_times),
                latency=np.mean(execution_times) / data_size,
                efficiency_score=np.mean(efficiency_scores),
                timestamp=datetime.now()
            )

            benchmark_results.append(avg_metrics)

        # 运行模型训练基准测试
        for model_name in self.test_config['models']:
            for data_size in self.test_config['data_sizes']:
                self.logger.info(f"测试模型训练性能，模型: {model_name}，数据大小: {data_size}")

                execution_times = []
                memory_usages = []
                cpu_usages = []
                efficiency_scores = []

                for i in range(self.test_config['iterations']):
                    metrics = self.run_model_training_benchmark(model_name, data_size)
                    execution_times.append(metrics.execution_time)
                    memory_usages.append(metrics.memory_usage)
                    cpu_usages.append(metrics.cpu_usage)
                    efficiency_scores.append(metrics.efficiency_score)

                    gc.collect()

                avg_metrics = PerformanceMetrics(
                    test_name=f"model_training_{model_name}_{data_size}",
                    model_name=model_name,
                    execution_time=np.mean(execution_times),
                    memory_usage=np.mean(memory_usages),
                    cpu_usage=np.mean(cpu_usages),
                    throughput=data_size / np.mean(execution_times),
                    latency=np.mean(execution_times) / data_size,
                    efficiency_score=np.mean(efficiency_scores),
                    timestamp=datetime.now()
                )

                benchmark_results.append(avg_metrics)

        # 运行回测基准测试
        for model_name in self.test_config['models']:
            for data_size in self.test_config['data_sizes']:
                self.logger.info(f"测试回测性能，模型: {model_name}，数据大小: {data_size}")

                execution_times = []
                memory_usages = []
                cpu_usages = []
                efficiency_scores = []

                for i in range(self.test_config['iterations']):
                    metrics = self.run_backtest_benchmark(model_name, data_size)
                    execution_times.append(metrics.execution_time)
                    memory_usages.append(metrics.memory_usage)
                    cpu_usages.append(metrics.cpu_usage)
                    efficiency_scores.append(metrics.efficiency_score)

                    gc.collect()

                avg_metrics = PerformanceMetrics(
                    test_name=f"backtest_{model_name}_{data_size}",
                    model_name=model_name,
                    execution_time=np.mean(execution_times),
                    memory_usage=np.mean(memory_usages),
                    cpu_usage=np.mean(cpu_usages),
                    throughput=data_size / np.mean(execution_times),
                    latency=np.mean(execution_times) / data_size,
                    efficiency_score=np.mean(efficiency_scores),
                    timestamp=datetime.now()
                )

                benchmark_results.append(avg_metrics)

        # 生成基准报告
        report_content = self.generate_benchmark_report(benchmark_results, system_info)
        report_file = self.output_dir / "performance_benchmark_report.md"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

        # 保存基准数据
        baseline_data = {
            'timestamp': datetime.now().isoformat(),
            'system_info': system_info,
            'baseline_metrics': {metrics.test_name: asdict(metrics) for metrics in benchmark_results}
        }

        baseline_file = self.output_dir / "performance_baseline.json"
        with open(baseline_file, 'w', encoding='utf-8') as f:
            json.dump(baseline_data, f, indent=2, ensure_ascii=False, default=str)

        self.logger.info(f"基准测试完成，报告已保存到: {report_file}")

        return {
            'benchmark_results': benchmark_results,
            'system_info': system_info,
            'report_file': str(report_file),
            'baseline_file': str(baseline_file)
        }

    def generate_benchmark_report(self, benchmark_results: List[PerformanceMetrics], system_info: Dict[str, Any]) -> str:
        """生成基准测试报告"""
        report_lines = []

        # 报告头部
        report_lines.extend([
            "# 性能测试基准报告",
            "",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**测试数量**: {len(benchmark_results)}",
            "",
            "## 💻 系统信息",
            "",
            f"- **CPU核心数**: {system_info['cpu_count']}",
            f"- **总内存**: {system_info['memory_total']:.1f}GB",
            f"- **可用内存**: {system_info['memory_available']:.1f}GB",
            f"- **CPU使用率**: {system_info['cpu_percent']:.1f}%",
            f"- **内存使用率**: {system_info['memory_percent']:.1f}%",
            "",
            "## 📊 性能统计",
            "",
            "| 指标 | 平均值 | 最小值 | 最大值 |",
            "|------|--------|--------|--------|"
        ])

        # 计算统计信息
        execution_times = [m.execution_time for m in benchmark_results]
        memory_usages = [m.memory_usage for m in benchmark_results]
        cpu_usages = [m.cpu_usage for m in benchmark_results]
        efficiency_scores = [m.efficiency_score for m in benchmark_results]

        report_lines.extend([
            f"| 执行时间(秒) | {np.mean(execution_times):.3f} | {np.min(execution_times):.3f} | {np.max(execution_times):.3f} |",
            f"| 内存使用(MB) | {np.mean(memory_usages):.1f} | {np.min(memory_usages):.1f} | {np.max(memory_usages):.1f} |",
            f"| CPU使用率(%) | {np.mean(cpu_usages):.1f} | {np.min(cpu_usages):.1f} | {np.max(cpu_usages):.1f} |",
            f"| 效率评分 | {np.mean(efficiency_scores):.3f} | {np.min(efficiency_scores):.3f} | {np.max(efficiency_scores):.3f} |",
            "",
            "## 📈 模型性能对比",
            "",
            "| 模型 | 平均执行时间(秒) | 平均内存使用(MB) | 平均CPU使用率(%) | 平均效率评分 |",
            "|------|-----------------|------------------|------------------|--------------|"
        ])

        # 按模型分组统计
        model_stats = {}
        for metrics in benchmark_results:
            model = metrics.model_name
            if model not in model_stats:
                model_stats[model] = []
            model_stats[model].append(metrics)

        for model, metrics_list in model_stats.items():
            avg_time = np.mean([m.execution_time for m in metrics_list])
            avg_memory = np.mean([m.memory_usage for m in metrics_list])
            avg_cpu = np.mean([m.cpu_usage for m in metrics_list])
            avg_efficiency = np.mean([m.efficiency_score for m in metrics_list])

            report_lines.append(
                f"| {model} | {avg_time:.3f} | {avg_memory:.1f} | {avg_cpu:.1f} | {avg_efficiency:.3f} |"
            )

        report_lines.extend([
            "",
            "## 📋 详细测试结果",
            ""
        ])

        # 详细测试结果
        for metrics in benchmark_results:
            report_lines.extend([
                f"### {metrics.test_name}",
                f"- **模型**: {metrics.model_name}",
                f"- **执行时间**: {metrics.execution_time:.3f}秒",
                f"- **内存使用**: {metrics.memory_usage:.1f}MB",
                f"- **CPU使用率**: {metrics.cpu_usage:.1f}%",
                f"- **吞吐量**: {metrics.throughput:.0f} 数据点/秒",
                f"- **延迟**: {metrics.latency:.6f} 秒/数据点",
                f"- **效率评分**: {metrics.efficiency_score:.3f}",
                ""
            ])

        # 性能优化建议
        worst_performance = min(benchmark_results, key=lambda x: x.efficiency_score)
        best_performance = max(benchmark_results, key=lambda x: x.efficiency_score)

        report_lines.extend([
            "## 💡 性能优化建议",
            "",
            f"**性能最差**: {worst_performance.test_name} (效率评分: {worst_performance.efficiency_score:.3f})",
            f"**性能最佳**: {best_performance.test_name} (效率评分: {best_performance.efficiency_score:.3f})",
            "",
            "",
            "## 📈 监控建议",
            "",
            "- 📊 **定期基准测试**: 建议每周运行一次基准测试",
            "- 🔍 **性能监控**: 实时监控系统资源使用情况",
            "- ⚠️ **告警设置**: 设置内存和CPU使用率告警",
            "- 📈 **趋势分析**: 跟踪性能变化趋势",
            "- 🔄 **自动化测试**: 集成到CI/CD流程中",
            ""
        ])

        return "\n".join(report_lines)


def main():
    """主函数"""
    print("🚀 启动性能测试基准系统")
    print("="*60)

    # 创建性能基准运行器
    benchmark_runner = PerformanceBenchmarkRunner()

    try:
        # 运行综合基准测试
        print("\n📊 运行综合性能基准测试...")
        results = benchmark_runner.run_comprehensive_benchmark()

        print(f"✅ 基准测试完成")
        print(f"📄 报告文件: {results['report_file']}")
        print(f"📊 基准数据: {results['baseline_file']}")

        print("\n🎉 性能测试基准系统运行完成！")

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
