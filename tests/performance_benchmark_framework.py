#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
性能基准测试框架
提供性能基准测试、监控和回归检测功能
"""

import time
import psutil
import statistics
from pathlib import Path
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import logging
import threading
import sys
from tests.test_architecture_config import (
    PerformanceBenchmark,
    test_architecture_config
)

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class PerformanceMeasurement:
    """性能测量结果"""
    benchmark_name: str
    metric_name: str
    measured_value: float
    baseline_value: float
    tolerance_percent: float
    comparison_operator: str
    status: str  # "pass", "fail", "warning"
    deviation_percent: float
    timestamp: datetime
    sample_size: int = 1
    raw_measurements: List[float] = field(default_factory=list)


@dataclass
class SystemMetrics:
    """系统性能指标"""
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_usage_percent: float
    network_io: Dict[str, int]
    timestamp: datetime


class PerformanceBenchmarkFramework:
    """性能基准测试框架"""

    def __init__(self):
        self.benchmarks = test_architecture_config.performance_baselines
        self.logger = self._setup_logger()
        self.measurements_history = []
        self.system_monitoring_active = False
        self.system_metrics = []

    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger("PerformanceBenchmark")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def run_benchmark(self, benchmark_name: str, test_function: Callable,
                    iterations: int = 10, warmup_iterations: int = 3) -> PerformanceMeasurement:
        """运行性能基准测试"""
        if benchmark_name not in self.benchmarks:
            raise ValueError(f"未知的基准测试: {benchmark_name}")

        benchmark = self.benchmarks[benchmark_name]

        self.logger.info(f"开始运行基准测试: {benchmark_name}")

        # 预热运行
        self.logger.info(f"执行预热运行: {warmup_iterations} 次")
        for i in range(warmup_iterations):
            test_function()

        # 正式测量
        measurements = []
        system_metrics_during_test = []

        self.start_system_monitoring()

        try:
            for i in range(iterations):
                self.logger.info(f"执行测试迭代 {i + 1}/{iterations}")

                # 记录系统状态
                start_metrics = self._capture_system_metrics()

                # 执行测试并测量时间
                start_time = time.perf_counter()
                result = test_function()
                end_time = time.perf_counter()

                execution_time = (end_time - start_time) * 1000  # 转换为毫秒

                # 记录结束系统状态
                end_metrics = self._capture_system_metrics()

                measurements.append(execution_time)
                system_metrics_during_test.append({
                    'iteration': i,
                    'start_metrics': start_metrics,
                    'end_metrics': end_metrics,
                    'execution_time': execution_time
                })

        finally:
            self.stop_system_monitoring()

        # 计算统计结果
        if benchmark.metric_name.endswith('_time') or benchmark.metric_name in ['average_response_time', 'query_execution_time']:
            # 时间相关指标：使用平均值
            measured_value = statistics.mean(measurements)
        elif benchmark.metric_name in ['max_concurrent_users', 'requests_per_second']:
            # 吞吐量指标：使用最大值或平均值
            measured_value = statistics.mean(measurements)
        else:
            # 其他指标：使用平均值
            measured_value = statistics.mean(measurements)

        # 计算偏差
        deviation_percent = ((measured_value - benchmark.baseline_value) / benchmark.baseline_value) * 100

        # 确定状态
        status = self._determine_status(measured_value, benchmark)

        measurement = PerformanceMeasurement(
            benchmark_name=benchmark_name,
            metric_name=benchmark.metric_name,
            measured_value=measured_value,
            baseline_value=benchmark.baseline_value,
            tolerance_percent=benchmark.tolerance_percent,
            comparison_operator=benchmark.comparison_operator,
            status=status,
            deviation_percent=deviation_percent,
            timestamp=datetime.now(),
            sample_size=len(measurements),
            raw_measurements=measurements
        )

        self.measurements_history.append(measurement)

        self.logger.info(
            f"基准测试完成: {benchmark_name} = {measured_value:.2f}, "
            f"基线: {benchmark.baseline_value:.2f}, "
            f"偏差: {deviation_percent:.1f}%, "
            f"状态: {status.upper()}"
        )

        return measurement

    def run_multiple_benchmarks(self, benchmark_tests: Dict[str, Callable],
                            iterations: int = 10) -> List[PerformanceMeasurement]:
        """运行多个基准测试"""
        results = []

        for benchmark_name, test_function in benchmark_tests.items():
            try:
                result = self.run_benchmark(benchmark_name, test_function, iterations)
                results.append(result)
            except Exception as e:
                self.logger.error(f"运行基准测试 {benchmark_name} 时发生错误: {e}")
                # 创建错误结果
                benchmark = self.benchmarks.get(benchmark_name)
                if benchmark:
                    error_result = PerformanceMeasurement(
                        benchmark_name=benchmark_name,
                        metric_name=benchmark.metric_name,
                        measured_value=0.0,
                        baseline_value=benchmark.baseline_value,
                        tolerance_percent=benchmark.tolerance_percent,
                        comparison_operator=benchmark.comparison_operator,
                        status="error",
                        deviation_percent=0.0,
                        timestamp=datetime.now(),
                        sample_size=0,
                        raw_measurements=[]
                    )
                    results.append(error_result)

        return results

    def _determine_status(self, measured_value: float, benchmark: PerformanceBenchmark) -> str:
        """确定测试状态"""
        deviation_percent = abs(((measured_value - benchmark.baseline_value) / benchmark.baseline_value) * 100)

        # 检查是否在容差范围内
        if deviation_percent <= benchmark.tolerance_percent:
            return "pass"

        # 检查是否严重偏离（超过容差的2倍）
        if deviation_percent > benchmark.tolerance_percent * 2:
            return "fail"

        # 轻微偏离
        return "warning"

    def start_system_monitoring(self, interval: float = 1.0):
        """开始系统监控"""
        if self.system_monitoring_active:
            return

        self.system_monitoring_active = True
        self.system_metrics = []

        def monitor():
            while self.system_monitoring_active:
                metrics = self._capture_system_metrics()
                self.system_metrics.append(metrics)
                time.sleep(interval)

        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()

        self.logger.info("系统监控已启动")

    def stop_system_monitoring(self):
        """停止系统监控"""
        if not self.system_monitoring_active:
            return

        self.system_monitoring_active = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=2.0)

        self.logger.info(f"系统监控已停止，共收集 {len(self.system_metrics)} 个数据点")

    def _capture_system_metrics(self) -> SystemMetrics:
        """捕获系统指标"""
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()

        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_mb=memory.used / 1024 / 1024,  # MB
            disk_usage_percent=disk.percent,
            network_io={
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            },
            timestamp=datetime.now()
        )

    def check_performance_regression(self, benchmark_name: str,
                                recent_measurements: List[PerformanceMeasurement],
                                threshold: float = 10.0) -> Dict[str, Any]:
        """检查性能回归"""
        if benchmark_name not in self.benchmarks:
            return {"status": "unknown", "message": f"未知的基准测试: {benchmark_name}"}

        if len(recent_measurements) < 2:
            return {"status": "insufficient_data", "message": "数据不足，无法检测回归"}

        # 计算趋势
        values = [m.measured_value for m in recent_measurements[-10:]]  # 最近10个测量值
        if len(values) < 2:
            return {"status": "insufficient_data", "message": "历史数据不足"}

        # 计算移动平均
        recent_avg = statistics.mean(values[-3:])  # 最近3次的平均
        older_avg = statistics.mean(values[:-3]) if len(values) > 3 else values[0]

        # 计算变化百分比
        if older_avg == 0:
            change_percent = 0.0
        else:
            change_percent = ((recent_avg - older_avg) / older_avg) * 100

        # 判断是否为性能回归
        benchmark = self.benchmarks[benchmark_name]

        # 对于时间指标，正向变化（变慢）是坏的
        # 对于吞吐量指标，负向变化（变慢）是坏的
        if benchmark.metric_name.endswith('_time') or benchmark.metric_name in ['average_response_time', 'query_execution_time']:
            # 时间指标：增加表示性能变差
            is_regression = change_percent > threshold
            regression_type = "performance_degradation"
        else:
            # 吞吐量指标：减少表示性能变差
            is_regression = change_percent < -threshold
            regression_type = "throughput_decline"

        result = {
            "status": "regression_detected" if is_regression else "no_regression",
            "change_percent": change_percent,
            "threshold": threshold,
            "recent_average": recent_avg,
            "older_average": older_avg,
            "regression_type": regression_type if is_regression else None
        }

        if is_regression:
            self.logger.warning(
                f"检测到性能回归: {benchmark_name}, "
                f"变化: {change_percent:.1f}%, "
                f"阈值: {threshold}%"
            )

        return result

    def generate_performance_report(self, measurements: List[PerformanceMeasurement]) -> str:
        """生成性能报告"""
        report_lines = []
        report_lines.append("# 性能基准测试报告")
        report_lines.append(f"生成时间: {datetime.now().isoformat()}")
        report_lines.append("")

        # 总体统计
        total_tests = len(measurements)
        passed_tests = sum(1 for m in measurements if m.status == "pass")
        warning_tests = sum(1 for m in measurements if m.status == "warning")
        failed_tests = sum(1 for m in measurements if m.status == "fail")
        error_tests = sum(1 for m in measurements if m.status == "error")

        report_lines.append("## 总体统计")
        report_lines.append(f"- 总测试数: {total_tests}")
        report_lines.append(f"- 通过测试: {passed_tests}")
        report_lines.append(f"- 警告测试: {warning_tests}")
        report_lines.append(f"- 失败测试: {failed_tests}")
        report_lines.append(f"- 错误测试: {error_tests}")
        report_lines.append("")

        # 详细结果
        report_lines.append("## 详细结果")
        for measurement in measurements:
            status_emoji = {
                "pass": "✅",
                "warning": "⚠️",
                "fail": "❌",
                "error": "🔥"
            }.get(measurement.status, "❓")

            report_lines.append(f"### {status_emoji} {measurement.benchmark_name}")
            report_lines.append(f"- 指标: {measurement.metric_name}")
            report_lines.append(f"- 测量值: {measurement.measured_value:.2f}")
            report_lines.append(f"- 基线值: {measurement.baseline_value:.2f}")
            report_lines.append(f"- 偏差: {measurement.deviation_percent:.1f}%")
            report_lines.append(f"- 容差: {measurement.tolerance_percent:.1f}%")
            report_lines.append(f"- 状态: {measurement.status.upper()}")
            if measurement.sample_size > 1:
                report_lines.append(f"- 样本数: {measurement.sample_size}")
                report_lines.append(f"- 原始测量: {measurement.raw_measurements[:5]}...")  # 只显示前5个
            report_lines.append("")

        return "\n".join(report_lines)

    def save_performance_report(self, measurements: List[PerformanceMeasurement],
                            report_file: str = "performance_benchmark_report.md"):
        """保存性能报告"""
        report_path = project_root / "test_logs" / report_file
        report_path.parent.mkdir(exist_ok=True)

        report_content = self.generate_performance_report(measurements)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        self.logger.info(f"性能基准测试报告已保存到: {report_path}")

    def export_measurements_to_json(self, measurements: List[PerformanceMeasurement],
                                json_file: str = "performance_measurements.json"):
        """导出测量结果到JSON"""
        json_path = project_root / "test_logs" / json_file
        json_path.parent.mkdir(exist_ok=True)

        # 转换为可序列化的字典
        measurements_data = []
        for m in measurements:
            measurement_dict = {
                'benchmark_name': m.benchmark_name,
                'metric_name': m.metric_name,
                'measured_value': m.measured_value,
                'baseline_value': m.baseline_value,
                'tolerance_percent': m.tolerance_percent,
                'comparison_operator': m.comparison_operator,
                'status': m.status,
                'deviation_percent': m.deviation_percent,
                'timestamp': m.timestamp.isoformat(),
                'sample_size': m.sample_size,
                'raw_measurements': m.raw_measurements
            }
            measurements_data.append(measurement_dict)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(measurements_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"性能测量结果已导出到: {json_path}")


# 全局框架实例
performance_framework = PerformanceBenchmarkFramework()


def run_performance_benchmark(benchmark_name: str, test_function: Callable,
                            iterations: int = 10) -> PerformanceMeasurement:
    """运行性能基准测试的便捷函数"""
    return performance_framework.run_benchmark(benchmark_name, test_function, iterations)


def check_performance_regression(benchmark_name: str,
                            recent_measurements: List[PerformanceMeasurement],
                            threshold: float = 10.0) -> Dict[str, Any]:
    """检查性能回归的便捷函数"""
    return performance_framework.check_performance_regression(benchmark_name, recent_measurements, threshold)
