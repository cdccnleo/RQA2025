#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能基准测试脚本
进行全面的性能基准测试
"""

import json
import time
import statistics
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, asdict


@dataclass
class BenchmarkConfig:
    """基准测试配置"""
    test_duration: int = 30
    concurrent_users: int = 5
    request_interval: float = 0.5
    warmup_duration: int = 5
    cooldown_duration: int = 3


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    test_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    throughput: float
    error_rate: float
    timestamp: float


class PerformanceBenchmark:
    """性能基准测试器"""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = []

    def benchmark_cache_performance(self) -> BenchmarkResult:
        """测试缓存性能"""
        print("🔧 开始缓存性能测试...")

        # 模拟缓存操作
        def cache_operation():
            time.sleep(0.01)  # 模拟缓存操作时间
            return True

        return self._run_benchmark("cache_performance", cache_operation)

    def benchmark_monitoring_performance(self) -> BenchmarkResult:
        """测试监控性能"""
        print("🔧 开始监控性能测试...")

        # 模拟监控操作
        def monitoring_operation():
            time.sleep(0.005)  # 模拟监控操作时间
            return True

        return self._run_benchmark("monitoring_performance", monitoring_operation)

    def benchmark_parameter_optimization_performance(self) -> BenchmarkResult:
        """测试参数优化性能"""
        print("🔧 开始参数优化性能测试...")

        # 模拟参数优化操作
        def parameter_optimization_operation():
            time.sleep(0.02)  # 模拟参数优化计算时间
            return True

        return self._run_benchmark("parameter_optimization_performance", parameter_optimization_operation)

    def _run_benchmark(self, test_name: str, operation_func) -> BenchmarkResult:
        """运行基准测试"""
        response_times = []
        successful_requests = 0
        failed_requests = 0
        total_requests = 0

        # 预热
        print(f"🔥 预热 {self.config.warmup_duration} 秒...")
        time.sleep(self.config.warmup_duration)

        # 开始测试
        print(f"🚀 开始性能测试，持续 {self.config.test_duration} 秒...")
        start_time = time.time()
        end_time = start_time + self.config.test_duration

        while time.time() < end_time:
            # 执行操作
            operation_start = time.time()
            try:
                result = operation_func()
                if result:
                    successful_requests += 1
                else:
                    failed_requests += 1
                total_requests += 1

                response_time = (time.time() - operation_start) * 1000  # 转换为毫秒
                response_times.append(response_time)

            except Exception as e:
                failed_requests += 1
                total_requests += 1
                print(f"❌ 操作失败: {e}")

            # 控制请求间隔
            time.sleep(self.config.request_interval)

        # 冷却
        print(f"❄️ 冷却 {self.config.cooldown_duration} 秒...")
        time.sleep(self.config.cooldown_duration)

        # 计算统计信息
        avg_response_time = statistics.mean(response_times) if response_times else 0

        # 计算吞吐量和错误率
        test_duration = self.config.test_duration
        throughput = total_requests / test_duration if test_duration > 0 else 0
        error_rate = (failed_requests / total_requests * 100) if total_requests > 0 else 0

        return BenchmarkResult(
            test_name=test_name,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_response_time=avg_response_time,
            throughput=throughput,
            error_rate=error_rate,
            timestamp=time.time()
        )


class BenchmarkReporter:
    """基准测试报告器"""

    def generate_report(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """生成测试报告"""
        report = {
            "timestamp": time.time(),
            "summary": self._generate_summary(results),
            "detailed_results": [asdict(result) for result in results],
            "recommendations": self._generate_recommendations(results)
        }

        return report

    def _generate_summary(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """生成摘要"""
        if not results:
            return {}

        total_requests = sum(r.total_requests for r in results)
        total_successful = sum(r.successful_requests for r in results)
        total_failed = sum(r.failed_requests for r in results)

        avg_throughput = statistics.mean([r.throughput for r in results])
        avg_error_rate = statistics.mean([r.error_rate for r in results])
        avg_response_time = statistics.mean([r.avg_response_time for r in results])

        return {
            "total_requests": total_requests,
            "total_successful": total_successful,
            "total_failed": total_failed,
            "success_rate": (total_successful / total_requests * 100) if total_requests > 0 else 0,
            "avg_throughput": avg_throughput,
            "avg_error_rate": avg_error_rate,
            "avg_response_time": avg_response_time,
            "test_count": len(results)
        }

    def _generate_recommendations(self, results: List[BenchmarkResult]) -> List[str]:
        """生成建议"""
        recommendations = []

        for result in results:
            if result.error_rate > 5.0:
                recommendations.append(
                    f"{result.test_name}: 错误率过高({result.error_rate:.1f}%)，建议检查系统稳定性")

            if result.avg_response_time > 1000:
                recommendations.append(
                    f"{result.test_name}: 平均响应时间过长({result.avg_response_time:.1f}ms)，建议优化性能")

            if result.throughput < 10:
                recommendations.append(
                    f"{result.test_name}: 吞吐量较低({result.throughput:.1f} req/s)，建议增加并发能力")

        if not recommendations:
            recommendations.append("所有测试项目性能表现良好")

        return recommendations


def main():
    """主函数"""
    print("🔧 启动性能基准测试...")

    # 创建基准测试配置
    config = BenchmarkConfig(
        test_duration=30,
        concurrent_users=5,
        request_interval=0.5,
        warmup_duration=5,
        cooldown_duration=3
    )

    # 创建基准测试器
    benchmark = PerformanceBenchmark(config)

    # 运行各项测试
    results = []

    # 缓存性能测试
    cache_result = benchmark.benchmark_cache_performance()
    results.append(cache_result)

    # 监控性能测试
    monitoring_result = benchmark.benchmark_monitoring_performance()
    results.append(monitoring_result)

    # 参数优化性能测试
    parameter_result = benchmark.benchmark_parameter_optimization_performance()
    results.append(parameter_result)

    # 生成报告
    reporter = BenchmarkReporter()
    report = reporter.generate_report(results)

    print("✅ 性能基准测试完成!")

    # 打印结果
    print("\n" + "="*50)
    print("🎯 性能测试结果:")
    print("="*50)

    summary = report["summary"]
    print(f"总请求数: {summary['total_requests']}")
    print(f"成功率: {summary['success_rate']:.1f}%")
    print(f"平均吞吐量: {summary['avg_throughput']:.1f} req/s")
    print(f"平均响应时间: {summary['avg_response_time']:.1f}ms")
    print(f"平均错误率: {summary['avg_error_rate']:.1f}%")

    print("\n📊 详细结果:")
    for result in results:
        print(f"\n{result.test_name}:")
        print(f"  请求数: {result.total_requests}")
        print(
            f"  成功率: {(result.successful_requests/result.total_requests*100):.1f}%" if result.total_requests > 0 else "  成功率: 0%")
        print(f"  吞吐量: {result.throughput:.1f} req/s")
        print(f"  平均响应时间: {result.avg_response_time:.1f}ms")
        print(f"  错误率: {result.error_rate:.1f}%")

    print("\n💡 建议:")
    for recommendation in report["recommendations"]:
        print(f"  - {recommendation}")

    print("="*50)

    # 保存测试报告
    output_dir = Path("reports/optimization/")
    output_dir.mkdir(parents=True, exist_ok=True)

    report_file = output_dir / "performance_benchmark_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"📄 测试报告已保存: {report_file}")


if __name__ == "__main__":
    main()
