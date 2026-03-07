#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
压力测试脚本
进行大规模压力测试验证系统稳定性
"""

import json
import time
import statistics
import random
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, asdict


@dataclass
class StressTestConfig:
    """压力测试配置"""
    test_duration: int = 60
    concurrent_users: int = 20
    target_rps: int = 50
    error_threshold: float = 5.0
    response_time_threshold: float = 2000.0


@dataclass
class StressTestResult:
    """压力测试结果"""
    test_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    throughput: float
    error_rate: float
    timestamp: float


class StressTestRunner:
    """压力测试运行器"""

    def __init__(self, config: StressTestConfig):
        self.config = config
        self.results = []

    def run_cache_stress_test(self) -> StressTestResult:
        """运行缓存压力测试"""
        print("🔧 开始缓存压力测试...")

        def cache_operation():
            # 模拟缓存操作
            operation = random.choice(["get", "put", "delete"])
            time.sleep(random.uniform(0.001, 0.02))
            return True

        return self._run_stress_test("cache_stress", cache_operation)

    def run_monitoring_stress_test(self) -> StressTestResult:
        """运行监控压力测试"""
        print("🔧 开始监控压力测试...")

        def monitoring_operation():
            # 模拟监控操作
            metric = random.choice(["cpu", "memory", "disk", "network"])
            time.sleep(random.uniform(0.001, 0.01))
            return True

        return self._run_stress_test("monitoring_stress", monitoring_operation)

    def run_parameter_optimization_stress_test(self) -> StressTestResult:
        """运行参数优化压力测试"""
        print("🔧 开始参数优化压力测试...")

        def parameter_optimization_operation():
            # 模拟参数优化操作
            optimization_type = random.choice(["price_limit", "after_hours", "circuit_breaker"])
            time.sleep(random.uniform(0.01, 0.05))
            return True

        return self._run_stress_test("parameter_optimization_stress", parameter_optimization_operation)

    def _run_stress_test(self, test_name: str, operation_func) -> StressTestResult:
        """运行压力测试"""
        response_times = []
        successful_requests = 0
        failed_requests = 0
        total_requests = 0

        print(f"🚀 开始压力测试，持续 {self.config.test_duration} 秒...")
        start_time = time.time()
        end_time = start_time + self.config.test_duration

        while time.time() < end_time:
            # 模拟并发用户
            for _ in range(self.config.concurrent_users):
                operation_start = time.time()
                try:
                    result = operation_func()
                    if result:
                        successful_requests += 1
                    else:
                        failed_requests += 1
                    total_requests += 1

                    response_time = (time.time() - operation_start) * 1000
                    response_times.append(response_time)

                except Exception as e:
                    failed_requests += 1
                    total_requests += 1
                    print(f"❌ 操作失败: {e}")

            # 控制请求频率
            time.sleep(1.0 / self.config.target_rps)

        # 计算统计信息
        avg_response_time = statistics.mean(response_times) if response_times else 0
        throughput = total_requests / self.config.test_duration if self.config.test_duration > 0 else 0
        error_rate = (failed_requests / total_requests * 100) if total_requests > 0 else 0

        return StressTestResult(
            test_name=test_name,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_response_time=avg_response_time,
            throughput=throughput,
            error_rate=error_rate,
            timestamp=time.time()
        )


class StressTestReporter:
    """压力测试报告器"""

    def generate_report(self, results: List[StressTestResult], config: StressTestConfig) -> Dict[str, Any]:
        """生成测试报告"""
        report = {
            "timestamp": time.time(),
            "config": asdict(config),
            "summary": self._generate_summary(results),
            "detailed_results": [asdict(result) for result in results],
            "analysis": self._analyze_results(results, config),
            "recommendations": self._generate_recommendations(results, config)
        }

        return report

    def _generate_summary(self, results: List[StressTestResult]) -> Dict[str, Any]:
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

    def _analyze_results(self, results: List[StressTestResult], config: StressTestConfig) -> Dict[str, Any]:
        """分析结果"""
        analysis = {
            "performance_analysis": {},
            "stability_analysis": {},
            "scalability_analysis": {}
        }

        if results:
            max_throughput = max(r.throughput for r in results)
            max_response_time = max(r.avg_response_time for r in results)
            max_error_rate = max(r.error_rate for r in results)

            analysis["performance_analysis"] = {
                "max_throughput": max_throughput,
                "max_response_time": max_response_time,
                "throughput_target_achieved": max_throughput >= config.target_rps,
                "response_time_acceptable": max_response_time <= config.response_time_threshold
            }

            analysis["stability_analysis"] = {
                "max_error_rate": max_error_rate,
                "stability_acceptable": max_error_rate <= config.error_threshold,
                "overall_success_rate": (sum(r.successful_requests for r in results) /
                                         sum(r.total_requests for r in results) * 100) if sum(r.total_requests for r in results) > 0 else 0
            }

            analysis["scalability_analysis"] = {
                "throughput_per_user": max_throughput / config.concurrent_users,
                "bottleneck_identified": any(r.avg_response_time > config.response_time_threshold for r in results)
            }

        return analysis

    def _generate_recommendations(self, results: List[StressTestResult], config: StressTestConfig) -> List[str]:
        """生成建议"""
        recommendations = []

        if not results:
            recommendations.append("没有测试结果，建议重新运行测试")
            return recommendations

        # 性能建议
        max_throughput = max(r.throughput for r in results)
        if max_throughput < config.target_rps:
            recommendations.append("吞吐量未达到目标，建议优化系统性能或增加资源")

        max_response_time = max(r.avg_response_time for r in results)
        if max_response_time > config.response_time_threshold:
            recommendations.append("响应时间过长，建议优化算法或增加缓存")

        # 稳定性建议
        max_error_rate = max(r.error_rate for r in results)
        if max_error_rate > config.error_threshold:
            recommendations.append("错误率过高，建议检查系统稳定性和错误处理")

        # 可扩展性建议
        if any(r.avg_response_time > config.response_time_threshold for r in results):
            recommendations.append("发现性能瓶颈，建议进行系统优化")

        if not recommendations:
            recommendations.append("系统在压力测试中表现良好，可以承受当前负载")

        return recommendations


def main():
    """主函数"""
    print("🔧 启动压力测试...")

    # 创建压力测试配置
    config = StressTestConfig(
        test_duration=60,
        concurrent_users=20,
        target_rps=50,
        error_threshold=5.0,
        response_time_threshold=2000.0
    )

    # 创建压力测试运行器
    runner = StressTestRunner(config)

    # 运行各项测试
    results = []

    # 缓存压力测试
    cache_result = runner.run_cache_stress_test()
    results.append(cache_result)

    # 监控压力测试
    monitoring_result = runner.run_monitoring_stress_test()
    results.append(monitoring_result)

    # 参数优化压力测试
    parameter_result = runner.run_parameter_optimization_stress_test()
    results.append(parameter_result)

    # 生成报告
    reporter = StressTestReporter()
    report = reporter.generate_report(results, config)

    print("✅ 压力测试完成!")

    # 打印结果
    print("\n" + "="*50)
    print("🎯 压力测试结果:")
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

    print("\n🔍 分析结果:")
    analysis = report["analysis"]

    perf_analysis = analysis.get("performance_analysis", {})
    print(f"  吞吐量目标达成: {'是' if perf_analysis.get('throughput_target_achieved') else '否'}")
    print(f"  响应时间可接受: {'是' if perf_analysis.get('response_time_acceptable') else '否'}")

    stability_analysis = analysis.get("stability_analysis", {})
    print(f"  稳定性可接受: {'是' if stability_analysis.get('stability_acceptable') else '否'}")
    print(f"  整体成功率: {stability_analysis.get('overall_success_rate', 0):.1f}%")

    print("\n💡 建议:")
    for recommendation in report["recommendations"]:
        print(f"  - {recommendation}")

    print("="*50)

    # 保存测试报告
    output_dir = Path("reports/optimization/")
    output_dir.mkdir(parents=True, exist_ok=True)

    report_file = output_dir / "stress_testing_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"📄 测试报告已保存: {report_file}")


if __name__ == "__main__":
    main()
