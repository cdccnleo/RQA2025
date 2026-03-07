#!/usr/bin/env python3
"""
RQA2026 性能基准测试框架

提供全面的性能测试和基准测试功能
"""

import asyncio
import time
import logging
import statistics
import psutil
from typing import Dict, List, Any, Optional, Callable
import numpy as np
from dataclasses import dataclass
from pathlib import Path

# 添加项目路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.rqa2026.quantum.portfolio_optimizer import (
        QuantumPortfolioOptimizer, AssetData, PortfolioConstraints
    )
    from src.rqa2026.ai.market_analyzer import MarketSentimentAnalyzer
    from src.rqa2026.bmi.signal_processor import RealtimeSignalProcessor
    from src.rqa2026.infrastructure.service_registry import ServiceRegistry

    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  某些组件不可用: {e}")
    COMPONENTS_AVAILABLE = False


@dataclass
class PerformanceResult:
    """性能测试结果"""
    operation: str
    iterations: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    median_time: float
    std_dev: float
    throughput: float  # operations per second
    memory_usage: float  # MB
    cpu_usage: float  # percentage
    success_rate: float  # percentage


@dataclass
class BenchmarkSuite:
    """基准测试套件"""
    name: str
    description: str
    test_cases: Dict[str, Callable]


class RQA2026PerformanceBenchmark:
    """
    RQA2026性能基准测试

    提供三大引擎的全面性能测试和优化建议
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.results: Dict[str, PerformanceResult] = {}

        # 初始化引擎
        if COMPONENTS_AVAILABLE:
            self.quantum_optimizer = QuantumPortfolioOptimizer(use_quantum=False)
            self.quantum_optimizer._initialized = True
            self.ai_analyzer = MarketSentimentAnalyzer()
            self.bmi_processor = RealtimeSignalProcessor()
            self.service_registry = ServiceRegistry()

    async def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """运行全面性能基准测试"""
        self.logger.info("🚀 开始RQA2026全面性能基准测试")

        results = {}

        # 量子引擎性能测试
        results["quantum_engine"] = await self._benchmark_quantum_engine()

        # AI引擎性能测试
        results["ai_engine"] = await self._benchmark_ai_engine()

        # BMI引擎性能测试
        results["bmi_engine"] = await self._benchmark_bmi_engine()

        # 基础设施性能测试
        results["infrastructure"] = await self._benchmark_infrastructure()

        # 系统集成性能测试
        results["system_integration"] = await self._benchmark_system_integration()

        # 生成性能报告
        report = self._generate_performance_report(results)

        self.logger.info("✅ 性能基准测试完成")
        return report

    async def _benchmark_quantum_engine(self) -> Dict[str, PerformanceResult]:
        """量子引擎性能测试"""
        self.logger.info("⚡ 测试量子引擎性能")

        results = {}

        # 投资组合优化测试
        async def portfolio_optimization_test():
            assets = [
                AssetData('AAPL', 0.12, 0.25, 150.0, [145, 148, 152, 149, 151]),
                AssetData('GOOGL', 0.10, 0.30, 2500.0, [2480, 2490, 2520, 2500, 2510]),
                AssetData('MSFT', 0.15, 0.28, 300.0, [295, 298, 305, 302, 299]),
                AssetData('NVDA', 0.18, 0.35, 400.0, [390, 395, 410, 405, 415]),
                AssetData('TSLA', 0.20, 0.45, 800.0, [780, 795, 820, 810, 825])
            ]
            constraints = PortfolioConstraints(min_weight=0.05, max_weight=0.4)
            return await self.quantum_optimizer.optimize_portfolio(assets, constraints)

        results["portfolio_optimization"] = await self._run_performance_test(
            "portfolio_optimization",
            portfolio_optimization_test,
            iterations=10
        )

        # 小规模测试 (快速验证)
        async def small_portfolio_test():
            assets = [
                AssetData('AAPL', 0.12, 0.25, 150.0, [145, 148, 152]),
                AssetData('GOOGL', 0.10, 0.30, 2500.0, [2480, 2490, 2520])
            ]
            constraints = PortfolioConstraints(min_weight=0.1, max_weight=0.8)
            return await self.quantum_optimizer.optimize_portfolio(assets, constraints)

        results["small_portfolio"] = await self._run_performance_test(
            "small_portfolio",
            small_portfolio_test,
            iterations=20
        )

        return results

    async def _benchmark_ai_engine(self) -> Dict[str, PerformanceResult]:
        """AI引擎性能测试"""
        self.logger.info("🤖 测试AI引擎性能")

        results = {}

        # 情绪分析测试
        async def sentiment_analysis_test():
            news_sources = [
                "Tech stocks rally as AI breakthrough announced",
                "Federal Reserve signals potential rate cut",
                "Market volatility increases amid economic uncertainty",
                "Strong earnings reports boost investor confidence",
                "Geopolitical tensions affect commodity prices"
            ]
            return await self.ai_analyzer.analyze_market_sentiment(news_sources)

        results["sentiment_analysis"] = await self._run_performance_test(
            "sentiment_analysis",
            sentiment_analysis_test,
            iterations=15
        )

        # 交易信号生成测试
        async def signal_generation_test():
            assets = ["AAPL", "GOOGL", "TSLA"]
            market_data = {
                "AAPL": {"prices": np.random.randn(50) + 150, "volume": np.random.randint(1000000, 2000000, 50)},
                "GOOGL": {"prices": np.random.randn(50) + 2500, "volume": np.random.randint(500000, 1000000, 50)},
                "TSLA": {"prices": np.random.randn(50) + 800, "volume": np.random.randint(2000000, 4000000, 50)}
            }
            sentiment_data = {
                "AAPL": {"news": ["AAPL shows strong growth", "Positive earnings report"]},
                "GOOGL": {"news": ["Google AI breakthrough", "Revenue beats expectations"]},
                "TSLA": {"news": ["EV market share increases", "Production ramp up"]}
            }
            return await self.ai_analyzer.generate_signals(assets, market_data, sentiment_data)

        results["signal_generation"] = await self._run_performance_test(
            "signal_generation",
            signal_generation_test,
            iterations=10
        )

        return results

    async def _benchmark_bmi_engine(self) -> Dict[str, PerformanceResult]:
        """BMI引擎性能测试"""
        self.logger.info("🧠 测试BMI引擎性能")

        results = {}

        # EEG信号处理测试
        async def eeg_processing_test():
            eeg_data = np.random.randn(32, 250)  # 32通道，1秒数据
            await self.bmi_processor.add_signal_data(eeg_data)
            return self.bmi_processor.get_signal_quality_metrics()

        results["eeg_processing"] = await self._run_performance_test(
            "eeg_processing",
            eeg_processing_test,
            iterations=25
        )

        # 意图识别测试
        async def intent_recognition_test():
            eeg_data = np.random.randn(32, 250) + np.sin(np.linspace(0, 10*np.pi, 250)) * 0.5
            await self.bmi_processor.add_signal_data(eeg_data)
            processed = await self.bmi_processor.process_eeg_data(eeg_data)
            return self.bmi_processor.classify_intent(processed)

        results["intent_recognition"] = await self._run_performance_test(
            "intent_recognition",
            intent_recognition_test,
            iterations=20
        )

        return results

    async def _benchmark_infrastructure(self) -> Dict[str, PerformanceResult]:
        """基础设施性能测试"""
        self.logger.info("🏗️ 测试基础设施性能")

        results = {}

        # 服务注册测试
        await self.service_registry.start()

        async def service_registration_test():
            from src.rqa2026.infrastructure.service_registry import ServiceInstance
            service = ServiceInstance(
                service_name="test-service",
                instance_id=f"test-{time.time()}",
                host="localhost",
                port=9999,
                metadata={"test": True}
            )
            return await self.service_registry.register_service(service)

        results["service_registration"] = await self._run_performance_test(
            "service_registration",
            service_registration_test,
            iterations=30
        )

        # 服务发现测试
        async def service_discovery_test():
            return await self.service_registry.discover_service("test-service")

        results["service_discovery"] = await self._run_performance_test(
            "service_discovery",
            service_discovery_test,
            iterations=50
        )

        await self.service_registry.stop()

        return results

    async def _benchmark_system_integration(self) -> Dict[str, PerformanceResult]:
        """系统集成性能测试"""
        self.logger.info("🔗 测试系统集成性能")

        results = {}

        # 端到端交易流程测试
        async def end_to_end_trading_test():
            # 1. 准备资产数据
            assets = [
                AssetData('AAPL', 0.12, 0.25, 150.0, [145, 148, 152]),
                AssetData('GOOGL', 0.10, 0.30, 2500.0, [2480, 2490, 2520]),
                AssetData('TSLA', 0.20, 0.45, 800.0, [780, 795, 820])
            ]
            constraints = PortfolioConstraints(min_weight=0.1, max_weight=0.6)

            # 2. 量子优化
            portfolio_result = await self.quantum_optimizer.optimize_portfolio(assets, constraints)

            # 3. AI分析
            sentiment_result = await self.ai_analyzer.analyze_market_sentiment([
                "Market shows bullish signals",
                "Strong economic indicators"
            ])

            # 4. BMI信号处理
            eeg_data = np.random.randn(16, 125)  # 16通道，0.5秒数据
            await self.bmi_processor.add_signal_data(eeg_data)
            signal_quality = self.bmi_processor.get_signal_quality_metrics()

            return {
                "portfolio": portfolio_result,
                "sentiment": sentiment_result,
                "signal_quality": signal_quality
            }

        results["end_to_end_trading"] = await self._run_performance_test(
            "end_to_end_trading",
            end_to_end_trading_test,
            iterations=5
        )

        return results

    async def _run_performance_test(
        self,
        operation: str,
        test_func: Callable,
        iterations: int = 10
    ) -> PerformanceResult:
        """运行性能测试"""
        self.logger.info(f"运行性能测试: {operation} ({iterations} 次)")

        execution_times = []
        memory_usage = []
        cpu_usage = []
        success_count = 0

        for i in range(iterations):
            try:
                # 记录开始时间和资源使用
                start_time = time.time()
                start_memory = psutil.virtual_memory().percent
                start_cpu = psutil.cpu_percent(interval=None)

                # 执行测试
                result = await test_func()

                # 记录结束时间和资源使用
                end_time = time.time()
                end_memory = psutil.virtual_memory().percent
                end_cpu = psutil.cpu_percent(interval=None)

                # 计算执行时间和资源使用
                execution_time = end_time - start_time
                avg_memory = (start_memory + end_memory) / 2
                avg_cpu = (start_cpu + end_cpu) / 2

                execution_times.append(execution_time)
                memory_usage.append(avg_memory)
                cpu_usage.append(avg_cpu)
                success_count += 1

                self.logger.debug(".4f"
            except Exception as e:
                self.logger.warning(f"测试 {operation} 第 {i+1} 次失败: {e}")
                # 记录失败的执行时间 (使用超时值)
                execution_times.append(10.0)  # 10秒超时
                memory_usage.append(psutil.virtual_memory().percent)
                cpu_usage.append(psutil.cpu_percent(interval=None))

        # 计算统计指标
        total_time = sum(execution_times)
        avg_time = statistics.mean(execution_times)
        min_time = min(execution_times)
        max_time = max(execution_times)
        median_time = statistics.median(execution_times)
        std_dev = statistics.stdev(execution_times) if len(execution_times) > 1 else 0
        throughput = iterations / total_time if total_time > 0 else 0
        success_rate = (success_count / iterations) * 100

        avg_memory = statistics.mean(memory_usage) if memory_usage else 0
        avg_cpu = statistics.mean(cpu_usage) if cpu_usage else 0

        result = PerformanceResult(
            operation=operation,
            iterations=iterations,
            total_time=total_time,
            avg_time=avg_time,
            min_time=min_time,
            max_time=max_time,
            median_time=median_time,
            std_dev=std_dev,
            throughput=throughput,
            memory_usage=avg_memory,
            cpu_usage=avg_cpu,
            success_rate=success_rate
        )

        self.results[operation] = result
        return result

    def _generate_performance_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """生成性能报告"""
        self.logger.info("生成性能基准测试报告")

        report = {
            "summary": {},
            "detailed_results": results,
            "recommendations": [],
            "performance_score": {},
            "timestamp": time.time()
        }

        # 计算总体性能指标
        all_results = []
        for engine_results in results.values():
            if isinstance(engine_results, dict):
                all_results.extend(engine_results.values())

        if all_results:
            avg_throughput = statistics.mean([r.throughput for r in all_results])
            avg_response_time = statistics.mean([r.avg_time for r in all_results])
            avg_success_rate = statistics.mean([r.success_rate for r in all_results])
            avg_memory_usage = statistics.mean([r.memory_usage for r in all_results])
            avg_cpu_usage = statistics.mean([r.cpu_usage for r in all_results])

            report["summary"] = {
                "total_operations": len(all_results),
                "average_throughput": avg_throughput,
                "average_response_time": avg_response_time,
                "average_success_rate": avg_success_rate,
                "average_memory_usage": avg_memory_usage,
                "average_cpu_usage": avg_cpu_usage
            }

            # 生成性能评分 (0-100)
            throughput_score = min(100, avg_throughput * 10)  # 假设10 ops/sec 为满分
            response_time_score = max(0, 100 - avg_response_time * 100)  # 响应时间越短分数越高
            success_rate_score = avg_success_rate
            resource_score = 100 - (avg_memory_usage + avg_cpu_usage) / 2

            overall_score = (throughput_score * 0.3 + response_time_score * 0.3 +
                           success_rate_score * 0.2 + resource_score * 0.2)

            report["performance_score"] = {
                "throughput_score": throughput_score,
                "response_time_score": response_time_score,
                "success_rate_score": success_rate_score,
                "resource_efficiency_score": resource_score,
                "overall_score": overall_score
            }

            # 生成优化建议
            recommendations = []

            if avg_response_time > 1.0:
                recommendations.append("响应时间较长，建议优化算法复杂度或启用缓存")
            if avg_success_rate < 95:
                recommendations.append("成功率偏低，建议加强错误处理和异常恢复")
            if avg_memory_usage > 80:
                recommendations.append("内存使用率较高，建议优化内存管理或增加缓存")
            if avg_cpu_usage > 70:
                recommendations.append("CPU使用率较高，建议优化并发处理或增加计算资源")
            if avg_throughput < 5:
                recommendations.append("吞吐量偏低，建议优化并发处理或使用异步架构")

            report["recommendations"] = recommendations

        return report

    def save_report(self, report: Dict[str, Any], output_file: str = "performance_report.json"):
        """保存性能报告"""
        import json
        from pathlib import Path

        output_path = Path("test_logs") / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        self.logger.info(f"性能报告已保存至: {output_path}")

    def print_report(self, report: Dict[str, Any]):
        """打印性能报告"""
        print("\\n" + "="*80)
        print("🎯 RQA2026性能基准测试报告")
        print("="*80)

        if "summary" in report:
            summary = report["summary"]
            print("\\n📊 总体性能指标:")
            print(".2f"            print(".4f"            print(".1f"            print(".1f"            print(".1f"        if "performance_score" in report:
            scores = report["performance_score"]
            print("\\n🏆 性能评分 (0-100):")
            print(".1f"            print(".1f"            print(".1f"            print(".1f"            print(".1f"            # 评分等级
            overall_score = scores["overall_score"]
            if overall_score >= 90:
                grade = "优秀 (A+)"
            elif overall_score >= 80:
                grade = "良好 (A)"
            elif overall_score >= 70:
                grade = "中等 (B)"
            elif overall_score >= 60:
                grade = "及格 (C)"
            else:
                grade = "需改进 (D)"

            print(f"   等级评定: {grade}")

        if "recommendations" in report and report["recommendations"]:
            print("\\n💡 优化建议:")
            for i, rec in enumerate(report["recommendations"], 1):
                print(f"   {i}. {rec}")

        print("\\n" + "="*80)


async def main():
    """主函数"""
    if not COMPONENTS_AVAILABLE:
        print("❌ RQA2026组件不可用，无法运行性能测试")
        return

    # 创建性能基准测试器
    benchmark = RQA2026PerformanceBenchmark()

    try:
        # 运行全面性能测试
        print("🚀 启动RQA2026性能基准测试...")
        report = await benchmark.run_comprehensive_benchmark()

        # 保存和显示报告
        benchmark.save_report(report)
        benchmark.print_report(report)

        print("\\n✅ 性能基准测试完成！")
        print("📄 详细报告已保存至: test_logs/performance_report.json")

    except Exception as e:
        print(f"❌ 性能测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 运行性能测试
    asyncio.run(main())