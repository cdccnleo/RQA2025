#!/usr/bin/env python3
"""
RQA2025量化交易系统性能基准测试
测试系统核心组件的性能表现
"""
from tests.unit.strategy.test_execution_engine_mock import (
    MockExecutionEngine, MockExecutionConfig,
    MockExecutionOrder
)
from tests.unit.strategy.test_real_time_data_stream_mock import (
    MockStreamProcessor, MockStreamConfig,
    MockStreamData
)
from performance_benchmark import PerformanceBenchmark
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class RQASystemBenchmark:
    """RQA系统性能基准测试"""

    def __init__(self):
        self.benchmark = PerformanceBenchmark()
        self.results = {}

    async def run_all_benchmarks(self):
        """运行所有性能基准测试"""
        print("🚀 RQA2025量化交易系统性能基准测试开始")
        print("=" * 80)

        # 1. 数据流处理器性能测试
        print("\n📊 1. 数据流处理器性能测试")
        await self.benchmark_stream_processor()

        # 2. 数据流过滤器性能测试
        print("\n🔍 2. 数据流过滤器性能测试")
        await self.benchmark_stream_filter()

        # 3. 数据流聚合器性能测试
        print("\n📈 3. 数据流聚合器性能测试")
        await self.benchmark_stream_aggregator()

        # 4. 执行引擎性能测试
        print("\n⚡ 4. 执行引擎性能测试")
        await self.benchmark_execution_engine()

        # 5. 内存使用测试
        print("\n💾 5. 内存使用测试")
        self.benchmark_memory_usage()

        # 保存结果
        self.save_results()

        # 生成报告
        self.generate_report()

    async def benchmark_stream_processor(self):
        """数据流处理器性能测试"""
        config = MockStreamConfig(stream_id="perf_test_processor")

        async def process_data():
            processor = MockStreamProcessor(config)
            await processor.start()

            data = MockStreamData(
                symbol="AAPL",
                timestamp=asyncio.get_event_loop().time(),
                data_type="quote",
                values={"price": 150.0, "volume": 1000}
            )

            await processor.process_data(data)
            await processor.stop()

        results = self.benchmark.benchmark_async_function(
            process_data,
            iterations=1000,
            concurrency=5,
            warmup_iterations=100
        )

        self.results['stream_processor'] = results
        self.benchmark.print_results(results, "数据流处理器性能")

    async def benchmark_stream_filter(self):
        """数据流过滤器性能测试"""
        from tests.unit.strategy.test_real_time_data_stream_mock import MockStreamFilter

        filter_obj = MockStreamFilter()

        # 添加过滤器
        async def price_filter(data):
            return data.values.get('price', 0) > 100

        filter_obj.add_filter(price_filter, "price_filter")

        async def filter_data():
            data = MockStreamData(
                symbol="AAPL",
                timestamp=asyncio.get_event_loop().time(),
                data_type="quote",
                values={"price": 150.0, "volume": 1000}
            )
            await filter_obj.apply_filters(data)

        results = self.benchmark.benchmark_async_function(
            filter_data,
            iterations=2000,
            concurrency=10,
            warmup_iterations=200
        )

        self.results['stream_filter'] = results
        self.benchmark.print_results(results, "数据流过滤器性能")

    async def benchmark_stream_aggregator(self):
        """数据流聚合器性能测试"""
        from tests.unit.strategy.test_real_time_data_stream_mock import MockStreamAggregator

        aggregator = MockStreamAggregator(window_size=10)

        async def aggregate_data():
            data = MockStreamData(
                symbol="AAPL",
                timestamp=asyncio.get_event_loop().time(),
                data_type="quote",
                values={"price": 150.0 + asyncio.get_event_loop().time() % 10, "volume": 1000}
            )
            await aggregator.add_data(data)

        results = self.benchmark.benchmark_async_function(
            aggregate_data,
            iterations=1000,
            concurrency=5,
            warmup_iterations=100
        )

        self.results['stream_aggregator'] = results
        self.benchmark.print_results(results, "数据流聚合器性能")

    async def benchmark_execution_engine(self):
        """执行引擎性能测试"""
        config = MockExecutionConfig(engine_id="perf_test_engine")

        async def execute_order():
            engine = MockExecutionEngine(config, auto_execute=True)
            await engine.initialize()

            order = MockExecutionOrder(
                order_id=f"ORD_{asyncio.get_event_loop().time()}",
                symbol="AAPL",
                side="BUY",
                quantity=100,
                execution_algorithm="market"
            )

            await engine.submit_order(order)
            await asyncio.sleep(0.01)  # 等待执行完成

        results = self.benchmark.benchmark_async_function(
            execute_order,
            iterations=500,
            concurrency=3,
            warmup_iterations=50
        )

        self.results['execution_engine'] = results
        self.benchmark.print_results(results, "执行引擎性能")

    def benchmark_memory_usage(self):
        """内存使用基准测试"""
        print("测试数据结构内存使用...")

        # 测试数据流数据内存使用
        def create_stream_data():
            return MockStreamData(
                symbol="AAPL",
                timestamp=asyncio.get_event_loop().time(),
                data_type="quote",
                values={"price": 150.0, "volume": 1000}
            )

        results = self.benchmark.benchmark_memory_usage(
            create_stream_data,
            iterations=10000
        )

        self.results['memory_usage'] = results
        self.benchmark.print_results(results, "内存使用性能")

    def save_results(self):
        """保存测试结果"""
        self.benchmark.save_results(self.results, "rqa_system_benchmark_baseline.json")
        print("\n💾 基准测试结果已保存到 rqa_system_benchmark_baseline.json")

    def generate_report(self):
        """生成性能报告"""
        print("\n" + "="*80)
        print("📋 RQA2025性能基准测试报告")
        print("="*80)

        print("\n🎯 性能指标汇总:")
        print(
            f"   • 数据流处理器: {self.results.get('stream_processor', {}).get('avg_latency', 'N/A'):.2f}ms 平均延迟")
        print(
            f"   • 数据流过滤器: {self.results.get('stream_filter', {}).get('throughput', 'N/A'):.0f} ops/sec 吞吐量")
        print(
            f"   • 数据流聚合器: {self.results.get('stream_aggregator', {}).get('avg_latency', 'N/A'):.2f}ms 平均延迟")
        print(
            f"   • 执行引擎: {self.results.get('execution_engine', {}).get('avg_latency', 'N/A'):.2f}ms 平均延迟")

        memory_results = self.results.get('memory_usage', {})
        if memory_results:
            print("\n💾 内存使用:")
            print(f"   • 峰值内存: {memory_results.get('memory_peak', 0):,} bytes")
            print(f"   • 每次操作内存: {memory_results.get('avg_memory_per_operation', 0):.0f} bytes")

        print("\n⚠️  性能优化目标:")
        print("   • 响应时间: < 10ms")
        print("   • 吞吐量: > 1000 TPS")
        print("   • 内存使用: < 500MB")
        print("   • CPU使用: < 30%")

        print("\n🚀 优化方向:")
        print("   • Phase 16.1: 异步架构深度优化")
        print("   • Phase 16.2: 内存管理深度优化")
        print("   • Phase 16.3: 数据处理性能优化")
        print("   • Phase 16.4: 缓存策略深度优化")
        print("   • Phase 16.5: 数据库性能深度优化")

        print("\n" + "="*80)


def main():
    """主函数"""
    try:
        # 检查是否已经在事件循环中
        loop = asyncio.get_running_loop()
        print("⚠️  检测到已在运行的事件循环，使用同步方式运行测试")

        # 创建新的事件循环来运行异步测试
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)

        try:
            benchmark = RQASystemBenchmark()
            new_loop.run_until_complete(benchmark.run_all_benchmarks())
        finally:
            new_loop.close()

    except RuntimeError:
        # 不在事件循环中，可以直接运行
        benchmark = RQASystemBenchmark()
        asyncio.run(benchmark.run_all_benchmarks())


if __name__ == "__main__":
    main()
