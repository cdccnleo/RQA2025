"""
performance_optimizer 模块

提供 performance_optimizer 相关功能和接口。
"""

import sys

import gc
import traceback
import psutil
import threading
import time

# from infrastructure.cache.cache_components import CacheComponentFactory  # Disabled: causing import issues
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Any
from src.infrastructure.constants import PerformanceConstants, SizeConstants
#!/usr/bin/env python3
"""
基础设施层性能优化系统

基于统一ComponentFactory架构进行性能优化：
1. 内存优化 - 对象池化、GC调优
2. CPU优化 - 异步处理、并发优化
3. I/O优化 - 连接池、缓存策略
4. 算法优化 - 数据结构、查找算法

作者: RQA2025 Team
版本: 1.0.0
更新: 2025年9月21日
"""


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    timestamp: float
    memory_usage: float  # MB
    cpu_usage: float     # %
    response_time: float  # ms
    throughput: float    # ops/sec
    error_rate: float    # %


@dataclass
class OptimizationResult:
    """优化结果数据类"""
    optimization_type: str
    before_metrics: PerformanceMetrics
    after_metrics: PerformanceMetrics
    improvement_percentage: float
    description: str


class ComponentFactoryPerformanceOptimizer:
    """ComponentFactory性能优化器"""

    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.optimization_results: List[OptimizationResult] = []
        self.object_pool: Dict[str, List[Any]] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)

    @staticmethod
    def _calculate_improvement(before: float, after: float, *, higher_is_better: bool) -> float:
        """安全计算改善比例，避免除零"""
        if before == 0:
            if higher_is_better:
                return 100.0 if after > 0 else 0.0
            return 0.0

        if higher_is_better:
            return ((after - before) / before) * 100
        return ((before - after) / before) * 100

    def optimize_memory_usage(self) -> OptimizationResult:
        """优化内存使用"""

        print("🧠 优化内存使用...")

        # 记录优化前指标
        before_metrics = self._collect_performance_metrics()

        # 1. 实现对象池化
        self._implement_object_pooling()

        # 2. 优化垃圾回收
        self._optimize_garbage_collection()

        # 3. 减少内存碎片
        self._reduce_memory_fragmentation()

        # 记录优化后指标
        after_metrics = self._collect_performance_metrics()

        improvement = self._calculate_improvement(
            before_metrics.memory_usage,
            after_metrics.memory_usage,
            higher_is_better=False
        )

        result = OptimizationResult(
            optimization_type="memory_optimization",
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            improvement_percentage=improvement,
            description=f"内存使用优化: {improvement:.1f}% 改善"
        )

        self.optimization_results.append(result)
        return result

    def optimize_cpu_usage(self) -> OptimizationResult:
        """优化CPU使用"""

        print("⚡ 优化CPU使用...")

        before_metrics = self._collect_performance_metrics()

        # 1. 异步处理优化
        self._optimize_async_processing()

        # 2. 并发优化
        self._optimize_concurrency()

        # 3. 算法优化
        self._optimize_algorithms()

        after_metrics = self._collect_performance_metrics()

        improvement = self._calculate_improvement(
            before_metrics.cpu_usage,
            after_metrics.cpu_usage,
            higher_is_better=False
        )

        result = OptimizationResult(
            optimization_type="cpu_optimization",
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            improvement_percentage=improvement,
            description=f"CPU使用优化: {improvement:.1f}% 改善"
        )

        self.optimization_results.append(result)
        return result

    def optimize_io_operations(self) -> OptimizationResult:
        """优化I/O操作"""

        print("💾 优化I/O操作...")

        before_metrics = self._collect_performance_metrics()

        # 1. 连接池优化
        self._optimize_connection_pooling()

        # 2. 缓存策略优化
        self._optimize_cache_strategy()

        # 3. 批量操作优化
        self._optimize_batch_operations()

        after_metrics = self._collect_performance_metrics()

        # I/O优化的衡量标准是响应时间改善
        improvement = self._calculate_improvement(
            before_metrics.response_time,
            after_metrics.response_time,
            higher_is_better=False
        )

        result = OptimizationResult(
            optimization_type="io_optimization",
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            improvement_percentage=improvement,
            description=f"I/O操作优化: {improvement:.1f}% 响应时间改善"
        )

        self.optimization_results.append(result)
        return result

    def optimize_data_structures(self) -> OptimizationResult:
        """优化数据结构"""

        print("📊 优化数据结构...")

        before_metrics = self._collect_performance_metrics()

        # 1. 字典优化
        self._optimize_dictionaries()

        # 2. 列表操作优化
        self._optimize_list_operations()

        # 3. 集合操作优化
        self._optimize_set_operations()

        after_metrics = self._collect_performance_metrics()

        # 数据结构优化的衡量标准是吞吐量改善
        improvement = self._calculate_improvement(
            before_metrics.throughput,
            after_metrics.throughput,
            higher_is_better=True
        )

        result = OptimizationResult(
            optimization_type="data_structure_optimization",
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            improvement_percentage=improvement,
            description=f"数据结构优化: {improvement:.1f}% 吞吐量改善"
        )

        self.optimization_results.append(result)
        return result

    def _implement_object_pooling(self):
        """实现对象池化"""

        # 为常用的ComponentFactory子类创建对象池
        common_classes = [
            "infrastructure.cache.cache_components.CacheComponentFactory",
            "infrastructure.health.health_components.HealthComponentFactory",
            "infrastructure.utils.util_components.UtilComponentFactory"
        ]

        for class_path in common_classes:
            try:
                module_path, class_name = class_path.rsplit(".", 1)
                module = __import__(module_path, fromlist=[class_name])
                cls = getattr(module, class_name)

                # 创建对象池
                pool_key = class_path
                if pool_key not in self.object_pool:
                    self.object_pool[pool_key] = []

                # 预创建一些对象
                for _ in range(5):  # 池中保持5个对象
                    obj = cls()
                    self.object_pool[pool_key].append(obj)

            except Exception as e:
                print(f"  ⚠️ 对象池化失败 {class_path}: {e}")

    def _optimize_garbage_collection(self):
        """优化垃圾回收"""

        # 调整GC参数
        gc.set_threshold(700, 10, 10)  # 降低GC触发频率

        # 手动GC
        collected = gc.collect()
        print(f"  🗑️ 手动GC回收了 {collected} 个对象")

    def _reduce_memory_fragmentation(self):
        """减少内存碎片"""

        # 重新分配大对象
        large_objects = [obj for obj in gc.get_objects() if sys.getsizeof(obj) > SizeConstants.LARGE_OBJECT]
        print(f"  📏 发现 {len(large_objects)} 个大对象")

        # 这里可以实现大对象的重新分配策略

    def _optimize_async_processing(self):
        """异步处理优化"""

        # 使用线程池执行异步任务
        def async_task():
            time.sleep(0.01)  # 模拟异步操作

        # 并发执行多个任务
        futures = []
        for _ in range(10):
            future = self.executor.submit(async_task)
            futures.append(future)

        # 等待所有任务完成
        for future in futures:
            future.result()

    def _optimize_concurrency(self):
        """并发优化"""

        # 使用锁优化
        lock = threading.Lock()

        def concurrent_operation():
            with lock:
                time.sleep(0.001)  # 模拟并发操作

        threads = []
        for _ in range(5):
            thread = threading.Thread(target=concurrent_operation)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

    def _optimize_algorithms(self):
        """算法优化"""

        # 使用更高效的数据结构
        # 这里可以实现算法优化的具体逻辑

    def _optimize_connection_pooling(self):
        """连接池优化"""

        # 模拟连接池优化
        # 在实际应用中，这里会优化数据库连接池、Redis连接池等

    def _optimize_cache_strategy(self):
        """缓存策略优化"""

        # 优化缓存策略
        # 实现更智能的缓存淘汰算法

    def _optimize_batch_operations(self):
        """批量操作优化"""

        # 实现批量数据库操作、批量缓存操作等

    def _optimize_dictionaries(self):
        """字典优化"""

        # 使用有序字典、默认字典等优化

    def _optimize_list_operations(self):
        """列表操作优化"""

        # 使用列表推导式、生成器等优化

    def _optimize_set_operations(self):
        """集合操作优化"""

        # 优化集合操作

    def _collect_performance_metrics(self) -> PerformanceMetrics:
        """收集性能指标"""

        process = psutil.Process()

        return PerformanceMetrics(
            timestamp=time.time(),
            memory_usage=process.memory_info().rss / 1024 / 1024,  # MB
            cpu_usage=process.cpu_percent(interval=0.1),
            response_time=10.0,  # 模拟响应时间，实际应用中需要测量
            throughput=1000.0,   # 模拟吞吐量
            error_rate=0.1       # 模拟错误率
        )

    def run_full_optimization(self) -> List[OptimizationResult]:
        """运行完整的性能优化"""

        print("🚀 开始基础设施层性能优化...")

        optimizations = [
            self.optimize_memory_usage,
            self.optimize_cpu_usage,
            self.optimize_io_operations,
            self.optimize_data_structures
        ]

        results = []
        for optimization in optimizations:
            try:
                result = optimization()
                results.append(result)
                print(f"  ✅ {result.optimization_type}: {result.improvement_percentage:.1f}% 改善")
            except Exception as e:
                print(f"  ❌ 优化失败: {e}")

        # 生成优化报告
        self._generate_optimization_report(results)

        return results

    def _generate_optimization_report(self, results: List[OptimizationResult]):
        """生成优化报告"""

        print("\n📊 性能优化报告")
        print("=" * 50)

        total_improvement = 0
        successful_optimizations = 0

        for result in results:
            if result.improvement_percentage > 0:
                successful_optimizations += 1
                total_improvement += result.improvement_percentage

            print(f"\\n🔧 {result.optimization_type}:")
            print(f"  描述: {result.description}")
            print(f"  优化前内存: {result.before_metrics.memory_usage:.1f} MB")
            print(f"  优化后内存: {result.after_metrics.memory_usage:.1f} MB")
            print(f"  改善程度: {result.improvement_percentage:.1f}%")

        if successful_optimizations > 0:
            avg_improvement = total_improvement / successful_optimizations
            print(f"\\n🎯 平均改善幅度: {avg_improvement:.1f}%")
        else:
            print("\\n⚠️ 没有成功的优化")

    def benchmark_component_creation(self, iterations: int = 1000) -> Dict[str, float]:
        """基准测试Component创建性能"""

        print(f"📈 基准测试Component创建性能 ({iterations} 次迭代)...")

        try:
            factory = CacheComponentFactory()

            # 测试创建性能
            start_time = time.time()

            for i in range(iterations):
                component = factory.create_component(1, {})
                if component:
                    # 模拟使用
                    _ = component.component_id

            end_time = time.time()

            total_time = end_time - start_time
            avg_time = total_time / iterations * 1000  # 毫秒
            throughput = iterations / total_time

            results = {
                'total_time': total_time,
                'avg_time_per_operation': avg_time,
                'throughput': throughput,
                'iterations': iterations
            }

            print(f"  ⏱️ 平均响应时间: {avg_time:.2f} ms")
            print(f"  🚀 吞吐量: {throughput:.0f} ops/sec")
            return results

        except Exception as e:
            print(f"❌ 基准测试失败: {e}")
            return {}


def main():
    """主函数"""

    print("⚡ 基础设施层性能优化系统")
    print("=" * 40)

    optimizer = ComponentFactoryPerformanceOptimizer()

    try:
        # 运行基准测试
        print("📊 运行性能基准测试...")
        benchmark_results = optimizer.benchmark_component_creation()

        # 运行优化
        print("\\n🔧 运行性能优化...")
        optimization_results = optimizer.run_full_optimization()

        # 输出最终结果
        print("\\n🎯 性能优化完成!")

        if benchmark_results:
            print("\\n📈 基准测试结果:")
            print(f"  ⏱️ 平均响应时间: {benchmark_results.get('avg_time_per_operation', 0):.2f} ms")
            print(f"  🚀 吞吐量: {benchmark_results.get('throughput', 0):.0f} ops/sec")
        if optimization_results:
            successful_opts = sum(1 for r in optimization_results if r.improvement_percentage > 0)
            total_improvement = sum(
                r.improvement_percentage for r in optimization_results if r.improvement_percentage > 0)

            if successful_opts > 0:
                print(
                    f"\\n🎯 优化成果: {successful_opts} 个优化成功，平均改善 {total_improvement/successful_opts:.1f}%")
    except Exception as e:
        print(f"❌ 性能优化失败: {e}")
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
