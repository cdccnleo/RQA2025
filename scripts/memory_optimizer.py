#!/usr/bin/env python3
"""
内存优化分析和实施工具

用于分析内存使用情况、检测内存泄漏、实施内存优化策略。
支持LRU缓存、内存池管理、对象压缩等。
"""

import gc
import psutil
import tracemalloc
import time
from typing import Dict, Any, List, Optional, Callable
from collections import OrderedDict
import threading
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryProfiler:
    """内存分析器"""

    def __init__(self):
        self.snapshots = []
        self.baseline = None

    def start_tracking(self):
        """开始内存跟踪"""
        tracemalloc.start()
        self.baseline = tracemalloc.take_snapshot()
        logger.info("内存跟踪已启动")

    def take_snapshot(self, label: str = "") -> Dict[str, Any]:
        """获取内存快照"""
        if not tracemalloc.is_tracing():
            return {"error": "内存跟踪未启动"}

        snapshot = tracemalloc.take_snapshot()

        # 分析内存统计
        stats = snapshot.statistics('filename')
        top_stats = stats[:10]  # 前10个占用内存的文件

        current_memory = psutil.virtual_memory()
        process_memory = psutil.Process().memory_info()

        snapshot_data = {
            "timestamp": time.time(),
            "label": label,
            "current_memory_percent": current_memory.percent,
            "current_memory_used": current_memory.used,
            "process_memory_rss": process_memory.rss,
            "process_memory_vms": process_memory.vms,
            "top_memory_files": [
                {
                    "file": stat.traceback[0].filename if stat.traceback else "unknown",
                    "line": stat.traceback[0].lineno if stat.traceback else 0,
                    "size": stat.size,
                    "count": stat.count
                }
                for stat in top_stats
            ]
        }

        self.snapshots.append(snapshot_data)
        return snapshot_data

    def analyze_memory_growth(self) -> Dict[str, Any]:
        """分析内存增长"""
        if len(self.snapshots) < 2:
            return {"error": "需要至少2个快照才能分析增长"}

        first_snapshot = self.snapshots[0]
        last_snapshot = self.snapshots[-1]

        memory_growth = last_snapshot["process_memory_rss"] - first_snapshot["process_memory_rss"]
        memory_growth_mb = memory_growth / 1024 / 1024

        return {
            "total_growth_mb": memory_growth_mb,
            "growth_rate_percent": (memory_growth / first_snapshot["process_memory_rss"]) * 100,
            "start_memory_mb": first_snapshot["process_memory_rss"] / 1024 / 1024,
            "end_memory_mb": last_snapshot["process_memory_rss"] / 1024 / 1024,
            "snapshots_count": len(self.snapshots)
        }

    def detect_memory_leaks(self) -> List[Dict[str, Any]]:
        """检测内存泄漏"""
        if not self.snapshots:
            return []

        # 分析内存增长趋势
        memory_over_time = [s["process_memory_rss"] for s in self.snapshots]

        # 简单的泄漏检测：内存持续增长
        leaks = []
        if len(memory_over_time) >= 3:
            # 计算增长趋势
            growth_trend = []
            for i in range(1, len(memory_over_time)):
                growth = memory_over_time[i] - memory_over_time[i-1]
                growth_trend.append(growth)

            avg_growth = sum(growth_trend) / len(growth_trend)

            if avg_growth > 1024 * 1024:  # 1MB增长
                leaks.append({
                    "type": "continuous_growth",
                    "description": f"内存持续增长，平均每快照增长 {avg_growth/1024/1024:.2f} MB",
                    "severity": "high",
                    "recommendation": "检查是否有未释放的对象引用"
                })

        return leaks

    def get_memory_recommendations(self) -> List[str]:
        """生成内存优化建议"""
        recommendations = []

        if not self.snapshots:
            return ["需要先收集内存快照数据"]

        # 分析当前内存使用
        latest = self.snapshots[-1] if self.snapshots else None
        if latest:
            memory_percent = latest["current_memory_percent"]

            if memory_percent > 80:
                recommendations.append("💾 内存使用率过高 (>80%)，建议实施内存优化")
            elif memory_percent > 70:
                recommendations.append("⚠️ 内存使用率较高 (>70%)，建议监控内存使用")

            # 分析大对象
            top_files = latest.get("top_memory_files", [])
            for file_info in top_files[:3]:
                if file_info["size"] > 50 * 1024 * 1024:  # 50MB
                    recommendations.append(
                        f"📊 大对象检测: {file_info['file']} 占用 {file_info['size']/1024/1024:.1f} MB")

        # 通用内存优化建议
        recommendations.extend([
            "🔄 实施LRU缓存策略，自动清理过期数据",
            "📦 使用内存池管理机制，减少分配/释放开销",
            "🗜️ 对大型模型和数据集实施压缩存储",
            "🧵 定期执行垃圾回收，释放无用对象",
            "📊 实施内存使用监控和告警机制"
        ])

        return recommendations


class LRUCache:
    """LRU缓存实现"""

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.access_times = {}
        self.hits = 0
        self.misses = 0

    def get(self, key):
        """获取缓存项"""
        if key in self.cache:
            self.hits += 1
            self.access_times[key] = time.time()
            self.cache.move_to_end(key)
            return self.cache[key]
        else:
            self.misses += 1
            return None

    def put(self, key, value):
        """设置缓存项"""
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                # 移除最少使用的项
                removed_key, _ = self.cache.popitem(last=False)
                logger.debug(f"LRU缓存移除: {removed_key}")

        self.cache[key] = value
        self.access_times[key] = time.time()

    def cleanup(self, max_age: int = 3600) -> int:
        """清理过期缓存项"""
        current_time = time.time()
        expired_keys = [
            key for key, access_time in self.access_times.items()
            if current_time - access_time > max_age
        ]

        for key in expired_keys:
            if key in self.cache:
                del self.cache[key]
            del self.access_times[key]

        logger.info(f"清理了 {len(expired_keys)} 个过期缓存项")
        return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0

        return {
            "capacity": self.capacity,
            "current_size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "utilization": len(self.cache) / self.capacity
        }


class MemoryPool:
    """内存池管理器"""

    def __init__(self, pool_size: int = 100):
        self.pool_size = pool_size
        self.pool = []
        self.allocated = set()
        self.lock = threading.Lock()

    def allocate(self, size: int) -> Optional[bytes]:
        """从内存池分配内存"""
        with self.lock:
            if len(self.pool) < self.pool_size:
                memory = bytes(size)
                self.pool.append(memory)
                self.allocated.add(id(memory))
                return memory
            else:
                logger.warning("内存池已满，无法分配新内存")
                return None

    def deallocate(self, memory: bytes) -> bool:
        """释放内存回到池中"""
        with self.lock:
            if id(memory) in self.allocated:
                # 内存仍然在池中，不需要释放
                return True
            else:
                logger.warning("尝试释放不在池中的内存")
                return False

    def cleanup(self) -> int:
        """清理内存池"""
        with self.lock:
            # 强制垃圾回收
            collected = gc.collect()

            # 清理弱引用
            self.pool = [mem for mem in self.pool if mem is not None]

            logger.info(f"内存池清理完成，收集了 {collected} 个对象")
            return collected

    def get_stats(self) -> Dict[str, Any]:
        """获取内存池统计"""
        return {
            "pool_size": self.pool_size,
            "allocated_count": len(self.pool),
            "utilization": len(self.pool) / self.pool_size,
            "total_memory_mb": sum(len(mem) for mem in self.pool if mem) / 1024 / 1024
        }


class ModelCompressor:
    """模型压缩器"""

    def __init__(self):
        self.compression_stats = {}

    def compress_model(self, model, method: str = "quantization") -> Dict[str, Any]:
        """压缩模型"""
        original_size = self._estimate_model_size(model)

        if method == "quantization":
            compressed_model = self._quantize_model(model)
        elif method == "pruning":
            compressed_model = self._prune_model(model)
        else:
            return {"error": f"不支持的压缩方法: {method}"}

        compressed_size = self._estimate_model_size(compressed_model)
        compression_ratio = compressed_size / original_size if original_size > 0 else 0

        result = {
            "method": method,
            "original_size_mb": original_size,
            "compressed_size_mb": compressed_size,
            "compression_ratio": compression_ratio,
            "memory_saved_mb": original_size - compressed_size,
            "compressed_model": compressed_model
        }

        self.compression_stats[method] = result
        return result

    def _quantize_model(self, model):
        """量化压缩（模拟）"""
        # 这里应该是实际的量化逻辑
        # 为了演示，我们返回一个标记，表示模型已被量化
        return {"quantized": True, "original_model": model}

    def _prune_model(self, model):
        """剪枝压缩（模拟）"""
        # 这里应该是实际的剪枝逻辑
        return {"pruned": True, "original_model": model}

    def _estimate_model_size(self, model) -> float:
        """估算模型大小"""
        # 简单的估算逻辑
        if isinstance(model, dict):
            return len(str(model)) / 1024 / 1024  # 估算为MB
        else:
            return 100.0  # 默认100MB

    def get_compression_stats(self) -> Dict[str, Any]:
        """获取压缩统计"""
        return self.compression_stats


class MemoryOptimizer:
    """内存优化器"""

    def __init__(self):
        self.profiler = MemoryProfiler()
        self.lru_cache = LRUCache(capacity=1000)
        self.memory_pool = MemoryPool(pool_size=50)
        self.model_compressor = ModelCompressor()

    def optimize_cache_memory(self, cache_data: Dict[str, Any]) -> Dict[str, Any]:
        """优化缓存内存使用"""
        logger.info("开始优化缓存内存使用...")

        # 将数据放入LRU缓存
        for key, value in cache_data.items():
            self.lru_cache.put(key, value)

        # 清理过期数据
        cleaned_count = self.lru_cache.cleanup(max_age=1800)  # 30分钟过期

        cache_stats = self.lru_cache.get_stats()

        return {
            "cache_stats": cache_stats,
            "cleaned_items": cleaned_count,
            "memory_optimized": True
        }

    def optimize_model_memory(self, models: List[Any]) -> Dict[str, Any]:
        """优化模型内存使用"""
        logger.info("开始优化模型内存使用...")

        optimized_models = []
        total_memory_saved = 0

        for model in models:
            # 对每个模型应用量化压缩
            compression_result = self.model_compressor.compress_model(model, "quantization")
            optimized_models.append(compression_result["compressed_model"])
            total_memory_saved += compression_result.get("memory_saved_mb", 0)

        return {
            "optimized_models": optimized_models,
            "total_memory_saved_mb": total_memory_saved,
            "compression_stats": self.model_compressor.get_compression_stats()
        }

    def implement_memory_pool(self, data_operations: List[Callable]) -> Dict[str, Any]:
        """实施内存池管理"""
        logger.info("开始实施内存池管理...")

        pool_stats_before = self.memory_pool.get_stats()

        # 执行数据操作，使用内存池
        results = []
        for operation in data_operations:
            # 从内存池分配内存进行操作
            memory_block = self.memory_pool.allocate(1024 * 1024)  # 1MB
            if memory_block:
                try:
                    result = operation(memory_block)
                    results.append(result)
                finally:
                    self.memory_pool.deallocate(memory_block)

        pool_stats_after = self.memory_pool.get_stats()

        # 清理内存池
        collected = self.memory_pool.cleanup()

        return {
            "operations_completed": len(results),
            "pool_stats_before": pool_stats_before,
            "pool_stats_after": pool_stats_after,
            "objects_collected": collected
        }

    def monitor_memory_usage(self, duration: int = 60) -> Dict[str, Any]:
        """监控内存使用情况"""
        logger.info(f"开始监控内存使用情况，持续{duration}秒...")

        self.profiler.start_tracking()

        # 定期获取快照
        snapshots = []
        for i in range(duration // 10):  # 每10秒一个快照
            time.sleep(10)
            snapshot = self.profiler.take_snapshot(f"snapshot_{i}")
            snapshots.append(snapshot)

        # 分析内存增长
        growth_analysis = self.profiler.analyze_memory_growth()

        # 检测内存泄漏
        leaks = self.profiler.detect_memory_leaks()

        # 生成优化建议
        recommendations = self.profiler.get_memory_recommendations()

        return {
            "duration": duration,
            "snapshots_count": len(snapshots),
            "memory_growth_analysis": growth_analysis,
            "detected_leaks": leaks,
            "optimization_recommendations": recommendations
        }


def main():
    """主函数"""
    print("🚀 CPU/内存性能优化专项 - 内存优化分析")
    print("=" * 60)

    # 创建内存优化器
    optimizer = MemoryOptimizer()

    # 1. 内存监控
    print("\n1. 执行内存监控...")
    memory_monitoring = optimizer.monitor_memory_usage(20)  # 20秒监控

    # 2. 缓存优化
    print("\n2. 执行缓存优化...")
    sample_cache_data = {f"key_{i}": f"value_{i}" * 100 for i in range(100)}
    cache_optimization = optimizer.optimize_cache_memory(sample_cache_data)

    # 3. 模型优化
    print("\n3. 执行模型内存优化...")
    sample_models = [{"model": "sample_model", "size": 200}]  # 模拟模型
    model_optimization = optimizer.optimize_model_memory(sample_models)

    # 4. 内存池测试
    print("\n4. 执行内存池测试...")

    def sample_operation(memory_block):
        # 模拟内存操作
        return len(memory_block)

    pool_operations = [sample_operation] * 10
    pool_test = optimizer.implement_memory_pool(pool_operations)

    # 5. 生成报告
    print("\n5. 生成内存优化报告...")

    report = {
        "memory_monitoring": memory_monitoring,
        "cache_optimization": cache_optimization,
        "model_optimization": model_optimization,
        "memory_pool_test": pool_test,
        "timestamp": time.time()
    }

    # 保存报告
    with open("memory_optimization_report.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    # 输出摘要
    print("\n📊 内存优化报告摘要")
    print("-" * 50)

    monitoring = memory_monitoring
    print("🖥️ 内存监控结果:")
    if "memory_growth_analysis" in monitoring:
        growth = monitoring["memory_growth_analysis"]
        if "total_growth_mb" in growth:
            print(f"   内存增长: {growth['total_growth_mb']:.2f} MB")
        if "detected_leaks" in monitoring:
            print(f"   检测到内存泄漏: {len(monitoring['detected_leaks'])} 个")

    cache_opt = cache_optimization
    if "cache_stats" in cache_opt:
        stats = cache_opt["cache_stats"]
        print("\n💾 缓存优化结果:")
        print(f"   缓存容量: {stats.get('capacity', 0)}")
        print(f"   当前大小: {stats.get('current_size', 0)}")
        print(f"   缓存命中率: {stats.get('hit_rate', 0):.1f}%")

    model_opt = model_optimization
    if "total_memory_saved_mb" in model_opt:
        print("\n🗜️ 模型优化结果:")
        print(f"   节省内存: {model_opt['total_memory_saved_mb']:.1f} MB")

    pool_test_result = pool_test
    if "operations_completed" in pool_test_result:
        print("\n🏊 内存池测试结果:")
        print(f"   完成的内存操作: {pool_test_result['operations_completed']}")

    print("\n💡 内存优化建议:")
    recommendations = memory_monitoring.get("optimization_recommendations", [])
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"   {i}. {rec}")

    print("\n📄 详细报告已保存到: memory_optimization_report.json")
    print("\n✅ 内存优化分析完成！")


if __name__ == "__main__":
    main()
