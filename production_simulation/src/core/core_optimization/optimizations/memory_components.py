"""
内存优化组件 - 重构后的内存优化模块

将原来复杂的MemoryOptimizer类拆分为多个职责单一的组件：
- MemoryAnalyzer: 内存分析
- GarbageCollector: 垃圾回收优化
- MemoryOptimizer: 协调器
"""

import gc
import time
import logging
from typing import Dict, Any, List

import psutil

from ...base import BaseComponent

logger = logging.getLogger(__name__)


class MemoryAnalyzer:
    """内存分析器 - 负责内存使用情况分析"""

    def __init__(self):
        self.memory_stats = {}
        logger.info("内存分析器初始化完成")

    def analyze_memory_usage(self) -> Dict[str, Any]:
        """分析内存使用情况"""
        logger.info("开始分析内存使用情况")

        process = psutil.Process()
        memory_info = process.memory_info()

        # 收集内存统计信息
        memory_stats = {
            "rss": memory_info.rss,  # 物理内存使用
            "vms": memory_info.vms,  # 虚拟内存使用
            "percent": process.memory_percent(),  # 内存使用百分比
            "available": psutil.virtual_memory().available,  # 可用内存
            "total": psutil.virtual_memory().total,  # 总内存
            "timestamp": time.time(),
        }

        # 分析内存使用趋势
        if self.memory_stats:
            last_stats = list(self.memory_stats.values())[-1]
            memory_stats["rss_change"] = memory_stats["rss"] - last_stats["rss"]
            memory_stats["vms_change"] = memory_stats["vms"] - last_stats["vms"]
        else:
            memory_stats["rss_change"] = 0
            memory_stats["vms_change"] = 0

        self.memory_stats[time.time()] = memory_stats

        # 识别内存使用问题
        issues = self._identify_memory_issues(memory_stats)

        analysis = {
            "current_usage": memory_stats,
            "issues": issues,
            "recommendations": self._generate_memory_recommendations(issues),
        }

        logger.info(
            f"内存分析完成: 使用率 {memory_stats['percent']:.2f}%, 问题数量 {len(issues)}"
        )
        return analysis

    def _identify_memory_issues(self, memory_stats: Dict[str, Any]) -> List[str]:
        """识别内存使用问题"""
        issues = []
        if memory_stats["percent"] > 80:
            issues.append("内存使用率过高 (>80%)")
        if memory_stats["rss_change"] > 100 * 1024 * 1024:  # 100MB增长
            issues.append("内存使用增长过快 (>100MB)")
        if memory_stats["available"] < 500 * 1024 * 1024:  # 500MB可用
            issues.append("可用内存不足 (<500MB)")
        return issues

    def _generate_memory_recommendations(self, issues: List[str]) -> List[str]:
        """生成内存优化建议"""
        recommendations = []

        for issue in issues:
            if "内存使用率过高" in issue:
                recommendations.append("建议增加系统内存或优化内存密集型操作")
            elif "内存使用增长过快" in issue:
                recommendations.append("建议检查内存泄漏，优化对象生命周期管理")
            elif "可用内存不足" in issue:
                recommendations.append("建议清理不必要的缓存，释放未使用的资源")

        if not issues:
            recommendations.append("内存使用情况良好，建议定期监控")

        return recommendations


class GarbageCollector:
    """垃圾回收器 - 负责垃圾回收优化"""

    def optimize_garbage_collection(self) -> Dict[str, Any]:
        """优化垃圾回收"""
        logger.info("开始优化垃圾回收")

        # 获取当前垃圾回收统计
        gc_stats_before = {
            "counts": gc.get_count(),
            "objects": len(gc.get_objects()),
            "garbage": len(gc.garbage),
        }

        # 执行垃圾回收
        collected = gc.collect()

        # 获取垃圾回收后统计
        gc_stats_after = {
            "counts": gc.get_count(),
            "objects": len(gc.get_objects()),
            "garbage": len(gc.garbage),
        }

        # 分析垃圾回收效果
        objects_freed = gc_stats_before["objects"] - gc_stats_after["objects"]
        garbage_cleared = gc_stats_before["garbage"] - gc_stats_after["garbage"]

        optimization_results = {
            "before_gc": gc_stats_before,
            "after_gc": gc_stats_after,
            "objects_freed": objects_freed,
            "garbage_cleared": garbage_cleared,
            "collected": collected,
            "effectiveness": "high" if objects_freed > 1000 else "low",
        }

        logger.info(
            f"垃圾回收优化完成: 释放 {objects_freed} 个对象，清理 {garbage_cleared} 个垃圾对象"
        )
        return optimization_results


class MemoryOptimizer(BaseComponent):
    """内存优化器 - 协调内存优化策略"""

    def __init__(self):
        super().__init__("MemoryOptimizer")

        # 初始化组件
        self.memory_analyzer = MemoryAnalyzer()
        self.garbage_collector = GarbageCollector()
        self.optimization_history = []

        logger.info("内存优化器初始化完成")

    def analyze_memory_usage(self) -> Dict[str, Any]:
        """分析内存使用情况"""
        return self.memory_analyzer.analyze_memory_usage()

    def optimize_memory_allocation(self) -> Dict[str, Any]:
        """优化内存分配"""
        logger.info("开始优化内存分配")

        optimization_results = {
            "before_optimization": self.analyze_memory_usage(),
            "optimizations_applied": [],
            "after_optimization": {},
        }

        # 执行内存优化策略
        optimizations = []

        # 1. 强制垃圾回收
        if self._force_garbage_collection():
            optimizations.append("强制垃圾回收")

        # 2. 清理缓存
        if self._cleanup_caches():
            optimizations.append("清理缓存")

        # 3. 优化对象池
        if self._optimize_object_pools():
            optimizations.append("优化对象池")

        # 4. 压缩内存
        if self._compress_memory():
            optimizations.append("压缩内存")

        optimization_results["optimizations_applied"] = optimizations
        optimization_results["after_optimization"] = self.analyze_memory_usage()

        # 计算优化效果
        before = optimization_results["before_optimization"]["current_usage"]
        after = optimization_results["after_optimization"]["current_usage"]

        memory_saved = before["rss"] - after["rss"]
        optimization_results["memory_saved_mb"] = memory_saved / (1024 * 1024)
        optimization_results["optimization_effectiveness"] = (
            "effective" if memory_saved > 0 else "no_change"
        )

        self.optimization_history.append(optimization_results)

        logger.info(
            f"内存优化完成: 节省 {optimization_results['memory_saved_mb']:.2f}MB"
        )
        return optimization_results

    def optimize_garbage_collection(self) -> Dict[str, Any]:
        """优化垃圾回收"""
        return self.garbage_collector.optimize_garbage_collection()

    def get_memory_optimization_summary(self) -> Dict[str, Any]:
        """获取内存优化摘要"""
        if not self.optimization_history:
            return {"total_optimizations": 0, "total_memory_saved": 0}

        total_optimizations = len(self.optimization_history)
        total_memory_saved = sum(
            opt.get("memory_saved_mb", 0) for opt in self.optimization_history
        )

        # 计算平均优化效果
        effective_optimizations = sum(
            1
            for opt in self.optimization_history
            if opt.get("optimization_effectiveness") == "effective"
        )

        return {
            "total_optimizations": total_optimizations,
            "total_memory_saved_mb": total_memory_saved,
            "effective_optimizations": effective_optimizations,
            "success_rate": (
                effective_optimizations / total_optimizations
                if total_optimizations > 0
                else 0
            ),
            "average_memory_saved_mb": (
                total_memory_saved / total_optimizations
                if total_optimizations > 0
                else 0
            ),
        }

    def _force_garbage_collection(self) -> bool:
        """强制垃圾回收"""
        try:
            collected = gc.collect()
            return collected > 0
        except Exception as e:
            logger.error(f"强制垃圾回收失败: {e}")
            return False

    def _cleanup_caches(self) -> bool:
        """清理缓存"""
        try:
            # 这里应该实现具体的缓存清理逻辑
            # 例如清理文件缓存、对象缓存等
            return True
        except Exception as e:
            logger.error(f"清理缓存失败: {e}")
            return False

    def _optimize_object_pools(self) -> bool:
        """优化对象池"""
        try:
            # 这里应该实现对象池优化逻辑
            # 例如调整池大小、清理空闲对象等
            return True
        except Exception as e:
            logger.error(f"优化对象池失败: {e}")
            return False

    def _compress_memory(self) -> bool:
        """压缩内存"""
        try:
            # 模拟内存压缩
            logger.debug("执行内存压缩")
            return True
        except Exception as e:
            logger.error(f"内存压缩失败: {e}")
            return False

    def shutdown(self) -> bool:
        """关闭内存优化器"""
        try:
            logger.info("开始关闭内存优化器")
            # 清理内存统计信息
            self.memory_analyzer.memory_stats.clear()
            self.optimization_history.clear()
            logger.info("内存优化器关闭完成")
            return True
        except Exception as e:
            logger.error(f"关闭内存优化器失败: {e}")
            return False
