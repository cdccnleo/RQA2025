#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RQA2025系统性能优化工具包
提供全面的性能分析、优化和监控工具
"""

import time
import psutil
import threading
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path
import sys

# 添加src路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used: int
    disk_usage: float
    network_io: Dict[str, int]
    response_time: Optional[float] = None
    throughput: Optional[int] = None
    error_rate: Optional[float] = None

class PerformanceAnalyzer:
    """性能分析器"""

    def __init__(self):
        self.baseline_metrics = {}
        self.current_metrics = []
        self.monitoring_active = False
        self.monitoring_thread = None

    def start_monitoring(self, interval: float = 1.0):
        """启动性能监控"""
        if self.monitoring_active:
            logger.warning("性能监控已在运行中")
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, args=(interval,))
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        logger.info(f"性能监控已启动，采集间隔: {interval}秒")

    def stop_monitoring(self):
        """停止性能监控"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        logger.info("性能监控已停止")

    def _monitoring_loop(self, interval: float):
        """监控循环"""
        while self.monitoring_active:
            metrics = self.collect_system_metrics()
            self.current_metrics.append(metrics)

            # 保持最近1小时的数据
            if len(self.current_metrics) > 3600:
                self.current_metrics = self.current_metrics[-3600:]

            time.sleep(interval)

    def collect_system_metrics(self) -> PerformanceMetrics:
        """收集系统性能指标"""
        timestamp = time.time()

        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # 内存使用情况
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used = memory.used

        # 磁盘使用情况
        disk = psutil.disk_usage('/')
        disk_usage = disk.percent

        # 网络I/O
        network = psutil.net_io_counters()
        network_io = {
            'bytes_sent': network.bytes_sent,
            'bytes_recv': network.bytes_recv,
            'packets_sent': network.packets_sent,
            'packets_recv': network.packets_recv
        }

        return PerformanceMetrics(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used=memory_used,
            disk_usage=disk_usage,
            network_io=network_io
        )

    def analyze_performance_trends(self, time_window: int = 300) -> Dict[str, Any]:
        """分析性能趋势"""
        if len(self.current_metrics) < 2:
            return {"error": "没有足够的性能数据进行分析"}

        # 获取最近时间窗口的数据
        recent_metrics = [m for m in self.current_metrics
                         if time.time() - m.timestamp <= time_window]

        if len(recent_metrics) < 2:
            return {"error": "时间窗口内没有足够的性能数据"}

        # 计算趋势
        cpu_trend = self._calculate_trend([m.cpu_percent for m in recent_metrics])
        memory_trend = self._calculate_trend([m.memory_percent for m in recent_metrics])
        disk_trend = self._calculate_trend([m.disk_usage for m in recent_metrics])

        # 计算峰值
        cpu_peak = max(m.cpu_percent for m in recent_metrics)
        memory_peak = max(m.memory_percent for m in recent_metrics)

        # 计算平均值
        cpu_avg = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        memory_avg = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)

        return {
            "time_window_seconds": time_window,
            "data_points": len(recent_metrics),
            "cpu": {
                "average": round(cpu_avg, 2),
                "peak": round(cpu_peak, 2),
                "trend": cpu_trend
            },
            "memory": {
                "average": round(memory_avg, 2),
                "peak": round(memory_peak, 2),
                "trend": memory_trend
            },
            "disk": {
                "usage": round(recent_metrics[-1].disk_usage, 2),
                "trend": disk_trend
            },
            "analysis_timestamp": time.time()
        }

    def _calculate_trend(self, values: List[float]) -> str:
        """计算趋势"""
        if len(values) < 3:
            return "insufficient_data"

        # 计算线性趋势
        n = len(values)
        x = list(range(n))
        y = values

        # 简单线性回归
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_xx = sum(xi * xi for xi in x)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)

        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"

    def generate_performance_report(self) -> Dict[str, Any]:
        """生成性能报告"""
        if not self.current_metrics:
            return {"error": "没有性能数据可以生成报告"}

        # 基础统计
        total_points = len(self.current_metrics)
        duration = self.current_metrics[-1].timestamp - self.current_metrics[0].timestamp

        # CPU统计
        cpu_values = [m.cpu_percent for m in self.current_metrics]
        cpu_stats = self._calculate_statistics(cpu_values)

        # 内存统计
        memory_values = [m.memory_percent for m in self.current_metrics]
        memory_stats = self._calculate_statistics(memory_values)

        # 磁盘统计
        disk_values = [m.disk_usage for m in self.current_metrics]
        disk_stats = self._calculate_statistics(disk_values)

        # 性能评估
        performance_score = self._calculate_performance_score(cpu_stats, memory_stats)

        return {
            "report_type": "system_performance_analysis",
            "generated_at": time.time(),
            "monitoring_duration_seconds": round(duration, 2),
            "total_data_points": total_points,
            "performance_score": performance_score,
            "metrics": {
                "cpu": cpu_stats,
                "memory": memory_stats,
                "disk": disk_stats
            },
            "recommendations": self._generate_recommendations(performance_score, cpu_stats, memory_stats)
        }

    def _calculate_statistics(self, values: List[float]) -> Dict[str, float]:
        """计算统计信息"""
        if not values:
            return {}

        return {
            "average": round(sum(values) / len(values), 2),
            "minimum": round(min(values), 2),
            "maximum": round(max(values), 2),
            "median": round(sorted(values)[len(values) // 2], 2),
            "standard_deviation": round((sum((x - sum(values)/len(values))**2 for x in values) / len(values))**0.5, 2)
        }

    def _calculate_performance_score(self, cpu_stats: Dict, memory_stats: Dict) -> Dict[str, Any]:
        """计算性能评分"""
        cpu_score = max(0, 100 - cpu_stats.get("average", 0))
        memory_score = max(0, 100 - memory_stats.get("average", 0))

        overall_score = (cpu_score + memory_score) / 2

        # 评分等级
        if overall_score >= 90:
            grade = "优秀"
        elif overall_score >= 80:
            grade = "良好"
        elif overall_score >= 70:
            grade = "一般"
        elif overall_score >= 60:
            grade = "需改进"
        else:
            grade = "严重不足"

        return {
            "overall_score": round(overall_score, 2),
            "cpu_score": round(cpu_score, 2),
            "memory_score": round(memory_score, 2),
            "grade": grade
        }

    def _generate_recommendations(self, performance_score: Dict, cpu_stats: Dict, memory_stats: Dict) -> List[str]:
        """生成优化建议"""
        recommendations = []

        overall_score = performance_score["overall_score"]

        if overall_score < 70:
            recommendations.append("系统性能需要紧急优化")

        if cpu_stats.get("average", 0) > 80:
            recommendations.append("CPU使用率过高，建议优化CPU密集型操作")

        if memory_stats.get("average", 0) > 85:
            recommendations.append("内存使用率过高，建议检查内存泄漏并优化内存管理")

        if cpu_stats.get("maximum", 0) > 95:
            recommendations.append("检测到CPU峰值过高，建议实施负载均衡")

        if memory_stats.get("maximum", 0) > 95:
            recommendations.append("检测到内存峰值过高，建议增加内存或优化内存使用")

        if not recommendations:
            recommendations.append("系统性能表现良好，继续保持")

        return recommendations

    def save_report(self, filepath: str):
        """保存性能报告"""
        report = self.generate_performance_report()

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"性能报告已保存到: {filepath}")
        except Exception as e:
            logger.error(f"保存性能报告失败: {e}")

class PerformanceOptimizer:
    """性能优化器"""

    def __init__(self):
        self.analyzer = PerformanceAnalyzer()
        self.optimization_strategies = {
            "cpu_optimization": self._optimize_cpu_usage,
            "memory_optimization": self._optimize_memory_usage,
            "io_optimization": self._optimize_io_operations,
            "cache_optimization": self._optimize_caching_strategy
        }

    def run_comprehensive_optimization(self) -> Dict[str, Any]:
        """运行综合性能优化"""
        logger.info("开始执行综合性能优化...")

        # 收集当前性能数据
        self.analyzer.start_monitoring(interval=0.5)
        time.sleep(5)  # 收集5秒数据
        self.analyzer.stop_monitoring()

        # 分析性能瓶颈
        analysis = self.analyzer.analyze_performance_trends(time_window=5)

        if "error" in analysis:
            return {"error": "无法进行性能分析", "details": analysis["error"]}

        # 执行优化策略
        optimization_results = {}
        recommendations = []

        # CPU优化
        if analysis["cpu"]["average"] > 70:
            cpu_result = self._optimize_cpu_usage()
            optimization_results["cpu_optimization"] = cpu_result
            recommendations.extend(cpu_result.get("recommendations", []))

        # 内存优化
        if analysis["memory"]["average"] > 75:
            memory_result = self._optimize_memory_usage()
            optimization_results["memory_optimization"] = memory_result
            recommendations.extend(memory_result.get("recommendations", []))

        # I/O优化
        io_result = self._optimize_io_operations()
        optimization_results["io_optimization"] = io_result
        recommendations.extend(io_result.get("recommendations", []))

        # 缓存优化
        cache_result = self._optimize_caching_strategy()
        optimization_results["cache_optimization"] = cache_result
        recommendations.extend(cache_result.get("recommendations", []))

        return {
            "optimization_timestamp": time.time(),
            "baseline_analysis": analysis,
            "optimization_results": optimization_results,
            "consolidated_recommendations": list(set(recommendations)),  # 去重
            "expected_improvements": self._estimate_improvements(optimization_results)
        }

    def _optimize_cpu_usage(self) -> Dict[str, Any]:
        """CPU使用优化"""
        recommendations = [
            "实施异步处理减少阻塞操作",
            "优化算法复杂度，减少不必要的计算",
            "使用多线程或协程提高并发处理能力",
            "实施CPU亲和性优化，合理分配CPU资源"
        ]

        return {
            "strategy": "cpu_optimization",
            "status": "recommendations_generated",
            "recommendations": recommendations,
            "estimated_improvement": "CPU使用率降低15-25%"
        }

    def _optimize_memory_usage(self) -> Dict[str, Any]:
        """内存使用优化"""
        recommendations = [
            "实施对象池化减少GC压力",
            "优化数据结构选择，减少内存占用",
            "实施内存泄漏检测和修复",
            "使用流式处理减少内存需求"
        ]

        return {
            "strategy": "memory_optimization",
            "status": "recommendations_generated",
            "recommendations": recommendations,
            "estimated_improvement": "内存使用率降低20-30%"
        }

    def _optimize_io_operations(self) -> Dict[str, Any]:
        """I/O操作优化"""
        recommendations = [
            "实施异步I/O操作提高并发能力",
            "使用连接池减少连接开销",
            "实施批量操作减少I/O次数",
            "使用缓存减少磁盘访问"
        ]

        return {
            "strategy": "io_optimization",
            "status": "recommendations_generated",
            "recommendations": recommendations,
            "estimated_improvement": "I/O性能提升30-50%"
        }

    def _optimize_caching_strategy(self) -> Dict[str, Any]:
        """缓存策略优化"""
        recommendations = [
            "实施多级缓存架构(L1/L2/L3)",
            "优化缓存失效策略",
            "实施缓存预热机制",
            "使用分布式缓存提高扩展性"
        ]

        return {
            "strategy": "cache_optimization",
            "status": "recommendations_generated",
            "recommendations": recommendations,
            "estimated_improvement": "响应时间降低40-60%"
        }

    def _estimate_improvements(self, optimization_results: Dict) -> Dict[str, str]:
        """估算优化效果"""
        improvements = {}

        if "cpu_optimization" in optimization_results:
            improvements["cpu_usage"] = "-15-25%"
            improvements["throughput"] = "+20-30%"

        if "memory_optimization" in optimization_results:
            improvements["memory_usage"] = "-20-30%"
            improvements["gc_overhead"] = "-30-40%"

        if "io_optimization" in optimization_results:
            improvements["io_performance"] = "+30-50%"
            improvements["response_time"] = "-20-30%"

        if "cache_optimization" in optimization_results:
            improvements["cache_hit_rate"] = "+40-60%"
            improvements["overall_performance"] = "+25-40%"

        return improvements

def main():
    """主函数 - 演示性能优化工具包使用"""
    print("🚀 RQA2025系统性能优化工具包")
    print("=" * 60)

    # 创建性能优化器
    optimizer = PerformanceOptimizer()

    print("📊 正在分析系统性能...")
    optimization_plan = optimizer.run_comprehensive_optimization()

    if "error" in optimization_plan:
        print(f"❌ 性能分析失败: {optimization_plan['error']}")
        return

    print("✅ 性能分析完成！")
    print()

    # 显示分析结果
    baseline = optimization_plan["baseline_analysis"]
    print("📈 当前系统性能状态:")
    print(f"   CPU使用率: 平均{baseline['cpu']['average']}%, 峰值{baseline['cpu']['peak']}%")
    print(f"   内存使用率: 平均{baseline['memory']['average']}%, 峰值{baseline['memory']['peak']}%")
    print(f"   磁盘使用率: {baseline['disk']['usage']}%")
    print()

    # 显示优化建议
    recommendations = optimization_plan["consolidated_recommendations"]
    print("💡 性能优化建议:")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    print()

    # 显示预期改善
    improvements = optimization_plan["expected_improvements"]
    if improvements:
        print("🎯 预期性能改善:")
        for metric, improvement in improvements.items():
            print(f"   {metric}: {improvement}")
        print()

    # 保存详细报告
    report_file = "performance_optimization_report.json"
    optimizer.analyzer.save_report(report_file)
    print(f"📄 详细性能报告已保存: {report_file}")

    print()
    print("🎉 性能优化分析完成！请根据建议实施优化措施。")

if __name__ == "__main__":
    main()
