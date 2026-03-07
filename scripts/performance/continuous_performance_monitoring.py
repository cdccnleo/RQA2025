#!/usr/bin/env python3
"""
持续性能监控脚本
实时监控RQA2025系统性能并提供持续优化建议
"""

from src.infrastructure.monitoring.application_monitor import ApplicationMonitor
import asyncio
import psutil
import logging
import json
import threading
import multiprocessing
from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
import sys
import os
import numpy as np
from collections import deque
import gc

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """性能指标"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    network_io: Dict[str, float]
    process_count: int
    thread_count: int
    gc_stats: Dict[str, Any]


@dataclass
class OptimizationRecommendation:
    """优化建议"""
    priority: str  # high, medium, low
    category: str
    description: str
    expected_improvement: float
    implementation_cost: str  # low, medium, high
    details: Dict[str, Any]


class ContinuousPerformanceMonitor:
    """持续性能监控器"""

    def __init__(self, monitoring_interval: int = 30):
        self.app_monitor = ApplicationMonitor()
        self.monitoring_interval = monitoring_interval

        # 性能指标历史
        self.performance_history: deque = deque(maxlen=1000)

        # 优化建议
        self.optimization_recommendations: List[OptimizationRecommendation] = []

        # 监控状态
        self.is_monitoring = False
        self.monitoring_task = None

        # 系统信息
        self.cpu_count = multiprocessing.cpu_count()
        self.memory_gb = psutil.virtual_memory().total / (1024**3)

        # 性能阈值
        self.thresholds = {
            'cpu_high': 80.0,
            'cpu_critical': 95.0,
            'memory_high': 85.0,
            'memory_critical': 95.0,
            'disk_high': 90.0,
            'disk_critical': 98.0
        }

        logger.info(
            f"ContinuousPerformanceMonitor initialized - CPU: {self.cpu_count}, Memory: {self.memory_gb:.1f}GB")

    async def start_monitoring(self):
        """开始持续监控"""
        logger.info("开始持续性能监控...")

        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())

        logger.info("持续性能监控已启动")

    async def stop_monitoring(self):
        """停止持续监控"""
        logger.info("停止持续性能监控...")

        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("持续性能监控已停止")

    async def _monitoring_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                # 收集性能指标
                metrics = await self._collect_performance_metrics()
                self.performance_history.append(metrics)

                # 分析性能趋势
                await self._analyze_performance_trends()

                # 生成优化建议
                await self._generate_optimization_recommendations()

                # 检查性能告警
                await self._check_performance_alerts(metrics)

                # 等待下次监控
                await asyncio.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error(f"监控循环出错: {e}")
                await asyncio.sleep(5)

    async def _collect_performance_metrics(self) -> PerformanceMetric:
        """收集性能指标"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)

        # 内存使用率
        memory_percent = psutil.virtual_memory().percent

        # 磁盘使用率
        disk_usage = psutil.disk_usage('/').percent

        # 网络IO
        network_io = psutil.net_io_counters()
        network_metrics = {
            'bytes_sent': network_io.bytes_sent,
            'bytes_recv': network_io.bytes_recv,
            'packets_sent': network_io.packets_sent,
            'packets_recv': network_io.packets_recv
        }

        # 进程和线程数
        process_count = len(psutil.pids())
        thread_count = threading.active_count()

        # 垃圾回收统计
        gc_stats = {
            'collections': gc.get_stats(),
            'count': gc.get_count(),
            'objects': len(gc.get_objects())
        }

        metrics = PerformanceMetric(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            disk_usage=disk_usage,
            network_io=network_metrics,
            process_count=process_count,
            thread_count=thread_count,
            gc_stats=gc_stats
        )

        return metrics

    async def _analyze_performance_trends(self):
        """分析性能趋势"""
        if len(self.performance_history) < 10:
            return

        # 计算最近10个数据点的趋势
        recent_metrics = list(self.performance_history)[-10:]

        # CPU趋势
        cpu_trend = self._calculate_trend([m.cpu_percent for m in recent_metrics])

        # 内存趋势
        memory_trend = self._calculate_trend([m.memory_percent for m in recent_metrics])

        # 磁盘趋势
        disk_trend = self._calculate_trend([m.disk_usage for m in recent_metrics])

        # 记录趋势分析
        logger.info(
            f"性能趋势分析 - CPU: {cpu_trend:.2f}%/min, 内存: {memory_trend:.2f}%/min, 磁盘: {disk_trend:.2f}%/min")

    def _calculate_trend(self, values: List[float]) -> float:
        """计算趋势（每分钟变化率）"""
        if len(values) < 2:
            return 0.0

        # 简单线性回归
        x = np.arange(len(values))
        y = np.array(values)

        # 计算斜率（每分钟变化率）
        slope = np.polyfit(x, y, 1)[0]

        return slope * 60  # 转换为每分钟变化率

    async def _generate_optimization_recommendations(self):
        """生成优化建议"""
        if len(self.performance_history) < 5:
            return

        # 清空之前的建议
        self.optimization_recommendations.clear()

        # 分析最近的性能指标
        recent_metrics = list(self.performance_history)[-5:]
        avg_cpu = np.mean([m.cpu_percent for m in recent_metrics])
        avg_memory = np.mean([m.memory_percent for m in recent_metrics])
        avg_disk = np.mean([m.disk_usage for m in recent_metrics])

        # 基于CPU使用率的建议
        if avg_cpu > self.thresholds['cpu_critical']:
            self.optimization_recommendations.append(OptimizationRecommendation(
                priority="high",
                category="cpu_optimization",
                description="CPU使用率过高，建议立即优化",
                expected_improvement=30.0,
                implementation_cost="medium",
                details={
                    "current_cpu": avg_cpu,
                    "threshold": self.thresholds['cpu_critical'],
                    "suggestions": [
                        "优化算法复杂度",
                        "启用多线程处理",
                        "使用GPU加速计算"
                    ]
                }
            ))
        elif avg_cpu > self.thresholds['cpu_high']:
            self.optimization_recommendations.append(OptimizationRecommendation(
                priority="medium",
                category="cpu_optimization",
                description="CPU使用率较高，建议优化",
                expected_improvement=20.0,
                implementation_cost="low",
                details={
                    "current_cpu": avg_cpu,
                    "threshold": self.thresholds['cpu_high'],
                    "suggestions": [
                        "优化代码逻辑",
                        "减少不必要的计算",
                        "使用缓存机制"
                    ]
                }
            ))

        # 基于内存使用率的建议
        if avg_memory > self.thresholds['memory_critical']:
            self.optimization_recommendations.append(OptimizationRecommendation(
                priority="high",
                category="memory_optimization",
                description="内存使用率过高，建议立即优化",
                expected_improvement=40.0,
                implementation_cost="medium",
                details={
                    "current_memory": avg_memory,
                    "threshold": self.thresholds['memory_critical'],
                    "suggestions": [
                        "优化内存分配",
                        "启用垃圾回收优化",
                        "使用内存池"
                    ]
                }
            ))
        elif avg_memory > self.thresholds['memory_high']:
            self.optimization_recommendations.append(OptimizationRecommendation(
                priority="medium",
                category="memory_optimization",
                description="内存使用率较高，建议优化",
                expected_improvement=25.0,
                implementation_cost="low",
                details={
                    "current_memory": avg_memory,
                    "threshold": self.thresholds['memory_high'],
                    "suggestions": [
                        "减少内存泄漏",
                        "优化数据结构",
                        "使用内存映射"
                    ]
                }
            ))

        # 基于磁盘使用率的建议
        if avg_disk > self.thresholds['disk_critical']:
            self.optimization_recommendations.append(OptimizationRecommendation(
                priority="high",
                category="disk_optimization",
                description="磁盘使用率过高，建议立即清理",
                expected_improvement=50.0,
                implementation_cost="low",
                details={
                    "current_disk": avg_disk,
                    "threshold": self.thresholds['disk_critical'],
                    "suggestions": [
                        "清理临时文件",
                        "压缩数据文件",
                        "迁移到更大磁盘"
                    ]
                }
            ))
        elif avg_disk > self.thresholds['disk_high']:
            self.optimization_recommendations.append(OptimizationRecommendation(
                priority="medium",
                category="disk_optimization",
                description="磁盘使用率较高，建议清理",
                expected_improvement=30.0,
                implementation_cost="low",
                details={
                    "current_disk": avg_disk,
                    "threshold": self.thresholds['disk_high'],
                    "suggestions": [
                        "清理日志文件",
                        "优化数据存储",
                        "使用数据压缩"
                    ]
                }
            ))

    async def _check_performance_alerts(self, metrics: PerformanceMetric):
        """检查性能告警"""
        alerts = []

        # CPU告警
        if metrics.cpu_percent > self.thresholds['cpu_critical']:
            alerts.append(f"🚨 CPU使用率过高: {metrics.cpu_percent:.1f}%")
        elif metrics.cpu_percent > self.thresholds['cpu_high']:
            alerts.append(f"⚠️ CPU使用率较高: {metrics.cpu_percent:.1f}%")

        # 内存告警
        if metrics.memory_percent > self.thresholds['memory_critical']:
            alerts.append(f"🚨 内存使用率过高: {metrics.memory_percent:.1f}%")
        elif metrics.memory_percent > self.thresholds['memory_high']:
            alerts.append(f"⚠️ 内存使用率较高: {metrics.memory_percent:.1f}%")

        # 磁盘告警
        if metrics.disk_usage > self.thresholds['disk_critical']:
            alerts.append(f"🚨 磁盘使用率过高: {metrics.disk_usage:.1f}%")
        elif metrics.disk_usage > self.thresholds['disk_high']:
            alerts.append(f"⚠️ 磁盘使用率较高: {metrics.disk_usage:.1f}%")

        # 输出告警
        for alert in alerts:
            logger.warning(alert)

    async def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.performance_history:
            return {"error": "没有性能数据"}

        recent_metrics = list(self.performance_history)[-10:]

        summary = {
            "timestamp": datetime.now().isoformat(),
            "monitoring_duration": len(self.performance_history) * self.monitoring_interval,
            "current_metrics": {
                "cpu_percent": recent_metrics[-1].cpu_percent,
                "memory_percent": recent_metrics[-1].memory_percent,
                "disk_usage": recent_metrics[-1].disk_usage,
                "process_count": recent_metrics[-1].process_count,
                "thread_count": recent_metrics[-1].thread_count
            },
            "average_metrics": {
                "cpu_percent": np.mean([m.cpu_percent for m in recent_metrics]),
                "memory_percent": np.mean([m.memory_percent for m in recent_metrics]),
                "disk_usage": np.mean([m.disk_usage for m in recent_metrics])
            },
            "optimization_recommendations": [
                {
                    "priority": rec.priority,
                    "category": rec.category,
                    "description": rec.description,
                    "expected_improvement": rec.expected_improvement,
                    "implementation_cost": rec.implementation_cost,
                    "details": rec.details
                }
                for rec in self.optimization_recommendations
            ]
        }

        return summary

    async def generate_performance_report(self):
        """生成性能报告"""
        logger.info("生成性能报告...")

        summary = await self.get_performance_summary()

        # 保存报告
        report_file = f"reports/performance/continuous_monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(os.path.dirname(report_file), exist_ok=True)

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"性能报告已保存到: {report_file}")

        # 输出摘要
        logger.info("=== 性能监控摘要 ===")
        logger.info(f"监控时长: {summary['monitoring_duration']}秒")
        logger.info(f"当前CPU使用率: {summary['current_metrics']['cpu_percent']:.1f}%")
        logger.info(f"当前内存使用率: {summary['current_metrics']['memory_percent']:.1f}%")
        logger.info(f"当前磁盘使用率: {summary['current_metrics']['disk_usage']:.1f}%")
        logger.info(f"优化建议数量: {len(summary['optimization_recommendations'])}")

        # 输出高优先级建议
        high_priority_recommendations = [
            rec for rec in summary['optimization_recommendations']
            if rec['priority'] == 'high'
        ]

        if high_priority_recommendations:
            logger.info("=== 高优先级优化建议 ===")
            for rec in high_priority_recommendations:
                logger.info(f"🚨 {rec['description']} (预期改进: {rec['expected_improvement']:.1f}%)")


async def main():
    """主函数"""
    monitor = ContinuousPerformanceMonitor(monitoring_interval=10)

    try:
        # 启动监控
        await monitor.start_monitoring()

        # 运行一段时间
        await asyncio.sleep(60)  # 监控1分钟

        # 生成报告
        await monitor.generate_performance_report()

    finally:
        # 停止监控
        await monitor.stop_monitoring()


if __name__ == "__main__":
    asyncio.run(main())
