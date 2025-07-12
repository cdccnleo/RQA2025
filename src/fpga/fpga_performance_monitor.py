#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FPGA性能监控器
实时监控FPGA加速模块的性能指标
"""

import time
from typing import Dict, List
import logging
from datetime import datetime
from .fpga_manager import FPGAManager

class FPGAPerformanceMonitor:
    def __init__(self, fpga_manager: FPGAManager):
        self.fpga_manager = fpga_manager
        self.logger = logging.getLogger(__name__)
        self.metrics = {
            'latency': [],
            'throughput': [],
            'utilization': []
        }
        self.warning_thresholds = {
            'latency': 0.1,  # 100ms
            'utilization': 0.9  # 90%
        }

    def record_latency(self, operation: str, latency: float):
        """记录操作延迟

        Args:
            operation: 操作名称
            latency: 延迟时间(秒)
        """
        timestamp = datetime.now()
        self.metrics['latency'].append({
            'timestamp': timestamp,
            'operation': operation,
            'value': latency
        })

        # 检查是否超过阈值
        if latency > self.warning_thresholds['latency']:
            self.logger.warning(
                f"FPGA延迟过高: {operation} 耗时 {latency:.3f}s")

    def record_throughput(self, operation: str, count: int):
        """记录吞吐量

        Args:
            operation: 操作名称
            count: 操作次数
        """
        timestamp = datetime.now()
        self.metrics['throughput'].append({
            'timestamp': timestamp,
            'operation': operation,
            'value': count
        })

    def update_utilization(self):
        """更新FPGA资源利用率"""
        status = self.fpga_manager.get_device_status()
        if not status:
            return

        utilization = status.get('utilization', 0)
        timestamp = datetime.now()
        self.metrics['utilization'].append({
            'timestamp': timestamp,
            'value': utilization
        })

        # 检查是否超过阈值
        if utilization > self.warning_thresholds['utilization']:
            self.logger.warning(
                f"FPGA资源利用率过高: {utilization:.1%}")

    def get_recent_metrics(self, metric_name: str, minutes: int = 5) -> List[Dict]:
        """获取最近一段时间内的性能指标

        Args:
            metric_name: 指标名称 (latency/throughput/utilization)
            minutes: 时间范围(分钟)

        Returns:
            指标数据列表
        """
        if metric_name not in self.metrics:
            return []

        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [m for m in self.metrics[metric_name]
                if m['timestamp'] >= cutoff]

    def generate_report(self) -> Dict:
        """生成性能报告

        Returns:
            性能报告字典
        """
        report = {
            'timestamp': datetime.now(),
            'latency_stats': self._calculate_stats('latency'),
            'throughput_stats': self._calculate_stats('throughput'),
            'utilization_stats': self._calculate_stats('utilization'),
            'warning_count': self._count_warnings()
        }
        return report

    def _calculate_stats(self, metric_name: str) -> Dict:
        """计算指标统计信息

        Args:
            metric_name: 指标名称

        Returns:
            统计信息字典
        """
        if not self.metrics[metric_name]:
            return {}

        values = [m['value'] for m in self.metrics[metric_name]]
        return {
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'count': len(values)
        }

    def _count_warnings(self) -> Dict:
        """统计各类告警数量

        Returns:
            告警统计字典
        """
        latency_warnings = sum(1 for m in self.metrics['latency']
                             if m['value'] > self.warning_thresholds['latency'])
        utilization_warnings = sum(1 for m in self.metrics['utilization']
                                 if m['value'] > self.warning_thresholds['utilization'])

        return {
            'latency': latency_warnings,
            'utilization': utilization_warnings,
            'total': latency_warnings + utilization_warnings
        }
