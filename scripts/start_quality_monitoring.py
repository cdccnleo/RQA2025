#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
质量监控启动脚本
"""
import time
import psutil
from datetime import datetime
import json
from pathlib import Path


class QualityMonitor:
    """质量监控器"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.metrics_file = self.project_root / 'reports' / 'quality_metrics.json'
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)

    def collect_metrics(self):
        """收集质量指标"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'system_metrics': {
                'cpu_usage': psutil.cpu_percent(interval=1),
                'memory_usage': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent
            },
            'quality_metrics': {
                'test_coverage': 46.0,  # 待实际计算
                'e2e_pass_rate': 92.5,  # 待实际计算
                'environment_stability': 85.0  # 待实际计算
            }
        }

        return metrics

    def save_metrics(self, metrics):
        """保存指标数据"""
        existing_metrics = []
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r', encoding='utf-8') as f:
                    existing_metrics = json.load(f)
            except:
                existing_metrics = []

        existing_metrics.append(metrics)

        # 保留最近1000条记录
        if len(existing_metrics) > 1000:
            existing_metrics = existing_metrics[-1000:]

        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(existing_metrics, f, indent=2, ensure_ascii=False)

    def start_monitoring(self):
        """启动监控"""
        print("启动质量监控系统...")
        print("按Ctrl+C停止监控")

        try:
            while True:
                metrics = self.collect_metrics()
                self.save_metrics(metrics)

                print(f"[{metrics['timestamp']}] CPU: {metrics['system_metrics']['cpu_usage']}%, "
                      f"内存: {metrics['system_metrics']['memory_usage']}%, "
                      f"覆盖率: {metrics['quality_metrics']['test_coverage']}%")

                time.sleep(300)  # 5分钟收集一次
        except KeyboardInterrupt:
            print("\n监控系统已停止")


if __name__ == '__main__':
    monitor = QualityMonitor()
    monitor.start_monitoring()
