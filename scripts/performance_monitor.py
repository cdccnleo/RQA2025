#!/usr/bin/env python3
"""
性能监控脚本
"""

import time
import psutil
import json
from pathlib import Path


def monitor_performance():
    """监控性能指标"""
    metrics = {
        'timestamp': time.time(),
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_usage': psutil.disk_usage('/').percent
    }

    # 保存指标
    metrics_file = Path('performance_metrics.json')
    existing_metrics = []

    if metrics_file.exists():
        try:
            with open(metrics_file, 'r', encoding='utf-8') as f:
                existing_metrics = json.load(f)
        except Exception:
            existing_metrics = []

    existing_metrics.append(metrics)

    # 只保留最近100个指标
    if len(existing_metrics) > 100:
        existing_metrics = existing_metrics[-100:]

    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(existing_metrics, f, indent=2)

    print(f"性能指标已记录: CPU {metrics['cpu_percent']}%, 内存 {metrics['memory_percent']}%")
    return metrics


if __name__ == "__main__":
    monitor_performance()
