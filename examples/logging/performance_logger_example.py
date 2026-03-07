#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PerformanceLogger 使用示例
演示性能监控系统的日志记录功能
"""

from infrastructure.logging import PerformanceLogger
import time
import random
import sys
import os
# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    import psutil
except ImportError:
    psutil = None


def get_system_metrics():
    """获取系统性能指标"""
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_io": psutil.disk_io_counters().read_bytes + psutil.disk_io_counters().write_bytes if psutil.disk_io_counters() else 0
    }


def simulate_performance_monitoring():
    """模拟性能监控日志记录"""

    # 创建性能Logger
    perf_logger = PerformanceLogger(
        name="performance.monitor",
        log_dir="logs/performance"
    )

    print("=== 性能Logger演示 ===")

    # API端点性能监控
    endpoints = [
        "/api/trade",
        "/api/portfolio",
        "/api/market-data",
        "/api/user/profile",
        "/api/reports/generate"
    ]

    for endpoint in endpoints:
        # 模拟API响应时间
        response_times = [random.uniform(10, 200) for _ in range(100)]
        avg_time = sum(response_times) / len(response_times)
        p95_time = sorted(response_times)[94]  # 95th percentile
        p99_time = sorted(response_times)[98]  # 99th percentile

        # 模拟请求率和错误率
        requests_per_sec = random.uniform(100, 2000)
        error_rate = random.uniform(0.001, 0.05)

        perf_logger.info("API性能指标",
                         endpoint=endpoint,
                         avg_response_time=round(avg_time, 2),
                         p95_response_time=round(p95_time, 2),
                         p99_response_time=round(p99_time, 2),
                         requests_per_second=round(requests_per_sec, 2),
                         error_rate=round(error_rate, 4),
                         timestamp=time.time()
                         )

        print(f"📊 {endpoint}: 平均 {avg_time:.1f}ms, P95 {p95_time:.1f}ms")

        time.sleep(0.1)

    # 数据库查询性能
    queries = [
        {"type": "SELECT", "table": "trades", "complexity": "simple"},
        {"type": "INSERT", "table": "orders", "complexity": "simple"},
        {"type": "UPDATE", "table": "portfolio", "complexity": "medium"},
        {"type": "SELECT", "table": "market_data", "complexity": "complex"}
    ]

    for query in queries:
        query_time = random.uniform(0.5, 50.0)
        rows_affected = random.randint(
            1, 10000) if query["type"] != "SELECT" else random.randint(1, 100000)

        perf_logger.info("数据库查询性能",
                         operation=query["type"],
                         table=query["table"],
                         query_time=round(query_time, 2),
                         rows_affected=rows_affected,
                         complexity=query["complexity"],
                         slow_query_threshold=10.0,
                         is_slow_query=query_time > 10.0,
                         timestamp=time.time()
                         )

        print(f"🗄️ {query['type']} {query['table']}: {query_time:.2f}ms")

    # 系统资源监控
    for _ in range(5):
        metrics = get_system_metrics()

        perf_logger.info("系统资源使用情况",
                         cpu_percent=round(metrics["cpu_percent"], 1),
                         memory_percent=round(metrics["memory_percent"], 1),
                         disk_io_mb=round(metrics["disk_io"] / (1024*1024), 2),
                         active_connections=random.randint(50, 200),
                         thread_count=random.randint(20, 100),
                         timestamp=time.time()
                         )

        print(f"💻 CPU: {metrics['cpu_percent']:.1f}%, 内存: {metrics['memory_percent']:.1f}%")
        time.sleep(1)

    # 缓存性能指标
    perf_logger.info("缓存性能统计",
                     cache_hit_rate=round(random.uniform(0.85, 0.98), 3),
                     cache_miss_rate=round(random.uniform(0.02, 0.15), 3),
                     cache_size_mb=round(random.uniform(100, 500), 2),
                     eviction_rate=round(random.uniform(0.001, 0.01), 4),
                     hot_keys_count=random.randint(100, 1000),
                     timestamp=time.time()
                     )

    # 业务流程性能
    perf_logger.info("业务流程性能",
                     process_name="order_processing",
                     avg_processing_time=round(random.uniform(50, 200), 2),
                     throughput_per_hour=random.randint(1000, 5000),
                     queue_length=random.randint(0, 50),
                     bottleneck_stage="payment_validation",
                     timestamp=time.time()
                     )

    print("\n性能日志记录完成")
    print(f"Logger名称: {perf_logger.name}")
    print(f"日志级别: {perf_logger.level}")
    print(f"日志分类: {perf_logger.category}")
    print(f"日志目录: {perf_logger.log_dir}")


if __name__ == "__main__":
    try:
        simulate_performance_monitoring()
    except ImportError:
        print("警告: psutil未安装，使用模拟数据")
        # 如果没有psutil，使用简化版本

        def get_system_metrics():
            return {
                "cpu_percent": random.uniform(10, 90),
                "memory_percent": random.uniform(20, 80),
                "disk_io": random.randint(1000000, 10000000)
            }

        simulate_performance_monitoring()
