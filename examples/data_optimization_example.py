#!/usr/bin/env python3
"""
数据层优化使用示例

展示如何使用数据优化器、性能监控器和数据预加载器。
"""

import asyncio
import time
import numpy as np

# 导入数据优化模块
from src.data.optimization import (
    DataOptimizer,
    OptimizationConfig
)
from src.data.optimization.performance_monitor import DataPerformanceMonitor
from src.data.optimization.data_preloader import DataPreloader, PreloadConfig


async def basic_optimization_example():
    """基本优化示例"""
    print("=== 基本数据优化示例 ===")

    # 创建优化配置
    config = OptimizationConfig(
        max_workers=4,
        enable_parallel_loading=True,
        enable_cache=True,
        enable_quality_monitor=True,
        quality_threshold=0.8
    )

    # 创建优化器
    optimizer = DataOptimizer(config)

    try:
        # 优化数据加载
        result = await optimizer.optimize_data_loading(
            data_type="stock",
            start_date="2024-01-01",
            end_date="2024-01-31",
            frequency="1d",
            symbols=["AAPL", "GOOGL", "MSFT"]
        )

        print(f"加载成功: {result.success}")
        print(f"加载时间: {result.load_time_ms:.2f}ms")
        print(f"缓存命中: {result.cache_hit}")

        if result.quality_metrics:
            print(f"质量分数: {result.quality_metrics.get('overall_score', 0):.2f}")
            print(f"质量等级: {result.quality_metrics.get('overall_level', 'unknown')}")

        # 获取优化报告
        report = optimizer.get_optimization_report()
        print(f"缓存命中率: {report['performance_metrics']['cache_hit_rate']:.2%}")
        print(f"平均质量分数: {report['performance_metrics']['avg_quality_score']:.2f}")

    finally:
        optimizer.cleanup()


def performance_monitoring_example():
    """性能监控示例"""
    print("\n=== 性能监控示例 ===")

    # 创建性能监控器
    monitor = DataPerformanceMonitor()

    try:
        # 开始监控
        monitor.start_monitoring(interval_seconds=10)

        # 模拟一些操作
        operations = [
            ("data_load", 150.5, True, {"symbol": "AAPL"}),
            ("data_process", 300.2, True, {"operation": "normalize"}),
            ("data_export", 75.8, True, {"format": "csv"}),
            ("data_load", 5000.0, False, {"symbol": "INVALID"}, "Connection timeout"),
            ("data_validate", 120.3, True, {"validation": "schema"})
        ]

        for operation, duration, success, metadata, *error in operations:
            error_msg = error[0] if error else None
            monitor.record_operation(
                operation=operation,
                duration_ms=duration,
                success=success,
                error_message=error_msg,
                metadata=metadata
            )
            time.sleep(0.1)  # 模拟操作间隔

        # 等待一段时间让监控器收集数据
        time.sleep(2)

        # 获取性能报告
        report = monitor.get_performance_report(hours=1)

        print(f"总操作数: {report['total_operations']}")
        print(f"成功操作: {report['successful_operations']}")
        print(f"失败操作: {report['failed_operations']}")
        print(f"平均加载时间: {report['operation_summary']['data_load']['avg_duration_ms']:.2f}ms")
        print(f"系统平均CPU使用率: {report['system_metrics']['avg_cpu_percent']:.1f}%")
        print(f"系统平均内存使用率: {report['system_metrics']['avg_memory_percent']:.1f}%")
        print(f"活跃告警数: {report['active_alerts']}")

    finally:
        monitor.stop_monitoring()


def data_preloading_example():
    """数据预加载示例"""
    print("\n=== 数据预加载示例 ===")

    # 创建预加载配置
    config = PreloadConfig(
        max_concurrent_tasks=3,
        enable_auto_preload=True,
        auto_preload_symbols=["AAPL", "GOOGL"]
    )

    # 创建预加载器
    preloader = DataPreloader(config)

    try:
        # 添加预加载任务
        task_ids = []
        symbols_list = [
            ["MSFT", "AMZN"],
            ["TSLA", "NVDA"],
            ["META", "NFLX"]
        ]

        for i, symbols in enumerate(symbols_list):
            task_id = preloader.add_preload_task(
                data_type="stock",
                start_date="2024-01-01",
                end_date="2024-01-31",
                frequency="1d",
                symbols=symbols,
                priority=i + 1
            )
            task_ids.append(task_id)
            print(f"添加预加载任务 {task_id}，优先级 {i + 1}")

        # 等待一段时间让任务执行
        time.sleep(3)

        # 检查任务状态
        for task_id in task_ids:
            status = preloader.get_task_status(task_id)
            print(f"任务 {task_id} 状态: {status['status']}")

        # 获取统计信息
        stats = preloader.get_stats()
        print(f"总任务数: {stats['total_tasks']}")
        print(f"完成任务数: {stats['completed_tasks']}")
        print(f"失败任务数: {stats['failed_tasks']}")
        print(f"队列大小: {stats['queue_size']}")
        print(f"活跃工作线程: {stats['active_workers']}")

        # 显示任务状态统计
        status_counts = stats['task_status_counts']
        for status, count in status_counts.items():
            print(f"状态 {status}: {count} 个任务")

    finally:
        preloader.shutdown()


async def advanced_optimization_example():
    """高级优化示例"""
    print("\n=== 高级优化示例 ===")

    # 创建高性能配置
    config = OptimizationConfig(
        max_workers=8,
        enable_parallel_loading=True,
        enable_cache=True,
        enable_quality_monitor=True,
        quality_threshold=0.9,
        performance_threshold_ms=3000
    )

    # 创建优化器
    optimizer = DataOptimizer(config)

    try:
        # 模拟多次数据加载
        symbols_groups = [
            ["AAPL", "GOOGL", "MSFT"],
            ["AMZN", "TSLA", "NVDA"],
            ["META", "NFLX", "ADBE"]
        ]

        results = []
        for i, symbols in enumerate(symbols_groups):
            print(f"加载第 {i+1} 组数据: {symbols}")

            result = await optimizer.optimize_data_loading(
                data_type="stock",
                start_date="2024-01-01",
                end_date="2024-01-31",
                frequency="1d",
                symbols=symbols
            )

            results.append(result)
            print(f"  加载时间: {result.load_time_ms:.2f}ms")
            print(f"  缓存命中: {result.cache_hit}")
            print(f"  质量分数: {result.quality_metrics.get('overall_score', 0):.2f}")

        # 分析结果
        total_time = sum(r.load_time_ms for r in results)
        cache_hits = sum(1 for r in results if r.cache_hit)
        avg_quality = np.mean([r.quality_metrics.get('overall_score', 0)
                              for r in results if r.quality_metrics])

        print(f"\n优化效果分析:")
        print(f"总加载时间: {total_time:.2f}ms")
        print(f"缓存命中次数: {cache_hits}/{len(results)}")
        print(f"平均质量分数: {avg_quality:.2f}")

        # 获取详细报告
        report = optimizer.get_optimization_report()
        print(f"整体缓存命中率: {report['performance_metrics']['cache_hit_rate']:.2%}")
        print(f"平均质量分数: {report['performance_metrics']['avg_quality_score']:.2f}")

    finally:
        optimizer.cleanup()


async def error_handling_example():
    """错误处理示例"""
    print("\n=== 错误处理示例 ===")

    config = OptimizationConfig(
        max_workers=2,
        enable_parallel_loading=True,
        enable_cache=True,
        enable_quality_monitor=True
    )

    optimizer = DataOptimizer(config)

    try:
        # 模拟错误情况
        result = await optimizer.optimize_data_loading(
            data_type="invalid_type",
            start_date="2024-01-01",
            end_date="2024-01-31",
            frequency="1d",
            symbols=["INVALID_SYMBOL"]
        )

        print(f"加载成功: {result.success}")
        if not result.success:
            print(f"错误信息: {result.error_message}")
        print(f"加载时间: {result.load_time_ms:.2f}ms")

    except Exception as e:
        print(f"捕获到异常: {str(e)}")
    finally:
        optimizer.cleanup()


async def main():
    """主函数"""
    print("数据层优化模块使用示例")
    print("=" * 50)

    # 运行各种示例
    await basic_optimization_example()
    performance_monitoring_example()
    data_preloading_example()
    await advanced_optimization_example()
    await error_handling_example()

    print("\n" + "=" * 50)
    print("所有示例运行完成！")


if __name__ == "__main__":
    # 运行示例
    asyncio.run(main())
