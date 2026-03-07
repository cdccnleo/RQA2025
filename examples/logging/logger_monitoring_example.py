#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 基础设施层日志系统 - Logger池监控集成示例

演示Logger池监控功能与统一监控系统的集成。
"""

from infrastructure.monitoring import (
    ApplicationMonitor,
    get_logger_pool_metrics,
    LoggerPoolMonitor
)
from infrastructure.logging.core.interfaces import get_pooled_logger
from infrastructure.logging import BaseLogger, BusinessLogger, AuditLogger
import sys
import os
import time
import json

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def basic_monitoring_example():
    """基础监控功能示例"""
    print("=== 基础监控功能示例 ===\n")

    # 创建应用监控器
    monitor = ApplicationMonitor()

    # 记录一些应用指标
    monitor.record_metric("cpu_usage", 45.2, {"server": "app-01"})
    monitor.record_metric("memory_usage", 2.3, {"server": "app-01"})
    monitor.record_metric("request_count", 1250, {"endpoint": "/api/v1/orders"})

    # 获取所有指标
    all_metrics = monitor.get_all_metrics()
    print("当前应用指标:")
    for name, metric in all_metrics.items():
        if name != 'logger_pool':  # 单独处理Logger池指标
            print(f"  {name}: {metric['value']} (tags: {metric['tags']})")

    print()

    # 获取Logger池统计
    logger_stats = monitor.get_logger_pool_stats()
    if logger_stats:
        print("Logger池统计:")
        current_stats = logger_stats.get('current_stats', {})
        if current_stats:
            print(f"  池大小: {current_stats.get('pool_size', 0)}")
            print(".2f" print(f"  创建计数: {current_stats.get('created_count', 0)}")
            print(f"  命中计数: {current_stats.get('hit_count', 0)}")
            print(".1f" print()


def prometheus_metrics_example():
    """Prometheus指标导出示例"""
    print("=== Prometheus指标导出示例 ===\n")

    # 创建监控器并记录指标
    monitor=ApplicationMonitor()
    monitor.record_metric("response_time", 125.5, {"endpoint": "/api/orders"})
    monitor.record_metric("error_rate", 0.02, {"service": "order_service"})

    # 生成Prometheus格式指标
    prometheus_metrics=monitor.get_prometheus_metrics()

    print("Prometheus格式指标:")
    print(prometheus_metrics)
    print()


def logger_pool_monitoring_example():
    """Logger池监控详细示例"""
    print("=== Logger池监控详细示例 ===\n")

    # 创建专用监控器
    pool_monitor=LoggerPoolMonitor("demo_pool", collection_interval=5)

    print("启动Logger池监控...")
    pool_monitor.start_monitoring()

    try:
        # 模拟Logger使用
        loggers=[]

        print("创建Logger实例...")
        for i in range(15):
            # 使用对象池获取Logger
            logger=get_pooled_logger(f"demo.service.{i % 5}")  # 只使用5个不同名称
            logger.info(f"测试日志消息 {i}", iteration=i)
            loggers.append(logger)

        print(f"创建了 {len(loggers)} 个Logger引用")

        # 等待监控数据收集
        print("等待监控数据收集...")
        time.sleep(6)  # 等待一个收集周期

        # 获取监控统计
        stats=pool_monitor.get_current_stats()
        if stats:
            print("
Logger池监控统计: "            print(f"  池大小: {stats.pool_size}")
            print(f"  最大容量: {stats.max_size}")
            print(".2f" print(f"  创建计数: {stats.created_count}")
            print(f"  命中计数: {stats.hit_count}")
            print(f"  Logger数量: {stats.logger_count}")
            print(".1f" print(".3f"        # 获取性能汇总
        performance=pool_monitor.get_performance_summary()
        if performance:
            perf_metrics=performance.get('performance_metrics', {})
            alert_status=performance.get('alert_status', {})
            recommendations=performance.get('recommendations', [])

            print("
性能指标: "            print(".1f"            print(".3f"            print(f"  内存效率: {perf_metrics.get('memory_efficiency', 'unknown')}")

            print("
告警状态: " for alert_type, is_active in alert_status.items():
                status="🔴 激活" if is_active else "✅ 正常"
                print(f"  {alert_type}: {status}")

            if recommendations:
                print("
优化建议: " for rec in recommendations:
                    print(f"  • {rec}")

        # 获取历史数据
        history=pool_monitor.get_history_stats(limit=3)
        if history:
            print(f"\n历史数据点: {len(history)}")
            for i, hist_stats in enumerate(history[-3:]):  # 显示最后3个
                print(f"  数据点{i+1}: 池大小={hist_stats.pool_size}, 命中率={hist_stats.hit_rate:.2f}")

    finally:
        print("
停止Logger池监控..."        pool_monitor.stop_monitoring()

    print()


def real_time_monitoring_example():
    """实时监控演示"""
    print("=== 实时监控演示 ===\n")

    # 创建监控器
    pool_monitor=LoggerPoolMonitor("realtime_demo", collection_interval=2)
    pool_monitor.start_monitoring()

    try:
        print("开始实时Logger操作监控...")
        print("每2秒收集一次统计数据，运行10秒")
        print()

        for round_num in range(5):
            print(f"第 {round_num + 1} 轮操作:")

            # 执行一些Logger操作
            for i in range(20):
                logger=get_pooled_logger(f"realtime.service.{i % 3}")
                logger.info(f"实时操作 {round_num}-{i}",
                          operation_id=f"op_{round_num}_{i}")

            # 显示当前统计
            stats=pool_monitor.get_current_stats()
            if stats:
                print(f"  池大小: {stats.pool_size}, 命中率: {stats.hit_rate:.2f}")

            time.sleep(2.1)  # 略微超过收集间隔

        print("\n实时监控完成")

    finally:
        pool_monitor.stop_monitoring()

    print()


def alert_system_integration_example():
    """告警系统集成示例"""
    print("=== 告警系统集成示例 ===\n")

    # 创建监控器，设置较低的告警阈值来触发告警
    pool_monitor=LoggerPoolMonitor("alert_demo", collection_interval=3)

    # 修改告警阈值以便演示
    pool_monitor.alert_thresholds['hit_rate_low']=0.95  # 非常高的阈值
    pool_monitor.alert_thresholds['pool_usage_high']=0.5  # 较低的阈值

    pool_monitor.start_monitoring()

    try:
        print("模拟触发告警的情况...")

        # 创建很多不同的Logger名称来降低命中率
        for i in range(100):  # 创建100个不同的Logger
            logger=get_pooled_logger(f"alert_test.unique_{i}")
            logger.debug(f"唯一Logger {i}")

        print("等待告警检查...")
        time.sleep(4)  # 等待监控收集和告警检查

        # 检查告警状态
        performance=pool_monitor.get_performance_summary()
        alert_status=performance.get('alert_status', {})

        print("告警状态检查:")
        for alert_type, is_active in alert_status.items():
            status="🔥 触发" if is_active else "✅ 正常"
            print(f"  {alert_type}: {status}")

        if any(alert_status.values()):
            print("\n注意：以上告警是故意触发的演示效果")

    finally:
        pool_monitor.stop_monitoring()

    print()


def main():
    """主函数"""
    print("RQA2025 基础设施层日志系统 - Logger池监控集成示例")
    print("=" * 70)
    print()

    try:
        basic_monitoring_example()
        prometheus_metrics_example()
        logger_pool_monitoring_example()
        real_time_monitoring_example()
        alert_system_integration_example()

        print("🎉 所有Logger监控示例执行完成！")
        print("\n监控功能亮点:")
        print("📊 实时性能指标收集")
        print("🚨 智能告警系统集成")
        print("📈 Prometheus指标导出")
        print("🔍 历史数据趋势分析")
        print("💡 自动化优化建议生成")

    except Exception as e:
        print(f"❌ 示例执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
