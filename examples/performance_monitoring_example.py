#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征性能监控示例

演示如何使用特征层性能监控功能。
"""

import time
import pandas as pd
import numpy as np
from src.features.monitoring import (
    get_performance_monitor,
    monitor_operation
)
from src.features.plugins import FeaturePluginManager


def create_sample_data(rows: int = 1000) -> pd.DataFrame:
    """创建示例数据"""
    np.random.seed(42)

    # 股票数据
    dates = pd.date_range('2024-01-01', periods=rows, freq='D')
    prices = np.random.randn(rows).cumsum() + 100

    data = pd.DataFrame({
        'date': dates,
        'open': prices + np.random.randn(rows) * 0.5,
        'high': prices + np.random.randn(rows) * 1.0,
        'low': prices - np.random.randn(rows) * 1.0,
        'close': prices,
        'volume': np.random.randint(1000, 10000, rows),
        'title': [f'新闻标题{i}' for i in range(rows)],
        'content': [f'这是第{i}条新闻内容，包含一些情感词汇。' for i in range(rows)]
    })

    return data


@monitor_operation("数据预处理")
def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """数据预处理"""
    # 模拟预处理操作
    time.sleep(0.1)
    return data.copy()


@monitor_operation("特征工程")
def extract_features(data: pd.DataFrame) -> pd.DataFrame:
    """特征工程"""
    # 模拟特征工程操作
    time.sleep(0.2)

    # 添加一些基本特征
    data['price_change'] = data['close'].pct_change()
    data['volume_ma'] = data['volume'].rolling(5).mean()
    data['price_ma'] = data['close'].rolling(10).mean()

    return data


@monitor_operation("技术指标计算")
def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """计算技术指标"""
    # 模拟技术指标计算
    time.sleep(0.3)

    # 简单的技术指标
    data['rsi'] = 50 + np.random.randn(len(data)) * 20
    data['macd'] = np.random.randn(len(data)) * 0.1
    data['bollinger_upper'] = data['close'] + np.random.randn(len(data)) * 2

    return data


@monitor_operation("情感分析")
def perform_sentiment_analysis(data: pd.DataFrame) -> pd.DataFrame:
    """情感分析"""
    # 模拟情感分析
    time.sleep(0.15)

    # 简单的情感分析
    data['sentiment_score'] = np.random.randn(len(data)) * 0.5
    data['sentiment_label'] = np.random.choice(['positive', 'negative', 'neutral'], len(data))

    return data


@monitor_operation("特征选择")
def select_features(data: pd.DataFrame) -> pd.DataFrame:
    """特征选择"""
    # 模拟特征选择
    time.sleep(0.1)

    # 选择重要特征
    important_features = ['close', 'volume', 'price_change', 'rsi', 'sentiment_score']
    return data[important_features]


def example_basic_monitoring():
    """基础性能监控示例"""
    print("=== 基础性能监控示例 ===")

    # 获取性能监控器
    monitor = get_performance_monitor()

    # 创建示例数据
    data = create_sample_data(100)
    print(f"原始数据形状: {data.shape}")

    # 使用监控器监控操作
    with monitor.monitor_operation("完整特征处理流程", {"data_size": len(data)}):
        # 数据预处理
        data = preprocess_data(data)

        # 特征工程
        data = extract_features(data)

        # 技术指标计算
        data = calculate_technical_indicators(data)

        # 情感分析
        data = perform_sentiment_analysis(data)

        # 特征选择
        data = select_features(data)

    print(f"处理后数据形状: {data.shape}")

    # 获取性能报告
    report = monitor.get_performance_report()
    print("\n性能报告:")
    print(f"  运行时间: {report['uptime']:.2f}秒")
    print(f"  操作次数: {report['operation_count']}")
    print(f"  错误次数: {report['error_count']}")
    print(f"  成功率: {report['success_rate']:.2f}%")


def example_detailed_metrics():
    """详细指标监控示例"""
    print("\n=== 详细指标监控示例 ===")

    monitor = get_performance_monitor()

    # 执行多个操作
    data = create_sample_data(200)

    operations = [
        ("数据预处理", lambda: preprocess_data(data)),
        ("特征工程", lambda: extract_features(data)),
        ("技术指标", lambda: calculate_technical_indicators(data)),
        ("情感分析", lambda: perform_sentiment_analysis(data)),
        ("特征选择", lambda: select_features(data))
    ]

    for op_name, op_func in operations:
        with monitor.monitor_operation(op_name, {"data_size": len(data)}):
            op_func()

    # 获取各类型指标摘要
    print("\n各类型指标摘要:")
    for metric_type in monitor._thresholds.keys():
        summary = monitor.get_metrics_summary(metric_type)
        if summary:
            print(f"  {metric_type.value}:")
            print(f"    次数: {summary['count']}")
            print(f"    平均值: {summary['mean']:.4f}")
            print(f"    最大值: {summary['max']:.4f}")
            print(f"    最小值: {summary['min']:.4f}")


def example_plugin_monitoring():
    """插件性能监控示例"""
    print("\n=== 插件性能监控示例 ===")

    monitor = get_performance_monitor()

    # 创建插件管理器
    plugin_manager = FeaturePluginManager()

    # 注册插件
    from examples.plugins.technical_plugin import TechnicalIndicatorPlugin
    from examples.plugins.sentiment_plugin import SentimentAnalysisPlugin

    technical_plugin = TechnicalIndicatorPlugin()
    sentiment_plugin = SentimentAnalysisPlugin()

    plugin_manager.register_plugin(technical_plugin)
    plugin_manager.register_plugin(sentiment_plugin)

    # 创建示例数据
    data = create_sample_data(150)
    print(f"原始数据形状: {data.shape}")

    # 监控插件处理
    with monitor.monitor_operation("插件处理", {"plugins": 2, "data_size": len(data)}):
        # 使用技术指标插件
        technical_result = technical_plugin.process(data.copy())

        # 使用情感分析插件
        sentiment_result = sentiment_plugin.process(data.copy())

    print(f"插件处理后数据形状: {technical_result.shape}")

    # 获取插件相关性能指标
    execution_summary = monitor.get_metrics_summary(
        monitor._thresholds.keys().__iter__().__next__()  # 获取第一个MetricType
    )
    if execution_summary:
        print(f"插件执行时间: {execution_summary['mean']:.4f}秒")


def example_threshold_monitoring():
    """阈值监控示例"""
    print("\n=== 阈值监控示例 ===")

    monitor = get_performance_monitor()

    # 设置更严格的阈值
    from src.features.monitoring.performance_monitor import MetricType

    monitor.set_threshold(MetricType.EXECUTION_TIME, 0.1)  # 100ms
    monitor.set_threshold(MetricType.MEMORY_USAGE, 10.0)   # 10%

    print("设置性能阈值:")
    thresholds = monitor.get_thresholds()
    for metric_type, threshold in thresholds.items():
        print(f"  {metric_type.value}: {threshold}")

    # 执行可能超阈值的操作
    data = create_sample_data(500)

    print("\n执行可能超阈值的操作...")
    with monitor.monitor_operation("大数据处理", {"data_size": len(data)}):
        # 模拟耗时操作
        time.sleep(0.2)  # 可能超过100ms阈值

        # 模拟内存密集型操作
        large_array = np.random.rand(10000, 1000)
        result = np.dot(large_array, large_array.T)

    print("操作完成，检查是否有阈值告警")


def example_snapshot_analysis():
    """快照分析示例"""
    print("\n=== 快照分析示例 ===")

    monitor = get_performance_monitor()

    # 执行一些操作
    data = create_sample_data(100)

    for i in range(5):
        with monitor.monitor_operation(f"操作{i+1}", {"iteration": i+1}):
            preprocess_data(data)
            extract_features(data)
            time.sleep(0.05)

    # 创建快照
    snapshot = monitor.create_snapshot()
    print(f"创建快照时间: {snapshot.timestamp}")
    print(f"快照包含指标数: {len(snapshot.metrics)}")

    # 获取快照历史
    snapshots = monitor.get_snapshots(limit=3)
    print(f"历史快照数: {len(snapshots)}")

    if snapshots:
        latest_snapshot = snapshots[-1]
        print(f"最新快照成功率: {latest_snapshot.summary.get('success_rate', 0):.2f}%")


def main():
    """主函数"""
    print("特征层性能监控示例")
    print("=" * 50)

    try:
        # 基础监控示例
        example_basic_monitoring()

        # 详细指标示例
        example_detailed_metrics()

        # 插件监控示例
        example_plugin_monitoring()

        # 阈值监控示例
        example_threshold_monitoring()

        # 快照分析示例
        example_snapshot_analysis()

        print("\n" + "=" * 50)
        print("所有示例执行完成！")

    except Exception as e:
        print(f"执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
