#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础特征工程示例

本示例演示如何使用RQA2025特征层进行基础的特征工程操作，
包括数据准备、特征生成、特征选择和特征标准化。

作者: 开发团队
日期: 2025-01-27
版本: 1.0.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 导入特征层模块
from src.features.feature_engineer import FeatureEngineer
from src.features.processors.feature_selector import FeatureSelector
from src.features.processors.feature_standardizer import FeatureStandardizer
from src.features.monitoring import FeaturesMonitor, MetricType


def create_sample_data(n_days=100):
    """
    创建示例股票数据

    参数:
        n_days (int): 数据天数

    返回:
        pd.DataFrame: 示例股票数据
    """
    print("创建示例数据...")

    # 生成日期序列
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]

    # 生成价格数据（模拟真实股票价格）
    np.random.seed(42)
    base_price = 100.0
    prices = []

    for i in range(n_days):
        # 添加随机波动
        change = np.random.normal(0, 0.02)
        base_price *= (1 + change)
        prices.append(base_price)

    # 生成OHLCV数据
    data = []
    for i, price in enumerate(prices):
        # 生成当日价格范围
        daily_volatility = np.random.uniform(0.01, 0.03)
        high = price * (1 + daily_volatility)
        low = price * (1 - daily_volatility)
        open_price = np.random.uniform(low, high)
        close_price = np.random.uniform(low, high)

        # 生成成交量
        volume = np.random.randint(1000, 10000)

        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })

    # 创建DataFrame
    df = pd.DataFrame(data, index=dates)

    print(f"创建了 {len(df)} 天的示例数据")
    print(f"数据形状: {df.shape}")
    print(f"数据范围: {df.index[0].date()} 到 {df.index[-1].date()}")

    return df


def basic_feature_engineering_example():
    """
    基础特征工程示例
    """
    print("\n" + "="*50)
    print("基础特征工程示例")
    print("="*50)

    # 1. 创建示例数据
    data = create_sample_data(100)

    # 2. 创建特征工程器
    print("\n2. 创建特征工程器...")
    engineer = FeatureEngineer()

    # 3. 生成技术指标特征
    print("\n3. 生成技术指标特征...")
    features = engineer.generate_technical_features(
        stock_data=data,
        indicators=["ma", "rsi", "macd", "bollinger"],
        params={
            "ma": {"window": [5, 10, 20]},
            "rsi": {"window": 14},
            "macd": {"fast_window": 12, "slow_window": 26, "signal_window": 9},
            "bollinger": {"window": 20, "num_std": 2}
        }
    )

    print(f"生成特征数量: {len(features.columns)}")
    print(f"特征列: {features.columns.tolist()}")

    # 4. 数据验证
    print("\n4. 验证数据质量...")
    is_valid = engineer.validate_data(data)
    print(f"数据验证结果: {'通过' if is_valid else '失败'}")

    # 5. 显示特征统计信息
    print("\n5. 特征统计信息:")
    print(features.describe())

    return features


def feature_selection_example(features):
    """
    特征选择示例

    参数:
        features (pd.DataFrame): 特征数据
    """
    print("\n" + "="*50)
    print("特征选择示例")
    print("="*50)

    # 创建目标变量（模拟）
    np.random.seed(42)
    y = pd.Series(np.random.randn(len(features)), index=features.index)

    # 1. 使用重要性方法选择特征
    print("\n1. 使用重要性方法选择特征...")
    selector_importance = FeatureSelector(method='importance', n_features=5)
    selected_importance = selector_importance.fit_transform(features, y)

    print(f"重要性方法选择的特征: {selected_importance.columns.tolist()}")

    # 2. 使用相关性方法选择特征
    print("\n2. 使用相关性方法选择特征...")
    selector_correlation = FeatureSelector(method='correlation', threshold=0.8)
    selected_correlation = selector_correlation.fit_transform(features)

    print(f"相关性方法选择的特征: {selected_correlation.columns.tolist()}")

    # 3. 获取特征重要性
    if hasattr(selector_importance, 'get_feature_importance'):
        importance = selector_importance.get_feature_importance()
        print("\n特征重要性:")
        for feature, score in importance.items():
            print(f"  {feature}: {score:.4f}")

    return selected_importance


def feature_standardization_example(features):
    """
    特征标准化示例

    参数:
        features (pd.DataFrame): 特征数据
    """
    print("\n" + "="*50)
    print("特征标准化示例")
    print("="*50)

    # 1. Z-score标准化
    print("\n1. Z-score标准化...")
    standardizer_zscore = FeatureStandardizer(method='zscore')
    standardized_zscore = standardizer_zscore.fit_transform(features)

    print("Z-score标准化后统计:")
    print(standardized_zscore.describe())

    # 2. Min-Max标准化
    print("\n2. Min-Max标准化...")
    standardizer_minmax = FeatureStandardizer(method='minmax')
    standardized_minmax = standardizer_minmax.fit_transform(features)

    print("Min-Max标准化后统计:")
    print(standardized_minmax.describe())

    # 3. Robust标准化
    print("\n3. Robust标准化...")
    standardizer_robust = FeatureStandardizer(method='robust')
    standardized_robust = standardizer_robust.fit_transform(features)

    print("Robust标准化后统计:")
    print(standardized_robust.describe())

    return standardized_zscore


def monitoring_example():
    """
    监控示例
    """
    print("\n" + "="*50)
    print("监控示例")
    print("="*50)

    # 创建监控器
    monitor = FeaturesMonitor()

    # 注册组件
    monitor.register_component("feature_engineer", "processor")
    monitor.register_component("technical_processor", "processor")
    monitor.register_component("feature_selector", "processor")
    monitor.register_component("feature_standardizer", "processor")

    # 启动监控
    monitor.start_monitoring()

    # 模拟处理过程
    print("\n模拟特征处理过程...")

    # 收集性能指标
    monitor.collect_metrics("feature_engineer", "processing_time", 1.5, MetricType.HISTOGRAM)
    monitor.collect_metrics("feature_engineer", "features_generated", 15, MetricType.COUNTER)
    monitor.collect_metrics("feature_engineer", "memory_usage", 256, MetricType.GAUGE)

    monitor.collect_metrics("technical_processor", "indicators_calculated", 8, MetricType.COUNTER)
    monitor.collect_metrics("technical_processor", "calculation_time", 0.8, MetricType.HISTOGRAM)

    monitor.collect_metrics("feature_selector", "features_selected", 5, MetricType.COUNTER)
    monitor.collect_metrics("feature_selector", "selection_time", 0.3, MetricType.HISTOGRAM)

    monitor.collect_metrics("feature_standardizer", "features_standardized", 5, MetricType.COUNTER)
    monitor.collect_metrics("feature_standardizer", "standardization_time",
                            0.2, MetricType.HISTOGRAM)

    # 获取性能报告
    report = monitor.get_performance_report()

    print("\n性能监控报告:")
    print(f"总组件数: {report.get('total_components', 0)}")
    print(f"运行中组件: {report.get('running_components', 0)}")
    print(f"平均处理时间: {report.get('avg_processing_time', 0):.3f}秒")
    print(f"平均内存使用: {report.get('avg_memory_usage', 0):.1f}MB")

    # 停止监控
    monitor.stop_monitoring()

    return report


def visualization_example(data, features):
    """
    可视化示例

    参数:
        data (pd.DataFrame): 原始数据
        features (pd.DataFrame): 特征数据
    """
    print("\n" + "="*50)
    print("可视化示例")
    print("="*50)

    try:
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('特征工程可视化示例', fontsize=16)

        # 1. 价格走势
        axes[0, 0].plot(data.index, data['close'], label='收盘价', linewidth=2)
        if 'MA_5' in features.columns:
            axes[0, 0].plot(features.index, features['MA_5'], label='MA5', alpha=0.7)
        if 'MA_10' in features.columns:
            axes[0, 0].plot(features.index, features['MA_10'], label='MA10', alpha=0.7)
        axes[0, 0].set_title('价格走势与技术指标')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. RSI指标
        if 'RSI' in features.columns:
            axes[0, 1].plot(features.index, features['RSI'], label='RSI', color='orange')
            axes[0, 1].axhline(y=70, color='r', linestyle='--', alpha=0.5, label='超买线')
            axes[0, 1].axhline(y=30, color='g', linestyle='--', alpha=0.5, label='超卖线')
            axes[0, 1].set_title('RSI指标')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # 3. MACD指标
        if 'MACD_DIF' in features.columns and 'MACD_DEA' in features.columns:
            axes[1, 0].plot(features.index, features['MACD_DIF'], label='MACD DIF', linewidth=2)
            axes[1, 0].plot(features.index, features['MACD_DEA'], label='MACD DEA', linewidth=2)
            if 'MACD_Histogram' in features.columns:
                axes[1, 0].bar(features.index, features['MACD_Histogram'],
                               label='MACD Histogram', alpha=0.5, width=0.8)
            axes[1, 0].set_title('MACD指标')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # 4. 布林带
        if 'BOLL_UPPER' in features.columns and 'BOLL_LOWER' in features.columns:
            axes[1, 1].plot(data.index, data['close'], label='收盘价', linewidth=2)
            axes[1, 1].plot(features.index, features['BOLL_UPPER'], label='布林上轨', alpha=0.7)
            axes[1, 1].plot(features.index, features['BOLL_LOWER'], label='布林下轨', alpha=0.7)
            if 'BOLL_MIDDLE' in features.columns:
                axes[1, 1].plot(features.index, features['BOLL_MIDDLE'], label='布林中轨', alpha=0.7)
            axes[1, 1].set_title('布林带指标')
            axes[1, 1].legend()
        else:
            # 如果没有布林带，显示成交量
            axes[1, 1].bar(data.index, data['volume'], alpha=0.6, label='成交量')
            axes[1, 1].set_title('成交量')
            axes[1, 1].legend()

        axes[1, 1].grid(True, alpha=0.3)

        # 调整布局
        plt.tight_layout()

        # 保存图片
        plt.savefig('feature_engineering_example.png', dpi=300, bbox_inches='tight')
        print("可视化图表已保存为: feature_engineering_example.png")

        # 显示图表
        plt.show()

    except Exception as e:
        print(f"可视化过程中出现错误: {e}")
        print("请确保已安装matplotlib库")


def main():
    """
    主函数
    """
    print("RQA2025 特征层基础特征工程示例")
    print("="*50)

    try:
        # 1. 基础特征工程
        features = basic_feature_engineering_example()

        # 2. 特征选择
        selected_features = feature_selection_example(features)

        # 3. 特征标准化
        standardized_features = feature_standardization_example(selected_features)

        # 4. 监控示例
        monitoring_report = monitoring_example()

        # 5. 可视化
        data = create_sample_data(100)
        visualization_example(data, features)

        print("\n" + "="*50)
        print("示例执行完成！")
        print("="*50)

        # 总结
        print("\n总结:")
        print(f"- 原始数据形状: {data.shape}")
        print(f"- 生成特征数量: {len(features.columns)}")
        print(f"- 选择特征数量: {len(selected_features.columns)}")
        print(f"- 标准化特征数量: {len(standardized_features.columns)}")
        print(f"- 监控组件数量: {monitoring_report.get('total_components', 0)}")

    except Exception as e:
        print(f"示例执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
