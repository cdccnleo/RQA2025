#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
插件使用示例

演示如何使用特征层插件系统。
"""

from examples.plugins.sentiment_plugin import SentimentAnalysisPlugin
from examples.plugins.technical_plugin import TechnicalIndicatorPlugin
from src.features.plugins import FeaturePluginManager, PluginType
import pandas as pd
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_sample_data() -> pd.DataFrame:
    """创建示例数据"""
    # 股票数据
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)

    stock_data = pd.DataFrame({
        'date': dates,
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 105,
        'low': np.random.randn(100).cumsum() + 95,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    })

    # 新闻数据
    news_data = pd.DataFrame({
        'date': dates,
        'title': [
            f"股票市场表现{'优秀' if i % 3 == 0 else '一般' if i % 3 == 1 else '糟糕'}"
            for i in range(100)
        ],
        'content': [
            f"今日市场{'表现良好，投资者信心充足' if i % 3 == 0 else '表现平平，投资者观望情绪浓厚' if i % 3 == 1 else '表现不佳，投资者担忧情绪上升'}"
            for i in range(100)
        ]
    })

    # 合并数据
    data = pd.merge(stock_data, news_data, on='date', how='inner')
    return data


def example_plugin_management():
    """示例：插件管理"""
    print("=== 插件管理示例 ===")

    # 创建插件管理器
    plugin_manager = FeaturePluginManager()

    # 创建插件实例
    technical_plugin = TechnicalIndicatorPlugin({
        "sma_periods": [5, 10, 20],
        "ema_periods": [12, 26],
        "rsi_period": 14
    })

    sentiment_plugin = SentimentAnalysisPlugin({
        "language": "zh-cn",
        "confidence_threshold": 0.6
    })

    # 注册插件
    print("注册插件...")
    success1 = plugin_manager.register_plugin(technical_plugin)
    success2 = plugin_manager.register_plugin(sentiment_plugin)

    print(f"技术指标插件注册: {'成功' if success1 else '失败'}")
    print(f"情感分析插件注册: {'成功' if success2 else '失败'}")

    # 列出所有插件
    print(f"\n已注册插件数量: {len(plugin_manager)}")
    all_plugins = plugin_manager.list_plugins()
    print(f"插件列表: {all_plugins}")

    # 按类型获取插件
    processors = plugin_manager.get_plugins_by_type(PluginType.PROCESSOR)
    analyzers = plugin_manager.get_plugins_by_type(PluginType.ANALYZER)

    print(f"处理器插件: {len(processors)} 个")
    print(f"分析器插件: {len(analyzers)} 个")

    # 获取插件信息
    for plugin_name in all_plugins:
        info = plugin_manager.get_plugin_info(plugin_name)
        if info:
            metadata = info['metadata']
            print(f"\n插件: {metadata['name']}")
            print(f"  版本: {metadata['version']}")
            print(f"  描述: {metadata['description']}")
            print(f"  类型: {metadata['plugin_type']}")
            print(f"  标签: {metadata['tags']}")

    # 获取统计信息
    stats = plugin_manager.get_plugin_stats()
    print(f"\n插件统计: {stats}")

    return plugin_manager


def example_plugin_processing():
    """示例：插件处理"""
    print("\n=== 插件处理示例 ===")

    # 创建数据
    data = create_sample_data()
    print(f"原始数据形状: {data.shape}")
    print(f"原始列: {list(data.columns)}")

    # 创建插件管理器
    plugin_manager = FeaturePluginManager()

    # 注册插件
    technical_plugin = TechnicalIndicatorPlugin({
        "sma_periods": [5, 10, 20],
        "ema_periods": [12, 26],
        "rsi_period": 14
    })

    sentiment_plugin = SentimentAnalysisPlugin({
        "language": "zh-cn",
        "confidence_threshold": 0.6
    })

    plugin_manager.register_plugin(technical_plugin)
    plugin_manager.register_plugin(sentiment_plugin)

    # 使用技术指标插件处理数据
    print("\n使用技术指标插件...")
    technical_result = technical_plugin.process(data)
    print(f"技术指标处理后数据形状: {technical_result.shape}")

    # 查看新增的技术指标列
    original_cols = set(data.columns)
    new_cols = set(technical_result.columns) - original_cols
    print(f"新增技术指标列: {list(new_cols)}")

    # 使用情感分析插件处理数据
    print("\n使用情感分析插件...")
    sentiment_result = sentiment_plugin.process(data)
    print(f"情感分析处理后数据形状: {sentiment_result.shape}")

    # 查看新增的情感分析列
    new_sentiment_cols = set(sentiment_result.columns) - original_cols
    print(f"新增情感分析列: {list(new_sentiment_cols)}")

    # 组合处理
    print("\n组合处理...")
    combined_result = sentiment_plugin.process(technical_result)
    print(f"组合处理后数据形状: {combined_result.shape}")

    return combined_result


def example_plugin_validation():
    """示例：插件验证"""
    print("\n=== 插件验证示例 ===")

    plugin_manager = FeaturePluginManager()

    # 创建插件
    technical_plugin = TechnicalIndicatorPlugin()
    sentiment_plugin = SentimentAnalysisPlugin()

    # 注册插件
    plugin_manager.register_plugin(technical_plugin)
    plugin_manager.register_plugin(sentiment_plugin)

    # 验证插件
    print("验证插件...")
    validation_results = plugin_manager.validate_all_plugins()

    for plugin_name, is_valid in validation_results.items():
        status = "通过" if is_valid else "失败"
        print(f"插件 {plugin_name}: {status}")

    # 验证特定插件
    print("\n验证特定插件...")
    for plugin_name in plugin_manager.list_plugins():
        is_valid = plugin_manager.validate_plugin(plugin_name)
        status = "通过" if is_valid else "失败"
        print(f"插件 {plugin_name}: {status}")

        # 获取插件信息
        info = plugin_manager.get_plugin_info(plugin_name)
        if info:
            capabilities = info.get('capabilities', {})
            print(f"  能力: {capabilities}")


def example_plugin_lifecycle():
    """示例：插件生命周期"""
    print("\n=== 插件生命周期示例 ===")

    plugin_manager = FeaturePluginManager()

    # 创建插件
    technical_plugin = TechnicalIndicatorPlugin()

    # 注册插件
    print("1. 注册插件")
    success = plugin_manager.register_plugin(technical_plugin)
    print(f"注册结果: {'成功' if success else '失败'}")

    # 初始化插件
    print("\n2. 初始化插件")
    init_success = plugin_manager.initialize_plugin("technical_indicator_plugin")
    print(f"初始化结果: {'成功' if init_success else '失败'}")

    # 获取插件信息
    print("\n3. 获取插件信息")
    info = plugin_manager.get_plugin_info("technical_indicator_plugin")
    if info:
        metadata = info['metadata']
        print(f"状态: {metadata['status']}")
        print(f"加载时间: {metadata['load_time']}")

    # 处理数据
    print("\n4. 处理数据")
    data = create_sample_data()
    plugin = plugin_manager.get_plugin("technical_indicator_plugin")
    if plugin:
        result = plugin.process(data)
        print(f"处理结果形状: {result.shape}")

    # 清理插件
    print("\n5. 清理插件")
    cleanup_success = plugin_manager.cleanup_plugin("technical_indicator_plugin")
    print(f"清理结果: {'成功' if cleanup_success else '失败'}")

    # 注销插件
    print("\n6. 注销插件")
    unregister_success = plugin_manager.unregister_plugin("technical_indicator_plugin")
    print(f"注销结果: {'成功' if unregister_success else '失败'}")


def example_plugin_configuration():
    """示例：插件配置"""
    print("\n=== 插件配置示例 ===")

    # 不同配置的技术指标插件
    configs = [
        {
            "name": "短期技术指标",
            "config": {
                "sma_periods": [5, 10],
                "ema_periods": [12],
                "rsi_period": 7
            }
        },
        {
            "name": "长期技术指标",
            "config": {
                "sma_periods": [20, 50, 100],
                "ema_periods": [26, 50],
                "rsi_period": 21
            }
        },
        {
            "name": "完整技术指标",
            "config": {
                "sma_periods": [5, 10, 20, 50],
                "ema_periods": [12, 26, 50],
                "rsi_period": 14,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9
            }
        }
    ]

    plugin_manager = FeaturePluginManager()
    data = create_sample_data()

    for config_info in configs:
        print(f"\n使用配置: {config_info['name']}")

        # 创建插件
        plugin = TechnicalIndicatorPlugin(config_info['config'])

        # 注册插件
        plugin_manager.register_plugin(plugin)

        # 处理数据
        result = plugin.process(data)

        # 查看结果
        original_cols = set(data.columns)
        new_cols = set(result.columns) - original_cols
        print(f"  新增列数: {len(new_cols)}")
        print(f"  新增列: {list(new_cols)[:5]}...")  # 只显示前5个

        # 注销插件
        plugin_manager.unregister_plugin(plugin.metadata.name)


def main():
    """主函数"""
    print("特征层插件系统使用示例")
    print("=" * 50)

    try:
        # 插件管理示例
        plugin_manager = example_plugin_management()

        # 插件处理示例
        result = example_plugin_processing()

        # 插件验证示例
        example_plugin_validation()

        # 插件生命周期示例
        example_plugin_lifecycle()

        # 插件配置示例
        example_plugin_configuration()

        print("\n" + "=" * 50)
        print("所有示例执行完成！")

    except Exception as e:
        print(f"示例执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
