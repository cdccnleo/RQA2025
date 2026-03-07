#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征层使用示例

展示优化后的特征层如何使用，包括特征引擎、特征工程器、情感分析器等组件。
"""

import pandas as pd
import numpy as np

# 导入特征层组件
from src.features import (
    FeatureEngine,
    FeatureEngineer,
    SentimentAnalyzer,
    DefaultConfigs
)
from src.features.feature_config import FeatureType, FeatureProcessingConfig


def create_sample_stock_data():
    """创建示例股票数据"""
    dates = pd.date_range('2024-01-01', periods=100, freq='D')

    # 生成模拟股票数据
    np.random.seed(42)
    base_price = 100
    returns = np.random.normal(0.001, 0.02, 100)
    prices = [base_price]

    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    data = pd.DataFrame({
        'close': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)

    return data


def create_sample_news_data():
    """创建示例新闻数据"""
    dates = pd.date_range('2024-01-01', periods=50, freq='D')

    news_data = pd.DataFrame({
        'date': dates,
        'content': [
            '公司发布利好消息，股价有望上涨',
            '市场对该公司前景持乐观态度',
            '分析师预测该公司业绩将超预期',
            '行业整体表现良好，带动相关股票上涨',
            '公司新产品获得市场认可，销售表现强劲'
        ] * 10  # 重复使用这些新闻内容
    })

    return news_data


def example_feature_engine_usage():
    """示例：使用特征引擎（推荐方式）"""
    print("=== 特征引擎使用示例 ===")

    # 创建示例数据
    stock_data = create_sample_stock_data()
    print(f"原始数据形状: {stock_data.shape}")

    # 使用默认配置
    engine = FeatureEngine()
    config = DefaultConfigs.basic_technical()

    try:
        # 处理特征
        features = engine.process_features(stock_data, config)
        print(f"处理后特征形状: {features.shape}")
        print(f"特征列: {list(features.columns)}")

        # 获取引擎统计信息
        stats = engine.get_stats()
        print(f"处理统计: {stats}")

        # 获取支持的处理器
        processors = engine.list_processors()
        print(f"可用处理器: {processors}")

    except Exception as e:
        print(f"特征处理失败: {e}")


def example_feature_engineer_usage():
    """示例：直接使用特征工程器"""
    print("\n=== 特征工程器使用示例 ===")

    # 创建示例数据
    stock_data = create_sample_stock_data()

    # 创建特征工程器
    engineer = FeatureEngineer()

    try:
        # 验证数据
        engineer._validate_stock_data(stock_data)
        print("数据验证通过")

        # 注册特征
        from src.features.feature_config import FeatureConfig
        config = FeatureConfig("test", FeatureType.TECHNICAL)
        engineer.register_feature(config)
        print("特征注册成功")

    except Exception as e:
        print(f"特征工程失败: {e}")


def example_sentiment_analyzer_usage():
    """示例：使用情感分析器"""
    print("\n=== 情感分析器使用示例 ===")

    # 创建情感分析器
    analyzer = SentimentAnalyzer()

    try:
        # 分析单条文本
        sample_text = "公司业绩表现优秀，市场前景看好"
        sentiment_result = analyzer.analyze_text(sample_text)
        print(f"单条文本情感分析: {sentiment_result}")

    except Exception as e:
        print(f"情感分析失败: {e}")


def example_custom_config_usage():
    """示例：使用自定义配置"""
    print("\n=== 自定义配置使用示例 ===")

    # 创建自定义配置
    custom_config = FeatureProcessingConfig(
        technical_indicators=['sma', 'ema', 'rsi', 'macd', 'bbands'],
        sentiment_analysis_enabled=True,
        enable_caching=True,
        max_features=20,
        standardization_method='zscore'
    )

    print("自定义配置:")
    print(f"- 技术指标: {custom_config.technical_indicators}")
    print(f"- 情感分析启用: {custom_config.sentiment_analysis_enabled}")
    print(f"- 缓存启用: {custom_config.enable_caching}")
    print(f"- 最大特征数: {custom_config.max_features}")
    print(f"- 标准化方法: {custom_config.standardization_method}")

    # 配置创建成功
    print("配置创建成功")


def example_error_handling():
    """示例：错误处理"""
    print("\n=== 错误处理示例 ===")

    # 创建空数据
    empty_data = pd.DataFrame()

    # 创建引擎
    engine = FeatureEngine()
    config = DefaultConfigs.basic_technical()

    try:
        # 尝试处理空数据
        features = engine.process_features(empty_data, config)
        print("处理成功")
    except ValueError as e:
        print(f"数据验证错误: {e}")
    except Exception as e:
        print(f"其他错误: {e}")


def main():
    """主函数"""
    print("特征层使用示例")
    print("=" * 50)

    # 运行各种示例
    example_feature_engine_usage()
    example_feature_engineer_usage()
    example_sentiment_analyzer_usage()
    example_custom_config_usage()
    example_error_handling()

    print("\n" + "=" * 50)
    print("示例运行完成")


if __name__ == "__main__":
    main()
