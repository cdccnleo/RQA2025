#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征层错误处理使用示例

展示优化后的特征层错误处理机制，包括异常类型、错误恢复和最佳实践。
"""

import pandas as pd
import numpy as np
from datetime import datetime

# 导入特征层组件和异常类
from src.features import (
    FeatureEngine,
    FeatureEngineer,
    SentimentAnalyzer,
    DefaultConfigs,
    # 异常类
    FeatureDataValidationError,
    FeatureConfigValidationError,
    FeatureProcessingError,
    FeatureStandardizationError,
    FeatureExceptionFactory,
    FeatureExceptionHandler,
    handle_feature_exception
)
from src.features.feature_config import FeatureType


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


def example_data_validation_errors():
    """示例：数据验证错误处理"""
    print("=== 数据验证错误处理示例 ===")

    # 创建异常工厂
    factory = FeatureExceptionFactory()

    # 1. 空数据错误
    try:
        empty_data = pd.DataFrame()
        if empty_data.empty:
            raise factory.create_data_validation_error(
                "输入数据为空",
                data_shape=empty_data.shape
            )
    except FeatureDataValidationError as e:
        print(f"捕获数据验证错误: {e}")
        print(f"错误类型: {e.error_type}")
        print(f"数据形状: {e.data_shape}")

    # 2. 缺失列错误
    try:
        incomplete_data = pd.DataFrame({
            'close': [100, 101, 102],
            'high': [102, 103, 104]
            # 缺少 'low' 和 'volume' 列
        })

        required_columns = ['close', 'high', 'low', 'volume']
        missing_columns = [col for col in required_columns if col not in incomplete_data.columns]

        if missing_columns:
            raise factory.create_data_validation_error(
                "缺失必要列",
                missing_columns=missing_columns,
                data_shape=incomplete_data.shape
            )
    except FeatureDataValidationError as e:
        print(f"捕获缺失列错误: {e}")
        print(f"缺失列: {e.missing_columns}")

    # 3. 数据类型错误
    try:
        invalid_data = pd.DataFrame({
            'close': ['100', '101', '102'],  # 字符串而不是数值
            'high': [102, 103, 104],
            'low': [99, 100, 101],
            'volume': [1000000, 1100000, 1200000]
        })

        invalid_types = []
        for col in ['close', 'high', 'low', 'volume']:
            if not pd.api.types.is_numeric_dtype(invalid_data[col]):
                invalid_types.append(col)

        if invalid_types:
            raise factory.create_data_validation_error(
                "数据类型不正确",
                invalid_types=invalid_types,
                data_shape=invalid_data.shape
            )
    except FeatureDataValidationError as e:
        print(f"捕获数据类型错误: {e}")
        print(f"无效类型列: {e.invalid_types}")


def example_config_validation_errors():
    """示例：配置验证错误处理"""
    print("\n=== 配置验证错误处理示例 ===")

    factory = FeatureExceptionFactory()

    # 1. 无效配置参数
    try:
        # 模拟无效配置
        invalid_config = {
            'max_features': 5,
            'min_features': 10  # 最大特征数小于最小特征数
        }

        if invalid_config['max_features'] < invalid_config['min_features']:
            raise factory.create_config_validation_error(
                "最大特征数不能小于最小特征数",
                config_field="max_features",
                expected_value=f">= {invalid_config['min_features']}",
                actual_value=invalid_config['max_features'],
                config_dict=invalid_config
            )
    except FeatureConfigValidationError as e:
        print(f"捕获配置验证错误: {e}")
        print(f"配置字段: {e.config_field}")
        print(f"期望值: {e.expected_value}")
        print(f"实际值: {e.actual_value}")

    # 2. 不支持的特征类型
    try:
        unsupported_feature_type = "UNKNOWN_TYPE"
        supported_types = [FeatureType.TECHNICAL, FeatureType.SENTIMENT]

        if unsupported_feature_type not in [ft.value for ft in supported_types]:
            raise factory.create_config_validation_error(
                "不支持的特征类型",
                config_field="feature_types",
                expected_value=supported_types,
                actual_value=unsupported_feature_type
            )
    except FeatureConfigValidationError as e:
        print(f"捕获不支持特征类型错误: {e}")


def example_processing_errors():
    """示例：处理错误处理"""
    print("\n=== 处理错误处理示例 ===")

    factory = FeatureExceptionFactory()

    # 1. 特征计算错误
    try:
        data = create_sample_stock_data()

        # 模拟特征计算失败
        feature_name = "invalid_feature"
        try:
            if feature_name == "sma":
                result = data['close'].rolling(window=20).mean()
            else:
                raise ValueError(f"不支持的特征: {feature_name}")
        except Exception as original_error:
            raise factory.create_processing_error(
                f"特征计算失败: {feature_name}",
                processor_name="technical_processor",
                step="feature_computation",
                original_error=original_error,
                feature_name=feature_name
            )
    except FeatureProcessingError as e:
        print(f"捕获处理错误: {e}")
        print(f"处理器: {e.processor_name}")
        print(f"步骤: {e.step}")
        print(f"特征: {e.feature_name}")
        print(f"原始错误: {e.original_error}")


def example_standardization_errors():
    """示例：标准化错误处理"""
    print("\n=== 标准化错误处理示例 ===")

    factory = FeatureExceptionFactory()

    # 1. 标准化器未拟合错误
    try:
        from src.features import FeatureStandardizer
        from pathlib import Path

        standardizer = FeatureStandardizer(Path("./models/features"))

        # 尝试在未拟合的情况下转换数据
        data = create_sample_stock_data()
        try:
            result = standardizer.transform(data)
        except Exception as original_error:
            raise factory.create_standardization_error(
                "标准化器尚未拟合",
                method=standardizer.method,
                is_fitted=standardizer.is_fitted
            )
    except FeatureStandardizationError as e:
        print(f"捕获标准化错误: {e}")
        print(f"标准化方法: {e.method}")
        print(f"已拟合: {e.is_fitted}")


def example_error_handler_usage():
    """示例：错误处理器使用"""
    print("\n=== 错误处理器使用示例 ===")

    handler = FeatureExceptionHandler()

    # 处理各种异常
    exceptions = [
        FeatureDataValidationError("测试数据验证错误", data_shape=(0, 0)),
        FeatureConfigValidationError("测试配置验证错误", config_field="test"),
        FeatureProcessingError("测试处理错误", processor_name="test_processor")
    ]

    for exception in exceptions:
        enhanced_exception = handler.handle_exception(exception, {
            "test_context": "example",
            "timestamp": datetime.now().isoformat()
        })
        print(f"增强的异常: {enhanced_exception}")

    # 获取错误摘要
    summary = handler.get_error_summary()
    print(f"错误摘要: {summary}")


@handle_feature_exception
def example_decorator_usage(data: pd.DataFrame, config):
    """示例：装饰器使用"""
    """使用错误处理装饰器的函数"""
    print(f"处理数据: {data.shape}")
    print(f"使用配置: {config}")

    # 模拟处理过程
    if data.empty:
        raise FeatureDataValidationError("数据为空")

    return data


def example_robust_feature_processing():
    """示例：健壮的特征处理"""
    print("\n=== 健壮的特征处理示例 ===")

    # 创建引擎和配置
    engine = FeatureEngine()
    config = DefaultConfigs.basic_technical()

    # 1. 正常数据处理
    try:
        data = create_sample_stock_data()
        features = engine.process_features(data, config)
        print(f"正常处理成功，特征形状: {features.shape}")
    except Exception as e:
        print(f"正常处理失败: {e}")

    # 2. 错误数据处理
    try:
        # 使用装饰器处理
        result = example_decorator_usage(pd.DataFrame(), config)
        print(f"装饰器处理结果: {result.shape}")
    except Exception as e:
        print(f"装饰器处理失败: {e}")

    # 3. 错误恢复
    try:
        # 尝试处理无效数据
        invalid_data = pd.DataFrame({
            'close': ['invalid', 'data'],  # 无效数据类型
            'high': [102, 103],
            'low': [99, 100],
            'volume': [1000000, 1100000]
        })

        # 使用错误处理装饰器
        result = example_decorator_usage(invalid_data, config)
        print(f"错误恢复结果: {result.shape}")
    except Exception as e:
        print(f"错误恢复失败: {e}")


def example_error_recovery_strategies():
    """示例：错误恢复策略"""
    print("\n=== 错误恢复策略示例 ===")

    factory = FeatureExceptionFactory()
    config = DefaultConfigs.basic_technical()

    def safe_feature_processing(data: pd.DataFrame, config):
        """安全的特征处理函数"""
        try:
            # 1. 数据验证
            if data.empty:
                raise factory.create_data_validation_error("输入数据为空")

            required_columns = ['close', 'high', 'low', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise factory.create_data_validation_error("缺失必要列", missing_columns=missing_columns)

            # 2. 特征处理
            engine = FeatureEngine()
            return engine.process_features(data, config)

        except FeatureDataValidationError as e:
            print(f"数据验证错误，尝试清理数据: {e}")
            # 尝试清理数据
            cleaned_data = data.copy()
            if not cleaned_data.empty:
                # 填充缺失值
                cleaned_data = cleaned_data.fillna(method='ffill').fillna(method='bfill')
                # 移除无效列
                cleaned_data = cleaned_data.select_dtypes(include=[np.number])
                if not cleaned_data.empty:
                    engine = FeatureEngine()
                    return engine.process_features(cleaned_data, config)
            return pd.DataFrame()

        except FeatureConfigValidationError as e:
            print(f"配置验证错误，使用默认配置: {e}")
            # 使用默认配置
            default_config = DefaultConfigs.basic_technical()
            engine = FeatureEngine()
            return engine.process_features(data, default_config)

        except FeatureProcessingError as e:
            print(f"处理错误，返回原始数据: {e}")
            # 返回原始数据
            return data

        except Exception as e:
            print(f"未知错误: {e}")
            return pd.DataFrame()

    # 测试各种错误情况
    test_cases = [
        ("空数据", pd.DataFrame()),
        ("缺失列数据", pd.DataFrame({'close': [100, 101]})),
        ("正常数据", create_sample_stock_data())
    ]

    for case_name, test_data in test_cases:
        print(f"\n测试: {case_name}")
        try:
            result = safe_feature_processing(test_data, config)
            print(f"处理结果形状: {result.shape}")
        except Exception as e:
            print(f"处理失败: {e}")


def main():
    """主函数"""
    print("特征层错误处理使用示例")
    print("=" * 60)

    # 运行各种示例
    example_data_validation_errors()
    example_config_validation_errors()
    example_processing_errors()
    example_standardization_errors()
    example_error_handler_usage()
    example_robust_feature_processing()
    example_error_recovery_strategies()

    print("\n" + "=" * 60)
    print("错误处理示例运行完成")


if __name__ == "__main__":
    main()
