#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试工厂创建功能
"""

from src.features.core.factory import FeatureProcessorFactory
from src.features.feature_config import FeatureType, FeatureConfig


def debug_factory():
    """调试工厂创建功能"""
    print("开始调试工厂...")

    # 创建工厂
    factory = FeatureProcessorFactory()
    print(f"工厂创建成功: {type(factory)}")

    # 列出可用处理器
    processors = factory.list_available_processors()
    print(f"可用处理器: {processors}")

    # 检查manager处理器的详细信息
    manager_info = factory._processors['manager']
    print(f"Manager处理器详细信息: {manager_info}")
    print(f"Manager处理器类: {manager_info['class']}")
    print(f"Manager处理器类名: {manager_info['class'].__name__}")
    print(f"Manager处理器模块: {manager_info['class'].__module__}")

    # 检查类的构造函数参数
    import inspect
    sig = inspect.signature(manager_info['class'].__init__)
    print(f"Manager构造函数签名: {sig}")

    # 测试FeatureConfig创建
    try:
        config = {
            'feature_types': [FeatureType.TECHNICAL],
            'technical_indicators': ["sma", "rsi"],
            'enable_feature_selection': False,
            'enable_standardization': True
        }

        print(f"尝试创建FeatureConfig: {config}")
        feature_config = FeatureConfig(
            feature_types=config.get('feature_types', [FeatureType.TECHNICAL]),
            technical_indicators=config.get('technical_indicators', ["sma", "rsi"]),
            enable_feature_selection=config.get('enable_feature_selection', False),
            enable_standardization=config.get('enable_standardization', True)
        )
        print(f"FeatureConfig创建成功: {feature_config}")

        # 尝试直接创建FeatureManager
        print("尝试直接创建FeatureManager...")
        manager_class = manager_info['class']
        manager = manager_class(feature_config)
        print(f"FeatureManager直接创建成功: {type(manager)}")

    except Exception as e:
        print(f"创建失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_factory()
