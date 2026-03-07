#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征层模块

提供特征工程、处理、选择、标准化等功能。

架构分层：
- 核心组件层：FeatureEngineer、FeatureProcessor、FeatureSelector、FeatureStandardizer、FeatureSaver
- 类型定义层：FeatureType等枚举和类型定义
- 处理器模块层：BaseFeatureProcessor、TechnicalProcessor、ProcessorFeatureEngineer
- 分析器模块层：SentimentAnalyzer等专业分析器
- 高频优化模块：HighFreqOptimizer（预留）
- 订单簿分析模块：OrderBookAnalyzer（预留）
- 插件系统：FeaturePluginManager、BaseFeaturePlugin等

典型用法：
```python
# 合理跨层级导入：features模块初始化文件导入自己的子模块
# 合理跨层级导入：这是模块内部的正常导入，不涉及跨层问题
from . import FeatureEngineer, FeatureProcessor, FeaturePluginManager

# 特征工程
engineer = FeatureEngineer()
    features = engineer.extract_features(data)

# 特征处理
processor = FeatureProcessor()
    processed_features = processor.process(features)

# 插件系统
plugin_manager = FeaturePluginManager()
    plugin_manager.discover_and_load_plugins()

# 使用统一工厂（推荐）
# 合理跨层级导入：features模块初始化文件导入自己的子模块
# 合理跨层级导入：这是模块内部的正常导入，不涉及跨层问题
from . import get_unified_feature_manager
manager = get_unified_feature_manager()
    features = manager.process_features(data, feature_configs)
        ```

作者：RQA Team
版本：1.1.0
"""

# 核心组件
from .core.engine import FeatureEngine
from .core.config import (
    FeatureConfig,
    FeatureType,
    TechnicalParams,
    SentimentParams,
    OrderBookConfig,
    OrderBookType
)
from .core.feature_engineer import FeatureEngineer
from .processors.general_processor import FeatureProcessor
from .processors.feature_selector import FeatureSelector
from .processors.feature_standardizer import FeatureStandardizer
from .core.feature_saver import FeatureSaver
from .quality_assessor import QualityAssessor, QualityAssessorConfig
from .feature_store import FeatureStore, StoreConfig, FeatureMetadata

# 统一工厂（新增）
from .core.factory import (
    FeatureProcessorFactory,
    UnifiedFeatureManager,
    ProcessorType,
    feature_processor_factory,
    create_feature_processor,
    get_unified_feature_manager
)

# 处理器
from .processors.base_processor import BaseFeatureProcessor
from .processors.technical import TechnicalProcessor
# from .processors.processor_feature_engineer import ProcessorFeatureEngineer  # 暂未实现

# 分析器
from .sentiment.sentiment_analyzer import SentimentAnalyzer

# 插件系统
from .plugins import (
    FeaturePluginManager,
    BaseFeaturePlugin,
    PluginMetadata,
    PluginType,
    PluginStatus,
    PluginRegistry,
    PluginLoader,
    PluginValidator
)

# 性能监控
from .monitoring import (
    FeaturesMonitor,
    MetricsCollector,
    AlertManager,
    PerformanceAnalyzer
)

# 分布式计算
from .distributed import (
    FeatureTaskScheduler,
    FeatureWorkerManager,
    DistributedFeatureProcessor,
    FeatureLoadBalancer
)

# 异常处理
from .core.exceptions import (
    FeatureDataValidationError,
    FeatureConfigValidationError,
    FeatureProcessingError,
    FeatureStandardizationError,
    FeatureSelectionError,
    FeatureSentimentError,
    FeatureTechnicalError,
    FeatureGeneralError,
    FeatureExceptionFactory,
    FeatureExceptionHandler,
    handle_feature_exception
)

# 向后兼容性别名
FeatureManager = FeatureEngine  # 保持向后兼容

__all__ = [
    # 核心组件
    'FeatureEngine',
    'FeatureConfig',
    'FeatureType',
    'TechnicalParams',
    'SentimentParams',
    'FeatureEngineer',
    'FeatureProcessor',
    'FeatureSelector',
    'FeatureStandardizer',
    'FeatureSaver',
    'QualityAssessor',
    'QualityAssessorConfig',
    'FeatureStore',
    'StoreConfig',
    'FeatureMetadata',

    # 统一工厂（新增）
    'FeatureProcessorFactory',
    'UnifiedFeatureManager',
    'ProcessorType',
    'feature_processor_factory',
    'create_feature_processor',
    'get_unified_feature_manager',

    # 处理器
    'BaseFeatureProcessor',
    'TechnicalProcessor',
    # 'ProcessorFeatureEngineer',  # 暂未实现

    # 分析器
    'SentimentAnalyzer',

    # 插件系统
    'FeaturePluginManager',
    'BaseFeaturePlugin',
    'PluginMetadata',
    'PluginType',
    'PluginStatus',
    'PluginRegistry',
    'PluginLoader',
    'PluginValidator',

    # 性能监控
    'FeaturesMonitor',
    'MetricsCollector',
    'AlertManager',
    'PerformanceAnalyzer',

    # 分布式计算
    'FeatureTaskScheduler',
    'FeatureWorkerManager',
    'DistributedFeatureProcessor',
    'FeatureLoadBalancer',

    # 异常处理
    'FeatureDataValidationError',
    'FeatureConfigValidationError',
    'FeatureProcessingError',
    'FeatureStandardizationError',
    'FeatureSelectionError',
    'FeatureSentimentError',
    'FeatureTechnicalError',
    'FeatureGeneralError',
    'FeatureExceptionFactory',
    'FeatureExceptionHandler',
    'handle_feature_exception',

    # 向后兼容
    'FeatureManager'
]
