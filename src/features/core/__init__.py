"""
特征核心模块

提供特征层的核心组件，包括配置管理、引擎和处理器等。
"""

from .config import (
    FeatureConfig,
    FeatureProcessingConfig,
    TechnicalParams,
    SentimentParams,
    TechnicalIndicatorType,
    SentimentType,
    OrderBookConfig,
    OrderBookType,
    DefaultConfigs
)

from .config_integration import *
from .dependency_manager import DependencyManager, safe_import, get_gpu_count
from .factory import *
from .feature_engineer import *
from .feature_manager import *
from .feature_saver import *
from .parallel_feature_processor import *
from .signal_generator import *
from .version_management import *

from .engine import FeatureEngine
from .manager import FeatureManager

__all__ = [
    'FeatureConfig',
    'FeatureProcessingConfig',
    'TechnicalParams',
    'SentimentParams',
    'TechnicalIndicatorType',
    'SentimentType',
    'OrderBookConfig',
    'OrderBookType',
    'DefaultConfigs',
    'FeatureEngine',
    'FeatureManager',
    # 新增导入的模块
    'FeatureEngineer',
    'FeatureSaver',
    'DependencyManager',
    'ParallelFeatureProcessor',
    'SignalGenerator',
    'VersionManagement'
]
