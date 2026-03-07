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
    'FeatureManager'
]
