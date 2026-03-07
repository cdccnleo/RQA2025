#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
另类数据适配器模块

提供非传统市场数据获取能力，支持：
- 社交媒体情绪数据
- 新闻情绪分析
- 搜索趋势数据
- 卫星/替代数据
- 数据融合引擎
"""

from .base_alternative_adapter import (
    AlternativeDataAdapter,
    AlternativeDataType,
    SentimentPolarity,
    SentimentDataPoint,
    TrendDataPoint,
    AlternativeDataRequest,
    AdapterMetrics,
    DataFusionEngine,
    AlternativeDataError
)

__all__ = [
    # 基类和数据模型
    'AlternativeDataAdapter',
    'AlternativeDataType',
    'SentimentPolarity',
    'SentimentDataPoint',
    'TrendDataPoint',
    'AlternativeDataRequest',
    'AdapterMetrics',
    'DataFusionEngine',
    'AlternativeDataError',
]
