#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征工程模块

提供自动化特征工程功能：
- 时序特征提取
- 技术指标特征
- 交叉特征生成
- 特征选择
"""

from .automated_feature_engineer import (
    AutomatedFeatureEngineer,
    FeatureSet,
    get_feature_engineer
)

__all__ = [
    'AutomatedFeatureEngineer',
    'FeatureSet',
    'get_feature_engineer',
]
