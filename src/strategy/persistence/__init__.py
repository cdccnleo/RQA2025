#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
策略持久化层
Strategy Persistence Layer

提供策略数据的持久化存储和管理功能。
"""

from .strategy_persistence import StrategyPersistence
StrategyRepository = StrategyPersistence  # 别名兼容性

__all__ = [
    'StrategyPersistence',
    'StrategyRepository'
]
