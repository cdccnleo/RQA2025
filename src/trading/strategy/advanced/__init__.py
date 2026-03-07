#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级策略模块

提供先进的量化策略实现：
- 多因子策略
- 统计套利策略
- 策略组合优化
"""

from .multi_factor_strategy import (
    MultiFactorStrategy,
    Factor,
    FactorType,
    FactorDirection,
    FactorScore,
    MultiFactorSignal,
    get_multi_factor_strategy
)

from .statistical_arbitrage_strategy import (
    StatisticalArbitrageStrategy,
    ArbitrageType,
    PairSignal,
    CointegrationResult,
    get_statistical_arbitrage_strategy
)

__all__ = [
    # 多因子策略
    'MultiFactorStrategy',
    'Factor',
    'FactorType',
    'FactorDirection',
    'FactorScore',
    'MultiFactorSignal',
    'get_multi_factor_strategy',
    
    # 统计套利策略
    'StatisticalArbitrageStrategy',
    'ArbitrageType',
    'PairSignal',
    'CointegrationResult',
    'get_statistical_arbitrage_strategy',
]
