#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计套利策略

功能：
- 配对交易（协整性检验）
- 均值回归策略
- 统计套利信号生成
- 动态阈值调整

作者: AI Assistant
创建日期: 2026-02-21
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import pandas as pd
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class ArbitrageType(Enum):
    """套利类型"""
    PAIR_TRADING = "pair_trading"       # 配对交易
    MEAN_REVERSION = "mean_reversion"   # 均值回归
    STATISTICAL = "statistical"         # 统计套利


@dataclass
class PairSignal:
    """配对交易信号"""
    symbol_pair: Tuple[str, str]        # 股票对
    timestamp: datetime
    spread: float                       # 价差
    zscore: float                       # Z-score
    signal_type: str                    # 'long_spread' / 'short_spread'
    confidence: float                   # 置信度
    entry_threshold: float
    exit_threshold: float
    expected_return: float


@dataclass
class CointegrationResult:
    """协整性检验结果"""
    symbol_pair: Tuple[str, str]
    is_cointegrated: bool
    p_value: float
    test_statistic: float
    critical_values: Dict[str, float]
    hedge_ratio: float
    half_life: float                    # 半衰期（天数）


class StatisticalArbitrageStrategy:
    """
    统计套利策略
    
    实现配对交易和均值回归策略：
    - 协整性检验
    - 价差计算和Z-score
    - 动态阈值调整
    - 信号生成
    """
    
    def __init__(
        self,
        name: str = "StatisticalArbitrage",
        lookback_period: int = 60,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5
    ):
        """
        初始化统计套利策略
        
        Args:
            name: 策略名称
            lookback_period: 回看周期
            entry_zscore: 入场Z-score阈值
            exit_zscore: 出场Z-score阈值
        """
        self.name = name
        self.lookback_period = lookback_period
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        
        # 配对缓存
        self.cointegration_cache: Dict[Tuple[str, str], CointegrationResult] = {}
        self.spread_history: Dict[Tuple[str, str], List[float]] = {}
        
        logger.info(f"统计套利策略 {name} 初始化完成")
    
    async def find_cointegrated_pairs(
        self,
        symbols: List[str],
        data_dict: Dict[str, pd.DataFrame],
        p_value_threshold: float = 0.05
    ) -> List[CointegrationResult]:
        """
        寻找协整的股票对
        
        Args:
            symbols: 股票代码列表
            data_dict: 股票数据字典
            p_value_threshold: p值阈值
            
        Returns:
            协整性检验结果列表
        """
        cointegrated_pairs = []
        
        # 遍历所有可能的配对
        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i+1:]:
                if symbol1 not in data_dict or symbol2 not in data_dict:
                    continue
                
                # 进行协整性检验
                result = await self._test_cointegration(
                    symbol1, symbol2,
                    data_dict[symbol1], data_dict[symbol2]
                )
                
                if result and result.is_cointegrated and result.p_value < p_value_threshold:
                    cointegrated_pairs.append(result)
                    self.cointegration_cache[(symbol1, symbol2)] = result
        
        # 按p值排序
        cointegrated_pairs.sort(key=lambda x: x.p_value)
        
        logger.info(f"找到 {len(cointegrated_pairs)} 对协整股票")
        return cointegrated_pairs
    
    async def _test_cointegration(
        self,
        symbol1: str,
        symbol2: str,
        data1: pd.DataFrame,
        data2: pd.DataFrame
    ) -> Optional[CointegrationResult]:
        """
        检验两只股票的协整性
        
        Args:
            symbol1: 股票1代码
            symbol2: 股票2代码
            data1: 股票1数据
            data2: 股票2数据
            
        Returns:
            协整性检验结果
        """
        try:
            # 对齐数据
            merged = pd.merge(
                data1[['close']].rename(columns={'close': 'close1'}),
                data2[['close']].rename(columns={'close': 'close2'}),
                left_index=True, right_index=True,
                how='inner'
            )
            
            if len(merged) < self.lookback_period:
                return None
            
            # 使用最近的数据
            merged = merged.tail(self.lookback_period)
            
            # Engle-Granger协整性检验
            from statsmodels.tsa.stattools import coint
            
            score, p_value, critical_values = coint(
                merged['close1'],
                merged['close2']
            )
            
            # 计算对冲比率（OLS回归）
            X = merged['close2'].values
            y = merged['close1'].values
            
            # 添加常数项
            X_with_const = np.column_stack([np.ones(len(X)), X])
            beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
            hedge_ratio = beta[1]
            
            # 计算价差
            spread = y - hedge_ratio * X
            
            # 计算半衰期（Ornstein-Uhlenbeck过程）
            spread_lag = np.roll(spread, 1)
            spread_lag[0] = spread_lag[1]
            
            delta_spread = spread - spread_lag
            X_ou = np.column_stack([np.ones(len(spread_lag)), spread_lag])
            
            theta = np.linalg.lstsq(X_ou, delta_spread, rcond=None)[0]
            half_life = -np.log(2) / theta[1] if theta[1] < 0 else np.inf
            
            is_cointegrated = p_value < 0.05 and half_life < 30
            
            return CointegrationResult(
                symbol_pair=(symbol1, symbol2),
                is_cointegrated=is_cointegrated,
                p_value=p_value,
                test_statistic=score,
                critical_values={
                    '1%': critical_values[0],
                    '5%': critical_values[1],
                    '10%': critical_values[2]
                },
                hedge_ratio=hedge_ratio,
                half_life=half_life
            )
            
        except Exception as e:
            logger.error(f"协整性检验失败 {symbol1}-{symbol2}: {e}")
            return None
    
    async def generate_signals(
        self,
        pair: Tuple[str, str],
        data1: pd.DataFrame,
        data2: pd.DataFrame
    ) -> Optional[PairSignal]:
        """
        生成配对交易信号
        
        Args:
            pair: 股票对
            data1: 股票1数据
            data2: 股票2数据
            
        Returns:
            交易信号
        """
        try:
            # 获取协整结果
            coint_result = self.cointegration_cache.get(pair)
            if not coint_result:
                return None
            
            # 对齐数据
            merged = pd.merge(
                data1[['close']].rename(columns={'close': 'close1'}),
                data2[['close']].rename(columns={'close': 'close2'}),
                left_index=True, right_index=True,
                how='inner'
            )
            
            if len(merged) < self.lookback_period:
                return None
            
            # 使用最近的数据
            merged = merged.tail(self.lookback_period)
            
            # 计算价差
            spread = (
                merged['close1'].values -
                coint_result.hedge_ratio * merged['close2'].values
            )
            
            # 计算Z-score
            spread_mean = np.mean(spread)
            spread_std = np.std(spread)
            
            if spread_std == 0:
                return None
            
            current_spread = spread[-1]
            zscore = (current_spread - spread_mean) / spread_std
            
            # 生成信号
            signal_type = None
            confidence = min(abs(zscore) / self.entry_zscore, 1.0)
            
            if zscore > self.entry_zscore:
                # 价差过高，做空价差（卖股票1，买股票2）
                signal_type = "short_spread"
            elif zscore < -self.entry_zscore:
                # 价差过低，做多价差（买股票1，卖股票2）
                signal_type = "long_spread"
            elif abs(zscore) < self.exit_zscore:
                # 价差回归，平仓
                signal_type = "exit"
                confidence = 1.0
            
            if signal_type:
                return PairSignal(
                    symbol_pair=pair,
                    timestamp=datetime.now(),
                    spread=current_spread,
                    zscore=zscore,
                    signal_type=signal_type,
                    confidence=confidence,
                    entry_threshold=self.entry_zscore,
                    exit_threshold=self.exit_zscore,
                    expected_return=abs(zscore) * 0.1  # 预期收益估算
                )
            
            return None
            
        except Exception as e:
            logger.error(f"生成信号失败 {pair}: {e}")
            return None
    
    async def analyze_all_pairs(
        self,
        data_dict: Dict[str, pd.DataFrame]
    ) -> List[PairSignal]:
        """
        分析所有配对
        
        Args:
            data_dict: 股票数据字典
            
        Returns:
            交易信号列表
        """
        signals = []
        
        for pair in self.cointegration_cache.keys():
            symbol1, symbol2 = pair
            
            if symbol1 not in data_dict or symbol2 not in data_dict:
                continue
            
            signal = await self.generate_signals(
                pair,
                data_dict[symbol1],
                data_dict[symbol2]
            )
            
            if signal:
                signals.append(signal)
        
        # 按置信度排序
        signals.sort(key=lambda x: x.confidence, reverse=True)
        
        return signals
    
    def adjust_thresholds(
        self,
        volatility: float,
        market_regime: str = "normal"
    ):
        """
        根据市场状态调整阈值
        
        Args:
            volatility: 市场波动率
            market_regime: 市场状态（normal/high_volatility/low_volatility）
        """
        if market_regime == "high_volatility":
            # 高波动时增加阈值
            self.entry_zscore = 2.5
            self.exit_zscore = 0.8
        elif market_regime == "low_volatility":
            # 低波动时降低阈值
            self.entry_zscore = 1.5
            self.exit_zscore = 0.3
        else:
            # 正常状态
            self.entry_zscore = 2.0
            self.exit_zscore = 0.5
        
        logger.info(f"阈值已调整: entry={self.entry_zscore}, exit={self.exit_zscore}")
    
    def get_strategy_stats(self) -> Dict[str, Any]:
        """
        获取策略统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'name': self.name,
            'lookback_period': self.lookback_period,
            'entry_zscore': self.entry_zscore,
            'exit_zscore': self.exit_zscore,
            'cointegrated_pairs': len(self.cointegration_cache),
            'pairs': list(self.cointegration_cache.keys())
        }


# 全局策略实例
_strategy_instance: Optional[StatisticalArbitrageStrategy] = None


def get_statistical_arbitrage_strategy(
    name: str = "StatisticalArbitrage",
    lookback_period: int = 60
) -> StatisticalArbitrageStrategy:
    """
    获取统计套利策略实例（单例模式）
    
    Args:
        name: 策略名称
        lookback_period: 回看周期
        
    Returns:
        StatisticalArbitrageStrategy实例
    """
    global _strategy_instance
    
    if _strategy_instance is None:
        _strategy_instance = StatisticalArbitrageStrategy(name, lookback_period)
    
    return _strategy_instance
