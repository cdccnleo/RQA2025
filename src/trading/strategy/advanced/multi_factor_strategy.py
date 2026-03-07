#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多因子策略框架

功能：
- 支持多种因子类型（价值、成长、质量、动量等）
- 因子评分和排序
- 多因子组合和加权
- 动态因子调整

作者: AI Assistant
创建日期: 2026-02-21
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class FactorType(Enum):
    """因子类型"""
    VALUE = "value"                 # 价值因子
    GROWTH = "growth"               # 成长因子
    QUALITY = "quality"             # 质量因子
    MOMENTUM = "momentum"           # 动量因子
    VOLATILITY = "volatility"       # 波动率因子
    LIQUIDITY = "liquidity"         # 流动性因子
    SENTIMENT = "sentiment"         # 情绪因子


class FactorDirection(Enum):
    """因子方向"""
    POSITIVE = 1                    # 正向因子（值越大越好）
    NEGATIVE = -1                   # 负向因子（值越小越好）


@dataclass
class Factor:
    """因子定义"""
    name: str
    factor_type: FactorType
    direction: FactorDirection
    weight: float = 1.0
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FactorScore:
    """因子评分结果"""
    symbol: str
    factor_name: str
    raw_value: float
    normalized_score: float           # 0-100
    rank: int
    percentile: float                 # 0-1
    timestamp: datetime


@dataclass
class MultiFactorSignal:
    """多因子信号"""
    symbol: str
    timestamp: datetime
    composite_score: float            # 综合评分
    factor_scores: Dict[str, float]   # 各因子评分
    factor_contributions: Dict[str, float]  # 各因子贡献度
    confidence: float                 # 置信度
    recommendation: str               # 建议（买入/卖出/持有）


class FactorCalculator:
    """因子计算器基类"""
    
    def __init__(self, factor: Factor):
        """
        初始化因子计算器
        
        Args:
            factor: 因子定义
        """
        self.factor = factor
    
    async def calculate(
        self,
        symbol: str,
        data: pd.DataFrame
    ) -> Optional[FactorScore]:
        """
        计算因子值
        
        Args:
            symbol: 股票代码
            data: 股票数据
            
        Returns:
            因子评分
        """
        raise NotImplementedError


class ValueFactorCalculator(FactorCalculator):
    """价值因子计算器"""
    
    async def calculate(
        self,
        symbol: str,
        data: pd.DataFrame
    ) -> Optional[FactorScore]:
        """计算价值因子（PE、PB、PS等）"""
        try:
            # 获取最新数据
            latest = data.iloc[-1]
            
            # 计算价值分数（PE越低越好）
            pe = latest.get('pe_ratio', np.nan)
            pb = latest.get('pb_ratio', np.nan)
            
            if pd.isna(pe) or pd.isna(pb):
                return None
            
            # 价值分数（PE和PB的倒数，标准化到0-100）
            value_score = (1 / max(pe, 0.1) + 1 / max(pb, 0.1)) * 50
            
            return FactorScore(
                symbol=symbol,
                factor_name=self.factor.name,
                raw_value=pe,
                normalized_score=min(value_score, 100),
                rank=0,
                percentile=0.0,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"计算价值因子失败 {symbol}: {e}")
            return None


class MomentumFactorCalculator(FactorCalculator):
    """动量因子计算器"""
    
    async def calculate(
        self,
        symbol: str,
        data: pd.DataFrame
    ) -> Optional[FactorScore]:
        """计算动量因子（收益率、价格趋势等）"""
        try:
            if len(data) < 20:
                return None
            
            # 计算不同周期的收益率
            returns_5d = (data['close'].iloc[-1] / data['close'].iloc[-5] - 1) * 100
            returns_20d = (data['close'].iloc[-1] / data['close'].iloc[-20] - 1) * 100
            
            # 计算趋势强度（使用RSI）
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            latest_rsi = rsi.iloc[-1]
            
            # 动量分数
            momentum_score = (returns_5d + returns_20d * 0.5) * 10 + latest_rsi
            momentum_score = max(0, min(momentum_score, 100))
            
            return FactorScore(
                symbol=symbol,
                factor_name=self.factor.name,
                raw_value=returns_20d,
                normalized_score=momentum_score,
                rank=0,
                percentile=0.0,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"计算动量因子失败 {symbol}: {e}")
            return None


class QualityFactorCalculator(FactorCalculator):
    """质量因子计算器"""
    
    async def calculate(
        self,
        symbol: str,
        data: pd.DataFrame
    ) -> Optional[FactorScore]:
        """计算质量因子（ROE、盈利稳定性等）"""
        try:
            latest = data.iloc[-1]
            
            # 获取财务指标
            roe = latest.get('roe', np.nan)
            debt_ratio = latest.get('debt_ratio', np.nan)
            
            if pd.isna(roe):
                return None
            
            # 质量分数
            quality_score = roe * 10  # ROE越高越好
            
            # 负债率调整（负债率越低越好）
            if not pd.isna(debt_ratio):
                quality_score -= debt_ratio * 20
            
            quality_score = max(0, min(quality_score, 100))
            
            return FactorScore(
                symbol=symbol,
                factor_name=self.factor.name,
                raw_value=roe,
                normalized_score=quality_score,
                rank=0,
                percentile=0.0,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"计算质量因子失败 {symbol}: {e}")
            return None


class VolatilityFactorCalculator(FactorCalculator):
    """波动率因子计算器"""
    
    async def calculate(
        self,
        symbol: str,
        data: pd.DataFrame
    ) -> Optional[FactorScore]:
        """计算波动率因子（历史波动率、Beta等）"""
        try:
            if len(data) < 20:
                return None
            
            # 计算历史波动率
            returns = data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100  # 年化波动率
            
            # 波动率分数（波动率越低越好，所以取反）
            volatility_score = max(0, 100 - volatility * 10)
            
            return FactorScore(
                symbol=symbol,
                factor_name=self.factor.name,
                raw_value=volatility,
                normalized_score=volatility_score,
                rank=0,
                percentile=0.0,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"计算波动率因子失败 {symbol}: {e}")
            return None


class MultiFactorStrategy:
    """
    多因子策略
    
    实现多因子选股策略，支持：
    - 多因子评分
    - 因子权重动态调整
    - 综合信号生成
    """
    
    def __init__(
        self,
        name: str = "MultiFactorStrategy",
        top_n: int = 20
    ):
        """
        初始化多因子策略
        
        Args:
            name: 策略名称
            top_n: 选股数量
        """
        self.name = name
        self.top_n = top_n
        
        # 因子列表
        self.factors: List[Factor] = []
        self.factor_calculators: Dict[str, FactorCalculator] = {}
        
        # 初始化默认因子
        self._init_default_factors()
        
        logger.info(f"多因子策略 {name} 初始化完成")
    
    def _init_default_factors(self):
        """初始化默认因子"""
        default_factors = [
            Factor(
                name="value_factor",
                factor_type=FactorType.VALUE,
                direction=FactorDirection.POSITIVE,
                weight=0.25,
                description="价值因子（PE、PB）"
            ),
            Factor(
                name="momentum_factor",
                factor_type=FactorType.MOMENTUM,
                direction=FactorDirection.POSITIVE,
                weight=0.25,
                description="动量因子（收益率、趋势）"
            ),
            Factor(
                name="quality_factor",
                factor_type=FactorType.QUALITY,
                direction=FactorDirection.POSITIVE,
                weight=0.25,
                description="质量因子（ROE、盈利稳定性）"
            ),
            Factor(
                name="volatility_factor",
                factor_type=FactorType.VOLATILITY,
                direction=FactorDirection.NEGATIVE,
                weight=0.25,
                description="波动率因子（低波动）"
            )
        ]
        
        for factor in default_factors:
            self.add_factor(factor)
    
    def add_factor(self, factor: Factor):
        """
        添加因子
        
        Args:
            factor: 因子定义
        """
        self.factors.append(factor)
        
        # 创建对应的计算器
        if factor.factor_type == FactorType.VALUE:
            self.factor_calculators[factor.name] = ValueFactorCalculator(factor)
        elif factor.factor_type == FactorType.MOMENTUM:
            self.factor_calculators[factor.name] = MomentumFactorCalculator(factor)
        elif factor.factor_type == FactorType.QUALITY:
            self.factor_calculators[factor.name] = QualityFactorCalculator(factor)
        elif factor.factor_type == FactorType.VOLATILITY:
            self.factor_calculators[factor.name] = VolatilityFactorCalculator(factor)
        
        logger.info(f"添加因子: {factor.name} ({factor.factor_type.value})")
    
    async def analyze(
        self,
        symbols: List[str],
        data_dict: Dict[str, pd.DataFrame]
    ) -> List[MultiFactorSignal]:
        """
        分析多只股票
        
        Args:
            symbols: 股票代码列表
            data_dict: 股票数据字典 {symbol: DataFrame}
            
        Returns:
            多因子信号列表
        """
        all_scores: Dict[str, List[FactorScore]] = {}
        
        # 计算所有股票的因子评分
        for symbol in symbols:
            if symbol not in data_dict:
                continue
            
            data = data_dict[symbol]
            symbol_scores = []
            
            for factor in self.factors:
                calculator = self.factor_calculators.get(factor.name)
                if calculator:
                    score = await calculator.calculate(symbol, data)
                    if score:
                        symbol_scores.append(score)
            
            if symbol_scores:
                all_scores[symbol] = symbol_scores
        
        # 标准化评分（排名和百分位）
        self._normalize_scores(all_scores)
        
        # 生成综合信号
        signals = self._generate_signals(all_scores)
        
        # 排序并返回Top N
        signals.sort(key=lambda x: x.composite_score, reverse=True)
        
        return signals[:self.top_n]
    
    def _normalize_scores(
        self,
        all_scores: Dict[str, List[FactorScore]]
    ):
        """
        标准化评分
        
        Args:
            all_scores: 所有股票的因子评分
        """
        # 按因子分组
        factor_groups: Dict[str, List[FactorScore]] = {}
        
        for symbol_scores in all_scores.values():
            for score in symbol_scores:
                if score.factor_name not in factor_groups:
                    factor_groups[score.factor_name] = []
                factor_groups[score.factor_name].append(score)
        
        # 计算排名和百分位
        for factor_name, scores in factor_groups.items():
            # 排序
            scores.sort(key=lambda x: x.normalized_score, reverse=True)
            
            n = len(scores)
            for i, score in enumerate(scores):
                score.rank = i + 1
                score.percentile = 1 - (i / n) if n > 1 else 1.0
    
    def _generate_signals(
        self,
        all_scores: Dict[str, List[FactorScore]]
    ) -> List[MultiFactorSignal]:
        """
        生成综合信号
        
        Args:
            all_scores: 所有股票的因子评分
            
        Returns:
            多因子信号列表
        """
        signals = []
        
        for symbol, scores in all_scores.items():
            if not scores:
                continue
            
            # 计算加权综合评分
            composite_score = 0.0
            factor_scores = {}
            factor_contributions = {}
            total_weight = 0.0
            
            for score in scores:
                factor = next((f for f in self.factors if f.name == score.factor_name), None)
                if factor:
                    weight = factor.weight
                    direction = factor.direction.value
                    
                    # 应用因子方向
                    adjusted_score = score.normalized_score * direction
                    composite_score += adjusted_score * weight
                    total_weight += weight
                    
                    factor_scores[score.factor_name] = score.normalized_score
                    factor_contributions[score.factor_name] = adjusted_score * weight
            
            if total_weight > 0:
                composite_score /= total_weight
            
            # 归一化到0-100
            composite_score = max(0, min(composite_score + 50, 100))
            
            # 确定建议
            if composite_score >= 70:
                recommendation = "强烈买入"
            elif composite_score >= 60:
                recommendation = "买入"
            elif composite_score >= 40:
                recommendation = "持有"
            elif composite_score >= 30:
                recommendation = "卖出"
            else:
                recommendation = "强烈卖出"
            
            # 计算置信度（基于因子覆盖度）
            confidence = len(scores) / len(self.factors)
            
            signal = MultiFactorSignal(
                symbol=symbol,
                timestamp=datetime.now(),
                composite_score=composite_score,
                factor_scores=factor_scores,
                factor_contributions=factor_contributions,
                confidence=confidence,
                recommendation=recommendation
            )
            
            signals.append(signal)
        
        return signals
    
    def adjust_factor_weights(self, performance_data: Dict[str, float]):
        """
        根据表现调整因子权重
        
        Args:
            performance_data: 因子表现数据 {factor_name: return}
        """
        if not performance_data:
            return
        
        # 计算新的权重（基于表现）
        total_return = sum(abs(r) for r in performance_data.values())
        
        if total_return == 0:
            return
        
        for factor in self.factors:
            if factor.name in performance_data:
                # 表现好的因子增加权重
                return_ratio = abs(performance_data[factor.name]) / total_return
                factor.weight = 0.2 + return_ratio * 0.6  # 权重范围0.2-0.8
        
        # 归一化权重
        total_weight = sum(f.weight for f in self.factors)
        for factor in self.factors:
            factor.weight /= total_weight
        
        logger.info(f"因子权重已调整: {[(f.name, f.weight) for f in self.factors]}")
    
    def get_factor_stats(self) -> Dict[str, Any]:
        """
        获取因子统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'total_factors': len(self.factors),
            'factor_types': [f.factor_type.value for f in self.factors],
            'factor_weights': {f.name: f.weight for f in self.factors},
            'top_n': self.top_n
        }


# 全局策略实例
_strategy_instance: Optional[MultiFactorStrategy] = None


def get_multi_factor_strategy(
    name: str = "MultiFactorStrategy",
    top_n: int = 20
) -> MultiFactorStrategy:
    """
    获取多因子策略实例（单例模式）
    
    Args:
        name: 策略名称
        top_n: 选股数量
        
    Returns:
        MultiFactorStrategy实例
    """
    global _strategy_instance
    
    if _strategy_instance is None:
        _strategy_instance = MultiFactorStrategy(name, top_n)
    
    return _strategy_instance
