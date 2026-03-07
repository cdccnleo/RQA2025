#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信号验证引擎

功能：
- 历史回测验证
- 质量评分算法
- 风险评分算法
- 综合评分计算

作者: AI Assistant
创建日期: 2026-02-21
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SignalValidationResult:
    """信号验证结果"""
    signal_id: str
    overall_score: float  # 0-100
    quality_score: float  # 0-100
    risk_score: float     # 0-100 (越低越好)
    backtest_score: float # 0-100
    is_valid: bool
    validation_time: datetime
    details: Dict[str, Any]


@dataclass
class BacktestResult:
    """回测结果"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_return: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float


class QualityScorer:
    """质量评分器"""
    
    def __init__(self):
        """初始化质量评分器"""
        self.weights = {
            'signal_strength': 0.3,
            'market_condition': 0.25,
            'volume_confirmation': 0.2,
            'trend_alignment': 0.25
        }
    
    def calculate_score(
        self,
        signal: Dict[str, Any],
        market_data: pd.DataFrame
    ) -> Tuple[float, Dict[str, float]]:
        """
        计算质量评分
        
        Args:
            signal: 信号数据
            market_data: 市场数据
            
        Returns:
            (总分, 分项得分)
        """
        scores = {}
        
        # 1. 信号强度评分
        scores['signal_strength'] = self._score_signal_strength(signal)
        
        # 2. 市场条件评分
        scores['market_condition'] = self._score_market_condition(market_data)
        
        # 3. 成交量确认评分
        scores['volume_confirmation'] = self._score_volume_confirmation(signal, market_data)
        
        # 4. 趋势一致性评分
        scores['trend_alignment'] = self._score_trend_alignment(signal, market_data)
        
        # 计算加权总分
        total_score = sum(
            scores[key] * self.weights[key]
            for key in scores
        )
        
        return min(total_score * 100, 100), scores
    
    def _score_signal_strength(self, signal: Dict[str, Any]) -> float:
        """评分信号强度"""
        confidence = signal.get('confidence', 0.5)
        return confidence
    
    def _score_market_condition(self, market_data: pd.DataFrame) -> float:
        """评分市场条件"""
        if market_data.empty or len(market_data) < 20:
            return 0.5
        
        # 计算波动率
        returns = market_data['close'].pct_change().dropna()
        volatility = returns.std()
        
        # 波动率适中得分最高
        if 0.01 <= volatility <= 0.03:
            return 0.8
        elif volatility < 0.01:
            return 0.6
        else:
            return 0.4
    
    def _score_volume_confirmation(
        self,
        signal: Dict[str, Any],
        market_data: pd.DataFrame
    ) -> float:
        """评分成交量确认"""
        if market_data.empty or 'volume' not in market_data.columns:
            return 0.5
        
        # 获取最近成交量
        recent_volume = market_data['volume'].iloc[-1]
        avg_volume = market_data['volume'].rolling(window=20).mean().iloc[-1]
        
        if avg_volume == 0:
            return 0.5
        
        # 成交量比率
        volume_ratio = recent_volume / avg_volume
        
        if volume_ratio >= 1.5:
            return 0.9
        elif volume_ratio >= 1.0:
            return 0.7
        else:
            return 0.5
    
    def _score_trend_alignment(
        self,
        signal: Dict[str, Any],
        market_data: pd.DataFrame
    ) -> float:
        """评分趋势一致性"""
        if market_data.empty or len(market_data) < 20:
            return 0.5
        
        # 计算趋势
        close = market_data['close']
        sma20 = close.rolling(window=20).mean().iloc[-1]
        current_price = close.iloc[-1]
        
        signal_type = signal.get('signal_type', 'hold')
        
        if signal_type == 'buy':
            # 买入信号：价格应该在均线之上
            return 0.8 if current_price > sma20 else 0.4
        elif signal_type == 'sell':
            # 卖出信号：价格应该在均线之下
            return 0.8 if current_price < sma20 else 0.4
        else:
            return 0.5


class RiskScorer:
    """风险评分器"""
    
    def __init__(self):
        """初始化风险评分器"""
        self.weights = {
            'volatility_risk': 0.3,
            'drawdown_risk': 0.3,
            'concentration_risk': 0.2,
            'liquidity_risk': 0.2
        }
    
    def calculate_score(
        self,
        signal: Dict[str, Any],
        market_data: pd.DataFrame,
        portfolio: Optional[Dict] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        计算风险评分（越低越好）
        
        Args:
            signal: 信号数据
            market_data: 市场数据
            portfolio: 投资组合信息
            
        Returns:
            (总分, 分项得分)
        """
        scores = {}
        
        # 1. 波动率风险
        scores['volatility_risk'] = self._score_volatility_risk(market_data)
        
        # 2. 回撤风险
        scores['drawdown_risk'] = self._score_drawdown_risk(market_data)
        
        # 3. 集中度风险
        scores['concentration_risk'] = self._score_concentration_risk(signal, portfolio)
        
        # 4. 流动性风险
        scores['liquidity_risk'] = self._score_liquidity_risk(market_data)
        
        # 计算加权总分
        total_score = sum(
            scores[key] * self.weights[key]
            for key in scores
        )
        
        return min(total_score * 100, 100), scores
    
    def _score_volatility_risk(self, market_data: pd.DataFrame) -> float:
        """评分波动率风险"""
        if market_data.empty or len(market_data) < 20:
            return 0.5
        
        returns = market_data['close'].pct_change().dropna()
        volatility = returns.std()
        
        # 波动率越高，风险越高
        return min(volatility * 10, 1.0)
    
    def _score_drawdown_risk(self, market_data: pd.DataFrame) -> float:
        """评分回撤风险"""
        if market_data.empty or len(market_data) < 20:
            return 0.5
        
        # 计算最大回撤
        close = market_data['close']
        rolling_max = close.expanding().max()
        drawdown = (close - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())
        
        return min(max_drawdown * 2, 1.0)
    
    def _score_concentration_risk(
        self,
        signal: Dict[str, Any],
        portfolio: Optional[Dict]
    ) -> float:
        """评分集中度风险"""
        if not portfolio:
            return 0.5
        
        # 检查该股票在投资组合中的权重
        symbol = signal.get('symbol', '')
        positions = portfolio.get('positions', {})
        
        if symbol not in positions:
            return 0.3
        
        position_weight = positions[symbol].get('weight', 0)
        
        # 权重越高，风险越高
        if position_weight > 0.3:
            return 0.9
        elif position_weight > 0.2:
            return 0.7
        else:
            return 0.5
    
    def _score_liquidity_risk(self, market_data: pd.DataFrame) -> float:
        """评分流动性风险"""
        if market_data.empty or 'volume' not in market_data.columns:
            return 0.5
        
        avg_volume = market_data['volume'].mean()
        
        # 成交量越低，流动性风险越高
        if avg_volume < 1000000:
            return 0.8
        elif avg_volume < 5000000:
            return 0.5
        else:
            return 0.3


class BacktestEngine:
    """回测引擎"""
    
    def run_backtest(
        self,
        signal: Dict[str, Any],
        historical_data: pd.DataFrame,
        lookforward_days: int = 5
    ) -> BacktestResult:
        """
        运行回测
        
        Args:
            signal: 信号数据
            historical_data: 历史数据
            lookforward_days: 前瞻天数
            
        Returns:
            回测结果
        """
        if historical_data.empty:
            return BacktestResult(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                avg_return=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                profit_factor=0.0
            )
        
        # 简化的回测逻辑
        returns = historical_data['close'].pct_change().dropna()
        
        if len(returns) < lookforward_days:
            return BacktestResult(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                avg_return=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                profit_factor=0.0
            )
        
        # 计算未来收益
        future_returns = returns.iloc[-lookforward_days:]
        
        winning_trades = sum(1 for r in future_returns if r > 0)
        losing_trades = sum(1 for r in future_returns if r <= 0)
        total_trades = winning_trades + losing_trades
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        avg_return = future_returns.mean()
        
        # 计算最大回撤
        cumulative = (1 + future_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())
        
        # 计算夏普比率（简化版）
        sharpe_ratio = avg_return / future_returns.std() if future_returns.std() > 0 else 0.0
        
        # 计算盈亏比
        avg_win = future_returns[future_returns > 0].mean() if winning_trades > 0 else 0
        avg_loss = abs(future_returns[future_returns < 0].mean()) if losing_trades > 0 else 1
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 0.0
        
        return BacktestResult(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_return=avg_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            profit_factor=profit_factor
        )
    
    def calculate_backtest_score(self, result: BacktestResult) -> float:
        """
        计算回测评分
        
        Args:
            result: 回测结果
            
        Returns:
            评分 (0-100)
        """
        if result.total_trades == 0:
            return 50.0
        
        # 综合评分
        score = 0.0
        
        # 胜率权重 30%
        score += result.win_rate * 30
        
        # 夏普比率权重 30%
        sharpe_score = min(max(result.sharpe_ratio, 0), 3) / 3 * 30
        score += sharpe_score
        
        # 最大回撤权重 20%（越低越好）
        drawdown_score = (1 - min(result.max_drawdown, 1)) * 20
        score += drawdown_score
        
        # 盈亏比权重 20%
        pf_score = min(result.profit_factor, 3) / 3 * 20
        score += pf_score
        
        return min(score, 100)


class SignalValidationEngine:
    """
    信号验证引擎
    
    职责：
    1. 历史回测验证
    2. 质量评分算法
    3. 风险评分算法
    4. 综合评分计算
    """
    
    def __init__(
        self,
        quality_threshold: float = 60.0,
        risk_threshold: float = 70.0,
        backtest_threshold: float = 50.0
    ):
        """
        初始化信号验证引擎
        
        Args:
            quality_threshold: 质量评分阈值
            risk_threshold: 风险评分阈值
            backtest_threshold: 回测评分阈值
        """
        self.quality_scorer = QualityScorer()
        self.risk_scorer = RiskScorer()
        self.backtest_engine = BacktestEngine()
        
        self.quality_threshold = quality_threshold
        self.risk_threshold = risk_threshold
        self.backtest_threshold = backtest_threshold
        
        logger.info("信号验证引擎初始化完成")
    
    def validate_signal(
        self,
        signal: Dict[str, Any],
        market_data: pd.DataFrame,
        portfolio: Optional[Dict] = None
    ) -> SignalValidationResult:
        """
        验证信号
        
        Args:
            signal: 信号数据
            market_data: 市场数据
            portfolio: 投资组合信息
            
        Returns:
            验证结果
        """
        signal_id = signal.get('symbol', 'unknown') + '_' + datetime.now().strftime('%Y%m%d%H%M%S')
        
        # 1. 计算质量评分
        quality_score, quality_details = self.quality_scorer.calculate_score(
            signal, market_data
        )
        
        # 2. 计算风险评分
        risk_score, risk_details = self.risk_scorer.calculate_score(
            signal, market_data, portfolio
        )
        
        # 3. 运行回测
        backtest_result = self.backtest_engine.run_backtest(
            signal, market_data
        )
        backtest_score = self.backtest_engine.calculate_backtest_score(backtest_result)
        
        # 4. 计算综合评分
        # 质量40% + 回测35% + (100-风险)25%
        overall_score = (
            quality_score * 0.4 +
            backtest_score * 0.35 +
            (100 - risk_score) * 0.25
        )
        
        # 5. 判断是否有效
        is_valid = (
            quality_score >= self.quality_threshold and
            risk_score <= self.risk_threshold and
            backtest_score >= self.backtest_threshold
        )
        
        return SignalValidationResult(
            signal_id=signal_id,
            overall_score=overall_score,
            quality_score=quality_score,
            risk_score=risk_score,
            backtest_score=backtest_score,
            is_valid=is_valid,
            validation_time=datetime.now(),
            details={
                'quality_details': quality_details,
                'risk_details': risk_details,
                'backtest_result': {
                    'total_trades': backtest_result.total_trades,
                    'win_rate': backtest_result.win_rate,
                    'avg_return': backtest_result.avg_return,
                    'max_drawdown': backtest_result.max_drawdown,
                    'sharpe_ratio': backtest_result.sharpe_ratio,
                    'profit_factor': backtest_result.profit_factor
                }
            }
        )


# 单例实例
_engine: Optional[SignalValidationEngine] = None


def get_signal_validation_engine(
    quality_threshold: float = 60.0,
    risk_threshold: float = 70.0,
    backtest_threshold: float = 50.0
) -> SignalValidationEngine:
    """
    获取信号验证引擎单例
    
    Args:
        quality_threshold: 质量评分阈值
        risk_threshold: 风险评分阈值
        backtest_threshold: 回测评分阈值
        
    Returns:
        SignalValidationEngine实例
    """
    global _engine
    if _engine is None:
        _engine = SignalValidationEngine(
            quality_threshold=quality_threshold,
            risk_threshold=risk_threshold,
            backtest_threshold=backtest_threshold
        )
    return _engine
