#!/usr/bin/env python3
"""
RQA2025 智能交易决策支持系统

基于AI / ML的智能决策支持系统，提供全面的交易决策辅助功能
包括市场分析、风险评估、策略建议、可视化界面等
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from collections import deque
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc

from ..monitoring.deep_learning_predictor import get_deep_learning_predictor
from ..strategy.intelligence.multi_strategy_optimizer import get_strategy_engine
from ..core.business_process_optimizer import get_business_process_optimizer
from ..monitoring.performance_analyzer import get_performance_analyzer
from ..core.ai_performance_optimizer import get_performance_optimizer

logger = logging.getLogger(__name__)


class DecisionType(Enum):

    """决策类型"""
    TRADE_SIGNAL = "trade_signal"          # 交易信号
    PORTFOLIO_ADJUSTMENT = "portfolio_adjustment"  # 组合调整
    RISK_MANAGEMENT = "risk_management"    # 风险管理
    MARKET_TIMING = "market_timing"        # 市场时机
    STRATEGY_SELECTION = "strategy_selection"  # 策略选择


class ConfidenceLevel(Enum):

    """置信度级别"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class DecisionRecommendation:

    """决策建议"""
    decision_id: str
    decision_type: DecisionType
    title: str
    description: str
    confidence: ConfidenceLevel
    confidence_score: float
    recommended_action: str
    expected_outcome: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    expiry_time: Optional[datetime] = None


@dataclass
class MarketAnalysis:

    """市场分析"""
    analysis_id: str
    symbol: str
    timeframe: str
    technical_indicators: Dict[str, Any]
    sentiment_analysis: Dict[str, Any]
    fundamental_data: Dict[str, Any]
    ai_predictions: Dict[str, Any]
    market_regime: str
    volatility_level: str
    trend_direction: str
    support_resistance_levels: List[float]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RiskProfile:

    """风险画像"""
    risk_tolerance: str
    max_drawdown_limit: float
    volatility_threshold: float
    concentration_limit: float
    liquidity_requirements: Dict[str, Any]
    stress_test_results: Dict[str, Any]
    last_updated: datetime = field(default_factory=datetime.now)


class IntelligentDecisionEngine:

    """
    智能决策引擎

    整合多种AI分析能力，提供智能交易决策支持
    """

    def __init__(self):

        self.dl_predictor = get_deep_learning_predictor()
        self.strategy_engine = get_strategy_engine()
        self.business_optimizer = get_business_process_optimizer()
        self.performance_analyzer = get_performance_analyzer()
        self.performance_optimizer = get_performance_optimizer()

        self.decision_history = deque(maxlen=1000)
        self.market_analyses = {}
        self.risk_profiles = {}

        # 决策规则和阈值
        self.decision_rules = self._initialize_decision_rules()

        logger.info("智能决策引擎初始化完成")

    async def analyze_market_and_generate_decisions(self, market_data: Dict[str, Any],
                                                    portfolio_data: Dict[str, Any],
                                                    risk_profile: Dict[str, Any]) -> List[DecisionRecommendation]:
        """
        分析市场并生成决策建议

        Args:
            market_data: 市场数据
            portfolio_data: 组合数据
            risk_profile: 风险画像

        Returns:
            决策建议列表
        """
        try:
            recommendations = []

            # 1. 执行市场分析
            market_analysis = await self._perform_comprehensive_market_analysis(market_data)

            # 2. 评估投资组合
            portfolio_assessment = await self._assess_portfolio_status(portfolio_data, market_analysis)

            # 3. 执行风险评估
            risk_assessment = await self._perform_risk_assessment(portfolio_data, market_analysis, risk_profile)

            # 4. 生成交易信号决策
            trade_signals = await self._generate_trade_signal_decisions(market_analysis, portfolio_assessment)
            recommendations.extend(trade_signals)

            # 5. 生成组合调整决策
            portfolio_decisions = await self._generate_portfolio_adjustment_decisions(
                portfolio_assessment, risk_assessment
            )
            recommendations.extend(portfolio_decisions)

            # 6. 生成风险管理决策
            risk_decisions = await self._generate_risk_management_decisions(risk_assessment)
            recommendations.extend(risk_decisions)

            # 7. 生成市场时机决策
            timing_decisions = await self._generate_market_timing_decisions(market_analysis)
            recommendations.extend(timing_decisions)

            # 8. 应用决策规则过滤
            filtered_recommendations = await self._apply_decision_rules(recommendations, market_analysis)

            # 记录决策历史
            for rec in filtered_recommendations:
                self.decision_history.append(rec)

            return filtered_recommendations

        except Exception as e:
            logger.error(f"智能决策生成失败: {e}")
            return []

    async def _perform_comprehensive_market_analysis(self, market_data: Dict[str, Any]) -> MarketAnalysis:
        """执行综合市场分析"""
        try:
            symbol = market_data.get('symbol', 'UNKNOWN')
            analysis_id = f"analysis_{symbol}_{int(time.time())}"

            # 技术指标分析
            technical_indicators = await self._calculate_technical_indicators(market_data)

            # 情感分析
            sentiment_analysis = await self._perform_sentiment_analysis(market_data)

            # 基本面分析
            fundamental_data = await self._gather_fundamental_data(symbol)

            # AI预测
            ai_predictions = await self._generate_ai_predictions(market_data)

            # 市场状态识别
            market_regime = await self._identify_market_regime(market_data)
            volatility_level = await self._assess_volatility_level(market_data)
            trend_direction = await self._determine_trend_direction(market_data)

            # 支撑阻力位
            support_resistance = await self._calculate_support_resistance_levels(market_data)

            analysis = MarketAnalysis(
                analysis_id=analysis_id,
                symbol=symbol,
                timeframe=market_data.get('timeframe', '1D'),
                technical_indicators=technical_indicators,
                sentiment_analysis=sentiment_analysis,
                fundamental_data=fundamental_data,
                ai_predictions=ai_predictions,
                market_regime=market_regime,
                volatility_level=volatility_level,
                trend_direction=trend_direction,
                support_resistance_levels=support_resistance
            )

            # 缓存分析结果
            self.market_analyses[analysis_id] = analysis

            return analysis

        except Exception as e:
            logger.error(f"综合市场分析失败: {e}")
            return MarketAnalysis(
                analysis_id=f"error_{int(time.time())}",
                symbol=market_data.get('symbol', 'UNKNOWN'),
                timeframe='1D',
                technical_indicators={},
                sentiment_analysis={},
                fundamental_data={},
                ai_predictions={},
                market_regime='unknown',
                volatility_level='unknown',
                trend_direction='unknown',
                support_resistance_levels=[]
            )

    async def _calculate_technical_indicators(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """计算技术指标"""
        try:
            prices = market_data.get('prices', [])
            if not prices:
                return {}

            # 移动平均线
            sma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else np.mean(prices)
            sma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else np.mean(prices)

            # RSI
            rsi = await self._calculate_rsi(prices)

            # MACD
            macd, signal, hist = await self._calculate_macd(prices)

            # 布林带
            upper_band, middle_band, lower_band = await self._calculate_bollinger_bands(prices)

            return {
                'sma_20': sma_20,
                'sma_50': sma_50,
                'rsi': rsi,
                'macd': {'macd': macd, 'signal': signal, 'histogram': hist},
                'bollinger_bands': {
                    'upper': upper_band,
                    'middle': middle_band,
                    'lower': lower_band
                },
                'current_price': prices[-1] if prices else 0
            }

        except Exception as e:
            logger.warning(f"技术指标计算失败: {e}")
            return {}

    async def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """计算RSI指标"""
        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    async def _calculate_macd(self, prices: List[float]) -> Tuple[float, float, float]:
        """计算MACD指标"""
        if len(prices) < 26:
            return 0.0, 0.0, 0.0

        # 计算EMA
        ema_12 = await self._calculate_ema(prices, 12)
        ema_26 = await self._calculate_ema(prices, 26)

        macd = ema_12 - ema_26
        signal = await self._calculate_ema([macd], 9) if len([macd]) >= 9 else macd
        histogram = macd - signal

        return macd, signal, histogram

    async def _calculate_ema(self, prices: List[float], period: int) -> float:
        """计算指数移动平均"""
        if len(prices) < period:
            return np.mean(prices)

        multiplier = 2 / (period + 1)
        ema = prices[0]

        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))

        return ema

    async def _calculate_bollinger_bands(self, prices: List[float],
                                         period: int = 20, std_dev: float = 2.0) -> Tuple[float, float, float]:
        """计算布林带"""
        if len(prices) < period:
            mean_price = np.mean(prices)
            return mean_price, mean_price, mean_price

        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])

        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)

        return upper_band, sma, lower_band

    async def _perform_sentiment_analysis(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行情感分析"""
        try:
            # 这里可以集成新闻情感分析、社交媒体情感分析等
            # 暂时返回模拟数据
            return {
                'overall_sentiment': 'neutral',
                'sentiment_score': 0.0,
                'news_sentiment': {'positive': 45, 'negative': 35, 'neutral': 20},
                'social_sentiment': {'bullish': 52, 'bearish': 48},
                'confidence': 0.7
            }
        except Exception as e:
            logger.warning(f"情感分析失败: {e}")
            return {'error': str(e)}

    async def _gather_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """收集基本面数据"""
        try:
            # 这里可以集成财务数据API
            # 暂时返回模拟数据
            return {
                'pe_ratio': 18.5,
                'pb_ratio': 3.2,
                'roe': 0.15,
                'debt_to_equity': 0.8,
                'revenue_growth': 0.12,
                'earnings_growth': 0.08,
                'dividend_yield': 0.025
            }
        except Exception as e:
            logger.warning(f"基本面数据收集失败: {e}")
            return {}

    async def _generate_ai_predictions(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成AI预测"""
        try:
            # 使用深度学习预测器进行预测
            predictions = {}

            # 价格预测
            price_data = market_data.get('prices', [])
            if price_data:
                df_price = pd.DataFrame({
                    'timestamp': pd.date_range(end=datetime.now(), periods=len(price_data), freq='H'),
                    'value': price_data
                }).set_index('timestamp')

                price_prediction = self.dl_predictor.get_optimized_prediction(
                    'price_prediction', df_price, steps=5
                )

                if price_prediction.get('status') == 'success':
                    predictions['price'] = price_prediction

            # 波动率预测
            if len(price_data) > 1:
                returns = np.diff(price_data) / price_data[:-1]
                df_volatility = pd.DataFrame({
                    'timestamp': pd.date_range(end=datetime.now(), periods=len(returns), freq='H'),
                    'value': returns
                }).set_index('timestamp')

                volatility_prediction = self.dl_predictor.get_optimized_prediction(
                    'volatility_prediction', df_volatility, steps=3
                )

                if volatility_prediction.get('status') == 'success':
                    predictions['volatility'] = volatility_prediction

            return predictions

        except Exception as e:
            logger.warning(f"AI预测生成失败: {e}")
            return {}

    async def _identify_market_regime(self, market_data: Dict[str, Any]) -> str:
        """识别市场状态"""
        try:
            prices = market_data.get('prices', [])
            if len(prices) < 20:
                return 'unknown'

            # 基于波动率和趋势识别市场状态
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns)

            # 计算趋势强度
            sma_short = np.mean(prices[-10:])
            sma_long = np.mean(prices[-30:])
            trend_strength = (sma_short - sma_long) / sma_long

            if volatility > 0.05:  # 高波动
                if trend_strength > 0.05:
                    return 'bullish_volatile'
                elif trend_strength < -0.05:
                    return 'bearish_volatile'
                else:
                    return 'sideways_volatile'
            else:  # 低波动
                if trend_strength > 0.02:
                    return 'bullish_trending'
                elif trend_strength < -0.02:
                    return 'bearish_trending'
                else:
                    return 'range_bound'

        except Exception:
            return 'unknown'

    async def _assess_volatility_level(self, market_data: Dict[str, Any]) -> str:
        """评估波动率水平"""
        try:
            prices = market_data.get('prices', [])
            if len(prices) < 10:
                return 'unknown'

            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns)

            if volatility > 0.08:
                return 'very_high'
            elif volatility > 0.05:
                return 'high'
            elif volatility > 0.02:
                return 'moderate'
            elif volatility > 0.01:
                return 'low'
            else:
                return 'very_low'

        except Exception:
            return 'unknown'

    async def _determine_trend_direction(self, market_data: Dict[str, Any]) -> str:
        """确定趋势方向"""
        try:
            prices = market_data.get('prices', [])
            if len(prices) < 20:
                return 'unknown'

            # 使用简单移动平均判断趋势
            sma_short = np.mean(prices[-10:])
            sma_long = np.mean(prices[-30:])

            if sma_short > sma_long * 1.02:
                return 'strong_uptrend'
            elif sma_short > sma_long * 1.005:
                return 'uptrend'
            elif sma_short < sma_long * 0.98:
                return 'strong_downtrend'
            elif sma_short < sma_long * 0.995:
                return 'downtrend'
            else:
                return 'sideways'

        except Exception:
            return 'unknown'

    async def _calculate_support_resistance_levels(self, market_data: Dict[str, Any]) -> List[float]:
        """计算支撑阻力位"""
        try:
            prices = market_data.get('prices', [])
            if len(prices) < 50:
                return []

            # 使用价格聚类方法识别支撑阻力位
            # 这里使用简单的峰谷检测
            support_levels = []
            resistance_levels = []

            window = 10
            for i in range(window, len(prices) - window):
                # 检测局部最低点（支撑）
                if all(prices[i] <= prices[j] for j in range(i - window, i + window + 1) if j != i):
                    support_levels.append(prices[i])

                # 检测局部最高点（阻力）
                if all(prices[i] >= prices[j] for j in range(i - window, i + window + 1) if j != i):
                    resistance_levels.append(prices[i])

            # 返回最近的支撑阻力位
            levels = []
            if support_levels:
                levels.append(min(support_levels[-3:]))  # 最近支撑位
            if resistance_levels:
                levels.append(max(resistance_levels[-3:]))  # 最近阻力位

            return sorted(levels)

        except Exception as e:
            logger.warning(f"支撑阻力位计算失败: {e}")
            return []

    async def _assess_portfolio_status(self, portfolio_data: Dict[str, Any],
                                       market_analysis: MarketAnalysis) -> Dict[str, Any]:
        """评估投资组合状态"""
        try:
            # 这里可以实现详细的组合评估逻辑
            return {
                'total_value': portfolio_data.get('total_value', 0),
                'positions': portfolio_data.get('positions', {}),
                'diversification_score': 0.7,  # 多样化评分
                'risk_exposure': 0.6,  # 风险暴露
                'performance_score': 0.8,  # 表现评分
                'rebalancing_needed': True
            }
        except Exception as e:
            logger.warning(f"组合评估失败: {e}")
            return {}

    async def _perform_risk_assessment(self, portfolio_data: Dict[str, Any],
                                       market_analysis: MarketAnalysis,
                                       risk_profile: Dict[str, Any]) -> Dict[str, Any]:
        """执行风险评估"""
        try:
            # 这里可以实现详细的风险评估逻辑
            return {
                'overall_risk_score': 0.4,
                'volatility_risk': 0.3,
                'liquidity_risk': 0.2,
                'concentration_risk': 0.5,
                'market_risk': 0.6,
                'risk_limits_breached': [],
                'stress_test_results': {}
            }
        except Exception as e:
            logger.warning(f"风险评估失败: {e}")
            return {}

    async def _generate_trade_signal_decisions(self, market_analysis: MarketAnalysis,
                                               portfolio_assessment: Dict[str, Any]) -> List[DecisionRecommendation]:
        """生成交易信号决策"""
        recommendations = []

        try:
            # 基于技术指标生成信号
            technical_signals = await self._analyze_technical_signals(market_analysis)

            for signal in technical_signals:
                confidence_score = signal.get('confidence', 0.5)
                confidence_level = self._map_confidence_score(confidence_score)

                recommendation = DecisionRecommendation(
                    decision_id=f"signal_{int(time.time())}_{len(recommendations)}",
                    decision_type=DecisionType.TRADE_SIGNAL,
                    title=f"交易信号: {signal['action']} {market_analysis.symbol}",
                    description=signal['reasoning'],
                    confidence=confidence_level,
                    confidence_score=confidence_score,
                    recommended_action=signal['action'],
                    expected_outcome={
                        'target_price': signal.get('target_price'),
                        'stop_loss': signal.get('stop_loss'),
                        'potential_return': signal.get('potential_return', 0)
                    },
                    risk_assessment={
                        'risk_level': signal.get('risk_level', 'medium'),
                        'max_loss': signal.get('max_loss', 0)
                    },
                    supporting_data={
                        'technical_indicators': market_analysis.technical_indicators,
                        'market_regime': market_analysis.market_regime
                    }
                )
                recommendations.append(recommendation)

        except Exception as e:
            logger.warning(f"交易信号决策生成失败: {e}")

        return recommendations

    async def _analyze_technical_signals(self, market_analysis: MarketAnalysis) -> List[Dict[str, Any]]:
        """分析技术信号"""
        signals = []

        try:
            indicators = market_analysis.technical_indicators

            # RSI信号
            rsi = indicators.get('rsi', 50)
            if rsi < 30:
                signals.append({
                    'action': 'BUY',
                    'confidence': 0.7,
                    'reasoning': f"RSI超卖 ({rsi:.1f})，可能反弹",
                    'target_price': indicators.get('current_price', 0) * 1.05,
                    'stop_loss': indicators.get('current_price', 0) * 0.95,
                    'potential_return': 0.05,
                    'risk_level': 'medium',
                    'max_loss': 0.05
                })
            elif rsi > 70:
                signals.append({
                    'action': 'SELL',
                    'confidence': 0.7,
                    'reasoning': f"RSI超买 ({rsi:.1f})，可能回调",
                    'target_price': indicators.get('current_price', 0) * 0.95,
                    'stop_loss': indicators.get('current_price', 0) * 1.05,
                    'potential_return': 0.05,
                    'risk_level': 'medium',
                    'max_loss': 0.05
                })

            # 移动平均信号
            sma_20 = indicators.get('sma_20', 0)
            sma_50 = indicators.get('sma_50', 0)
            current_price = indicators.get('current_price', 0)

            if current_price > sma_20 > sma_50:
                signals.append({
                    'action': 'BUY',
                    'confidence': 0.6,
                    'reasoning': "价格在短期均线上方，长期均线支撑",
                    'target_price': current_price * 1.03,
                    'stop_loss': sma_50,
                    'potential_return': 0.03,
                    'risk_level': 'low',
                    'max_loss': (current_price - sma_50) / current_price
                })

        except Exception as e:
            logger.warning(f"技术信号分析失败: {e}")

        return signals

    def _map_confidence_score(self, score: float) -> ConfidenceLevel:
        """映射置信度分数到级别"""
        if score >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif score >= 0.7:
            return ConfidenceLevel.HIGH
        elif score >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif score >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    async def _generate_portfolio_adjustment_decisions(self, portfolio_assessment: Dict[str, Any],
                                                       risk_assessment: Dict[str, Any]) -> List[DecisionRecommendation]:
        """生成组合调整决策"""
        recommendations = []

        try:
            # 检查是否需要重新平衡
            if portfolio_assessment.get('rebalancing_needed', False):
                recommendation = DecisionRecommendation(
                    decision_id=f"rebalance_{int(time.time())}",
                    decision_type=DecisionType.PORTFOLIO_ADJUSTMENT,
                    title="投资组合重新平衡",
                    description="组合权重偏离目标配置，建议重新平衡",
                    confidence=ConfidenceLevel.MEDIUM,
                    confidence_score=0.6,
                    recommended_action="REBALANCE",
                    expected_outcome={
                        'improved_diversification': True,
                        'risk_reduction': 0.1
                    },
                    risk_assessment={
                        'transaction_costs': 0.02,
                        'market_impact': 'low'
                    },
                    supporting_data={
                        'current_allocation': portfolio_assessment.get('positions', {}),
                        'target_allocation': {},  # 这里应该有目标配置
                        'diversification_score': portfolio_assessment.get('diversification_score', 0)
                    }
                )
                recommendations.append(recommendation)

        except Exception as e:
            logger.warning(f"组合调整决策生成失败: {e}")

        return recommendations

    async def _generate_risk_management_decisions(self, risk_assessment: Dict[str, Any]) -> List[DecisionRecommendation]:
        """生成风险管理决策"""
        recommendations = []

        try:
            # 检查风险限额是否突破
            breached_limits = risk_assessment.get('risk_limits_breached', [])

            for limit in breached_limits:
                recommendation = DecisionRecommendation(
                    decision_id=f"risk_{limit}_{int(time.time())}",
                    decision_type=DecisionType.RISK_MANAGEMENT,
                    title=f"风险限额突破: {limit}",
                    description=f"当前{limit}已超过设定限额",
                    confidence=ConfidenceLevel.HIGH,
                    confidence_score=0.8,
                    recommended_action="REDUCE_EXPOSURE",
                    expected_outcome={
                        'risk_reduction': 0.2,
                        'portfolio_stability': 'improved'
                    },
                    risk_assessment={
                        'current_risk': risk_assessment.get('overall_risk_score', 0),
                        'recommended_risk': risk_assessment.get('overall_risk_score', 0) * 0.8
                    }
                )
                recommendations.append(recommendation)

        except Exception as e:
            logger.warning(f"风险管理决策生成失败: {e}")

        return recommendations

    async def _generate_market_timing_decisions(self, market_analysis: MarketAnalysis) -> List[DecisionRecommendation]:
        """生成市场时机决策"""
        recommendations = []

        try:
            # 基于市场状态生成时机建议
            regime = market_analysis.market_regime
            volatility = market_analysis.volatility_level

            if regime == 'bullish_trending' and volatility == 'low':
                recommendation = DecisionRecommendation(
                    decision_id=f"timing_bullish_{int(time.time())}",
                    decision_type=DecisionType.MARKET_TIMING,
                    title="市场时机: 看涨趋势",
                    description="市场处于温和上涨趋势，适合积极策略",
                    confidence=ConfidenceLevel.MEDIUM,
                    confidence_score=0.65,
                    recommended_action="INCREASE_EXPOSURE",
                    expected_outcome={
                        'market_participation': 'high',
                        'timing_advantage': 0.15
                    },
                    risk_assessment={
                        'market_risk': 0.4,
                        'volatility_risk': 0.2
                    },
                    supporting_data={
                        'market_regime': regime,
                        'volatility_level': volatility,
                        'trend_direction': market_analysis.trend_direction
                    }
                )
                recommendations.append(recommendation)

            elif regime in ['bearish_volatile', 'strong_downtrend']:
                recommendation = DecisionRecommendation(
                    decision_id=f"timing_bearish_{int(time.time())}",
                    decision_type=DecisionType.MARKET_TIMING,
                    title="市场时机: 谨慎观望",
                    description="市场处于下跌或高波动状态，建议降低风险",
                    confidence=ConfidenceLevel.HIGH,
                    confidence_score=0.75,
                    recommended_action="REDUCE_EXPOSURE",
                    expected_outcome={
                        'capital_preservation': 'high',
                        'risk_avoidance': 0.8
                    },
                    risk_assessment={
                        'market_risk': 0.8,
                        'volatility_risk': 0.7
                    },
                    supporting_data={
                        'market_regime': regime,
                        'volatility_level': volatility
                    }
                )
                recommendations.append(recommendation)

        except Exception as e:
            logger.warning(f"市场时机决策生成失败: {e}")

        return recommendations

    async def _apply_decision_rules(self, recommendations: List[DecisionRecommendation],
                                    market_analysis: MarketAnalysis) -> List[DecisionRecommendation]:
        """应用决策规则过滤"""
        filtered = []

        try:
            for rec in recommendations:
                # 检查置信度阈值
                min_confidence = self.decision_rules.get('min_confidence', 0.5)
                if rec.confidence_score < min_confidence:
                    continue

                # 检查风险限额
                max_risk = self.decision_rules.get('max_risk_score', 0.8)
                risk_score = rec.risk_assessment.get('risk_level', 'low')
                risk_score_map = {'very_low': 0.2, 'low': 0.4,
                                  'medium': 0.6, 'high': 0.8, 'very_high': 1.0}
                if risk_score_map.get(risk_score, 0) > max_risk:
                    continue

                # 检查市场条件
                if market_analysis.volatility_level == 'very_high':
                    # 在极高波动情况下降低决策置信度
                    rec.confidence_score *= 0.8

                filtered.append(rec)

        except Exception as e:
            logger.warning(f"决策规则应用失败: {e}")
            return recommendations  # 返回原始列表

        return filtered

    def _initialize_decision_rules(self) -> Dict[str, Any]:
        """初始化决策规则"""
        return {
            'min_confidence': 0.5,  # 最小置信度
            'max_risk_score': 0.8,  # 最大风险评分
            'max_decisions_per_hour': 10,  # 每小时最大决策数
            'enable_market_filtering': True,  # 启用市场条件过滤
            'enable_risk_filtering': True,  # 启用风险过滤
            'enable_confidence_filtering': True  # 启用置信度过滤
        }

    def get_decision_history(self, limit: int = 50) -> List[DecisionRecommendation]:
        """获取决策历史"""
        return list(self.decision_history)[-limit:]

    def get_market_analysis(self, analysis_id: str) -> Optional[MarketAnalysis]:
        """获取市场分析结果"""
        return self.market_analyses.get(analysis_id)

    def get_recent_analyses(self, limit: int = 10) -> List[MarketAnalysis]:
        """获取最近的市场分析"""
        return list(self.market_analyses.values())[-limit:]

    def update_decision_rules(self, rules: Dict[str, Any]):
        """更新决策规则"""
        self.decision_rules.update(rules)
        logger.info(f"决策规则已更新: {rules}")


class IntelligentDecisionDashboard:

    """
    智能决策仪表板

    提供基于Web的智能决策可视化界面
    """

    def __init__(self, decision_engine: IntelligentDecisionEngine):

        self.decision_engine = decision_engine
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

        # 仪表板数据
        self.current_market_data = {}
        self.current_portfolio_data = {}
        self.current_risk_profile = {}

        self._setup_layout()
        self._setup_callbacks()

    def _setup_layout(self):
        """设置仪表板布局"""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("智能交易决策支持系统", className="text - center mb - 4"),
                    html.Hr()
                ], width=12)
            ]),

            # 市场概览卡片
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("市场概览"),
                        dbc.CardBody([
                            html.Div(id="market - overview")
                        ])
                    ], className="mb - 4")
                ], width=6),

                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("投资组合状态"),
                        dbc.CardBody([
                            html.Div(id="portfolio - status")
                        ])
                    ], className="mb - 4")
                ], width=6)
            ]),

            # 决策建议区域
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("智能决策建议"),
                        dbc.CardBody([
                            html.Div(id="decision - recommendations")
                        ])
                    ], className="mb - 4")
                ], width=12)
            ]),

            # 技术分析图表
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("技术分析图表"),
                        dbc.CardBody([
                            dcc.Graph(id="technical - chart")
                        ])
                    ], className="mb - 4")
                ], width=12)
            ]),

            # 风险分析
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("风险分析面板"),
                        dbc.CardBody([
                            dcc.Graph(id="risk - analysis - chart")
                        ])
                    ], className="mb - 4")
                ], width=6),

                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("性能指标"),
                        dbc.CardBody([
                            dcc.Graph(id="performance - metrics - chart")
                        ])
                    ], className="mb - 4")
                ], width=6)
            ]),

            # 控制面板
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("控制面板"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button("刷新数据", id="refresh - btn",
                                               color="primary", className="me - 2"),
                                    dbc.Button("生成决策", id="generate - decisions - btn",
                                               color="success", className="me - 2"),
                                    dbc.Button(
                                        "执行建议", id="execute - recommendations - btn", color="warning")
                                ], width=12)
                            ]),
                            html.Br(),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("置信度阈值"),
                                    dcc.Slider(
                                        id="confidence - threshold",
                                        min=0.1,
                                        max=1.0,
                                        step=0.1,
                                        value=0.5,
                                        marks={i / 10: f"{i / 10}" for i in range(1, 11)}
                                    )
                                ], width=6),
                                dbc.Col([
                                    html.Label("风险容忍度"),
                                    dcc.Slider(
                                        id="risk - tolerance",
                                        min=0.1,
                                        max=1.0,
                                        step=0.1,
                                        value=0.6,
                                        marks={i / 10: f"{i / 10}" for i in range(1, 11)}
                                    )
                                ], width=6)
                            ])
                        ])
                    ], className="mb - 4")
                ], width=12)
            ]),

            # 实时更新间隔
            dcc.Interval(
                id="interval - component",
                interval=30000,  # 30秒更新一次
                n_intervals=0
            ),

            # 存储组件用于数据传递
            dcc.Store(id="market - data - store"),
            dcc.Store(id="portfolio - data - store"),
            dcc.Store(id="decisions - data - store")

        ], fluid=True)

    def _setup_callbacks(self):
        """设置回调函数"""
        # 设置数据更新回调
        self._setup_data_update_callback()
        
        # 设置市场概览回调
        self._setup_market_overview_callback()
        
        # 设置投资组合状态回调
        self._setup_portfolio_status_callback()
        
        # 设置决策建议回调
        self._setup_decision_recommendations_callback()
        
        # 设置技术分析图表回调
        self._setup_technical_chart_callback()
        
        # 设置风险分析图表回调
        self._setup_risk_analysis_chart_callback()
        
        # 设置性能指标图表回调
        self._setup_performance_metrics_chart_callback()

    def _setup_data_update_callback(self):
        """设置数据更新回调"""
        @self.app.callback(
            [Output("market - data - store", "data"),
             Output("portfolio - data - store", "data"),
             Output("decisions - data - store", "data")],
            [Input("interval - component", "n_intervals"),
             Input("refresh - btn", "n_clicks")]
        )
        def update_data_stores(n_intervals, n_clicks):
            """更新数据存储"""
            # 这里应该从实际数据源获取数据
            # 暂时返回模拟数据

            market_data = {
                'symbol': 'AAPL',
                'prices': [150 + i * 0.5 + np.sin(i / 5) * 2 for i in range(100)],
                'volume': [1000000 + np.random.randint(-100000, 100000) for _ in range(100)],
                'timestamp': datetime.now().isoformat()
            }

            portfolio_data = {
                'total_value': 100000,
                'positions': {
                    'AAPL': {'quantity': 100, 'value': 15000, 'weight': 0.15},
                    'GOOGL': {'quantity': 50, 'value': 20000, 'weight': 0.20},
                    'MSFT': {'quantity': 80, 'value': 18000, 'weight': 0.18}
                },
                'cash': 35000
            }

            decisions_data = {
                'recommendations': [],
                'last_updated': datetime.now().isoformat()
            }

            return market_data, portfolio_data, decisions_data

    def _setup_market_overview_callback(self):
        """设置市场概览回调"""
        @self.app.callback(
            Output("market - overview", "children"),
            Input("market - data - store", "data")
        )
        def update_market_overview(market_data):
            """更新市场概览"""
            if not market_data:
                return html.Div("暂无市场数据")

            symbol = market_data.get('symbol', 'N / A')
            current_price = market_data.get('prices', [0])[-1]
            price_change = current_price - \
                market_data.get('prices', [current_price]
                                )[-2] if len(market_data.get('prices', [])) > 1 else 0

            return html.Div([
                html.H4(f"{symbol} 市场概览"),
                html.P(f"当前价格: ${current_price:.2f}"),
                html.P(f"价格变化: {price_change:+.2f} ({price_change / current_price * 100:+.2f}%)",
                       style={'color': 'green' if price_change >= 0 else 'red'}),
                html.P(f"成交量: {market_data.get('volume', [0])[-1]:,}")
            ])

    def _setup_portfolio_status_callback(self):
        """设置投资组合状态回调"""
        @self.app.callback(
            Output("portfolio - status", "children"),
            Input("portfolio - data - store", "data")
        )
        def update_portfolio_status(portfolio_data):
            """更新投资组合状态"""
            if not portfolio_data:
                return html.Div("暂无组合数据")

            total_value = portfolio_data.get('total_value', 0)
            cash = portfolio_data.get('cash', 0)
            positions = portfolio_data.get('positions', {})

            return html.Div([
                html.H4("投资组合状态"),
                html.P(f"总价值: ${total_value:,.2f}"),
                html.P(f"现金: ${cash:,.2f}"),
                html.P(f"持仓数量: {len(positions)}"),
                html.Ul([html.Li(f"{symbol}: {pos['quantity']}股 (${pos['value']:,.2f})")
                        for symbol, pos in positions.items()])
            ])

    def _setup_decision_recommendations_callback(self):
        """设置决策建议回调"""
        @self.app.callback(
            Output("decision - recommendations", "children"),
            Input("decisions - data - store", "data")
        )
        def update_decision_recommendations(decisions_data):
            """更新决策建议"""
            recommendations = decisions_data.get('recommendations', [])

            if not recommendations:
                return html.Div("暂无决策建议")

            return html.Div([
                dbc.ListGroup([
                    dbc.ListGroupItem([
                        html.H6(rec['title']),
                        html.P(rec['description']),
                        html.Small(f"置信度: {rec['confidence']} | 建议操作: {rec['recommended_action']}")
                    ], color="success" if rec['confidence'] == 'HIGH' else "warning")
                    for rec in recommendations[:5]  # 显示前5个建议
                ])
            ])

    def _setup_technical_chart_callback(self):
        """设置技术分析图表回调"""
        @self.app.callback(
            Output("technical - chart", "figure"),
            Input("market - data - store", "data")
        )
        def update_technical_chart(market_data):
            """更新技术分析图表"""
            if not market_data:
                return {}

            prices = market_data.get('prices', [])
            timestamps = list(range(len(prices)))

            # 创建价格图表
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # 添加价格线
            fig.add_trace(
                go.Scatter(x=timestamps, y=prices, name="价格", line=dict(color='blue')),
                secondary_y=False
            )

            # 添加移动平均线
            if len(prices) >= 20:
                sma_20 = [np.mean(prices[max(0, i - 20):i + 1]) for i in range(len(prices))]
                fig.add_trace(
                    go.Scatter(x=timestamps, y=sma_20, name="SMA 20", line=dict(color='orange')),
                    secondary_y=False
                )

            # 添加成交量子图
            volume = market_data.get('volume', [])
            if volume:
                fig.add_trace(
                    go.Bar(x=timestamps, y=volume, name="成交量", opacity=0.3),
                    secondary_y=True
                )

            fig.update_layout(
                title="技术分析图表",
                xaxis_title="时间",
                yaxis_title="价格",
                yaxis2_title="成交量"
            )

            return fig

    def _setup_risk_analysis_chart_callback(self):
        """设置风险分析图表回调"""
        @self.app.callback(
            Output("risk - analysis - chart", "figure"),
            Input("portfolio - data - store", "data")
        )
        def update_risk_analysis_chart(portfolio_data):
            """更新风险分析图表"""
            if not portfolio_data:
                return {}

            positions = portfolio_data.get('positions', {})

            # 创建风险分析饼图
            labels = list(positions.keys()) + ['现金']
            values = [pos['weight'] for pos in positions.values(
            )] + [portfolio_data.get('cash', 0) / portfolio_data.get('total_value', 1)]

            fig = go.Figure(data=[go.Pie(labels=labels, values=values, title="资产配置")])
            fig.update_layout(title="风险分析 - 资产配置")

            return fig

    def _setup_performance_metrics_chart_callback(self):
        """设置性能指标图表回调"""
        @self.app.callback(
            Output("performance - metrics - chart", "figure"),
            Input("portfolio - data - store", "data")
        )
        def update_performance_metrics_chart(portfolio_data):
            """更新性能指标图表"""
            if not portfolio_data:
                return {}

            # 创建性能指标条形图
            metrics = {
                '总价值': portfolio_data.get('total_value', 0),
                '现金占比': portfolio_data.get('cash', 0) / max(portfolio_data.get('total_value', 1), 1) * 100,
                '持仓数量': len(portfolio_data.get('positions', {}))
            }

            fig = go.Figure(data=[
                go.Bar(x=list(metrics.keys()), y=list(metrics.values()))
            ])
            fig.update_layout(title="性能指标")

            return fig

    def run_server(self, host: str = "0.0.0.0", port: int = 8050, debug: bool = False):
        """运行服务器"""
        logger.info(f"启动智能决策仪表板服务器: http://{host}:{port}")
        self.app.run_server(host=host, port=port, debug=debug)


# 全局实例
_decision_engine_instance = None
_decision_dashboard_instance = None


def get_intelligent_decision_engine() -> IntelligentDecisionEngine:
    """获取智能决策引擎实例"""
    global _decision_engine_instance
    if _decision_engine_instance is None:
        _decision_engine_instance = IntelligentDecisionEngine()
    return _decision_engine_instance


def get_intelligent_decision_dashboard() -> IntelligentDecisionDashboard:
    """获取智能决策仪表板实例"""
    global _decision_dashboard_instance
    if _decision_dashboard_instance is None:
        _decision_dashboard_instance = IntelligentDecisionDashboard(
            get_intelligent_decision_engine())
    return _decision_dashboard_instance


if __name__ == "__main__":
    # 测试代码
    print("智能交易决策支持系统测试")

    async def test_decision_engine():
        # 获取决策引擎实例
        engine = get_intelligent_decision_engine()

        # 模拟市场数据
        market_data = {
            'symbol': 'AAPL',
            'prices': [150 + i * 0.5 + np.sin(i / 5) * 2 for i in range(100)],
            'volume': [1000000 + np.secrets.randint(-100000, 100000) for _ in range(100)],
            'timeframe': '1H'
        }

        # 模拟组合数据
        portfolio_data = {
            'total_value': 100000,
            'positions': {
                'AAPL': {'quantity': 100, 'value': 15000},
                'GOOGL': {'quantity': 50, 'value': 20000},
                'MSFT': {'quantity': 80, 'value': 18000}
            },
            'cash': 35000
        }

        # 模拟风险画像
        risk_profile = {
            'risk_tolerance': 'medium',
            'max_drawdown': 0.2,
            'max_volatility': 0.3
        }

        # 生成决策
        print("正在生成智能决策建议...")
        recommendations = await engine.analyze_market_and_generate_decisions(
            market_data, portfolio_data, risk_profile
        )

        print(f"生成了 {len(recommendations)} 个决策建议:")

        for rec in recommendations:
            print(f"  - {rec.title}")
            print(f"    描述: {rec.description}")
            print(f"    置信度: {rec.confidence.value} ({rec.confidence_score:.2f})")
            print(f"    建议操作: {rec.recommended_action}")
            print()

        # 获取决策历史
        history = engine.get_decision_history(5)
        print(f"最近5个决策历史: {len(history)} 条")

    # 运行测试
    asyncio.run(test_decision_engine())
    print("智能交易决策支持系统测试完成")
