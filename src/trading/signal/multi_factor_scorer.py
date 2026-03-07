"""
多因子信号评分器
基于技术指标、基本面、市场情绪等多维度因子进行信号评分
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class FactorCategory(Enum):
    """因子类别"""
    TECHNICAL = "technical"      # 技术指标
    FUNDAMENTAL = "fundamental"  # 基本面
    SENTIMENT = "sentiment"      # 市场情绪
    MONEY_FLOW = "money_flow"    # 资金流向
    VOLATILITY = "volatility"    # 波动率


@dataclass
class FactorScore:
    """因子评分"""
    name: str
    category: FactorCategory
    score: float  # 0-100
    weight: float
    description: str


@dataclass
class MultiFactorScoreResult:
    """多因子评分结果"""
    signal_id: str
    symbol: str
    total_score: float  # 0-100
    category_scores: Dict[str, float]
    factor_scores: List[FactorScore]
    recommendation: str
    confidence: float
    timestamp: datetime


class MultiFactorScorer:
    """
    多因子信号评分器
    
    职责：
    1. 技术指标因子评分
    2. 基本面因子评分
    3. 市场情绪因子评分
    4. 资金流向因子评分
    5. 综合评分计算
    """
    
    def __init__(self):
        """初始化多因子评分器"""
        # 因子权重配置
        self.category_weights = {
            FactorCategory.TECHNICAL: 0.35,
            FactorCategory.FUNDAMENTAL: 0.25,
            FactorCategory.SENTIMENT: 0.20,
            FactorCategory.MONEY_FLOW: 0.15,
            FactorCategory.VOLATILITY: 0.05
        }
        
        logger.info("多因子信号评分器初始化完成")
    
    def calculate_score(
        self,
        signal: Dict[str, Any],
        market_data: pd.DataFrame,
        fundamental_data: Optional[Dict] = None,
        sentiment_data: Optional[Dict] = None,
        money_flow_data: Optional[pd.DataFrame] = None
    ) -> MultiFactorScoreResult:
        """
        计算多因子评分
        
        Args:
            signal: 信号数据
            market_data: 市场数据
            fundamental_data: 基本面数据（可选）
            sentiment_data: 情绪数据（可选）
            money_flow_data: 资金流向数据（可选）
            
        Returns:
            多因子评分结果
        """
        try:
            factor_scores = []
            category_scores = {}
            
            # 1. 技术指标因子评分
            technical_score = self._calculate_technical_factors(signal, market_data)
            factor_scores.extend(technical_score)
            category_scores[FactorCategory.TECHNICAL.value] = np.mean([
                f.score for f in technical_score
            ]) if technical_score else 50
            
            # 2. 基本面因子评分
            if fundamental_data:
                fundamental_score = self._calculate_fundamental_factors(fundamental_data)
                factor_scores.extend(fundamental_score)
                category_scores[FactorCategory.FUNDAMENTAL.value] = np.mean([
                    f.score for f in fundamental_score
                ]) if fundamental_score else 50
            else:
                category_scores[FactorCategory.FUNDAMENTAL.value] = 50
            
            # 3. 市场情绪因子评分
            if sentiment_data:
                sentiment_score = self._calculate_sentiment_factors(sentiment_data)
                factor_scores.extend(sentiment_score)
                category_scores[FactorCategory.SENTIMENT.value] = np.mean([
                    f.score for f in sentiment_score
                ]) if sentiment_score else 50
            else:
                category_scores[FactorCategory.SENTIMENT.value] = 50
            
            # 4. 资金流向因子评分
            if money_flow_data is not None and not money_flow_data.empty:
                money_flow_score = self._calculate_money_flow_factors(money_flow_data)
                factor_scores.extend(money_flow_score)
                category_scores[FactorCategory.MONEY_FLOW.value] = np.mean([
                    f.score for f in money_flow_score
                ]) if money_flow_score else 50
            else:
                category_scores[FactorCategory.MONEY_FLOW.value] = 50
            
            # 5. 波动率因子评分
            volatility_score = self._calculate_volatility_factors(market_data)
            factor_scores.extend(volatility_score)
            category_scores[FactorCategory.VOLATILITY.value] = np.mean([
                f.score for f in volatility_score
            ]) if volatility_score else 50
            
            # 计算综合评分
            total_score = sum(
                category_scores[cat.value] * weight
                for cat, weight in self.category_weights.items()
            )
            
            # 计算置信度
            confidence = self._calculate_confidence(factor_scores)
            
            # 生成建议
            recommendation = self._generate_recommendation(total_score, category_scores)
            
            return MultiFactorScoreResult(
                signal_id=signal.get('id', ''),
                symbol=signal.get('symbol', ''),
                total_score=round(total_score, 2),
                category_scores=category_scores,
                factor_scores=factor_scores,
                recommendation=recommendation,
                confidence=round(confidence, 2),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"计算多因子评分失败: {e}")
            return MultiFactorScoreResult(
                signal_id=signal.get('id', ''),
                symbol=signal.get('symbol', ''),
                total_score=50,
                category_scores={},
                factor_scores=[],
                recommendation="评分失败",
                confidence=0,
                timestamp=datetime.now()
            )
    
    def _calculate_technical_factors(
        self,
        signal: Dict[str, Any],
        market_data: pd.DataFrame
    ) -> List[FactorScore]:
        """计算技术指标因子"""
        factors = []
        
        try:
            if market_data.empty or len(market_data) < 20:
                return factors
            
            closes = market_data['close']
            
            # 1. RSI因子
            rsi = self._calculate_rsi(closes, 14)
            if len(rsi) > 0:
                rsi_value = rsi.iloc[-1]
                # RSI < 30 超卖（买入信号加分），RSI > 70 超买（卖出信号加分）
                signal_type = signal.get('type', 'buy')
                if signal_type == 'buy':
                    rsi_score = max(0, min(100, (70 - rsi_value) / 40 * 100))
                else:
                    rsi_score = max(0, min(100, (rsi_value - 30) / 40 * 100))
                
                factors.append(FactorScore(
                    name="RSI",
                    category=FactorCategory.TECHNICAL,
                    score=rsi_score,
                    weight=0.25,
                    description=f"RSI={rsi_value:.1f}"
                ))
            
            # 2. MACD因子
            macd, macd_signal, macd_hist = self._calculate_macd(closes)
            if len(macd) > 0:
                macd_value = macd.iloc[-1]
                macd_signal_value = macd_signal.iloc[-1]
                
                if signal.get('type') == 'buy':
                    macd_score = 100 if macd_value > macd_signal_value else 30
                else:
                    macd_score = 100 if macd_value < macd_signal_value else 30
                
                factors.append(FactorScore(
                    name="MACD",
                    category=FactorCategory.TECHNICAL,
                    score=macd_score,
                    weight=0.25,
                    description=f"MACD={'金叉' if macd_value > macd_signal_value else '死叉'}"
                ))
            
            # 3. 均线排列因子
            ma5 = closes.rolling(5).mean()
            ma10 = closes.rolling(10).mean()
            ma20 = closes.rolling(20).mean()
            
            if len(ma5) > 0 and len(ma10) > 0 and len(ma20) > 0:
                if ma5.iloc[-1] > ma10.iloc[-1] > ma20.iloc[-1]:
                    ma_score = 100 if signal.get('type') == 'buy' else 20
                    ma_desc = "多头排列"
                elif ma5.iloc[-1] < ma10.iloc[-1] < ma20.iloc[-1]:
                    ma_score = 20 if signal.get('type') == 'buy' else 100
                    ma_desc = "空头排列"
                else:
                    ma_score = 50
                    ma_desc = "震荡"
                
                factors.append(FactorScore(
                    name="均线排列",
                    category=FactorCategory.TECHNICAL,
                    score=ma_score,
                    weight=0.25,
                    description=ma_desc
                ))
            
            # 4. 布林带因子
            bb_upper, bb_lower, bb_percent = self._calculate_bollinger_bands(closes)
            if len(bb_percent) > 0:
                bb_value = bb_percent.iloc[-1]
                
                if signal.get('type') == 'buy':
                    bb_score = max(0, min(100, (0.3 - bb_value) / 0.3 * 100))
                else:
                    bb_score = max(0, min(100, (bb_value - 0.7) / 0.3 * 100))
                
                factors.append(FactorScore(
                    name="布林带",
                    category=FactorCategory.TECHNICAL,
                    score=bb_score,
                    weight=0.25,
                    description=f"位置={bb_value:.1%}"
                ))
            
        except Exception as e:
            logger.error(f"计算技术指标因子失败: {e}")
        
        return factors
    
    def _calculate_fundamental_factors(
        self,
        fundamental_data: Dict[str, Any]
    ) -> List[FactorScore]:
        """计算基本面因子"""
        factors = []
        
        try:
            # 1. ROE因子
            roe = fundamental_data.get('roe', 0)
            roe_score = min(100, max(0, roe / 20 * 100))  # ROE 20%为满分
            factors.append(FactorScore(
                name="ROE",
                category=FactorCategory.FUNDAMENTAL,
                score=roe_score,
                weight=0.30,
                description=f"ROE={roe:.1%}"
            ))
            
            # 2. PE因子
            pe = fundamental_data.get('pe', 0)
            # PE 10-20为合理区间，低于10或高于40扣分
            if 10 <= pe <= 20:
                pe_score = 100
            elif pe < 10:
                pe_score = 80
            elif pe > 40:
                pe_score = 30
            else:
                pe_score = max(0, 100 - abs(pe - 15) / 25 * 100)
            
            factors.append(FactorScore(
                name="PE",
                category=FactorCategory.FUNDAMENTAL,
                score=pe_score,
                weight=0.25,
                description=f"PE={pe:.1f}"
            ))
            
            # 3. PB因子
            pb = fundamental_data.get('pb', 0)
            pb_score = min(100, max(0, (3 - pb) / 3 * 100))  # PB越低越好
            factors.append(FactorScore(
                name="PB",
                category=FactorCategory.FUNDAMENTAL,
                score=pb_score,
                weight=0.25,
                description=f"PB={pb:.2f}"
            ))
            
            # 4. 营收增长率
            revenue_growth = fundamental_data.get('revenue_growth', 0)
            growth_score = min(100, max(0, revenue_growth / 0.3 * 100))  # 30%增长为满分
            factors.append(FactorScore(
                name="营收增长",
                category=FactorCategory.FUNDAMENTAL,
                score=growth_score,
                weight=0.20,
                description=f"增长={revenue_growth:.1%}"
            ))
            
        except Exception as e:
            logger.error(f"计算基本面因子失败: {e}")
        
        return factors
    
    def _calculate_sentiment_factors(
        self,
        sentiment_data: Dict[str, Any]
    ) -> List[FactorScore]:
        """计算市场情绪因子"""
        factors = []
        
        try:
            # 1. 情绪分数
            sentiment_score = sentiment_data.get('sentiment_score', 0.5)
            sentiment_factor_score = sentiment_score * 100
            factors.append(FactorScore(
                name="情绪分数",
                category=FactorCategory.SENTIMENT,
                score=sentiment_factor_score,
                weight=0.40,
                description=f"情绪={sentiment_score:.2f}"
            ))
            
            # 2. 新闻数量
            news_count = sentiment_data.get('news_count', 0)
            news_score = min(100, news_count / 20 * 100)  # 20篇新闻为满分
            factors.append(FactorScore(
                name="新闻热度",
                category=FactorCategory.SENTIMENT,
                score=news_score,
                weight=0.30,
                description=f"新闻数={news_count}"
            ))
            
            # 3. 正负面比率
            positive = sentiment_data.get('positive_count', 0)
            negative = sentiment_data.get('negative_count', 1)  # 避免除零
            pos_neg_ratio = positive / (positive + negative)
            ratio_score = pos_neg_ratio * 100
            factors.append(FactorScore(
                name="正负面比率",
                category=FactorCategory.SENTIMENT,
                score=ratio_score,
                weight=0.30,
                description=f"正面率={pos_neg_ratio:.1%}"
            ))
            
        except Exception as e:
            logger.error(f"计算情绪因子失败: {e}")
        
        return factors
    
    def _calculate_money_flow_factors(
        self,
        money_flow_data: pd.DataFrame
    ) -> List[FactorScore]:
        """计算资金流向因子"""
        factors = []
        
        try:
            if money_flow_data.empty:
                return factors
            
            # 1. 主力资金流向
            net_inflow = money_flow_data['net_inflow'].iloc[-1] if 'net_inflow' in money_flow_data.columns else 0
            inflow_score = 50 + min(50, max(-50, net_inflow / 1000000))  # 归一化到0-100
            factors.append(FactorScore(
                name="主力资金",
                category=FactorCategory.MONEY_FLOW,
                score=inflow_score,
                weight=0.50,
                description=f"净流入={net_inflow/10000:.0f}万"
            ))
            
            # 2. 资金流向趋势
            if len(money_flow_data) >= 5:
                recent_inflow = money_flow_data['net_inflow'].tail(5).mean()
                trend_score = 50 + min(50, max(-50, recent_inflow / 1000000))
                factors.append(FactorScore(
                    name="资金趋势",
                    category=FactorCategory.MONEY_FLOW,
                    score=trend_score,
                    weight=0.50,
                    description="5日平均"
                ))
            
        except Exception as e:
            logger.error(f"计算资金流向因子失败: {e}")
        
        return factors
    
    def _calculate_volatility_factors(
        self,
        market_data: pd.DataFrame
    ) -> List[FactorScore]:
        """计算波动率因子"""
        factors = []
        
        try:
            if market_data.empty or len(market_data) < 20:
                return factors
            
            # 计算历史波动率
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # 年化波动率
            
            # 波动率适中最好（20%-40%）
            if 0.20 <= volatility <= 0.40:
                vol_score = 100
            elif volatility < 0.20:
                vol_score = 80  # 波动率太低，缺乏机会
            else:
                vol_score = max(0, 100 - (volatility - 0.40) / 0.40 * 100)
            
            factors.append(FactorScore(
                name="波动率",
                category=FactorCategory.VOLATILITY,
                score=vol_score,
                weight=1.0,
                description=f"年化波动={volatility:.1%}"
            ))
            
        except Exception as e:
            logger.error(f"计算波动率因子失败: {e}")
        
        return factors
    
    def _calculate_confidence(self, factor_scores: List[FactorScore]) -> float:
        """计算置信度"""
        if not factor_scores:
            return 0.0
        
        # 基于因子数量和覆盖的类别计算置信度
        categories = set(f.category for f in factor_scores)
        category_coverage = len(categories) / len(FactorCategory)
        
        # 基于因子评分的一致性计算置信度
        scores = [f.score for f in factor_scores]
        score_std = np.std(scores)
        consistency = 1 - min(1, score_std / 50)  # 标准差越小，一致性越高
        
        return (category_coverage * 0.6 + consistency * 0.4)
    
    def _generate_recommendation(
        self,
        total_score: float,
        category_scores: Dict[str, float]
    ) -> str:
        """生成建议"""
        if total_score >= 80:
            return "强烈推荐"
        elif total_score >= 65:
            return "推荐"
        elif total_score >= 50:
            return "中性"
        elif total_score >= 35:
            return "谨慎"
        else:
            return "不推荐"
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> tuple:
        """计算MACD"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    def _calculate_bollinger_bands(
        self,
        prices: pd.Series,
        period: int = 20,
        std_dev: int = 2
    ) -> tuple:
        """计算布林带"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        percent = (prices - lower) / (upper - lower)
        return upper, lower, percent


# 单例实例
_multi_factor_scorer: Optional[MultiFactorScorer] = None


def get_multi_factor_scorer() -> MultiFactorScorer:
    """获取多因子评分器实例"""
    global _multi_factor_scorer
    if _multi_factor_scorer is None:
        _multi_factor_scorer = MultiFactorScorer()
    return _multi_factor_scorer
