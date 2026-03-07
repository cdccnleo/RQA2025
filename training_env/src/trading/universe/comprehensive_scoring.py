"""综合评分模型实现"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ScoringDimension(Enum):
    """评分维度枚举"""
    LIQUIDITY = "liquidity"
    VOLATILITY = "volatility"
    FUNDAMENTAL = "fundamental"
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    RISK = "risk"


@dataclass
class ScoringResult:
    """评分结果"""
    stock: str
    composite_score: float
    dimension_scores: Dict[str, float]
    rank: int
    percentile: float


class ComprehensiveScoringModel:
    """综合评分模型"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scoring_weights = {
            ScoringDimension.LIQUIDITY: config.get('liquidity_weight', 0.20),
            ScoringDimension.VOLATILITY: config.get('volatility_weight', 0.15),
            ScoringDimension.FUNDAMENTAL: config.get('fundamental_weight', 0.25),
            ScoringDimension.TECHNICAL: config.get('technical_weight', 0.20),
            ScoringDimension.SENTIMENT: config.get('sentiment_weight', 0.10),
            ScoringDimension.RISK: config.get('risk_weight', 0.10)
        }

        # 确保权重总和为1
        total_weight = sum(self.scoring_weights.values())
        if total_weight != 1.0:
            for dimension in self.scoring_weights:
                self.scoring_weights[dimension] /= total_weight

        self.scoring_history = []
        self.performance_tracker = ScoringPerformanceTracker()

    def calculate_scores(self, universe: List[str], market_data: pd.DataFrame) -> List[ScoringResult]:
        """计算综合评分"""
        logger.info(f"开始计算综合评分，股票数量: {len(universe)}")

        if not universe or market_data.empty:
            return []

        results = []

        for stock in universe:
            # 计算各维度评分
            dimension_scores = self._calculate_dimension_scores(stock, market_data)

            # 计算综合评分
            composite_score = self._calculate_composite_score(dimension_scores)

            # 创建评分结果
            result = ScoringResult(
                stock=stock,
                composite_score=composite_score,
                dimension_scores=dimension_scores,
                rank=0,  # 稍后设置
                percentile=0.0  # 稍后设置
            )

            results.append(result)

        # 排序并设置排名
        results.sort(key=lambda x: x.composite_score, reverse=True)
        for i, result in enumerate(results):
            result.rank = i + 1
            result.percentile = (len(results) - i) / len(results) * 100

        # 记录评分历史
        self._record_scoring_history(results)

        logger.info(
            f"评分完成，最高分: {results[0].composite_score:.4f}, 最低分: {results[-1].composite_score:.4f}")

        return results

    def _calculate_dimension_scores(self, stock: str, market_data: pd.DataFrame) -> Dict[str, float]:
        """计算各维度评分"""
        dimension_scores = {}

        # 流动性评分
        dimension_scores[ScoringDimension.LIQUIDITY.value] = self._calculate_liquidity_score(
            stock, market_data)

        # 波动率评分
        dimension_scores[ScoringDimension.VOLATILITY.value] = self._calculate_volatility_score(
            stock, market_data)

        # 基本面评分
        dimension_scores[ScoringDimension.FUNDAMENTAL.value] = self._calculate_fundamental_score(
            stock, market_data)

        # 技术面评分
        dimension_scores[ScoringDimension.TECHNICAL.value] = self._calculate_technical_score(
            stock, market_data)

        # 情感面评分
        dimension_scores[ScoringDimension.SENTIMENT.value] = self._calculate_sentiment_score(
            stock, market_data)

        # 风险面评分
        dimension_scores[ScoringDimension.RISK.value] = self._calculate_risk_score(
            stock, market_data)

        return dimension_scores

    def _calculate_composite_score(self, dimension_scores: Dict[str, float]) -> float:
        """计算综合评分"""
        composite_score = 0.0

        for dimension, score in dimension_scores.items():
            weight = self.scoring_weights.get(ScoringDimension(dimension), 0.0)
            composite_score += score * weight

        return composite_score

    def _calculate_liquidity_score(self, stock: str, market_data: pd.DataFrame) -> float:
        """计算流动性评分"""
        if stock not in market_data.index:
            return 0.0

        stock_data = market_data.loc[stock]

        # 流动性指标
        volume_score = self._normalize_metric(stock_data.get('volume', 0), 1000000, 100000000)
        turnover_score = self._normalize_metric(stock_data.get('turnover_rate', 0), 0.01, 0.1)
        market_cap_score = self._normalize_metric(
            stock_data.get('market_cap', 0), 1000000000, 10000000000)

        # 综合流动性评分
        liquidity_score = (volume_score + turnover_score + market_cap_score) / 3

        return min(max(liquidity_score, 0.0), 1.0)

    def _calculate_volatility_score(self, stock: str, market_data: pd.DataFrame) -> float:
        """计算波动率评分"""
        if stock not in market_data.index:
            return 0.0

        stock_data = market_data.loc[stock]

        # 波动率指标（越低越好，需要反转）
        volatility = stock_data.get('volatility', 0.5)
        beta = stock_data.get('beta', 1.0)
        sharpe_ratio = stock_data.get('sharpe_ratio', 0.0)

        # 波动率评分（反转）
        volatility_score = 1.0 - self._normalize_metric(volatility, 0.1, 0.5)
        beta_score = 1.0 - self._normalize_metric(beta, 0.5, 2.0)
        sharpe_score = self._normalize_metric(sharpe_ratio, 0.0, 1.0)

        # 综合波动率评分
        volatility_score = (volatility_score + beta_score + sharpe_score) / 3

        return min(max(volatility_score, 0.0), 1.0)

    def _calculate_fundamental_score(self, stock: str, market_data: pd.DataFrame) -> float:
        """计算基本面评分"""
        if stock not in market_data.index:
            return 0.0

        stock_data = market_data.loc[stock]

        # 基本面指标
        roe = stock_data.get('roe', 0.0)
        pe = stock_data.get('pe', 50.0)
        pb = stock_data.get('pb', 5.0)
        profit_growth = stock_data.get('profit_growth', 0.0)

        # 基本面评分
        roe_score = self._normalize_metric(roe, 0.05, 0.3)
        pe_score = 1.0 - self._normalize_metric(pe, 10, 50)  # PE越低越好
        pb_score = 1.0 - self._normalize_metric(pb, 1, 5)     # PB越低越好
        growth_score = self._normalize_metric(profit_growth, 0.05, 0.3)

        # 综合基本面评分
        fundamental_score = (roe_score + pe_score + pb_score + growth_score) / 4

        return min(max(fundamental_score, 0.0), 1.0)

    def _calculate_technical_score(self, stock: str, market_data: pd.DataFrame) -> float:
        """计算技术面评分"""
        if stock not in market_data.index:
            return 0.0

        stock_data = market_data.loc[stock]

        # 技术指标
        rsi = stock_data.get('rsi', 50.0)
        macd = stock_data.get('macd', 0.0)
        ma_trend = stock_data.get('ma_trend', 0.0)

        # 技术面评分
        rsi_score = 1.0 - abs(rsi - 50) / 50  # RSI接近50最好
        macd_score = 1.0 - min(abs(macd), 0.2) / 0.2  # MACD接近0最好
        trend_score = self._normalize_metric(ma_trend, -0.1, 0.1)

        # 综合技术面评分
        technical_score = (rsi_score + macd_score + trend_score) / 3

        return min(max(technical_score, 0.0), 1.0)

    def _calculate_sentiment_score(self, stock: str, market_data: pd.DataFrame) -> float:
        """计算情感面评分"""
        if stock not in market_data.index:
            return 0.0

        stock_data = market_data.loc[stock]

        # 情感指标
        sentiment_score = stock_data.get('sentiment_score', 0.5)
        news_volume = stock_data.get('news_volume', 50)

        # 情感面评分
        sentiment_raw = self._normalize_metric(sentiment_score, 0.3, 0.8)
        news_score = 1.0 - self._normalize_metric(news_volume, 0, 100)  # 新闻量适中最好

        # 综合情感面评分
        sentiment_score = (sentiment_raw + news_score) / 2

        return min(max(sentiment_score, 0.0), 1.0)

    def _calculate_risk_score(self, stock: str, market_data: pd.DataFrame) -> float:
        """计算风险面评分"""
        if stock not in market_data.index:
            return 0.0

        stock_data = market_data.loc[stock]

        # 风险指标
        credit_risk = stock_data.get('credit_risk', 0.3)
        concentration_risk = stock_data.get('concentration_risk', 0.2)
        rating = stock_data.get('rating', 'BBB')

        # 风险面评分（反转）
        credit_score = 1.0 - self._normalize_metric(credit_risk, 0.1, 0.5)
        concentration_score = 1.0 - self._normalize_metric(concentration_risk, 0.05, 0.3)
        rating_score = self._calculate_rating_score(rating)

        # 综合风险面评分
        risk_score = (credit_score + concentration_score + rating_score) / 3

        return min(max(risk_score, 0.0), 1.0)

    def _normalize_metric(self, value: float, min_val: float, max_val: float) -> float:
        """标准化指标值到[0,1]区间"""
        if max_val == min_val:
            return 0.5

        normalized = (value - min_val) / (max_val - min_val)
        return min(max(normalized, 0.0), 1.0)

    def _calculate_rating_score(self, rating: str) -> float:
        """计算评级分数"""
        rating_scores = {
            'AAA': 1.0, 'AA': 0.95, 'A': 0.9, 'BBB': 0.8,
            'BB': 0.6, 'B': 0.4, 'CCC': 0.2, 'CC': 0.1, 'C': 0.05, 'D': 0.0
        }
        return rating_scores.get(rating, 0.5)

    def _record_scoring_history(self, results: List[ScoringResult]):
        """记录评分历史"""
        history_record = {
            'timestamp': datetime.now(),
            'total_stocks': len(results),
            'avg_score': np.mean([r.composite_score for r in results]),
            'std_score': np.std([r.composite_score for r in results]),
            'top_10_avg': np.mean([r.composite_score for r in results[:10]]),
            'bottom_10_avg': np.mean([r.composite_score for r in results[-10:]])
        }

        self.scoring_history.append(history_record)

        # 保留最近100条记录
        if len(self.scoring_history) > 100:
            self.scoring_history.pop(0)

    def get_scoring_statistics(self) -> Dict[str, any]:
        """获取评分统计信息"""
        if not self.scoring_history:
            return {}

        recent_history = self.scoring_history[-10:]  # 最近10次评分

        return {
            'avg_total_stocks': np.mean([r['total_stocks'] for r in recent_history]),
            'avg_score': np.mean([r['avg_score'] for r in recent_history]),
            'score_volatility': np.std([r['avg_score'] for r in recent_history]),
            'top_bottom_spread': np.mean([r['top_10_avg'] - r['bottom_10_avg'] for r in recent_history])
        }

    def update_weights(self, new_weights: Dict[str, float]):
        """更新评分权重"""
        for dimension, weight in new_weights.items():
            if dimension in self.scoring_weights:
                self.scoring_weights[dimension] = weight

        # 重新归一化
        total_weight = sum(self.scoring_weights.values())
        if total_weight != 1.0:
            for dimension in self.scoring_weights:
                self.scoring_weights[dimension] /= total_weight

        logger.info(f"更新评分权重: {self.scoring_weights}")


class ScoringPerformanceTracker:
    """评分表现跟踪器"""

    def __init__(self):
        self.performance_data = []
        self.dimension_performance = {}

    def track_performance(self, scoring_results: List[ScoringResult],
                          actual_returns: Dict[str, float]):
        """跟踪评分表现"""
        if not scoring_results or not actual_returns:
            return

        # 计算评分与实际收益的相关性
        scores = [r.composite_score for r in scoring_results]
        returns = [actual_returns.get(r.stock, 0.0) for r in scoring_results]

        if len(scores) > 1:
            correlation = np.corrcoef(scores, returns)[0, 1]

            performance_record = {
                'timestamp': datetime.now(),
                'correlation': correlation,
                'avg_score': np.mean(scores),
                'avg_return': np.mean(returns),
                'score_std': np.std(scores),
                'return_std': np.std(returns)
            }

            self.performance_data.append(performance_record)

            # 保留最近50条记录
            if len(self.performance_data) > 50:
                self.performance_data.pop(0)

    def get_performance_summary(self) -> Dict[str, float]:
        """获取表现摘要"""
        if not self.performance_data:
            return {}

        correlations = [r['correlation'] for r in self.performance_data]

        return {
            'mean_correlation': np.mean(correlations),
            'std_correlation': np.std(correlations),
            'recent_correlation': np.mean(correlations[-5:]) if len(correlations) >= 5 else np.mean(correlations),
            'positive_correlation_rate': sum(1 for c in correlations if c > 0) / len(correlations)
        }
