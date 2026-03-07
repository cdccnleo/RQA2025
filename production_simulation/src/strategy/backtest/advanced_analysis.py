#!/usr / bin / env python
# -*- coding: utf-8 -*-

"""
高级分析功能模块
包括多因子分析、情绪分析和宏观经济分析
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class AdvancedAnalysis:

    """高级分析功能"""

    def __init__(self):

        self.factor_weights = {
            'momentum': 0.3,
            'value': 0.25,
            'quality': 0.25,
            'volatility': 0.2
        }

    def analyze_multifactor(self, market_data: pd.DataFrame,


                            factor_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """多因子分析"""
        logger.info("开始多因子分析...")

        results = []

        for factor_name, factor_df in factor_data.items():
            if factor_name not in self.factor_weights:
                continue

        try:
            factor_value = factor_df.iloc[-1].mean() if len(factor_df) > 0 else 0.0
            factor_contribution = factor_value * self.factor_weights[factor_name]

            results.append({
                'factor_name': factor_name,
                'factor_value': factor_value,
                'factor_weight': self.factor_weights[factor_name],
                'factor_contribution': factor_contribution
            })

        except Exception as e:
            logger.error(f"因子 {factor_name} 分析失败: {e}")

        return results

    def analyze_sentiment(self, news_data: List[Dict],


                          social_data: List[Dict]) -> Dict:
        """情绪分析"""
        logger.info("开始情绪分析...")

        all_texts = []

        for news in news_data:
            if 'title' in news:
                all_texts.append(news['title'])

        for post in social_data:
            if 'text' in post:
                all_texts.append(post['text'])

        if not all_texts:
            return {
                'sentiment_score': 0.0,
                'sentiment_type': 'neutral',
                'confidence': 0.0
            }

        sentiment_score = self._calculate_sentiment_score(all_texts)

        if sentiment_score > 0.1:
            sentiment_type = 'positive'
        elif sentiment_score < -0.1:
            sentiment_type = 'negative'
        else:
            sentiment_type = 'neutral'

        confidence = min(abs(sentiment_score) * 2, 1.0)

        return {
            'sentiment_score': sentiment_score,
            'sentiment_type': sentiment_type,
            'confidence': confidence
        }

    def _calculate_sentiment_score(self, texts: List[str]) -> float:
        """计算情感得分"""
        positive_words = {'上涨', '利好', '增长', '盈利', '突破', '强势', '看好', '推荐', '买入'}
        negative_words = {'下跌', '利空', '亏损', '风险', '担忧', '看空', '卖出', '减持', '回避'}

        total_score = 0.0
        total_words = 0

        for text in texts:
            if not isinstance(text, str):
                continue

            words = text.lower().split()
            text_score = 0.0

            for word in words:
                if word in positive_words:
                    text_score += 1.0
                elif word in negative_words:
                    text_score -= 1.0

            total_score += text_score
            total_words += len(words)

        return total_score / total_words if total_words > 0 else 0.0

    def analyze_macro_economy(self, macro_data: Dict[str, float]) -> Dict:
        """宏观经济分析"""
        logger.info("开始宏观经济分析...")

        gdp_growth = macro_data.get('gdp_growth', 0.0)
        inflation_rate = macro_data.get('inflation_rate', 0.0)

        # 分析市场情绪
        sentiment_score = 0.0
        if gdp_growth > 3.0:
            sentiment_score += 1.0
        elif gdp_growth < 0:
            sentiment_score -= 1.0

        if 1.0 <= inflation_rate <= 3.0:
            sentiment_score += 0.5
        elif inflation_rate > 5.0:
            sentiment_score -= 1.0

        if sentiment_score > 0.5:
            market_sentiment = 'positive'
        elif sentiment_score < -0.5:
            market_sentiment = 'negative'
        else:
            market_sentiment = 'neutral'

        # 评估风险水平
        risk_score = 0.0
        if gdp_growth < 0:
            risk_score += 2.0
        if inflation_rate > 5.0:
            risk_score += 2.0

        if risk_score >= 3.0:
            risk_level = 'high'
        elif risk_score >= 1.0:
            risk_level = 'medium'
        else:
            risk_level = 'low'

        return {
            'gdp_growth': gdp_growth,
            'inflation_rate': inflation_rate,
            'market_sentiment': market_sentiment,
            'risk_level': risk_level
        }

    def generate_comprehensive_report(self,


                                      market_data: pd.DataFrame,
                                      factor_data: Dict[str, pd.DataFrame],
                                      news_data: List[Dict],
                                      social_data: List[Dict],
                                      macro_data: Dict[str, float]) -> Dict[str, Any]:
        """生成综合分析报告"""
        logger.info("生成综合分析报告...")

        # 多因子分析
        factor_analysis = self.analyze_multifactor(market_data, factor_data)

        # 情绪分析
        sentiment_analysis = self.analyze_sentiment(news_data, social_data)

        # 宏观经济分析
        macro_analysis = self.analyze_macro_economy(macro_data)

        # 综合评分
        composite_score = self._calculate_composite_score(
            factor_analysis, sentiment_analysis, macro_analysis
        )

        return {
            'timestamp': datetime.now(),
            'factor_analysis': factor_analysis,
            'sentiment_analysis': sentiment_analysis,
            'macro_analysis': macro_analysis,
            'composite_score': composite_score,
            'recommendation': self._generate_recommendation(composite_score)
        }

    def _calculate_composite_score(self,


                                   factor_analysis: List[Dict],
                                   sentiment_analysis: Dict,
                                   macro_analysis: Dict) -> float:
        """计算综合评分"""
        score = 0.0

        # 因子分析贡献 (40%)
        factor_score = sum(f['factor_contribution'] for f in factor_analysis)
        score += factor_score * 0.4

        # 情绪分析贡献 (30%)
        score += sentiment_analysis['sentiment_score'] * 0.3

        # 宏观经济贡献 (30%)
        macro_score = 0.5 if macro_analysis['market_sentiment'] == 'positive' else \
            -0.5 if macro_analysis['market_sentiment'] == 'negative' else 0.0
        score += macro_score * 0.3

        return np.clip(score, -1, 1)

    def _generate_recommendation(self, composite_score: float) -> str:
        """生成投资建议"""
        if composite_score > 0.3:
            return 'strong_buy'
        elif composite_score > 0.1:
            return 'buy'
        elif composite_score > -0.1:
            return 'hold'
        elif composite_score > -0.3:
            return 'sell'
        else:
            return 'strong_sell'

        if __name__ == "__main__":
            analysis = AdvancedAnalysis()
            print("高级分析功能模块初始化完成")
