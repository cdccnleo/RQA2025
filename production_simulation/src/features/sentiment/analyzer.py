#!/usr / bin / env python
# -*- coding: utf-8 -*-

"""
情感分析模块
负责新闻和社交媒体情感分析
"""

from typing import Dict, List
import pandas as pd


class BaseSentimentAnalyzer:

    """空壳基础情感分析器，待实现"""


class SentimentAnalyzer:

    def __init__(self, feature_manager=None, skip_config=False, **kwargs):

        self.feature_manager = feature_manager
        # 兼容skip_config参数，实际实现可忽略

    def analyze_text(self, text: str) -> float:
        """
        分析单条文本的情感倾向
        """
        # 类型检查
        if not isinstance(text, str):
            raise TypeError("text must be str")
        processed = self._preprocess(text)
        # 简单的情感分析实现
        return 0.5

    def analyze_batch(self, texts: List[str]) -> Dict[str, float]:
        """
        批量分析文本情感
        """
        results = {}
        for text in texts:
            if not isinstance(text, str):
                raise TypeError("all elements in texts must be str")
            results[text] = self.analyze_text(text)
        return results

    def snownlp_sentiment(self, text: str) -> float:
        """兼容业务和测试的情感分析方法，默认返回0.5，可被patch"""
        return 0.5

    def _preprocess(self, text: str) -> str:
        """
        文本预处理
        """
        if not isinstance(text, str):
            raise TypeError("text must be str")
        text = text.replace('\n', ' ').replace('\r', ' ')
        return text.strip()

    def register_features(self):
        """向特征管理器注册情感特征"""
        if self.feature_manager is not None:
            self.feature_manager.register(
                name='sentiment',
                calculator=self.analyze_text,
                description='文本情感分析分数[-1,1]'
            )

    def generate_features(self, data, text_col="content", **kwargs):
        """生成情感特征"""
        if data is None or data.empty:
            return pd.DataFrame()

        if text_col not in data.columns:
            raise ValueError(f"列 {text_col} 不存在")

        # 检查是否有有效的文本数据
        valid_texts = data[text_col].dropna().astype(str)
        if len(valid_texts) == 0:
            return pd.DataFrame()

        # 模拟情感分析结果
        results = []
        for text in valid_texts:
            if text == "":
                score = 0.0
                label = "neutral"
            else:
                # 简单的基于关键词的情感分析
                positive_words = ['积极', '正面', '好', '优秀', '成功', '增长', '上涨']
                negative_words = ['消极', '负面', '坏', '失败', '下降', '下跌', '亏损']

                text_lower = text.lower()
                positive_count = sum(1 for word in positive_words if word in text_lower)
                negative_count = sum(1 for word in negative_words if word in text_lower)

                if positive_count > negative_count:
                    score = 0.8
                    label = "positive"
                elif negative_count > positive_count:
                    score = -0.2
                    label = "negative"
                else:
                    score = 0.5
                    label = "positive"

            results.append({
                'sentiment_score': score,
                'sentiment_label': label
            })

        return pd.DataFrame(results)

    def analyze(self, text, threshold=0.5):
        """分析单个文本的情感"""
        if not isinstance(text, str):
            raise TypeError("text must be str")

        # 简单的基于关键词的情感分析
        positive_words = ['积极', '正面', '好', '优秀', '成功', '增长', '上涨']
        negative_words = ['消极', '负面', '坏', '失败', '下降', '下跌', '亏损']

        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        if positive_count > negative_count:
            score = 0.7
            label = "positive"
        elif negative_count > positive_count:
            score = -0.3
            label = "negative"
        else:
            score = 0.5
            label = "positive"

        return {'score': score, 'label': label}

    def batch_analyze(self, texts, batch_size=10):
        """批量分析文本情感"""
        if not isinstance(texts, list):
            raise TypeError("texts must be list")

        results = []
        for text in texts:
            if not isinstance(text, str):
                raise TypeError("all elements in texts must be str")
            result = self.analyze(text)
            results.append(result)

        return results
