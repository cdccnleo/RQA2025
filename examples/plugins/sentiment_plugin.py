#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
情感分析插件示例

演示如何创建情感分析插件。
"""

import pandas as pd
from typing import Any, Dict, List
from src.features.plugins import BaseFeaturePlugin, PluginMetadata


class SentimentAnalysisPlugin(BaseFeaturePlugin):
    """情感分析插件"""

    def _get_metadata(self) -> PluginMetadata:
        """获取插件元数据"""
        from src.features.plugins import PluginMetadata, PluginType

        return PluginMetadata(
            name="sentiment_analysis_plugin",
            version="1.0.0",
            description="情感分析插件，提供文本情感分析功能",
            author="RQA Team",
            plugin_type=PluginType.ANALYZER,
            dependencies=["pandas", "numpy"],
            tags=["sentiment", "analysis", "nlp"],
            config_schema={
                "language": {"type": str, "default": "zh-cn"},
                "confidence_threshold": {"type": float, "default": 0.6},
                "max_keywords": {"type": int, "default": 100},
                "batch_size": {"type": int, "default": 1000}
            },
            min_api_version="1.0.0",
            max_api_version="2.0.0"
        )

    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        处理数据，进行情感分析

        Args:
            data: 输入数据（包含文本列）
            **kwargs: 额外参数

        Returns:
            包含情感分析结果的数据框
        """
        if data.empty:
            self.logger.warning("输入数据为空")
            return data

        # 获取配置
        language = self.config.get("language", "zh-cn")
        confidence_threshold = self.config.get("confidence_threshold", 0.6)
        max_keywords = self.config.get("max_keywords", 100)
        batch_size = self.config.get("batch_size", 1000)

        result = data.copy()

        # 查找文本列
        text_columns = self._find_text_columns(data)
        if not text_columns:
            self.logger.warning("未找到文本列")
            return result

        # 对每个文本列进行情感分析
        for text_col in text_columns:
            if text_col in data.columns:
                # 情感分析
                sentiment_scores = self._analyze_sentiment(
                    data[text_col], language, confidence_threshold)
                result[f'{text_col}_sentiment_score'] = sentiment_scores['score']
                result[f'{text_col}_sentiment_label'] = sentiment_scores['label']
                result[f'{text_col}_sentiment_confidence'] = sentiment_scores['confidence']

                # 关键词提取
                keywords = self._extract_keywords(data[text_col], max_keywords)
                result[f'{text_col}_keywords'] = keywords

        self.logger.info(f"情感分析完成，处理了 {len(text_columns)} 个文本列")
        return result

    def _find_text_columns(self, data: pd.DataFrame) -> List[str]:
        """查找文本列"""
        text_columns = []

        # 常见的文本列名
        common_text_names = ['text', 'content', 'message',
                             'comment', 'title', 'description', 'summary']

        for col in data.columns:
            col_lower = col.lower()
            # 检查是否包含文本相关关键词
            if any(keyword in col_lower for keyword in common_text_names):
                text_columns.append(col)
            # 检查数据类型
            elif data[col].dtype == 'object':
                # 检查是否主要是文本数据
                sample_values = data[col].dropna().head(10)
                if len(sample_values) > 0:
                    avg_length = sample_values.astype(str).str.len().mean()
                    if avg_length > 10:  # 平均长度大于10认为是文本
                        text_columns.append(col)

        return text_columns

    def _analyze_sentiment(self, text_series: pd.Series, language: str,
                           confidence_threshold: float) -> Dict[str, pd.Series]:
        """
        分析情感

        Args:
            text_series: 文本序列
            language: 语言
            confidence_threshold: 置信度阈值

        Returns:
            情感分析结果
        """
        scores = []
        labels = []
        confidences = []

        for text in text_series:
            if pd.isna(text) or text == "":
                scores.append(0.0)
                labels.append("neutral")
                confidences.append(0.0)
            else:
                # 简化的情感分析逻辑
                sentiment_result = self._simple_sentiment_analysis(str(text), language)
                scores.append(sentiment_result['score'])
                labels.append(sentiment_result['label'])
                confidences.append(sentiment_result['confidence'])

        return {
            'score': pd.Series(scores, index=text_series.index),
            'label': pd.Series(labels, index=text_series.index),
            'confidence': pd.Series(confidences, index=text_series.index)
        }

    def _simple_sentiment_analysis(self, text: str, language: str) -> Dict[str, Any]:
        """
        简单的情感分析

        Args:
            text: 文本
            language: 语言

        Returns:
            情感分析结果
        """
        # 简化的情感分析实现
        # 在实际应用中，这里应该使用专业的情感分析模型

        # 中文情感词典（简化版）
        positive_words = ['好', '棒', '优秀', '赞', '喜欢', '满意', '高兴', '开心', '成功', '盈利']
        negative_words = ['差', '坏', '糟糕', '失望', '讨厌', '失败', '亏损', '下跌', '风险', '问题']

        text_lower = text.lower()

        # 计算情感分数
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        total_words = len(text.split())
        if total_words == 0:
            return {'score': 0.0, 'label': 'neutral', 'confidence': 0.0}

        # 计算情感分数 (-1 到 1)
        sentiment_score = (positive_count - negative_count) / max(total_words, 1)
        sentiment_score = max(-1.0, min(1.0, sentiment_score))

        # 确定标签
        if sentiment_score > 0.1:
            label = 'positive'
        elif sentiment_score < -0.1:
            label = 'negative'
        else:
            label = 'neutral'

        # 计算置信度
        confidence = min(1.0, abs(sentiment_score) * 2)

        return {
            'score': sentiment_score,
            'label': label,
            'confidence': confidence
        }

    def _extract_keywords(self, text_series: pd.Series, max_keywords: int) -> pd.Series:
        """
        提取关键词

        Args:
            text_series: 文本序列
            max_keywords: 最大关键词数量

        Returns:
            关键词序列
        """
        keywords_list = []

        for text in text_series:
            if pd.isna(text) or text == "":
                keywords_list.append([])
            else:
                # 简化的关键词提取
                keywords = self._simple_keyword_extraction(str(text), max_keywords)
                keywords_list.append(keywords)

        return pd.Series(keywords_list, index=text_series.index)

    def _simple_keyword_extraction(self, text: str, max_keywords: int) -> List[str]:
        """
        简单的关键词提取

        Args:
            text: 文本
            max_keywords: 最大关键词数量

        Returns:
            关键词列表
        """
        # 简化的关键词提取实现
        # 在实际应用中，这里应该使用专业的关键词提取算法

        # 分词（简化版）
        import re
        words = re.findall(r'\w+', text.lower())

        # 过滤停用词
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
                      '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        filtered_words = [word for word in words if word not in stop_words and len(word) > 1]

        # 统计词频
        from collections import Counter
        word_counts = Counter(filtered_words)

        # 返回最频繁的词作为关键词
        keywords = [word for word, count in word_counts.most_common(max_keywords)]

        return keywords

    def _get_capabilities(self) -> Dict[str, Any]:
        """获取插件能力"""
        return {
            "languages": ["zh-cn", "en-us"],
            "features": ["sentiment_analysis", "keyword_extraction"],
            "input_types": ["text", "string"],
            "output_features": ["sentiment_score", "sentiment_label", "sentiment_confidence", "keywords"],
            "configurable": True
        }

    def _validate_input(self, data: Any) -> bool:
        """验证输入数据"""
        if not isinstance(data, pd.DataFrame):
            return False

        # 检查是否有文本列
        text_columns = self._find_text_columns(data)
        if not text_columns:
            self.logger.warning("未找到文本列")
            return False

        return True
