#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
情感分析模块
负责新闻和社交媒体文本的情感分析
"""

from typing import List, Dict
import numpy as np
from ..manager import FeatureManager
from .models import load_pretrained_model

class SentimentAnalyzer:
    def __init__(self, feature_manager: FeatureManager):
        self.feature_manager = feature_manager
        self.model = load_pretrained_model()

    def analyze_text(self, text: str) -> float:
        """分析单条文本的情感倾向

        Args:
            text: 待分析的文本内容

        Returns:
            情感分数，范围[-1,1]，正值表示积极，负值表示消极
        """
        # 预处理文本
        processed = self._preprocess(text)

        # 使用预训练模型分析
        score = self.model.predict(processed)

        # 标准化到[-1,1]范围
        return float(2 * (score - 0.5))

    def analyze_batch(self, texts: List[str]) -> Dict[str, float]:
        """批量分析文本情感

        Args:
            texts: 待分析的文本列表

        Returns:
            字典格式结果 {text: score}
        """
        results = {}
        for text in texts:
            results[text] = self.analyze_text(text)
        return results

    def _preprocess(self, text: str) -> str:
        """文本预处理

        Args:
            text: 原始文本

        Returns:
            清洗后的文本
        """
        # 去除特殊字符
        text = text.replace('\n', ' ').replace('\r', ' ')
        # 其他预处理步骤...
        return text.strip()

    def register_features(self):
        """向特征管理器注册情感特征"""
        self.feature_manager.register(
            name='sentiment',
            calculator=self.analyze_text,
            description='文本情感分析分数[-1,1]'
        )
