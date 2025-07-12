#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from src.features.sentiment.analyzer import SentimentAnalyzer
from src.features.manager import FeatureManager

class TestSentimentAnalyzer(unittest.TestCase):
    def setUp(self):
        self.manager = FeatureManager()
        self.analyzer = SentimentAnalyzer(self.manager)

    def test_analyze_text(self):
        # 测试中性文本
        score = self.analyzer.analyze_text("这是一个普通的句子")
        self.assertTrue(-1 <= score <= 1)

        # 测试空文本
        score = self.analyzer.analyze_text("")
        self.assertEqual(score, 0.0)

    def test_analyze_batch(self):
        texts = [
            "这个产品很棒",
            "我不喜欢这个服务",
            "一般般吧"
        ]
        results = self.analyzer.analyze_batch(texts)

        self.assertEqual(len(results), 3)
        for text, score in results.items():
            self.assertTrue(-1 <= score <= 1)

    def test_register_features(self):
        # 测试特征注册
        self.analyzer.register_features()
        self.assertTrue('sentiment' in self.manager.features)

if __name__ == '__main__':
    unittest.main()
