#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sentiment分析器测试覆盖
测试sentiment/analyzer.py
"""

import pytest
import pandas as pd
from unittest.mock import Mock, MagicMock
from typing import List, Dict

from src.features.sentiment.analyzer import SentimentAnalyzer, BaseSentimentAnalyzer


class TestSentimentAnalyzer:
    """SentimentAnalyzer测试"""

    def test_sentiment_analyzer_initialization(self):
        """测试初始化"""
        analyzer = SentimentAnalyzer()
        assert analyzer.feature_manager is None

    def test_sentiment_analyzer_initialization_with_feature_manager(self):
        """测试带feature_manager初始化"""
        mock_manager = Mock()
        analyzer = SentimentAnalyzer(feature_manager=mock_manager)
        assert analyzer.feature_manager == mock_manager

    def test_sentiment_analyzer_initialization_with_skip_config(self):
        """测试skip_config参数（兼容性）"""
        analyzer = SentimentAnalyzer(skip_config=True)
        assert analyzer.feature_manager is None

    def test_analyze_text_valid(self):
        """测试分析有效文本"""
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze_text("这是一个测试文本")
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_analyze_text_invalid_type(self):
        """测试分析无效类型文本"""
        analyzer = SentimentAnalyzer()
        with pytest.raises(TypeError, match="text must be str"):
            analyzer.analyze_text(123)

    def test_analyze_text_empty_string(self):
        """测试分析空字符串"""
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze_text("")
        assert isinstance(result, float)

    def test_analyze_batch_valid(self):
        """测试批量分析有效文本"""
        analyzer = SentimentAnalyzer()
        texts = ["文本1", "文本2", "文本3"]
        results = analyzer.analyze_batch(texts)
        assert isinstance(results, dict)
        assert len(results) == 3
        for text, score in results.items():
            assert text in texts
            assert isinstance(score, float)

    def test_analyze_batch_invalid_type(self):
        """测试批量分析无效类型"""
        analyzer = SentimentAnalyzer()
        with pytest.raises(TypeError, match="all elements in texts must be str"):
            analyzer.analyze_batch(["text1", 123, "text3"])

    def test_analyze_batch_empty_list(self):
        """测试批量分析空列表"""
        analyzer = SentimentAnalyzer()
        results = analyzer.analyze_batch([])
        assert isinstance(results, dict)
        assert len(results) == 0

    def test_snownlp_sentiment(self):
        """测试snownlp情感分析（兼容方法）"""
        analyzer = SentimentAnalyzer()
        result = analyzer.snownlp_sentiment("测试文本")
        assert isinstance(result, float)
        assert result == 0.5  # 默认返回值

    def test_preprocess_valid(self):
        """测试预处理有效文本"""
        analyzer = SentimentAnalyzer()
        text = "测试\n文本\r内容"
        result = analyzer._preprocess(text)
        assert isinstance(result, str)
        assert "\n" not in result
        assert "\r" not in result

    def test_preprocess_invalid_type(self):
        """测试预处理无效类型"""
        analyzer = SentimentAnalyzer()
        with pytest.raises(TypeError, match="text must be str"):
            analyzer._preprocess(123)

    def test_preprocess_empty_string(self):
        """测试预处理空字符串"""
        analyzer = SentimentAnalyzer()
        result = analyzer._preprocess("")
        assert result == ""

    def test_register_features_with_manager(self):
        """测试注册特征（有manager）"""
        mock_manager = Mock()
        analyzer = SentimentAnalyzer(feature_manager=mock_manager)
        analyzer.register_features()
        mock_manager.register.assert_called_once()

    def test_register_features_without_manager(self):
        """测试注册特征（无manager）"""
        analyzer = SentimentAnalyzer()
        # 应该不报错
        analyzer.register_features()

    def test_generate_features_empty_dataframe(self):
        """测试生成特征（空DataFrame）"""
        analyzer = SentimentAnalyzer()
        result = analyzer.generate_features(pd.DataFrame())
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_generate_features_none_data(self):
        """测试生成特征（None数据）"""
        analyzer = SentimentAnalyzer()
        result = analyzer.generate_features(None)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_generate_features_missing_column(self):
        """测试生成特征（缺少列）"""
        analyzer = SentimentAnalyzer()
        data = pd.DataFrame({'other_col': ['text1', 'text2']})
        with pytest.raises(ValueError, match="列 content 不存在"):
            analyzer.generate_features(data, text_col="content")

    def test_generate_features_valid_data(self):
        """测试生成特征（有效数据）"""
        analyzer = SentimentAnalyzer()
        data = pd.DataFrame({
            'content': ['积极正面的消息', '消极负面的消息', '中性消息']
        })
        result = analyzer.generate_features(data, text_col="content")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert 'sentiment_score' in result.columns
        assert 'sentiment_label' in result.columns

    def test_generate_features_with_empty_texts(self):
        """测试生成特征（空文本）"""
        analyzer = SentimentAnalyzer()
        data = pd.DataFrame({
            'content': ['', '', '']
        })
        result = analyzer.generate_features(data, text_col="content")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert all(result['sentiment_score'] == 0.0)

    def test_generate_features_with_na_values(self):
        """测试生成特征（包含NA值）"""
        analyzer = SentimentAnalyzer()
        data = pd.DataFrame({
            'content': ['积极消息', None, '消极消息', pd.NA]
        })
        result = analyzer.generate_features(data, text_col="content")
        assert isinstance(result, pd.DataFrame)
        # 应该只处理有效文本
        assert len(result) <= 2

    def test_analyze_positive_text(self):
        """测试分析正面文本"""
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze("这是一个积极正面的消息", threshold=0.5)
        assert isinstance(result, dict)
        assert 'score' in result
        assert 'label' in result
        assert result['label'] == "positive"

    def test_analyze_negative_text(self):
        """测试分析负面文本"""
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze("这是一个消极负面的消息", threshold=0.5)
        assert isinstance(result, dict)
        assert 'score' in result
        assert 'label' in result
        assert result['label'] == "negative"

    def test_analyze_neutral_text(self):
        """测试分析中性文本"""
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze("这是一个普通消息", threshold=0.5)
        assert isinstance(result, dict)
        assert 'score' in result
        assert 'label' in result

    def test_analyze_invalid_type(self):
        """测试分析无效类型"""
        analyzer = SentimentAnalyzer()
        with pytest.raises(TypeError, match="text must be str"):
            analyzer.analyze(123)

    def test_batch_analyze_valid(self):
        """测试批量分析有效文本"""
        analyzer = SentimentAnalyzer()
        texts = ["积极消息", "消极消息", "中性消息"]
        results = analyzer.batch_analyze(texts)
        assert isinstance(results, list)
        assert len(results) == 3
        for result in results:
            assert isinstance(result, dict)
            assert 'score' in result
            assert 'label' in result

    def test_batch_analyze_invalid_type(self):
        """测试批量分析无效类型"""
        analyzer = SentimentAnalyzer()
        with pytest.raises(TypeError, match="texts must be list"):
            analyzer.batch_analyze("not a list")

    def test_batch_analyze_invalid_element_type(self):
        """测试批量分析无效元素类型"""
        analyzer = SentimentAnalyzer()
        with pytest.raises(TypeError, match="all elements in texts must be str"):
            analyzer.batch_analyze(["text1", 123, "text3"])

    def test_batch_analyze_with_batch_size(self):
        """测试批量分析（指定batch_size）"""
        analyzer = SentimentAnalyzer()
        texts = ["文本1", "文本2", "文本3", "文本4", "文本5"]
        results = analyzer.batch_analyze(texts, batch_size=2)
        assert isinstance(results, list)
        assert len(results) == 5

    def test_generate_features_positive_keywords(self):
        """测试生成特征（正面关键词）"""
        analyzer = SentimentAnalyzer()
        data = pd.DataFrame({
            'content': ['这是一个优秀的成功案例，增长很快']
        })
        result = analyzer.generate_features(data, text_col="content")
        assert len(result) == 1
        assert result.iloc[0]['sentiment_label'] == "positive"

    def test_generate_features_negative_keywords(self):
        """测试生成特征（负面关键词）"""
        analyzer = SentimentAnalyzer()
        data = pd.DataFrame({
            'content': ['这是一个失败的案例，亏损严重']
        })
        result = analyzer.generate_features(data, text_col="content")
        assert len(result) == 1
        assert result.iloc[0]['sentiment_label'] == "negative"

    def test_generate_features_custom_text_col(self):
        """测试生成特征（自定义文本列）"""
        analyzer = SentimentAnalyzer()
        data = pd.DataFrame({
            'text': ['测试文本内容']
        })
        result = analyzer.generate_features(data, text_col="text")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1


class TestBaseSentimentAnalyzer:
    """BaseSentimentAnalyzer测试"""

    def test_base_sentiment_analyzer_initialization(self):
        """测试基础分析器初始化"""
        analyzer = BaseSentimentAnalyzer()
        # 这是一个空壳类，应该可以实例化
        assert analyzer is not None

