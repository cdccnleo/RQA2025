#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sentiment/analyzer补充测试覆盖
针对未覆盖的代码分支编写测试
"""

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from src.features.sentiment.analyzer import SentimentAnalyzer, BaseSentimentAnalyzer


class TestSentimentAnalyzerCoverageSupplement:
    """sentiment/analyzer补充测试"""

    def test_analyze_text_type_error(self):
        """测试analyze_text类型错误"""
        analyzer = SentimentAnalyzer()
        
        with pytest.raises(TypeError, match="text must be str"):
            analyzer.analyze_text(123)

    def test_analyze_batch_type_error_in_list(self):
        """测试analyze_batch列表中元素类型错误"""
        analyzer = SentimentAnalyzer()
        
        with pytest.raises(TypeError, match="all elements in texts must be str"):
            analyzer.analyze_batch(['text1', 123, 'text2'])

    def test_preprocess_type_error(self):
        """测试_preprocess类型错误"""
        analyzer = SentimentAnalyzer()
        
        with pytest.raises(TypeError, match="text must be str"):
            analyzer._preprocess(123)

    def test_register_features_with_feature_manager(self):
        """测试register_features（有feature_manager）"""
        feature_manager = MagicMock()
        analyzer = SentimentAnalyzer(feature_manager=feature_manager)
        
        analyzer.register_features()
        
        # 验证register被调用
        feature_manager.register.assert_called_once()
        call_args = feature_manager.register.call_args
        assert call_args[1]['name'] == 'sentiment'
        assert call_args[1]['calculator'] == analyzer.analyze_text
        assert 'description' in call_args[1]

    def test_register_features_without_feature_manager(self):
        """测试register_features（无feature_manager）"""
        analyzer = SentimentAnalyzer()
        
        # 不应该抛出异常
        analyzer.register_features()

    def test_generate_features_empty_dataframe(self):
        """测试generate_features（空DataFrame）"""
        analyzer = SentimentAnalyzer()
        
        result = analyzer.generate_features(pd.DataFrame())
        assert result.empty

    def test_generate_features_none_data(self):
        """测试generate_features（None数据）"""
        analyzer = SentimentAnalyzer()
        
        result = analyzer.generate_features(None)
        assert result.empty

    def test_generate_features_missing_column(self):
        """测试generate_features（列不存在）"""
        analyzer = SentimentAnalyzer()
        data = pd.DataFrame({'other_col': ['text1', 'text2']})
        
        with pytest.raises(ValueError, match="列 content 不存在"):
            analyzer.generate_features(data, text_col='content')

    def test_generate_features_no_valid_texts(self):
        """测试generate_features（无有效文本）"""
        analyzer = SentimentAnalyzer()
        data = pd.DataFrame({'content': [None, None, None]})
        
        result = analyzer.generate_features(data)
        assert result.empty

    def test_generate_features_empty_string(self):
        """测试generate_features（空字符串）"""
        analyzer = SentimentAnalyzer()
        data = pd.DataFrame({'content': ['', 'text1', 'text2']})
        
        result = analyzer.generate_features(data)
        assert len(result) == 3
        assert result.iloc[0]['sentiment_score'] == 0.0
        assert result.iloc[0]['sentiment_label'] == 'neutral'

    def test_generate_features_positive_sentiment(self):
        """测试generate_features（正面情感）"""
        analyzer = SentimentAnalyzer()
        data = pd.DataFrame({'content': ['这是一个积极正面的消息，增长上涨']})
        
        result = analyzer.generate_features(data)
        assert result.iloc[0]['sentiment_score'] == 0.8
        assert result.iloc[0]['sentiment_label'] == 'positive'

    def test_generate_features_negative_sentiment(self):
        """测试generate_features（负面情感）"""
        analyzer = SentimentAnalyzer()
        data = pd.DataFrame({'content': ['这是一个消极负面的消息，失败下降亏损']})
        
        result = analyzer.generate_features(data)
        assert result.iloc[0]['sentiment_score'] == -0.2
        assert result.iloc[0]['sentiment_label'] == 'negative'

    def test_generate_features_neutral_sentiment(self):
        """测试generate_features（中性情感）"""
        analyzer = SentimentAnalyzer()
        data = pd.DataFrame({'content': ['这是一个普通的消息']})
        
        result = analyzer.generate_features(data)
        assert result.iloc[0]['sentiment_score'] == 0.5
        assert result.iloc[0]['sentiment_label'] == 'positive'

    def test_analyze_type_error(self):
        """测试analyze类型错误"""
        analyzer = SentimentAnalyzer()
        
        with pytest.raises(TypeError, match="text must be str"):
            analyzer.analyze(123)

    def test_analyze_positive(self):
        """测试analyze（正面情感）"""
        analyzer = SentimentAnalyzer()
        
        result = analyzer.analyze('这是一个积极正面的消息，增长上涨')
        assert result['score'] == 0.7
        assert result['label'] == 'positive'

    def test_analyze_negative(self):
        """测试analyze（负面情感）"""
        analyzer = SentimentAnalyzer()
        
        result = analyzer.analyze('这是一个消极负面的消息，失败下降亏损')
        assert result['score'] == -0.3
        assert result['label'] == 'negative'

    def test_analyze_neutral(self):
        """测试analyze（中性情感）"""
        analyzer = SentimentAnalyzer()
        
        result = analyzer.analyze('这是一个普通的消息')
        assert result['score'] == 0.5
        assert result['label'] == 'positive'

    def test_batch_analyze_type_error_not_list(self):
        """测试batch_analyze类型错误（不是列表）"""
        analyzer = SentimentAnalyzer()
        
        with pytest.raises(TypeError, match="texts must be list"):
            analyzer.batch_analyze('not a list')

    def test_batch_analyze_type_error_in_list(self):
        """测试batch_analyze类型错误（列表中元素不是字符串）"""
        analyzer = SentimentAnalyzer()
        
        with pytest.raises(TypeError, match="all elements in texts must be str"):
            analyzer.batch_analyze(['text1', 123, 'text2'])

    def test_batch_analyze_success(self):
        """测试batch_analyze成功"""
        analyzer = SentimentAnalyzer()
        
        texts = ['正面消息', '负面消息', '中性消息']
        results = analyzer.batch_analyze(texts)
        
        assert len(results) == 3
        assert all('score' in r and 'label' in r for r in results)

    def test_batch_analyze_with_batch_size(self):
        """测试batch_analyze（指定batch_size）"""
        analyzer = SentimentAnalyzer()
        
        texts = ['text1', 'text2', 'text3']
        results = analyzer.batch_analyze(texts, batch_size=2)
        
        assert len(results) == 3

    def test_base_sentiment_analyzer(self):
        """测试BaseSentimentAnalyzer（空类）"""
        analyzer = BaseSentimentAnalyzer()
        assert analyzer is not None

    def test_snownlp_sentiment(self):
        """测试snownlp_sentiment方法"""
        analyzer = SentimentAnalyzer()
        
        result = analyzer.snownlp_sentiment('test text')
        assert result == 0.5

    def test_preprocess_with_newlines(self):
        """测试_preprocess（包含换行符）"""
        analyzer = SentimentAnalyzer()
        
        text = "line1\nline2\rline3"
        result = analyzer._preprocess(text)
        assert '\n' not in result
        assert '\r' not in result

    def test_preprocess_with_whitespace(self):
        """测试_preprocess（包含前后空白）"""
        analyzer = SentimentAnalyzer()
        
        text = "  test text  "
        result = analyzer._preprocess(text)
        assert result == "test text"

