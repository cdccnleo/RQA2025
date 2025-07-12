import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from src.features.processors.feature_selector import FeatureSelector

@pytest.mark.usefixtures("sample_features")
class TestFeatureSelector:
    """FeatureSelector特征选择测试"""

    @pytest.fixture
    def empty_df(self, sample_features):
        return sample_features.iloc[:0]

    @pytest.fixture
    def no_features_df(self, sample_features):
        return sample_features[[]]

    @pytest.mark.parametrize("strategy,expected", [
        ("variance", ["close", "sentiment"]),
        ("correlation", ["close", "trend"]),
        ("importance", ["sentiment", "close"])
    ])
    def test_selection_strategies(self, strategy, expected, sample_features):
        """测试不同选择策略"""
        selector = FeatureSelector(strategy=strategy)

        # 模拟特征重要性
        with patch.object(selector, '_get_feature_importance') as mock_imp:
            mock_imp.return_value = pd.Series({
                'close': 0.3,
                'volume': 0.1,
                'sentiment': 0.4,
                'trend': 0.2
            })

            result = selector.select(sample_features)
            assert set(result.columns) == set(expected)

    def test_constant_feature_removal(self, sample_features):
        """测试常量特征过滤"""
        df = sample_features.copy()
        df['constant'] = 1.0  # 添加常量特征

        selector = FeatureSelector(strategy="variance")
        result = selector.select(df)
        assert 'constant' not in result.columns

    def test_high_correlation_removal(self, sample_features):
        """测试高相关性特征过滤"""
        df = sample_features.copy()
        df['close_copy'] = df['close'] * 1.01  # 添加高相关特征

        selector = FeatureSelector(strategy="correlation", threshold=0.9)
        result = selector.select(df)
        assert 'close_copy' not in result.columns

    def test_custom_selection_function(self, sample_features):
        """测试自定义选择函数"""
        def custom_selector(df):
            return df[['sentiment']]

        selector = FeatureSelector(custom_strategy=custom_selector)
        result = selector.select(sample_features)
        assert list(result.columns) == ['sentiment']

    @pytest.mark.parametrize("invalid_input", [
        "empty_df",  # 0行数据
        "no_features_df",  # 无特征列
        pd.DataFrame(),  # 空DataFrame
    ])
    def test_invalid_input_handling(self, request, invalid_input):
        """测试异常输入处理"""
        selector = FeatureSelector()
        if isinstance(invalid_input, str):
            invalid_input = request.getfixturevalue(invalid_input)
        with pytest.raises(ValueError, match="No valid features"):
            selector.select(invalid_input)
