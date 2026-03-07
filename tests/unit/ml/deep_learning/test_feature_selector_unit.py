import pandas as pd

from src.ml.deep_learning.feature_selector import (
    FeatureSelector,
    AdvancedFeatureSelector,
    select_features_auto,
    select_features_univariate,
    select_features_model_based
)


def test_feature_selector_selects_numeric_columns():
    data = pd.DataFrame(
        {
            "num1": [1, 2],
            "num2": [3, 4],
            "category": ["a", "b"],
        }
    )
    selector = FeatureSelector(top_k=1)
    result = selector.select(data)
    assert result == ["num1"]


def test_feature_selector_handles_small_dataset():
    data = pd.DataFrame({"value": [1]})
    selector = FeatureSelector(top_k=5)
    assert selector.select(data) == ["value"]


def test_advanced_feature_selector_correlation_method():
    """测试高级特征选择器的相关性方法"""
    # 创建具有高度相关特征的数据
    data = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [1.1, 2.1, 3.1, 4.1, 5.1],  # 与feature1高度相关
        "feature3": [0.5, 1.5, 2.5, 3.5, 4.5],  # 与feature1相关但不高
        "category": ["a", "b", "c", "d", "e"]  # 非数值列
    })

    selector = AdvancedFeatureSelector(method="correlation", threshold=0.9, top_k=2)
    result = selector.select(data)

    # 应该移除高度相关的feature2
    assert "feature1" in result
    assert "feature2" not in result  # 被移除
    assert "feature3" in result
    assert len(result) <= 2  # 限制top_k


def test_advanced_feature_selector_default_method():
    """测试高级特征选择器的默认方法"""
    data = pd.DataFrame({
        "num1": [1, 2, 3],
        "num2": [4, 5, 6],
        "text": ["a", "b", "c"]
    })

    selector = AdvancedFeatureSelector(method="unknown", top_k=1)
    result = selector.select(data)

    # 未知方法应该使用父类方法
    assert result == ["num1"]


def test_advanced_feature_selector_empty_data():
    """测试高级特征选择器空数据处理"""
    data = pd.DataFrame()  # 空数据框

    selector = AdvancedFeatureSelector(method="correlation")
    result = selector.select(data)
    assert result == []


def test_advanced_feature_selector_no_numeric_columns():
    """测试高级特征选择器无数值列的情况"""
    data = pd.DataFrame({
        "text1": ["a", "b", "c"],
        "text2": ["x", "y", "z"]
    })

    selector = AdvancedFeatureSelector(method="correlation")
    result = selector.select(data)
    assert result == []


def test_select_features_auto():
    """测试自动特征选择函数"""
    data = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [1.1, 2.1, 3.1, 4.1, 5.1],
        "text": ["a", "b", "c", "d", "e"]
    })

    result = select_features_auto(data, method="correlation")
    assert isinstance(result, list)
    assert len(result) > 0
    # 应该包含数值列
    assert any(col in result for col in ["feature1", "feature2"])


def test_select_features_univariate():
    """测试单变量特征选择函数"""
    data = pd.DataFrame({
        "num1": [1, 2, 3],
        "num2": [4, 5, 6],
        "num3": [7, 8, 9],
        "text": ["a", "b", "c"]
    })

    result = select_features_univariate(data, target="num1", k=2)
    assert isinstance(result, list)
    assert len(result) == 2
    assert "num1" in result
    assert "num2" in result


def test_select_features_model_based():
    """测试基于模型的特征选择函数"""
    data = pd.DataFrame({
        "feature1": [1, 2, 3, 4],
        "feature2": [0.1, 0.2, 0.3, 0.4],
        "feature3": [10, 20, 30, 40],
        "text": ["a", "b", "c", "d"]
    })

    result = select_features_model_based(data, target="feature1", k=3)
    assert isinstance(result, list)
    assert len(result) == 3
    # 应该只包含数值列
    assert all(col in ["feature1", "feature2", "feature3"] for col in result)

