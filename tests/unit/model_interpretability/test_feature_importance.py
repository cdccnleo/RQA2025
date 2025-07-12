import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import matplotlib.pyplot as plt
from features.feature_importance import (
    FeatureImportanceAnalyzer
)

@pytest.fixture
def sample_data():
    """创建测试数据"""
    X = np.random.rand(100, 5)  # 100 samples, 5 features
    y = np.random.randint(0, 2, 100)  # binary target
    feature_names = [f"feature_{i}" for i in range(5)]
    return X, y, feature_names

@pytest.fixture
def mock_model():
    """创建模拟模型"""
    model = MagicMock()
    model.predict.return_value = np.random.randint(0, 2, 100)
    model.score.return_value = 0.85
    return model

@pytest.fixture
def analyzer(mock_model):
    """创建分析器实例"""
    return FeatureImportanceAnalyzer(mock_model)

def test_permutation_importance(analyzer, sample_data):
    """测试排列重要性计算"""
    X, y, feature_names = sample_data

    # 模拟permutation_importance结果
    mock_result = MagicMock()
    mock_result.importances_mean = np.array([0.1, 0.3, 0.2, 0.05, 0.15])
    mock_result.importances_std = np.array([0.01, 0.02, 0.015, 0.005, 0.01])

    with patch('sklearn.inspection.permutation_importance',
               return_value=mock_result) as mock_pi:
        importance = analyzer.calculate_permutation_importance(
            X, y, feature_names
        )

        # 验证调用
        mock_pi.assert_called_once()

        # 验证结果
        assert isinstance(importance, dict)
        assert len(importance) == 5
        assert importance["feature_1"] == 0.3  # 第二个特征重要性最高

        # 验证标准差
        assert analyzer.importance_std["feature_1"] == 0.02

def test_get_top_features(analyzer, sample_data):
    """测试获取重要特征"""
    X, y, feature_names = sample_data

    # 模拟permutation_importance结果
    mock_result = MagicMock()
    mock_result.importances_mean = np.array([0.1, 0.3, 0.2, 0.05, 0.15])
    mock_result.importances_std = np.array([0.01, 0.02, 0.015, 0.005, 0.01])

    with patch('sklearn.inspection.permutation_importance',
               return_value=mock_result):
        analyzer.calculate_permutation_importance(X, y, feature_names)

        # 获取top 3特征
        top_features = analyzer.get_top_features(3)

        # 验证结果
        assert len(top_features) == 3
        assert top_features == ["feature_1", "feature_2", "feature_4"]

def test_plot_importance(analyzer, sample_data):
    """测试重要性绘图"""
    X, y, feature_names = sample_data

    # 模拟permutation_importance结果
    mock_result = MagicMock()
    mock_result.importances_mean = np.array([0.1, 0.3, 0.2, 0.05, 0.15])
    mock_result.importances_std = np.array([0.01, 0.02, 0.015, 0.005, 0.01])

    with patch('sklearn.inspection.permutation_importance',
               return_value=mock_result):
        analyzer.calculate_permutation_importance(X, y, feature_names)

        # 测试绘图
        fig = analyzer.plot_importance(top_n=3)

        # 验证返回对象
        assert isinstance(fig, plt.Figure)

        # 验证图表内容
        ax = fig.axes[0]
        yticklabels = [t.get_text() for t in ax.get_yticklabels()]
        assert "feature_1" in yticklabels
        assert "feature_0" not in yticklabels  # 不在top3中

        plt.close(fig)

@patch.dict('sys.modules', {'shap': MagicMock()})
def test_shap_values(analyzer, sample_data):
    """测试SHAP值计算"""
    X, y, feature_names = sample_data

    # 模拟shap返回值
    mock_shap_values = np.random.rand(100, 5)

    with patch('shap.TreeExplainer') as mock_explainer:
        mock_explainer.return_value.shap_values.return_value = mock_shap_values

        shap_results = analyzer.calculate_shap_values(
            X, feature_names
        )

        # 验证调用
        mock_explainer.assert_called_once_with(analyzer.model)

        # 验证结果
        assert isinstance(shap_results, dict)
        assert len(shap_results) == 5
        assert "feature_1" in shap_results
        assert shap_results["feature_1"].shape == (100,)

@patch.dict('sys.modules', {'shap': MagicMock()})
def test_shap_summary_plot(analyzer, sample_data):
    """测试SHAP摘要图"""
    X, y, feature_names = sample_data

    # 模拟shap返回值
    mock_shap_values = np.random.rand(100, 5)

    with patch('shap.TreeExplainer') as mock_explainer, \
         patch('shap.summary_plot') as mock_summary_plot:
        mock_explainer.return_value.shap_values.return_value = mock_shap_values

        # 测试绘图
        fig = analyzer.plot_shap_summary(X, feature_names)

        # 验证调用
        mock_summary_plot.assert_called_once()

        # 验证返回对象
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

def test_error_handling(analyzer, sample_data):
    """测试错误处理"""
    X, y, feature_names = sample_data

    # 未计算重要性时获取top特征
    with pytest.raises(RuntimeError, match="请先计算特征重要性"):
        analyzer.get_top_features()

    # 特征数量不匹配
    with pytest.raises(ValueError, match="特征数量与名称数量不匹配"):
        analyzer.calculate_permutation_importance(
            X, y, feature_names[:3]  # 只提供3个特征名
        )

    # SHAP包未安装
    with patch.dict('sys.modules', {'shap': None}):
        with pytest.raises(ImportError, match="请先安装shap包"):
            analyzer.calculate_shap_values(X, feature_names)
