import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from models.model_explainability import (
    SHAPExplainer,
    LIMEExplainer,
    FeatureImportanceExplainer,
    ModelExplainer,
    ModelExplanationSystem,
    ExplanationMethod
)

@pytest.fixture
def sample_data():
    """生成测试数据"""
    X = pd.DataFrame({
        'feature1': np.random.rand(10),
        'feature2': np.random.rand(10),
        'feature3': np.random.rand(10)
    })
    return X

@pytest.fixture
def mock_model():
    """创建模拟模型"""
    model = MagicMock()
    model.predict.return_value = np.array([1, 0, 1])
    model.predict_proba.return_value = np.array([[0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
    model.feature_importances_ = np.array([0.1, 0.3, 0.6])
    return model

def test_shap_explainer(mock_model, sample_data):
    """测试SHAP解释器"""
    explainer = SHAPExplainer(mock_model, ['f1', 'f2', 'f3'])

    # 测试解释生成
    explanation = explainer.explain(sample_data)

    assert 'values' in explanation
    assert 'base_values' in explanation
    assert len(explanation['feature_names']) == 3

    # 测试可视化(不实际显示)
    explainer.visualize(explanation)

def test_lime_explainer(mock_model, sample_data):
    """测试LIME解释器"""
    explainer = LIMEExplainer(mock_model, ['f1', 'f2', 'f3'])

    # 测试解释生成
    explanation = explainer.explain(sample_data.iloc[:3])

    assert 'explanations' in explanation
    assert len(explanation['explanations']) == 3
    assert all('features' in exp for exp in explanation['explanations'])

    # 测试可视化(不实际显示)
    explainer.visualize(explanation)

def test_feature_importance_explainer(mock_model):
    """测试特征重要性解释器"""
    explainer = FeatureImportanceExplainer(mock_model, ['f1', 'f2', 'f3'])

    # 测试解释生成
    explanation = explainer.explain()

    assert 'importances' in explanation
    assert len(explanation['importances']) == 3
    assert explanation['importances'][2] == 0.6  # 最高重要性

    # 测试可视化(不实际显示)
    explainer.visualize(explanation)

def test_model_explainer(mock_model, sample_data):
    """测试模型解释统一接口"""
    explainer = ModelExplainer(mock_model, ['f1', 'f2', 'f3'])

    # 测试SHAP解释
    shap_exp = explainer.explain(sample_data, method=ExplanationMethod.SHAP)
    assert 'values' in shap_exp

    # 测试LIME解释
    lime_exp = explainer.explain(sample_data.iloc[:1], method=ExplanationMethod.LIME)
    assert 'explanations' in lime_exp

    # 测试特征重要性
    fi_exp = explainer.explain(method=ExplanationMethod.FEATURE_IMPORTANCE)
    assert 'importances' in fi_exp

    # 测试特征贡献度
    contribs = explainer.get_feature_contributions(sample_data)
    assert isinstance(contribs, pd.DataFrame)
    assert contribs.iloc[0]['contribution'] == pytest.approx(0.6)  # 最高贡献

def test_explanation_system(mock_model, sample_data):
    """测试模型解释系统"""
    config = {
        'model1': {
            'model': mock_model,
            'feature_names': ['f1', 'f2', 'f3']
        }
    }

    system = ModelExplanationSystem(config)

    # 测试解释生成
    exp = system.explain_model('model1', sample_data, 'shap')
    assert 'values' in exp

    # 测试添加模型
    new_model = MagicMock()
    new_model.feature_importances_ = np.array([0.4, 0.3, 0.3])
    system.add_model('model2', new_model, ['f1', 'f2', 'f3'])

    # 测试特征贡献度
    contribs = system.get_model_contributions('model2', sample_data, 'feature_importance')
    assert contribs.iloc[0]['feature'] == 'f1'
    assert contribs.iloc[0]['contribution'] == pytest.approx(0.4)

def test_invalid_method(mock_model, sample_data):
    """测试无效解释方法"""
    explainer = ModelExplainer(mock_model, ['f1', 'f2', 'f3'])

    with pytest.raises(ValueError):
        explainer.explain(sample_data, method='invalid_method')

def test_missing_feature_importance(mock_model):
    """测试缺失特征重要性"""
    mock_model.feature_importances_ = None
    explainer = FeatureImportanceExplainer(mock_model, ['f1', 'f2', 'f3'])

    with pytest.raises(ValueError):
        explainer.explain()

def test_visualization_with_invalid_index(mock_model, sample_data):
    """测试无效索引可视化"""
    explainer = LIMEExplainer(mock_model, ['f1', 'f2', 'f3'])
    explanation = explainer.explain(sample_data.iloc[:1])

    with pytest.raises(IndexError):
        explainer.visualize(explanation, sample_idx=1)

def test_model_not_found(sample_data):
    """测试模型不存在情况"""
    system = ModelExplanationSystem({})

    with pytest.raises(ValueError):
        system.explain_model('nonexistent', sample_data)

def test_contributions_sorting(mock_model, sample_data):
    """测试贡献度排序"""
    explainer = ModelExplainer(mock_model, ['f1', 'f2', 'f3'])
    contribs = explainer.get_feature_contributions(sample_data, 'feature_importance')

    assert contribs['contribution'].is_monotonic_decreasing
