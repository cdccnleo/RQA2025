import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from src.ensemble.ensemble_predictor import (
    AverageEnsemble,
    StackingEnsemble,
    BayesianModelAveraging,
    DynamicWeightedEnsemble,
    EnsembleResult,
    ModelPrediction,
    EnsembleMethod
)

@pytest.fixture
def sample_models():
    """生成模拟模型"""
    models = {
        'lstm': MagicMock(),
        'rf': MagicMock(),
        'nn': MagicMock()
    }
    # 设置各模型预测结果
    models['lstm'].predict.return_value = np.array([0.8, 0.7, 0.6])
    models['rf'].predict.return_value = np.array([0.7, 0.6, 0.5])
    models['nn'].predict.return_value = np.array([0.9, 0.8, 0.7])
    return models

@pytest.fixture
def sample_X():
    """生成测试数据"""
    return pd.DataFrame({
        'feature1': [0.1, 0.2, 0.3],
        'feature2': [0.4, 0.5, 0.6]
    })

@pytest.fixture
def sample_y():
    """生成测试标签"""
    return pd.Series([1, 0, 1])

def test_average_ensemble(sample_models, sample_X):
    """测试简单平均集成"""
    ensemble = AverageEnsemble(sample_models)
    result = ensemble.predict(sample_X)

    assert isinstance(result, EnsembleResult)
    assert np.allclose(result.prediction, [0.8, 0.7, 0.6], atol=0.1)
    assert 'lstm' in result.weights
    assert result.uncertainty is not None

def test_stacking_ensemble(sample_models, sample_X, sample_y):
    """测试堆叠集成"""
    # 模拟元模型
    meta_model = MagicMock()
    meta_model.coef_ = np.array([0.4, 0.3, 0.3])
    meta_model.predict.return_value = np.array([0.75, 0.65, 0.55])

    ensemble = StackingEnsemble(sample_models, meta_model)
    ensemble.fit(sample_X, sample_y)
    result = ensemble.predict(sample_X)

    assert isinstance(result, EnsembleResult)
    assert np.allclose(result.prediction, [0.75, 0.65, 0.55])
    assert result.weights['lstm'] == pytest.approx(0.4)

def test_bayesian_ensemble(sample_models, sample_X, sample_y):
    """测试贝叶斯模型平均"""
    ensemble = BayesianModelAveraging(sample_models)
    ensemble.fit(sample_X, sample_y)
    result = ensemble.predict(sample_X)

    assert isinstance(result, EnsembleResult)
    assert len(result.weights) == 3
    assert sum(result.weights.values()) == pytest.approx(1.0)

def test_dynamic_ensemble(sample_models, sample_X, sample_y):
    """测试动态加权集成"""
    ensemble = DynamicWeightedEnsemble(sample_models, lookback=2)
    ensemble.fit(sample_X, sample_y)
    result = ensemble.predict(sample_X)

    assert isinstance(result, EnsembleResult)
    assert len(result.weights) == 3
    assert sum(result.weights.values()) == pytest.approx(1.0)

def test_dynamic_ensemble_not_fitted(sample_models, sample_X):
    """测试未拟合的动态集成"""
    ensemble = DynamicWeightedEnsemble(sample_models)
    with pytest.raises(ValueError):
        ensemble.predict(sample_X)

def test_ensemble_visualization():
    """测试集成可视化"""
    from src.ensemble.ensemble_predictor import EnsembleVisualizer

    # 测试权重图
    weights = {'lstm': 0.4, 'rf': 0.3, 'nn': 0.3}
    fig = EnsembleVisualizer.plot_weights(weights)
    assert fig is not None

    # 测试不确定性图
    pred = np.array([0.8, 0.7, 0.6])
    uncertainty = np.array([0.1, 0.05, 0.15])
    fig = EnsembleVisualizer.plot_uncertainty(pred, uncertainty)
    assert fig is not None

    # 测试贡献图
    predictions = {
        'lstm': np.array([0.8, 0.7, 0.6]),
        'rf': np.array([0.7, 0.6, 0.5]),
        'nn': np.array([0.9, 0.8, 0.7])
    }
    final_pred = np.array([0.8, 0.7, 0.6])
    fig = EnsembleVisualizer.plot_contributions(predictions, final_pred)
    assert fig is not None

def test_average_ensemble_empty_models(sample_X):
    """测试空模型输入"""
    with pytest.raises(ValueError):
        AverageEnsemble({}).predict(sample_X)

def test_stacking_ensemble_fit_predict_consistency(sample_models, sample_X, sample_y):
    """测试堆叠集成拟合预测一致性"""
    ensemble = StackingEnsemble(sample_models)
    ensemble.fit(sample_X, sample_y)
    result1 = ensemble.predict(sample_X)
    result2 = ensemble.predict(sample_X)

    assert np.allclose(result1.prediction, result2.prediction)
