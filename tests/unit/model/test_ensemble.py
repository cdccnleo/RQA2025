import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from models.ensemble import (
    ModelWeight,
    ModelEnsemble,
    EnsembleManager
)

@pytest.fixture
def sample_models():
    """创建模拟模型"""
    model1 = MagicMock()
    model1.predict.return_value = np.array([0.6, 0.7, 0.8])

    model2 = MagicMock()
    model2.predict.return_value = np.array([0.4, 0.3, 0.2])

    return [
        ModelWeight(name="model1", weight=0.6, model=model1),
        ModelWeight(name="model2", weight=0.4, model=model2)
    ]

@pytest.fixture
def sample_data():
    """生成测试数据"""
    return pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6]
    })

@pytest.fixture
def sample_labels():
    """生成测试标签"""
    return np.array([1, 0, 1])

def test_weighted_predict(sample_models, sample_data):
    """测试加权组合预测"""
    ensemble = ModelEnsemble(sample_models)
    pred = ensemble.weighted_predict(sample_data)

    # 验证加权组合计算
    expected = np.array([0.6*0.6 + 0.4*0.4,
                        0.6*0.7 + 0.4*0.3,
                        0.6*0.8 + 0.4*0.2])
    assert np.allclose(pred, expected)

    # 验证模型调用
    for model in sample_models:
        model.model.predict.assert_called_once_with(sample_data)

def test_stacked_predict(sample_models, sample_data, sample_labels):
    """测试堆叠集成预测"""
    ensemble = ModelEnsemble(sample_models)

    # 首次调用训练元模型
    pred1 = ensemble.stacked_predict(sample_data, sample_labels)
    assert pred1.shape == (3,)

    # 验证基模型调用
    for model in sample_models:
        model.model.predict.assert_called_once_with(sample_data)

    # 后续调用使用已训练元模型
    pred2 = ensemble.stacked_predict(sample_data)
    assert np.array_equal(pred1, pred2)

def test_evaluate(sample_models, sample_data, sample_labels):
    """测试模型评估"""
    ensemble = ModelEnsemble(sample_models)

    # 测试加权评估
    weighted_metrics = ensemble.evaluate(sample_data, sample_labels, 'weighted')
    assert 'accuracy' in weighted_metrics
    assert 0 <= weighted_metrics['accuracy'] <= 1

    # 测试堆叠评估
    stacked_metrics = ensemble.evaluate(sample_data, sample_labels, 'stacked')
    assert 'accuracy' in stacked_metrics
    assert 0 <= stacked_metrics['accuracy'] <= 1

def test_explain(sample_models, sample_data):
    """测试模型解释"""
    # 设置模型特征重要性
    sample_models[0].model.feature_importances_ = np.array([0.7, 0.3])
    sample_models[1].model.feature_importances_ = np.array([0.4, 0.6])

    ensemble = ModelEnsemble(sample_models)

    # 测试加权解释
    weighted_exp = ensemble.explain(sample_data, 'weighted')
    assert len(weighted_exp) == 2
    assert np.isclose(weighted_exp['feature1'], 0.6*0.7 + 0.4*0.4)
    assert np.isclose(weighted_exp['feature2'], 0.6*0.3 + 0.4*0.6)

    # 测试堆叠解释
    stacked_exp = ensemble.explain(sample_data, 'stacked')
    assert len(stacked_exp) == 2

def test_ensemble_manager():
    """测试集成模型管理器"""
    manager = EnsembleManager()

    # 测试添加和获取
    ensemble = ModelEnsemble([])
    manager.add_ensemble('test', ensemble)
    assert manager.get_ensemble('test') == ensemble

    # 测试列表
    assert 'test' in manager.list_ensembles()

    # 测试移除
    manager.remove_ensemble('test')
    assert manager.get_ensemble('test') is None

def test_weight_validation():
    """测试权重验证"""
    model = MagicMock()

    # 测试有效权重
    ModelEnsemble([ModelWeight('m1', 0.6, model),
                  ModelWeight('m2', 0.4, model)])

    # 测试无效权重
    with pytest.raises(ValueError):
        ModelEnsemble([ModelWeight('m1', 0.6, model),
                      ModelWeight('m2', 0.5, model)])

@pytest.mark.parametrize("method", ['weighted', 'stacked'])
def test_evaluate_methods(method, sample_models, sample_data, sample_labels):
    """测试不同评估方法"""
    ensemble = ModelEnsemble(sample_models)
    metrics = ensemble.evaluate(sample_data, sample_labels, method)
    assert 'accuracy' in metrics
