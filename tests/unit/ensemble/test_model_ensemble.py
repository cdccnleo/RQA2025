import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from src.ensemble.model_ensemble import (
    WeightedEnsemble,
    EnsembleResult,
    EnsembleMethod,
    WeightUpdateRule,
    EnsembleMonitor
)

@pytest.fixture
def sample_predictions():
    """生成测试预测数据"""
    return {
        'model1': np.array([0.9, 0.8, 0.3, 0.1]),
        'model2': np.array([0.7, 0.6, 0.4, 0.2]),
        'model3': np.array([0.8, 0.7, 0.2, 0.3])
    }

@pytest.fixture
def sample_labels():
    """生成测试标签数据"""
    return np.array([1, 1, 0, 0])

def test_weighted_ensemble(sample_predictions, sample_labels):
    """测试加权集成"""
    ensemble = WeightedEnsemble(
        update_rule=WeightUpdateRule.PERFORMANCE,
        decay_factor=0.9
    )

    # 首次预测
    result = ensemble.predict(sample_predictions, sample_labels)
    assert isinstance(result, EnsembleResult)
    assert len(result.model_weights) == 3
    assert pytest.approx(sum(result.model_weights.values())) == 1.0
    assert 'accuracy' in result.performance_metrics
    assert result.uncertainty is not None

    # 第二次预测
    new_predictions = {
        'model1': np.array([0.85, 0.75, 0.25, 0.15]),
        'model2': np.array([0.65, 0.55, 0.35, 0.25]),
        'model3': np.array([0.75, 0.65, 0.15, 0.25])
    }
    new_labels = np.array([1, 1, 0, 0])
    result2 = ensemble.predict(new_predictions, new_labels)

    # 验证权重更新
    assert result2.model_weights != result.model_weights
    assert pytest.approx(sum(result2.model_weights.values())) == 1.0

def test_ensemble_monitor(sample_predictions, sample_labels):
    """测试集成监控"""
    monitor = EnsembleMonitor(list(sample_predictions.keys()))

    # 模拟集成预测
    ensemble_pred = np.mean(list(sample_predictions.values()), axis=0)

    # 更新监控
    monitor.update(sample_predictions, sample_labels, ensemble_pred)

    # 验证数据记录
    for name in sample_predictions:
        assert len(monitor.model_performance[name]) == 1
    assert len(monitor.ensemble_performance) == 1
    assert not monitor.correlation_matrix.isnull().any().any()

    # 第二次更新
    new_predictions = {
        'model1': np.array([0.85, 0.75, 0.25, 0.15]),
        'model2': np.array([0.65, 0.55, 0.35, 0.25]),
        'model3': np.array([0.75, 0.65, 0.15, 0.25])
    }
    new_labels = np.array([1, 1, 0, 0])
    new_ensemble = np.mean(list(new_predictions.values()), axis=0)
    monitor.update(new_predictions, new_labels, new_ensemble)

    # 验证数据累积
    for name in new_predictions:
        assert len(monitor.model_performance[name]) == 2
    assert len(monitor.ensemble_performance) == 2

def test_visualization():
    """测试可视化工具"""
    from src.ensemble.model_ensemble import EnsembleVisualizer

    # 测试权重分布图
    weights = {'model1': 0.5, 'model2': 0.3, 'model3': 0.2}
    fig = EnsembleVisualizer.plot_weight_distribution(weights)
    assert fig is not None

    # 测试不确定性图
    pred = np.array([0.8, 0.7, 0.6, 0.5])
    uncertainty = np.array([0.1, 0.15, 0.2, 0.1])
    fig = EnsembleVisualizer.plot_uncertainty(pred, uncertainty)
    assert fig is not None

def test_edge_cases():
    """测试边界情况"""
    # 测试单模型集成
    ensemble = WeightedEnsemble()
    result = ensemble.predict(
        {'model1': np.array([0.9, 0.8, 0.1, 0.2])},
        np.array([1, 1, 0, 0])
    )
    assert pytest.approx(result.model_weights['model1']) == 1.0

    # 测试无标签预测
    ensemble = WeightedEnsemble()
    predictions = {
        'model1': np.array([0.9, 0.8, 0.1, 0.2]),
        'model2': np.array([0.8, 0.7, 0.2, 0.3])
    }
    result = ensemble.predict(predictions)
    assert result.performance_metrics == {}
    assert pytest.approx(sum(result.model_weights.values())) == 1.0
