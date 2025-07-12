import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from src.model.model_ensemble import (
    ModelEnsemble,
    WeightMethod,
    ModelPrediction,
    RiskAwareEnsemble,
    OnlineModelEnsemble
)

@pytest.fixture
def sample_predictions():
    """生成测试预测数据"""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    return {
        "lstm": ModelPrediction(
            model_name="lstm",
            predictions=np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100),
            confidence=np.linspace(0.7, 0.9, 100)
        ),
        "rf": ModelPrediction(
            model_name="rf",
            predictions=np.cos(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100),
            confidence=np.linspace(0.6, 0.8, 100)
        ),
        "nn": ModelPrediction(
            model_name="nn",
            predictions=np.sin(np.linspace(0, 5, 100)) + np.random.normal(0, 0.1, 100),
            confidence=np.linspace(0.5, 0.7, 100)
        )
    }

@pytest.fixture
def sample_actual():
    """生成测试实际值"""
    return np.sin(np.linspace(0, 10, 100)) + 0.5 * np.cos(np.linspace(0, 5, 100))

def test_equal_weight_combination(sample_predictions):
    """测试等权重组合"""
    ensemble = ModelEnsemble(weight_method=WeightMethod.EQUAL)
    combined = ensemble.combine_predictions(sample_predictions)

    assert len(combined) == 100
    assert np.all(combined >= min(p.predictions.min() for p in sample_predictions.values()))
    assert np.all(combined <= max(p.predictions.max() for p in sample_predictions.values()))

def test_optimal_combination(sample_predictions, sample_actual):
    """测试最优组合"""
    ensemble = ModelEnsemble(weight_method=WeightMethod.OPTIMAL)
    combined = ensemble.combine_predictions(sample_predictions, sample_actual)

    assert len(combined) == 100
    assert ensemble.meta_model is not None

def test_dynamic_weight_adjustment(sample_predictions):
    """测试动态权重调整"""
    ensemble = ModelEnsemble(weight_method=WeightMethod.DYNAMIC)
    combined = ensemble.combine_predictions(sample_predictions)

    assert len(combined) == 100
    assert len(ensemble.weights) == 3

def test_risk_parity_combination(sample_predictions):
    """测试风险平价组合"""
    ensemble = ModelEnsemble(weight_method=WeightMethod.RISK_PARITY)
    combined = ensemble.combine_predictions(sample_predictions)

    assert len(combined) == 100
    assert all(w > 0 for w in ensemble.weights.values())

def test_risk_aware_ensemble(sample_predictions):
    """测试风险感知组合"""
    ensemble = RiskAwareEnsemble(risk_target=0.1)
    combined = ensemble.combine_predictions(sample_predictions)

    assert len(combined) == 100
    risk = ensemble.calculate_portfolio_risk(sample_predictions)
    assert 0 < risk < 0.2

def test_online_learning(sample_predictions, sample_actual):
    """测试在线学习"""
    ensemble = OnlineModelEnsemble(learning_rate=0.1)

    # 模拟在线更新
    for i in range(10):
        current_preds = {
            name: ModelPrediction(
                model_name=name,
                predictions=np.array([pred.predictions[i]]),
                confidence=np.array([pred.confidence[i]]) if pred.confidence else None
            )
            for name, pred in sample_predictions.items()
        }
        new_weights = ensemble.update_online(current_preds, sample_actual[i])

    assert len(new_weights) == 3
    assert all(0 <= w <= 1 for w in new_weights.values())

def test_model_evaluation(sample_predictions, sample_actual):
    """测试模型评估"""
    ensemble = ModelEnsemble()
    eval_df = ensemble.evaluate_models(sample_predictions, sample_actual)

    assert len(eval_df) == 3
    assert all(col in eval_df.columns for col in ['Model', 'MSE', 'Correlation', 'StdDev'])

def test_rolling_combination(sample_predictions, sample_actual):
    """测试滚动组合"""
    ensemble = ModelEnsemble()
    combined = ensemble.rolling_combination(sample_predictions, sample_actual, window=30)

    assert len(combined) == 100
    assert np.all(np.isnan(combined[:30]))  # 前30天无预测
    assert np.all(~np.isnan(combined[30:]))

def test_empty_predictions():
    """测试空预测输入"""
    ensemble = ModelEnsemble()
    with pytest.raises(ValueError):
        ensemble.combine_predictions({})

def test_single_model_combination():
    """测试单模型组合"""
    ensemble = ModelEnsemble()
    predictions = {
        "single": ModelPrediction(
            model_name="single",
            predictions=np.array([1, 2, 3])
        )
    }
    combined = ensemble.combine_predictions(predictions)

    assert np.array_equal(combined, np.array([1, 2, 3]))

def test_weight_normalization(sample_predictions):
    """测试权重归一化"""
    ensemble = ModelEnsemble(weight_method=WeightMethod.DYNAMIC)
    combined = ensemble.combine_predictions(sample_predictions)

    total_weight = sum(ensemble.weights.values())
    assert abs(total_weight - 1.0) < 1e-6

def test_risk_target_adjustment():
    """测试风险目标调整"""
    ensemble = RiskAwareEnsemble(risk_target=0.05)
    predictions = {
        "high_risk": ModelPrediction(
            model_name="high_risk",
            predictions=np.random.normal(0, 0.2, 100)
        ),
        "low_risk": ModelPrediction(
            model_name="low_risk",
            predictions=np.random.normal(0, 0.05, 100)
        )
    }
    combined = ensemble.combine_predictions(predictions)
    assert np.std(combined) < 0.06
