import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from models.interpretability import (
    ModelInterpreter,
    GlobalInterpretation,
    LocalInterpretation
)

@pytest.fixture
def sample_model():
    """创建模拟模型"""
    model = MagicMock()
    model.predict.return_value = np.array([0.5])
    return model

@pytest.fixture
def sample_data():
    """生成测试数据"""
    np.random.seed(42)
    data = {
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(5, 2, 100),
        'feature3': np.random.binomial(1, 0.3, 100)
    }
    return pd.DataFrame(data)

def test_global_interpretation(sample_model, sample_data):
    """测试全局解释"""
    features = ['feature1', 'feature2', 'feature3']
    interpreter = ModelInterpreter(sample_model, features)

    # 测试全局解释
    global_exp = interpreter.global_interpret(sample_data)
    assert isinstance(global_exp, GlobalInterpretation)
    assert len(global_exp.feature_importance) == 3
    assert all(feat in features for feat in global_exp.feature_importance)
    assert all(isinstance(v, float) for v in global_exp.feature_importance.values())

    # 测试部分依赖
    assert len(global_exp.partial_dependence) > 0
    assert all(isinstance(v, np.ndarray) for v in global_exp.partial_dependence.values())

    # 测试特征交互
    assert len(global_exp.interaction_strength) > 0
    assert all(isinstance(k, tuple) for k in global_exp.interaction_strength)

def test_local_interpretation(sample_model, sample_data):
    """测试局部解释"""
    features = ['feature1', 'feature2', 'feature3']
    interpreter = ModelInterpreter(sample_model, features)
    instance = sample_data.iloc[0:1]

    # 测试局部解释
    local_exp = interpreter.local_interpret(instance)
    assert isinstance(local_exp, LocalInterpretation)
    assert 0 <= local_exp.prediction <= 1
    assert len(local_exp.feature_contributions) == 3
    assert all(feat in features for feat in local_exp.feature_contributions)

    # 测试决策路径
    assert len(local_exp.decision_path) > 0
    assert all(isinstance(item, tuple) for item in local_exp.decision_path)

def test_feature_importance_plot(sample_model):
    """测试特征重要性可视化"""
    features = ['feature1', 'feature2', 'feature3']
    interpreter = ModelInterpreter(sample_model, features)

    # 模拟重要性数据
    importance = {'feature1': 0.5, 'feature2': 0.3, 'feature3': 0.2}

    # 测试绘图功能
    fig = interpreter.plot_feature_importance(importance)
    assert fig is not None

def test_contributions_plot(sample_model):
    """测试特征贡献可视化"""
    features = ['feature1', 'feature2', 'feature3']
    interpreter = ModelInterpreter(sample_model, features)

    # 模拟贡献数据
    contributions = {'feature1': 0.2, 'feature2': -0.1, 'feature3': 0.05}

    # 测试绘图功能
    fig = interpreter.plot_contributions(contributions)
    assert fig is not None

def test_edge_cases(sample_model):
    """测试边界情况"""
    # 测试空特征名
    with pytest.raises(ValueError):
        ModelInterpreter(sample_model, [])

    # 测试单特征
    interpreter = ModelInterpreter(sample_model, ['feature1'])
    global_exp = interpreter.global_interpret(pd.DataFrame({'feature1': [0]}))
    assert len(global_exp.feature_importance) == 1

    # 测试空数据
    with pytest.raises(ValueError):
        interpreter.global_interpret(pd.DataFrame())

@pytest.mark.parametrize("num_features", [1, 5, 10])
def test_varying_features(sample_model, num_features):
    """测试不同特征数量"""
    features = [f'feature{i}' for i in range(1, num_features+1)]
    data = pd.DataFrame(np.random.rand(10, num_features), columns=features)

    interpreter = ModelInterpreter(sample_model, features)
    global_exp = interpreter.global_interpret(data)
    assert len(global_exp.feature_importance) == num_features
