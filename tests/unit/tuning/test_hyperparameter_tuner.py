import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from src.tuning.core import (
    OptunaTuner,
    MultiObjectiveTuner,
    TuningResult,
    SearchMethod,
    ObjectiveDirection,
    EarlyStopping
)

@pytest.fixture
def sample_param_space():
    """生成测试参数空间"""
    return {
        'learning_rate': {'type': 'float', 'low': 1e-5, 'high': 1e-2, 'log': True},
        'batch_size': {'type': 'int', 'low': 16, 'high': 256},
        'optimizer': {'type': 'categorical', 'choices': ['adam', 'sgd', 'rmsprop']}
    }

@pytest.fixture
def sample_objective():
    """生成测试目标函数"""
    def objective(params):
        # 模拟一个简单的目标函数
        score = np.sin(params['learning_rate'] * 100) + \
                np.log(params['batch_size']) / 10 + \
                (0.5 if params['optimizer'] == 'adam' else 0.3)
        return score
    return objective

@pytest.fixture
def sample_multi_objectives():
    """生成多目标测试函数"""
    def obj1(params):
        return params['learning_rate'] * 100 + params['batch_size']

    def obj2(params):
        return -params['learning_rate'] * 50 + np.sqrt(params['batch_size'])

    return [
        (obj1, ObjectiveDirection.MAXIMIZE),
        (obj2, ObjectiveDirection.MINIMIZE)
    ]

def test_optuna_tuner(sample_param_space, sample_objective):
    """测试Optuna调参器"""
    tuner = OptunaTuner(method=SearchMethod.TPE)
    result = tuner.tune(
        objective_func=sample_objective,
        param_space=sample_param_space,
        n_trials=10,
        direction=ObjectiveDirection.MAXIMIZE
    )

    assert isinstance(result, TuningResult)
    assert len(result.best_params) == 3
    assert 'learning_rate' in result.best_params
    assert 'batch_size' in result.best_params
    assert 'optimizer' in result.best_params
    assert isinstance(result.trials, pd.DataFrame)
    assert len(result.trials) == 10
    assert result.importance is not None

def test_multi_objective_tuner(sample_param_space, sample_multi_objectives):
    """测试多目标调参器"""
    tuner = MultiObjectiveTuner(sample_multi_objectives)
    results = tuner.tune(
        param_space=sample_param_space,
        n_trials=10
    )

    assert isinstance(results, dict)
    assert len(results) == 2
    assert 'obj1' in results
    assert 'obj2' in results
    assert isinstance(results['obj1'], TuningResult)
    assert isinstance(results['obj2'], TuningResult)

def test_early_stopping():
    """测试早停机制"""
    early_stop = EarlyStopping(patience=3, min_delta=0.1)

    # 模拟连续下降
    assert not early_stop(1.0)  # 初始最佳
    assert not early_stop(0.95)  # 下降
    assert not early_stop(0.9)   # 继续下降
    assert early_stop(0.85)      # 触发早停

    # 测试恢复
    early_stop = EarlyStopping(patience=2, min_delta=0.1)
    assert not early_stop(1.0)
    assert not early_stop(0.9)
    assert not early_stop(1.1)  # 恢复
    assert not early_stop(1.0)  # 重新计数
    assert not early_stop(0.9)
    assert early_stop(0.8)

def test_tuning_visualization():
    """测试调参可视化"""
    from src.tuning.core import TuningVisualizer

    # 测试优化历史图
    trials = pd.DataFrame({
        'number': range(10),
        'value': np.random.random(10)
    })
    fig = TuningVisualizer.plot_optimization_history(trials)
    assert fig is not None

    # 测试参数重要性图
    importance = {'param1': 0.8, 'param2': 0.5, 'param3': 0.3}
    fig = TuningVisualizer.plot_param_importance(importance)
    assert fig is not None

    # 测试平行坐标图
    trials = pd.DataFrame({
        'param1': np.random.random(10),
        'param2': np.random.random(10),
        'value': np.random.random(10)
    })
    fig = TuningVisualizer.plot_parallel_coordinate(trials, ['param1', 'param2'])
    assert fig is not None

def test_edge_cases():
    """测试边界情况"""
    # 测试单参数优化
    param_space = {'lr': {'type': 'float', 'low': 0.001, 'high': 0.1}}
    tuner = OptunaTuner()
    result = tuner.tune(
        objective_func=lambda p: p['lr'],
        param_space=param_space,
        n_trials=5,
        direction=ObjectiveDirection.MAXIMIZE
    )
    assert len(result.best_params) == 1
    assert 0.001 <= result.best_params['lr'] <= 0.1

    # 测试空参数空间
    with pytest.raises(ValueError):
        tuner.tune(
            objective_func=lambda p: 0,
            param_space={},
            n_trials=5
        )
