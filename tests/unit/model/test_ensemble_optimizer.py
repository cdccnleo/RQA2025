import pytest
import numpy as np
import pandas as pd
from models.ensemble_optimizer import (
    EnsembleOptimizer,
    OptimizationMethod,
    RiskBudgetOptimizer,
    ModelPerformance
)

@pytest.fixture
def sample_predictions():
    """生成测试预测数据"""
    dates = pd.date_range("2025-01-01", periods=100)
    return {
        "LSTM": pd.DataFrame({
            'return': np.random.normal(0.001, 0.02, 100),
            'prediction': np.random.normal(0, 1, 100)
        }, index=dates),
        "RandomForest": pd.DataFrame({
            'return': np.random.normal(0.0008, 0.015, 100),
            'prediction': np.random.normal(0.1, 0.8, 100)
        }, index=dates),
        "NeuralNet": pd.DataFrame({
            'return': np.random.normal(0.0005, 0.01, 100),
            'prediction': np.random.normal(-0.1, 0.5, 100)
        }, index=dates)
    }

def test_mean_variance_optimization(sample_predictions):
    """测试均值-方差优化"""
    optimizer = EnsembleOptimizer(method=OptimizationMethod.MEAN_VARIANCE)
    weights = optimizer.optimize_weights(sample_predictions)

    assert len(weights) == 3
    assert pytest.approx(sum(weights.values()), 1e-6) == 1.0
    assert all(0 <= w <= 1 for w in weights.values())

def test_risk_parity_optimization(sample_predictions):
    """测试风险平价优化"""
    optimizer = EnsembleOptimizer(method=OptimizationMethod.RISK_PARITY)
    weights = optimizer.optimize_weights(sample_predictions)

    # 验证风险贡献大致相等
    returns = pd.DataFrame({name: pred['return'] for name, pred in sample_predictions.items()})
    cov_matrix = returns.cov()
    port_var = sum(w * sum(cov_matrix[name] * w for name in weights) for name, w in weights.items())
    risk_contrib = {name: w * sum(cov_matrix[name] * w for w in weights.values()) / port_var
                   for name in weights}

    assert len(weights) == 3
    assert pytest.approx(sum(weights.values()), 1e-6) == 1.0
    assert all(0.3 <= rc <= 0.4 for rc in risk_contrib.values())  # 大致均衡

def test_performance_analysis(sample_predictions):
    """测试绩效分析"""
    optimizer = EnsembleOptimizer()
    performance = optimizer.calculate_performance(sample_predictions)

    assert len(performance) == 3
    assert all(isinstance(p, ModelPerformance) for p in performance.values())
    assert all(p.volatility > 0 for p in performance.values())
    assert all(p.max_drawdown <= 0 for p in performance.values())

def test_adaptive_weight_update(sample_predictions):
    """测试自适应权重更新"""
    optimizer = EnsembleOptimizer()
    current_weights = {'LSTM': 0.5, 'RandomForest': 0.3, 'NeuralNet': 0.2}
    new_weights = optimizer.adaptive_update(sample_predictions, current_weights)

    assert len(new_weights) == 3
    assert pytest.approx(sum(new_weights.values()), 1e-6) == 1.0
    # 新权重应在原始权重和最优权重之间
    assert all(abs(new_weights[name] - current_weights.get(name, 0)) <= 0.2
               for name in sample_predictions)

def test_correlation_calculation(sample_predictions):
    """测试相关性计算"""
    optimizer = EnsembleOptimizer()
    corr_matrix = optimizer.calculate_correlation(sample_predictions)

    assert corr_matrix.shape == (3, 3)
    assert all(-1 <= corr_matrix.loc[name1, name2] <= 1
              for name1 in corr_matrix.index
              for name2 in corr_matrix.columns)
    assert all(corr_matrix.loc[name, name] == 1 for name in corr_matrix.index)

def test_risk_budget_optimizer(sample_predictions):
    """测试风险预算优化"""
    risk_budget = {'LSTM': 0.5, 'RandomForest': 0.3, 'NeuralNet': 0.2}
    optimizer = RiskBudgetOptimizer(risk_budget=risk_budget)
    weights = optimizer.optimize(sample_predictions)

    # 验证风险贡献比例接近预算
    returns = pd.DataFrame({name: pred['return'] for name, pred in sample_predictions.items()})
    cov_matrix = returns.cov()
    port_var = sum(w * sum(cov_matrix[name] * w for name in weights) for name, w in weights.items())
    risk_contrib = {name: w * sum(cov_matrix[name] * w for w in weights.values()) / port_var
                   for name in weights}

    assert len(weights) == 3
    assert pytest.approx(sum(weights.values()), 1e-6) == 1.0
    assert all(abs(risk_contrib[name] - risk_budget[name]) < 0.1
              for name in risk_budget)

def test_min_variance_optimization(sample_predictions):
    """测试最小方差优化"""
    optimizer = EnsembleOptimizer(method=OptimizationMethod.MIN_VARIANCE)
    weights = optimizer.optimize_weights(sample_predictions)

    # 验证组合方差确实最小
    returns = pd.DataFrame({name: pred['return'] for name, pred in sample_predictions.items()})
    cov_matrix = returns.cov()
    min_var = sum(w * sum(cov_matrix[name] * w for name in weights) for name, w in weights.items())

    # 与其他简单权重组合比较
    equal_weights = {'LSTM': 1/3, 'RandomForest': 1/3, 'NeuralNet': 1/3}
    equal_var = sum(w * sum(cov_matrix[name] * w for name in equal_weights)
                   for name, w in equal_weights.items())

    assert min_var <= equal_var

def test_performance_history_tracking(sample_predictions):
    """测试绩效历史跟踪"""
    optimizer = EnsembleOptimizer()

    # 第一次优化
    weights1 = optimizer.optimize_weights(sample_predictions)
    perf1 = optimizer.analyze_performance(sample_predictions, weights1)

    # 第二次优化
    new_pred = {name: df.iloc[:50] for name, df in sample_predictions.items()}
    weights2 = optimizer.optimize_weights(new_pred)
    perf2 = optimizer.analyze_performance(new_pred, weights2)

    assert len(optimizer.weights_history) == 2
    assert len(optimizer.performance_history) == 2
    assert optimizer.performance_history[0]['return'] == perf1['return']
    assert optimizer.performance_history[1]['return'] == perf2['return']
