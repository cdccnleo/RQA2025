import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from src.models.model_ensemble import (
    ModelEnsembler,
    EnsembleMethod,
    ModelPrediction,
    PortfolioOptimizer,
    ModelPortfolioManager
)

@pytest.fixture
def sample_predictions():
    """生成测试预测数据"""
    np.random.seed(42)
    actuals = np.random.randn(100)
    preds = [
        ModelPrediction(
            model_name="LSTM",
            predictions=actuals + np.random.normal(0, 0.1, 100),
            confidence=np.random.uniform(0.7, 0.9, 100)
        ),
        ModelPrediction(
            model_name="RandomForest",
            predictions=actuals + np.random.normal(0, 0.2, 100),
            confidence=np.random.uniform(0.6, 0.8, 100)
        ),
        ModelPrediction(
            model_name="NeuralNet",
            predictions=actuals + np.random.normal(0, 0.15, 100),
            confidence=np.random.uniform(0.5, 0.9, 100)
        )
    ]
    return preds, actuals

@pytest.fixture
def sample_asset_data():
    """生成测试资产数据"""
    np.random.seed(42)
    n_assets = 5
    dates = pd.date_range("2025-01-01", periods=100)
    returns = pd.DataFrame(
        np.random.randn(100, n_assets) * 0.01,
        index=dates,
        columns=[f"Asset_{i}" for i in range(n_assets)]
    )
    cov = returns.cov()
    return returns, cov

def test_average_ensembler(sample_predictions):
    """测试平均集成方法"""
    preds, actuals = sample_predictions
    ensembler = ModelEnsembler(method=EnsembleMethod.AVERAGE)
    weights = ensembler.fit(preds, actuals)

    assert len(weights) == 3
    assert np.allclose(weights, [1/3, 1/3, 1/3])

    combined_pred = ensembler.predict(preds)
    assert combined_pred.shape == (100,)

def test_stacking_ensembler(sample_predictions):
    """测试堆叠集成方法"""
    preds, actuals = sample_predictions
    ensembler = ModelEnsembler(method=EnsembleMethod.STACKING)
    weights = ensembler.fit(preds, actuals)

    assert len(weights) == 3
    assert np.allclose(np.sum(weights), 1.0, rtol=0.1)  # 允许一定误差

    combined_pred = ensembler.predict(preds)
    assert combined_pred.shape == (100,)

def test_correlation_ensembler(sample_predictions):
    """测试相关性加权方法"""
    preds, actuals = sample_predictions
    ensembler = ModelEnsembler(method=EnsembleMethod.CORRELATION)
    weights = ensembler.fit(preds, actuals)

    assert len(weights) == 3
    assert np.all(weights >= 0)
    assert np.allclose(np.sum(weights), 1.0)

    combined_pred = ensembler.predict(preds)
    assert combined_pred.shape == (100,)

def test_performance_ensembler(sample_predictions):
    """测试表现加权方法"""
    preds, actuals = sample_predictions
    ensembler = ModelEnsembler(method=EnsembleMethod.PERFORMANCE)
    weights = ensembler.fit(preds, actuals)

    assert len(weights) == 3
    assert np.all(weights >= 0)
    assert np.allclose(np.sum(weights), 1.0)

    combined_pred = ensembler.predict(preds)
    assert combined_pred.shape == (100,)

def test_mean_variance_optimizer(sample_asset_data):
    """测试均值-方差优化"""
    returns, cov = sample_asset_data
    optimizer = PortfolioOptimizer()
    weights = optimizer.mean_variance(
        returns=returns.mean(),
        cov_matrix=cov
    )

    assert len(weights) == 5
    assert np.all(weights >= 0)
    assert np.allclose(np.sum(weights), 1.0)

    # 测试目标收益率约束
    target_ret = returns.mean().max()
    weights = optimizer.mean_variance(
        returns=returns.mean(),
        cov_matrix=cov,
        target_return=target_ret
    )
    achieved_ret = np.dot(weights, returns.mean())
    assert achieved_ret >= target_ret - 1e-6

def test_risk_parity_optimizer(sample_asset_data):
    """测试风险平价优化"""
    _, cov = sample_asset_data
    optimizer = PortfolioOptimizer()
    weights = optimizer.risk_parity(cov_matrix=cov)

    assert len(weights) == 5
    assert np.all(weights >= 0)
    assert np.allclose(np.sum(weights), 1.0)

    # 验证风险贡献
    port_risk = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    marginal_risk = np.dot(cov, weights) / port_risk
    risk_contrib = weights * marginal_risk
    assert np.allclose(risk_contrib / port_risk, 0.2, atol=0.05)

def test_max_diversification_optimizer(sample_asset_data):
    """测试最大分散化优化"""
    _, cov = sample_asset_data
    optimizer = PortfolioOptimizer()
    weights = optimizer.max_diversification(cov_matrix=cov)

    assert len(weights) == 5
    assert np.all(weights >= 0)
    assert np.allclose(np.sum(weights), 1.0)

    # 验证分散化比率
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    weighted_vol = np.dot(weights, np.sqrt(np.diag(cov)))
    diversification = weighted_vol / port_vol
    assert diversification > 1.0  # 应该大于1

def test_portfolio_manager(sample_predictions, sample_asset_data):
    """测试组合管理器"""
    preds, _ = sample_predictions
    returns, cov = sample_asset_data

    ensembler = ModelEnsembler(method=EnsembleMethod.STACKING)
    optimizer = PortfolioOptimizer()
    manager = ModelPortfolioManager(ensembler, optimizer)

    # 测试信号生成
    signals = manager.generate_signals(
        model_predictions=preds,
        asset_returns=returns,
        asset_cov=cov
    )

    assert len(signals) == 5
    assert 'asset' in signals.columns
    assert 'weight' in signals.columns
    assert 'pred_return' in signals.columns
    assert np.allclose(signals['weight'].sum(), 1.0)

    # 测试重新平衡
    old_weights = signals['weight'].values
    signals = manager.generate_signals(
        model_predictions=preds,
        asset_returns=returns,
        asset_cov=cov,
        rebalance=False
    )
    assert np.array_equal(old_weights, signals['weight'].values)

def test_turnover_constraint(sample_asset_data):
    """测试换手率约束"""
    returns, cov = sample_asset_data
    optimizer = PortfolioOptimizer()

    # 初始权重
    current_weights = np.array([0.3, 0.2, 0.2, 0.2, 0.1])

    # 无约束优化
    new_weights = optimizer.mean_variance(
        returns=returns.mean(),
        cov_matrix=cov
    )
    turnover = np.sum(np.abs(new_weights - current_weights))

    # 有约束优化
    constrained_weights = optimizer.mean_variance(
        returns=returns.mean(),
        cov_matrix=cov,
        turnover_constraint=turnover/2,
        current_weights=current_weights
    )
    constrained_turnover = np.sum(np.abs(constrained_weights - current_weights))

    assert constrained_turnover <= turnover/2 + 1e-6

def test_negative_correlation_handling():
    """测试负相关性处理"""
    # 构造一个负相关预测
    actuals = np.random.randn(100)
    preds = [
        ModelPrediction(
            model_name="Good",
            predictions=actuals + np.random.normal(0, 0.1, 100)
        ),
        ModelPrediction(
            model_name="Bad",
            predictions=-actuals + np.random.normal(0, 0.1, 100)  # 负相关
        )
    ]

    ensembler = ModelEnsembler(method=EnsembleMethod.CORRELATION)
    weights = ensembler.fit(preds, actuals)

    assert weights[0] > 0.9  # 正相关模型应获得几乎全部权重
    assert weights[1] < 0.1  # 负相关模型权重应接近0
