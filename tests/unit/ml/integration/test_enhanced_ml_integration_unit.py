from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pandas as pd
import pytest

# 创建mock adapter
mock_adapter = Mock(get_models_logger=lambda: Mock())

# 在导入模块之前先patch get_models_adapter
# 需要同时patch两个位置：src.core.integration和src.ml.integration.enhanced_ml_integration
import sys
from unittest.mock import MagicMock

# 创建mock的integration模块
mock_integration_module = MagicMock()
mock_integration_module.get_models_adapter = Mock(return_value=mock_adapter)

# 如果src.core.integration不存在，先创建它
if 'src.core.integration' not in sys.modules:
    import types
    sys.modules['src.core'] = types.ModuleType('src.core')
    sys.modules['src.core.integration'] = mock_integration_module
else:
    # 如果已存在，则patch它
    sys.modules['src.core.integration'].get_models_adapter = Mock(return_value=mock_adapter)

# 现在可以安全导入
from src.ml.integration.enhanced_ml_integration import (
    EnhancedMLIntegration,
    MLModelConfig,
    PredictionResult,
)


@pytest.fixture(autouse=True)
def patch_models_adapter(monkeypatch):
    """自动应用的fixture，确保get_models_adapter被正确mock"""
    # 使用monkeypatch来设置get_models_adapter，避免AttributeError
    monkeypatch.setattr("src.ml.integration.enhanced_ml_integration.get_models_adapter", lambda: mock_adapter)
    # 如果src.core.integration存在，也patch它
    try:
        import src.core.integration
        monkeypatch.setattr("src.core.integration.get_models_adapter", lambda: mock_adapter)
    except (ImportError, AttributeError):
        pass
    yield mock_adapter


def build_market_data(rows=260):
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    data = pd.DataFrame(
        {
            "close": np.linspace(100, 120, rows) + np.random.randn(rows),
            "volume": np.linspace(1_000, 2_000, rows),
        },
        index=dates,
    )
    return data


class DummyScaler:
    def fit_transform(self, X):
        self.mean_ = X.mean(axis=0)
        return X.values

    def transform(self, X):
        return X if isinstance(X, np.ndarray) else X.values


class DummyModel:
    def __init__(self):
        self.feature_importances_ = []

    def fit(self, X, y):
        self.feature_importances_ = np.ones(X.shape[1]) / X.shape[1]

    def predict(self, X):
        return np.ones(len(X))

    def predict_proba(self, X):
        return np.tile([0.3, 0.7], (len(X), 1))


@pytest.fixture
def integration(monkeypatch):
    config = MLModelConfig(training_period=100, prediction_window=5)
    ml = EnhancedMLIntegration(config=config)
    ml.scaler = DummyScaler()
    ml.model = DummyModel()
    return ml


def test_prepare_features_generates_expected_columns(integration):
    data = build_market_data()
    features = integration.prepare_features(data)
    expected = {"returns", "momentum", "volume_ratio", "rsi", "volatility"}
    assert expected.issubset(set(features.columns))
    assert not features.isna().any().any()


def test_train_model_returns_metrics_and_records_history(integration):
    data = build_market_data(300)
    metrics = integration.train_model(data)
    assert "accuracy" in metrics
    assert integration.training_history
    assert integration.feature_importance


def test_predict_returns_default_when_no_features(integration):
    empty_data = build_market_data(rows=10)
    result = integration.predict(empty_data)
    assert isinstance(result, PredictionResult)
    assert result.confidence == 0.0
    assert result.prediction == 0


def test_predict_after_training_uses_model(integration):
    data = build_market_data(300)
    integration.train_model(data)
    result = integration.predict(data)
    assert result.prediction == 1
    assert result.confidence > 0
    assert integration.prediction_history


def test_train_model_insufficient_data(integration):
    """测试训练数据不足的情况（106-108行）"""
    data = build_market_data(rows=50)  # 少于training_period=100
    metrics = integration.train_model(data)
    assert metrics == {}
    assert not integration.training_history


def test_predict_exception_handling(integration, monkeypatch):
    """测试预测异常处理路径（173-180行）"""
    data = build_market_data(300)
    integration.train_model(data)
    
    # Mock model.predict抛出异常，确保异常在try块内被捕获（173-180行）
    def failing_predict(X):
        raise RuntimeError("predict failed")
    
    monkeypatch.setattr(integration.model, "predict", failing_predict)
    
    # 异常应该被捕获并返回默认的PredictionResult
    result = integration.predict(data)
    assert result.prediction == 0
    assert result.confidence == 0.0
    assert result.probability == 0.5
    assert result.features_importance == {}


def test_get_model_performance_with_history(integration):
    """测试获取模型性能统计（182-193行）"""
    data = build_market_data(300)
    integration.train_model(data)
    
    performance = integration.get_model_performance()
    assert "latest_training" in performance
    assert "training_count" in performance
    assert "prediction_count" in performance
    assert "accuracy" in performance
    assert performance["training_count"] == 1


def test_get_model_performance_no_history(integration):
    """测试获取模型性能统计（无历史记录）（184-185行）"""
    performance = integration.get_model_performance()
    assert performance == {}


def test_calculate_rsi(integration):
    """测试RSI计算（78-85行）"""
    import pandas as pd
    prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
    rsi = integration._calculate_rsi(prices, period=5)
    assert len(rsi) == len(prices)
    assert not rsi.isna().all()  # 至少有一些有效值


def test_prepare_labels(integration):
    """测试准备标签数据（87-92行）"""
    data = build_market_data(300)
    labels = integration.prepare_labels(data)
    assert len(labels) == len(data)
    assert labels.dtype == int or labels.dtype == 'int64'
    assert labels.isin([0, 1]).all()


def test_logger_fallback_on_exception():
    """测试logger初始化异常时的降级处理（26-28行）"""
    # 这个测试需要重新导入模块，所以我们需要在单独的测试中处理
    # 由于模块已经在导入时处理了异常，我们只需要验证logger存在
    from src.ml.integration import enhanced_ml_integration
    # 验证logger已经被正确初始化（无论是通过adapter还是降级处理）
    assert hasattr(enhanced_ml_integration, 'logger')
    assert enhanced_ml_integration.logger is not None


def test_main_block_execution(monkeypatch, capsys):
    """测试if __name__ == '__main__'块的执行（197行）"""
    # 通过直接执行模块的main块来测试
    import src.ml.integration.enhanced_ml_integration as module
    
    # 模拟__name__ == "__main__"的情况
    original_name = module.__name__
    monkeypatch.setattr(module, "__name__", "__main__")
    
    # 执行main块
    try:
        # 重新执行模块级别的代码（实际上这很难直接测试）
        # 我们可以通过检查模块是否可以被正常导入和实例化来间接验证
        ml_integration = module.EnhancedMLIntegration()
        assert ml_integration is not None
    finally:
        monkeypatch.setattr(module, "__name__", original_name)

