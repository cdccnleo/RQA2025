import sys
import logging
import os
from pathlib import Path
from typing import Dict

# 确保正确的模块路径
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
project_root = os.path.abspath(os.path.join(current_dir, '../../../../'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import pytest

from src.ml.core.ml_core import MLCore


@pytest.fixture
def ml_core_factory(monkeypatch):
    def failing_adapter():
        raise RuntimeError("adapter unavailable")

    monkeypatch.setattr("ml.core.ml_core._get_models_adapter", failing_adapter)

    def factory(config: Dict = None) -> MLCore:
        return MLCore(config)

    return factory


def make_dataset():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [0.5, 0.0, -1.0, 1.5]})
    y = pd.Series([0.0, 1.0, 1.0, 0.0])
    return X, y


def test_initialization_fallback(ml_core_factory):
    core = ml_core_factory({"epochs": 5})
    assert core.cache_manager is None
    assert core.config["epochs"] == 5
    assert core.models == {}


def test_initialization_without_model_cache_flag_applies_defaults(monkeypatch):
    sentinel = object()

    class Adapter:
        def get_models_cache_manager(self):
            return sentinel

        def get_models_config_manager(self):
            return sentinel

        def get_models_logger(self):
            return logging.getLogger("mlcore.test")

    monkeypatch.setattr("ml.core.ml_core._get_models_adapter", lambda: Adapter())

    core = MLCore({})

    assert core.config["model_cache_enabled"] is True
    assert core.cache_manager is None
    assert core.config_manager is None


def test_train_and_predict_linear_model(ml_core_factory):
    core = ml_core_factory({"random_state": 0})
    X, y = make_dataset()
    model_id = core.train_model(X, y, model_type="linear")
    assert model_id in core.list_models()
    predictions = core.predict(model_id, X)
    assert len(predictions) == len(X)


def test_create_model_invalid_type(ml_core_factory):
    core = ml_core_factory()
    with pytest.raises(ValueError):
        core._create_model("unsupported")


def test_get_feature_importance_with_importances(ml_core_factory):
    core = ml_core_factory()

    class FakeModel:
        feature_importances_ = np.array([0.3, 0.7])

    core.models["model"] = {
        "model": FakeModel(),
        "feature_names": ["x1", "x2"],
    }
    importance = core.get_feature_importance("model")
    assert importance == {"x1": 0.3, "x2": 0.7}


def test_save_load_and_delete_model(tmp_path: Path, ml_core_factory):
    core = ml_core_factory({"random_state": 0})
    X, y = make_dataset()
    model_id = core.train_model(X, y, model_type="linear")

    path = tmp_path / "model.pkl"
    assert core.save_model(model_id, path)

    new_core = ml_core_factory()
    loaded_id = new_core.load_model(path)
    assert loaded_id is not None
    assert new_core.delete_model(loaded_id)
    assert new_core.delete_model(loaded_id) is False


def test_cross_validate_returns_metrics(ml_core_factory):
    core = ml_core_factory({"cross_validation_folds": 2, "random_state": 0})
    X, y = make_dataset()
    results = core.cross_validate(X, y, model_type="linear")
    assert "mean_score" in results
    assert results["folds"] == 2


def test_feature_processor_workflow(ml_core_factory):
    core = ml_core_factory()
    processor_id = core.create_feature_processor("standard")
    assert processor_id in core.feature_processors
    X, _ = make_dataset()
    core.fit_feature_processor(processor_id, X)
    transformed = core.transform_features(processor_id, X)
    assert transformed.shape == X.values.shape


def test_initialization_force_fallback_and_null_adapters(monkeypatch):
    """测试初始化时强制fallback和adapter返回None的情况（46-49, 69, 79行）"""
    import os
    
    # 测试强制fallback（69行）
    monkeypatch.setenv("ML_CORE_FORCE_FALLBACK", "1")
    
    class NullAdapter:
        def get_models_cache_manager(self):
            return None
        def get_models_config_manager(self):
            return None
        def get_models_logger(self):
            return logging.getLogger("mlcore.test")
    
    monkeypatch.setattr("src.ml.core.ml_core._get_models_adapter", lambda: NullAdapter())
    
    core = MLCore({})
    # 应该应用默认服务（79行）
    assert core.cache_manager is None
    assert core.config_manager is None


def test_train_model_exception_path(ml_core_factory, monkeypatch):
    """测试模型训练异常处理路径（150-153行）"""
    core = ml_core_factory()
    X, y = make_dataset()
    
    # Mock _create_model抛出异常
    original_create = core._create_model
    def failing_create(*args, **kwargs):
        raise RuntimeError("model creation failed")
    
    monkeypatch.setattr(core, "_create_model", failing_create)
    
    with pytest.raises(Exception):  # ModelTrainingError
        core.train_model(X, y, model_type="linear")


def test_prepare_features_else_branch(ml_core_factory):
    """测试_prepare_features中numpy数组的分支（163, 170行）"""
    core = ml_core_factory()
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    feature_names = ["x1", "x2"]
    
    X_processed, names = core._prepare_features(X, feature_names)
    assert isinstance(X_processed, np.ndarray)
    assert names == feature_names


def test_predict_model_not_found(ml_core_factory):
    """测试predict中模型不存在的异常路径（198行）"""
    core = ml_core_factory()
    X, _ = make_dataset()
    
    from src.ml.core.exceptions import ModelNotFoundError

    with pytest.raises(ModelNotFoundError, match="模型未找到"):
        core.predict("nonexistent_model", X)


def test_predict_exception_handling(ml_core_factory, monkeypatch):
    """测试predict中预测失败的异常处理路径（208, 213-215行）"""
    core = ml_core_factory({"random_state": 0})
    X, y = make_dataset()
    model_id = core.train_model(X, y, model_type="linear")
    
    # Mock model.predict抛出异常
    original_predict = core.models[model_id]['model'].predict
    def failing_predict(*args, **kwargs):
        raise RuntimeError("prediction failed")
    
    monkeypatch.setattr(core.models[model_id]['model'], "predict", failing_predict)
    
    with pytest.raises(Exception):
        core.predict(model_id, X)


def test_evaluate_model_exception_handling(ml_core_factory, monkeypatch):
    """测试evaluate_model中的异常处理路径（232-242行）"""
    core = ml_core_factory({"random_state": 0})
    X, y = make_dataset()
    model_id = core.train_model(X, y, model_type="linear")
    
    # Mock predict抛出异常
    original_predict = core.predict
    def failing_predict(*args, **kwargs):
        raise RuntimeError("prediction failed in evaluate")
    
    monkeypatch.setattr(core, "predict", failing_predict)
    
    with pytest.raises(Exception):
        core.evaluate_model(model_id, X, y)


def test_create_model_rf_type(ml_core_factory):
    """测试创建随机森林模型（255-260行）"""
    core = ml_core_factory({"random_state": 42})
    model = core._create_model("rf", {"n_estimators": 50})
    assert model is not None
    assert model.n_estimators == 50


def test_create_model_xgb_with_import_error(ml_core_factory, monkeypatch):
    """测试创建XGBoost模型时的ImportError处理路径（263-277行）"""
    core = ml_core_factory({"random_state": 42})
    
    # Mock xgboost导入失败
    import sys
    original_import = __import__
    
    def mock_import(name, *args, **kwargs):
        if name == "xgboost" or name.startswith("xgboost"):
            raise ImportError("No module named 'xgboost'")
        return original_import(name, *args, **kwargs)
    
    monkeypatch.setattr("builtins.__import__", mock_import)
    
    # 应该fallback到RandomForest
    model = core._create_model("xgb", {})
    assert model is not None
    # 验证是RandomForestRegressor而不是XGBRegressor
    from sklearn.ensemble import RandomForestRegressor
    assert isinstance(model, RandomForestRegressor)


def test_create_simple_neural_network_with_import_error(ml_core_factory, monkeypatch):
    """测试创建神经网络时的ImportError处理路径（293-309行）"""
    core = ml_core_factory({"random_state": 42, "epochs": 100})
    
    # Mock sklearn.neural_network导入失败
    import sys
    original_import = __import__
    
    def mock_import(name, *args, **kwargs):
        if name == "sklearn.neural_network" or (name == "sklearn" and args and args[0] == "neural_network"):
            raise ImportError("No module named 'sklearn.neural_network'")
        return original_import(name, *args, **kwargs)
    
    monkeypatch.setattr("builtins.__import__", mock_import)
    
    # 应该fallback到LinearRegression
    model = core._create_simple_neural_network({})
    assert model is not None
    from sklearn.linear_model import LinearRegression
    assert isinstance(model, LinearRegression)


def test_preprocess_features_exception_path(ml_core_factory, monkeypatch):
    """测试_preprocess_features中的异常处理路径（324-326行）"""
    core = ml_core_factory()
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    
    # Mock fillna抛出异常
    original_fillna = pd.DataFrame.fillna
    def failing_fillna(self, *args, **kwargs):
        raise RuntimeError("fillna failed")
    
    monkeypatch.setattr(pd.DataFrame, "fillna", failing_fillna)
    
    # 应该捕获异常并返回values
    result = core._preprocess_features(X, ["a", "b"])
    assert isinstance(result, np.ndarray)


def test_calculate_metrics_exception_path(ml_core_factory, monkeypatch):
    """测试_calculate_metrics中的异常处理路径（334-351行）"""
    core = ml_core_factory()
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 2.1, 2.9])
    
    # Mock sklearn.metrics导入失败
    original_import = __import__
    def mock_import(name, *args, **kwargs):
        if name == "sklearn.metrics" or (name == "sklearn" and args and args[0] == "metrics"):
            raise ImportError("No module named 'sklearn.metrics'")
        return original_import(name, *args, **kwargs)
    
    monkeypatch.setattr("builtins.__import__", mock_import)
    
    # 应该返回空字典（351行）
    result = core._calculate_metrics(y_true, y_pred)
    assert result == {}


def test_create_feature_processor_exception_path(ml_core_factory, monkeypatch):
    """测试create_feature_processor中的异常处理路径（408-410, 453-455行）"""
    core = ml_core_factory()
    
    # Mock StandardScaler导入失败
    original_import = __import__
    def mock_import(name, *args, **kwargs):
        if name == "sklearn.preprocessing" or (name == "sklearn" and args and args[0] == "preprocessing"):
            raise ImportError("No module named 'sklearn.preprocessing'")
        return original_import(name, *args, **kwargs)
    
    monkeypatch.setattr("builtins.__import__", mock_import)
    
    with pytest.raises(Exception):
        core.create_feature_processor("standard")


def test_fit_feature_processor_exception_paths(ml_core_factory, monkeypatch):
    """测试fit_feature_processor中的异常处理路径（461, 470-472行）"""
    core = ml_core_factory()
    X, _ = make_dataset()
    
    # 测试处理器不存在（461行）
    with pytest.raises(ValueError, match="特征处理器 .* 不存在"):
        core.fit_feature_processor("nonexistent", X)
    
    # 创建处理器并测试拟合异常
    processor_id = core.create_feature_processor("standard")
    
    # Mock fit抛出异常
    def failing_fit(*args, **kwargs):
        raise RuntimeError("fit failed")
    
    monkeypatch.setattr(core.feature_processors[processor_id]['processor'], "fit", failing_fit)
    
    with pytest.raises(Exception):
        core.fit_feature_processor(processor_id, X)


def test_transform_features_exception_paths(ml_core_factory, monkeypatch):
    """测试transform_features中的异常处理路径（478, 488-490行）"""
    core = ml_core_factory()
    X, _ = make_dataset()
    
    # 测试处理器不存在（478行）
    with pytest.raises(ValueError, match="特征处理器 .* 不存在"):
        core.transform_features("nonexistent", X)
    
    # 创建并拟合处理器，然后测试转换异常
    processor_id = core.create_feature_processor("standard")
    core.fit_feature_processor(processor_id, X)
    
    # Mock transform抛出异常
    original_transform = core.feature_processors[processor_id]['processor'].transform
    def failing_transform(*args, **kwargs):
        raise RuntimeError("transform failed")
    
    monkeypatch.setattr(core.feature_processors[processor_id]['processor'], "transform", failing_transform)
    
    with pytest.raises(Exception):
        core.transform_features(processor_id, X)


def test_prepare_target_with_numpy_array(ml_core_factory):
    """测试_prepare_target处理numpy数组的分支（170行）"""
    core = ml_core_factory()
    y_numpy = np.array([0.0, 1.0, 1.0, 0.0])
    result = core._prepare_target(y_numpy)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, y_numpy)


def test_predict_with_numpy_array(ml_core_factory):
    """测试predict处理numpy数组的else分支（208行）"""
    core = ml_core_factory()
    X, y = make_dataset()
    model_id = core.train_model(X, y)
    
    # 使用numpy数组而不是DataFrame
    X_numpy = X.values
    predictions = core.predict(model_id, X_numpy)
    assert len(predictions) == len(X_numpy)


def test_evaluate_model_normal_path(ml_core_factory):
    """测试evaluate_model的正常执行路径（235-239行）"""
    core = ml_core_factory()
    X, y = make_dataset()
    model_id = core.train_model(X, y)
    
    # 正常评估路径
    metrics = core.evaluate_model(model_id, X, y)
    assert "mse" in metrics or "accuracy" in metrics or "r2" in metrics


def test_initialization_with_null_adapters_else_branch(monkeypatch):
    """测试初始化时cache_manager和config_manager都为None的else分支（79行）"""
    class MockAdapter:
        def get_models_cache_manager(self):
            return None
        def get_models_config_manager(self):
            return None
        def get_models_logger(self):
            return logging.getLogger(__name__)
    
    def mock_get_adapter():
        return MockAdapter()
    
    monkeypatch.setattr("src.ml.core.ml_core._get_models_adapter", mock_get_adapter)
    
    core = MLCore()
    # 应该应用默认服务
    assert hasattr(core, 'logger')


def test_get_feature_importance_branches(ml_core_factory):
    """测试get_feature_importance中的各种分支（496, 504-507, 512行）"""
    core = ml_core_factory()
    
    # 测试模型不存在（496行）
    assert core.get_feature_importance("nonexistent") is None
    
    # 测试模型没有feature_importances_也没有coef_（507行）
    class NoImportanceModel:
        pass
    
    core.models["no_importance"] = {
        "model": NoImportanceModel(),
        "feature_names": []
    }
    assert core.get_feature_importance("no_importance") is None
    
    # 测试模型有coef_但没有feature_names（504-506行）
    class CoefModel:
        coef_ = np.array([0.5, -0.3])
    
    core.models["coef_only"] = {
        "model": CoefModel(),
        "feature_names": []
    }
    importance = core.get_feature_importance("coef_only")
    assert importance is not None

