import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import joblib
from datetime import datetime
from fastapi.testclient import TestClient
from src.production.model_serving import (
    ModelService,
    ModelWrapper,
    ABTestManager,
    PredictionRequest
)

@pytest.fixture
def mock_model():
    """创建模拟模型"""
    model = MagicMock()
    model.predict.return_value = np.array([0.5])
    model.predict_proba.return_value = np.array([[0.3, 0.7]])
    model.feature_names_in_ = np.array(['feature1', 'feature2'])
    return model

@pytest.fixture
def model_wrapper(mock_model, tmp_path):
    """创建模型包装器实例"""
    model_path = tmp_path / "test_model.pkl"
    joblib.dump(mock_model, model_path)
    return ModelWrapper(str(model_path))

@pytest.fixture
def model_service():
    """创建模型服务实例"""
    return ModelService()

@pytest.fixture
def test_client(model_service):
    """创建测试客户端"""
    return TestClient(model_service.app)

def test_model_wrapper_loading(tmp_path, mock_model):
    """测试模型加载"""
    model_path = tmp_path / "test_model.pkl"
    joblib.dump(mock_model, model_path)

    wrapper = ModelWrapper(str(model_path))
    assert wrapper.model is not None
    assert wrapper.version == "1.0"
    assert wrapper.feature_names == ['feature1', 'feature2']

def test_model_wrapper_predict(model_wrapper, mock_model):
    """测试模型预测"""
    features = {'feature1': 0.1, 'feature2': 0.2}
    result = model_wrapper.predict(features)

    mock_model.predict.assert_called_once()
    assert isinstance(result, dict)
    assert 'prediction' in result
    assert 'probabilities' in result

def test_model_service_load_unload(model_service, tmp_path, mock_model):
    """测试模型加载与卸载"""
    model_path = tmp_path / "test_model.pkl"
    joblib.dump(mock_model, model_path)

    # 测试加载
    model_service.load_model("test_model", str(model_path))
    assert "test_model" in model_service.models

    # 测试卸载
    model_service.unload_model("test_model")
    assert "test_model" not in model_service.models

def test_model_service_predict_api(test_client, model_service, tmp_path, mock_model):
    """测试预测API"""
    model_path = tmp_path / "test_model.pkl"
    joblib.dump(mock_model, model_path)
    model_service.load_model("test_model", str(model_path))

    request_data = {
        "model_id": "test_model",
        "features": {"feature1": 0.1, "feature2": 0.2},
        "request_id": "req123"
    }

    response = test_client.post("/predict", json=request_data)
    assert response.status_code == 200
    assert response.json()["request_id"] == "req123"
    assert "prediction" in response.json()

def test_model_service_invalid_model(test_client):
    """测试无效模型请求"""
    request_data = {
        "model_id": "invalid_model",
        "features": {"feature1": 0.1, "feature2": 0.2}
    }
    response = test_client.post("/predict", json=request_data)
    assert response.status_code == 404

def test_abtest_manager(model_service, tmp_path, mock_model):
    """测试AB测试管理器"""
    # 加载两个测试模型
    model_path1 = tmp_path / "model1.pkl"
    model_path2 = tmp_path / "model2.pkl"
    joblib.dump(mock_model, model_path1)
    joblib.dump(mock_model, model_path2)

    model_service.load_model("model1", str(model_path1))
    model_service.load_model("model2", str(model_path2))

    # 创建AB测试实验
    ab_test = ABTestManager(model_service)
    ab_test.create_experiment("exp1", {"model1": 0.5, "model2": 0.5})

    # 测试预测
    features = {'feature1': 0.1, 'feature2': 0.2}
    result = ab_test.predict("exp1", features)

    assert result["experiment_id"] == "exp1"
    assert result["model_id"] in ["model1", "model2"]
    assert ab_test.experiments["exp1"]["stats"][result["model_id"]]["requests"] == 1

def test_abtest_invalid_experiment(model_service):
    """测试无效AB实验"""
    ab_test = ABTestManager(model_service)
    with pytest.raises(ValueError, match="Experiment not found"):
        ab_test.predict("invalid_exp", {})

def test_model_list_api(test_client, model_service, tmp_path, mock_model):
    """测试模型列表API"""
    model_path = tmp_path / "test_model.pkl"
    joblib.dump(mock_model, model_path)
    model_service.load_model("test_model", str(model_path))

    response = test_client.get("/models")
    assert response.status_code == 200
    assert "test_model" in response.json()["models"]

def test_model_info_api(test_client, model_service, tmp_path, mock_model):
    """测试模型信息API"""
    model_path = tmp_path / "test_model.pkl"
    joblib.dump(mock_model, model_path)
    model_service.load_model("test_model", str(model_path), version="1.1")

    response = test_client.get("/model/test_model")
    assert response.status_code == 200
    assert response.json()["version"] == "1.1"
    assert "feature1" in response.json()["feature_names"]

def test_model_info_not_found(test_client):
    """测试模型信息不存在"""
    response = test_client.get("/model/unknown_model")
    assert response.status_code == 404
