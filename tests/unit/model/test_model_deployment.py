import pytest
import numpy as np
import tempfile
from unittest.mock import MagicMock
from models.model_deployment import (
    ModelDeployment,
    ModelFormat,
    DeploymentService,
    ABTestFramework
)
from fastapi.testclient import TestClient

@pytest.fixture
def sample_data():
    """生成测试数据"""
    return np.random.rand(10, 5)

@pytest.fixture
def mock_model():
    """创建模拟模型"""
    model = MagicMock()
    model.predict.return_value = np.array([1, 0, 1, 0, 1])
    return model

@pytest.fixture
def temp_deployment():
    """创建临时部署环境"""
    with tempfile.TemporaryDirectory() as temp_dir:
        deployment = ModelDeployment(model_dir=temp_dir)
        yield deployment

def test_model_export_load(temp_deployment, mock_model, sample_data):
    """测试模型导出和加载"""
    # 测试Pickle格式
    version = temp_deployment.export_model(
        mock_model,
        "test_model",
        sample_data,
        ModelFormat.PICKLE
    )

    # 验证元数据
    assert f"test_model_{version}" in temp_deployment.list_models()

    # 测试加载
    loaded_model = temp_deployment.load_model("test_model", version)
    assert loaded_model.predict(sample_data[:1])[0] in [0, 1]

    # 测试删除
    temp_deployment.delete_model("test_model", version)
    assert f"test_model_{version}" not in temp_deployment.list_models()

def test_prediction(temp_deployment, mock_model, sample_data):
    """测试预测功能"""
    version = temp_deployment.export_model(
        mock_model,
        "predict_model",
        sample_data,
        ModelFormat.PICKLE
    )

    # 测试预测
    result = temp_deployment.predict("predict_model", version, sample_data[:2])
    assert len(result) == 2

def test_deployment_service(temp_deployment, mock_model, sample_data):
    """测试部署服务"""
    # 添加测试模型
    version = temp_deployment.export_model(
        mock_model,
        "service_model",
        sample_data,
        ModelFormat.PICKLE
    )

    # 创建测试客户端
    service = DeploymentService(temp_deployment)
    client = TestClient(service.app)

    # 测试预测API
    response = client.post(
        "/predict",
        json={
            "model_name": "service_model",
            "version": version,
            "data": sample_data[:1].tolist()
        }
    )
    assert response.status_code == 200
    assert "prediction" in response.json()

    # 测试模型列表API
    response = client.get("/models")
    assert response.status_code == 200
    assert "service_model" in str(response.json())

def test_ab_testing(temp_deployment, mock_model, sample_data):
    """测试AB测试框架"""
    # 添加两个版本模型
    v1 = temp_deployment.export_model(
        mock_model,
        "ab_model",
        sample_data,
        ModelFormat.PICKLE
    )

    mock_model_v2 = MagicMock()
    mock_model_v2.predict.return_value = np.array([0, 1, 0, 1, 0])
    v2 = temp_deployment.export_model(
        mock_model_v2,
        "ab_model",
        sample_data,
        ModelFormat.PICKLE
    )

    # 初始化AB测试框架
    ab_test = ABTestFramework(temp_deployment)
    ab_test.start_experiment(
        "test_exp",
        "ab_model", v1,
        "ab_model", v2,
        0.5
    )

    # 测试预测
    results = []
    for _ in range(10):
        result = ab_test.predict("test_exp", sample_data[:1])
        results.append(result["model_used"])

    # 验证流量分配
    assert "ab_model" in results
    stats = ab_test.get_experiment_stats("test_exp")
    assert stats["total_requests"] == 10
    assert stats["a_requests"] + stats["b_requests"] == 10

    # 结束实验
    final_stats = ab_test.end_experiment("test_exp")
    assert final_stats["total_requests"] == 10

def test_invalid_model(temp_deployment):
    """测试无效模型处理"""
    with pytest.raises(ValueError):
        temp_deployment.load_model("nonexistent", "1.0")

def test_invalid_format(temp_deployment, mock_model, sample_data):
    """测试无效格式处理"""
    with pytest.raises(ValueError):
        temp_deployment.export_model(
            mock_model,
            "invalid_model",
            sample_data,
            "invalid_format"  # type: ignore
        )

def test_service_error_handling(temp_deployment):
    """测试服务错误处理"""
    service = DeploymentService(temp_deployment)
    client = TestClient(service.app)

    # 测试无效预测请求
    response = client.post(
        "/predict",
        json={
            "model_name": "nonexistent",
            "version": "1.0",
            "data": [[1, 2, 3]]
        }
    )
    assert response.status_code == 400

def test_ab_test_error_handling(temp_deployment):
    """测试AB测试错误处理"""
    ab_test = ABTestFramework(temp_deployment)

    with pytest.raises(ValueError):
        ab_test.predict("nonexistent", np.array([[1, 2, 3]]))

def test_model_metadata(temp_deployment, mock_model, sample_data):
    """测试模型元数据"""
    version = temp_deployment.export_model(
        mock_model,
        "meta_model",
        sample_data,
        ModelFormat.PICKLE
    )

    models = temp_deployment.list_models()
    model_info = models[f"meta_model_{version}"]

    assert model_info["name"] == "meta_model"
    assert model_info["version"] == version
    assert model_info["format"] == "pickle"
    assert "hash" in model_info

def test_onnx_integration(temp_deployment, mock_model, sample_data):
    """测试ONNX格式集成"""
    # 跳过实际ONNX转换的测试(需要真实模型)
    pass
