import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import pickle
import json
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from src.model.model_manager import ModelManager
from src.model.base_model import BaseModel

@pytest.fixture
def temp_model_dir(tmp_path):
    """临时模型目录fixture"""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir

@pytest.fixture
def mock_models():
    """模拟模型实例"""
    mock_lstm = MagicMock(spec=BaseModel)
    mock_nn = MagicMock(spec=BaseModel)
    mock_rf = MagicMock(spec=BaseModel)

    return {
        "attention_lstm": mock_lstm,
        "neural_network": mock_nn,
        "random_forest": mock_rf
    }

@pytest.fixture
def model_manager(temp_model_dir, mock_models):
    """模型管理器测试实例"""
    with patch('src.model.model_manager.AttentionLSTM', return_value=mock_models["attention_lstm"]), \
         patch('src.model.model_manager.NeuralNetworkModel', return_value=mock_models["neural_network"]), \
         patch('src.model.model_manager.RandomForestModel', return_value=mock_models["random_forest"]):

        mm = ModelManager(model_dir=str(temp_model_dir))
        yield mm

        # 清理
        mm.executor.shutdown()

def test_model_training(model_manager, mock_models, temp_model_dir):
    """测试模型训练和保存"""
    # 准备测试数据
    model_name = "attention_lstm"
    features = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]})
    targets = pd.Series([0, 1, 0])
    params = {"units": 64, "epochs": 10}

    # 设置模拟返回值
    mock_model = mock_models[model_name]
    mock_model.score.return_value = 0.95

    # 训练模型
    model, version = model_manager.train_model(
        model_name=model_name,
        features=features,
        targets=targets,
        params=params
    )

    # 验证结果
    assert model == mock_model
    assert isinstance(version, str)

    # 验证模型保存
    model_path = temp_model_dir / f"{model_name}_{version}.pkl"
    assert model_path.exists()

    # 验证元数据更新
    meta_key = f"{model_name}_{version}"
    assert meta_key in model_manager.model_metadata
    assert model_manager.model_metadata[meta_key]["train_score"] == 0.95

def test_model_loading(model_manager, mock_models, temp_model_dir):
    """测试模型加载"""
    # 准备测试模型
    model_name = "neural_network"
    version = "test_version"
    mock_model = mock_models[model_name]

    # 保存模拟模型
    model_path = temp_model_dir / f"{model_name}_{version}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(mock_model, f)

    # 更新元数据
    model_manager.model_metadata[f"{model_name}_{version}"] = {
        "model": model_name,
        "version": version,
        "train_date": datetime.now().isoformat(),
        "train_score": 0.92,
        "training_time": 10.5,
        "params": {}
    }

    # 加载模型
    loaded_model = model_manager.load_model(model_name, version)

    # 验证结果
    assert loaded_model == mock_model

def test_model_prediction(model_manager, mock_models):
    """测试模型预测"""
    # 准备测试数据
    model_name = "random_forest"
    version = "test_pred_version"
    features = pd.DataFrame({"f1": [1, 2], "f2": [3, 4]})

    # 设置模拟返回值
    mock_model = mock_models[model_name]
    mock_model.predict.return_value = np.array([0, 1])

    # 模拟加载模型
    with patch.object(model_manager, 'load_model', return_value=mock_model):
        predictions = model_manager.predict(
            model_name=model_name,
            version=version,
            features=features
        )

    # 验证结果
    assert np.array_equal(predictions, np.array([0, 1]))
    mock_model.predict.assert_called_once()

def test_batch_training(model_manager, mock_models):
    """测试批量训练"""
    # 准备测试数据
    model_configs = [
        {"model_name": "attention_lstm", "params": {"units": 64}},
        {"model_name": "random_forest", "params": {"n_estimators": 100}}
    ]
    features = pd.DataFrame({"f1": [1, 2, 3]})
    targets = pd.Series([0, 1, 0])

    # 设置模拟返回值
    mock_models["attention_lstm"].score.return_value = 0.96
    mock_models["random_forest"].score.return_value = 0.93

    # 执行批量训练
    results = model_manager.batch_train(model_configs, features, targets)

    # 验证结果
    assert len(results) == 2
    assert "attention_lstm" in results
    assert "random_forest" in results
    assert results["attention_lstm"][0] == mock_models["attention_lstm"]
    assert results["random_forest"][0] == mock_models["random_forest"]

def test_model_version_management(model_manager, mock_models, temp_model_dir):
    """测试模型版本管理"""
    # 训练多个版本的模型
    model_name = "neural_network"
    features = pd.DataFrame({"f1": [1, 2]})
    targets = pd.Series([0, 1])

    # 版本1
    mock_models["neural_network"].score.return_value = 0.90
    _, v1 = model_manager.train_model(
        model_name=model_name,
        features=features,
        targets=targets,
        version="v1"
    )

    # 版本2
    mock_models["neural_network"].score.return_value = 0.95
    _, v2 = model_manager.train_model(
        model_name=model_name,
        features=features,
        targets=targets,
        version="v2"
    )

    # 测试获取版本列表
    versions = model_manager.get_model_versions(model_name)
    assert len(versions) == 2
    assert {v["version"] for v in versions} == {"v1", "v2"}

    # 测试获取最佳模型
    best_model, best_version = model_manager.get_best_model(model_name)
    assert best_version == "v2"
    assert best_model == mock_models["neural_network"]

    # 测试删除模型
    model_manager.delete_model(model_name, "v1")
    assert f"{model_name}_v1" not in model_manager.model_metadata
    assert not (temp_model_dir / f"{model_name}_v1.pkl").exists()

def test_error_handling(model_manager, mock_models):
    """测试错误处理"""
    # 测试加载不存在的模型
    with pytest.raises(FileNotFoundError):
        model_manager.load_model("invalid_model", "v1")

    # 测试训练不存在的模型
    with pytest.raises(ValueError):
        model_manager.train_model(
            model_name="invalid_model",
            features=pd.DataFrame(),
            targets=pd.Series()
        )
