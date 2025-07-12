from pathlib import Path
from unittest.mock import patch

import pytest
import pandas as pd
import numpy as np
import torch

from src.models.lstm import LSTMModelWrapper
from src.models.nn import NeuralNetworkModel
from src.models.rf import RandomForestModel
from src.models.model_manager import ModelManager
from src.models.utils import DeviceManager


# 测试数据生成夹具
@pytest.fixture
def sample_data():
    # 生成示例特征数据和目标数据
    np.random.seed(42)
    features = pd.DataFrame(np.random.randn(100, 5), columns=[f'feature_{i}' for i in range(5)])
    target = pd.Series(np.random.randn(100), name='target')
    return features, target


# 测试模型夹具
@pytest.fixture
def lstm_model(sample_data):
    """初始化LSTM模型"""
    features, _ = sample_data
    return LSTMModelWrapper(
        input_size=features.shape[1],  # 动态获取特征维度
        seq_length=10,
        hidden_size=32,
        num_layers=3,
        output_size=1,
        dropout=0.5,
        device="cpu"
    )


@pytest.fixture
def nn_model(sample_data) -> NeuralNetworkModel:
    features, _ = sample_data
    return NeuralNetworkModel(
        input_size=features.shape[1],
        hidden_layers=[32, 16],
        output_size=1,
        dropout_rate=0.5,
        device="cpu"
    )


@pytest.fixture
def rf_model(sample_data):
    model = RandomForestModel(
        n_estimators=50,
        max_depth=5
    )
    return model


@pytest.fixture(params=["lstm", "nn", "rf"])
def sample_model(request, sample_data):
    """创建不同模型的示例实例"""
    features, target = sample_data
    if request.param == "lstm":
        model = LSTMModelWrapper(
            input_size=features.shape[1],
            seq_length=10,
            hidden_size=32,
            device="cpu"
        )
    elif request.param == "nn":
        model = NeuralNetworkModel(
            input_size=features.shape[1],
            hidden_layers=[32, 16],
            output_size=1,
            dropout_rate=0.5,
            device="cpu"
        )
    elif request.param == "rf":
        model = RandomForestModel(
            n_estimators=50,
            max_depth=5
        )
    else:
        raise ValueError("Unsupported model type")

    # 训练模型
    model.train(features, target)
    return model


# 模型管理器夹具
@pytest.fixture
def model_manager(tmp_path):
    return ModelManager(base_path=tmp_path, device="cpu")


@pytest.fixture
def sample_metadata():
    """示例元数据"""
    return {
        "model_name": "test_model",
        "version": "1.0.0",
        "timestamp": "2023-10-01T12:00:00",
        "feature_columns": [f"feat_{i}" for i in range(10)],
        "metadata": {
            "description": "Test model for unit testing",
            "author": "Test User",
            "created_at": "2023-10-01"
        }
    }


@patch("pathlib.Path.mkdir")
@patch("joblib.dump")
def test_save_with_mocked_fs(mock_dump, mock_mkdir):
    """模拟文件系统操作测试保存流程"""
    model = NeuralNetworkModel(10)
    model.save(Path("/fake/path"), "test")
    mock_mkdir.assert_called_once()


@pytest.fixture
def sample_sequence_data():
    """生成时间序列测试数据"""
    return pd.DataFrame(np.random.randn(100, 5), columns=list("abcde"))


@pytest.fixture(params=[
    {"model_type": "lstm", "input_size": 5},
    {"model_type": "nn", "input_size": 5},
    {"model_type": "rf", "input_size": 5}
])
def configured_model(request):
    """参数化模型配置"""
    if request.param["model_type"] == "lstm":
        return LSTMModelWrapper(input_size=5, seq_length=10)
    elif request.param["model_type"] == "nn":
        return NeuralNetworkModel(input_size=5)
    return RandomForestModel()


# 覆盖设备选择分支
@pytest.mark.parametrize("device_str", ["auto", "cpu", "cuda"])
def test_device_selection(device_str):
    device = DeviceManager.get_device(device_str)
    assert isinstance(device, torch.device)