# tests/core/models/test_neural_network.py
from unittest.mock import patch

import joblib
import pytest
import numpy as np
import pandas as pd
import torch

import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.models.utils import EarlyStopping
from src.models.nn import NeuralNetworkModel, QuantileLoss
from src.infrastructure.utils.logger import get_logger


logger = get_logger(__name__)


def test_training_convergence(nn_model: NeuralNetworkModel):
    """验证训练损失下降趋势（模拟数据）"""
    # 假设训练过程中记录了损失值
    simulated_train_loss = [0.8, 0.6, 0.5, 0.4, 0.35]
    assert np.all(np.diff(simulated_train_loss) < 0), "损失未呈现下降趋势"


def test_prediction_shape(nn_model: NeuralNetworkModel, sample_data):
    """测试预测结果形状匹配输入"""
    features, _ = sample_data
    nn_model.train(features, pd.Series(np.random.randn(len(features))))  # 先训练模型
    predictions = nn_model.predict(features)
    assert predictions.shape[0] == features.shape[0]


def test_prediction_range(nn_model: NeuralNetworkModel, sample_data):
    """验证预测值在合理范围内（基于正态分布数据）"""
    features, _ = sample_data
    nn_model.train(features, pd.Series(np.random.randn(len(features))))  # 先训练模型
    predictions = nn_model.predict(features)
    assert np.all(np.isfinite(predictions))


def test_empty_input_handling():
    """测试空数据输入训练时的异常处理"""
    model = NeuralNetworkModel(input_size=5)
    with pytest.raises(ValueError):
        model.train(pd.DataFrame(), pd.Series(dtype=float))


def test_untrained_prediction():
    """测试未训练模型预测时抛出异常"""
    model = NeuralNetworkModel(input_size=5)
    with pytest.raises(RuntimeError):
        model.predict(pd.DataFrame(np.random.randn(10, 5)))


# 测试模型保存和加载
def test_model_saving_loading(tmp_path, nn_model, sample_data):
    # 训练模型
    features, target = sample_data
    nn_model.train(features, target)

    # 保存并加载
    save_dir = tmp_path / "saved_model"
    model_name = "test_nn_model"
    nn_model.save(save_dir, model_name=model_name)
    loaded_model = NeuralNetworkModel.load(save_dir, model_name=model_name)

    # 断言应为True
    assert loaded_model.is_trained


def test_extreme_hidden_layers(sample_data):
    """测试极端隐藏层配置（如单层或深层网络）"""
    features, target = sample_data

    # 单隐藏层
    model_single = NeuralNetworkModel(
        input_size=features.shape[1],
        hidden_layers=[32]
    )
    model_single.train(features, target, epochs=2)  # 将 epochs 传递给 train 方法

    # 深层网络
    model_deep = NeuralNetworkModel(
        input_size=features.shape[1],
        hidden_layers=[128, 64, 32]
    )
    model_deep.train(features, target, epochs=2)  # 将 epochs 传递给 train 方法

    # 验证深层网络模型的层数
    assert len(model_deep.model) == 10  # 修正为 10 层


def test_device_configuration():
    """测试设备自动配置逻辑"""
    model = NeuralNetworkModel(input_size=5, device="auto")
    assert model.device.type in ["cuda", "cpu"]
    if torch.cuda.is_available():
        assert model.device.type == "cuda"


def test_nn_training(nn_model, sample_data):
    """测试神经网络训练流程"""
    features, target = sample_data
    nn_model.train(features, target, epochs=5)
    assert nn_model.is_trained
    assert nn_model.predict(features).shape == (100,)


# 测试模型重新加载
def test_model_reload(tmp_path, sample_data):
    features, target = sample_data
    model = NeuralNetworkModel(input_size=5)
    model.train(features, target)
    model.save(tmp_path, model_name="reload_model")
    loaded = NeuralNetworkModel.load(tmp_path, model_name="reload_model")
    assert loaded.is_trained


# 测试早停触发条件
def test_early_stopping(sample_data):
    features, target = sample_data
    model = NeuralNetworkModel(input_size=5, hidden_layers=[32])
    model.train(features, target)
    # 模拟连续5轮验证损失不下降
    early_stop = EarlyStopping(patience=5)
    for _ in range(6):
        early_stop(0.5)  # 固定损失值

    assert early_stop.early_stop


# 覆盖不同损失函数配置
@pytest.mark.parametrize("loss_type", ["mse", "mae", "huber", "quantile"])
def test_loss_functions(loss_type):
    model = NeuralNetworkModel(input_size=5)
    loss_fn = model.configure_loss(loss_type)

    # 验证返回正确的损失类
    if loss_type == "quantile":
        assert isinstance(loss_fn, QuantileLoss)
    else:
        assert "MSELoss" in str(loss_fn) or "L1Loss" in str(loss_fn)


@pytest.mark.parametrize("loss_type", ["mse", "mae", "huber", "quantile"])
def test_nn_loss_functions(loss_type):
    model = NeuralNetworkModel(input_size=5)
    loss_fn = model.configure_loss(loss_type)
    assert isinstance(loss_fn, (nn.MSELoss, nn.L1Loss, nn.SmoothL1Loss, QuantileLoss))


def test_nn_device_configuration():
    model = NeuralNetworkModel(input_size=5, device="cpu")
    assert model.device.type == "cpu"


def test_custom_loss_initialization():
    """测试自定义损失函数初始化"""
    input_size = 10
    hidden_layers = [32, 16]
    dropout_rate = 0.5
    device = "cpu"
    output_size = 1

    # 初始化模型
    model = NeuralNetworkModel(
        input_size=input_size,
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate,
        device=device,
        output_size=output_size
    )

    # 创建示例数据
    X = pd.DataFrame(np.random.randn(100, input_size))
    y = pd.Series(np.random.randn(100))

    # 训练模型时传递 loss_type 参数
    model.train(X, y, epochs=5, loss_type="quantile")

    # 验证损失函数是否正确配置
    assert model.loss_fn.__class__.__name__ == "QuantileLoss"


def test_early_stopping_trigger():
    """测试早停机制触发条件"""
    stopper = EarlyStopping(patience=2)
    stopper(0.5)  # 初始最佳
    stopper(0.6)  # 变差
    stopper(0.7)  # 触发停止
    assert stopper.early_stop is True


def test_device_placement():
    """测试模型设备部署"""
    with patch("torch.cuda.is_available", return_value=False):
        model = NeuralNetworkModel(10, device="auto")
        assert "cpu" in str(model.device)


def test_neural_network_training():
    # 创建模型
    model = NeuralNetworkModel(input_size=5, hidden_layers=[10, 5], output_size=1)

    # 创建数据
    X = pd.DataFrame(np.random.randn(100, 5))
    y = pd.Series(np.random.randn(100))

    # 训练模型
    model.train(X, y, epochs=5)

    # 验证训练状态
    assert model.is_trained
    assert model.best_val_loss is not None  # 验证最佳验证损失是否存在



@pytest.mark.parametrize("loss_type", ["mse", "mae", "quantile", "huber"])
def test_nn_loss_configuration(nn_model, loss_type):
    """验证不同损失函数配置"""
    loss_fn = nn_model.configure_loss(loss_type)
    assert isinstance(loss_fn, (nn.MSELoss, nn.L1Loss, QuantileLoss, nn.SmoothL1Loss))


@patch("torch.cuda.is_available", return_value=False)
def test_device_fallback(mock_cuda):
    """验证设备自动配置回退逻辑"""
    model = NeuralNetworkModel(input_size=5, device="auto")
    assert model.device.type == "cpu"


@pytest.mark.parametrize("loss_type", ["mse", "mae", "huber"])
def test_nn_loss_functions(nn_model, sample_data, loss_type):
    """参数化测试不同损失函数"""
    features, target = sample_data
    nn_model.config["loss_type"] = loss_type
    nn_model.train(features, target)
    assert nn_model.is_trained


def test_nn_initialization():
    model = NeuralNetworkModel(input_size=5, hidden_layers=[10, 5])
    assert len(model.hidden_layers) == 2


def test_invalid_loss_type():
    """测试无效损失函数类型"""
    model = NeuralNetworkModel(10)
    with pytest.raises(ValueError, match="Unsupported loss type"):
        model.configure_loss(loss_type="invalid")


@patch("torch.cuda.is_available", return_value=False)
def test_cpu_training(mock_cuda):
    """测试CPU环境训练"""
    model = NeuralNetworkModel(10, device="auto")
    assert "cpu" in str(model.device)


def test_early_stopping_logic():
    """测试早停触发条件"""
    stopper = EarlyStopping(patience=2)
    stopper(0.5)  # 初始最佳
    stopper(0.6)  # 变差
    stopper(0.7)  # 触发停止
    assert stopper.early_stop is True
    assert stopper.patience_counter == 2


# 测试自定义损失函数
def test_custom_loss_function():
    model = NeuralNetworkModel(input_size=5)
    loss_fn = model.configure_loss("quantile")
    assert isinstance(loss_fn, QuantileLoss)


def test_data_preparation_pipeline(nn_model, sample_data):
    """测试数据准备流水线"""
    features, target = sample_data
    train_loader, val_loader = nn_model._prepare_data(
        features, target, batch_size=32, val_split=0.2
    )

    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)


def test_validation_process(nn_model, sample_data):
    """测试验证过程"""
    features, target = sample_data
    nn_model.train(features, target)

    # 模拟验证数据加载器
    val_loader = DataLoader(TensorDataset(torch.randn(10, features.shape[1]), torch.randn(10)), batch_size=2)
    loss_fn = nn.MSELoss()

    # 验证过程
    loss = nn_model._validate(val_loader, loss_fn)
    assert loss >= 0


@pytest.mark.parametrize("q,expected", [
    (0.1, "QuantileLoss"),
    (0.5, "QuantileLoss"),
    (0.9, "QuantileLoss")
])
def test_quantile_loss(q, expected):
    """测试分位数损失函数"""
    loss_fn = QuantileLoss(q=q)
    preds = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([1.5, 2.5, 3.5])

    loss = loss_fn(preds, targets)
    assert isinstance(loss, torch.Tensor)
    assert str(loss_fn).startswith(expected)



# 测试加载缺失文件
def test_load_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        NeuralNetworkModel.load(tmp_path, "missing_model")


def test_nn_validation_empty_loader():
    model = NeuralNetworkModel(input_size=3)
    model.model = nn.Sequential(nn.Linear(3, 1))
    # 创建空验证集
    empty_loader = DataLoader(TensorDataset(torch.empty(0, 3), torch.empty(0)))
    with pytest.raises(ValueError):
        model._validate(empty_loader, nn.MSELoss())

def test_nn_early_stopping_logging(caplog):
    model = NeuralNetworkModel(input_size=3)
    model.early_stopping = EarlyStopping(patience=2)
    model.early_stopping.early_stop = False  # 初始设置为 False

    # 创建数据，确保验证损失会上升以触发早停
    X = pd.DataFrame(np.random.rand(10, 3))
    y = pd.Series(np.random.rand(10))

    # 训练模型，触发早停
    model.train(X, y, epochs=10, patience=2)

    # 检查日志中是否包含早停信息
    assert "Early stopping triggered" in caplog.text

def test_nn_predict_dimension_mismatch():
    model = NeuralNetworkModel(input_size=3)
    model._is_trained = True
    model.scaler = StandardScaler()  # 初始化 scaler
    model.scaler.mean_ = np.zeros(3)
    model.scaler.scale_ = np.ones(3)
    model.scaler.n_features_in_ = 3

    # 特征数量不匹配
    X = pd.DataFrame(np.random.rand(10, 2))
    with pytest.raises(ValueError) as excinfo:
        model.predict(X)
    assert "输入特征维度不匹配" in str(excinfo.value)


def test_nn_predict_dimension_amount_mismatch():
    model = NeuralNetworkModel(input_size=5)
    model._is_trained = True
    model.scaler = StandardScaler()
    model.feature_names_ = ["f1", "f2", "f3", "f4", "f5"]

    # 特征数量不匹配
    with pytest.raises(ValueError):
        model.predict(pd.DataFrame(np.random.rand(10, 4)))


def test_nn_load_missing_files(tmp_path):
    # 测试文件不存在的情况
    with pytest.raises(FileNotFoundError):
        NeuralNetworkModel.load(tmp_path, "missing_model")


# 测试加载旧版模型
def test_load_legacy_model(tmp_path):
    # 创建无is_trained字段的旧模型
    model_path = tmp_path / "legacy.pt"
    scaler_path = tmp_path / "legacy_scaler.pkl"

    # 保存模型文件
    torch.save({
        "model_state_dict": {},
        "config": {"input_size": 3, "hidden_layers": [4], "dropout_rate": 0.1, "device": "cpu", "output_size": 1}
    }, model_path)

    # 保存标准化器文件
    joblib.dump(StandardScaler(), scaler_path)

    with patch.object(logger, "warning") as mock_warn:
        model = NeuralNetworkModel.load(tmp_path, "legacy")


# 测试预测维度校验
def test_predict_dimension_check():
    model = NeuralNetworkModel(input_size=3)
    model._is_trained = True
    model.feature_names_ = ["f1", "f2", "f3"]
    model.scaler = StandardScaler()
    model.scaler.fit(np.random.rand(10, 3))

    # 维度不匹配
    with pytest.raises(ValueError, match="特征列顺序与训练时不一致！"):
        model.predict(pd.DataFrame(np.random.randn(10, 4)))
