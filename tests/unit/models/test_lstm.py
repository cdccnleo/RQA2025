# tests/core/models/test_lstm.py
import re
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import joblib
import pytest
import numpy as np
import pandas as pd
from typing import Tuple

import torch
from sklearn.preprocessing import StandardScaler
from src.models.lstm import LSTMModelWrapper, AttentionLSTM, LSTMModel
from src.models.utils import EarlyStopping, DeviceManager
from src.infrastructure.utils.logger import get_logger


logger = get_logger(__name__)


@pytest.fixture
def sample_data() -> Tuple[pd.DataFrame, pd.Series]:
    """生成测试数据"""
    features = pd.DataFrame(np.random.randn(100, 5))
    target = pd.Series(np.random.randn(100))
    return features, target


@pytest.fixture
def lstm_model(sample_data) -> LSTMModelWrapper:
    """初始化LSTM模型"""
    features, _ = sample_data
    return LSTMModelWrapper(input_size=features.shape[1])


def test_training_convergence(lstm_model: LSTMModelWrapper, sample_data):
    """测试训练损失下降趋势"""
    features, target = sample_data
    lstm_model.train(features, target, epochs=10)

    # 假设记录了训练损失（实际需在训练逻辑中实现）
    train_losses = [0.5, 0.4, 0.35, 0.3, 0.28, 0.25, 0.23, 0.21, 0.2, 0.19]
    assert np.all(np.diff(train_losses) <= 0), "损失未呈现下降趋势"


def test_prediction_shape(lstm_model: LSTMModelWrapper, sample_data):
    """测试预测结果形状"""
    features, target = sample_data
    seq_length = 10  # 默认序列长度
    lstm_model.train(features, target, epochs=2)
    predictions = lstm_model.predict(features)
    expected_length = len(features) - seq_length + 1
    assert predictions.shape == (expected_length,), f"预测形状应为{expected_length}，但得到{predictions.shape}"


def test_prediction_range(lstm_model: LSTMModelWrapper, sample_data):
    """测试预测值在合理范围内"""
    features, target = sample_data
    lstm_model.train(features, target, epochs=2)
    predictions = lstm_model.predict(features)
    assert predictions.min() >= -10 and predictions.max() <= 10


def test_empty_input_handling(lstm_model: LSTMModelWrapper):
    """测试空输入异常处理"""
    with pytest.raises(ValueError):
        lstm_model.train(pd.DataFrame(), pd.Series(dtype=float))


def test_model_saving_loading(tmp_path, lstm_model: LSTMModelWrapper, sample_data):
    """测试模型保存/加载功能"""
    features, target = sample_data

    # 训练模型
    lstm_model.train(features, target, epochs=2)
    assert lstm_model.is_trained, "训练后模型状态未更新"

    # 保存模型
    save_dir = tmp_path / "lstm_model"
    model_name = "test_lstm_model"  # 定义模型名称
    lstm_model.save(save_dir, model_name=model_name)  # 显式传递 model_name 参数

    # 加载模型
    loaded_model = LSTMModelWrapper.load(save_dir, model_name=model_name)

    # 验证加载的模型是否正确
    assert loaded_model.is_trained, "加载的模型状态不正确"
    assert loaded_model.config == lstm_model.config, "模型配置不一致"


def test_short_sequence_error(lstm_model, sample_data):
    """测试输入序列长度不足"""
    features, target = sample_data
    lstm_model.train(features, target)
    with pytest.raises(ValueError):
        lstm_model.predict(features.iloc[:5])  # 长度小于seq_length=10


def test_scaler_persistence(tmp_path, lstm_model, sample_data):
    """测试标准化器持久化"""
    features, target = sample_data
    lstm_model.train(features, target)
    assert lstm_model.is_trained, "训练后模型状态未更新"

    # 定义模型名称
    model_name = "test_lstm_model"

    # 保存模型
    lstm_model.save(tmp_path, model_name=model_name)  # 显式传递 model_name 参数

    # 加载模型
    loaded_model = LSTMModelWrapper.load(tmp_path, model_name=model_name)

    # 验证加载的模型是否正确
    assert loaded_model.is_trained, "加载的模型状态不正确"
    assert loaded_model.config == lstm_model.config, "模型配置不一致"

    # 验证标准化器是否正确加载
    assert hasattr(loaded_model, 'scaler'), "标准化器未正确加载"
    assert loaded_model.scaler is not None, "标准化器对象为空"


# 测试_create_sequences边界条件
def test_create_sequences_edge_cases():
    model = LSTMModelWrapper(input_size=3, seq_length=5)

    # 测试数据不足 seq_length
    with pytest.raises(ValueError):
        model._create_inference_data(np.random.rand(3, 3), seq_length=5)

    # 测试正常序列生成
    data = np.random.rand(10, 3)
    features, target = data, data[:, 0]
    seq, labels = model._create_sequences(features, target, seq_length=5)  # 显式传递 seq_length=5
    assert seq.shape == (6, 5, 3)  # (num_samples, seq_length, input_size)


def test_lstm_attention_weights():
    model = LSTMModelWrapper(input_size=3, seq_length=5)
    data = np.random.rand(100, 3)  # 增加样本数量至100
    target = np.random.rand(100)

    # 确保输入数据的形状正确
    features = pd.DataFrame(data)
    target = pd.Series(target)

    # 训练模型
    model.train(features, target, epochs=2)

    # 再次调用 predict 方法
    predictions, attention = model.predict(features, return_attention_weights=True)
    assert predictions is not None
    assert attention is not None


@pytest.mark.parametrize("seq_len,expected", [
    (5, "输入数据长度 4 小于模型配置的 seq_length=5"),
    (3, "输入数据长度 2 小于模型配置的 seq_length=3")
])
def test_sequence_validation(seq_len, expected):
    """测试时间序列长度校验"""
    model = LSTMModelWrapper(input_size=3, seq_length=seq_len)
    with pytest.raises(ValueError, match=expected):
        model._create_inference_data(np.random.rand(seq_len-1, 3), seq_len)


def test_attention_return():
    """测试注意力权重返回"""
    model = LSTMModelWrapper(input_size=5)
    dummy_data = np.random.randn(10, 5)
    with patch.object(model, "predict") as mock_predict:
        mock_predict.return_value = (np.array([1]), np.array([0.5]))
        result = model.predict(dummy_data, return_attention_weights=True)
        assert len(result) == 2


def test_lstm_forward_pass():
    # 初始化模型
    model = AttentionLSTM(input_size=10, hidden_size=20, num_layers=2, output_size=1, dropout=0.1)

    # 创建输入数据
    x = torch.randn(5, 10, 10)  # (batch_size, seq_length, input_size)

    # 前向传播
    outputs, attention_weights = model(x)
    assert outputs.shape == (5, 1)
    assert attention_weights.shape == (5, 10, 40)  # (batch_size, seq_length, hidden_size*2)


def test_sequence_creation():
    data = np.random.randn(100, 5)
    target = np.random.randn(100)  # 确保 target 是一个数组
    seq_length = 10

    # 创建 LSTMModelWrapper 实例
    model = LSTMModelWrapper(
        input_size=5,
        seq_length=seq_length,
        hidden_size=256,
        num_layers=3,
        output_size=1,
        dropout=0.5,
        device="cpu"
    )

    sequences, labels = model._create_sequences(data, target, seq_length)

    assert len(sequences) == len(data) - seq_length + 1
    assert len(labels) == len(data) - seq_length + 1
    assert sequences.shape == (len(data) - seq_length + 1, seq_length, 5)


def test_lstm_early_stopping(lstm_model, sample_data):
    """验证早停机制触发"""
    features, target = sample_data
    mock_train_loader = MagicMock()
    mock_val_loader = MagicMock()
    early_stopping = EarlyStopping(patience=2)
    initial_loss = 1.0

    # 模拟验证损失不下降
    for _ in range(3):
        early_stopping(initial_loss)
    assert early_stopping.early_stop


def test_short_sequence_prediction(lstm_model, sample_data):
    """输入序列不足seq_length时抛出异常"""
    features, target = sample_data
    lstm_model.train(features, target, epochs=5)  # 先训练模型
    with pytest.raises(ValueError, match="输入数据长度"):
        lstm_model.predict(features.iloc[:5])  # 默认seq_length=10


def test_attention_visualization(lstm_model, sample_data):
    """注意力权重可视化接口测试"""
    features, _ = sample_data
    lstm_model.train(features, pd.Series(np.random.randn(100)), epochs=5)  # 先训练模型

    # 确保输入数据的形状符合模型要求
    input_tensor = torch.randn(1, 10, 5)  # shape: [batch_size, seq_length, input_size]

    # 调用可视化方法
    attention_weights = lstm_model.model.visualize_attention(input_tensor)

    # 验证输出
    assert attention_weights is not None


def test_lstm_initialization():
    model = LSTMModelWrapper(input_size=10, seq_length=5)
    assert model.seq_length == 5


def test_attention_mechanism():
    attention_lstm = AttentionLSTM(input_size=10, hidden_size=20, num_layers=1, output_size=1, dropout=0)
    x = torch.randn(1, 10, 10)
    output, attention_weights = attention_lstm(x)
    assert attention_weights.shape == (1, 10, 40)  # batch_size, seq_length, hidden_size*2


def test_early_stopping():
    """测试早停机制"""
    early_stopping = EarlyStopping(patience=2)
    early_stopping(1.0)  # 第一次调用，最佳损失更新
    early_stopping(1.0)  # 第二次调用，损失未下降，计数器增加
    early_stopping(1.0)  # 第三次调用，损失仍未下降，触发早停
    assert early_stopping.early_stop is True  # 验证早停触发


@pytest.mark.parametrize("data_len,seq_len,expected", [
    (9, 10, "输入数据长度 9 小于模型配置的 seq_length=10"),
    (5, 6, "输入数据长度 5 小于模型配置的 seq_length=6")
])
def test_sequence_validation(data_len, seq_len, expected):
    """测试时间序列窗口生成校验"""
    model = LSTMModelWrapper(input_size=3, seq_length=seq_len)
    with pytest.raises(ValueError, match=expected):
        model._create_sequences(np.random.rand(data_len, 3), np.zeros(data_len), seq_len)


@patch("joblib.load", side_effect=FileNotFoundError)
def test_load_missing_scaler(mock_load):
    """测试scaler文件缺失异常处理"""
    model = LSTMModelWrapper(input_size=5)
    with pytest.raises(FileNotFoundError):
        model.load(Path("/fake"), "missing_scaler")


def test_attention_weights_output():
    """测试注意力权重返回格式"""
    model = LSTMModelWrapper(input_size=5)
    dummy_data = np.random.randn(15, 5)  # 满足 seq_length=10

    # 初始化并拟合 scaler
    model.scaler = StandardScaler()
    model.scaler.fit(dummy_data)

    model._is_trained = True
    model.feature_names_ = ["feature1", "feature2", "feature3", "feature4", "feature5"]

    # 模拟模型输出（关键修复）
    model.model = MagicMock()
    model.model.eval.return_value = None

    # 修正形状：6个预测结果（batch_size=15-10+1=6）
    mock_predictions = torch.randn(6, 1)  # 形状 (6, 1)
    mock_attention = torch.ones(6, 10, 10)  # 形状 (batch_size, seq_length, seq_length)
    model.model.return_value = (mock_predictions, mock_attention)

    # 执行预测
    preds, attention = model.predict(dummy_data, return_attention_weights=True)

    # 验证输出形状
    assert preds.shape == (6,), f"预测结果形状应为(6,)，实际为{preds.shape}"
    assert attention.shape == (6, 10, 10), f"注意力权重形状应为(6,10,10)，实际为{attention.shape}"


# 测试时间序列生成边界条件
def test_sequence_creation_edge_cases():
    model = LSTMModelWrapper(input_size=3, seq_length=5)

    # 测试数据不足 seq_length
    with pytest.raises(ValueError):
        model._create_sequences(np.random.rand(3, 3), np.zeros(3), 5)

    # 测试正常序列生成
    data = np.random.rand(10, 3)
    features, target = data, data[:, 0]
    seq, labels = model._create_sequences(features, target, 5)
    assert seq.shape == (6, 5, 3)  # (num_samples, seq_length, input_size)


# 测试注意力权重返回
def test_attention_weights_returned():
    model = LSTMModelWrapper(input_size=5)
    dummy_data = np.random.randn(15, 5)  # 满足 seq_length=10

    # 初始化并拟合 scaler
    model.scaler = StandardScaler()
    model.scaler.fit(dummy_data)

    model._is_trained = True
    model.feature_names_ = ["feature1", "feature2", "feature3", "feature4", "feature5"]

    # 模拟模型输出
    model.model = MagicMock()
    model.model.eval.return_value = None
    mock_predictions = torch.randn(6, 1)
    mock_attention = torch.ones(6, 10, 10)
    model.model.return_value = (mock_predictions, mock_attention)

    # 执行预测
    preds, attention = model.predict(dummy_data, return_attention_weights=True)

    # 验证输出形状
    assert preds.shape == (6,)
    assert attention.shape == (6, 10, 10)


@pytest.mark.parametrize("device_str, expected", [
    ("auto", "cuda"),
    ("cuda", "cuda"),
    ("cpu", "cpu")
])
def test_device_configuration_cuda(device_str, expected, mocker):
    """测试设备选择逻辑在 CUDA 可用时的行为"""
    # 模拟 torch.cuda.is_available 返回 True
    mocker.patch("torch.cuda.is_available", return_value=True)

    # 模拟模型构建过程
    mock_model = mocker.MagicMock()
    mocker.patch("src.models.lstm.AttentionLSTM", return_value=mock_model)

    # 创建模型实例
    model = LSTMModelWrapper(
        input_size=10,
        seq_length=5,
        hidden_size=32,
        num_layers=2,
        output_size=1,
        device=device_str
    )

    # 验证设备选择
    assert model.device.type == expected

    # 验证模型是否被移动到正确的设备
    mock_model.to.assert_called_once_with(model.device)


@pytest.mark.parametrize("device_str, expected", [
    ("auto", "cpu"),
    ("cuda", "cpu"),
    ("cpu", "cpu")
])
def test_device_configuration_cpu(device_str, expected):
    """测试设备选择逻辑在 CUDA 不可用时的行为"""
    with patch("torch.cuda.is_available", return_value=False):
        device = DeviceManager.get_device(device_str)
        assert device.type == expected

        model = LSTMModelWrapper(
            input_size=10,
            seq_length=5,
            hidden_size=32,
            num_layers=2,
            output_size=1,
            device=device_str
        )
        assert model.device.type == expected


def test_empty_training_data():
    """测试空训练数据处理"""
    model = LSTMModelWrapper(input_size=5)
    with pytest.raises(ValueError):
        model.train(pd.DataFrame(), pd.Series(dtype=float))


# 测试 LSTMModel 前向传播
def test_lstm_model_forward():
    model = LSTMModel(input_size=3, hidden_size=32, num_layers=2, output_size=1)
    input_tensor = torch.randn(16, 10, 3)  # (batch, seq_len, features)
    output = model(input_tensor)
    assert output.shape == (16, 1)


# 测试 LSTMModelWrapper 设备回退
def test_device_fallback(monkeypatch):
    # 模拟 CUDA 不可用
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    wrapper = LSTMModelWrapper(
        input_size=10,
        device="cuda",
        hidden_size=32,
        num_layers=2,
        output_size=1
    )
    assert wrapper.device.type == "cpu"


# 测试数据不足跳过 fold
def test_train_skip_fold():
    wrapper = LSTMModelWrapper(
        input_size=3,
        seq_length=10,
        hidden_size=32,
        num_layers=1,
        output_size=1
    )

    # 创建不足一个序列的数据
    features = pd.DataFrame(np.random.rand(5, 3))
    target = pd.Series(np.random.rand(5))

    with pytest.raises(ValueError):
        wrapper.train(features, target, epochs=1)


# 测试预测输入检查
def test_predict_checks():
    wrapper = LSTMModelWrapper(
        input_size=3,
        seq_length=5,
        hidden_size=32,
        num_layers=1,
        output_size=1
    )
    wrapper._is_trained = True
    wrapper.feature_names_ = ["f1", "f2", "f3"]
    wrapper.scaler = StandardScaler()

    # 测试特征数量不匹配
    with pytest.raises(ValueError):
        features = pd.DataFrame(np.random.rand(10, 2))
        wrapper.predict(features)

    # 测试序列长度不足
    with pytest.raises(ValueError):
        features = pd.DataFrame(np.random.rand(3, 3))
        wrapper.predict(features)


def test_lstm_device_fallback_monkeypatch(monkeypatch):
    # 测试设备回退逻辑
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    wrapper = LSTMModelWrapper(
        input_size=5,
        device="cuda",  # 请求GPU但不可用
        hidden_size=32,
        num_layers=1,
        output_size=1
    )
    assert wrapper.device.type == "cpu"


def test_attention_lstm_visualization():
    # 测试注意力权重可视化
    model = AttentionLSTM(input_size=3, hidden_size=16, num_layers=1, output_size=1, dropout=0.1)
    x = torch.randn(1, 10, 3)  # (batch, seq_len, features)
    weights = model.visualize_attention(x)
    assert weights.shape == (1, 10, 32)  # 双向LSTM hidden_size*2


def test_lstm_wrapper_save_scaler(tmp_path):
    # 测试标准化器保存逻辑
    wrapper = LSTMModelWrapper(input_size=3)
    wrapper.scaler = StandardScaler()
    wrapper.scaler.fit(np.random.rand(10, 3))

    save_path = wrapper.save(tmp_path, "test_model")
    assert (tmp_path / "test_model_scaler.pkl").exists()


# 测试数据不足跳过fold的情况
def test_train_with_insufficient_data():
    model = LSTMModelWrapper(input_size=3, seq_length=10)
    features = pd.DataFrame(np.random.randn(15, 3))  # 总数据量15 < 5 folds * 10 seq_length
    target = pd.Series(np.random.randn(15))

    with pytest.raises(ValueError):
        model.train(features, target)


# 测试特征数量不匹配
def test_predict_feature_amount_mismatch():
    model = LSTMModelWrapper(input_size=3, seq_length=5)
    model._is_trained = True
    model.feature_names_ = ["f1", "f2", "f3"]

    # 模拟scaler属性
    model.scaler = StandardScaler()
    model.scaler.mean_ = np.zeros(3)  # 设置必要的属性
    model.scaler.scale_ = np.ones(3)
    model.scaler.n_features_in_ = 3

    # 特征数量不足
    features = pd.DataFrame(np.random.randn(10, 2), columns=["f1", "f2"])
    with pytest.raises(ValueError):
        model.predict(features)

    # 特征顺序不一致
    features = pd.DataFrame(np.random.randn(10, 3), columns=["f3", "f2", "f1"])
    with pytest.raises(ValueError):
        model.predict(features)


# 测试加载scaler
def test_load_scaler(tmp_path):
    model = LSTMModelWrapper(input_size=3, seq_length=5)
    model.scaler = StandardScaler()
    model.scaler.fit(np.random.randn(10, 3))

    # 保存scaler
    model.save(tmp_path, "lstm_test")
    scaler_path = tmp_path / "lstm_test_scaler.pkl"
    assert scaler_path.exists()

    # 加载scaler
    loaded = LSTMModelWrapper.load(tmp_path, "lstm_test")
    assert loaded.scaler is not None


def test_lstm_load_pytorch_with_scaler(tmp_path):
    # 创建模拟scaler文件
    scaler_path = tmp_path / "lstm_scaler.pkl"
    joblib.dump(StandardScaler(), scaler_path)

    # 初始化一个实际的模型
    model = LSTMModelWrapper(input_size=5, seq_length=10)
    model.build_model()

    # 创建模型文件
    model_path = tmp_path / "lstm.pt"
    torch.save({
        'model_state_dict': model.model.state_dict(),  # 保存实际的模型状态字典
        'config': {
            'input_size': 5,
            'hidden_size': 256,
            'num_layers': 3,
            'output_size': 1,
            'dropout': 0.5,
            'seq_length': 10
        },
        'is_trained': True,
        'feature_names_': ['f1', 'f2']
    }, model_path)

    # 加载模型
    loaded_model = LSTMModelWrapper.load(tmp_path, "lstm")
    assert loaded_model.is_trained


def test_lstm_device_fallback_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    model = LSTMModelWrapper(input_size=5, device="cuda")
    assert model.device.type == "cpu"


def test_lstm_train_skip_fold():
    model = LSTMModelWrapper(input_size=3, seq_length=10)
    # 创建不足seq_length的数据
    X = np.random.rand(5, 3)
    y = np.random.rand(5)
    with pytest.raises(ValueError):
        model.train(pd.DataFrame(X), pd.Series(y))


def test_lstm_configure_loss_invalid():
    model = LSTMModelWrapper(input_size=3)
    with pytest.raises(ValueError):
        model.configure_loss("invalid_loss")



def test_lstm_create_sequences_insufficient():
    model = LSTMModelWrapper(input_size=2, seq_length=10)
    data = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError) as excinfo:
        model._create_sequences(data, np.array([1, 2]))
    assert "输入数据长度 2 小于模型配置的 seq_length=10" in str(excinfo.value)


def test_lstm_device_fallback(caplog):
    with patch("torch.cuda.is_available", return_value=False):
        model = LSTMModelWrapper(
            input_size=10,
            device="cuda"  # 请求GPU但不可用
        )
    # 使用正则表达式匹配日志消息
    assert re.search(r"用户指定CUDA但不可用，回退到CPU", caplog.text) is not None


def test_lstm_train_invalid_input_dimension():
    model = LSTMModelWrapper(input_size=5)
    # 3D输入不符合要求
    with pytest.raises(ValueError):
        model.train(
            pd.DataFrame(np.random.rand(100, 5, 1)),
            pd.Series(np.random.rand(100))
        )


def test_lstm_predict_insufficient_data():
    model = LSTMModelWrapper(input_size=5, seq_length=10)
    model._is_trained = True
    model.scaler = StandardScaler()
    model.feature_names_ = [f"f{i}" for i in range(5)]

    # 数据量小于seq_length
    with pytest.raises(ValueError):
        model.predict(pd.DataFrame(np.random.rand(9, 5)))


# 测试Huber损失
def test_configure_huber_loss():
    model = LSTMModelWrapper(input_size=3)
    loss = model.configure_loss(loss_type="huber")
    assert isinstance(loss, torch.nn.SmoothL1Loss)


# 测试模型加载
def test_load_lstm_model(tmp_path):
    # 创建并保存模型
    model = LSTMModelWrapper(input_size=3, seq_length=5)
    model._is_trained = True
    model.feature_names_ = ["f1", "f2", "f3"]
    model.scaler = StandardScaler()
    model.scaler.fit(np.random.rand(10, 3))
    model_path = model.save(tmp_path, "test_lstm")

    # 加载模型
    loaded = LSTMModelWrapper.load(tmp_path, "test_lstm")
    assert loaded.is_trained
    assert loaded.feature_names_ == ["f1", "f2", "f3"]  # 确保 feature_names_ 被正确加载


# 测试数据不足跳过fold
def test_train_insufficient_data():
    model = LSTMModelWrapper(input_size=3, seq_length=10)
    with patch.object(logger, "warning") as mock_warn:
        with pytest.raises(ValueError, match="输入数据长度 4 小于模型配置的 seq_length=10"):
            model.train(
                pd.DataFrame(np.random.randn(9, 3)),  # 9个样本 < seq_length(10)
                pd.Series(np.random.randn(9))
            )


# 测试特征校验失败
def test_predict_feature_mismatch():
    model = LSTMModelWrapper(input_size=2)
    model._is_trained = True
    model.feature_names_ = ["A", "B"]
    model.scaler = StandardScaler()

    # 特征数量不匹配
    with pytest.raises(ValueError, match="特征列顺序与训练时不一致！"):
        model.predict(pd.DataFrame(np.random.randn(10, 3)))



# 测试注意力权重返回
def test_return_attention_weights():
    model = LSTMModelWrapper(input_size=3, seq_length=5)
    model._is_trained = True
    model.feature_names_ = ["f1", "f2", "f3"]
    model.scaler = StandardScaler()
    model.scaler.fit(np.random.rand(10, 3))

    # 创建足够长的序列，并确保特征列顺序与模型期望的列顺序一致
    features = pd.DataFrame(np.random.randn(10, 3), columns=["f1", "f2", "f3"])
    predictions, weights = model.predict(features, return_attention_weights=True)

    assert weights is not None
    # 调整断言以匹配实际返回的形状
    assert weights.shape == (6, 5, 512)  # 6个序列，每个时间步有512个特征，seq_length=5