# tests/models/test_utils.py
from unittest.mock import patch

import torch

from src.models.utils import EarlyStopping, DeviceManager


def test_early_stopping():
    """测试早停机制"""
    early_stopping = EarlyStopping(patience=2)
    early_stopping(1.0)  # 第一次调用，最佳损失更新
    early_stopping(1.0)  # 第二次调用，损失未下降，计数器增加
    # 需要第三次调用，早停条件才会被触发
    early_stopping(1.0)  # 第三次调用，损失仍未下降，触发早停
    assert early_stopping.early_stop is True  # 验证早停触发


def verify_device_configuration():
    """验证设备选择逻辑"""
    # 验证自动选择逻辑
    with patch('torch.cuda.is_available', return_value=True):
        device = DeviceManager.get_device("auto")
        assert device == torch.device("cuda")

    with patch('torch.cuda.is_available', return_value=False):
        device = DeviceManager.get_device("auto")
        assert device == torch.device("cpu")

    # 验证手动选择逻辑
    device = DeviceManager.get_device("cuda")
    assert device == torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = DeviceManager.get_device("cpu")
    assert device == torch.device("cpu")


def test_device_manager_gpu_fallback(monkeypatch, caplog):
    # 模拟 torch.cuda.is_available 返回 True
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)

    # 模拟 torch.cuda.get_device_name 抛出 RuntimeError
    def mock_get_device_name(device):
        raise RuntimeError("GPU error")

    monkeypatch.setattr(torch.cuda, 'get_device_name', mock_get_device_name)

    # 获取设备
    device = DeviceManager.get_device("auto")

    # 检查是否回退到 CPU
    assert device.type == "cpu"
    assert "回退到CPU设备" in caplog.text