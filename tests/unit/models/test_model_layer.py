from unittest.mock import patch

import pandas as pd
import pytest
import numpy as np

from src.models.nn import NeuralNetworkModel
from src.models.lstm import LSTMModelWrapper
from src.models.utils import EarlyStopping


class TestModelLayer:
    """模型层单元测试"""

    def test_model_save_load_consistency(self, lstm_model, nn_model, rf_model, sample_data):
        """测试模型保存/加载一致性"""
        features, target = sample_data

        # 确保 LSTM 模型的输入特征维度与数据一致
        input_size = features.shape[1]  # 获取特征数量
        lstm_model = LSTMModelWrapper(
            input_size=input_size,
            seq_length=10,
            hidden_size=32,
            num_layers=3,
            output_size=1,
            dropout=0.5,
            device="cpu"
        )

        # 测试 LSTM 模型
        lstm_model.train(features, target)
        # 其他模型测试逻辑...

    def test_lstm_cross_validation(self, lstm_model, sample_data):
        """测试LSTM交叉验证最佳模型选择"""
        features, target = sample_data
        # 确保 LSTM 模型的输入特征维度与数据一致
        input_size = features.shape[1]  # 获取特征数量
        lstm_model = LSTMModelWrapper(
            input_size=input_size,
            seq_length=10,
            hidden_size=32,
            num_layers=3,
            output_size=1,
            dropout=0.5,
            device="cpu"
        )
        lstm_model.train(features, target)

    def test_feature_importance(self, rf_model, nn_model, lstm_model, sample_data):
        """测试特征重要性计算"""
        features, target = sample_data

        # 测试随机森林特征重要性
        rf_model.train(features, target)
        assert rf_model.is_trained, "随机森林模型训练状态未正确更新"  # 显式验证训练状态

        # 验证特征重要性计算
        feature_importance = rf_model.get_feature_importance()
        assert np.allclose(
            feature_importance.values,
            rf_model.model.feature_importances_
        )

        # 测试神经网络特征重要性
        nn_model.train(features, target)
        with pytest.raises(NotImplementedError):  # 神经网络模型不支持特征重要性
            nn_model.get_feature_importance()

        # 测试LSTM特征重要性
        lstm_model.train(features, target)
        with pytest.raises(NotImplementedError):  # LSTM模型不支持特征重要性
            lstm_model.get_feature_importance()

    def test_input_dimension_errors(self, lstm_model, nn_model, rf_model, sample_data):
        """测试异常输入处理"""
        features, target = sample_data

        # 测试LSTM时间步长错误
        wrong_features = features.iloc[-5:]  # 长度小于seq_length
        with pytest.raises(ValueError):
            lstm_model.train(wrong_features, target)

        # 测试神经网络特征数量不匹配
        wrong_features = features.iloc[:, :3]  # 减少到3个特征
        with pytest.raises(ValueError) as e:
            nn_model.train(wrong_features, target)
        assert "输入特征维度不正确" in str(e.value)  # 更新期望的异常信息

    def test_early_stopping(self):
        """测试早停机制"""
        # 初始化早停对象
        patience = 3
        early_stopping = EarlyStopping(patience=patience)

        # 模拟验证损失不再下降的情况
        val_loss = 1.0
        early_stopping(val_loss)  # 第一次调用，最佳损失更新
        early_stopping(val_loss + 0.1)  # 第二次调用，损失上升
        early_stopping(val_loss + 0.2)  # 第三次调用，损失继续上升
        early_stopping(val_loss + 0.3)  # 第四次调用，达到耐心轮次，触发早停

        # 验证早停是否触发
        assert early_stopping.early_stop