#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度学习模型神经网络测试
覆盖LSTMPredictor和Autoencoder类的各种场景
"""

import sys
import importlib
from pathlib import Path
import pytest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

# 确保Python路径正确配置
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
project_root_str = str(project_root)
src_path_str = str(project_root / "src")

if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

# 动态导入深度学习模型模块
try:
    dl_models_module = importlib.import_module('src.monitoring.ai.dl_models')
    LSTMPredictor = getattr(dl_models_module, 'LSTMPredictor', None)
    Autoencoder = getattr(dl_models_module, 'Autoencoder', None)
    
    if LSTMPredictor is None:
        pytest.skip("深度学习模型模块不可用", allow_module_level=True)
except ImportError:
    pytest.skip("深度学习模型模块导入失败", allow_module_level=True)


class TestLSTMPredictor:
    """测试LSTMPredictor类"""

    def test_lstm_predictor_init_default(self):
        """测试LSTMPredictor默认初始化"""
        model = LSTMPredictor()
        
        assert model.input_size == 1
        assert model.hidden_size == 64
        assert model.num_layers == 2
        assert model.output_size == 1
        assert isinstance(model.lstm, nn.LSTM)
        assert isinstance(model.dropout, nn.Dropout)
        assert isinstance(model.fc, nn.Linear)

    def test_lstm_predictor_init_custom(self):
        """测试LSTMPredictor自定义初始化"""
        model = LSTMPredictor(
            input_size=10,
            hidden_size=128,
            num_layers=3,
            output_size=5,
            dropout=0.3
        )
        
        assert model.input_size == 10
        assert model.hidden_size == 128
        assert model.num_layers == 3
        assert model.output_size == 5

    def test_lstm_predictor_forward_single_batch(self):
        """测试LSTMPredictor前向传播（单批次）"""
        model = LSTMPredictor(input_size=1, hidden_size=32, num_layers=1, output_size=1)
        model.eval()  # 设置为评估模式
        
        # 创建输入：batch_size=1, seq_length=10, input_size=1
        x = torch.randn(1, 10, 1)
        
        output = model(x)
        
        assert output.shape == (1, 1)  # batch_size=1, output_size=1

    def test_lstm_predictor_forward_batch(self):
        """测试LSTMPredictor前向传播（多批次）"""
        model = LSTMPredictor(input_size=1, hidden_size=32, num_layers=1, output_size=1)
        model.eval()
        
        # 创建输入：batch_size=5, seq_length=10, input_size=1
        x = torch.randn(5, 10, 1)
        
        output = model(x)
        
        assert output.shape == (5, 1)  # batch_size=5, output_size=1

    def test_lstm_predictor_forward_multi_input(self):
        """测试LSTMPredictor前向传播（多输入特征）"""
        model = LSTMPredictor(input_size=5, hidden_size=32, num_layers=1, output_size=3)
        model.eval()
        
        # 创建输入：batch_size=2, seq_length=10, input_size=5
        x = torch.randn(2, 10, 5)
        
        output = model(x)
        
        assert output.shape == (2, 3)  # batch_size=2, output_size=3

    def test_lstm_predictor_forward_multi_output(self):
        """测试LSTMPredictor前向传播（多输出）"""
        model = LSTMPredictor(input_size=1, hidden_size=32, num_layers=1, output_size=5)
        model.eval()
        
        # 创建输入：batch_size=3, seq_length=10, input_size=1
        x = torch.randn(3, 10, 1)
        
        output = model(x)
        
        assert output.shape == (3, 5)  # batch_size=3, output_size=5

    def test_lstm_predictor_forward_with_dropout(self):
        """测试LSTMPredictor前向传播（带dropout）"""
        model = LSTMPredictor(input_size=1, hidden_size=32, num_layers=2, output_size=1, dropout=0.5)
        model.train()  # 训练模式下dropout生效
        
        x = torch.randn(2, 10, 1)
        
        output = model(x)
        
        assert output.shape == (2, 1)

    def test_lstm_predictor_forward_no_dropout_single_layer(self):
        """测试LSTMPredictor前向传播（单层无dropout）"""
        model = LSTMPredictor(input_size=1, hidden_size=32, num_layers=1, output_size=1, dropout=0.5)
        # 单层时dropout应为0
        
        x = torch.randn(1, 10, 1)
        
        output = model(x)
        
        assert output.shape == (1, 1)

    def test_lstm_predictor_forward_different_seq_lengths(self):
        """测试LSTMPredictor前向传播（不同序列长度）"""
        model = LSTMPredictor(input_size=1, hidden_size=32, num_layers=1, output_size=1)
        model.eval()
        
        # 测试不同序列长度
        for seq_length in [5, 10, 20, 50]:
            x = torch.randn(1, seq_length, 1)
            output = model(x)
            assert output.shape == (1, 1)


class TestAutoencoder:
    """测试Autoencoder类"""

    def test_autoencoder_init_default(self):
        """测试Autoencoder默认初始化"""
        model = Autoencoder()
        
        assert model.input_size == 10
        assert model.encoding_size == 5
        assert isinstance(model.encoder, nn.Sequential)
        assert isinstance(model.decoder, nn.Sequential)

    def test_autoencoder_init_custom(self):
        """测试Autoencoder自定义初始化"""
        model = Autoencoder(input_size=20, encoding_size=10)
        
        assert model.input_size == 20
        assert model.encoding_size == 10

    def test_autoencoder_forward_single_batch(self):
        """测试Autoencoder前向传播（单批次）"""
        model = Autoencoder(input_size=10, encoding_size=5)
        model.eval()
        
        # 创建输入：batch_size=1, input_size=10
        x = torch.randn(1, 10)
        
        output = model(x)
        
        assert output.shape == (1, 10)  # 输出应该与输入相同维度

    def test_autoencoder_forward_batch(self):
        """测试Autoencoder前向传播（多批次）"""
        model = Autoencoder(input_size=10, encoding_size=5)
        model.eval()
        
        # 创建输入：batch_size=5, input_size=10
        x = torch.randn(5, 10)
        
        output = model(x)
        
        assert output.shape == (5, 10)

    def test_autoencoder_forward_reconstruction(self):
        """测试Autoencoder重构能力"""
        model = Autoencoder(input_size=10, encoding_size=5)
        model.eval()
        
        # 创建输入
        x = torch.randn(3, 10)
        
        # 前向传播
        reconstructed = model(x)
        
        # 输出维度应该与输入相同
        assert reconstructed.shape == x.shape

    def test_autoencoder_forward_encoding(self):
        """测试Autoencoder编码过程"""
        model = Autoencoder(input_size=10, encoding_size=5)
        model.eval()
        
        x = torch.randn(2, 10)
        
        # 直接调用编码器
        encoded = model.encoder(x)
        
        # 编码后的维度应该是 encoding_size
        assert encoded.shape == (2, 5)

    def test_autoencoder_forward_decoding(self):
        """测试Autoencoder解码过程"""
        model = Autoencoder(input_size=10, encoding_size=5)
        model.eval()
        
        # 创建编码后的数据
        encoded = torch.randn(2, 5)
        
        # 解码
        decoded = model.decoder(encoded)
        
        # 解码后的维度应该是 input_size
        assert decoded.shape == (2, 10)

    def test_autoencoder_forward_different_input_sizes(self):
        """测试Autoencoder前向传播（不同输入大小）"""
        for input_size, encoding_size in [(10, 5), (20, 10), (50, 25)]:
            model = Autoencoder(input_size=input_size, encoding_size=encoding_size)
            model.eval()
            
            x = torch.randn(1, input_size)
            output = model(x)
            
            assert output.shape == (1, input_size)

    def test_autoencoder_forward_forward_method(self):
        """测试Autoencoder的forward方法"""
        model = Autoencoder(input_size=10, encoding_size=5)
        model.eval()
        
        x = torch.randn(1, 10)
        
        # 使用forward方法
        output = model.forward(x)
        
        assert output.shape == (1, 10)

    def test_autoencoder_forward_training_mode(self):
        """测试Autoencoder训练模式"""
        model = Autoencoder(input_size=10, encoding_size=5)
        model.train()  # 设置为训练模式
        
        x = torch.randn(2, 10)
        
        output = model(x)
        
        assert output.shape == (2, 10)

    def test_autoencoder_forward_with_nan_handling(self):
        """测试Autoencoder处理NaN值"""
        model = Autoencoder(input_size=10, encoding_size=5)
        model.eval()
        
        # 创建包含NaN的输入
        x = torch.randn(1, 10)
        x[0, 0] = float('nan')
        
        # 应该能够处理NaN或抛出异常
        try:
            output = model(x)
            # 如果成功，检查输出
            assert output.shape == (1, 10)
        except RuntimeError:
            # NaN可能在某些情况下导致错误，这也是预期的
            pass

    def test_autoencoder_forward_with_zero_input(self):
        """测试Autoencoder处理零输入"""
        model = Autoencoder(input_size=10, encoding_size=5)
        model.eval()
        
        x = torch.zeros(1, 10)
        
        output = model(x)
        
        assert output.shape == (1, 10)

    def test_autoencoder_forward_large_batch(self):
        """测试Autoencoder处理大批次"""
        model = Autoencoder(input_size=10, encoding_size=5)
        model.eval()
        
        # 大批次
        x = torch.randn(100, 10)
        
        output = model(x)
        
        assert output.shape == (100, 10)

    def test_autoencoder_encoder_structure(self):
        """测试Autoencoder编码器结构"""
        model = Autoencoder(input_size=10, encoding_size=5)
        
        # 检查编码器层数
        assert len(model.encoder) >= 2  # 至少应该有2层

    def test_autoencoder_decoder_structure(self):
        """测试Autoencoder解码器结构"""
        model = Autoencoder(input_size=10, encoding_size=5)
        
        # 检查解码器层数
        assert len(model.decoder) >= 2  # 至少应该有2层


class TestLSTMPredictorAndAutoencoderIntegration:
    """测试LSTMPredictor和Autoencoder集成场景"""

    def test_both_models_can_be_instantiated(self):
        """测试两个模型都可以实例化"""
        lstm = LSTMPredictor()
        autoencoder = Autoencoder()
        
        assert lstm is not None
        assert autoencoder is not None

    def test_both_models_can_forward_pass(self):
        """测试两个模型都可以进行前向传播"""
        lstm = LSTMPredictor(input_size=1, hidden_size=32, num_layers=1, output_size=1)
        autoencoder = Autoencoder(input_size=10, encoding_size=5)
        
        lstm.eval()
        autoencoder.eval()
        
        # LSTMPredictor输入
        lstm_input = torch.randn(1, 10, 1)
        lstm_output = lstm(lstm_input)
        assert lstm_output.shape == (1, 1)
        
        # Autoencoder输入
        ae_input = torch.randn(1, 10)
        ae_output = autoencoder(ae_input)
        assert ae_output.shape == (1, 10)

    def test_models_different_parameters(self):
        """测试两个模型可以使用不同参数"""
        lstm1 = LSTMPredictor(input_size=1, hidden_size=32)
        lstm2 = LSTMPredictor(input_size=5, hidden_size=128)
        
        assert lstm1.input_size != lstm2.input_size
        assert lstm1.hidden_size != lstm2.hidden_size
        
        ae1 = Autoencoder(input_size=10, encoding_size=5)
        ae2 = Autoencoder(input_size=20, encoding_size=10)
        
        assert ae1.input_size != ae2.input_size
        assert ae1.encoding_size != ae2.encoding_size



