"""
深度学习模型定义

LSTM预测器和Autoencoder模型定义。

从deep_learning_predictor.py中提取以改善代码组织。

Author: RQA2025 Development Team
Date: 2025-11-01
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    """时序数据数据集"""

    def __init__(self, data: np.ndarray, seq_length: int):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class LSTMPredictor(nn.Module):
    """
    LSTM时序预测模型

    用于预测系统指标的未来趋势，支持多步预测
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 64,
                 num_layers: int = 2, output_size: int = 1,
                 dropout: float = 0.2):
        super(LSTMPredictor, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Dropout层
        self.dropout = nn.Dropout(dropout)

        # 全连接层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM前向传播
        lstm_out, _ = self.lstm(x)

        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]

        # Dropout
        dropped = self.dropout(last_output)

        # 全连接层
        output = self.fc(dropped)

        return output


class Autoencoder(nn.Module):
    """
    自编码器模型

    用于异常检测，通过重构误差判断数据是否异常
    """

    def __init__(self, input_size: int = 10, encoding_size: int = 5):
        super(Autoencoder, self).__init__()

        # 保存参数
        self.input_size = input_size
        self.encoding_size = encoding_size

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_size, encoding_size * 2),
            nn.ReLU(),
            nn.Linear(encoding_size * 2, encoding_size),
            nn.ReLU()
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(encoding_size, encoding_size * 2),
            nn.ReLU(),
            nn.Linear(encoding_size * 2, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


__all__ = [
    'TimeSeriesDataset',
    'LSTMPredictor',
    'Autoencoder'
]

