"""
深度学习预测器核心

DeepLearningPredictor主类实现。

从deep_learning_predictor.py中提取以改善代码组织。

Author: RQA2025 Development Team
Date: 2025-11-01
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import List, Dict, Any, Optional
import logging
import os
import time
from datetime import datetime
import warnings

from .dl_models import TimeSeriesDataset, LSTMPredictor, Autoencoder
from .dl_optimizer import GPUResourceManager, AIModelOptimizer, DynamicBatchOptimizer

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class ModelCacheManager:
    """
    模型缓存管理器

    缓存已训练的模型，避免重复训练
    """

    def __init__(self, max_cache_size: int = 10):
        self.cache: Dict[str, Any] = {}
        self.max_cache_size = max_cache_size
        self.access_count: Dict[str, int] = {}

        logger.info(f"模型缓存管理器初始化，最大缓存: {max_cache_size}")

    def get(self, key: str) -> Optional[Any]:
        """获取缓存的模型"""
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        return None

    def set(self, key: str, model: Any):
        """缓存模型"""
        # 如果缓存已满，移除访问次数最少的
        if len(self.cache) >= self.max_cache_size:
            lru_key = min(self.access_count.items(), key=lambda x: x[1])[0]
            del self.cache[lru_key]
            del self.access_count[lru_key]

        self.cache[key] = model
        self.access_count[key] = 0

    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.access_count.clear()


class DeepLearningPredictor:
    """
    深度学习预测器

    整合LSTM预测、Autoencoder异常检测等功能

    Author: RQA2025 Development Team
    Date: 2025-11-01
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # 基础配置
        self.seq_length = self.config.get('seq_length', 20)
        self.hidden_size = self.config.get('hidden_size', 64)
        self.num_layers = self.config.get('num_layers', 2)
        self.dropout = self.config.get('dropout', 0.2)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.epochs = self.config.get('epochs', 100)
        self.batch_size = self.config.get('batch_size', 32)

        # 组件
        self.gpu_manager = GPUResourceManager()
        self.model_optimizer = AIModelOptimizer()
        self.batch_optimizer = DynamicBatchOptimizer(initial_batch_size=self.batch_size)
        self.model_cache = ModelCacheManager()

        # 设备
        self.device = self.gpu_manager.get_device()

        # 数据预处理器
        self.scaler = StandardScaler()

        # 模型
        self.lstm_model: Optional[LSTMPredictor] = None
        self.autoencoder_model: Optional[Autoencoder] = None

        # 训练历史
        self.training_history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }

        logger.info(f"深度学习预测器初始化完成，设备: {self.device}")

    def train_lstm(self, data: np.ndarray, epochs: Optional[int] = None,
                   validation_split: float = 0.2) -> Dict[str, Any]:
        """训练LSTM模型"""
        try:
            epochs = epochs or self.epochs

            # 数据预处理
            scaled_data = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()

            # 划分训练集和验证集
            split_idx = int(len(scaled_data) * (1 - validation_split))
            train_data = scaled_data[:split_idx]
            val_data = scaled_data[split_idx:]

            # 创建数据集
            train_dataset = TimeSeriesDataset(train_data, self.seq_length)
            val_dataset = TimeSeriesDataset(val_data, self.seq_length)

            # 创建数据加载器
            batch_size = self.batch_optimizer.get_batch_size()
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            # 创建模型
            self.lstm_model = LSTMPredictor(
                input_size=1,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                output_size=1,
                dropout=self.dropout
            ).to(self.device)

            # 优化器和损失函数
            optimizer = optim.Adam(self.lstm_model.parameters(), lr=self.learning_rate)
            criterion = nn.MSELoss()

            # 训练循环
            train_losses = []
            val_losses = []

            for epoch in range(epochs):
                # 训练阶段
                self.lstm_model.train()
                train_loss = 0.0

                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.unsqueeze(-1).to(self.device)
                    batch_y = batch_y.unsqueeze(-1).to(self.device)

                    optimizer.zero_grad()
                    outputs = self.lstm_model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                train_loss /= len(train_loader)
                train_losses.append(train_loss)

                # 验证阶段
                self.lstm_model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.unsqueeze(-1).to(self.device)
                        batch_y = batch_y.unsqueeze(-1).to(self.device)

                        outputs = self.lstm_model(batch_x)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()

                val_loss /= len(val_loader)
                val_losses.append(val_loss)

                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

                # 动态调整批量大小
                if (epoch + 1) % 20 == 0:
                    batch_size = self.batch_optimizer.adjust_batch_size()

            # 保存训练历史
            self.training_history['train_loss'] = train_losses
            self.training_history['val_loss'] = val_losses

            return {
                'success': True,
                'final_train_loss': train_losses[-1],
                'final_val_loss': val_losses[-1],
                'epochs': epochs
            }

        except Exception as e:
            logger.error(f"LSTM训练失败: {e}")
            return {'success': False, 'error': str(e)}

    def predict(self, data: np.ndarray, steps: int = 1) -> np.ndarray:
        """预测未来值"""
        if self.lstm_model is None:
            logger.error("模型未训练")
            return np.array([])

        try:
            self.lstm_model.eval()

            # 数据预处理
            scaled_data = self.scaler.transform(data.reshape(-1, 1)).flatten()

            # 准备输入序列
            input_seq = scaled_data[-self.seq_length:]
            predictions = []

            with torch.no_grad():
                for _ in range(steps):
                    x = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device)
                    pred = self.lstm_model(x)
                    pred_value = pred.item()

                    predictions.append(pred_value)

                    # 更新输入序列
                    input_seq = np.append(input_seq[1:], pred_value)

            # 反标准化
            predictions = self.scaler.inverse_transform(
                np.array(predictions).reshape(-1, 1)
            ).flatten()

            return predictions

        except Exception as e:
            logger.error(f"预测失败: {e}")
            return np.array([])

    def detect_anomaly(self, data: np.ndarray, threshold: float = 0.1) -> Dict[str, Any]:
        """异常检测"""
        if self.autoencoder_model is None:
            logger.error("Autoencoder模型未训练")
            return {'anomalies': [], 'reconstruction_errors': []}

        try:
            self.autoencoder_model.eval()

            # 数据预处理
            scaled_data = self.scaler.transform(data.reshape(-1, 1))

            with torch.no_grad():
                x = torch.tensor(scaled_data, dtype=torch.float32).to(self.device)
                reconstructed = self.autoencoder_model(x)

                # 计算重构误差
                errors = torch.mean((x - reconstructed) ** 2, dim=1).cpu().numpy()

            # 识别异常点
            anomalies = np.where(errors > threshold)[0].tolist()

            return {
                'anomalies': anomalies,
                'reconstruction_errors': errors.tolist(),
                'threshold': threshold
            }

        except Exception as e:
            logger.error(f"异常检测失败: {e}")
            return {'anomalies': [], 'reconstruction_errors': [], 'error': str(e)}


__all__ = ['ModelCacheManager', 'DeepLearningPredictor']

