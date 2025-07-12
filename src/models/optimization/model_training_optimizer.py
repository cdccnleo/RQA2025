import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import logging
import time
from datetime import datetime
import torch
import tensorflow as tf
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import os

logger = logging.getLogger(__name__)

class ModelTrainingOptimizer:
    """模型训练优化器"""

    def __init__(
        self,
        use_gpu: bool = True,
        mixed_precision: bool = True,
        batch_size: Optional[int] = None,
        n_jobs: int = -1,
        early_stopping: bool = True,
        early_stopping_patience: int = 10,
        early_stopping_delta: float = 0.001
    ):
        """
        初始化模型训练优化器

        Args:
            use_gpu: 是否使用GPU
            mixed_precision: 是否使用混合精度训练
            batch_size: 批处理大小，None表示自动确定
            n_jobs: 并行作业数，-1表示使用所有可用CPU
            early_stopping: 是否使用早停
            early_stopping_patience: 早停耐心值
            early_stopping_delta: 早停增量阈值
        """
        self.use_gpu = use_gpu and (torch.cuda.is_available() or tf.config.list_physical_devices('GPU'))
        self.mixed_precision = mixed_precision
        self.batch_size = batch_size
        self.n_jobs = n_jobs if n_jobs > 0 else max(1, os.cpu_count() - 1)
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delta = early_stopping_delta

        # 初始化设备
        self._init_devices()

        # 初始化混合精度
        self._init_mixed_precision()

    def _init_devices(self) -> None:
        """初始化设备"""
        self.devices = []

        if self.use_gpu:
            # PyTorch设备
            if torch.cuda.is_available():
                self.torch_device = torch.device("cuda")
                for i in range(torch.cuda.device_count()):
                    self.devices.append(f"cuda:{i}")
            else:
                self.torch_device = torch.device("cpu")
                self.devices.append("cpu")

            # TensorFlow设备
            self.tf_gpus = tf.config.list_physical_devices('GPU')
            if self.tf_gpus:
                # 配置TensorFlow使用所有GPU
                try:
                    for gpu in self.tf_gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except Exception as e:
                    logger.warning(f"Error configuring TensorFlow GPU: {e}")
        else:
            self.torch_device = torch.device("cpu")
            self.devices.append("cpu")
            self.tf_gpus = []

    def _init_mixed_precision(self) -> None:
        """初始化混合精度"""
        if self.mixed_precision and self.use_gpu:
            # PyTorch混合精度
            self.torch_scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

            # TensorFlow混合精度
            if self.tf_gpus:
                try:
                    tf.keras.mixed_precision.set_global_policy('mixed_float16')
                    self.tf_mixed_precision = True
                except Exception as e:
                    logger.warning(f"Error setting TensorFlow mixed precision: {e}")
                    self.tf_mixed_precision = False
            else:
                self.tf_mixed_precision = False
        else:
            self.torch_scaler = None
            self.tf_mixed_precision = False

    def optimize_batch_size(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        framework: str = 'pytorch',
        min_batch: int = 16,
        max_batch: int = 1024,
        max_memory_usage: float = 0.8
    ) -> int:
        """
        优化批处理大小

        Args:
            model: 模型
            X: 特征数据
            y: 目标变量
            framework: 框架，'pytorch'或'tensorflow'或'sklearn'
            min_batch: 最小批处理大小
            max_batch: 最大批处理大小
            max_memory_usage: 最大内存使用率

        Returns:
            int: 优化后的批处理大小
        """
        if framework == 'sklearn':
            # scikit-learn模型不使用批处理
            return len(X)

        # 如果没有GPU，使用较小的批处理大小
        if not self.use_gpu:
            return min_batch

        # 获取可用GPU内存
        if framework == 'pytorch' and torch.cuda.is_available():
            device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device).total_memory
            allocated_memory = torch.cuda.memory_allocated(device)
            available_memory = total_memory - allocated_memory
            max_allowed_memory = available_memory * max_memory_usage
        elif framework == 'tensorflow' and self.tf_gpus:
            # TensorFlow获取可用内存较复杂，使用简化方法
            try:
                gpu_info = tf.config.experimental.get_memory_info('GPU:0')
                available_memory = gpu_info['available']
                max_allowed_memory = available_memory * max_memory_usage
            except:
                # 如果无法获取内存信息，使用默认值
                return 128
        else:
            # 如果无法获取内存信息，使用默认值
            return 128

        # 二分查找最佳批处理大小
        best_batch_size = min_batch
        left, right = min_batch, max_batch

        while left <= right:
            mid = (left + right) // 2

            try:
                if framework == 'pytorch':
                    # 创建测试批次
                    X_batch = torch.tensor(X[:mid], device=self.torch_device)
                    y_batch = torch.tensor(y[:mid], device=self.torch_device)

                    # 尝试前向传播
                    with torch.no_grad():
                        _ = model(X_batch)

                    # 检查内存使用
                    current_memory = torch.cuda.memory_allocated(device)
                    if current_memory < max_allowed_memory:
                        best_batch_size = mid
                        left = mid + 1
                    else:
                        right = mid - 1

                    # 清理内存
                    del X_batch, y_batch
                    torch.cuda.empty_cache()

                elif framework == 'tensorflow':
                    # 创建测试批次
                    X_batch = tf.convert_to_tensor(X[:mid])
                    y_batch = tf.convert_to_tensor(y[:mid])

                    # 尝试前向传播
                    _ = model(X_batch, training=False)

                    # 由于TensorFlow内存管理复杂，使用简化方法
                    best_batch_size = mid
                    left = mid + 1

                    # 清理内存
                    del X_batch, y_batch
                    tf.keras.backend.clear_session()

            except Exception as e:
                # 如果出现内存错误，减小批处理大小
                logger.warning(f"Error with batch size {mid}: {e}")
                right = mid - 1

        return best_batch_size

    def optimize_pytorch_training(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        criterion: torch.nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        epochs: int = 100,
        callbacks: List[Callable] = None
    ) -> Dict:
        """
        优化PyTorch模型训练

        Args:
            model: PyTorch模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            criterion: 损失函数
            optimizer: 优化器
            epochs: 训练轮数
            callbacks: 回调函数列表

        Returns:
            Dict: 训练历史
        """
        # 将模型移动到设备
        model = model.to(self.torch_device)

        # 初始化训练历史
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_time': [],
            'val_time': []
        }

        # 初始化早停
        if self.early_stopping:
            best_val_loss = float('inf')
            patience_counter = 0

        # 训练循环
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_start_time = time.time()

            for batch_idx, (data, target) in enumerate(train_loader):
                # 将数据移动到设备
                data, target = data.to(self.torch_device), target.to(self.torch_device)

                # 清除梯度
                optimizer.zero_grad()

                if self.mixed_precision and self.torch_scaler:
                    # 使用混合精度训练
                    with torch.cuda.amp.autocast():
                        output = model(data)
                        loss = criterion(output, target)

                    # 缩放梯度并优化
                    self.torch_scaler.scale(l loss).backward()
                    self.torch_scaler.step(optimizer)
                    self.torch_scaler.update()
                else:
                    # 常规训练
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_time = time.time() - train_start_time
            history['train_loss'].append(train_loss)
            history['train_time'].append(train_time)

            # 验证阶段
            if val_loader:
                model.eval()
                val_loss = 0.0
                val_start_time = time.time()

                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(self.torch_device), target.to(self.torch_device)
                        output = model(data)
                        loss = criterion(output, target)
                        val_loss += loss.item()

                val_loss /= len(val_loader)
                val_time = time.time() - val_start_time
                history['val_loss'].append(val_loss)
                history['val_time'].append(val_time)

                # 早停检查
                if self.early_stopping:
                    if val_loss < best_val_loss - self.early_stopping_delta:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= self.early_stopping_patience:
                            logger.info(f"Early stopping at epoch {epoch+1}")
                            break

            # 执行回调
            if callbacks