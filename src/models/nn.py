# src/models/nn.py
import copy
from typing import Optional, Any, Union

import joblib
import torch
import torch.nn as nn
from src.models.utils import EarlyStopping
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
from pathlib import Path
from src.models.base_model import BaseModel, TorchModelMixin
from sklearn.preprocessing import StandardScaler
from src.infrastructure.utils.logger import get_logger
from sklearn.model_selection import TimeSeriesSplit

logger = get_logger(__name__)  # 自动继承全局配置


class NeuralNetworkModel(BaseModel, TorchModelMixin):
    """神经网络回归模型，继承自BaseModel和TorchModelMixin

    属性：
        model_name (str): 模型名称标识，默认"neural_network"
        input_size (int): 输入特征维度
        hidden_layers (list): 隐藏层维度列表，例如[256, 128]
        output_size (int): 输出维度，默认1
        dropout_rate (float): Dropout比例，默认0.3
        device (torch.device): 计算设备（CPU/GPU）
        model (Optional[nn.Module]): PyTorch模型实例
        scaler (Optional[StandardScaler]): 特征标准化器
    """

    def __init__(self, input_size, hidden_layers=None, dropout_rate=0.5, device="cpu", output_size=1):
        """初始化神经网络模型"""
        # 设置默认的 hidden_layers 值
        if hidden_layers is None:
            hidden_layers = [256, 128]

        super().__init__(
            model_name="neural_network",
            config={
                'input_size': input_size,
                'hidden_layers': hidden_layers,
                'dropout_rate': dropout_rate,
                'device': device,
                'output_size': output_size
            }
        )
        # 显式保存关键参数为实例属性
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.actual_epochs = 0  # 记录实际训练的总轮次
        # 设备配置
        self.device = self._configure_device(device)
        self.scaler: Optional[StandardScaler] = None
        self.build_model()
        # self.is_trained = False

        logger.info(f"模型初始化完成，配置: {self.config}")

    def _initialize_model(self):
        layers = []
        input_size = self.config['input_size']

        for hidden_size in self.config['hidden_layers']:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.config['dropout_rate']))
            input_size = hidden_size

        layers.append(nn.Linear(input_size, self.config['output_size']))
        return nn.Sequential(*layers)

    def get_model(self) -> torch.nn.Module:
        """返回具体的 PyTorch 模型实例"""
        return self.model

    def _configure_device(self, device: str) -> torch.device:
        """配置计算设备

        参数：
            device (str): 设备标识符

        返回：
            torch.device: 计算设备实例
        """
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def build_model(self):
        """构建PyTorch模型结构"""
        layers = []
        input_size = self.input_size

        for i, hidden_size in enumerate(self.hidden_layers):
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
            input_size = hidden_size

        layers.append(nn.Linear(input_size, self.output_size))
        self.model = nn.Sequential(*layers).to(self.device)

    def train(self, features: pd.DataFrame, target: pd.Series, epochs: int = 100, batch_size: int = 32,
              validation_split: float = 0.2, patience: int = 5, loss_type: str = 'mse',
              **kwargs: Any) -> "NeuralNetworkModel":
        # 校验输入特征维度
        if features.shape[1] != self.config['input_size']:
            raise ValueError(f"输入特征维度不正确: 预期 {self.config['input_size']}，实际 {features.shape[1]}")

        # 保存特征列顺序
        self.feature_names_ = features.columns.tolist()
        features_array = features.values
        target_array = target.values

        # 全局标准化
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features_array)

        # 创建数据集和数据加载器
        X_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(target_array, dtype=torch.float32).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        train_size = int(len(dataset) * (1 - validation_split))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 配置损失函数
        self.loss_fn = self.configure_loss(loss_type=loss_type)

        # 初始化模型和优化器
        model = self._initialize_model().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.get("lr", 1e-3))
        self.best_val_loss = float('inf')

        # 实例化早停类
        self.early_stopping = EarlyStopping(patience=patience, delta=0)
        best_model = None
        best_val_loss = float('inf')

        for epoch in range(epochs):
            # 训练步骤
            model.train()
            train_loss = 0.0
            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = self.loss_fn(outputs, labels.unsqueeze(1))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # 验证步骤
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = model(inputs)
                    val_loss += self.loss_fn(outputs, labels.unsqueeze(1)).item()

            # 打印日志
            logger.info(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # 更新最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(model)

            # 早停检查
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                logger.info("Early stopping triggered")  # 显式记录早停日志
                break

        # 保存最佳模型
        self.model = best_model
        self._is_trained = True

        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """生成预测结果

        参数:
            features (pd.DataFrame): 输入特征数据

        返回:
            np.ndarray: 预测结果数组

        异常:
            RuntimeError: 模型未训练时调用预测抛出
            ValueError: 输入特征数量不匹配时抛出
        """
        # 校验特征顺序
        logger.info("开始预测过程")
        self._validate_feature_order(features)

        if not self.is_trained or self.scaler is None:
            logger.error("模型尚未训练，无法进行预测")
            raise RuntimeError("模型尚未训练")

        # 检查特征数量
        if features.shape[1] != self.config['input_size']:
            logger.error(f"输入特征维度不匹配：预期 {self.config['input_size']}，实际 {features.shape[1]}")
            raise ValueError(f"输入特征维度不匹配：预期 {self.config['input_size']}，实际 {features.shape[1]}")

        # 标准化输入
        X = self.scaler.transform(features)
        logger.info("特征标准化完成")

        # 转换为 Tensor 并指定设备
        tensor_X = torch.FloatTensor(X)
        if not isinstance(tensor_X, torch.Tensor):
            logger.warning("输入数据转换为张量失败，重新转换")
            tensor_X = torch.FloatTensor(X)
        tensor_X = tensor_X.to(self.device)
        logger.info(f"数据转换为张量，设备: {self.device}, 形状: {tensor_X.shape}")

        # 强制设置为评估模式
        self.model.eval()
        logger.info("模型设置为评估模式")

        with torch.no_grad():
            logger.info("开始模型推理")
            predictions = self.model(tensor_X).cpu().numpy()
            logger.info(f"预测完成，结果形状: {predictions.shape}")

        return predictions.flatten()

    def save(self, dir_path: Union[str, Path], model_name: str, overwrite: bool = False) -> Path:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        model_path = dir_path / f"{model_name}.pt"
        scaler_path = dir_path / f"{model_name}_scaler.pkl"

        if model_path.exists() and not overwrite:
            raise FileExistsError(f"模型文件已存在: {model_path}")

        # 打印保存前的训练状态，方便调试
        logger.info(f"模型保存前，训练状态为：{self._is_trained}")

        # 保存模型状态
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'is_trained': self._is_trained,  # 明确保存训练状态
            'feature_names_': self.feature_names_
        }, model_path)

        # 保存 scaler
        joblib.dump(self.scaler, scaler_path)

        logger.info(f"模型保存成功: {model_path}")
        return model_path

    @classmethod
    def load(cls, dir_path: Union[str, Path], model_name: str) -> "NeuralNetworkModel":
        load_path = Path(dir_path) / f"{model_name}.pt"
        scaler_path = Path(dir_path) / f"{model_name}_scaler.pkl"
        if not load_path.exists() or not scaler_path.exists():
            logger.error(f"模型文件或标准化器文件不存在: {load_path} 或 {scaler_path}")
            raise FileNotFoundError(f"模型文件或标准化器文件不存在: {load_path} 或 {scaler_path}")

        logger.info(f"开始加载模型: {load_path}")

        checkpoint = torch.load(load_path, map_location='cpu')
        logger.info(f"模型配置: {checkpoint['config']}")

        # 检查 checkpoint 中是否包含 'is_trained' 键
        if 'is_trained' not in checkpoint:
            logger.warning("加载的模型 checkpoint 中未找到 'is_trained' 键，将设置为 False")
            is_trained = False
        else:
            is_trained = checkpoint['is_trained']

        # 重建模型实例
        instance = cls(
            input_size=checkpoint['config']['input_size'],
            hidden_layers=checkpoint['config']['hidden_layers'],
            dropout_rate=checkpoint['config']['dropout_rate'],
            device=checkpoint['config']['device'],
            output_size=checkpoint['config']['output_size']
        )
        logger.info(f"模型实例化完成")

        # 确保模型结构正确初始化
        if hasattr(instance, 'build_model'):
            instance.build_model()

        # 加载模型状态字典
        instance.model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        # 确保加载 scaler
        instance.scaler = joblib.load(scaler_path)

        # 显式设置训练状态
        instance._is_trained = is_trained
        instance.feature_names_ = checkpoint.get('feature_names_', None)

        return instance

    def _prepare_data(self, features, target, batch_size, val_split):
        """数据预处理流水线"""
        # 特征标准化
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(features)
        y = target.values.astype(np.float32)

        # 转换为Tensor
        X_tensor = torch.FloatTensor(X).to(self.device)  # 显式移动到设备
        y_tensor = torch.FloatTensor(y).to(self.device)  # 显式移动到设备
        dataset = TensorDataset(X_tensor, y_tensor)

        # 划分训练/验证集
        val_size = int(len(dataset) * val_split)
        train_set, val_set = torch.utils.data.random_split(
            dataset, [len(dataset) - val_size, val_size]
        )

        return (
            DataLoader(train_set, batch_size=batch_size, shuffle=True),
            DataLoader(val_set, batch_size=batch_size, shuffle=False)
        )

    def _validate(self, val_loader: DataLoader, loss_fn: nn.Module) -> float:
        """验证集评估"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = self.model(inputs)
                total_loss += loss_fn(outputs, labels.unsqueeze(1)).item()
                num_batches += 1
        if num_batches == 0:
            raise ValueError("验证数据加载器未提供任何数据")
        return total_loss / num_batches

    def configure_optimizer(self):
        """配置Adam优化器"""
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def configure_loss(self, loss_type: str = 'mse'):
        if loss_type == 'mse':
            return torch.nn.MSELoss()
        elif loss_type == 'mae':
            return torch.nn.L1Loss()
        elif loss_type == 'quantile':
            return QuantileLoss(q=0.5)
        elif loss_type == 'huber':
            return torch.nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")


class QuantileLoss(nn.Module):
    """分位数损失函数，用于处理非对称成本"""

    def __init__(self, q: float = 0.5):
        super(QuantileLoss, self).__init__()
        self.q = q

    def forward(self, preds, targets):
        assert 0 < self.q < 1
        errors = targets - preds
        return torch.mean(torch.max((self.q - 1) * errors, self.q * errors))


class EarlyStopping:
    """早停类，用于防止过拟合"""

    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.patience_counter = 0

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
            self.val_loss_min = val_loss
        elif val_loss > self.val_loss_min + self.delta:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.val_loss_min = val_loss
            self.patience_counter = 0
        return self.early_stop
