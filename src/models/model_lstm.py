# src/models/lstm.py
import copy
from pathlib import Path
from typing import Optional, Any, Union, Tuple, Type

import joblib
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd

from src.models.base_model import BaseModel, T
from src.infrastructure.utils.logger import get_logger
from src.models.utils import EarlyStopping, DeviceManager
from src.models.nn import QuantileLoss

logger = get_logger(__name__)  # 自动继承全局配置


class LSTMModel(nn.Module):
    """基础LSTM模型

    属性：
        input_size (int): 输入特征维度
        hidden_size (int): 隐藏层维度
        num_layers (int): LSTM层数
        output_size (int): 输出维度
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """前向传播

        参数：
            x (torch.Tensor): 输入张量，形状为(batch_size, seq_len, input_size)

        返回：
            torch.Tensor: 输出张量，形状为(batch_size, output_size)
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class LSTMModelWrapper(BaseModel):
    """LSTM模型包装器，集成训练/预测/保存功能

    属性：
        input_size (int): 输入特征维度
        seq_length (int): 时间序列窗口长度
        hidden_size (int): LSTM隐藏层维度，默认256
        num_layers (int): LSTM层数，默认3
        output_size (int): 输出维度，默认1
        dropout (float): Dropout比例，默认0.5
        model (Optional[nn.Module]): LSTM模型实例
        scaler (Optional[StandardScaler]): 特征标准化器
    """

    def __init__(
            self,
            input_size: int,
            seq_length: int = 10,
            hidden_size: int = 256,
            num_layers: int = 3,
            output_size: int = 1,
            dropout: float = 0.5,
            device: str = "auto"  # 默认自动选择设备
    ):
        super().__init__(model_name="lstm")
        # 保存初始化参数到配置字典
        self.config = {
            'input_size': input_size,
            'seq_length': seq_length,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'output_size': output_size,
            'dropout': dropout,
            'device': device
        }
        self.input_size = input_size
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.model: Optional[nn.Module] = None
        self.scaler: Optional[StandardScaler] = None  # 全局scaler
        self.best_val_loss = -np.inf
        self.early_stopping = None

        # 设备初始化 - 使用 DeviceManager 确保设备选择正确
        self.device = DeviceManager.get_device(device)

        # 构建模型
        self.build_model()

        # 显式初始化 strict_feature_order 属性
        self.strict_feature_order = True

        # 尝试将模型移动到目标设备
        try:
            self.model.to(self.device)
        except RuntimeError as e:
            logger.error(f"无法将模型移动到设备 {self.device}: {str(e)}")
            logger.warning("回退到CPU设备")
            self.device = torch.device("cpu")
            self.model.to(self.device)

    def get_model(self) -> nn.Module:
        """返回具体的 PyTorch 模型实例"""
        return self.model

    def build_model(self):
        """根据配置参数构建AttentionLSTM模型"""
        required_params = ['input_size', 'hidden_size', 'num_layers', 'output_size', 'dropout']
        lstm_config = {k: v for k, v in self.config.items() if k in required_params}

        # 直接构建模型
        self.model = AttentionLSTM(**lstm_config)

    @classmethod
    def _load_pytorch(cls, path: Path) -> BaseModel:
        checkpoint = torch.load(path)
        config = checkpoint['config']
        model_name = config.get('model_name', 'default_name')
        instance = cls(model_name=model_name, config=config)
        instance.build_model()
        instance.model.load_state_dict(checkpoint['model_state_dict'])
        instance._is_trained = checkpoint.get('is_trained', False)  # 显式恢复训练状态
        instance.feature_names_ = checkpoint.get('feature_names_', None)

        # 加载标准化器
        scaler_path = path.parent / f"{model_name}_scaler.pkl"
        if scaler_path.exists():
            instance.scaler = joblib.load(scaler_path)
        else:
            instance.scaler = None

        return instance

    def _initialize_model(self) -> nn.Module:
        """初始化模型实例"""
        logger.info(f"初始化模型，配置参数：{self.config}")
        model = AttentionLSTM(
            input_size=self.config['input_size'],
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            output_size=self.config['output_size'],
            dropout=self.config['dropout']
        )
        logger.info(f"模型初始化完成，类型：{type(model)}")
        return model

    def _configure_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                # 显式检查 CUDA 是否可用
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        elif device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA is not available, falling back to CPU")
            device = torch.device("cpu")
        else:
            device = torch.device(device)
        return device

    def train(self, features: pd.DataFrame, target: pd.Series, **kwargs: Any) -> "LSTMModelWrapper":
        """训练LSTM模型"""
        # 校验输入数据维度
        if len(features.shape) != 2:
            raise ValueError("输入数据维度不正确，应为 2D")

        # 保存特征列顺序
        self.feature_names_ = features.columns.tolist()
        raw_features = features.values
        raw_target = target.values

        # 在训练循环外进行全局标准化
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(raw_features)

        # 初始化时间序列交叉验证器
        tscv = TimeSeriesSplit(n_splits=5)
        fold_models = []  # 用于保存每个fold的最佳模型
        self.best_val_loss = float('inf')  # 初始化全局最佳验证损失
        best_global_model = None

        for fold, (train_idx, val_idx) in enumerate(tscv.split(features_scaled)):
            # 获取原始数据的训练集和验证集索引
            train_data = features_scaled[train_idx]
            train_target = raw_target[train_idx]
            val_data = features_scaled[val_idx]
            val_target = raw_target[val_idx]

            # 创建时间序列数据
            X_train, y_train = self._create_sequences(train_data, train_target)
            if len(X_train) == 0:
                logger.warning(f"Fold {fold + 1} 数据不足，跳过该fold")
                continue
            X_val, y_val = self._create_sequences(val_data, val_target)

            # 初始化模型
            model = self._initialize_model()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.get('lr', 0.001))
            criterion = self.configure_loss(loss_type=self.config.get('loss_type', 'mse'))

            # 确保数据在正确设备上
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device)
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(self.device)

            # 实例化早停类
            early_stopping = EarlyStopping(patience=kwargs.get('patience', 5), delta=0)
            best_fold_loss = float('inf')
            best_fold_model = None

            for epoch in range(kwargs.get('epochs', 100)):
                # 训练步骤
                model.train()
                optimizer.zero_grad()
                outputs, _ = model(X_train_tensor)  # 注意这里返回两个值
                outputs = outputs.view(-1, self.config['output_size'])
                y_train_tensor = y_train_tensor.view(-1, self.config['output_size'])

                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()

                # 验证步骤
                model.eval()
                with torch.no_grad():
                    val_outputs, _ = model(X_val_tensor)
                    val_outputs = val_outputs.view(-1, self.config['output_size'])
                    y_val_tensor = y_val_tensor.view(-1, self.config['output_size'])

                    val_loss = criterion(val_outputs, y_val_tensor)

                # 打印日志
                logger.info(
                    f"Fold {fold + 1}, Epoch {epoch + 1}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}"
                )

                # 更新最佳模型
                if val_loss < best_fold_loss:
                    best_fold_loss = val_loss
                    best_fold_model = copy.deepcopy(model)

                # 调用早停逻辑
                if early_stopping(val_loss):
                    logger.info(f"Fold {fold + 1} 早停触发：验证损失 {early_stopping.patience} 轮未下降，训练终止")
                    break

            # 创建一个临时的 LSTMModelWrapper 实例来保存最佳模型
            fold_wrapper = LSTMModelWrapper(
                input_size=self.input_size,
                seq_length=self.seq_length,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                output_size=self.output_size,
                dropout=self.dropout,
                device=self.device
            )
            fold_wrapper.model = best_fold_model
            fold_wrapper.best_val_loss = best_fold_loss  # 保存最佳验证损失
            fold_wrapper._is_trained = True

            fold_models.append(fold_wrapper)

            # 更新全局最佳模型
            if best_fold_loss < self.best_val_loss:
                self.best_val_loss = best_fold_loss
                best_global_model = copy.deepcopy(best_fold_model)

        # 使用全局最佳模型替换默认模型
        self.model = best_global_model

        # 保存所有fold的最佳模型
        self.fold_models = fold_models
        self._is_trained = True

        return self

    def configure_loss(self, loss_type: str = 'mse') -> torch.nn.Module:
        """配置损失函数，支持多种损失类型"""
        if loss_type == 'mse':
            return torch.nn.MSELoss()
        elif loss_type == 'mae':
            return torch.nn.L1Loss()
        elif loss_type == 'quantile':
            return QuantileLoss(q=0.5)  # 中位数损失
        elif loss_type == 'huber':
            return torch.nn.SmoothL1Loss()
        else:
            raise ValueError(f"不支持的损失函数: {loss_type}")

    def predict(self, features: Union[pd.DataFrame, np.ndarray],
                return_attention_weights: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """生成预测结果，可选返回注意力权重"""
        if not self.is_trained or self.model is None:
            raise RuntimeError("模型尚未训练")

        logger.info("开始预测过程")

        # 校验特征顺序（当输入为DataFrame时）
        if isinstance(features, pd.DataFrame):
            logger.info(f"输入特征为 DataFrame，列数: {features.shape[1]}")
            self._validate_feature_order(features)
            features_array = features.values
        else:
            # 若输入为numpy数组，需校验特征数量与顺序是否与训练一致
            logger.info(f"输入特征为 numpy 数组，形状: {features.shape}")
            if features.ndim < 2:
                logger.error("特征数据维度不足，应至少为二维")
                raise ValueError("特征数据维度不足，应至少为二维")
            if features.shape[1] != len(self.feature_names_):
                logger.error(f"输入特征维度与训练时不匹配: {features.shape[1]} vs {len(self.feature_names_)}")
                raise ValueError("输入特征维度与训练时不匹配")
            features_array = features

        logger.info("特征校验完成")

        # 强制转换为 numpy array
        features_scaled = self.scaler.transform(features_array)
        logger.info("特征标准化完成")

        # 校验输入数据长度是否足够
        if len(features_scaled) < self.config.get('seq_length', 10):
            logger.error(f"输入数据长度不足: {len(features_scaled)}")
            raise ValueError(
                f"输入数据长度 {len(features_scaled)} 小于模型配置的 seq_length={self.config.get('seq_length', 10)}"
            )

        # 生成时间窗口数据
        X = self._create_inference_data(features_scaled, seq_length=self.config.get('seq_length', 10))
        logger.info(f"生成时间窗口数据，形状: {X.shape}")

        # 确保数据在正确设备上
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        logger.info(f"数据转换为张量，设备: {self.device}")

        # 强制设置为评估模式
        self.model.eval()
        logger.info("模型设置为评估模式")

        with torch.no_grad():
            # 获取模型输出和注意力权重
            outputs, attention_weights = self.model(X_tensor)
            logger.info("模型推理完成")

            predictions = outputs.cpu().numpy()
            attention_weights = attention_weights.cpu().numpy()

        logger.info(f"预测完成，结果形状: {predictions.shape}")

        if return_attention_weights:
            logger.info("返回预测结果和注意力权重")
            return predictions.flatten(), attention_weights
        else:
            logger.info("仅返回预测结果")
            return predictions.flatten()

    def save(self, dir_path: Union[str, Path], model_name: str, overwrite: bool = False) -> Path:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        # 保存模型状态
        model_path = dir_path / f"{model_name}.pt"
        scaler_path = dir_path / f"{model_name}_scaler.pkl"

        if model_path.exists() and not overwrite:
            raise FileExistsError(f"模型文件已存在: {model_path}")

        # 保存模型状态字典，包括 is_trained 和 feature_names_
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'is_trained': self._is_trained,
            'feature_names_': self.feature_names_  # 显式保存 feature_names_
        }, model_path)

        # 保存标准化器
        joblib.dump(self.scaler, scaler_path)

        self.logger.info(f"模型保存成功: {model_path}")
        return model_path

    @classmethod
    def load(cls: Type[T], dir_path: Union[str, Path], model_name: str) -> T:
        dir_path = Path(dir_path)
        model_path = dir_path / f"{model_name}.pt"
        scaler_path = dir_path / f"{model_name}_scaler.pkl"

        if not model_path.exists() or not scaler_path.exists():
            raise FileNotFoundError(f"模型文件或标准化器文件不存在: {model_path} 或 {scaler_path}")

        # 加载模型状态
        checkpoint = torch.load(model_path, map_location='cpu')
        config = checkpoint['config']
        model = cls(**config)
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model._is_trained = checkpoint.get('is_trained', False)
        model.feature_names_ = checkpoint.get('feature_names_', None)  # 显式加载 feature_names_

        # 加载标准化器
        model.scaler = joblib.load(scaler_path)

        model.logger.info(f"模型加载成功: {model_path}")
        return model

    def _create_sequences(self, data, target, seq_length=10):
        """生成时间序列数据"""
        if len(data) < seq_length:
            logger.error(f"输入数据长度 {len(data)} 小于模型配置的 seq_length={seq_length}")
            raise ValueError(f"输入数据长度 {len(data)} 小于模型配置的 seq_length={seq_length}")
        sequences = []
        labels = []
        for i in range(len(data) - seq_length + 1):
            sequences.append(data[i:i + seq_length])
            labels.append(target[i + seq_length - 1])
        return np.array(sequences), np.array(labels)

    def _create_inference_data(self, data: np.ndarray, seq_length: int = 10) -> np.ndarray:
        """生成推理用的时间序列数据

        参数:
            data (np.ndarray): 标准化后的特征数据，形状为(n_samples, n_features)
            seq_length (int): 时间序列窗口长度，默认10

        返回:
            np.ndarray: 时间序列窗口数据，形状为(n_samples - seq_length + 1, seq_length, n_features)
        """
        if len(data) < seq_length:
            raise ValueError(
                f"输入数据长度 {len(data)} 小于模型配置的 seq_length={seq_length}"
            )

        sequences = []
        for i in range(len(data) - seq_length + 1):
            sequences.append(data[i:i + seq_length])

        return np.array(sequences)

    def get_feature_importance(self) -> pd.Series:
        """神经网络模型不支持特征重要性计算"""
        raise NotImplementedError("神经网络模型暂不支持特征重要性计算")


class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(AttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.output_size = output_size  # 添加 output_size 属性
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.attention = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        attention_weights = torch.tanh(self.attention(out))
        attention_weights = torch.softmax(attention_weights, dim=1)
        out = out * attention_weights
        out = out[:, -1, :]
        out = self.fc(out)
        return out, attention_weights  # 返回输出和注意力权重

    def visualize_attention(self, x):
        """可视化注意力权重接口"""
        with torch.no_grad():
            _, attention_weights = self.forward(x)
        return attention_weights.cpu().numpy()
