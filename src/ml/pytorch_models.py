"""
PyTorch深度学习模型实现
支持LSTM、CNN、Transformer模型用于量化交易
"""

import logging
import math
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# ==================== LSTM模型 ====================

class LSTMModel(nn.Module):
    """LSTM模型用于股票价格预测"""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, 
                 output_size: int = 1, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入数据，形状 (batch_size, sequence_length, input_size)
            
        Returns:
            输出预测，形状 (batch_size, output_size)
        """
        # LSTM前向传播
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        
        # 全连接层
        output = self.fc(last_output)
        return output


def create_sequences(X: pd.DataFrame, y: pd.Series, seq_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    创建时间序列数据
    
    Args:
        X: 特征数据
        y: 标签数据
        seq_length: 序列长度
        
    Returns:
        (X序列, y序列)
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X.iloc[i:i+seq_length].values)
        y_seq.append(y.iloc[i+seq_length])
    return torch.FloatTensor(X_seq), torch.FloatTensor(y_seq)


def train_lstm_model(X_train: pd.DataFrame, y_train: pd.Series, 
                     X_test: pd.DataFrame, y_test: pd.Series,
                     config: Dict[str, Any], progress_callback=None) -> Tuple[LSTMModel, Dict[str, Any]]:
    """
    训练LSTM模型
    
    Args:
        X_train: 训练特征
        y_train: 训练标签
        X_test: 测试特征
        y_test: 测试标签
        config: 配置参数
        progress_callback: 进度回调函数
        
    Returns:
        (训练好的模型, 训练信息)
    """
    # 超参数
    hidden_size = config.get('hidden_size', 64)
    num_layers = config.get('num_layers', 2)
    learning_rate = config.get('learning_rate', 0.001)
    epochs = config.get('epochs', 100)
    batch_size = config.get('batch_size', 32)
    dropout = config.get('dropout', 0.2)
    sequence_length = config.get('sequence_length', 20)
    early_stopping_patience = config.get('early_stopping_patience', 10)
    
    logger.info(f"开始训练LSTM模型: hidden_size={hidden_size}, num_layers={num_layers}, "
                f"sequence_length={sequence_length}")
    
    # 数据准备 - 转换为时间序列格式
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length)
    
    logger.info(f"序列数据形状: X_train={X_train_seq.shape}, y_train={y_train_seq.shape}")
    
    if len(X_train_seq) < batch_size:
        logger.warning(f"训练数据不足: {len(X_train_seq)} < batch_size={batch_size}")
        batch_size = max(1, len(X_train_seq) // 2)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train_seq, y_train_seq)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 模型初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    model = LSTMModel(
        input_size=X_train.shape[1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=1,
        dropout=dropout
    ).to(device)
    
    # 损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)
    
    # 训练循环
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # 前向传播
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_loss)
        
        # 早停检查
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # 保存最佳模型
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"早停于epoch {epoch}")
                break
        
        # 进度回调
        if progress_callback and epoch % 10 == 0:
            progress_callback(int((epoch / epochs) * 100), f"Epoch {epoch}, Loss: {avg_loss:.4f}")
        
        if epoch % 10 == 0:
            logger.info(f"Epoch [{epoch}/{epochs}], Loss: {avg_loss:.4f}")
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # 评估模型
    model.eval()
    with torch.no_grad():
        X_test_seq = X_test_seq.to(device)
        y_pred_proba = model(X_test_seq).cpu().numpy().squeeze()
        y_pred = (y_pred_proba > 0.5).astype(int)
    
    # 计算指标
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    accuracy = accuracy_score(y_test_seq.numpy(), y_pred)
    precision = precision_score(y_test_seq.numpy(), y_pred, zero_division=0)
    recall = recall_score(y_test_seq.numpy(), y_pred, zero_division=0)
    f1 = f1_score(y_test_seq.numpy(), y_pred, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_test_seq.numpy(), y_pred_proba)
    except:
        roc_auc = None
    
    training_info = {
        'best_loss': best_loss,
        'epochs_trained': epoch + 1,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'device': str(device)
    }
    
    logger.info(f"LSTM训练完成: accuracy={accuracy:.4f}, f1={f1:.4f}")
    
    return model, training_info


# ==================== CNN模型 ====================

class CNNModel(nn.Module):
    """CNN模型用于K线形态识别"""
    
    def __init__(self, input_size: int, num_classes: int = 1, dropout: float = 0.3):
        super(CNNModel, self).__init__()
        
        # 卷积层
        self.conv_layers = nn.Sequential(
            # 卷积层1
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout / 2),
            
            # 卷积层2
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout / 2),
            
            # 卷积层3
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),  # 全局池化
            nn.Dropout(dropout)
        )
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入数据，形状 (batch_size, sequence_length, input_size)
            
        Returns:
            输出预测，形状 (batch_size, num_classes)
        """
        # 转置为: (batch_size, input_size, sequence_length)
        x = x.transpose(1, 2)
        
        # 卷积层
        x = self.conv_layers(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.fc_layers(x)
        return x


def train_cnn_model(X_train: pd.DataFrame, y_train: pd.Series,
                    X_test: pd.DataFrame, y_test: pd.Series,
                    config: Dict[str, Any], progress_callback=None) -> Tuple[CNNModel, Dict[str, Any]]:
    """
    训练CNN模型
    
    Args:
        X_train: 训练特征
        y_train: 训练标签
        X_test: 测试特征
        y_test: 测试标签
        config: 配置参数
        progress_callback: 进度回调函数
        
    Returns:
        (训练好的模型, 训练信息)
    """
    # 超参数
    learning_rate = config.get('learning_rate', 0.001)
    epochs = config.get('epochs', 100)
    batch_size = config.get('batch_size', 32)
    dropout = config.get('dropout', 0.3)
    sequence_length = config.get('sequence_length', 20)
    early_stopping_patience = config.get('early_stopping_patience', 10)
    
    logger.info(f"开始训练CNN模型: sequence_length={sequence_length}")
    
    # 数据准备
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train_seq, y_train_seq)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 模型初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNModel(
        input_size=X_train.shape[1],
        num_classes=1,
        dropout=dropout
    ).to(device)
    
    # 损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    # 训练循环
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"早停于epoch {epoch}")
                break
        
        if progress_callback and epoch % 10 == 0:
            progress_callback(int((epoch / epochs) * 100), f"Epoch {epoch}, Loss: {avg_loss:.4f}")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # 评估
    model.eval()
    with torch.no_grad():
        X_test_seq = X_test_seq.to(device)
        y_pred_proba = model(X_test_seq).cpu().numpy().squeeze()
        y_pred = (y_pred_proba > 0.5).astype(int)
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    accuracy = accuracy_score(y_test_seq.numpy(), y_pred)
    precision = precision_score(y_test_seq.numpy(), y_pred, zero_division=0)
    recall = recall_score(y_test_seq.numpy(), y_pred, zero_division=0)
    f1 = f1_score(y_test_seq.numpy(), y_pred, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_test_seq.numpy(), y_pred_proba)
    except:
        roc_auc = None
    
    training_info = {
        'best_loss': best_loss,
        'epochs_trained': epoch + 1,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'device': str(device)
    }
    
    logger.info(f"CNN训练完成: accuracy={accuracy:.4f}, f1={f1:.4f}")
    
    return model, training_info


# ==================== Transformer模型 ====================

class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Transformer模型用于多因子建模"""
    
    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8, 
                 num_layers: int = 4, dim_feedforward: int = 512, 
                 dropout: float = 0.1, output_size: int = 1):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        
        # 输入嵌入层
        self.input_embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_size),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入数据，形状 (batch_size, sequence_length, input_size)
            
        Returns:
            输出预测，形状 (batch_size, output_size)
        """
        # 输入嵌入
        x = self.input_embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Transformer编码
        x = self.transformer_encoder(x)
        
        # 取最后一个时间步
        x = x[:, -1, :]
        
        # 输出
        x = self.output_layer(x)
        return x


def train_transformer_model(X_train: pd.DataFrame, y_train: pd.Series,
                           X_test: pd.DataFrame, y_test: pd.Series,
                           config: Dict[str, Any], progress_callback=None) -> Tuple[TransformerModel, Dict[str, Any]]:
    """
    训练Transformer模型
    
    Args:
        X_train: 训练特征
        y_train: 训练标签
        X_test: 测试特征
        y_test: 测试标签
        config: 配置参数
        progress_callback: 进度回调函数
        
    Returns:
        (训练好的模型, 训练信息)
    """
    # 超参数
    d_model = config.get('d_model', 128)
    nhead = config.get('nhead', 8)
    num_layers = config.get('num_layers', 4)
    dim_feedforward = config.get('dim_feedforward', 512)
    learning_rate = config.get('learning_rate', 0.0001)
    epochs = config.get('epochs', 100)
    batch_size = config.get('batch_size', 32)
    dropout = config.get('dropout', 0.1)
    sequence_length = config.get('sequence_length', 20)
    early_stopping_patience = config.get('early_stopping_patience', 10)
    
    logger.info(f"开始训练Transformer模型: d_model={d_model}, nhead={nhead}, num_layers={num_layers}")
    
    # 数据准备
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train_seq, y_train_seq)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 模型初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerModel(
        input_size=X_train.shape[1],
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        output_size=1
    ).to(device)
    
    # 损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    # 训练循环
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"早停于epoch {epoch}")
                break
        
        if progress_callback and epoch % 10 == 0:
            progress_callback(int((epoch / epochs) * 100), f"Epoch {epoch}, Loss: {avg_loss:.4f}")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # 评估
    model.eval()
    with torch.no_grad():
        X_test_seq = X_test_seq.to(device)
        y_pred_proba = model(X_test_seq).cpu().numpy().squeeze()
        y_pred = (y_pred_proba > 0.5).astype(int)
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    accuracy = accuracy_score(y_test_seq.numpy(), y_pred)
    precision = precision_score(y_test_seq.numpy(), y_pred, zero_division=0)
    recall = recall_score(y_test_seq.numpy(), y_pred, zero_division=0)
    f1 = f1_score(y_test_seq.numpy(), y_pred, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_test_seq.numpy(), y_pred_proba)
    except:
        roc_auc = None
    
    training_info = {
        'best_loss': best_loss,
        'epochs_trained': epoch + 1,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'device': str(device)
    }
    
    logger.info(f"Transformer训练完成: accuracy={accuracy:.4f}, f1={f1:.4f}")
    
    return model, training_info
