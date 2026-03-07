# PyTorch深度学习模型实现计划

## 评估目标

实现LSTM、CNN、Transformer深度学习模型支持，充分利用已安装的PyTorch（版本2.8.0+cu128）。

## 当前状态

### PyTorch安装状态 ✅

**验证结果**:
```
PyTorch版本: 2.8.0+cu128
```

- PyTorch已安装且支持CUDA（GPU加速）
- 无需额外安装步骤

### 当前模型支持状态

**文件**: `src/ml/real_model_trainer.py` 第355-365行

```python
elif model_type == 'LSTM':
    # LSTM需要PyTorch，暂不支持，使用RandomForest替代
    logger.warning("LSTM需要PyTorch，暂不支持，使用RandomForest替代")
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

elif model_type == 'CNN' or model_type == 'Transformer':
    # CNN和Transformer需要PyTorch，暂不支持
    logger.warning(f"{model_type}需要PyTorch，暂不支持，使用RandomForest替代")
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
```

**问题**: PyTorch已安装但深度学习模型未实现，用户选择LSTM/CNN/Transformer时实际得到RandomForest

---

## 模型实现方案

### 1. LSTM模型实现（高优先级）

**适用场景**: 时序数据建模，适合股票价格趋势预测

**实现代码**:
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class LSTMModel(nn.Module):
    """LSTM模型用于股票价格预测"""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
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
    
    def forward(self, x):
        # LSTM前向传播
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        
        # 全连接层
        output = self.fc(last_output)
        return output


def train_lstm_model(X_train, y_train, X_test, y_test, config, progress_callback=None):
    """训练LSTM模型"""
    
    # 超参数
    hidden_size = config.get('hidden_size', 64)
    num_layers = config.get('num_layers', 2)
    learning_rate = config.get('learning_rate', 0.001)
    epochs = config.get('epochs', 100)
    batch_size = config.get('batch_size', 32)
    dropout = config.get('dropout', 0.2)
    
    # 数据准备 - 转换为时间序列格式
    sequence_length = config.get('sequence_length', 20)  # 使用过去20天的数据预测
    
    def create_sequences(X, y, seq_length):
        """创建时间序列数据"""
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_length):
            X_seq.append(X.iloc[i:i+seq_length].values)
            y_seq.append(y.iloc[i+seq_length])
        return torch.FloatTensor(X_seq), torch.FloatTensor(y_seq)
    
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train_seq, y_train_seq)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 模型初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    # 训练循环
    best_loss = float('inf')
    patience = config.get('early_stopping_patience', 10)
    patience_counter = 0
    
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
            if patience_counter >= patience:
                print(f"早停于epoch {epoch}")
                break
        
        # 进度回调
        if progress_callback and epoch % 10 == 0:
            progress_callback(int((epoch / epochs) * 100), f"Epoch {epoch}, Loss: {avg_loss:.4f}")
    
    # 加载最佳模型
    model.load_state_dict(best_model_state)
    
    return model
```

---

### 2. CNN模型实现（中优先级）

**适用场景**: K线形态识别，图像化价格数据特征提取

**实现代码**:
```python
class CNNModel(nn.Module):
    """CNN模型用于K线形态识别"""
    
    def __init__(self, input_size, num_classes=1, dropout=0.3):
        super(CNNModel, self).__init__()
        
        # 将1D数据转换为2D图像格式（类似K线图）
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
    
    def forward(self, x):
        # x形状: (batch_size, sequence_length, features)
        # 转置为: (batch_size, features, sequence_length)
        x = x.transpose(1, 2)
        
        # 卷积层
        x = self.conv_layers(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.fc_layers(x)
        return x
```

---

### 3. Transformer模型实现（中优先级）

**适用场景**: 多因子模型，注意力机制捕捉长距离依赖

**实现代码**:
```python
class TransformerModel(nn.Module):
    """Transformer模型用于多因子建模"""
    
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=4, 
                 dim_feedforward=512, dropout=0.1, output_size=1):
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
    
    def forward(self, x):
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


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)
```

---

## 实施计划

### 阶段1: 实现LSTM模型（高优先级）

**任务**:
1. 创建 `src/ml/pytorch_models.py` 文件
2. 实现LSTM模型类和训练函数
3. 修改 `real_model_trainer.py` 集成LSTM支持
4. 添加LSTM超参数配置
5. 测试LSTM模型训练

**预期结果**:
- 用户可以选择LSTM模型进行训练
- 支持GPU加速训练
- 支持早停和学习率调度

---

### 阶段2: 实现CNN模型（中优先级）

**任务**:
1. 在 `pytorch_models.py` 中添加CNN模型
2. 实现CNN训练函数
3. 修改 `real_model_trainer.py` 集成CNN支持
4. 添加CNN超参数配置
5. 测试CNN模型训练

**预期结果**:
- 用户可以选择CNN模型进行训练
- CNN可以识别K线形态特征

---

### 阶段3: 实现Transformer模型（中优先级）

**任务**:
1. 在 `pytorch_models.py` 中添加Transformer模型
2. 实现位置编码
3. 实现Transformer训练函数
4. 修改 `real_model_trainer.py` 集成Transformer支持
5. 添加Transformer超参数配置
6. 测试Transformer模型训练

**预期结果**:
- 用户可以选择Transformer模型进行训练
- Transformer可以捕捉长距离依赖关系

---

### 阶段4: 前端界面更新（低优先级）

**任务**:
1. 更新模型类型选择说明
2. 添加深度学习模型超参数配置界面
3. 显示GPU使用情况
4. 添加模型训练可视化（损失曲线等）

---

## 检查清单

- [ ] 创建PyTorch模型文件
- [ ] 实现LSTM模型
- [ ] 实现CNN模型
- [ ] 实现Transformer模型
- [ ] 修改real_model_trainer.py支持深度学习模型
- [ ] 添加超参数配置
- [ ] 测试LSTM模型训练
- [ ] 测试CNN模型训练
- [ ] 测试Transformer模型训练
- [ ] 更新前端界面
- [ ] 更新文档

---

## 风险评估

| 风险项 | 风险等级 | 影响 | 缓解措施 |
|--------|---------|------|---------|
| GPU内存不足 | 中 | 大模型训练可能OOM | 添加梯度累积和混合精度训练 |
| 训练时间过长 | 中 | 深度学习模型训练慢 | 支持早停和模型检查点 |
| 模型过拟合 | 中 | 泛化能力差 | 添加正则化和Dropout |
| 超参数调优困难 | 低 | 模型性能不佳 | 提供默认超参数配置 |

---

## 优先级

**高** - LSTM模型是量化交易的核心需求，应尽快实现