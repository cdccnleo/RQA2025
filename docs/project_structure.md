# 量化交易系统项目结构文档

## 项目概述
本项目是一个适用于A股市场的量化交易系统，采用LSTM、随机森林和神经网络等多模型集成策略。系统采用分层架构设计，包含数据层、特征层、模型层和交易层。

## 系统架构图
```
[数据层] -> [特征层] -> [模型层] -> [交易层]
    ↑           ↑           ↑
[基础设施层]-------+-----------+
```

## 各层详细说明

### 数据层 (src/data)
核心职责：多源数据集成与管理

#### 核心模块
- `data_manager.py` - 数据管理核心类
- `loader/` - 数据加载器
  - `stock_loader.py` - 股票数据加载
  - `index_loader.py` - 指数数据加载
  - `financial_loader.py` - 财务数据加载
  - `news_loader.py` - 新闻数据加载
- `processing/` - 数据处理
- `validator.py` - 数据验证
- `version_control/` - 数据版本控制

#### 关键类
- `DataManager` - 统一数据入口
- `StockDataLoader` - 股票数据加载器
- `FinancialNewsLoader` - 财经新闻加载器

### 特征层 (src/features)
核心职责：特征工程处理

#### 核心模块
- `feature_manager.py` - 特征流程控制
- `processors/` - 特征处理器
  - `technical_processor.py` - 技术指标
  - `sentiment_analyzer.py` - 情感分析
- `feature_selector.py` - 特征选择
- `feature_standardizer.py` - 特征标准化

#### 关键类
- `FeatureManager` - 特征流程控制器
- `TechnicalProcessor` - 技术指标处理器
- `SentimentAnalyzer` - 多模型情感分析

### 模型层 (src/models)
核心职责：模型训练与预测

#### 核心模块
- `base_model.py` - 模型抽象基类
- `lstm.py` - LSTM模型实现
- `nn.py` - 神经网络模型
- `rf.py` - 随机森林模型
- `model_manager.py` - 模型生命周期管理

#### 关键类
- `BaseModel` - 模型抽象基类
- `AttentionLSTM` - 注意力LSTM模型
- `ModelManager` - 模型管理

### 交易层 (src/trading)
核心职责：交易策略执行

#### 核心模块
- `backtest_analyzer.py` - 回测分析
- `execution_engine.py` - 交易执行
- `gateway.py` - 交易网关
- `intelligent_rebalancing.py` - 智能调仓

#### 关键类
- `EnhancedTradingStrategy` - 增强交易策略
- `BacktestAnalyzer` - 回测分析器

## 核心数据流
1. 数据层获取原始数据
2. 特征层生成特征矩阵
3. 模型层进行预测
4. 交易层执行策略
5. 回测分析评估效果

## 基础设施层 (src/infrastructure)
- 配置管理
- 日志监控
- 异常处理
- 资源管理
