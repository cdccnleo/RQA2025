# 检查市场数据获取方式计划

## 目标
检查市场数据的获取方式，为 `SimpleSignalGenerator.generate_signals()` 方法提供所需的 `data` 参数。

## 检查范围

### 1. SimpleSignalGenerator 实现检查
- [ ] 检查 `generate_signals()` 方法的参数要求
- [ ] 检查 `data` 参数的数据类型和格式
- [ ] 检查是否有默认值或可选参数

### 2. 市场数据来源检查
- [ ] 检查实时数据流接口
- [ ] 检查历史数据获取方式
- [ ] 检查数据适配器/连接器

### 3. TradingSignalService 调用检查
- [ ] 检查 `get_realtime_signals()` 函数
- [ ] 检查如何获取市场数据
- [ ] 检查数据转换逻辑

### 4. 数据流架构检查
- [ ] 检查数据从数据源到信号生成器的流程
- [ ] 检查缓存机制
- [ ] 检查数据格式转换

## 具体检查项

### SimpleSignalGenerator.generate_signals 方法
```python
# 检查点：
# 1. data 参数的类型（DataFrame?）
# 2. data 参数包含哪些列（open, high, low, close, volume?）
# 3. 是否可以为空或使用默认数据
```

### 市场数据获取
```python
# 检查点：
# 1. 从实时引擎获取数据
# 2. 从数据适配器获取数据
# 3. 从缓存获取数据
```

### 数据格式
```python
# 检查点：
# 1. pandas DataFrame 格式
# 2. 列名要求
# 3. 时间序列索引
```

## 验证步骤

### 步骤 1: 检查 SimpleSignalGenerator 实现
1. 读取 `signal_signal_generator.py` 文件
2. 分析 `generate_signals()` 方法签名
3. 检查 `data` 参数要求

### 步骤 2: 检查 TradingSignalService
1. 读取 `trading_signal_service.py` 文件
2. 分析 `get_realtime_signals()` 函数
3. 检查如何获取市场数据

### 步骤 3: 检查数据适配器
1. 检查数据管理层接口
2. 检查实时数据流
3. 检查历史数据查询

### 步骤 4: 设计解决方案
1. 确定数据获取方式
2. 设计数据转换逻辑
3. 实现修复方案

## 预期结果

### 数据格式
```python
# 预期的 DataFrame 格式
import pandas as pd

data = pd.DataFrame({
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
}, index=pd.DatetimeIndex([...]))
```

### 修复方案
1. 在调用 `generate_signals()` 前获取市场数据
2. 转换数据格式为 pandas DataFrame
3. 处理数据缺失情况

## 问题排查

### 常见问题
1. **数据缺失**: 没有可用的市场数据
2. **格式不匹配**: 数据格式不符合要求
3. **实时性**: 数据不是最新的

### 修复方案
根据检查结果，可能需要：
1. 添加数据获取逻辑
2. 添加数据格式转换
3. 添加默认数据处理
