# FeatureEngine API 文档

## 概述
`FeatureEngine` 是特征层的核心引擎，负责协调和管理整个特征工程流程，包括数据预处理、特征生成、特征选择和特征标准化等功能。

## 类和方法

### FeatureEngine
特征工程核心引擎，提供统一的特征工程接口

#### 初始化
```python
def __init__(self, config=None):
```

**参数**:
- `config` (dict, optional): 引擎配置字典

**示例**:
```python
from src.features.core.engine import FeatureEngine

# 使用默认配置
engine = FeatureEngine()

# 使用自定义配置
config = {
    'preprocessing': {
        'validate_data': True,
        'auto_repair': True
    },
    'feature_generation': {
        'enable_caching': True,
        'parallel_processing': True
    },
    'feature_selection': {
        'method': 'importance',
        'n_features': 10
    }
}
engine = FeatureEngine(config=config)
```

#### 方法

##### process_data(data, indicators=None, params=None)
处理数据并生成特征

**参数**:
- `data` (pd.DataFrame): 输入数据
- `indicators` (list, optional): 技术指标列表
- `params` (dict, optional): 处理参数

**返回**:
- `pd.DataFrame`: 处理后的特征数据

**示例**:
```python
import pandas as pd

# 准备数据
data = pd.DataFrame({
    'open': [100, 101, 102, 103, 104],
    'high': [102, 103, 104, 105, 106],
    'low': [99, 100, 101, 102, 103],
    'close': [101, 102, 103, 104, 105],
    'volume': [1000, 1100, 1200, 1300, 1400]
})

# 处理数据
features = engine.process_data(
    data=data,
    indicators=["ma", "rsi", "macd"],
    params={
        "ma": {"window": [5, 10]},
        "rsi": {"window": 14},
        "macd": {"fast": 12, "slow": 26, "signal": 9}
    }
)

print("处理完成，特征形状:", features.shape)
```

##### preprocess_data(data)
预处理数据

**参数**:
- `data` (pd.DataFrame): 原始数据

**返回**:
- `pd.DataFrame`: 预处理后的数据

**示例**:
```python
# 预处理数据
processed_data = engine.preprocess_data(data)
print("预处理完成，数据形状:", processed_data.shape)
```

##### generate_features(data, indicators=None, params=None)
生成特征

**参数**:
- `data` (pd.DataFrame): 输入数据
- `indicators` (list, optional): 技术指标列表
- `params` (dict, optional): 特征参数

**返回**:
- `pd.DataFrame`: 生成的特征数据

**示例**:
```python
# 生成特征
features = engine.generate_features(
    data=data,
    indicators=["ma", "rsi"],
    params={"ma": {"window": [5, 10]}, "rsi": {"window": 14}}
)

print("特征生成完成，特征列:", features.columns.tolist())
```

##### select_features(features, method='importance', n_features=None)
选择特征

**参数**:
- `features` (pd.DataFrame): 特征数据
- `method` (str): 选择方法
- `n_features` (int, optional): 选择的特征数量

**返回**:
- `pd.DataFrame`: 选择后的特征数据

**示例**:
```python
# 选择特征
selected_features = engine.select_features(
    features=features,
    method='importance',
    n_features=5
)

print("特征选择完成，选择特征数:", selected_features.shape[1])
```

##### standardize_features(features, method='zscore')
标准化特征

**参数**:
- `features` (pd.DataFrame): 特征数据
- `method` (str): 标准化方法

**返回**:
- `pd.DataFrame`: 标准化后的特征数据

**示例**:
```python
# 标准化特征
standardized_features = engine.standardize_features(
    features=selected_features,
    method='zscore'
)

print("特征标准化完成")
```

##### get_processing_pipeline()
获取处理流水线

**返回**:
- `dict`: 处理流水线配置

**示例**:
```python
# 获取处理流水线
pipeline = engine.get_processing_pipeline()
print("处理流水线:", pipeline)
```

##### set_processing_pipeline(pipeline)
设置处理流水线

**参数**:
- `pipeline` (dict): 处理流水线配置

**返回**:
- `None`

**示例**:
```python
# 设置处理流水线
pipeline_config = {
    'preprocessing': True,
    'feature_generation': True,
    'feature_selection': True,
    'feature_standardization': True
}
engine.set_processing_pipeline(pipeline_config)
```

##### get_processing_stats()
获取处理统计信息

**返回**:
- `dict`: 处理统计信息

**示例**:
```python
# 获取处理统计
stats = engine.get_processing_stats()
print("处理统计:", stats)
```

## 配置选项

### 预处理配置
```python
preprocessing_config = {
    "validate_data": True,        # 数据验证
    "auto_repair": True,          # 自动修复
    "handle_missing": True,       # 处理缺失值
    "remove_outliers": False      # 移除异常值
}
```

### 特征生成配置
```python
feature_generation_config = {
    "enable_caching": True,       # 启用缓存
    "parallel_processing": True,   # 并行处理
    "memory_optimization": True,   # 内存优化
    "chunk_size": 1000            # 分块大小
}
```

### 特征选择配置
```python
feature_selection_config = {
    "method": "importance",        # 选择方法
    "n_features": 10,             # 特征数量
    "threshold": 0.01,            # 阈值
    "random_state": 42            # 随机种子
}
```

### 特征标准化配置
```python
feature_standardization_config = {
    "method": "zscore",           # 标准化方法
    "with_mean": True,            # 减去均值
    "with_std": True,             # 除以标准差
    "copy": True                  # 复制数据
}
```

## 使用示例

### 基础使用
```python
from src.features.core.engine import FeatureEngine
import pandas as pd

# 创建引擎
engine = FeatureEngine()

# 准备数据
data = pd.DataFrame({
    'open': [100, 101, 102, 103, 104],
    'high': [102, 103, 104, 105, 106],
    'low': [99, 100, 101, 102, 103],
    'close': [101, 102, 103, 104, 105],
    'volume': [1000, 1100, 1200, 1300, 1400]
})

# 处理数据
features = engine.process_data(
    data=data,
    indicators=["ma", "rsi"],
    params={"ma": {"window": [5, 10]}, "rsi": {"window": 14}}
)

print("处理完成，特征形状:", features.shape)
```

### 高级使用
```python
# 自定义配置
config = {
    'preprocessing': {
        'validate_data': True,
        'auto_repair': True
    },
    'feature_generation': {
        'enable_caching': True,
        'parallel_processing': True
    },
    'feature_selection': {
        'method': 'importance',
        'n_features': 5
    },
    'feature_standardization': {
        'method': 'zscore'
    }
}

engine = FeatureEngine(config=config)

# 分步处理
# 1. 预处理
processed_data = engine.preprocess_data(data)

# 2. 生成特征
features = engine.generate_features(
    processed_data,
    indicators=["ma", "rsi", "macd"],
    params={
        "ma": {"window": [5, 10, 20]},
        "rsi": {"window": 14},
        "macd": {"fast": 12, "slow": 26, "signal": 9}
    }
)

# 3. 选择特征
selected_features = engine.select_features(features, n_features=5)

# 4. 标准化特征
final_features = engine.standardize_features(selected_features)

print("最终特征形状:", final_features.shape)
print("最终特征列:", final_features.columns.tolist())
```

### 流水线处理
```python
# 设置自定义流水线
pipeline = {
    'preprocessing': True,
    'feature_generation': True,
    'feature_selection': True,
    'feature_standardization': True
}

engine.set_processing_pipeline(pipeline)

# 执行完整流水线
final_features = engine.process_data(
    data=data,
    indicators=["ma", "rsi", "macd", "bollinger"],
    params={
        "ma": {"window": [5, 10, 20]},
        "rsi": {"window": 14},
        "macd": {"fast": 12, "slow": 26, "signal": 9},
        "bollinger": {"window": 20, "num_std": 2}
    }
)

# 获取处理统计
stats = engine.get_processing_stats()
print("处理统计:", stats)
```

## 性能优化建议

### 1. 配置优化
- 根据数据规模调整分块大小
- 启用并行处理提高性能
- 使用缓存减少重复计算

### 2. 内存管理
- 设置合理的内存限制
- 及时释放不需要的数据
- 使用数据分块处理

### 3. 流水线优化
- 根据需求配置处理流水线
- 跳过不必要的处理步骤
- 优化处理顺序

## 故障排除

### 常见问题

#### 1. 内存不足
**问题**: 处理大数据集时内存不足
**解决方案**: 调整分块大小，启用内存优化

#### 2. 处理速度慢
**问题**: 特征生成速度慢
**解决方案**: 启用并行处理和缓存

#### 3. 配置错误
**问题**: 配置参数错误
**解决方案**: 检查配置格式，使用默认配置

### 调试技巧

#### 1. 检查处理统计
```python
stats = engine.get_processing_stats()
print("处理统计:", stats)
```

#### 2. 分步调试
```python
# 分步执行，检查每步结果
processed_data = engine.preprocess_data(data)
print("预处理完成")

features = engine.generate_features(processed_data)
print("特征生成完成")

selected_features = engine.select_features(features)
print("特征选择完成")
```

## 版本历史

### v1.0.0
- 初始版本
- 基础特征工程功能
- 流水线处理

### v1.1.0
- 添加缓存机制
- 优化性能
- 增强错误处理

### v1.2.0
- 添加并行处理
- 增强配置管理
- 完善文档

---

**文档版本**: 1.2.0  
**最后更新**: 2025-01-27  
**维护者**: 开发团队 