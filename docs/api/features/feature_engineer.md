# FeatureEngineer API 文档

## 概述
`FeatureEngineer` 是特征层的核心组件，负责特征工程的主要功能，包括技术指标计算、数据验证、特征生成等。

## 类和方法

### FeatureEngineer
特征工程器，提供统一的特征工程接口

#### 初始化
```python
def __init__(self, technical_processor=None, config=None):
```

**参数**:
- `technical_processor` (TechnicalProcessor, optional): 技术指标处理器实例
- `config` (dict, optional): 配置字典

**示例**:
```python
from src.features.feature_engineer import FeatureEngineer
from src.features.technical.technical_processor import TechnicalProcessor

# 使用默认技术指标处理器
engineer = FeatureEngineer()

# 使用自定义技术指标处理器
processor = TechnicalProcessor()
engineer = FeatureEngineer(technical_processor=processor)
```

#### 方法

##### generate_technical_features(stock_data, indicators=None, params=None)
生成技术指标特征

**参数**:
- `stock_data` (pd.DataFrame): 股票数据，必须包含OHLCV列
- `indicators` (list, optional): 技术指标列表，默认为["ma", "rsi"]
- `params` (dict, optional): 指标参数配置

**返回**:
- `pd.DataFrame`: 包含技术指标的特征数据框

**示例**:
```python
import pandas as pd

# 准备股票数据
data = pd.DataFrame({
    'open': [100, 101, 102, 103, 104],
    'high': [102, 103, 104, 105, 106],
    'low': [99, 100, 101, 102, 103],
    'close': [101, 102, 103, 104, 105],
    'volume': [1000, 1100, 1200, 1300, 1400]
}, index=pd.date_range('2023-01-01', periods=5))

# 生成技术指标
features = engineer.generate_technical_features(
    stock_data=data,
    indicators=["ma", "rsi", "macd"],
    params={
        "ma": {"window": [5, 10]},
        "rsi": {"window": 14},
        "macd": {"fast": 12, "slow": 26, "signal": 9}
    }
)
```

**异常**:
- `ValueError`: 当数据验证失败时抛出
- `KeyError`: 当缺少必需的列时抛出

##### validate_data(data)
验证输入数据的有效性

**参数**:
- `data` (pd.DataFrame): 待验证的数据

**返回**:
- `bool`: 验证结果，True表示有效

**示例**:
```python
# 验证数据
is_valid = engineer.validate_data(stock_data)
if not is_valid:
    print("数据验证失败")
```

##### get_available_indicators()
获取可用的技术指标列表

**返回**:
- `list`: 可用指标列表

**示例**:
```python
indicators = engineer.get_available_indicators()
print(f"可用指标: {indicators}")
```

##### set_config(config)
设置特征工程配置

**参数**:
- `config` (dict): 配置字典

**示例**:
```python
config = {
    "validation": {
        "strict_mode": False,
        "auto_repair": True
    },
    "caching": {
        "enable": True,
        "ttl": 3600
    }
}
engineer.set_config(config)
```

## 配置选项

### 数据验证配置
```python
validation_config = {
    "strict_mode": False,      # 严格模式，默认False
    "auto_repair": True,       # 自动修复，默认True
    "required_columns": ["open", "high", "low", "close", "volume"],
    "price_logic_check": True, # 价格逻辑检查
    "volume_check": True       # 成交量检查
}
```

### 缓存配置
```python
cache_config = {
    "enable": True,           # 启用缓存
    "ttl": 3600,             # 缓存时间（秒）
    "max_size": 1000,        # 最大缓存条目数
    "cleanup_interval": 300   # 清理间隔（秒）
}
```

### 性能配置
```python
performance_config = {
    "parallel_processing": True,  # 并行处理
    "max_workers": 4,            # 最大工作线程数
    "chunk_size": 1000,          # 数据分块大小
    "memory_limit": "2GB"        # 内存限制
}
```

## 使用示例

### 基础使用
```python
from src.features.feature_engineer import FeatureEngineer
import pandas as pd

# 创建特征工程器
engineer = FeatureEngineer()

# 准备数据
data = pd.DataFrame({
    'open': [100, 101, 102, 103, 104],
    'high': [102, 103, 104, 105, 106],
    'low': [99, 100, 101, 102, 103],
    'close': [101, 102, 103, 104, 105],
    'volume': [1000, 1100, 1200, 1300, 1400]
}, index=pd.date_range('2023-01-01', periods=5))

# 生成特征
features = engineer.generate_technical_features(data)
print(features.head())
```

### 高级使用
```python
# 自定义配置
config = {
    "validation": {
        "strict_mode": False,
        "auto_repair": True
    },
    "caching": {
        "enable": True,
        "ttl": 3600
    },
    "performance": {
        "parallel_processing": True,
        "max_workers": 4
    }
}

engineer = FeatureEngineer(config=config)

# 自定义指标和参数
indicators = ["ma", "rsi", "macd", "bollinger"]
params = {
    "ma": {"window": [5, 10, 20]},
    "rsi": {"window": 14},
    "macd": {"fast": 12, "slow": 26, "signal": 9},
    "bollinger": {"window": 20, "num_std": 2}
}

features = engineer.generate_technical_features(
    stock_data=data,
    indicators=indicators,
    params=params
)
```

### 错误处理
```python
try:
    features = engineer.generate_technical_features(data)
except ValueError as e:
    print(f"数据验证错误: {e}")
except KeyError as e:
    print(f"缺少必需列: {e}")
except Exception as e:
    print(f"未知错误: {e}")
```

## 性能优化建议

### 1. 数据预处理
- 确保输入数据格式正确
- 预先处理缺失值和异常值
- 使用适当的数据类型

### 2. 缓存策略
- 启用缓存以提高重复计算性能
- 合理设置缓存TTL
- 定期清理过期缓存

### 3. 并行处理
- 对于大数据集，启用并行处理
- 根据系统资源调整工作线程数
- 使用数据分块处理

### 4. 内存管理
- 设置合理的内存限制
- 及时释放不需要的数据
- 使用生成器处理大数据集

## 故障排除

### 常见问题

#### 1. 数据验证失败
**问题**: `ValueError: 缺少必需列: ['open']`
**解决方案**: 确保输入数据包含所有必需的OHLCV列

#### 2. 价格逻辑错误
**问题**: `ValueError: 价格高低值逻辑错误`
**解决方案**: 检查数据中的high >= low和high >= close >= low逻辑

#### 3. 内存不足
**问题**: 处理大数据集时内存不足
**解决方案**: 启用数据分块处理，调整chunk_size参数

#### 4. 计算性能慢
**问题**: 特征计算速度慢
**解决方案**: 启用并行处理和缓存机制

### 调试技巧

#### 1. 启用详细日志
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### 2. 检查数据质量
```python
# 检查数据基本信息
print(data.info())
print(data.describe())

# 检查缺失值
print(data.isnull().sum())
```

#### 3. 验证计算结果
```python
# 检查特征数据
print(features.info())
print(features.describe())

# 检查特定指标
if 'RSI' in features.columns:
    print(f"RSI范围: {features['RSI'].min()} - {features['RSI'].max()}")
```

## 版本历史

### v1.0.0
- 初始版本
- 基础特征工程功能
- 技术指标计算
- 数据验证机制

### v1.1.0
- 添加缓存机制
- 优化性能
- 增强错误处理
- 添加配置管理

### v1.2.0
- 添加并行处理
- 增强数据验证
- 添加自动修复功能
- 完善文档

---

**文档版本**: 1.2.0  
**最后更新**: 2025-01-27  
**维护者**: 开发团队 