# 特征层配置API文档

## 概述

特征层配置API提供了统一的配置管理接口，遵循单一来源原则，所有配置类都在core层统一定义。

## 核心配置类

### OrderBookConfig

订单簿配置类，单一来源定义。

#### 类定义
```python
@dataclass
class OrderBookConfig:
    """订单簿配置类 - 单一来源定义"""
    
    # 基础配置
    orderbook_type: OrderBookType = OrderBookType.LEVEL2
    depth: int = 10  # 订单簿深度
    update_frequency: float = 1.0  # 更新频率（秒）
    
    # 分析配置
    enable_imbalance_analysis: bool = True
    enable_skew_analysis: bool = True
    enable_spread_analysis: bool = True
    enable_depth_analysis: bool = True
    
    # 指标配置
    imbalance_threshold: float = 0.1
    skew_threshold: float = 0.05
    spread_threshold: float = 0.001
    
    # 缓存配置
    enable_caching: bool = True
    cache_ttl: int = 60  # 缓存时间（秒）
    
    # 性能配置
    max_workers: int = 4
    batch_size: int = 1000
    
    # 自定义配置
    custom_config: Dict[str, Any] = field(default_factory=dict)
```

#### 导入方式
```python
# 推荐方式：从core层直接导入
from src.features.core.config import OrderBookConfig, OrderBookType

# 兼容方式：从特征层主模块导入
from src.features import OrderBookConfig, OrderBookType
```

#### 使用示例
```python
# 创建基本配置
config = OrderBookConfig(
    depth=15,
    orderbook_type=OrderBookType.LEVEL2,
    enable_imbalance_analysis=True,
    imbalance_threshold=0.15
)

# 配置验证
if config.validate():
    print("配置有效")
else:
    print("配置无效")

# 字典转换
config_dict = config.to_dict()
new_config = OrderBookConfig.from_dict(config_dict)

# 自定义配置
config.custom_config['custom_param'] = 'custom_value'
```

#### 方法说明

##### validate()
验证配置有效性。

**返回值**: `bool`
- `True`: 配置有效
- `False`: 配置无效

**验证规则**:
- `depth` 必须大于0
- `update_frequency` 必须大于0
- `max_workers` 必须大于0
- `batch_size` 必须大于0

##### to_dict()
将配置转换为字典。

**返回值**: `Dict[str, Any]`

**示例**:
```python
config = OrderBookConfig(depth=10)
config_dict = config.to_dict()
# 结果: {'orderbook_type': 'level2', 'depth': 10, ...}
```

##### from_dict()
从字典创建配置。

**参数**:
- `config_dict`: `Dict[str, Any]` - 配置字典

**返回值**: `OrderBookConfig`

**示例**:
```python
config_dict = {
    'depth': 20,
    'orderbook_type': 'level3',
    'enable_caching': False
}
config = OrderBookConfig.from_dict(config_dict)
```

### OrderBookType

订单簿类型枚举。

```python
class OrderBookType(Enum):
    """订单簿类型枚举"""
    LEVEL1 = "level1"
    LEVEL2 = "level2"
    LEVEL3 = "level3"
```

## 其他配置类

### FeatureConfig

特征配置类，用于管理特征处理的整体配置。

```python
@dataclass
class FeatureConfig:
    """特征配置类"""
    
    # 基本配置
    feature_types: List[FeatureType] = field(default_factory=lambda: [FeatureType.TECHNICAL])
    enable_feature_selection: bool = True
    enable_standardization: bool = True
    enable_feature_saving: bool = False
    
    # 技术指标配置
    technical_indicators: List[str] = field(default_factory=lambda: [
        "sma", "ema", "rsi", "macd", "bbands", "atr"
    ])
    technical_params: TechnicalParams = field(default_factory=TechnicalParams)
    
    # 情感分析配置
    sentiment_types: List[str] = field(default_factory=lambda: [
        "news_sentiment", "social_sentiment"
    ])
    sentiment_params: SentimentParams = field(default_factory=SentimentParams)
    
    # 特征选择配置
    feature_selection_method: str = "mutual_info"
    max_features: int = 50
    min_features: int = 5
    
    # 标准化配置
    standardization_method: str = "zscore"  # zscore, minmax, robust
    handle_outliers: bool = True
    outlier_method: str = "iqr"  # iqr, zscore, isolation_forest
    
    # 缓存配置
    enable_caching: bool = True
    cache_dir: str = "./feature_cache"
    cache_ttl: int = 3600  # 1小时
    
    # 性能配置
    parallel_processing: bool = True
    max_workers: int = 4
    chunk_size: int = 1000
    
    # 验证配置
    enable_validation: bool = True
    strict_validation: bool = False
```

### TechnicalParams

技术指标参数配置。

```python
@dataclass
class TechnicalParams:
    """技术指标参数配置"""
    sma_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    ema_periods: List[int] = field(default_factory=lambda: [12, 26])
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    rsi_period: int = 14
    adx_period: int = 14
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    cci_period: int = 20
    roc_period: int = 10
    atr_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    kc_period: int = 20
    kc_multiplier: float = 2.0
    mfi_period: int = 14
```

### SentimentParams

情感分析参数配置。

```python
@dataclass
class SentimentParams:
    """情感分析参数配置"""
    news_lookback_days: int = 30
    social_lookback_days: int = 7
    min_confidence: float = 0.6
    max_keywords: int = 100
    language: str = "zh-cn"
```

### FeatureProcessingConfig

特征处理配置类。

```python
@dataclass
class FeatureProcessingConfig:
    """特征处理配置类"""
    
    # 处理参数
    batch_size: int = 1000
    timeout: float = 30.0
    retry_count: int = 3
    retry_delay: float = 1.0
    
    # 内存管理
    max_memory_usage: float = 1024.0  # MB
    enable_memory_monitoring: bool = True
    
    # 错误处理
    continue_on_error: bool = False
    log_errors: bool = True
    
    # 并行处理
    enable_parallel_processing: bool = True
    max_workers: int = 4
    chunk_size: int = 1000
```

## 预定义配置

### DefaultConfigs

提供预定义的标准配置。

```python
class DefaultConfigs:
    """预定义配置"""
    
    @staticmethod
    def basic_technical() -> FeatureConfig:
        """基础技术指标配置"""
        return FeatureConfig(
            feature_types=[FeatureType.TECHNICAL],
            technical_indicators=["sma", "rsi", "macd"],
            enable_feature_selection=False,
            enable_standardization=True
        )
    
    @staticmethod
    def comprehensive_technical() -> FeatureConfig:
        """全面技术指标配置"""
        return FeatureConfig(
            feature_types=[FeatureType.TECHNICAL],
            technical_indicators=["sma", "ema", "rsi", "macd", "bbands", "atr", "stoch", "cci"],
            enable_feature_selection=True,
            enable_standardization=True,
            max_features=30
        )
    
    @staticmethod
    def sentiment_analysis() -> FeatureConfig:
        """情感分析配置"""
        return FeatureConfig(
            feature_types=[FeatureType.SENTIMENT],
            sentiment_types=["news_sentiment", "social_sentiment"],
            enable_feature_selection=True,
            enable_standardization=True
        )
    
    @staticmethod
    def full_feature() -> FeatureConfig:
        """完整特征配置"""
        return FeatureConfig(
            feature_types=[FeatureType.TECHNICAL, FeatureType.SENTIMENT],
            technical_indicators=["sma", "ema", "rsi", "macd", "bbands", "atr"],
            sentiment_types=["news_sentiment"],
            enable_feature_selection=True,
            enable_standardization=True,
            max_features=50
        )
```

## 使用最佳实践

### 1. 配置创建
```python
# 使用预定义配置
config = DefaultConfigs.basic_technical()

# 自定义配置
config = FeatureConfig(
    feature_types=[FeatureType.TECHNICAL],
    technical_indicators=["sma", "rsi"],
    enable_feature_selection=True,
    max_features=20
)

# 订单簿配置
orderbook_config = OrderBookConfig(
    depth=15,
    orderbook_type=OrderBookType.LEVEL2,
    enable_imbalance_analysis=True
)
```

### 2. 配置验证
```python
# 验证配置
if config.validate():
    # 使用配置
    engine = FeatureEngine()
    features = engine.process_features(data, config)
else:
    print("配置无效，请检查参数")
```

### 3. 配置序列化
```python
# 保存配置
config_dict = config.to_dict()
with open('config.json', 'w') as f:
    json.dump(config_dict, f, indent=2)

# 加载配置
with open('config.json', 'r') as f:
    config_dict = json.load(f)
config = FeatureConfig.from_dict(config_dict)
```

### 4. 配置组合
```python
# 组合多个配置
feature_config = DefaultConfigs.comprehensive_technical()
orderbook_config = OrderBookConfig(depth=10)

# 在特征引擎中使用
engine = FeatureEngine()
engine.set_feature_config(feature_config)
engine.set_orderbook_config(orderbook_config)
```

## 错误处理

### 配置验证错误
```python
try:
    config = OrderBookConfig(depth=-1)  # 无效配置
    if not config.validate():
        print("配置验证失败")
except ValueError as e:
    print(f"配置错误: {e}")
```

### 导入错误
```python
try:
    from src.features.core.config import OrderBookConfig
except ImportError as e:
    print(f"导入错误: {e}")
    # 使用兼容导入
    from src.features import OrderBookConfig
```

## 版本兼容性

### 向后兼容性
- 保持原有的导入路径
- 现有代码无需修改
- 新增功能不影响现有功能

### 迁移指南
```python
# 旧版本导入（仍然支持）
from src.features.config import OrderBookConfig

# 新版本导入（推荐）
from src.features.core.config import OrderBookConfig
# 或者
from src.features import OrderBookConfig
```

## 性能考虑

### 配置缓存
```python
# 配置对象可以缓存以提高性能
config_cache = {}

def get_cached_config(config_name: str) -> FeatureConfig:
    if config_name not in config_cache:
        config_cache[config_name] = DefaultConfigs.basic_technical()
    return config_cache[config_name]
```

### 内存优化
```python
# 对于大量配置，使用字典而不是对象
config_dict = {
    'depth': 10,
    'orderbook_type': 'level2'
}

# 只在需要时创建对象
config = OrderBookConfig.from_dict(config_dict)
```

## 测试

### 单元测试
```python
def test_orderbook_config():
    config = OrderBookConfig(depth=10)
    assert config.validate() == True
    assert config.depth == 10
    assert config.orderbook_type == OrderBookType.LEVEL2
```

### 集成测试
```python
def test_config_integration():
    feature_config = DefaultConfigs.basic_technical()
    orderbook_config = OrderBookConfig(depth=15)
    
    engine = FeatureEngine()
    engine.set_feature_config(feature_config)
    engine.set_orderbook_config(orderbook_config)
    
    # 测试配置集成
    assert engine.feature_config == feature_config
    assert engine.orderbook_config == orderbook_config
``` 