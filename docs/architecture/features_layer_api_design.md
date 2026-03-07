# 特征层API文档

## 概述

本文档详细介绍了RQA特征层的API接口、使用方法和最佳实践。特征层提供了完整的特征工程、处理、选择、标准化和保存功能。

## 核心组件API

### FeatureEngine（特征引擎）

特征引擎是特征层的核心协调器，提供了统一的特征处理接口。

#### 初始化

```python
from src.features import FeatureEngine, DefaultConfigs

# 使用默认配置初始化
engine = FeatureEngine()

# 使用自定义配置初始化
from src.features.core.config import FeatureConfig
config = FeatureConfig(
    feature_types=[FeatureType.TECHNICAL],
    technical_indicators=["sma", "rsi", "macd"],
    enable_feature_selection=True,
    enable_standardization=True
)
engine = FeatureEngine(config)
```

#### 主要方法

##### process_features()

处理特征的主要方法。

```python
def process_features(self, data: pd.DataFrame, config: Optional[FeatureConfig] = None) -> pd.DataFrame:
    """
    处理特征
    
    Args:
        data: 输入数据，必须包含 'close', 'high', 'low', 'volume' 列
        config: 特征配置，如果为None则使用默认配置
        
    Returns:
        处理后的特征数据
        
    Raises:
        ValueError: 数据验证失败
        Exception: 特征处理失败
    """
```

**使用示例：**

```python
import pandas as pd
from src.features import FeatureEngine, DefaultConfigs

# 创建示例数据
data = pd.DataFrame({
    'close': [100, 101, 102, 103, 104],
    'high': [102, 103, 104, 105, 106],
    'low': [99, 100, 101, 102, 103],
    'volume': [1000000, 1100000, 1200000, 1300000, 1400000]
})

# 使用默认配置处理特征
engine = FeatureEngine()
config = DefaultConfigs.basic_technical()
features = engine.process_features(data, config)

print(f"处理后特征形状: {features.shape}")
print(f"特征列: {list(features.columns)}")
```

##### register_processor()

注册自定义处理器。

```python
def register_processor(self, name: str, processor: BaseFeatureProcessor) -> None:
    """
    注册特征处理器
    
    Args:
        name: 处理器名称
        processor: 处理器实例，必须继承自BaseFeatureProcessor
        
    Raises:
        ValueError: 处理器类型不正确
    """
```

**使用示例：**

```python
from src.features import BaseFeatureProcessor, ProcessorConfig

class CustomProcessor(BaseFeatureProcessor):
    def _compute_feature(self, data, feature_name, params):
        # 实现自定义特征计算
        if feature_name == "custom_ratio":
            return data['close'] / data['volume']
        return None

# 注册自定义处理器
custom_processor = CustomProcessor()
engine.register_processor("custom", custom_processor)
```

##### process_with_processor()

使用指定处理器处理特征。

```python
def process_with_processor(self, data: pd.DataFrame, processor_name: str, 
                         config: Optional[FeatureConfig] = None) -> pd.DataFrame:
    """
    使用指定处理器处理特征
    
    Args:
        data: 输入数据
        processor_name: 处理器名称
        config: 特征配置
        
    Returns:
        处理后的特征数据
        
    Raises:
        ValueError: 未找到指定处理器
    """
```

**使用示例：**

```python
# 使用技术指标处理器
technical_features = engine.process_with_processor(data, "technical", config)

# 使用情感分析处理器
sentiment_features = engine.process_with_processor(data, "sentiment", config)
```

##### 统计和监控方法

```python
# 获取处理统计信息
stats = engine.get_stats()
print(f"处理统计: {stats}")

# 重置统计信息
engine.reset_stats()

# 获取引擎信息
info = engine.get_engine_info()
print(f"引擎信息: {info}")

# 列出可用处理器
processors = engine.list_processors()
print(f"可用处理器: {processors}")

# 获取支持的特征列表
features = engine.get_supported_features()
print(f"支持的特征: {features}")
```

### FeatureConfig（配置管理）

配置管理类提供了统一的特征处理配置。

#### 基本配置

```python
from src.features.core.config import FeatureConfig, FeatureType

# 创建基本配置
config = FeatureConfig(
    feature_types=[FeatureType.TECHNICAL],
    technical_indicators=["sma", "rsi", "macd"],
    enable_feature_selection=True,
    enable_standardization=True,
    max_features=20,
    min_features=5
)
```

#### 高级配置

```python
from src.features.core.config import TechnicalParams, SentimentParams

# 技术指标参数配置
technical_params = TechnicalParams(
    sma_periods=[5, 10, 20],
    rsi_periods=[14],
    macd_params={"fast": 12, "slow": 26, "signal": 9}
)

# 情感分析参数配置
sentiment_params = SentimentParams(
    model_type="bert",
    batch_size=32,
    max_length=512
)

# 完整配置
config = FeatureConfig(
    feature_types=[FeatureType.TECHNICAL, FeatureType.SENTIMENT],
    technical_indicators=["sma", "rsi", "macd", "bbands"],
    technical_params=technical_params,
    sentiment_params=sentiment_params,
    enable_feature_selection=True,
    enable_standardization=True,
    standardization_method="zscore",
    max_features=30,
    min_features=10
)
```

#### 默认配置

```python
from src.features.core.config import DefaultConfigs

# 基础技术指标配置
basic_config = DefaultConfigs.basic_technical()

# 综合技术指标配置
comprehensive_config = DefaultConfigs.comprehensive_technical()

# 情感分析配置
sentiment_config = DefaultConfigs.sentiment_analysis()

# 完整配置
full_config = DefaultConfigs.full_features()
```

#### 配置验证

```python
# 验证配置
if config.validate():
    print("配置验证通过")
else:
    print("配置验证失败")

# 转换为字典
config_dict = config.to_dict()
print(f"配置字典: {config_dict}")
```

### FeatureEngineer（特征工程器）

特征工程器负责特征提取和工程。

#### 基本使用

```python
from src.features import FeatureEngineer

# 创建特征工程器
engineer = FeatureEngineer()

# 验证数据
engineer._validate_stock_data(data)

# 注册特征
from src.features.feature_config import FeatureConfig
config = FeatureConfig("test", FeatureType.TECHNICAL)
engineer.register_feature(config)
```

### SentimentAnalyzer（情感分析器）

情感分析器提供文本情感分析功能。

#### 基本使用

```python
from src.features import SentimentAnalyzer

# 创建情感分析器
analyzer = SentimentAnalyzer()

# 分析单条文本
text = "公司业绩表现优秀，市场前景看好"
sentiment_result = analyzer.analyze_text(text)
print(f"情感分析结果: {sentiment_result}")

# 分析批量文本
texts = [
    "公司发布利好消息，股价有望上涨",
    "市场对该公司前景持乐观态度",
    "分析师预测该公司业绩将超预期"
]
sentiment_results = analyzer.analyze_batch(texts)
print(f"批量情感分析结果: {sentiment_results}")
```

## 处理器API

### BaseFeatureProcessor（基础处理器）

所有处理器都继承自基础处理器，提供了标准化的接口。

#### 自定义处理器

```python
from src.features import BaseFeatureProcessor, ProcessorConfig

class CustomFeatureProcessor(BaseFeatureProcessor):
    def __init__(self, config: ProcessorConfig = None):
        super().__init__(config)
        self.supported_features = ["custom_ratio", "custom_momentum"]
    
    def _compute_feature(self, data, feature_name, params):
        """计算自定义特征"""
        if feature_name == "custom_ratio":
            return data['close'] / data['volume']
        elif feature_name == "custom_momentum":
            return data['close'].pct_change()
        return None
    
    def list_features(self):
        """列出支持的特征"""
        return self.supported_features
    
    def get_feature_info(self, feature_name):
        """获取特征信息"""
        return {
            "name": feature_name,
            "description": f"自定义特征: {feature_name}",
            "type": "numeric"
        }

# 使用自定义处理器
processor = CustomFeatureProcessor()
features = processor.process(data)
```

### FeatureSelector（特征选择器）

特征选择器提供特征选择和优化功能。

```python
from src.features import FeatureSelector

# 创建特征选择器
selector = FeatureSelector()

# 选择特征
selected_features = selector.select_features(
    features,
    target_column="target",
    method="rfecv",
    max_features=10
)

# 获取特征重要性
importance = selector.get_feature_importance()
print(f"特征重要性: {importance}")
```

### FeatureStandardizer（特征标准化器）

特征标准化器提供特征标准化功能。

```python
from src.features import FeatureStandardizer
from pathlib import Path

# 创建标准化器
model_path = Path("./models/features")
standardizer = FeatureStandardizer(model_path, method="standard")

# 拟合并转换特征
standardized_features = standardizer.fit_transform(features, is_training=True)

# 应用转换
transformed_features = standardizer.transform(features)

# 逆变换
original_features = standardizer.inverse_transform(transformed_features)

# 加载预训练标准化器
standardizer.load_scaler(model_path / "feature_scaler.pkl")
```

## 错误处理

### 异常类型

特征层定义了以下异常类型：

1. **ValueError**: 数据验证失败
2. **RuntimeError**: 运行时错误（如标准化器未拟合）
3. **FileNotFoundError**: 文件未找到
4. **Exception**: 其他通用错误

### 错误处理最佳实践

```python
from src.features import FeatureEngine, DefaultConfigs

def safe_feature_processing(data, config=None):
    """安全的特征处理函数"""
    engine = FeatureEngine()
    
    try:
        # 数据验证
        if not engine.validate_data(data):
            raise ValueError("输入数据验证失败")
        
        # 处理特征
        features = engine.process_features(data, config)
        return features
        
    except ValueError as e:
        print(f"数据验证错误: {e}")
        # 记录错误日志
        logger.error(f"数据验证失败: {e}")
        return None
        
    except RuntimeError as e:
        print(f"运行时错误: {e}")
        # 记录错误日志
        logger.error(f"运行时错误: {e}")
        return None
        
    except Exception as e:
        print(f"未知错误: {e}")
        # 记录错误日志
        logger.error(f"特征处理失败: {e}")
        return None

# 使用安全处理函数
result = safe_feature_processing(data)
if result is not None:
    print(f"处理成功，特征形状: {result.shape}")
else:
    print("处理失败")
```

### 输入验证

```python
def validate_input_data(data):
    """验证输入数据"""
    if data is None or data.empty:
        raise ValueError("输入数据为空")
    
    required_columns = ['close', 'high', 'low', 'volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        raise ValueError(f"缺失必要列: {missing_columns}")
    
    # 检查数据类型
    for col in required_columns:
        if not pd.api.types.is_numeric_dtype(data[col]):
            raise ValueError(f"列 {col} 不是数值类型")
    
    # 检查缺失值
    if data[required_columns].isnull().any().any():
        raise ValueError("数据包含缺失值")
    
    return True
```

## 性能优化

### 缓存机制

```python
from functools import lru_cache

class CachedFeatureEngine(FeatureEngine):
    def __init__(self, config=None):
        super().__init__(config)
        self._cache = {}
    
    @lru_cache(maxsize=128)
    def _compute_technical_features(self, data_hash, config_hash):
        """缓存技术指标计算"""
        # 实现缓存逻辑
        pass
```

### 并行处理

```python
import concurrent.futures
from src.features import FeatureEngine

def parallel_feature_processing(data_list, config):
    """并行处理多个数据集"""
    engine = FeatureEngine()
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(engine.process_features, data, config)
            for data in data_list
        ]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"处理失败: {e}")
    
    return results
```

### 内存优化

```python
def memory_efficient_processing(data, config, chunk_size=1000):
    """内存高效的特征处理"""
    engine = FeatureEngine()
    results = []
    
    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i+chunk_size]
        result = engine.process_features(chunk, config)
        results.append(result)
        
        # 清理内存
        del chunk
    
    return pd.concat(results, ignore_index=True)
```

## 最佳实践

### 1. 使用特征引擎（推荐）

```python
# 推荐：使用特征引擎
from src.features import FeatureEngine, DefaultConfigs

engine = FeatureEngine()
config = DefaultConfigs.basic_technical()
features = engine.process_features(data, config)
```

### 2. 配置管理

```python
# 使用默认配置
config = DefaultConfigs.basic_technical()

# 自定义配置
config = FeatureConfig(
    feature_types=[FeatureType.TECHNICAL],
    technical_indicators=["sma", "rsi", "macd"],
    enable_feature_selection=True,
    enable_standardization=True
)
```

### 3. 错误处理

```python
try:
    features = engine.process_features(data, config)
except ValueError as e:
    logger.error(f"数据验证失败: {e}")
except Exception as e:
    logger.error(f"特征处理失败: {e}")
```

### 4. 性能监控

```python
# 获取处理统计
stats = engine.get_stats()
print(f"处理时间: {stats['processing_time']:.2f}秒")
print(f"处理特征数: {stats['processed_features']}")
print(f"错误数: {stats['errors']}")
```

### 5. 自定义扩展

```python
# 自定义处理器
class CustomProcessor(BaseFeatureProcessor):
    def _compute_feature(self, data, feature_name, params):
        # 实现自定义逻辑
        pass

# 注册自定义处理器
engine.register_processor("custom", CustomProcessor())
```

## 总结

特征层API提供了完整、灵活且易于使用的特征处理功能。通过遵循最佳实践，可以构建高效、可靠的特征处理流程，为RQA系统提供高质量的特征数据服务。 