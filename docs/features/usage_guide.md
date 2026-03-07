# 特征层使用指南

## 概述

本指南介绍如何使用RQA2025项目的特征层功能。

## 快速开始

```python
from src.features import FeatureEngine, FeatureConfig

# 创建特征引擎
engine = FeatureEngine()

# 准备数据
import pandas as pd
import numpy as np

data = pd.DataFrame({
    'close': np.random.randn(100).cumsum() + 100,
    'high': np.random.randn(100).cumsum() + 105,
    'low': np.random.randn(100).cumsum() + 95,
    'volume': np.random.randint(1000, 10000, 100)
})

# 处理特征
result = engine.process_features(data)
print(f"处理完成，生成了 {len(result.columns)} 个特征")
```

## 核心组件

### 1. 特征引擎 (FeatureEngine)

特征引擎是特征层的核心组件，负责协调各个处理器。

#### 主要方法

- `process_features(data, config)`: 处理特征
- `register_processor(name, processor)`: 注册处理器
- `get_processor(name)`: 获取处理器
- `list_processors()`: 列出所有处理器
- `validate_data(data)`: 验证数据

### 2. 特征配置 (FeatureConfig)

特征配置用于控制特征处理的行为。

```python
from src.features import FeatureConfig, FeatureType

config = FeatureConfig(
    feature_types=[FeatureType.TECHNICAL],
    enable_feature_selection=True,
    enable_standardization=True,
    max_features=20
)
```

### 3. 处理器使用

#### 技术指标处理器

```python
from src.features.processors.technical import TechnicalProcessor

processor = TechnicalProcessor()
sma_result = processor.calculate_indicator(data, 'sma', {'period': 20, 'column': 'close'})
```

#### 情感分析器

```python
from src.features.sentiment import SentimentAnalyzer

analyzer = SentimentAnalyzer()
sentiment_result = analyzer.analyze_news_sentiment(news_data)
```

## 性能优化

### 1. 分布式处理

```python
from src.features.processors.distributed import DistributedFeatureProcessor

dist_processor = DistributedFeatureProcessor(max_workers=4)
result = dist_processor.process_in_parallel(data, processor, config)
```

### 2. 内存优化

```python
from src.features.processors.distributed import MemoryOptimizedProcessor

mem_processor = MemoryOptimizedProcessor(max_memory_mb=1024)
result = mem_processor.process_with_memory_optimization(data, processor, config)
```

### 3. 缓存处理

```python
from src.features.processors.distributed import CachingProcessor

cache_processor = CachingProcessor(cache_dir='./cache')
result = cache_processor.process_with_cache(data, processor, config)
```

## 监控和调试

```python
# 获取统计信息
stats = engine.get_stats()
print(f"处理特征数: {stats['processed_features']}")
print(f"处理时间: {stats['processing_time']:.2f}秒")

# 获取引擎信息
info = engine.get_engine_info()
print(f"引擎版本: {info['version']}")
print(f"注册处理器: {info['processors']}")
```

## 错误处理

```python
# 验证数据
if not engine.validate_data(data):
    print("数据验证失败")
    return

# 处理特征
try:
    result = engine.process_features(data)
except Exception as e:
    print(f"特征处理失败: {e}")
```

## 扩展开发

### 创建自定义处理器

```python
from src.features.processors.base_processor import BaseFeatureProcessor, ProcessorConfig

class CustomProcessor(BaseFeatureProcessor):
    def __init__(self):
        config = ProcessorConfig(processor_type='custom', feature_params={})
        super().__init__(config)
    
    def _compute_feature(self, data, feature_name, params):
        # 实现自定义特征计算
        return data['close'] * 2
    
    def _get_feature_metadata(self, feature_name):
        return {'name': feature_name, 'type': 'custom'}
    
    def _get_available_features(self):
        return ['custom_feature']
```

## 最佳实践

1. **数据验证**: 在处理前验证数据完整性
2. **错误处理**: 使用try-catch处理异常
3. **性能监控**: 定期检查处理统计
4. **缓存使用**: 对重复计算使用缓存
5. **内存管理**: 大数据集使用内存优化
6. **日志记录**: 记录关键操作和错误

## 常见问题

### 1. 内存不足
使用内存优化处理器处理大数据集

### 2. 处理速度慢
使用分布式处理器并行处理

### 3. 缓存命中率低
优化缓存配置和键生成策略 