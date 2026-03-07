# RQA2025 特征层设计文档

## 1. 概述

特征层负责将原始数据转换为模型可用的特征，主要功能包括：
- 技术指标计算
- 情感特征提取
- 特征标准化
- 特征选择

## 2. 核心组件

### 2.1 特征处理器
```text
TechnicalProcessor    - 技术指标计算
SentimentAnalyzer     - 情感特征提取
FeatureStandardizer   - 特征标准化
FeatureSelector       - 特征选择
```

### 2.2 特征引擎(FeatureEngineer)
```text
功能：
1. 统一特征计算接口
2. 特征缓存管理
3. 批量特征计算
4. 特征质量控制
```

## 3. 特征计算流程

### 3.1 基本计算流程
```python
# 初始化特征引擎
engineer = FeatureEngineer()

# 计算技术指标特征
tech_features = engineer.calculate_technical_features(
    symbol="600000",
    price_data=price_df,
    indicators=["SMA", "RSI"]
)

# 计算情感特征
sentiment_features = engineer.calculate_sentiment_features(
    text_data=news_df,
    models=["BERT"]
)
```

### 3.2 缓存机制
```python
# 自动缓存技术指标(默认24小时)
features = engineer.calculate_technical_features(..., use_cache=True)

# 自定义缓存时间(小时)
engineer._save_to_cache(key, features, ttl=12)

# 禁用缓存
features = engineer.calculate_technical_features(..., use_cache=False)
```

**缓存策略**:
- 技术指标: 24小时
- 情感特征: 12小时
- 标准化特征: 不缓存(依赖输入)

### 3.3 批量计算
```python
# 批量计算技术指标
symbols = ["600000", "000001", "601318"]
price_data = {
    "600000": price_df1,
    "000001": price_df2,
    "601318": price_df3
}
results = engineer.batch_calculate_technical_features(
    symbols, price_data, ["SMA", "RSI"]
)

# 批量计算情感特征
text_data = {
    "600000": news_df1,
    "000001": news_df2
}
results = {
    sym: engineer.calculate_sentiment_features(df, ["BERT"])
    for sym, df in text_data.items()
}
```

### 3.4 缓存管理
```python
# 清理过期缓存(默认保留7天)
engineer.clear_cache(older_than_days=7)

# 手动删除特定缓存
engineer._save_to_cache(key, data, ttl=0)  # 立即过期
```

## 4. 性能优化建议

1. **高频特征**设置较长TTL
2. **批量操作**使用batch方法
3. **内存管理**定期清理缓存
4. **并行计算**合理设置线程数

## 5. 最佳实践

```python
# 示例：组合特征生成流程
def generate_features(symbol, price_data, news_data):
    # 技术指标
    tech_features = engineer.calculate_technical_features(
        symbol, price_data, ["SMA", "RSI", "MACD"]
    )
    
    # 情感特征
    if not news_data.empty:
        sentiment_features = engineer.calculate_sentiment_features(
            news_data, models=["BERT", "TextBlob"]
        )
        features = pd.concat([tech_features, sentiment_features], axis=1)
    else:
        features = tech_features
    
    # 标准化
    features = FeatureStandardizer().transform(features)
    
    return features
```

## 5. 新增功能 (v1.3)

### 5.1 并行特征处理
```python
from src.features.parallel import ParallelFeatureProcessor

processor = ParallelFeatureProcessor(n_workers=4)
features = processor.batch_process(feature_tasks)
```

### 5.2 特征存储系统
```python
from src.features.store import FeatureStore

store = FeatureStore()
store.save_features("dataset1", features, metadata={...})
retrieved = store.load_features("dataset1")
```

### 5.3 质量评估报告
```python
from src.features.evaluation import FeatureQualityReport

report = FeatureQualityReport(features, target)
html_report = report.generate_html()
```

## 6. 版本历史

- v1.0 (2023-06-15): 初始版本
- v1.1 (2023-07-20): 添加特征缓存
- v1.2 (2023-08-10): 增加批量计算功能
- v1.3 (2023-09-01): 新增并行处理、特征存储和质量评估
