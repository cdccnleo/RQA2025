# 特征层 (Features Layer)

## 概述

特征层是RQA2025项目的核心组件，负责特征工程、技术指标计算、情感分析、特征选择、特征存储等功能。该层提供了完整的特征处理流水线，支持多种特征类型和配置管理。

## 主要组件

### 1. 特征工程 (Feature Engineering)
- **FeatureEngineer**: 特征工程主处理器
- **TechnicalProcessor**: 技术指标处理器
- **SentimentAnalyzer**: 情感分析器
- **FeatureSelector**: 特征选择器
- **FeatureStore**: 特征存储管理器

### 2. 配置管理集成 (Configuration Management Integration)

#### 配置作用域 (ConfigScope)
特征层支持多作用域配置管理：

- **GLOBAL**: 全局配置
- **TECHNICAL**: 技术指标配置
- **SENTIMENT**: 情感分析配置
- **PROCESSING**: 处理配置
- **MONITORING**: 监控配置

#### 配置热更新
所有主要组件都支持配置热更新，无需重启即可生效：

```python
from src.features.config_integration import get_config_integration_manager, ConfigScope

# 获取配置管理器
config_manager = get_config_integration_manager()

# 设置配置
config_manager.set_config(ConfigScope.TECHNICAL, "rsi_period", 16)

# 通知配置变更
config_manager.notify_config_change(ConfigScope.TECHNICAL, "rsi_period", 14, 16)
```

#### 配置管理方式
提供多种配置管理方式：

```bash
# 通过API管理配置
curl -X GET http://localhost:8000/api/v1/features/config

# 通过配置文件管理
# 编辑 config/features/ 目录下的配置文件

# 通过程序化接口管理
python -c "from src.features.config_integration import get_config_integration_manager; print(get_config_integration_manager().get_config('technical'))"
```

### 3. 技术指标 (Technical Indicators)

#### 支持的指标
- **RSI (相对强弱指数)**
- **MACD (移动平均收敛发散)**
- **Bollinger Bands (布林带)**
- **ATR (平均真实波幅)**
- **移动平均线**
- **成交量指标**

#### 配置参数
```python
# 技术指标配置示例
technical_config = {
    "rsi_period": 14,
    "rsi_overbought": 70.0,
    "rsi_oversold": 30.0,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bollinger_period": 20,
    "bollinger_std": 2.0,
    "atr_period": 14
}
```

### 4. 情感分析 (Sentiment Analysis)

#### 功能特性
- 文本清理和预处理
- 多语言支持
- BERT模型集成
- 情感极性分析
- 批量处理支持

#### 配置参数
```python
# 情感分析配置示例
sentiment_config = {
    "use_bert": True,
    "bert_model_path": "models/bert-base-chinese",
    "default_language": "zh",
    "batch_size": 32
}
```

### 5. 特征选择 (Feature Selection)

#### 选择策略
- **RFECV**: 递归特征消除与交叉验证
- **SelectKBest**: K最佳特征选择
- **自定义策略**: 支持自定义特征选择函数

#### 配置参数
```python
# 特征选择配置示例
selection_config = {
    "selector_type": "rfecv",
    "n_features": 20,
    "min_features_to_select": 5,
    "cv": 5,
    "threshold": 0.9
}
```

### 6. 特征存储 (Feature Storage)

#### 存储特性
- 文件系统存储
- 压缩支持
- TTL (生存时间) 管理
- 元数据管理
- 缓存机制

#### 配置参数
```python
# 特征存储配置示例
store_config = {
    "base_path": "./feature_cache",
    "max_size_mb": 1024,
    "ttl_hours": 24,
    "compression": "gzip",
    "use_filesystem": True,
    "max_workers": 4
}
```

## 使用示例

### 基本使用
```python
from src.features.feature_manager import FeatureManager
from src.features.config_integration import get_config_integration_manager, ConfigScope

# 获取配置管理器
config_manager = get_config_integration_manager()

# 创建特征管理器
feature_manager = FeatureManager()

# 处理特征
features = feature_manager.process_features(data)
```

### 配置热更新示例
```python
# 动态更新技术指标参数
config_manager.set_config(ConfigScope.TECHNICAL, "rsi_period", 16)
config_manager.notify_config_change(ConfigScope.TECHNICAL, "rsi_period", 14, 16)

# 动态更新处理参数
config_manager.set_config(ConfigScope.PROCESSING, "max_workers", 8)
config_manager.notify_config_change(ConfigScope.PROCESSING, "max_workers", 4, 8)
```

### 配置管理使用
```python
from src.features.config_integration import get_config_integration_manager

# 获取配置管理器
config_manager = get_config_integration_manager()
# 获取技术指标配置
tech_config = config_manager.get_config('technical')
print(f"RSI周期: {tech_config.get('rsi_period', 14)}")
```

## 测试

### 运行测试
```bash
# 运行特征层测试
python scripts/testing/run_tests.py --module features

# 运行配置集成测试
python scripts/testing/run_tests.py --module features --pytest-args -k config
```

### 配置集成测试
```python
# 测试配置热更新
def test_config_hot_update():
    config_manager = get_config_integration_manager()
    
    # 测试技术指标配置更新
    old_value = config_manager.get_config(ConfigScope.TECHNICAL, "rsi_period")
    config_manager.set_config(ConfigScope.TECHNICAL, "rsi_period", old_value + 2)
    config_manager.notify_config_change(ConfigScope.TECHNICAL, "rsi_period", old_value, old_value + 2)
    
    assert config_manager.get_config(ConfigScope.TECHNICAL, "rsi_period") == old_value + 2
```

## 性能优化

### 缓存机制
- 特征计算结果缓存
- 技术指标缓存
- 情感分析结果缓存
- 配置变更缓存

### 并行处理
- 多线程特征计算
- 批量数据处理
- 异步配置更新

### 内存管理
- 特征数据分块处理
- 缓存大小限制
- 自动垃圾回收

## 监控和日志

### 性能监控
- 特征计算时间
- 内存使用情况
- 缓存命中率
- 配置变更频率

### 日志记录
- 配置变更日志
- 错误和异常日志
- 性能指标日志
- 调试信息日志

## 最佳实践

### 配置管理
1. 使用API或配置文件进行配置管理
2. 定期备份配置历史记录
3. 在测试环境中验证配置变更
4. 使用配置热更新避免服务重启

### 性能优化
1. 根据数据量调整工作线程数
2. 合理设置缓存大小和TTL
3. 使用批量处理提高效率
4. 监控内存使用情况

### 错误处理
1. 实现配置验证机制
2. 提供配置回滚功能
3. 记录详细的错误日志
4. 实现优雅的错误恢复

## 故障排除

### 常见问题
1. **配置热更新不生效**: 检查组件是否正确注册了配置监听器
2. **内存使用过高**: 调整缓存大小和TTL设置
3. **特征计算速度慢**: 增加工作线程数或使用缓存
4. **配置无法加载**: 检查配置文件路径和权限

### 调试技巧
1. 启用详细日志记录
2. 使用API或配置文件查看当前配置
3. 检查配置历史记录
4. 运行单元测试验证功能

## 更新日志

### v4.0 (2025-01-27)
- ✅ 完成配置管理集成
- ✅ 实现配置热更新功能
- ✅ 实现API配置管理
- ✅ 建立多作用域配置管理
- ✅ 补充配置集成测试

### v3.0 (2025-01-26)
- ✅ 完成性能监控集成
- ✅ 建立监控数据持久化系统
- ✅ 建立监控数据API

### v2.0 (2025-01-25)
- ✅ 完成技术债务修复
- ✅ 优化代码结构和性能
- ✅ 完善错误处理机制

### v1.0 (2025-01-24)
- ✅ 基础特征工程功能
- ✅ 技术指标计算
- ✅ 情感分析功能
- ✅ 特征选择和存储