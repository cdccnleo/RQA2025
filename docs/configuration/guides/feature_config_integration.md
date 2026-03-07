# 特征层配置集成使用指南

## 概述

特征层配置集成提供了统一的配置管理功能，支持多作用域配置、热更新、可视化界面等特性。本指南详细介绍如何使用特征层的配置管理功能。

## 配置作用域

### 支持的作用域

特征层支持以下配置作用域：

```python
from src.features.config_integration import ConfigScope

# 全局配置
ConfigScope.GLOBAL      # 全局配置，影响所有组件

# 技术指标配置
ConfigScope.TECHNICAL   # 技术指标相关配置

# 情感分析配置
ConfigScope.SENTIMENT   # 情感分析相关配置

# 处理配置
ConfigScope.PROCESSING  # 特征处理相关配置

# 监控配置
ConfigScope.MONITORING  # 监控相关配置
```

### 配置项说明

#### 全局配置 (GLOBAL)
```python
global_config = {
    "cache_dir": "./feature_cache",      # 缓存目录
    "fallback_enabled": True,            # 启用回退机制
    "log_level": "INFO",                 # 日志级别
    "debug_mode": False                  # 调试模式
}
```

#### 技术指标配置 (TECHNICAL)
```python
technical_config = {
    "rsi_period": 14,                   # RSI周期
    "rsi_overbought": 70.0,             # RSI超买阈值
    "rsi_oversold": 30.0,               # RSI超卖阈值
    "macd_fast": 12,                    # MACD快线周期
    "macd_slow": 26,                    # MACD慢线周期
    "macd_signal": 9,                   # MACD信号线周期
    "bollinger_period": 20,             # 布林带周期
    "bollinger_std": 2.0,               # 布林带标准差
    "atr_period": 14                    # ATR周期
}
```

#### 情感分析配置 (SENTIMENT)
```python
sentiment_config = {
    "use_bert": True,                   # 使用BERT模型
    "bert_model_path": "models/bert",   # BERT模型路径
    "default_language": "zh",           # 默认语言
    "batch_size": 32,                   # 批处理大小
    "confidence_threshold": 0.8         # 置信度阈值
}
```

#### 处理配置 (PROCESSING)
```python
processing_config = {
    "max_workers": 4,                   # 最大工作线程数
    "batch_size": 1000,                 # 批处理大小
    "timeout": 300,                     # 超时时间(秒)
    "feature_selection_method": "rfecv", # 特征选择方法
    "max_features": 50,                 # 最大特征数
    "min_feature_importance": 0.01,     # 最小特征重要性
    "standardization_method": "zscore", # 标准化方法
    "robust_scaling": True,             # 使用鲁棒缩放
    "selector_type": "rfecv",           # 选择器类型
    "n_features": 20,                   # 选择特征数
    "min_features_to_select": 5,        # 最小选择特征数
    "cv": 5,                           # 交叉验证折数
    "threshold": 0.9,                   # 阈值
    "base_path": "./feature_cache",     # 存储基础路径
    "max_size_mb": 1024,               # 最大存储大小(MB)
    "ttl_hours": 24,                   # 生存时间(小时)
    "compression": "gzip",              # 压缩方式
    "use_filesystem": True              # 使用文件系统
}
```

#### 监控配置 (MONITORING)
```python
monitoring_config = {
    "enable_monitoring": True,          # 启用监控
    "monitoring_level": "standard",     # 监控级别
    "metrics_interval": 60,             # 指标收集间隔(秒)
    "alert_threshold": 0.8,             # 告警阈值
    "log_performance": True             # 记录性能日志
}
```

## 基本使用

### 1. 获取配置管理器

```python
from src.features.config_integration import get_config_integration_manager, ConfigScope

# 获取配置管理器实例
config_manager = get_config_integration_manager()
```

### 2. 获取配置

```python
# 获取整个作用域的配置
technical_config = config_manager.get_config(ConfigScope.TECHNICAL)

# 获取特定配置项
rsi_period = config_manager.get_config(ConfigScope.TECHNICAL, "rsi_period")
```

### 3. 设置配置

```python
# 设置单个配置项
config_manager.set_config(ConfigScope.TECHNICAL, "rsi_period", 16)

# 设置多个配置项
config_manager.set_config(ConfigScope.TECHNICAL, {
    "rsi_period": 16,
    "macd_fast": 10,
    "bollinger_std": 2.5
})
```

### 4. 配置变更通知

```python
# 通知配置变更（触发热更新）
config_manager.notify_config_change(
    ConfigScope.TECHNICAL, 
    "rsi_period", 
    14,  # 旧值
    16   # 新值
)
```

## 配置热更新

### 组件自动响应

所有主要组件都支持配置热更新，无需重启即可生效：

```python
# 技术指标处理器会自动响应配置变更
from src.features.technical.technical_processor import TechnicalProcessor

processor = TechnicalProcessor()

# 变更RSI周期，处理器会自动更新
config_manager.set_config(ConfigScope.TECHNICAL, "rsi_period", 16)
config_manager.notify_config_change(ConfigScope.TECHNICAL, "rsi_period", 14, 16)

# 处理器内部会自动更新RSI计算参数
```

### 线程池热更新

```python
# 特征工程师的线程池会自动响应max_workers变更
from src.features.feature_engineer import FeatureEngineer

engineer = FeatureEngineer()

# 变更最大工作线程数
config_manager.set_config(ConfigScope.PROCESSING, "max_workers", 8)
config_manager.notify_config_change(ConfigScope.PROCESSING, "max_workers", 4, 8)

# 特征工程师会自动重新创建线程池
```

## 配置管理界面

### 启动界面

```bash
# 启动配置管理界面
python scripts/features/start_config_interface.py
```

### 界面功能

1. **配置编辑**: 可视化编辑各作用域配置
2. **实时预览**: 实时查看当前配置状态
3. **历史记录**: 查看配置变更历史
4. **配置验证**: 验证配置有效性
5. **批量操作**: 支持批量配置变更

### 界面操作

1. **选择作用域**: 从下拉菜单选择配置作用域
2. **编辑配置**: 双击配置项进行编辑
3. **添加配置**: 点击"添加"按钮新增配置项
4. **删除配置**: 选择配置项后点击"删除"
5. **应用配置**: 点击"应用"按钮保存变更

## 高级功能

### 1. 配置验证

```python
# 验证配置有效性
def validate_config(scope: ConfigScope, config: dict) -> bool:
    try:
        if scope == ConfigScope.TECHNICAL:
            # 验证技术指标配置
            if config.get("rsi_period", 0) <= 0:
                return False
            if config.get("macd_fast", 0) >= config.get("macd_slow", 0):
                return False
        return True
    except Exception:
        return False

# 使用验证
if validate_config(ConfigScope.TECHNICAL, technical_config):
    config_manager.set_config(ConfigScope.TECHNICAL, technical_config)
```

### 2. 配置回滚

```python
# 保存配置快照
def save_config_snapshot():
    snapshot = {}
    for scope in ConfigScope:
        snapshot[scope.value] = config_manager.get_config(scope)
    return snapshot

# 回滚到快照
def rollback_to_snapshot(snapshot):
    for scope_name, config in snapshot.items():
        scope = ConfigScope(scope_name)
        for key, value in config.items():
            config_manager.set_config(scope, key, value)
            config_manager.notify_config_change(scope, key, None, value)
```

### 3. 配置监控

```python
# 监控配置变更
def monitor_config_changes():
    def on_config_change(scope: ConfigScope, key: str, old_value: Any, new_value: Any):
        print(f"配置变更: {scope.value}.{key} = {old_value} -> {new_value}")
    
    # 注册监控回调
    for scope in ConfigScope:
        config_manager.register_config_watcher(scope, on_config_change)
```

## 最佳实践

### 1. 配置管理

- **使用配置管理界面**: 优先使用可视化界面进行配置管理
- **定期备份**: 定期备份重要配置
- **版本控制**: 对配置文件进行版本控制
- **环境隔离**: 为不同环境使用不同的配置

### 2. 热更新使用

- **渐进式更新**: 逐步更新配置，避免一次性大量变更
- **验证变更**: 每次配置变更后验证功能正常
- **监控影响**: 监控配置变更对性能的影响
- **回滚准备**: 准备快速回滚机制

### 3. 性能优化

- **合理设置**: 根据系统资源合理设置配置参数
- **缓存利用**: 充分利用缓存机制
- **批量处理**: 使用批量处理提高效率
- **监控指标**: 监控关键性能指标

### 4. 错误处理

- **异常捕获**: 捕获配置相关的异常
- **默认值**: 为关键配置提供合理的默认值
- **日志记录**: 详细记录配置变更日志
- **错误恢复**: 实现优雅的错误恢复机制

## 故障排除

### 常见问题

1. **配置热更新不生效**
   - 检查组件是否正确注册了配置监听器
   - 确认配置变更通知是否正确发送
   - 验证配置作用域是否正确

2. **配置界面无法启动**
   - 检查tkinter依赖是否正确安装
   - 确认Python环境配置
   - 检查项目路径设置

3. **配置验证失败**
   - 检查配置项的数据类型
   - 验证配置值的范围
   - 确认配置项的名称正确

4. **性能问题**
   - 调整工作线程数
   - 优化缓存设置
   - 检查内存使用情况

### 调试技巧

1. **启用详细日志**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **使用配置摘要**
   ```python
   summary = config_manager.get_config_summary()
   print(json.dumps(summary, indent=2))
   ```

3. **检查配置历史**
   ```python
   # 查看配置变更历史
   history = config_manager.get_config_history()
   for entry in history:
       print(f"{entry['timestamp']}: {entry['scope']}.{entry['key']}")
   ```

## 示例代码

### 完整示例

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征层配置集成完整示例
"""

from src.features.config_integration import get_config_integration_manager, ConfigScope
from src.features.feature_engineer import FeatureEngineer
from src.features.technical.technical_processor import TechnicalProcessor
from src.features.sentiment_analyzer import SentimentAnalyzer

def main():
    """主函数"""
    # 获取配置管理器
    config_manager = get_config_integration_manager()
    
    # 初始化组件
    engineer = FeatureEngineer()
    processor = TechnicalProcessor()
    analyzer = SentimentAnalyzer()
    
    # 设置初始配置
    config_manager.set_config(ConfigScope.TECHNICAL, {
        "rsi_period": 14,
        "macd_fast": 12,
        "macd_slow": 26
    })
    
    # 模拟配置热更新
    print("原始RSI周期:", processor.rsi_period)
    
    # 更新RSI周期
    config_manager.set_config(ConfigScope.TECHNICAL, "rsi_period", 16)
    config_manager.notify_config_change(ConfigScope.TECHNICAL, "rsi_period", 14, 16)
    
    print("更新后RSI周期:", processor.rsi_period)
    
    # 更新处理配置
    config_manager.set_config(ConfigScope.PROCESSING, "max_workers", 8)
    config_manager.notify_config_change(ConfigScope.PROCESSING, "max_workers", 4, 8)
    
    print("更新后最大工作线程:", engineer.max_workers)

if __name__ == "__main__":
    main()
```

## 总结

特征层配置集成提供了强大的配置管理功能，支持多作用域配置、热更新、可视化界面等特性。通过合理使用这些功能，可以大大提高系统的灵活性和可维护性。

关键要点：
- 使用配置管理界面进行可视化配置
- 充分利用配置热更新功能
- 遵循最佳实践进行配置管理
- 建立完善的监控和错误处理机制 