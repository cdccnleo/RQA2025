# 特征层快速开始指南

## 概述
本指南将帮助您快速上手RQA2025项目的特征层功能，包括环境准备、基础使用和常见配置。

## 环境准备

### 1. 系统要求
- Python 3.8+
- 内存: 4GB+ (推荐8GB+)
- 存储: 2GB+ 可用空间

### 2. 依赖安装
```bash
# 激活conda环境
conda activate test

# 安装基础依赖
conda install pandas numpy scipy

# 安装可选依赖（用于性能优化）
pip install numba

# 安装监控依赖
pip install psutil
```

### 3. 验证安装
```python
# 验证基础功能
from src.features.feature_engineer import FeatureEngineer
from src.features.technical.technical_processor import TechnicalProcessor
from src.features.monitoring import FeaturesMonitor

print("特征层模块导入成功！")
```

## 基础使用

### 1. 特征工程器使用

#### 创建特征工程器
```python
from src.features.feature_engineer import FeatureEngineer
import pandas as pd

# 创建特征工程器
engineer = FeatureEngineer()

# 准备示例数据
data = pd.DataFrame({
    'open': [100, 101, 102, 103, 104],
    'high': [102, 103, 104, 105, 106],
    'low': [99, 100, 101, 102, 103],
    'close': [101, 102, 103, 104, 105],
    'volume': [1000, 1100, 1200, 1300, 1400]
}, index=pd.date_range('2023-01-01', periods=5))

print("数据准备完成")
```

#### 生成技术指标
```python
# 生成基础技术指标
features = engineer.generate_technical_features(
    stock_data=data,
    indicators=["ma", "rsi"],
    params={
        "ma": {"window": [5, 10]},
        "rsi": {"window": 14}
    }
)

print("特征生成完成")
print(f"生成特征数量: {len(features.columns)}")
print("特征列:", features.columns.tolist())
```

### 2. 技术指标处理器使用

#### 创建处理器
```python
from src.features.technical.technical_processor import TechnicalProcessor

# 创建技术指标处理器
processor = TechnicalProcessor()

# 计算单个指标
ma5 = processor.calculate_ma(data, window=5)
rsi = processor.calculate_rsi(data, window=14)

print("技术指标计算完成")
```

#### 批量计算指标
```python
# 批量计算多个指标
indicators = ["ma", "rsi", "macd", "bollinger"]
params = {
    "ma": {"window": [5, 10, 20]},
    "rsi": {"window": 14},
    "macd": {"fast_window": 12, "slow_window": 26, "signal_window": 9},
    "bollinger": {"window": 20, "num_std": 2}
}

result = processor.calculate_indicators(data, indicators, params)
print("批量计算完成")
print(f"结果形状: {result.shape}")
```

### 3. 监控系统使用

#### 设置监控
```python
from src.features.monitoring import FeaturesMonitor, MetricType

# 创建监控器
monitor = FeaturesMonitor()

# 注册组件
monitor.register_component("feature_engineer", "processor")
monitor.register_component("technical_processor", "processor")

# 启动监控
monitor.start_monitoring()
print("监控系统已启动")
```

#### 收集指标
```python
# 收集性能指标
monitor.collect_metrics(
    "feature_engineer",
    "processing_time",
    1.23,
    MetricType.HISTOGRAM
)

# 获取性能报告
report = monitor.get_performance_report()
print("性能报告:", report)
```

## 常见配置

### 1. 数据验证配置
```python
# 配置数据验证
config = {
    "validation": {
        "strict_mode": False,      # 非严格模式
        "auto_repair": True,       # 自动修复
        "required_columns": ["open", "high", "low", "close", "volume"]
    }
}

engineer = FeatureEngineer(config=config)
```

### 2. 缓存配置
```python
# 配置缓存
cache_config = {
    "enable": True,           # 启用缓存
    "ttl": 3600,             # 缓存时间1小时
    "max_size": 1000         # 最大缓存条目数
}

engineer.set_config({"caching": cache_config})
```

### 3. 监控配置
```python
# 配置监控
monitor_config = {
    "monitor_interval": 5.0,        # 监控间隔5秒
    "thresholds": {
        "cpu_usage": 80.0,          # CPU使用率阈值
        "memory_usage": 80.0,       # 内存使用率阈值
        "error_rate": 5.0           # 错误率阈值
    }
}

monitor = FeaturesMonitor(monitor_config)
```

## 完整示例

### 基础特征工程流程
```python
import pandas as pd
from src.features.feature_engineer import FeatureEngineer
from src.features.monitoring import FeaturesMonitor

def basic_feature_engineering_example():
    """基础特征工程示例"""
    
    # 1. 准备数据
    data = pd.DataFrame({
        'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'high': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
        'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
        'close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
    }, index=pd.date_range('2023-01-01', periods=10))
    
    # 2. 创建特征工程器
    engineer = FeatureEngineer()
    
    # 3. 生成特征
    features = engineer.generate_technical_features(
        stock_data=data,
        indicators=["ma", "rsi", "macd"],
        params={
            "ma": {"window": [5, 10]},
            "rsi": {"window": 14},
            "macd": {"fast": 12, "slow": 26, "signal": 9}
        }
    )
    
    # 4. 查看结果
    print("特征工程完成")
    print(f"输入数据形状: {data.shape}")
    print(f"输出特征形状: {features.shape}")
    print("特征列:", features.columns.tolist())
    
    return features

# 运行示例
if __name__ == "__main__":
    features = basic_feature_engineering_example()
```

### 带监控的特征工程流程
```python
def monitored_feature_engineering_example():
    """带监控的特征工程示例"""
    
    # 1. 设置监控
    monitor = FeaturesMonitor()
    monitor.register_component("feature_engineer", "processor")
    monitor.start_monitoring()
    
    # 2. 准备数据
    data = pd.DataFrame({
        'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'high': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
        'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
        'close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
    }, index=pd.date_range('2023-01-01', periods=10))
    
    # 3. 创建特征工程器
    engineer = FeatureEngineer()
    
    # 4. 执行特征工程（带监控）
    import time
    start_time = time.time()
    
    features = engineer.generate_technical_features(
        stock_data=data,
        indicators=["ma", "rsi", "macd", "bollinger"],
        params={
            "ma": {"window": [5, 10, 20]},
            "rsi": {"window": 14},
            "macd": {"fast": 12, "slow": 26, "signal": 9},
            "bollinger": {"window": 20, "num_std": 2}
        }
    )
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # 5. 收集监控指标
    monitor.collect_metrics(
        "feature_engineer",
        "processing_time",
        processing_time,
        MetricType.HISTOGRAM
    )
    
    monitor.collect_metrics(
        "feature_engineer",
        "features_generated",
        len(features.columns),
        MetricType.COUNTER
    )
    
    # 6. 获取性能报告
    report = monitor.get_performance_report()
    
    print("带监控的特征工程完成")
    print(f"处理时间: {processing_time:.3f}秒")
    print(f"生成特征数: {len(features.columns)}")
    print("性能报告:", report)
    
    # 7. 停止监控
    monitor.stop_monitoring()
    
    return features

# 运行示例
if __name__ == "__main__":
    features = monitored_feature_engineering_example()
```

## 故障排除

### 常见问题

#### 1. 导入错误
**问题**: `ModuleNotFoundError: No module named 'src.features'`
**解决方案**: 
```bash
# 确保在项目根目录
cd /path/to/RQA2025

# 设置PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

#### 2. 数据验证错误
**问题**: `ValueError: 缺少必需列: ['open']`
**解决方案**: 确保数据包含所有必需的OHLCV列
```python
required_columns = ['open', 'high', 'low', 'close', 'volume']
for col in required_columns:
    if col not in data.columns:
        print(f"缺少列: {col}")
```

#### 3. 内存不足
**问题**: 处理大数据集时内存不足
**解决方案**: 使用数据分块处理
```python
# 配置内存优化
config = {
    "performance": {
        "chunk_size": 1000,  # 分块大小
        "memory_limit": "2GB"  # 内存限制
    }
}
engineer = FeatureEngineer(config=config)
```

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

## 下一步

### 1. 学习高级功能
- 阅读完整的API文档
- 学习自定义特征开发
- 了解性能优化技巧

### 2. 探索监控功能
- 学习告警配置
- 了解性能分析
- 掌握指标导出

### 3. 参与开发
- 查看源代码
- 阅读架构文档
- 参与测试和优化

## 获取帮助

### 1. 文档资源
- API文档: `docs/api/features/`
- 架构文档: `docs/architecture/features/`
- 配置文档: `docs/configuration/features/`

### 2. 示例代码
- 基础示例: `examples/features/`
- 高级示例: `examples/advanced/`
- 集成示例: `examples/integration/`

### 3. 测试用例
- 单元测试: `tests/unit/features/`
- 集成测试: `tests/integration/features/`
- 性能测试: `tests/performance/features/`

---

**指南版本**: 1.0.0  
**最后更新**: 2025-01-27  
**维护者**: 开发团队 