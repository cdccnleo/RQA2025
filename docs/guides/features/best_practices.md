# 特征层最佳实践指南

## 概述
本指南提供RQA2025项目特征层开发和使用的最佳实践，包括代码规范、性能优化、错误处理、测试策略等方面的建议。

## 代码规范

### 1. 命名规范

#### 类命名
```python
# ✅ 正确：使用PascalCase
class FeatureEngineer:
    pass

class TechnicalProcessor:
    pass

# ❌ 错误：使用其他命名方式
class feature_engineer:
    pass
```

#### 函数命名
```python
# ✅ 正确：使用snake_case
def generate_technical_features():
    pass

def calculate_moving_average():
    pass

# ❌ 错误：使用其他命名方式
def generateTechnicalFeatures():
    pass
```

### 2. 文档规范

#### 类和函数文档
```python
class FeatureEngineer:
    """
    特征工程器，负责生成和管理技术指标特征。
    
    主要功能：
    - 生成技术指标特征
    - 数据验证和修复
    - 特征选择和标准化
    """
    
    def generate_technical_features(self, stock_data, indicators=None, params=None):
        """
        生成技术指标特征。
        
        参数：
            stock_data (pd.DataFrame): 股票数据，必须包含OHLCV列
            indicators (list, optional): 技术指标列表，默认为["ma", "rsi"]
            params (dict, optional): 指标参数配置
            
        返回：
            pd.DataFrame: 包含技术指标的特征数据框
        """
        pass
```

## 性能优化

### 1. 数据处理优化

#### 使用向量化操作
```python
# ✅ 正确：使用向量化操作
def calculate_returns(prices):
    """计算收益率"""
    return prices.pct_change()

def calculate_volatility(returns, window=20):
    """计算波动率"""
    return returns.rolling(window=window).std()

# ❌ 错误：使用循环
def calculate_returns_slow(prices):
    """计算收益率（慢速版本）"""
    returns = []
    for i in range(1, len(prices)):
        returns.append((prices[i] - prices[i-1]) / prices[i-1])
    return returns
```

### 2. 内存管理
```python
# ✅ 正确：及时释放内存
def process_large_dataset(data_iterator):
    """处理大型数据集"""
    results = []
    
    for chunk in data_iterator:
        processed_chunk = process_chunk(chunk)
        results.append(processed_chunk)
        
        # 及时释放内存
        del chunk
        del processed_chunk
    
    return pd.concat(results, ignore_index=True)
```

## 错误处理

### 1. 异常处理策略

```python
class FeatureProcessingError(Exception):
    """特征处理异常基类"""
    pass

def safe_feature_processing(data, processor):
    """安全的特征处理"""
    try:
        # 数据验证
        if not validate_data(data):
            raise ValueError("数据验证失败")
        
        # 特征处理
        result = processor.process(data)
        
        return result
        
    except ValueError as e:
        logging.error(f"数据验证错误: {e}")
        # 尝试修复数据
        repaired_data = repair_data(data)
        return safe_feature_processing(repaired_data, processor)
        
    except Exception as e:
        logging.error(f"未知错误: {e}")
        raise FeatureProcessingError(f"特征处理失败: {e}")
```

### 2. 数据验证

```python
class DataValidator:
    """数据验证器"""
    
    @staticmethod
    def validate_stock_data(data):
        """验证股票数据"""
        errors = []
        warnings = []
        
        # 基本检查
        if data.empty:
            errors.append("数据为空")
            return {'is_valid': False, 'errors': errors, 'warnings': warnings}
        
        # 列检查
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            errors.append(f"缺少必需列: {missing_columns}")
        
        # 逻辑检查
        if 'high' in data.columns and 'low' in data.columns:
            invalid_rows = data[data['high'] < data['low']]
            if not invalid_rows.empty:
                warnings.append(f"发现 {len(invalid_rows)} 行高低价逻辑错误")
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
```

## 测试策略

### 1. 单元测试

```python
import pytest
import pandas as pd
from src.features.feature_engineer import FeatureEngineer

class TestFeatureEngineer:
    """特征工程器测试类"""
    
    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        return pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [102, 103, 104, 105, 106],
            'low': [99, 100, 101, 102, 103],
            'close': [101, 102, 103, 104, 105],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
    
    def test_generate_technical_features_basic(self, sample_data):
        """测试基础特征生成"""
        engineer = FeatureEngineer()
        features = engineer.generate_technical_features(
            sample_data,
            indicators=["ma"],
            params={"ma": {"window": [5]}}
        )
        
        assert not features.empty
        assert "MA_5" in features.columns
        assert len(features) == len(sample_data)
    
    def test_generate_technical_features_invalid_data(self):
        """测试无效数据处理"""
        engineer = FeatureEngineer()
        invalid_data = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 103]
            # 缺少必需列
        })
        
        with pytest.raises(ValueError):
            engineer.generate_technical_features(invalid_data)
```

### 2. 性能测试

```python
import time
import psutil

class TestFeaturePerformance:
    """特征性能测试"""
    
    def test_large_dataset_performance(self):
        """测试大数据集性能"""
        # 创建大数据集
        large_data = create_large_test_data(10000)
        
        # 记录开始时间和内存
        start_time = time.time()
        start_memory = psutil.virtual_memory().used
        
        # 执行特征工程
        engineer = FeatureEngineer()
        features = engineer.generate_technical_features(
            large_data,
            indicators=["ma", "rsi", "macd", "bollinger"]
        )
        
        # 记录结束时间和内存
        end_time = time.time()
        end_memory = psutil.virtual_memory().used
        
        # 计算性能指标
        processing_time = end_time - start_time
        memory_usage = end_memory - start_memory
        
        # 性能断言
        assert processing_time < 10.0  # 处理时间应小于10秒
        assert memory_usage < 1024 * 1024 * 100  # 内存使用应小于100MB
        assert not features.empty
        assert features.shape[0] == len(large_data)
```

## 监控和日志

### 1. 日志配置

```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    """结构化日志记录器"""
    
    def __init__(self, name, level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # 配置处理器
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_feature_generation(self, component, indicators, data_shape, duration):
        """记录特征生成日志"""
        log_data = {
            'event': 'feature_generation',
            'component': component,
            'indicators': indicators,
            'data_shape': data_shape,
            'duration': duration,
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(json.dumps(log_data))
```

### 2. 性能监控

```python
import time
import psutil

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics = {}
    
    def start_timer(self, operation):
        """开始计时"""
        self.metrics[operation] = {
            'start_time': time.time(),
            'memory_start': psutil.virtual_memory().used
        }
    
    def end_timer(self, operation):
        """结束计时"""
        if operation in self.metrics:
            end_time = time.time()
            end_memory = psutil.virtual_memory().used
            
            duration = end_time - self.metrics[operation]['start_time']
            memory_usage = end_memory - self.metrics[operation]['memory_start']
            
            self.metrics[operation].update({
                'duration': duration,
                'memory_usage': memory_usage,
                'end_time': end_time
            })
    
    def get_performance_report(self):
        """获取性能报告"""
        return self.metrics
```

## 配置管理

### 1. 配置验证

```python
from typing import Dict, Any

class ConfigValidator:
    """配置验证器"""
    
    @staticmethod
    def validate_feature_config(config: Dict[str, Any]) -> bool:
        """验证特征配置"""
        required_keys = ['validation', 'caching', 'performance']
        
        for key in required_keys:
            if key not in config:
                print(f"缺少必需配置项: {key}")
                return False
        
        return True
```

### 2. 环境配置

```python
import os
from typing import Dict, Any

class EnvironmentConfig:
    """环境配置管理器"""
    
    @staticmethod
    def get_feature_config() -> Dict[str, Any]:
        """获取特征配置"""
        return {
            "validation": {
                "strict_mode": os.getenv("FEATURE_STRICT_MODE", "false").lower() == "true",
                "auto_repair": os.getenv("FEATURE_AUTO_REPAIR", "true").lower() == "true"
            },
            "caching": {
                "enable": os.getenv("FEATURE_CACHE_ENABLE", "true").lower() == "true",
                "ttl": int(os.getenv("FEATURE_CACHE_TTL", "3600"))
            },
            "performance": {
                "parallel_processing": os.getenv("FEATURE_PARALLEL", "true").lower() == "true",
                "max_workers": int(os.getenv("FEATURE_MAX_WORKERS", "4"))
            }
        }
```

## 最佳实践总结

### 1. 代码质量
- 遵循Python编码规范（PEP 8）
- 使用类型注解提高代码可读性
- 编写完整的文档字符串
- 实现全面的单元测试

### 2. 性能优化
- 使用向量化操作代替循环
- 实现智能缓存机制
- 采用并行处理处理大数据集
- 及时释放不需要的内存

### 3. 错误处理
- 实现分层的异常处理机制
- 提供数据验证和修复功能
- 记录详细的错误日志
- 实现优雅的降级策略

### 4. 监控和日志
- 使用结构化日志记录
- 收集关键性能指标
- 实现实时监控和告警
- 定期分析性能趋势

### 5. 测试策略
- 编写全面的单元测试
- 实现集成测试验证端到端功能
- 进行性能基准测试
- 建立持续集成流程

---

**指南版本**: 1.0.0  
**最后更新**: 2025-01-27  
**维护者**: 开发团队 