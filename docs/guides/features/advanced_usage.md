# 特征层高级使用指南

## 概述
本指南介绍RQA2025项目特征层的高级功能和使用技巧，包括自定义特征开发、性能优化、监控配置、错误处理等高级主题。

## 自定义特征开发

### 1. 创建自定义技术指标

#### 基础自定义指标
```python
from src.features.technical.technical_processor import TechnicalProcessor
import pandas as pd
import numpy as np

class CustomTechnicalProcessor(TechnicalProcessor):
    """自定义技术指标处理器"""
    
    def calculate_custom_ma(self, prices, window=10, method='sma'):
        """自定义移动平均线"""
        if method == 'sma':
            return prices.rolling(window=window).mean()
        elif method == 'ema':
            return prices.ewm(span=window).mean()
        elif method == 'wma':
            weights = np.arange(1, window + 1)
            return prices.rolling(window=window).apply(
                lambda x: np.dot(x, weights) / weights.sum()
            )
        else:
            raise ValueError(f"不支持的方法: {method}")
    
    def calculate_volatility_ratio(self, df, window=20):
        """计算波动率比率"""
        returns = df['close'].pct_change()
        volatility = returns.rolling(window=window).std()
        avg_volatility = volatility.rolling(window=window).mean()
        return volatility / avg_volatility
    
    def calculate_price_momentum(self, df, window=10):
        """计算价格动量"""
        return (df['close'] - df['close'].shift(window)) / df['close'].shift(window)

# 使用自定义处理器
custom_processor = CustomTechnicalProcessor()

# 准备数据
data = pd.DataFrame({
    'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
    'high': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
    'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
    'close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
})

# 计算自定义指标
custom_ma = custom_processor.calculate_custom_ma(data['close'], window=5, method='ema')
volatility_ratio = custom_processor.calculate_volatility_ratio(data, window=5)
momentum = custom_processor.calculate_price_momentum(data, window=3)

print("自定义指标计算结果:")
print("EMA:", custom_ma.values)
print("波动率比率:", volatility_ratio.values)
print("价格动量:", momentum.values)
```

### 2. 创建自定义特征工程器

```python
from src.features.feature_engineer import FeatureEngineer

class CustomFeatureEngineer(FeatureEngineer):
    """自定义特征工程器"""
    
    def __init__(self, technical_processor=None, config=None):
        super().__init__(technical_processor, config)
        self.custom_features = {}
    
    def add_custom_feature(self, name, feature_func):
        """添加自定义特征函数"""
        self.custom_features[name] = feature_func
    
    def generate_custom_features(self, stock_data):
        """生成自定义特征"""
        custom_features = pd.DataFrame(index=stock_data.index)
        
        for name, func in self.custom_features.items():
            try:
                custom_features[name] = func(stock_data)
            except Exception as e:
                print(f"生成特征 {name} 时出错: {e}")
                custom_features[name] = np.nan
        
        return custom_features
    
    def generate_all_features(self, stock_data, indicators=None, params=None):
        """生成所有特征（包括自定义特征）"""
        # 生成标准技术指标
        technical_features = self.generate_technical_features(stock_data, indicators, params)
        
        # 生成自定义特征
        custom_features = self.generate_custom_features(stock_data)
        
        # 合并特征
        all_features = pd.concat([technical_features, custom_features], axis=1)
        
        return all_features

# 使用自定义特征工程器
custom_engineer = CustomFeatureEngineer()

# 添加自定义特征
def volume_price_trend(stock_data):
    """成交量价格趋势"""
    return (stock_data['volume'] * stock_data['close']).pct_change()

def price_volume_ratio(stock_data):
    """价格成交量比率"""
    return stock_data['close'] / stock_data['volume']

custom_engineer.add_custom_feature('volume_price_trend', volume_price_trend)
custom_engineer.add_custom_feature('price_volume_ratio', price_volume_ratio)

# 生成所有特征
all_features = custom_engineer.generate_all_features(
    stock_data=data,
    indicators=["ma", "rsi"],
    params={"ma": {"window": [5, 10]}, "rsi": {"window": 14}}
)

print("所有特征列:", all_features.columns.tolist())
```

## 性能优化技巧

### 1. 并行处理优化

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

class ParallelFeatureEngineer(FeatureEngineer):
    """并行特征工程器"""
    
    def __init__(self, technical_processor=None, config=None, n_jobs=-1):
        super().__init__(technical_processor, config)
        self.n_jobs = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
    
    def generate_technical_features_parallel(self, stock_data, indicators=None, params=None):
        """并行生成技术指标"""
        if indicators is None:
            indicators = ["ma", "rsi"]
        
        # 准备并行任务
        tasks = []
        for indicator in indicators:
            indicator_params = params.get(indicator, {}) if params else {}
            tasks.append((indicator, indicator_params))
        
        # 并行计算
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = []
            for indicator, indicator_params in tasks:
                future = executor.submit(
                    self._calculate_single_indicator,
                    stock_data, indicator, indicator_params
                )
                futures.append((indicator, future))
            
            # 收集结果
            results = {}
            for indicator, future in futures:
                try:
                    results[indicator] = future.result()
                except Exception as e:
                    print(f"计算指标 {indicator} 时出错: {e}")
                    results[indicator] = pd.DataFrame()
        
        # 合并结果
        all_features = pd.concat(results.values(), axis=1)
        return all_features
    
    def _calculate_single_indicator(self, stock_data, indicator, params):
        """计算单个指标"""
        if indicator == "ma":
            return self.technical_processor.calculate_ma(stock_data, **params)
        elif indicator == "rsi":
            return self.technical_processor.calculate_rsi(stock_data, **params)
        elif indicator == "macd":
            return self.technical_processor.calculate_macd(stock_data, **params)
        else:
            raise ValueError(f"不支持的指标: {indicator}")

# 使用并行特征工程器
parallel_engineer = ParallelFeatureEngineer(n_jobs=4)

# 并行生成特征
features = parallel_engineer.generate_technical_features_parallel(
    stock_data=data,
    indicators=["ma", "rsi", "macd"],
    params={
        "ma": {"window": [5, 10, 20]},
        "rsi": {"window": 14},
        "macd": {"fast_window": 12, "slow_window": 26, "signal_window": 9}
    }
)

print("并行处理完成，特征形状:", features.shape)
```

### 2. 内存优化

```python
class MemoryOptimizedFeatureEngineer(FeatureEngineer):
    """内存优化的特征工程器"""
    
    def __init__(self, technical_processor=None, config=None, chunk_size=1000):
        super().__init__(technical_processor, config)
        self.chunk_size = chunk_size
    
    def generate_features_in_chunks(self, stock_data, indicators=None, params=None):
        """分块生成特征"""
        total_rows = len(stock_data)
        chunks = []
        
        for start_idx in range(0, total_rows, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, total_rows)
            chunk_data = stock_data.iloc[start_idx:end_idx]
            
            # 生成当前块的特征
            chunk_features = self.generate_technical_features(
                chunk_data, indicators, params
            )
            chunks.append(chunk_features)
            
            # 清理内存
            del chunk_data
        
        # 合并所有块
        all_features = pd.concat(chunks, axis=0)
        return all_features

# 使用内存优化的特征工程器
memory_engineer = MemoryOptimizedFeatureEngineer(chunk_size=500)

# 分块处理大数据集
large_features = memory_engineer.generate_features_in_chunks(
    stock_data=data,
    indicators=["ma", "rsi"],
    params={"ma": {"window": [5, 10]}, "rsi": {"window": 14}}
)

print("内存优化处理完成，特征形状:", large_features.shape)
```

## 监控配置

### 1. 高级监控配置

```python
from src.features.monitoring import FeaturesMonitor, MetricType
import time
import psutil

class AdvancedFeatureMonitor(FeaturesMonitor):
    """高级特征监控器"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.performance_history = []
        self.error_history = []
    
    def monitor_system_resources(self):
        """监控系统资源"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available': memory.available / (1024**3),  # GB
            'disk_percent': disk.percent,
            'disk_free': disk.free / (1024**3)  # GB
        }
    
    def monitor_feature_generation(self, component_name, start_time):
        """监控特征生成性能"""
        end_time = time.time()
        duration = end_time - start_time
        
        # 收集系统资源
        system_resources = self.monitor_system_resources()
        
        # 记录性能指标
        self.collect_metrics(
            component_name,
            "generation_time",
            duration,
            MetricType.HISTOGRAM
        )
        
        self.collect_metrics(
            component_name,
            "cpu_usage",
            system_resources['cpu_percent'],
            MetricType.GAUGE
        )
        
        return duration, system_resources

# 使用高级监控器
advanced_monitor = AdvancedFeatureMonitor()

# 注册组件
advanced_monitor.register_component("feature_engineer", "processor")
advanced_monitor.start_monitoring()

# 监控特征生成
start_time = time.time()
features = engineer.generate_technical_features(data, ["ma", "rsi"])
duration, resources = advanced_monitor.monitor_feature_generation(
    "feature_engineer", start_time
)

print("性能监控结果:")
print(f"生成时间: {duration:.3f}秒")
print(f"CPU使用率: {resources['cpu_percent']}%")
print(f"内存使用率: {resources['memory_percent']}%")
```

## 错误处理最佳实践

### 1. 异常处理装饰器

```python
import functools
import logging

def handle_feature_errors(func):
    """特征处理错误处理装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            logging.error(f"数据验证错误: {e}")
            raise
        except MemoryError as e:
            logging.error(f"内存不足: {e}")
            # 尝试清理内存
            import gc
            gc.collect()
            raise
        except Exception as e:
            logging.error(f"未知错误: {e}")
            raise
    return wrapper

# 使用装饰器
@handle_feature_errors
def safe_feature_generation(data, indicators):
    """安全的特征生成"""
    return engineer.generate_technical_features(data, indicators)

# 测试错误处理
try:
    features = safe_feature_generation(data, ["invalid_indicator"])
except Exception as e:
    print(f"错误已处理: {e}")
```

### 2. 数据验证和修复

```python
class DataValidator:
    """数据验证器"""
    
    @staticmethod
    def validate_stock_data(data):
        """验证股票数据"""
        errors = []
        warnings = []
        
        # 检查必需列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            errors.append(f"缺少必需列: {missing_columns}")
        
        # 检查数据类型
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in data.columns and not pd.api.types.is_numeric_dtype(data[col]):
                errors.append(f"列 {col} 不是数值类型")
        
        # 检查数据逻辑
        if 'high' in data.columns and 'low' in data.columns:
            invalid_rows = data[data['high'] < data['low']]
            if not invalid_rows.empty:
                warnings.append(f"发现 {len(invalid_rows)} 行高低价逻辑错误")
        
        # 检查缺失值
        missing_counts = data.isnull().sum()
        if missing_counts.sum() > 0:
            warnings.append(f"发现缺失值: {missing_counts.to_dict()}")
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    @staticmethod
    def repair_stock_data(data):
        """修复股票数据"""
        repaired_data = data.copy()
        
        # 修复缺失值
        repaired_data = repaired_data.fillna(method='ffill').fillna(method='bfill')
        
        # 修复负值
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in repaired_data.columns:
                repaired_data[col] = repaired_data[col].abs()
        
        # 修复价格逻辑
        if 'high' in repaired_data.columns and 'low' in repaired_data.columns:
            mask = repaired_data['high'] < repaired_data['low']
            repaired_data.loc[mask, 'high'] = repaired_data.loc[mask, 'low']
        
        return repaired_data

# 使用数据验证器
validator = DataValidator()

# 验证数据
validation_result = validator.validate_stock_data(data)
print("数据验证结果:", validation_result)

# 修复数据
if not validation_result['is_valid']:
    print("修复数据...")
    repaired_data = validator.repair_stock_data(data)
    print("数据修复完成")
```

## 最佳实践总结

### 1. 性能优化
- 使用并行处理处理大数据集
- 实现缓存机制避免重复计算
- 分块处理减少内存使用
- 定期清理不需要的数据

### 2. 监控和告警
- 设置合理的监控阈值
- 实现多级告警机制
- 记录详细的性能指标
- 定期分析性能趋势

### 3. 错误处理
- 实现全面的异常处理
- 提供数据验证和修复功能
- 记录详细的错误日志
- 实现优雅的降级机制

### 4. 代码质量
- 遵循Python编码规范
- 编写完整的文档和注释
- 实现全面的单元测试
- 定期进行代码审查

---

**指南版本**: 1.0.0  
**最后更新**: 2025-01-27  
**维护者**: 开发团队 