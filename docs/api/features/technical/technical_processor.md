# TechnicalProcessor API 文档

## 概述
`TechnicalProcessor` 是特征层的高性能技术指标处理器，提供各种技术指标的计算功能，支持多种调用方式和优化算法。

## 类和方法

### TechnicalProcessor
高性能技术指标处理器，兼容多种调用方式

#### 初始化
```python
def __init__(self, data=None, register_func: Optional[Callable] = None):
```

**参数**:
- `data` (pd.DataFrame, optional): 价格数据DataFrame
- `register_func` (Callable, optional): 特征注册函数

**示例**:
```python
from src.features.technical.technical_processor import TechnicalProcessor
import pandas as pd

# 使用默认初始化
processor = TechnicalProcessor()

# 使用价格数据初始化
data = pd.DataFrame({
    'open': [100, 101, 102, 103, 104],
    'high': [102, 103, 104, 105, 106],
    'low': [99, 100, 101, 102, 103],
    'close': [101, 102, 103, 104, 105],
    'volume': [1000, 1100, 1200, 1300, 1400]
})
processor = TechnicalProcessor(data=data)
```

#### 方法

##### calculate_ma(prices=None, window=5, price_col='close')
计算移动平均线

**参数**:
- `prices` (pd.DataFrame/pd.Series, optional): 价格数据
- `window` (int): 窗口大小，默认5
- `price_col` (str): 价格列名，默认'close'

**返回**:
- `pd.DataFrame`: 包含移动平均线的DataFrame

**示例**:
```python
# 计算5日和10日移动平均线
ma5 = processor.calculate_ma(data, window=5)
ma10 = processor.calculate_ma(data, window=10)

print(ma5.head())
print(ma10.head())
```

##### calculate_rsi(prices=None, window=14, price_col='close')
计算相对强弱指数(RSI)

**参数**:
- `prices` (pd.DataFrame/pd.Series, optional): 价格数据
- `window` (int): 窗口大小，默认14
- `price_col` (str): 价格列名，默认'close'

**返回**:
- `pd.DataFrame`: 包含RSI的DataFrame

**示例**:
```python
# 计算14日RSI
rsi = processor.calculate_rsi(data, window=14)
print(rsi.head())
```

##### calculate_macd(prices=None, fast_window=None, slow_window=None, signal_window=None, short_window=None, long_window=None, price_col='close')
计算MACD指标

**参数**:
- `prices` (pd.DataFrame/pd.Series, optional): 价格数据
- `fast_window` (int, optional): 快线周期，默认12
- `slow_window` (int, optional): 慢线周期，默认26
- `signal_window` (int, optional): 信号线周期，默认9
- `short_window` (int, optional): 短期周期（兼容参数）
- `long_window` (int, optional): 长期周期（兼容参数）
- `price_col` (str): 价格列名，默认'close'

**返回**:
- `pd.DataFrame`: 包含MACD_DIF、MACD_DEA、MACD_Histogram的DataFrame

**示例**:
```python
# 计算MACD指标
macd = processor.calculate_macd(
    data,
    fast_window=12,
    slow_window=26,
    signal_window=9
)
print(macd.head())
```

##### calculate_bollinger(prices=None, window=20, num_std=2, price_col='close')
计算布林带指标

**参数**:
- `prices` (pd.DataFrame/pd.Series, optional): 价格数据
- `window` (int): 窗口大小，默认20
- `num_std` (int): 标准差倍数，默认2
- `price_col` (str): 价格列名，默认'close'

**返回**:
- `pd.DataFrame`: 包含BOLL_UPPER、BOLL_MIDDLE、BOLL_LOWER的DataFrame

**示例**:
```python
# 计算布林带
bollinger = processor.calculate_bollinger(data, window=20, num_std=2)
print(bollinger.head())
```

##### calculate_obv(df, price_col='close', volume_col='volume')
计算能量潮指标(OBV)

**参数**:
- `df` (pd.DataFrame): 包含价格和成交量的DataFrame
- `price_col` (str): 价格列名，默认'close'
- `volume_col` (str): 成交量列名，默认'volume'

**返回**:
- `pd.DataFrame`: 包含OBV的DataFrame

**示例**:
```python
# 计算OBV
obv = processor.calculate_obv(data)
print(obv.head())
```

##### calculate_atr(df, window=14, high_col='high', low_col='low', close_col='close')
计算平均真实波幅(ATR)

**参数**:
- `df` (pd.DataFrame): 包含OHLC数据的DataFrame
- `window` (int): 窗口大小，默认14
- `high_col` (str): 最高价列名，默认'high'
- `low_col` (str): 最低价列名，默认'low'
- `close_col` (str): 收盘价列名，默认'close'

**返回**:
- `pd.DataFrame`: 包含ATR的DataFrame

**示例**:
```python
# 计算ATR
atr = processor.calculate_atr(data, window=14)
print(atr.head())
```

##### calculate_indicators(df, indicators, params=None)
批量计算技术指标

**参数**:
- `df` (pd.DataFrame): 价格数据
- `indicators` (list): 指标列表，如["ma", "rsi", "macd"]
- `params` (dict, optional): 指标参数配置

**返回**:
- `pd.DataFrame`: 包含所有计算指标的DataFrame

**示例**:
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
print(result.head())
```

##### calculate_volatility_moments(df, price_col='close', window=20)
计算波动率矩

**参数**:
- `df` (pd.DataFrame): 价格数据
- `price_col` (str): 价格列名，默认'close'
- `window` (int): 窗口大小，默认20

**返回**:
- `pd.DataFrame`: 包含波动率、偏度、峰度的DataFrame

**示例**:
```python
# 计算波动率矩
volatility = processor.calculate_volatility_moments(data, window=20)
print(volatility.head())
```

##### extreme_value_analysis(df, price_col='close', threshold=2)
极值分析

**参数**:
- `df` (pd.DataFrame): 价格数据
- `price_col` (str): 价格列名，默认'close'
- `threshold` (float): 极值阈值，默认2

**返回**:
- `pd.DataFrame`: 包含极值分析结果的DataFrame

**示例**:
```python
# 极值分析
extreme = processor.extreme_value_analysis(data, threshold=2)
print(extreme.head())
```

## 配置选项

### 性能优化配置
```python
performance_config = {
    "enable_numba": True,        # 启用Numba加速
    "parallel_processing": True,  # 启用并行处理
    "cache_results": True,        # 缓存计算结果
    "memory_optimization": True   # 内存优化
}
```

### 计算精度配置
```python
precision_config = {
    "float_precision": "float64",  # 浮点精度
    "round_decimals": 4,           # 小数位数
    "handle_nan": True,            # 处理NaN值
    "fill_method": "forward"       # 填充方法
}
```

## 使用示例

### 基础使用
```python
from src.features.technical.technical_processor import TechnicalProcessor
import pandas as pd

# 创建处理器
processor = TechnicalProcessor()

# 准备数据
data = pd.DataFrame({
    'open': [100, 101, 102, 103, 104],
    'high': [102, 103, 104, 105, 106],
    'low': [99, 100, 101, 102, 103],
    'close': [101, 102, 103, 104, 105],
    'volume': [1000, 1100, 1200, 1300, 1400]
}, index=pd.date_range('2023-01-01', periods=5))

# 计算单个指标
ma5 = processor.calculate_ma(data, window=5)
rsi = processor.calculate_rsi(data, window=14)
macd = processor.calculate_macd(data)

print("移动平均线:", ma5.head())
print("RSI:", rsi.head())
print("MACD:", macd.head())
```

### 高级使用
```python
# 批量计算多个指标
indicators = ["ma", "rsi", "macd", "bollinger", "obv", "atr"]
params = {
    "ma": {"window": [5, 10, 20, 50]},
    "rsi": {"window": 14},
    "macd": {"fast_window": 12, "slow_window": 26, "signal_window": 9},
    "bollinger": {"window": 20, "num_std": 2},
    "atr": {"window": 14}
}

# 批量计算
result = processor.calculate_indicators(data, indicators, params)

# 查看结果
print("计算结果形状:", result.shape)
print("指标列:", result.columns.tolist())
print("前5行:", result.head())
```

### 性能优化使用
```python
# 使用Numba加速的底层方法
import numpy as np

# 直接使用numpy数组
prices = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])

# 计算RSI（使用Numba加速）
rsi_values = processor.calculate_rsi(prices=prices, window=5)
print("RSI值:", rsi_values)

# 计算MACD（使用Numba加速）
macd_result = processor.calculate_macd(prices=prices)
print("MACD结果:", macd_result)
```

### 错误处理
```python
try:
    # 尝试计算指标
    result = processor.calculate_rsi(data, window=14)
except ValueError as e:
    print(f"数据验证错误: {e}")
except Exception as e:
    print(f"计算错误: {e}")
```

## 性能优化建议

### 1. 数据预处理
- 确保输入数据格式正确
- 预先处理缺失值和异常值
- 使用适当的数据类型

### 2. 批量计算
- 使用`calculate_indicators`批量计算多个指标
- 减少重复的数据验证和预处理
- 利用向量化操作提高性能

### 3. Numba加速
- 对于大数据集，启用Numba加速
- 使用numpy数组而不是pandas Series
- 避免在循环中调用pandas方法

### 4. 内存管理
- 及时释放不需要的中间结果
- 使用适当的数据类型减少内存占用
- 避免创建不必要的数据副本

## 故障排除

### 常见问题

#### 1. 数据格式错误
**问题**: `ValueError: 价格数据包含NaN值`
**解决方案**: 检查并清理输入数据中的NaN值

#### 2. 窗口大小错误
**问题**: `ValueError: 窗口必须为正数`
**解决方案**: 确保窗口大小为正整数

#### 3. 数据长度不足
**问题**: `ValueError: 数据长度不足`
**解决方案**: 确保数据长度大于等于窗口大小

#### 4. 内存不足
**问题**: 处理大数据集时内存不足
**解决方案**: 使用数据分块处理，启用内存优化

### 调试技巧

#### 1. 检查数据质量
```python
# 检查数据基本信息
print(data.info())
print(data.describe())

# 检查缺失值
print(data.isnull().sum())
```

#### 2. 验证计算结果
```python
# 检查计算结果
print(result.info())
print(result.describe())

# 检查特定指标
if 'RSI' in result.columns:
    print(f"RSI范围: {result['RSI'].min()} - {result['RSI'].max()}")
```

#### 3. 性能分析
```python
import time

# 测量计算时间
start_time = time.time()
result = processor.calculate_indicators(data, indicators, params)
end_time = time.time()

print(f"计算时间: {end_time - start_time:.3f}秒")
```

## 版本历史

### v1.0.0
- 初始版本
- 基础技术指标计算
- 支持pandas DataFrame输入

### v1.1.0
- 添加Numba加速
- 增强错误处理
- 支持numpy数组输入

### v1.2.0
- 添加批量计算功能
- 优化内存使用
- 增强数据验证

### v1.3.0
- 添加高级指标（波动率矩、极值分析）
- 完善文档
- 性能优化

---

**文档版本**: 1.3.0  
**最后更新**: 2025-01-27  
**维护者**: 开发团队 