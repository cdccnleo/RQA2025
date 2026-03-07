# FeatureStandardizer API 文档

## 概述
`FeatureStandardizer` 是特征层的特征标准化器，提供多种标准化方法，包括Z-score标准化、Min-Max标准化、Robust标准化等，帮助用户将特征数据转换为标准格式，提高模型性能。

## 类和方法

### FeatureStandardizer
特征标准化器，提供多种标准化方法

#### 初始化
```python
def __init__(self, method='zscore', **kwargs):
```

**参数**:
- `method` (str): 标准化方法，可选值：'zscore', 'minmax', 'robust', 'log', 'boxcox'
- `**kwargs`: 其他参数

**示例**:
```python
from src.features.processors.feature_standardizer import FeatureStandardizer

# 使用Z-score标准化
standardizer = FeatureStandardizer(method='zscore')

# 使用Min-Max标准化
standardizer = FeatureStandardizer(method='minmax', feature_range=(0, 1))

# 使用Robust标准化
standardizer = FeatureStandardizer(method='robust')
```

#### 方法

##### fit(X, y=None)
训练标准化器

**参数**:
- `X` (pd.DataFrame): 特征数据
- `y` (pd.Series, optional): 目标变量（通常不使用）

**返回**:
- `self`: 训练后的标准化器实例

**示例**:
```python
import pandas as pd
import numpy as np

# 准备数据
X = pd.DataFrame({
    'feature1': np.random.randn(100),
    'feature2': np.random.randn(100) * 10,
    'feature3': np.random.randn(100) * 100
})

# 训练标准化器
standardizer.fit(X)
```

##### transform(X)
应用标准化

**参数**:
- `X` (pd.DataFrame): 待标准化的特征数据

**返回**:
- `pd.DataFrame`: 标准化后的特征数据

**示例**:
```python
# 应用标准化
X_standardized = standardizer.transform(X)
print("标准化后的数据:")
print(X_standardized.describe())
```

##### fit_transform(X, y=None)
训练并应用标准化

**参数**:
- `X` (pd.DataFrame): 特征数据
- `y` (pd.Series, optional): 目标变量

**返回**:
- `pd.DataFrame`: 标准化后的特征数据

**示例**:
```python
# 一步完成训练和标准化
X_standardized = standardizer.fit_transform(X)
```

##### inverse_transform(X)
逆变换

**参数**:
- `X` (pd.DataFrame): 标准化后的特征数据

**返回**:
- `pd.DataFrame`: 原始尺度的特征数据

**示例**:
```python
# 逆变换
X_original = standardizer.inverse_transform(X_standardized)
```

##### get_params()
获取标准化参数

**返回**:
- `dict`: 标准化参数字典

**示例**:
```python
params = standardizer.get_params()
print("标准化参数:", params)
```

##### set_params(**params)
设置标准化参数

**参数**:
- `**params`: 参数字典

**返回**:
- `self`: 更新后的标准化器实例

**示例**:
```python
# 设置参数
standardizer.set_params(method='minmax', feature_range=(-1, 1))
```

## 配置选项

### Z-score标准化配置
```python
zscore_config = {
    "method": "zscore",
    "with_mean": True,      # 是否减去均值
    "with_std": True,       # 是否除以标准差
    "copy": True            # 是否复制数据
}
```

### Min-Max标准化配置
```python
minmax_config = {
    "method": "minmax",
    "feature_range": (0, 1),  # 特征范围
    "copy": True              # 是否复制数据
}
```

### Robust标准化配置
```python
robust_config = {
    "method": "robust",
    "quantile_range": (25.0, 75.0),  # 分位数范围
    "copy": True                      # 是否复制数据
}
```

### 对数变换配置
```python
log_config = {
    "method": "log",
    "base": "e",             # 对数底数
    "offset": 1,             # 偏移量
    "copy": True             # 是否复制数据
}
```

### Box-Cox变换配置
```python
boxcox_config = {
    "method": "boxcox",
    "lmbda": None,           # lambda参数
    "copy": True             # 是否复制数据
}
```

## 使用示例

### 基础使用
```python
from src.features.processors.feature_standardizer import FeatureStandardizer
import pandas as pd
import numpy as np

# 准备示例数据
np.random.seed(42)
X = pd.DataFrame({
    'feature1': np.random.randn(100),
    'feature2': np.random.randn(100) * 10,
    'feature3': np.random.randn(100) * 100
})

print("原始数据统计:")
print(X.describe())

# 创建标准化器
standardizer = FeatureStandardizer(method='zscore')

# 训练和标准化
X_standardized = standardizer.fit_transform(X)

print("\n标准化后数据统计:")
print(X_standardized.describe())
```

### 高级使用
```python
# 使用不同方法进行标准化
methods = ['zscore', 'minmax', 'robust']

for method in methods:
    print(f"\n使用 {method} 方法:")
    
    # 创建标准化器
    standardizer = FeatureStandardizer(method=method)
    
    # 训练和标准化
    X_standardized = standardizer.fit_transform(X)
    
    print("标准化后统计:")
    print(X_standardized.describe())
    
    # 获取参数
    params = standardizer.get_params()
    print("标准化参数:", params)
```

### 批量标准化
```python
def batch_standardization(X, methods):
    """批量标准化"""
    results = {}
    
    for method in methods:
        try:
            # 创建标准化器
            standardizer = FeatureStandardizer(method=method)
            
            # 训练和标准化
            X_standardized = standardizer.fit_transform(X)
            
            results[method] = {
                'data': X_standardized,
                'params': standardizer.get_params(),
                'stats': X_standardized.describe().to_dict()
            }
            
        except Exception as e:
            print(f"方法 {method} 失败: {e}")
            results[method] = {'error': str(e)}
    
    return results

# 执行批量标准化
methods = ['zscore', 'minmax', 'robust', 'log']

results = batch_standardization(X, methods)

# 显示结果
for method, result in results.items():
    print(f"\n{method} 方法:")
    if 'error' not in result:
        print("  参数:", result['params'])
        print("  统计信息:")
        for stat, values in result['stats'].items():
            print(f"    {stat}: {values}")
    else:
        print(f"  错误: {result['error']}")
```

### 特征工程流水线
```python
from src.features.processors.feature_selector import FeatureSelector

def feature_engineering_pipeline(X, y):
    """特征工程流水线"""
    
    # 1. 特征选择
    print("步骤1: 特征选择")
    selector = FeatureSelector(method='importance', n_features=2)
    X_selected = selector.fit_transform(X, y)
    print(f"选择的特征: {X_selected.columns.tolist()}")
    
    # 2. 特征标准化
    print("\n步骤2: 特征标准化")
    standardizer = FeatureStandardizer(method='zscore')
    X_standardized = standardizer.fit_transform(X_selected)
    
    # 3. 显示结果
    print("\n最终结果:")
    print("原始数据形状:", X.shape)
    print("处理后数据形状:", X_standardized.shape)
    print("处理后统计:")
    print(X_standardized.describe())
    
    return X_standardized

# 执行流水线
y = pd.Series(np.random.randn(100))
X_final = feature_engineering_pipeline(X, y)
```

### 交叉验证标准化
```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

def cross_validate_standardization(X, y, method='zscore', cv=5):
    """交叉验证标准化"""
    scores = []
    
    for i in range(cv):
        # 分割数据
        split_idx = len(X) // cv
        start_idx = i * split_idx
        end_idx = start_idx + split_idx
        
        X_train = X.iloc[:start_idx].append(X.iloc[end_idx:])
        y_train = y.iloc[:start_idx].append(y.iloc[end_idx:])
        X_test = X.iloc[start_idx:end_idx]
        y_test = y.iloc[start_idx:end_idx]
        
        # 标准化
        standardizer = FeatureStandardizer(method=method)
        X_train_standardized = standardizer.fit_transform(X_train)
        X_test_standardized = standardizer.transform(X_test)
        
        # 训练模型
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_standardized, y_train)
        
        # 评估
        score = model.score(X_test_standardized, y_test)
        scores.append(score)
    
    return {
        'scores': scores,
        'mean_score': np.mean(scores),
        'std_score': np.std(scores)
    }

# 执行交叉验证
cv_results = cross_validate_standardization(X, y, method='zscore')

print("交叉验证结果:")
print(f"平均得分: {cv_results['mean_score']:.4f}")
print(f"标准差: {cv_results['std_score']:.4f}")
```

## 性能优化建议

### 1. 数据预处理
- 确保输入数据格式正确
- 处理缺失值和异常值
- 检查数据类型

### 2. 方法选择
- **Z-score标准化**: 适用于正态分布数据，对异常值敏感
- **Min-Max标准化**: 适用于有界数据，保持零值
- **Robust标准化**: 适用于有异常值的数据
- **对数变换**: 适用于右偏分布数据
- **Box-Cox变换**: 适用于需要正态化的数据

### 3. 参数调优
- 根据数据分布选择合适的方法
- 调整特征范围（Min-Max方法）
- 设置合适的分位数范围（Robust方法）

### 4. 内存管理
- 对于大数据集，使用增量学习
- 及时释放不需要的数据
- 使用适当的数据类型

## 故障排除

### 常见问题

#### 1. 数据格式错误
**问题**: `ValueError: 输入数据必须为DataFrame`
**解决方案**: 确保输入数据为pandas DataFrame格式

#### 2. 零方差特征
**问题**: `ValueError: 特征方差为零`
**解决方案**: 移除零方差特征或使用其他标准化方法

#### 3. 负值数据
**问题**: `ValueError: 数据包含负值，无法进行对数变换`
**解决方案**: 使用偏移量或选择其他变换方法

#### 4. 内存不足
**问题**: 处理大数据集时内存不足
**解决方案**: 使用数据分块处理或选择更高效的方法

### 调试技巧

#### 1. 检查数据质量
```python
# 检查数据基本信息
print(X.info())
print(X.describe())

# 检查缺失值
print(X.isnull().sum())

# 检查零方差特征
zero_var_features = X.columns[X.var() == 0]
print("零方差特征:", zero_var_features.tolist())
```

#### 2. 验证标准化结果
```python
# 检查标准化后的数据
print("标准化后统计:")
print(X_standardized.describe())

# 检查是否在预期范围内
if method == 'minmax':
    print("数据范围:", X_standardized.min().min(), "到", X_standardized.max().max())
elif method == 'zscore':
    print("均值接近0:", abs(X_standardized.mean()).max() < 1e-10)
    print("标准差接近1:", abs(X_standardized.std() - 1).max() < 1e-10)
```

#### 3. 性能分析
```python
import time

# 测量标准化时间
start_time = time.time()
X_standardized = standardizer.fit_transform(X)
end_time = time.time()

print(f"标准化时间: {end_time - start_time:.3f}秒")
```

## 版本历史

### v1.0.0
- 初始版本
- 基础标准化功能
- 支持Z-score、Min-Max方法

### v1.1.0
- 添加Robust标准化
- 添加对数变换
- 增强错误处理

### v1.2.0
- 添加Box-Cox变换
- 优化性能
- 完善文档

---

**文档版本**: 1.2.0  
**最后更新**: 2025-01-27  
**维护者**: 开发团队 