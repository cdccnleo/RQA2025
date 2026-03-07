# FeatureSelector API 文档

## 概述
`FeatureSelector` 是特征层的特征选择器，提供多种特征选择算法，包括基于重要性、相关性、递归特征消除等方法，帮助用户从大量特征中选择最有价值的特征子集。

## 类和方法

### FeatureSelector
特征选择器，提供多种特征选择算法

#### 初始化
```python
def __init__(self, method='importance', **kwargs):
```

**参数**:
- `method` (str): 特征选择方法，可选值：'importance', 'correlation', 'rfe', 'lasso', 'tree'
- `**kwargs`: 其他参数

**示例**:
```python
from src.features.processors.feature_selector import FeatureSelector

# 使用重要性方法
selector = FeatureSelector(method='importance')

# 使用相关性方法
selector = FeatureSelector(method='correlation', threshold=0.8)

# 使用递归特征消除
selector = FeatureSelector(method='rfe', n_features=10)
```

#### 方法

##### fit(X, y=None)
训练特征选择器

**参数**:
- `X` (pd.DataFrame): 特征数据
- `y` (pd.Series, optional): 目标变量

**返回**:
- `self`: 训练后的选择器实例

**示例**:
```python
import pandas as pd
import numpy as np

# 准备数据
X = pd.DataFrame({
    'feature1': np.random.randn(100),
    'feature2': np.random.randn(100),
    'feature3': np.random.randn(100),
    'feature4': np.random.randn(100)
})
y = pd.Series(np.random.randn(100))

# 训练选择器
selector.fit(X, y)
```

##### transform(X)
应用特征选择

**参数**:
- `X` (pd.DataFrame): 待转换的特征数据

**返回**:
- `pd.DataFrame`: 选择后的特征数据

**示例**:
```python
# 应用特征选择
selected_features = selector.transform(X)
print(f"原始特征数: {X.shape[1]}")
print(f"选择后特征数: {selected_features.shape[1]}")
```

##### fit_transform(X, y=None)
训练并应用特征选择

**参数**:
- `X` (pd.DataFrame): 特征数据
- `y` (pd.Series, optional): 目标变量

**返回**:
- `pd.DataFrame`: 选择后的特征数据

**示例**:
```python
# 一步完成训练和选择
selected_features = selector.fit_transform(X, y)
```

##### get_feature_importance()
获取特征重要性

**返回**:
- `pd.Series`: 特征重要性得分

**示例**:
```python
importance = selector.get_feature_importance()
print("特征重要性:")
for feature, score in importance.items():
    print(f"{feature}: {score:.4f}")
```

##### get_selected_features()
获取选择的特征列表

**返回**:
- `list`: 选择的特征名称列表

**示例**:
```python
selected_features = selector.get_selected_features()
print("选择的特征:", selected_features)
```

##### get_feature_scores()
获取特征得分

**返回**:
- `dict`: 特征得分字典

**示例**:
```python
scores = selector.get_feature_scores()
print("特征得分:", scores)
```

## 配置选项

### 重要性方法配置
```python
importance_config = {
    "method": "importance",
    "n_features": 10,           # 选择特征数量
    "threshold": 0.01,          # 重要性阈值
    "random_state": 42          # 随机种子
}
```

### 相关性方法配置
```python
correlation_config = {
    "method": "correlation",
    "threshold": 0.8,           # 相关性阈值
    "method": "pearson",        # 相关性计算方法
    "target_correlation": 0.1   # 与目标变量的相关性阈值
}
```

### 递归特征消除配置
```python
rfe_config = {
    "method": "rfe",
    "n_features": 10,           # 目标特征数量
    "step": 1,                  # 每次移除的特征数
    "estimator": "random_forest"  # 使用的估计器
}
```

### Lasso方法配置
```python
lasso_config = {
    "method": "lasso",
    "alpha": 0.01,              # L1正则化参数
    "max_iter": 1000,           # 最大迭代次数
    "tol": 1e-4                 # 收敛容差
}
```

## 使用示例

### 基础使用
```python
from src.features.processors.feature_selector import FeatureSelector
import pandas as pd
import numpy as np

# 准备示例数据
np.random.seed(42)
X = pd.DataFrame({
    'feature1': np.random.randn(100),
    'feature2': np.random.randn(100),
    'feature3': np.random.randn(100),
    'feature4': np.random.randn(100),
    'feature5': np.random.randn(100)
})
y = pd.Series(np.random.randn(100))

# 创建特征选择器
selector = FeatureSelector(method='importance', n_features=3)

# 训练和选择
selected_features = selector.fit_transform(X, y)

print("原始特征:", X.columns.tolist())
print("选择的特征:", selected_features.columns.tolist())
print("特征重要性:", selector.get_feature_importance())
```

### 高级使用
```python
# 使用不同方法进行特征选择
methods = ['importance', 'correlation', 'rfe', 'lasso']

for method in methods:
    print(f"\n使用 {method} 方法:")
    
    # 创建选择器
    selector = FeatureSelector(method=method, n_features=3)
    
    # 训练和选择
    selected_features = selector.fit_transform(X, y)
    
    print(f"选择的特征: {selected_features.columns.tolist()}")
    print(f"特征数量: {selected_features.shape[1]}")
    
    if hasattr(selector, 'get_feature_importance'):
        importance = selector.get_feature_importance()
        print("特征重要性:", importance.to_dict())
```

### 批量特征选择
```python
def batch_feature_selection(X, y, methods, n_features_list):
    """批量特征选择"""
    results = {}
    
    for method in methods:
        for n_features in n_features_list:
            key = f"{method}_{n_features}"
            
            try:
                selector = FeatureSelector(
                    method=method, 
                    n_features=n_features
                )
                
                selected_features = selector.fit_transform(X, y)
                
                results[key] = {
                    'features': selected_features.columns.tolist(),
                    'n_features': selected_features.shape[1],
                    'importance': selector.get_feature_importance().to_dict() if hasattr(selector, 'get_feature_importance') else {}
                }
                
            except Exception as e:
                print(f"方法 {key} 失败: {e}")
                results[key] = {'error': str(e)}
    
    return results

# 执行批量选择
methods = ['importance', 'correlation']
n_features_list = [2, 3, 4]

results = batch_feature_selection(X, y, methods, n_features_list)

# 显示结果
for key, result in results.items():
    print(f"\n{key}:")
    if 'error' not in result:
        print(f"  特征: {result['features']}")
        print(f"  数量: {result['n_features']}")
        if result['importance']:
            print(f"  重要性: {result['importance']}")
    else:
        print(f"  错误: {result['error']}")
```

### 交叉验证特征选择
```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

def cross_validate_feature_selection(X, y, method='importance', cv=5):
    """交叉验证特征选择"""
    scores = []
    feature_sets = []
    
    for i in range(cv):
        # 分割数据
        split_idx = len(X) // cv
        start_idx = i * split_idx
        end_idx = start_idx + split_idx
        
        X_train = X.iloc[:start_idx].append(X.iloc[end_idx:])
        y_train = y.iloc[:start_idx].append(y.iloc[end_idx:])
        X_test = X.iloc[start_idx:end_idx]
        y_test = y.iloc[start_idx:end_idx]
        
        # 特征选择
        selector = FeatureSelector(method=method, n_features=3)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        # 训练模型
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_selected, y_train)
        
        # 评估
        score = model.score(X_test_selected, y_test)
        scores.append(score)
        feature_sets.append(X_train_selected.columns.tolist())
    
    return {
        'scores': scores,
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'feature_sets': feature_sets
    }

# 执行交叉验证
cv_results = cross_validate_feature_selection(X, y, method='importance')

print("交叉验证结果:")
print(f"平均得分: {cv_results['mean_score']:.4f}")
print(f"标准差: {cv_results['std_score']:.4f}")
print("特征集:")
for i, features in enumerate(cv_results['feature_sets']):
    print(f"  折{i+1}: {features}")
```

## 性能优化建议

### 1. 数据预处理
- 确保输入数据格式正确
- 处理缺失值和异常值
- 标准化数值特征

### 2. 方法选择
- **重要性方法**: 适用于有监督学习，计算快速
- **相关性方法**: 适用于无监督学习，计算简单
- **RFE方法**: 适用于复杂模型，计算较慢但效果好
- **Lasso方法**: 适用于线性模型，有正则化效果

### 3. 参数调优
- 根据数据规模调整特征数量
- 根据业务需求调整阈值
- 使用交叉验证选择最佳参数

### 4. 内存管理
- 对于大数据集，使用增量学习
- 及时释放不需要的数据
- 使用适当的数据类型

## 故障排除

### 常见问题

#### 1. 数据格式错误
**问题**: `ValueError: 输入数据必须为DataFrame`
**解决方案**: 确保输入数据为pandas DataFrame格式

#### 2. 特征数量不足
**问题**: `ValueError: 请求的特征数量超过可用特征数`
**解决方案**: 调整n_features参数或增加特征

#### 3. 方法不支持
**问题**: `ValueError: 不支持的方法`
**解决方案**: 检查method参数，使用支持的方法

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
```

#### 2. 验证选择结果
```python
# 检查选择的特征
selected_features = selector.get_selected_features()
print("选择的特征:", selected_features)

# 检查特征重要性
importance = selector.get_feature_importance()
print("特征重要性:", importance)
```

#### 3. 性能分析
```python
import time

# 测量选择时间
start_time = time.time()
selected_features = selector.fit_transform(X, y)
end_time = time.time()

print(f"特征选择时间: {end_time - start_time:.3f}秒")
```

## 版本历史

### v1.0.0
- 初始版本
- 基础特征选择功能
- 支持重要性、相关性方法

### v1.1.0
- 添加RFE方法
- 添加Lasso方法
- 增强错误处理

### v1.2.0
- 添加交叉验证支持
- 优化性能
- 完善文档

---

**文档版本**: 1.2.0  
**最后更新**: 2025-01-27  
**维护者**: 开发团队 