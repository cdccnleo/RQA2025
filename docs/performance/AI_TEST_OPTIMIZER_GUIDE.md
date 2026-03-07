# AI测试优化器使用指南

## 🎯 概述

AI测试优化器是一个基于机器学习的智能测试优化系统，旨在通过分析历史测试数据和执行模式，为测试团队提供智能化的测试用例选择、执行时间预测、故障预测和资源优化分配。

## 🏗️ 系统架构

### 核心组件

```
AI测试优化器
├── 特征工程器 (FeatureEngineer)
├── 执行时间预测器 (ExecutionTimePredictor)
├── 故障预测器 (FailurePredictor)
├── 测试用例选择器 (TestCaseSelector)
├── 资源优化分配器 (ResourceOptimizer)
└── 主优化器 (AITestOptimizer)
```

### 技术栈

- **机器学习**: scikit-learn (RandomForest, LinearRegression, LogisticRegression)
- **数据处理**: NumPy, Pandas
- **特征工程**: 标准化、分类编码、特征提取
- **并发处理**: 多线程、线程安全
- **模型持久化**: Pickle序列化

## 🚀 快速开始

### 1. 基本使用

```python
from src.infrastructure.performance import create_ai_optimizer, TestCase, TestExecutionRecord
from datetime import datetime

# 创建AI优化器
optimizer = create_ai_optimizer()

# 添加测试用例
test_case = TestCase(
    id="test_001",
    name="登录测试",
    module="auth",
    category="integration",
    priority=9,
    complexity=7.0,
    estimated_time=15.0,
    dependencies=["db_setup"],
    tags=["login", "auth"],
    created_at=datetime.now()
)
optimizer.add_test_case(test_case)

# 添加执行记录
record = TestExecutionRecord(
    test_id="test_001",
    execution_id="exec_001",
    start_time=datetime.now(),
    end_time=datetime.now(),
    duration=18.5,
    status="success"
)
optimizer.add_execution_record(record)

# 执行优化
results = optimizer.optimize_test_execution(max_count=10, strategy="hybrid")
```

### 2. 配置优化器

```python
from src.infrastructure.performance import OptimizationConfig, ModelType

config = OptimizationConfig(
    model_type=ModelType.RANDOM_FOREST,
    auto_retrain=True,
    retrain_threshold=100,
    confidence_threshold=0.8,
    max_features=50
)

optimizer = create_ai_optimizer(config)
```

## 📊 核心功能

### 1. 智能测试用例选择

#### 选择策略

- **风险基础选择 (risk_based)**: 基于故障概率、优先级和复杂度
- **时间基础选择 (time_based)**: 基于执行时间效率和置信度
- **混合策略 (hybrid)**: 结合风险和时间的平衡策略
- **自适应策略 (adaptive)**: 根据上下文动态调整

```python
# 使用不同策略选择测试用例
results = optimizer.optimize_test_execution(
    max_count=50,
    strategy="risk_based",  # 或 "time_based", "hybrid", "adaptive"
    context={"quality_focus": True}
)
```

#### 上下文配置

```python
context = {
    "time_constraint": True,      # 时间紧张时优先选择高效测试
    "quality_focus": True,        # 质量优先时选择高风险测试
    "available_resources": {      # 可用资源约束
        "cpu": 0.8,
        "memory": 0.6,
        "disk": 0.9
    }
}
```

### 2. 执行时间预测

#### 预测模型

- **随机森林回归器**: 默认模型，适合复杂特征关系
- **线性回归**: 适合线性特征关系
- **自动特征工程**: 提取50维特征向量

```python
# 预测单个测试用例执行时间
predicted_time, confidence = optimizer.execution_predictor.predict(test_case)

print(f"预测执行时间: {predicted_time:.2f}秒")
print(f"预测置信度: {confidence:.2f}")
```

#### 特征说明

```
特征向量 (50维):
├── 基础特征 (9维): 优先级、复杂度、预估时间、执行次数等
├── 时间特征 (1维): 距离上次执行的天数
├── 成功率特征 (1维): 历史成功率
├── 稳定性特征 (1维): 执行时间稳定性
├── 模块特征 (10维): 模块one-hot编码
├── 类别特征 (10维): 类别one-hot编码
└── 执行记录特征 (6维): 持续时间、状态、系统资源等
```

### 3. 故障预测

#### 预测模型

- **随机森林分类器**: 默认模型，适合复杂特征关系
- **逻辑回归**: 适合线性特征关系
- **历史数据回退**: 无模型时基于历史故障率

```python
# 预测故障概率
failure_prob = optimizer.failure_predictor.predict_failure_probability(test_case)

print(f"故障概率: {failure_prob:.2%}")
if failure_prob > 0.5:
    print("高风险测试用例，建议优先执行")
```

### 4. 资源优化分配

#### 资源配置文件

```python
resource_profiles = {
    "light": {"cpu": 0.1, "memory": 0.1, "disk": 0.1},      # 轻量级测试
    "medium": {"cpu": 0.3, "memory": 0.3, "disk": 0.3},     # 中等测试
    "heavy": {"cpu": 0.6, "memory": 0.6, "disk": 0.6}       # 重量级测试
}
```

#### 智能分配

- 根据测试复杂度自动选择配置
- 考虑执行时间和依赖关系调整资源
- 资源不足时按优先级调整分配

## ⚙️ 配置选项

### 模型配置

```yaml
model:
  type: "random_forest"  # 模型类型
  random_forest:
    n_estimators: 100    # 树的数量
    max_depth: 10        # 最大深度
    min_samples_split: 2 # 分裂最小样本数
```

### 特征工程配置

```yaml
feature_engineering:
  max_features: 50       # 最大特征数量
  feature_weights:       # 特征权重
    priority: 0.3
    complexity: 0.2
    execution_count: 0.15
```

### 优化策略配置

```yaml
optimization:
  default_strategy: "hybrid"  # 默认选择策略
  risk_based:
    failure_probability: 0.5  # 故障概率权重
    priority: 0.3            # 优先级权重
    complexity: 0.2          # 复杂度权重
```

## 🔧 高级功能

### 1. 模型管理

#### 自动重新训练

```python
# 启用自动重新训练
config = OptimizationConfig(
    auto_retrain=True,
    retrain_threshold=100  # 每100条记录重新训练
)
```

#### 模型持久化

```python
# 保存模型
success = optimizer.save_models("models/my_optimizer/")

# 加载模型
success = optimizer.load_models("models/my_optimizer/")
```

### 2. 性能监控

#### 统计信息

```python
stats = optimizer.get_optimization_stats()
print(f"测试用例总数: {stats['total_test_cases']}")
print(f"执行记录总数: {stats['total_execution_records']}")
print(f"模型训练状态: {stats['models_trained']}")
print(f"机器学习库可用: {stats['ml_available']}")
```

#### 缓存管理

```python
config = OptimizationConfig(
    prediction_cache_size=1000,  # 预测缓存大小
    cache_cleanup_threshold=0.8  # 清理阈值
)
```

### 3. 后台任务

- **自动优化检查**: 每5分钟检查一次
- **模型性能监控**: 每30分钟检查一次
- **缓存清理**: 每10分钟清理一次
- **统计信息更新**: 每分钟更新一次

## 📈 性能指标

### 预测准确性

- **执行时间预测**: RMSE (均方根误差)
- **故障预测**: 准确率、精确率、召回率、F1分数

### 系统性能

- **响应时间**: 预测 < 100ms, 优化 < 1s
- **内存使用**: < 100MB (基础配置)
- **并发支持**: 支持多线程并发访问

## 🚨 故障排除

### 常见问题

#### 1. 机器学习库不可用

```python
# 检查ML_AVAILABLE状态
if not ML_AVAILABLE:
    print("请安装scikit-learn: pip install scikit-learn")
```

#### 2. 训练数据不足

```python
# 检查训练数据量
if len(execution_records) < 10:
    print("需要至少10条执行记录进行训练")
```

#### 3. 特征向量长度不一致

```python
# 系统会自动处理特征长度不一致问题
# 使用最短长度截断所有特征
```

### 调试模式

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 查看详细日志信息
optimizer = create_ai_optimizer()
```

## 🔮 未来规划

### 短期优化 (1-2个月)

- **模型性能提升**: 集成更多算法，优化超参数
- **特征工程增强**: 支持更多特征类型和编码方式
- **实时学习**: 支持在线学习和增量训练

### 中期发展 (3-6个月)

- **深度学习集成**: 支持神经网络模型
- **自动化调优**: 自动超参数优化和模型选择
- **分布式训练**: 支持大规模数据训练

### 长期愿景 (6个月以上)

- **多模态学习**: 集成代码、日志、性能数据
- **智能推荐**: 基于业务场景的测试策略推荐
- **预测分析**: 趋势预测和异常检测

## 📚 参考资源

### 相关文档

- [性能测试框架指南](./PERFORMANCE_FRAMEWORK_GUIDE.md)
- [监控告警系统指南](./MONITORING_ALERT_SYSTEM_GUIDE.md)
- [Web管理界面指南](./WEB_MANAGEMENT_INTERFACE_GUIDE.md)

### 技术参考

- [scikit-learn官方文档](https://scikit-learn.org/)
- [机器学习最佳实践](https://developers.google.com/machine-learning/guides)
- [测试优化策略](https://martinfowler.com/articles/microservices-testing/)

---

*本指南涵盖了AI测试优化器的所有核心功能和使用方法。如有问题，请参考故障排除部分或联系开发团队。*
