# 集成层（ensemble）架构设计说明

## 1. 模块定位
ensemble模块为RQA2025系统提供多模型集成预测、优化、监控、可视化等能力，提升模型泛化能力和鲁棒性，是主流程的模型融合与优化核心。

## 2. 主要子系统
- **模型集成预测**：EnsemblePredictor、ModelEnsemble 支持加权平均、简单平均、堆叠、贝叶斯平均、动态加权等多种集成方法。
- **集成优化与权重更新**：WeightedEnsemble、DynamicWeightedEnsemble等，支持基于性能、等权、衰减等权重更新规则。
- **集成结果与不确定性分析**：EnsembleResult 输出集成预测、权重分布、不确定性、各模型贡献等。
- **集成监控与可视化**：EnsembleMonitor、EnsembleVisualizer 支持性能监控、相关性分析、权重/不确定性/贡献可视化。

## 3. 典型用法
### 加权平均集成
```python
from src.ensemble.ensemble_predictor import EnsemblePredictor
ensemble = EnsemblePredictor([model1, model2], weights=[0.7, 0.3])
pred = ensemble.predict(X)
```

### 堆叠集成
```python
from src.ensemble.ensemble_predictor import StackingEnsemble
stack = StackingEnsemble({'m1': model1, 'm2': model2}, meta_model=meta)
stack.fit(X, y)
result = stack.predict(X)
```

### 集成监控与可视化
```python
from src.ensemble.ensemble_predictor import EnsembleMonitor, EnsembleVisualizer
monitor = EnsembleMonitor(['m1', 'm2'])
monitor.update(predictions, y, ensemble_pred)
fig = EnsembleVisualizer.plot_weight_distribution(weights)
```

## 4. 在主流程中的地位
- 为models/trading等提供更优决策基础，提升模型泛化能力和鲁棒性。
- 支持多模型融合、动态权重调整、性能监控，保障主流程的灵活性和高性能。
- 接口抽象与注册机制，便于扩展新集成方法、适配新场景、Mock测试等。

## 5. 测试与质量保障
- 已实现高质量pytest单元测试，覆盖集成预测、优化、监控、可视化等主要功能和边界。
- 测试用例见：tests/unit/ensemble/ 目录下相关文件。 