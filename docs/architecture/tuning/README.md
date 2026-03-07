# 调优层（tuning）架构设计说明

## 1. 模块定位
tuning模块为RQA2025系统提供高效、灵活的超参数搜索、调优、早停、可视化等能力，是主流程的参数优化与性能提升核心。

## 2. 主要子系统
- **超参数搜索与优化**：BaseTuner、OptunaTuner、MultiObjectiveTuner 支持网格搜索、随机搜索、贝叶斯优化、TPE、CMA-ES等多种搜索方法。
- **单目标与多目标调优**：OptunaTuner、MultiObjectiveTuner 支持单目标和多目标调优。
- **早停机制**：EarlyStopping 支持训练过程中的自动早停，防止过拟合。
- **调优结果与可视化**：TuningResult 结构化输出最优参数、性能、重要性、历史；TuningVisualizer 支持优化历史、参数重要性、平行坐标等可视化。
- **搜索方法与优化方向**：SearchMethod、ObjectiveDirection 枚举，统一调优策略。

## 3. 典型用法
### 单目标调优
```python
from src.tuning.optimizers.optuna_tuner import OptunaTuner
from src.tuning.optimizers.base import ObjectiveDirection

def objective(params):
    ... # 返回分数
param_space = {...}
tuner = OptunaTuner()
result = tuner.tune(objective, param_space, n_trials=50, direction=ObjectiveDirection.MAXIMIZE)
```

### 多目标调优
```python
from src.tuning.optimizers.optuna_tuner import MultiObjectiveTuner
objectives = [(obj1, ObjectiveDirection.MAXIMIZE), (obj2, ObjectiveDirection.MINIMIZE)]
tuner = MultiObjectiveTuner(objectives)
results = tuner.tune(param_space, n_trials=50)
```

### 早停与可视化
```python
from src.tuning.evaluators.early_stopping import EarlyStopping
from src.tuning.utils.visualization import TuningVisualizer
es = EarlyStopping(patience=5)
for score in scores:
    if es(score): break
TuningVisualizer.plot_optimization_history(result.trials)
```

## 4. 在主流程中的地位
- 为models/features/trading等提供最优参数配置，提升系统性能和泛化能力。
- 支持多种调优策略、早停机制、可视化分析，保障主流程的灵活性和高效性。
- 接口抽象与注册机制，便于扩展新调优方法、适配新场景、Mock测试等。

## 5. 测试与质量保障
- 建议实现高质量pytest单元测试，覆盖调优主流程、早停、可视化等主要功能和边界。
- 测试用例建议存放于：tests/unit/tuning/ 目录下。 