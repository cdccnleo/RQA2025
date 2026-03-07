# 动态宇宙管理系统实践练习

## 练习概述

本练习材料包含6个实践模块，帮助用户深入理解和掌握动态宇宙管理系统的各项功能。

### 练习环境准备

```bash
# 1. 激活conda环境
conda activate test

# 2. 安装依赖
conda install pandas numpy scipy scikit-learn
pip install transformers seaborn backtrader

# 3. 验证环境
python -c "import pandas, numpy, scipy, sklearn; print('环境准备完成')"
```

## 练习1：系统基础配置和初始化

### 目标
- 掌握系统组件的创建和配置
- 理解配置参数的作用
- 验证系统初始化状态

### 操作步骤

#### 步骤1：创建基础配置文件
```python
# 创建配置文件
config = {
    'liquidity_threshold': 1500000,
    'volatility_threshold': 0.25,
    'fundamental_threshold': 0.6,
    'technical_threshold': 0.5,
    'performance_threshold': 0.1,
    'adjustment_sensitivity': 0.1,
    'min_weight': 0.1,
    'max_weight': 0.4
}
```

#### 步骤2：初始化系统组件
```python
from src.trading.universe.dynamic_universe_manager import DynamicUniverseManager
from src.trading.universe.intelligent_updater import IntelligentUniverseUpdater
from src.trading.universe.dynamic_weight_adjuster import DynamicWeightAdjuster

# 创建管理器
manager = DynamicUniverseManager(config)
updater = IntelligentUniverseUpdater(config)
adjuster = DynamicWeightAdjuster(config)

print("系统组件初始化完成")
```

## 练习2：数据处理和过滤功能

### 目标
- 掌握市场数据的准备和处理
- 理解多维过滤器的运作机制
- 验证过滤结果的准确性

### 操作步骤

#### 步骤1：准备测试数据
```python
import pandas as pd
import numpy as np

# 创建模拟股票数据
stocks = ['000001.SZ', '000002.SZ', '000858.SZ', '600000.SH', '600036.SH']
market_data = pd.DataFrame({
    'stock_code': stocks,
    'turnover': [1500000, 800000, 1200000, 2500000, 1800000],
    'volatility': [0.2, 0.35, 0.25, 0.15, 0.3],
    'pe_ratio': [15.2, 12.8, 18.5, 10.2, 22.1],
    'rsi': [65, 45, 70, 55, 80]
})
```

#### 步骤2：执行宇宙更新
```python
# 执行宇宙更新
updated_universe = manager.update_universe(market_data)

print(f"原始股票数量: {len(market_data)}")
print(f"过滤后股票数量: {len(updated_universe)}")
print(f"保留的股票: {list(updated_universe)}")
```

## 练习3：智能更新触发机制

### 目标
- 理解智能更新的触发条件
- 掌握不同市场状态下的更新策略
- 验证更新决策的准确性

### 操作步骤

#### 步骤1：测试不同市场状态
```python
# 定义测试场景
test_scenarios = [
    {
        'name': '牛市高收益',
        'market_state': 'bull',
        'performance': 0.15,
        'volatility': 0.25,
        'liquidity': 1500000
    },
    {
        'name': '熊市低收益',
        'market_state': 'bear',
        'performance': -0.08,
        'volatility': 0.45,
        'liquidity': 600000
    }
]

print("=== 智能更新测试 ===")
for scenario in test_scenarios:
    result = updater.should_update_universe(
        current_market_state=scenario['market_state'],
        current_performance=scenario['performance'],
        current_volatility=scenario['volatility'],
        current_liquidity=scenario['liquidity']
    )
    
    print(f"{scenario['name']}: 触发更新: {result.trigger}")
```

## 练习4：动态权重调整

### 目标
- 理解权重调整的机制
- 掌握不同市场条件下的调整策略
- 验证权重变化的合理性

### 操作步骤

#### 步骤1：测试权重调整
```python
# 测试不同市场条件下的权重调整
market_conditions = [
    ("bull", 0.15, "牛市高收益"),
    ("bear", -0.08, "熊市低收益"),
    ("sideways", 0.02, "震荡市微收益")
]

print("=== 权重调整测试 ===")
for state, perf, desc in market_conditions:
    adjusted = adjuster.adjust_weights(state, perf)
    print(f"{desc}:")
    for factor, weight in adjusted.items():
        print(f"  {factor}: {weight:.3f}")
```

## 练习5：系统集成测试

### 目标
- 验证系统各组件的协同工作
- 测试完整的工作流程
- 验证系统性能和稳定性

### 操作步骤

#### 步骤1：创建集成测试
```python
# 创建集成测试
def create_integrated_test():
    print("=== 集成测试开始 ===")
    
    # 准备测试数据
    test_stocks = ['000001.SZ', '000002.SZ', '000858.SZ', '600000.SH', '600036.SH']
    market_data = pd.DataFrame({
        'stock_code': test_stocks,
        'turnover': np.random.uniform(500000, 3000000, len(test_stocks)),
        'volatility': np.random.uniform(0.1, 0.5, len(test_stocks)),
        'pe_ratio': np.random.uniform(8, 35, len(test_stocks)),
        'rsi': np.random.uniform(20, 80, len(test_stocks))
    })
    
    # 执行宇宙更新
    updated_universe = manager.update_universe(market_data)
    print(f"宇宙大小: {len(updated_universe)}")
    
    # 检查是否需要更新
    should_update = updater.should_update_universe(
        current_market_state="bull",
        current_performance=0.1,
        current_volatility=0.3,
        current_liquidity=1000000
    )
    print(f"触发更新: {should_update.trigger}")
    
    # 调整权重
    adjusted_weights = adjuster.adjust_weights("bull", 0.1)
    print(f"权重调整: {dict(adjusted_weights)}")
    
    return {
        'universe_size': len(updated_universe),
        'update_triggered': should_update.trigger,
        'weights': adjusted_weights
    }

# 执行集成测试
result = create_integrated_test()
print(f"集成测试结果: {result}")
```

## 练习6：错误处理和异常情况

### 目标
- 掌握系统的错误处理机制
- 测试异常情况的处理能力
- 验证系统的健壮性

### 操作步骤

#### 步骤1：测试数据异常处理
```python
# 测试数据异常处理
def test_data_anomalies():
    print("=== 数据异常处理测试 ===")
    
    # 测试空数据
    print("\n1. 空数据处理:")
    try:
        empty_result = manager.update_universe(pd.DataFrame())
        print(f"  结果: {empty_result}")
    except Exception as e:
        print(f"  异常: {e}")
    
    # 测试缺失数据
    print("\n2. 缺失数据处理:")
    incomplete_data = market_data.copy()
    incomplete_data.loc[0, 'turnover'] = np.nan
    
    try:
        incomplete_result = manager.update_universe(incomplete_data)
        print(f"  结果: {len(incomplete_result)} 只股票")
    except Exception as e:
        print(f"  异常: {e}")

test_data_anomalies()
```

## 练习总结

### 完成情况检查
- [ ] 练习1：系统基础配置和初始化
- [ ] 练习2：数据处理和过滤功能
- [ ] 练习3：智能更新触发机制
- [ ] 练习4：动态权重调整
- [ ] 练习5：系统集成测试
- [ ] 练习6：错误处理和异常情况

### 技能掌握评估
- [ ] 能够独立配置系统参数
- [ ] 能够处理和分析市场数据
- [ ] 能够理解智能更新机制
- [ ] 能够调整和优化权重
- [ ] 能够进行系统集成测试
- [ ] 能够处理异常情况

### 后续学习建议
1. **深入理解算法**：研究过滤算法的具体实现
2. **性能优化**：学习如何优化系统性能
3. **扩展功能**：了解如何添加新的过滤维度
4. **实际应用**：在实际交易环境中应用系统
5. **持续监控**：学习如何监控和维护系统

---

*本练习材料将根据系统更新和用户反馈持续完善* 