# 🚀 Week 2 执行计划 - 三层提升至30%

**启动时间**: 2025-11-02  
**目标**: Strategy/Trading/Risk三层平均覆盖率达到30%  
**策略**: 创建真实集成测试，导入src/代码

---

## 🎯 Week 2目标

### 覆盖率提升目标

| 层级 | Week 1基线 | Week 2目标 | 提升幅度 | 新增测试 |
|------|-----------|-----------|---------|---------|
| Strategy | 7% | 25% | +18% | 50个 |
| Trading | 23% | 40% | +17% | 45个 |
| Risk | 4% | 25% | +21% | 55个 |
| **平均** | **9.2%** | **30%** | **+20.8%** | **150个** |

---

## 📋 Day-by-Day执行计划

### Day 1-2: Strategy层（7% → 25%）

**目标**: 新增50个测试，覆盖率提升18%

**重点测试区域**:
1. **BaseStrategy核心方法**（20测试）
   - 初始化和配置
   - 状态管理（start/stop/pause）
   - 参数管理
   - 信号生成接口

2. **StrategyFactory工厂**（15测试）
   - 策略创建
   - 策略注册
   - 策略获取

3. **具体策略类**（15测试）
   - MeanReversionStrategy
   - TrendFollowingStrategy
   - 策略执行逻辑

**预期成果**: Strategy层25%覆盖率

### Day 3-4: Trading层（23% → 40%）

**目标**: 新增45个测试，覆盖率提升17%

**重点测试区域**:
1. **OrderManager扩展**（20测试）
   - 订单生命周期管理
   - 订单状态跟踪
   - 订单取消和修改

2. **ExecutionEngine深化**（15测试）
   - 执行算法
   - 市场订单执行
   - 限价订单执行

3. **Portfolio管理**（10测试）
   - 持仓管理
   - 组合优化
   - 风险控制

**预期成果**: Trading层40%覆盖率

### Day 5-7: Risk层（4% → 25%）

**目标**: 新增55个测试，覆盖率提升21%

**重点测试区域**:
1. **RiskManager核心**（25测试）
   - 风险计算
   - 限额检查
   - 风险监控

2. **风险计算引擎**（20测试）
   - VaR计算
   - 波动率计算
   - 风险指标

3. **实时监控**（10测试）
   - 实时风险跟踪
   - 告警触发
   - 风险报告

**预期成果**: Risk层25%覆盖率

---

## 🔧 测试编写标准

### 必须遵守的原则

1. **真实导入src/代码**
```python
# ✅ 正确
from src.strategy.strategies.base_strategy import BaseStrategy
from src.trading.execution.order_manager import OrderManager
from src.risk.models.risk_manager import RiskManager

# ❌ 错误
import numpy as np  # 只测第三方库
```

2. **测试实际功能**
```python
# ✅ 正确
def test_base_strategy_initialization():
    strategy = BaseStrategy("test", "Test", "test")
    assert strategy.get_strategy_name() == "Test"

# ❌ 错误  
def test_simple_calculation():
    result = 1 + 1
    assert result == 2  # 不测试项目代码
```

3. **Mock外部依赖**
```python
@patch('src.strategy.data_source.get_data')
def test_strategy_with_data(mock_data):
    mock_data.return_value = test_data
    # 测试策略逻辑
```

4. **覆盖关键路径**
- 正常执行路径
- 边界条件
- 异常处理

---

## 📊 质量标准

### 每个测试文件必须
- ✅ 导入至少1个src/模块
- ✅ 测试至少5个方法/函数
- ✅ 100%测试通过
- ✅ 无linter错误
- ✅ 提升覆盖率≥5%

### Week 2整体必须
- ✅ 新增150个测试，全部通过
- ✅ 三层平均覆盖率≥30%
- ✅ 测试执行时间<5分钟
- ✅ 代码质量优秀

---

## 🎯 成功标准

### 必须达成
1. ✅ Strategy层≥25%
2. ✅ Trading层≥40%
3. ✅ Risk层≥25%
4. ✅ 三层平均≥30%
5. ✅ 所有新测试100%通过

### 期望达成
1. 🎯 Strategy层≥28%
2. 🎯 Trading层≥42%
3. 🎯 Risk层≥28%
4. 🎯  三层平均≥33%

---

## 📈 Week 2-7总览

| Week | 目标覆盖率 | 新增测试 | 累计覆盖率进展 |
|------|-----------|---------|--------------|
| Week 1 | 9.2% | 0 | 基线建立 ✅ |
| **Week 2** | **30%** | **150** | **+20.8%** |
| Week 3 | 45% | 120 | +15% |
| Week 4 | 55% | 100 | +10% |
| Week 5 | 65% | 100 | +10% |
| Week 6 | 73% | 80 | +8% |
| Week 7 | 80%+ | 70 | +7% |

---

## 🚀 立即开始

**Week 2 Day 1任务**: 创建Strategy层BaseStrategy核心测试

**第一个测试文件**: `test_base_strategy_core_coverage.py`
- 测试BaseStrategy初始化
- 测试状态管理方法
- 测试参数管理方法
- **目标**: Strategy层7% → 12%（+5%）

**现在开始创建！**


