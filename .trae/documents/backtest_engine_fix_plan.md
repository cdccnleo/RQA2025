# 回测引擎修复计划

## 问题分析

### 错误信息
```
'BacktestEngine' object has no attribute 'run_backtest_with_signals'
```

### 问题位置
- **调用方**: `src/gateway/web/backtest_service.py` 第1266行
- **被调用方**: `src/strategy/backtest/backtest_engine.py` 中的 `BacktestEngine` 类

### 根本原因
`BacktestEngine` 类缺少 `run_backtest_with_signals` 方法，但 `backtest_service.py` 中的 `run_model_driven_backtest` 函数尝试调用它。

## 架构设计参考

根据 `docs/architecture/strategy_layer_architecture_design.md`：
- 策略回测模块 (`src/strategy/backtest/`) 是策略层的核心组件
- `BacktestEngine` 是回测引擎的核心类
- 支持多种回测模式：SINGLE、MULTI、OPTIMIZE

## 修复方案

### 方案：在 BacktestEngine 类中添加 run_backtest_with_signals 方法

该方法需要：
1. 接收历史数据、交易信号和配置参数
2. 基于信号模拟交易执行
3. 计算回测指标（收益率、夏普比率、最大回撤等）
4. 返回回测结果字典

### 实现细节

```python
def run_backtest_with_signals(
    self,
    data: pd.DataFrame,
    signals: List[str],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    基于交易信号运行回测
    
    Args:
        data: 历史数据DataFrame
        signals: 交易信号列表 ['buy', 'sell', 'hold', ...]
        config: 回测配置
        
    Returns:
        回测结果字典
    """
    # 实现逻辑：
    # 1. 解析配置参数（初始资金、手续费率、滑点等）
    # 2. 遍历数据和信号，模拟交易执行
    # 3. 计算资金曲线、交易记录
    # 4. 计算绩效指标
    # 5. 返回结果字典
```

## 实施步骤

### 第一阶段：添加 run_backtest_with_signals 方法（30分钟）
1. 在 `BacktestEngine` 类中添加新方法
2. 实现基于信号的交易模拟逻辑
3. 实现绩效指标计算

### 第二阶段：测试验证（20分钟）
1. 验证方法可以被正常调用
2. 验证回测结果格式正确
3. 验证指标计算准确

## 验收标准

- [ ] `run_backtest_with_signals` 方法成功添加到 `BacktestEngine` 类
- [ ] 方法能够正确处理买入/卖出/持有信号
- [ ] 方法返回完整的回测结果（资金曲线、交易记录、绩效指标）
- [ ] 策略回测功能正常运行，不再报错

## 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 信号处理逻辑错误 | 回测结果不准确 | 添加详细的日志记录和边界条件检查 |
| 性能问题 | 大数据量回测慢 | 使用向量化计算，避免循环 |
| 数据格式不兼容 | 运行时错误 | 添加数据验证和错误处理 |
