# 修复 BacktestEngine 缺少 add_strategy 方法 Spec

## Why
策略优化运行失败，错误信息显示 `'BacktestEngine' object has no attribute 'add_strategy'`。这表明 `BacktestEngine` 类缺少 `add_strategy` 方法，导致参数优化器无法正常执行。

## What Changes
- 在 `BacktestEngine` 类中添加 `add_strategy` 方法
- 确保参数优化器可以正确调用该方法
- 保持与现有代码的兼容性

## Impact
- Affected code: `src/strategy/backtest/backtest_engine.py`
- Affected feature: 策略优化功能
- Affected system: 策略优化器页面

## ADDED Requirements
### Requirement: add_strategy 方法
The BacktestEngine class SHALL provide an `add_strategy` method for adding strategies to the engine.

#### Scenario: 策略优化执行
- **WHEN** 参数优化器调用 `engine.add_strategy(strategy)`
- **THEN** 策略应被成功添加到回测引擎
- **AND** 参数优化应正常执行

#### Scenario: 方法签名
- **GIVEN** BacktestEngine 实例
- **WHEN** 调用 `add_strategy(strategy)`
- **THEN** 方法应接受策略对象作为参数
- **AND** 返回成功状态或 None

## MODIFIED Requirements
### Requirement: BacktestEngine 类接口
The BacktestEngine class SHALL support strategy management through the `add_strategy` method.

**Current behavior**: 
- `BacktestEngine` 缺少 `add_strategy` 方法
- 参数优化器调用时抛出 `AttributeError`

**Expected behavior**:
- `BacktestEngine` 提供 `add_strategy(strategy)` 方法
- 参数优化器可以正常添加策略并执行优化
