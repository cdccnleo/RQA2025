# 修复策略优化模块导入问题 Spec

## Why
策略优化功能运行失败，后台线程中无法找到 `backtest_engine` 模块，导致参数优化无法执行。这是因为在后台线程中 Python 路径配置不正确，导致模块导入失败。

## What Changes
- 修复 `strategy_optimization_service.py` 中的模块导入路径
- 确保后台线程中正确设置 Python 路径
- 使用正确的绝对导入方式导入 `BacktestEngine` 和 `ParameterOptimizer`

## Impact
- Affected code: `src/gateway/web/strategy_optimization_service.py`
- Affected feature: 策略优化功能
- Affected system: 策略优化器页面

## ADDED Requirements
### Requirement: 模块导入修复
The system SHALL correctly import `BacktestEngine` and `ParameterOptimizer` in the background thread.

#### Scenario: 策略优化启动
- **WHEN** 用户启动策略优化
- **THEN** 后台线程应能正确导入回测引擎和优化器模块
- **AND** 参数优化应正常执行

#### Scenario: 模块路径配置
- **WHEN** 后台线程启动时
- **THEN** 应自动配置正确的 Python 路径
- **AND** 应能解析 `src/strategy/backtest/` 目录下的模块

## MODIFIED Requirements
### Requirement: 后台线程导入逻辑
The background thread SHALL use absolute imports with proper path configuration.

**Current behavior**: 
- 使用 `from strategy.backtest.backtest_engine import BacktestEngine` 导入失败
- 后台线程中 Python 路径不包含 `/app`

**Expected behavior**:
- 后台线程自动添加 `/app` 到 Python 路径
- 使用正确的导入语句成功加载模块
