# 修复策略执行监控页面暂停操作和增加退市功能 Spec

## Why
策略执行监控页面（strategy-execution-monitor）中策略 model_strategy_1771503574 的暂停操作无响应，用户无法停止正在运行的策略。此外，用户需要在策略执行列表中直接进行退市操作，而不必切换到策略生命周期管理页面。

## What Changes
- 修复策略执行监控页面的暂停/恢复操作无响应问题
- 在策略执行列表中增加退市策略操作按钮
- 确保暂停/恢复操作正确调用后端 API
- 确保退市操作正确停止策略并从执行列表中移除

## Impact
- Affected code: 
  - `web-static/strategy-execution-monitor.html`
  - `src/gateway/web/strategy_execution_routes.py`
  - `src/gateway/web/strategy_execution_service.py`
- Affected feature: 策略执行监控功能
- Affected system: 策略执行监控页面

## ADDED Requirements
### Requirement: 策略执行列表退市操作
The system SHALL provide a retire button in the strategy execution list for retiring running strategies.

#### Scenario: 从执行列表退市策略
- **GIVEN** 用户在策略执行监控页面
- **WHEN** 点击策略的"退市"按钮
- **THEN** 系统应先停止策略执行
- **AND** 将策略生命周期状态设置为 archived
- **AND** 从执行列表中移除该策略
- **AND** 显示成功提示

#### Scenario: 退市确认对话框
- **GIVEN** 用户点击退市按钮
- **WHEN** 系统显示确认对话框
- **THEN** 应提示"确定要退市此策略吗？此操作不可逆。"
- **AND** 用户确认后才执行退市操作

## MODIFIED Requirements
### Requirement: 策略暂停/恢复操作
The strategy execution monitor page SHALL correctly handle pause/resume operations.

**Current behavior**: 
- 点击暂停/恢复按钮无响应
- 操作可能未正确调用后端 API
- 状态更新后页面未正确刷新

**Expected behavior**:
- 点击暂停按钮后，策略状态变为 paused
- 点击恢复按钮后，策略状态变为 running
- 操作成功后页面自动刷新显示最新状态
- 显示操作成功或失败的提示信息

#### Scenario: 暂停策略
- **GIVEN** 策略正在运行
- **WHEN** 用户点击暂停按钮
- **THEN** 调用 `/strategy/execution/{id}/pause` API
- **AND** 策略状态更新为 paused
- **AND** 页面刷新显示新状态

#### Scenario: 恢复策略
- **GIVEN** 策略已暂停
- **WHEN** 用户点击恢复按钮
- **THEN** 调用 `/strategy/execution/{id}/start` API
- **AND** 策略状态更新为 running
- **AND** 页面刷新显示新状态

## REMOVED Requirements
None
