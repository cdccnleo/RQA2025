# 策略优化模块导入问题最终修复计划

## 任务目标
彻底解决策略优化运行失败的问题，修复 `No module named 'backtest_engine'` 错误。

## 当前状态
- 已多次尝试修复导入路径，但问题依然存在
- 需要深入分析 Python 模块导入机制
- 需要找到根本解决方案

## 问题分析

### 可能的原因
1. **Python 路径问题**：后台线程的 `sys.path` 不包含正确的路径
2. **模块命名冲突**：存在命名冲突或循环导入
3. **包结构问题**：`__init__.py` 文件缺失或配置错误
4. **导入时机问题**：模块在后台线程启动时才尝试导入，但环境未准备好
5. **Docker 环境问题**：容器内文件路径与预期不符

### 需要验证的内容
- [ ] 检查容器内 `/app/src/strategy/backtest/` 目录是否存在
- [ ] 检查 `backtest_engine.py` 文件是否在正确位置
- [ ] 检查 `__init__.py` 文件是否正确配置
- [ ] 检查后台线程的 `sys.path` 内容
- [ ] 尝试使用不同的导入方式（相对导入、绝对导入、动态导入）

## 检查步骤

### 第一阶段：环境检查（10分钟）
1. 进入容器检查文件系统结构
2. 验证 `backtest_engine.py` 文件位置
3. 检查 `sys.path` 在后台线程中的值
4. 尝试手动导入测试

### 第二阶段：导入机制分析（10分钟）
1. 分析 Python 模块导入机制
2. 检查 `strategy` 包的 `__init__.py` 配置
3. 检查 `backtest` 子包的 `__init__.py` 配置
4. 确定最佳导入方式

### 第三阶段：实现修复（15分钟）
1. 根据分析结果选择正确的导入方式
2. 修改 `strategy_optimization_service.py`
3. 添加详细的错误日志和调试信息
4. 实现备选导入方案

### 第四阶段：测试验证（10分钟）
1. 重新构建并部署容器
2. 检查日志确认问题已解决
3. 验证策略优化功能正常

## 可能的解决方案

### 方案一：动态导入
```python
import importlib
import sys
sys.path.insert(0, '/app')
backtest_engine = importlib.import_module('src.strategy.backtest.backtest_engine')
BacktestEngine = backtest_engine.BacktestEngine
```

### 方案二：使用 __import__
```python
import sys
sys.path.insert(0, '/app')
BacktestEngine = __import__('src.strategy.backtest.backtest_engine', fromlist=['BacktestEngine']).BacktestEngine
```

### 方案三：预导入机制
在主线程中先导入模块，后台线程直接使用

### 方案四：修复包结构
确保所有 `__init__.py` 文件正确配置

## 预期结果
- 策略优化服务正常运行
- 无模块导入错误
- 参数优化功能可用

## 风险与缓解
| 风险 | 缓解措施 |
|------|----------|
| 导入机制复杂 | 添加详细的日志和调试信息 |
| 环境问题 | 在容器内直接测试导入 |
| 兼容性问题 | 测试多种导入方式 |
