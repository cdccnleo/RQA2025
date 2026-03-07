# 修复 BacktestEngine 缺少 get_performance 方法计划

## 任务目标
修复策略优化运行失败的问题，错误信息：`'BacktestEngine' object has no attribute 'get_performance'`。

## 当前问题
策略优化器调用 `engine.get_performance()` 方法时失败，因为 `BacktestEngine` 类缺少该方法。

## 问题分析

### 错误信息
```
src.gateway.web.strategy_optimization_service - ERROR - 参数优化执行失败: 'BacktestEngine' object has no attribute 'get_performance'
```

### 可能的原因
1. `BacktestEngine` 类确实缺少 `get_performance` 方法
2. 方法名称可能不同（如 `get_results`、`performance` 等）
3. 参数优化器使用了错误的方法名

### 需要检查的内容
- [ ] 检查 `BacktestEngine` 类的所有方法
- [ ] 检查参数优化器如何使用 `get_performance`
- [ ] 确定正确的方法名或实现缺失的方法

## 检查步骤

### 第一阶段：分析方法调用（5分钟）
1. 检查参数优化器中 `get_performance` 的使用位置
2. 查看调用上下文和期望的返回值
3. 确定方法应该提供的功能

### 第二阶段：检查 BacktestEngine 类（5分钟）
1. 列出 `BacktestEngine` 类的所有方法
2. 查找类似功能的方法（如 `get_results`、`performance` 等）
3. 确定是添加新方法还是修改调用

### 第三阶段：实现修复（10分钟）
1. 在 `BacktestEngine` 类中添加 `get_performance` 方法
2. 或修改参数优化器使用正确的方法名
3. 确保返回值符合期望格式

### 第四阶段：测试验证（5分钟）
1. 重新构建 Docker 容器
2. 运行单元测试验证修复
3. 检查策略优化功能正常

## 可能的解决方案

### 方案一：添加 get_performance 方法
```python
def get_performance(self):
    """获取回测性能指标"""
    return self.results.get('performance', {})
```

### 方案二：修改参数优化器
如果 `BacktestEngine` 已有类似方法，修改参数优化器使用正确的方法名。

## 预期结果
- 策略优化服务正常运行
- 无 `'BacktestEngine' object has no attribute 'get_performance'` 错误
- 参数优化功能可用

## 风险与缓解
| 风险 | 缓解措施 |
|------|----------|
| 方法签名不匹配 | 仔细检查参数优化器的调用方式 |
| 返回值格式问题 | 确保返回值符合期望格式 |
