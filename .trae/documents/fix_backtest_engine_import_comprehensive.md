# 全面修复 backtest_engine 导入问题计划

## 任务目标
参考策略服务层架构设计文档，深入全面检查并彻底解决 `backtest_engine` 导入问题。

## 架构设计参考
根据 `docs/architecture/strategy_layer_architecture_design.md`：
- 回测引擎位于 `src/strategy/backtest/backtest_engine.py`
- 参数优化器位于 `src/strategy/backtest/parameter_optimizer.py`
- 策略层设计原则：**回测驱动**、**模块化扩展**、**性能优化**

## 当前问题分析

### 已尝试的修复方案
1. ✗ 直接导入 `from src.strategy.backtest.backtest_engine import BacktestEngine` - 失败
2. ✗ 使用 `strategy.backtest.backtest_engine` - 失败
3. ✗ 使用 `src.strategy.backtest.backtest_engine` - 失败
4. ✗ 使用 `importlib.import_module` - 可能仍有问题

### 根本原因分析
1. **后台线程环境差异**：后台线程的 Python 环境与主线程不同
2. **模块缓存问题**：Python 的模块缓存机制可能导致导入失败
3. **包初始化问题**：`__init__.py` 文件可能未正确执行
4. **循环导入**：可能存在循环导入问题

## 全面检查步骤

### 第一阶段：深度环境诊断（15分钟）
1. 检查容器内完整的 Python 环境
2. 检查 `sys.modules` 中是否存在相关模块
3. 检查 `src` 包和 `strategy` 包的初始化状态
4. 验证 `backtest_engine.py` 文件内容和语法
5. 检查是否存在循环导入

### 第二阶段：架构级修复（20分钟）
1. 参考架构设计文档，确保包结构符合规范
2. 修复所有 `__init__.py` 文件，确保正确导出
3. 实现预导入机制，在主线程中预先加载模块
4. 添加模块加载重试机制
5. 实现备选导入方案（多种方式尝试）

### 第三阶段：健壮性增强（15分钟）
1. 添加详细的导入日志和错误追踪
2. 实现模块存在性检查
3. 添加自动修复机制
4. 优化后台线程的 Python 环境

### 第四阶段：验证测试（10分钟）
1. 在容器内测试所有导入方式
2. 验证策略优化功能正常
3. 检查日志确认无错误
4. 进行端到端测试

## 具体修复方案

### 方案一：预导入机制（推荐）
在 `strategy_optimization_service.py` 模块级别预先导入，后台线程直接使用：
```python
# 模块级别预导入
try:
    from src.strategy.backtest.backtest_engine import BacktestEngine
    from src.strategy.backtest.parameter_optimizer import ParameterOptimizer
    _IMPORT_SUCCESS = True
except ImportError:
    _IMPORT_SUCCESS = False

# 后台线程中直接使用
if _IMPORT_SUCCESS:
    engine = BacktestEngine()
```

### 方案二：强制重新加载
清除模块缓存后重新导入：
```python
import sys
import importlib

# 清除可能损坏的模块缓存
modules_to_remove = [k for k in sys.modules.keys() if 'backtest' in k]
for mod in modules_to_remove:
    del sys.modules[mod]

# 重新导入
from src.strategy.backtest.backtest_engine import BacktestEngine
```

### 方案三：子进程隔离
将参数优化放在独立子进程中执行，避免后台线程环境问题：
```python
import subprocess
import json

# 通过子进程执行优化
result = subprocess.run(
    ['python', '-m', 'src.strategy.backtest.parameter_optimizer'],
    input=json.dumps(config),
    capture_output=True,
    text=True
)
```

### 方案四：动态路径修复
在导入前动态修复 Python 路径：
```python
import sys
import os

# 添加所有可能的路径
paths_to_add = [
    '/app',
    '/app/src',
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
]
for path in paths_to_add:
    if path not in sys.path:
        sys.path.insert(0, path)

# 然后导入
from strategy.backtest.backtest_engine import BacktestEngine
```

## 预期结果
- 策略优化服务正常运行
- 无 `No module named 'backtest_engine'` 错误
- 参数优化功能可用
- 符合策略层架构设计规范

## 风险与缓解
| 风险 | 缓解措施 |
|------|----------|
| 架构冲突 | 严格遵循架构设计文档 |
| 性能下降 | 使用预导入机制避免重复加载 |
| 兼容性问题 | 保留多种备选方案 |
| 循环导入 | 使用延迟导入或重构代码 |
