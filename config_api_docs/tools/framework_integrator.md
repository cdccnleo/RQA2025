# framework_integrator

**文件路径**: `tools\framework_integrator.py`

## 模块描述

性能测试框架集成器
将测试优化器与现有性能测试框架无缝集成

## 导入语句

```python
from infrastructure.config.core.imports import time
from infrastructure.config.core.imports import logging
from infrastructure.config.core.imports import Dict
from infrastructure.config.core.imports import Any
from infrastructure.config.core.imports import Optional
from infrastructure.config.core.imports import Callable
from infrastructure.config.core.imports import TYPE_CHECKING
from contextlib import contextmanager
from infrastructure.config.tools.test_optimizer import TestOptimizer
from infrastructure.config.tools.test_optimizer import TestOptimizationConfig
# ... 等4个导入
```

## 类

### PerformanceFrameworkIntegrator

framework_integrator - 配置管理

职责说明：
负责系统配置的统一管理、配置文件的读取、配置验证和配置分发

核心职责：
- 配置文件的读取和解析
- 配置参数的验证
- 配置的热重载
- 配置的分发和同步
- 环境变量管理
- 配置加密和安全

相关接口：
- IConfigComponent
- IConfigManager
- IConfigValidator

**方法**:

- `__init__`
- `integrate_benchmark_framework`
- `integrate_performance_dashboard`
- `setup_test_environment`
- `cleanup_test_environment`
- ... 等10个方法

## 函数

### get_framework_integrator

获取全局框架集成器实例

**返回值**: `PerformanceFrameworkIntegrator`

### integrate_performance_framework

集成性能测试框架

**参数**:

- `benchmark_framework: <ast.Subscript object at 0x0000022590260B20>`
- `performance_dashboard: <ast.Subscript object at 0x0000022590260C40>`

### quick_integration

快速集成

### run_optimized_test

快速运行优化的测试

**参数**:

- `test_name: str`
- `test_func: Callable`
- `test_mode: TestMode`
- `**kwargs`

### run_optimized_benchmark_test

快速运行优化的基准测试

**参数**:

- `benchmark_name: str`
- `test_func: Callable`
- `test_mode: TestMode`
- `**kwargs`

