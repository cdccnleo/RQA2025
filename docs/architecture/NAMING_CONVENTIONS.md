# RQA2025 命名规范

## 📋 概述

本文档定义了RQA2025项目中文件、类、方法和变量的命名规范，旨在提高代码可读性、可维护性和一致性。

## 🗂️ 文件命名规范

### 1. Python 文件命名

#### 核心业务文件
```
{domain}_{component}_{type}.py
```
- `domain`: 业务领域 (strategy, trading, risk, data, ml等)
- `component`: 组件名称 (manager, engine, service等)
- `type`: 文件类型 (component, interface, config等)

**示例**:
- `resource_optimization_engine.py` - 资源优化引擎
- `trading_execution_manager.py` - 交易执行管理器
- `risk_control_service.py` - 风控服务

#### 工具和基础设施文件
```
{purpose}_{type}.py
```
- `purpose`: 用途 (common, shared, base等)
- `type`: 类型 (validator, detector, handler等)

**示例**:
- `common_validators.py` - 通用验证器
- `shared_interfaces.py` - 共享接口
- `base_component.py` - 基础组件

#### 特殊文件
- `__init__.py` - 包初始化文件
- `conftest.py` - pytest配置
- `setup.py` - 项目安装脚本

### 2. 目录结构命名

#### 主要目录
```
src/
├── core/              # 核心业务逻辑
├── api/               # API接口层
├── services/          # 业务服务层
├── repositories/      # 数据访问层
├── models/            # 数据模型
├── utils/             # 工具函数
├── config/            # 配置管理
├── infrastructure/    # 基础设施层
├── adapters/          # 外部适配器
├── gateways/          # 网关层
└── tests/             # 测试代码
```

#### 子目录命名
- 使用小写字母和下划线
- 反映功能分组
- 保持层次清晰

**示例**:
```
monitoring/
├── health/           # 健康检查
├── metrics/          # 指标收集
├── alerts/           # 告警管理
├── performance/      # 性能监控
└── tests/            # 测试相关
```

## 🏗️ 类命名规范

### 1. 类命名格式
```
{PascalCase}{Type}
```

#### 核心业务类
```
{Component}{Type}
```
- `Component`: 组件名称
- `Type`: 类类型 (Manager, Service, Engine, Handler等)

**示例**:
```python
class ResourceOptimizationEngine:      # 资源优化引擎
class TradingExecutionManager:         # 交易执行管理器
class RiskControlService:             # 风控服务
```

#### 数据类和模型
```
{Entity}{Type}
```
```python
@dataclass
class PerformanceMetrics:             # 性能指标
class TradingStrategy:                # 交易策略
```

#### 工具类
```
{Function}{Type}
```
```python
class CommonValidators:               # 通用验证器
class MemoryLeakDetector:             # 内存泄漏检测器
```

### 2. 特殊类命名

#### 异常类
```
{ErrorType}Error
```
```python
class ValidationError(Exception):      # 验证错误
class ConfigurationError(Exception):   # 配置错误
```

#### 工厂类
```
{Component}Factory
```
```python
class ResourceManagerFactory:         # 资源管理器工厂
```

#### 适配器类
```
{Target}Adapter
```
```python
class DatabaseAdapter:                # 数据库适配器
class ApiGatewayAdapter:              # API网关适配器
```

## 🛠️ 方法命名规范

### 1. 方法命名格式
```
{verb}_{object}[_{modifier}]
```
使用小写字母和下划线分隔

#### 动作动词
- `get_` - 获取数据
- `set_` - 设置数据
- `create_` - 创建对象
- `update_` - 更新数据
- `delete_` - 删除数据
- `process_` - 处理数据
- `validate_` - 验证数据
- `convert_` - 转换数据
- `calculate_` - 计算数据
- `generate_` - 生成数据

**示例**:
```python
def get_system_status(self) -> Dict[str, Any]:          # 获取系统状态
def create_trading_order(self, order: Order) -> str:    # 创建交易订单
def validate_configuration(self, config: Dict) -> bool:  # 验证配置
def calculate_performance_metrics(self) -> Metrics:      # 计算性能指标
```

### 2. 私有方法
```
_{verb}_{object}[_{modifier}]
```
使用单下划线前缀

**示例**:
```python
def _validate_input_data(self, data: Any) -> bool:       # 验证输入数据
def _calculate_health_score(self, metrics: Metrics) -> float:  # 计算健康评分
```

### 3. 特殊方法类型

#### 工厂方法
```python
@classmethod
def create_from_config(cls, config: Dict) -> 'Self':     # 从配置创建实例
```

#### 属性方法
```python
@property
def is_healthy(self) -> bool:                            # 健康状态属性
```

## 📊 变量和常量命名

### 1. 变量命名
```
snake_case
```
使用小写字母和下划线

**示例**:
```python
system_status = "healthy"
current_metrics = get_metrics()
trading_orders = []
```

### 2. 常量命名
```
SCREAMING_SNAKE_CASE
```
使用大写字母和下划线

**示例**:
```python
MAX_RETRY_COUNT = 3
DEFAULT_TIMEOUT = 30
HEALTH_CHECK_INTERVAL = 60
```

### 3. 枚举值
```python
class HealthStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
```

## 🏷️ 包和模块导入

### 1. 导入顺序
```python
# 标准库导入
import os
import sys
from typing import Dict, List

# 第三方库导入
import pandas as pd
import numpy as np

# 本地模块导入
from .shared_interfaces import ILogger
from ..config import settings
```

### 2. 导入别名
```python
# 常用别名
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 避免过度缩写
from .resource_manager import ResourceManager as RM  # 不推荐
from .resource_manager import ResourceManager        # 推荐
```

## 🎯 命名一致性检查

### 1. 文件名与类名对应
文件名应该与主要类名对应：
- `resource_manager.py` → `class ResourceManager`
- `trading_engine.py` → `class TradingEngine`

### 2. 方法命名一致性
相似功能使用相同的前缀：
```python
def get_system_status(self):      # ✓
def fetch_system_state(self):     # ✗ - 不一致

def validate_input(self):         # ✓
def check_input_validity(self):   # ✗ - 不一致
```

## 📝 注释和文档规范

### 1. 模块级文档
```python
"""
模块功能描述

详细说明模块的职责、主要类和使用方法。
"""

# 示例
"""
资源优化引擎

负责执行资源优化策略和配置调整。
提供内存、CPU、磁盘和网络等资源的优化功能。
"""
```

### 2. 类级文档
```python
class ResourceOptimizationEngine:
    """
    资源优化引擎

    提供全面的资源优化功能，包括：
    - 内存优化和垃圾回收
    - CPU亲和性配置
    - 磁盘I/O优化
    - 并行化配置
    """
```

### 3. 方法文档
```python
def optimize_resources(self, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    执行资源优化

    Args:
        config: 优化配置参数

    Returns:
        包含优化结果的字典

    Raises:
        ValueError: 当配置无效时抛出
    """
```

## 🔍 命名规范检查工具

### 1. 自动检查脚本
```bash
# 运行命名规范检查
python scripts/check_naming_conventions.py src/

# 检查特定文件
python scripts/check_naming_conventions.py src/core/resource_manager.py
```

### 2. IDE 配置
- 配置代码格式化工具 (black, autopep8)
- 启用拼写检查和命名规范检查
- 设置代码模板和自动补全

## 📊 遵守程度评估

### 1. 评估指标
- **文件名一致性**: 95%以上的文件遵循命名规范
- **类名一致性**: 98%以上的类遵循命名规范
- **方法名一致性**: 90%以上的方法遵循命名规范
- **导入顺序**: 100%遵循标准导入顺序

### 2. 持续改进
- 定期审查和更新命名规范
- 新增代码必须遵循现有规范
- 重构时优先修正命名问题
- 团队培训和规范宣贯

---

**📋 总结**

良好的命名规范是高质量代码的基础。本规范通过标准化命名约定，提高了代码的可读性、可维护性和团队协作效率。在实际应用中，应结合具体项目特点适当调整，同时保持整体一致性。
