# unified_exceptions模块职责分析

## 📂 两个模块的职责定位

### 1. `src/infrastructure/error/exceptions/unified_exceptions.py`
**定位**: **基础设施层**的具体异常定义  
**职责**:
- 定义基础设施层的具体异常类
- ErrorCode枚举（错误代码1000-9999）
- 具体异常类：DataLoaderError, ConfigurationError, NetworkError等
- 工厂方法：file_not_found(), invalid_format()等
- 基础设施层专用异常

**特点**:
- 聚焦于基础设施组件的异常
- 提供具体的错误代码和分类
- 简洁的异常创建方法

### 2. `src/core/foundation/exceptions/unified_exceptions.py`
**定位**: **核心服务层**的统一异常处理框架  
**职责**:
- RQA2025Exception基类（所有系统异常的根）
- 异常分层架构：Business, Infrastructure, System, ExternalService
- 异常处理装饰器：handle_exceptions(), handle_business_exceptions()等
- 异常监控和统计：ExceptionMonitor, ExceptionStatistics
- 全局异常管理：global_exception_monitor, global_exception_config
- 验证工具函数：validate_not_none(), validate_range()等

**特点**:
- 提供完整的异常处理框架
- 支持监控、告警、统计
- 装饰器模式便捷使用
- 全局实例和配置

### 3. `src/core/unified_exceptions.py`（别名模块）
**定位**: 别名模块，向后兼容  
**职责**:
- 从foundation.exceptions.unified_exceptions导入
- 提供便捷的导入路径

---

## 🎯 职责边界

| 维度 | Infrastructure层 | Core Foundation层 |
|------|----------------|------------------|
| **异常定义** | 具体的基础设施异常 | 系统异常基类和框架 |
| **错误代码** | 1000-9999（具体） | 业务代码（抽象） |
| **处理机制** | 简单的异常类 | 装饰器、监控、统计 |
| **使用场景** | 基础设施组件内部 | 全系统统一处理 |
| **依赖关系** | 独立 | 可能依赖其他模块 |

---

## ✅ 使用建议

### 在基础设施层代码中
```python
from src.infrastructure.error.exceptions.unified_exceptions import (
    DataLoaderError,
    ConfigurationError,
    ErrorCode
)
```

### 在核心服务层代码中
```python
from src.core.foundation.exceptions.unified_exceptions import (
    handle_infrastructure_exceptions,
    ValidationError,
    BusinessException
)
```

### 在核心层简化导入
```python
from src.core.unified_exceptions import handle_infrastructure_exceptions
```

---

## 📋 当前状态

**别名模块**: ✅ 已正确创建  
**导入路径**: ✅ 符合规范  
**职责分离**: ✅ 清晰明确

---

_分析完成_

