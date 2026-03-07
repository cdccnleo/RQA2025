# 日志系统API参考文档

## 概述

RQA2025日志系统采用分层架构设计，提供企业级的结构化日志记录功能。系统已于2025年9月完成架构重构，消除重复代码，建立清晰的模块化架构。

**架构层次**：
- `core/` - 核心抽象接口和基础实现
- `handlers/` - 日志处理器实现
- `formatters/` - 日志格式化器
- `utils/` - 工具类和辅助功能
- `advanced/` - 高级功能扩展
- `monitors/` - 监控和统计功能

## 核心接口

### ILogger 接口

```python
from abc import ABC, abstractmethod
from src.infrastructure.logging.core.interfaces import LogLevel

class ILogger(ABC):
    """日志器统一抽象接口"""

    @abstractmethod
    def log(self, level: str, message: str, **kwargs) -> None:
        """记录日志"""

    @abstractmethod
    def debug(self, message: str, **kwargs) -> None:
        """记录DEBUG级别日志"""

    @abstractmethod
    def info(self, message: str, **kwargs) -> None:
        """记录INFO级别日志"""

    @abstractmethod
    def warning(self, message: str, **kwargs) -> None:
        """记录WARNING级别日志"""

    @abstractmethod
    def error(self, message: str, **kwargs) -> None:
        """记录ERROR级别日志"""

    @abstractmethod
    def critical(self, message: str, **kwargs) -> None:
        """记录CRITICAL级别日志"""
```

## 核心实现类

### BaseLogger

```python
from src.infrastructure.logging.core.base_logger import BaseLogger
from src.infrastructure.logging.core.interfaces import LogLevel

class BaseLogger(ILogger):
    """基础日志器实现"""

    def __init__(self, name: str = "BaseLogger", level: LogLevel = LogLevel.INFO):
        """初始化基础日志器"""

    def log(self, level: str, message: str, **kwargs) -> None:
        """记录日志"""

    def debug(self, message: str, **kwargs) -> None:
        """记录调试日志"""

    def info(self, message: str, **kwargs) -> None:
        """记录信息日志"""

    def warning(self, message: str, **kwargs) -> None:
        """记录警告日志"""

    def error(self, message: str, **kwargs) -> None:
        """记录错误日志"""

    def critical(self, message: str, **kwargs) -> None:
        """记录严重错误日志"""
```

### UnifiedLogger

```python
from src.infrastructure.logging.core.unified_logger import UnifiedLogger

class UnifiedLogger(BaseLogger):
    """统一日志器 - 提供企业级日志功能"""

    def __init__(self, name: str = "RQA2025", level: LogLevel = LogLevel.INFO):
        """初始化统一日志器"""

    def log_structured(self, level: LogLevel, message: str, **kwargs) -> None:
        """结构化日志记录"""

    def log_performance(self, operation: str, duration: float, **kwargs) -> None:
        """性能日志记录"""

    def log_business_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """业务事件日志"""

    def log_security_event(self, event_type: str, user_id: str = None, **kwargs) -> None:
        """安全事件日志"""

    def get_log_stats(self) -> Dict[str, Any]:
        """获取日志统计信息"""

    def shutdown(self) -> None:
        """关闭日志器，清理资源"""
```

### 专用Logger类

```python
from src.infrastructure.logging.core.base_logger import (
    BusinessLogger, AuditLogger, PerformanceLogger
)

# 业务日志器 - 专门用于业务逻辑日志
business_logger = BusinessLogger("BusinessLogger")

# 审计日志器 - 专门用于安全审计日志
audit_logger = AuditLogger("AuditLogger")

# 性能日志器 - 专门用于性能监控日志
performance_logger = PerformanceLogger("PerformanceLogger")
```

## 高级功能

### AdvancedLogger

```python
from src.infrastructure.logging.advanced import AdvancedLogger

class AdvancedLogger(UnifiedLogger):
    """高级日志器 - 提供异步处理和性能监控"""

    def __init__(self, name: str = "AdvancedLogger",
                 level: LogLevel = LogLevel.INFO,
                 enable_async: bool = True,
                 enable_monitoring: bool = True):
        """初始化高级日志器"""

    def log_async(self, level: LogLevel, message: str, **kwargs):
        """异步日志记录"""

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""

    def log_with_performance_tracking(self, level: LogLevel, message: str,
                                    operation: str, **kwargs):
        """带性能跟踪的日志记录"""
```

## 处理器和格式化器

### 日志处理器

```python
from src.infrastructure.logging.handlers import ConsoleHandler, FileHandler

# 控制台处理器
console_handler = ConsoleHandler(LogLevel.INFO)

# 文件处理器
file_handler = FileHandler("logs/app.log", LogLevel.DEBUG, max_bytes=10*1024*1024)
```

### 日志格式化器

```python
from src.infrastructure.logging.utils import LogFormatter

# 文本格式化
text_message = LogFormatter.format_text(log_record)

# JSON格式化
json_message = LogFormatter.format_json(log_record)

# 结构化格式化
structured_message = LogFormatter.format_structured(log_record)
```

## 数据库审计功能

### DatabaseAuditLogger

```python
from src.infrastructure.logging.audit_logger import DatabaseAuditLogger

# 创建数据库审计日志器
audit_logger = DatabaseAuditLogger(
    log_file="logs/database_audit.log",
    max_records=10000,
    enable_encryption=False
)

# 记录数据库操作
audit_logger.log_database_operation(
    operation_type="query",
    sql="SELECT * FROM users WHERE id = ?",
    params={"id": 123},
    execution_time=0.05,
    user_id="user123"
)

# 获取审计记录
records = audit_logger.get_audit_records(user_id="user123", limit=10)

# 获取统计信息
stats = audit_logger.get_statistics()
```

### 枚举类型

## 使用示例

### 基本使用

```python
from src.infrastructure.logging.core import UnifiedLogger, LogLevel

# 创建日志器
logger = UnifiedLogger("MyApp", LogLevel.INFO)

# 记录不同级别的日志
logger.debug("调试信息")
logger.info("一般信息")
logger.warning("警告信息")
logger.error("错误信息")
logger.critical("严重错误")
```

### 专用Logger使用

```python
from src.infrastructure.logging.core import BusinessLogger, AuditLogger, PerformanceLogger

# 业务日志
business_logger = BusinessLogger("BusinessModule")
business_logger.info("用户登录", user_id="12345", action="login")

# 审计日志
audit_logger = AuditLogger("SecurityAudit")
audit_logger.warning("可疑活动检测", user_id="12345", activity="multiple_failed_logins")

# 性能日志
perf_logger = PerformanceLogger("PerformanceMonitor")
perf_logger.info("API响应时间", endpoint="/api/users", duration=0.125)
```

### 结构化日志

```python
from src.infrastructure.logging.core import UnifiedLogger, LogLevel

logger = UnifiedLogger("StructuredLogger")

# 结构化业务事件
logger.log_business_event("user_registration", {
    "user_id": "12345",
    "email": "user@example.com",
    "registration_method": "email",
    "timestamp": "2025-09-23T10:30:00Z"
})

# 性能监控
logger.log_performance("database_query", 0.05, query_type="SELECT", table="users")

# 安全事件
logger.log_security_event("unauthorized_access", user_id="12345", resource="/admin")
```

### 高级功能使用

```python
from src.infrastructure.logging.advanced import AdvancedLogger

# 创建高级日志器
advanced_logger = AdvancedLogger("AdvancedApp", enable_async=True, enable_monitoring=True)

# 异步日志记录
advanced_logger.log_async(LogLevel.INFO, "异步处理的消息", task_id="123")

# 带性能跟踪的日志
advanced_logger.log_with_performance_tracking(
    LogLevel.INFO,
    "数据库查询完成",
    "user_lookup",
    user_id="12345",
    query_time=0.03
)

# 获取性能统计
stats = advanced_logger.get_performance_stats()
print(f"日志处理统计: {stats}")
```

## 最佳实践

### 1. Logger选择

- **UnifiedLogger**: 通用应用场景，推荐大多数情况使用
- **BusinessLogger**: 业务逻辑相关日志
- **AuditLogger**: 安全审计和合规日志
- **PerformanceLogger**: 性能监控和指标日志
- **AdvancedLogger**: 需要异步处理和高性能的应用

### 2. 日志级别使用

- **DEBUG**: 详细的调试信息，仅开发环境使用
- **INFO**: 重要的业务信息和正常操作
- **WARNING**: 潜在问题，不影响正常功能
- **ERROR**: 错误情况，需要关注
- **CRITICAL**: 严重错误，可能导致系统不可用

### 3. 结构化日志

```python
# 推荐：结构化日志
logger.info("用户操作", user_id=user.id, action="login", ip=request.ip)

# 不推荐：字符串拼接
logger.info(f"用户 {user.id} 执行登录操作，IP: {request.ip}")
```

### 4. 错误处理

```python
try:
    # 业务逻辑
    result = process_data(data)
    logger.info("数据处理成功", records_processed=len(result))
except Exception as e:
    logger.error("数据处理失败", error=str(e), data_id=data.id)
    raise
```

### 5. 性能考虑

```python
# 避免在循环中记录过多日志
for item in large_list:
    if item.status == 'error':
        logger.warning("项目状态异常", item_id=item.id, status=item.status)

# 使用采样或批量记录
logger.info("批量处理完成", total_items=len(large_list), success_count=success_count)
```

## 故障排除

### 常见问题

1. **导入错误**: 确保路径配置正确，检查模块是否存在
2. **日志不输出**: 检查日志级别设置，确认处理器配置
3. **性能问题**: 避免在热点路径记录过多日志，考虑异步处理
4. **文件权限**: 确保日志目录有写入权限

### 调试技巧

```python
# 启用DEBUG级别查看详细日志
logger = UnifiedLogger("DebugApp", LogLevel.DEBUG)

# 查看日志器状态
stats = logger.get_log_stats()
print(f"日志器状态: {stats}")

# 高级日志器的性能监控
if hasattr(logger, 'get_performance_stats'):
    perf_stats = logger.get_performance_stats()
    print(f"性能统计: {perf_stats}")
```

## 版本历史

- **v2.0 (2025-09-23)**: 架构重构完成，消除重复代码，建立分层架构
- **v1.5 (2025-09-20)**: 高级功能增强，添加异步处理和性能监控
- **v1.0 (2025-09-01)**: 初始版本，统一日志接口和基础实现

---

*最后更新: 2025-09-23*
*维护者: RQA2025基础设施团队*
