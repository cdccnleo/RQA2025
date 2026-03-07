# 日志系统开发指南

## 概述

本文档定义了日志系统的开发规范、接口设计原则和最佳实践，确保代码质量和一致性。

## 接口设计原则

### 1. 协议导向设计

所有Logger实现必须遵循`ILogger`协议：

```python
from typing import Protocol, runtime_checkable
from src.infrastructure.logging.core.interfaces import ILogger

@runtime_checkable
class ICustomLogger(ILogger):
    """自定义Logger必须实现ILogger协议"""

    def custom_method(self) -> None:
        """扩展方法"""
        pass
```

### 2. 单一职责原则

每个Logger类职责明确：

```python
# ✅ 正确：单一职责
class DatabaseLogger(BaseLogger):
    """专门处理数据库操作日志"""
    def log_query(self, sql: str, duration: float) -> None:
        pass

# ❌ 错误：职责混杂
class MixedLogger(BaseLogger):
    """同时处理数据库、缓存、网络日志"""
    def log_database(self): pass
    def log_cache(self): pass
    def log_network(self): pass
```

### 3. 开闭原则

通过继承和组合扩展功能：

```python
# 通过继承扩展
class CloudLogger(BaseLogger):
    """云环境专用Logger"""
    def __init__(self, cloud_provider: str, **kwargs):
        super().__init__(**kwargs)
        self.cloud_provider = cloud_provider

# 通过组合添加功能
class MonitoredLogger:
    """带监控功能的Logger包装器"""
    def __init__(self, logger: ILogger):
        self._logger = logger
        self._metrics = {}

    def info(self, message: str, **kwargs):
        self._metrics['info_count'] = self._metrics.get('info_count', 0) + 1
        self._logger.info(message, **kwargs)
```

## 代码规范

### 1. 命名规范

```python
# 类名
class BusinessLogger(BaseLogger):  # PascalCase
    pass

class AuditLogger(BaseLogger):
    pass

# 方法名
def log_business_event(self):  # snake_case
    pass

def get_operation_stats(self):
    pass

# 变量名
log_level = LogLevel.INFO  # snake_case
max_retry_count = 3

# 常量
DEFAULT_TIMEOUT = 30  # UPPER_SNAKE_CASE
MAX_LOG_SIZE = 100 * 1024 * 1024
```

### 2. 类型注解

```python
from typing import Dict, List, Optional, Any, Union
from src.infrastructure.logging.core.interfaces import LogLevel, LogFormat

def process_log_entry(
    self,
    message: str,
    level: LogLevel,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    处理日志条目

    Args:
        message: 日志消息
        level: 日志级别
        metadata: 可选的元数据

    Returns:
        处理后的日志数据
    """
    result: Dict[str, Any] = {
        'message': message,
        'level': level.value,
        'timestamp': datetime.now().isoformat()
    }

    if metadata:
        result.update(metadata)

    return result
```

### 3. 文档字符串

使用Google风格的docstring：

```python
def log_business_event(
    self,
    event_type: str,
    event_id: str,
    user_id: str,
    data: Dict[str, Any],
    status: str = "success",
    **kwargs: Any
) -> None:
    """
    记录业务事件

    Args:
        event_type: 事件类型，如"order", "payment"等
        event_id: 事件唯一标识符
        user_id: 用户标识符
        data: 事件相关数据
        status: 事件状态，默认为"success"
        **kwargs: 额外的上下文信息

    Raises:
        ValueError: 当event_type为空时抛出

    Example:
        >>> logger.log_business_event(
        ...     "order_created", "ord_123", "user_456",
        ...     {"amount": 99.99}, "success"
        ... )
    """
    if not event_type:
        raise ValueError("event_type cannot be empty")

    # 实现逻辑...
```

## 测试规范

### 1. 测试结构

```python
# tests/unit/infrastructure/logging/test_custom_logger.py
import pytest
from src.infrastructure.logging import CustomLogger, LogLevel

class TestCustomLogger:
    """CustomLogger单元测试"""

    def setup_method(self):
        """测试前准备"""
        self.logger = CustomLogger(name="test")

    def teardown_method(self):
        """测试后清理"""
        # 清理资源

    def test_initialization(self):
        """测试初始化"""
        assert self.logger.name == "test"

    def test_custom_functionality(self):
        """测试自定义功能"""
        # 测试逻辑

    @pytest.mark.parametrize("input_value,expected", [
        ("test1", "result1"),
        ("test2", "result2"),
    ])
    def test_parametrized_cases(self, input_value, expected):
        """参数化测试"""
        assert self.logger.process(input_value) == expected
```

### 2. 测试覆盖率要求

- **单元测试**: >80% 行覆盖率
- **分支覆盖**: >70%
- **关键路径**: 100% 覆盖

### 3. Mock和Stub使用

```python
import pytest
from unittest.mock import Mock, patch, MagicMock

def test_with_mock():
    """使用Mock的测试"""
    with patch('src.infrastructure.logging.external_service') as mock_service:
        mock_service.call.return_value = {"status": "success"}

        logger = CustomLogger()
        result = logger.call_external_service()

        assert result["status"] == "success"
        mock_service.call.assert_called_once()
```

## 错误处理规范

### 1. 异常层次结构

```python
# src/infrastructure/logging/exceptions.py
class LoggingError(Exception):
    """日志系统基础异常"""
    pass

class ConfigurationError(LoggingError):
    """配置相关异常"""
    pass

class HandlerError(LoggingError):
    """处理器相关异常"""
    pass

class FormatterError(LoggingError):
    """格式化器相关异常"""
    pass
```

### 2. 防御性编程

```python
def safe_log_operation(self, operation: str, **kwargs) -> bool:
    """
    安全的日志操作，总是返回成功状态

    Returns:
        操作是否成功
    """
    try:
        # 核心逻辑
        self._perform_operation(operation, **kwargs)
        return True
    except Exception as e:
        # 记录错误但不抛出
        self._logger.error(f"日志操作失败: {operation}", {
            'error': str(e),
            'error_type': type(e).__name__,
            'operation': operation,
            'kwargs': kwargs
        })
        return False
```

### 3. 优雅降级

```python
def initialize_with_fallback(self):
    """带降级的初始化"""
    try:
        # 尝试高级功能
        self._init_advanced_features()
    except Exception as e:
        self._logger.warning("高级功能初始化失败，使用基础功能", {
            'error': str(e)
        })
        # 降级到基础功能
        self._init_basic_features()
```

## 性能优化规范

### 1. 异步处理

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncLogger(BaseLogger):
    """异步Logger实现"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._executor = ThreadPoolExecutor(max_workers=4)

    async def log_async(self, level: LogLevel, message: str, **kwargs):
        """异步日志记录"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor,
            self.log,
            level, message, kwargs
        )
```

### 2. 缓冲和批处理

```python
from collections import deque
import threading
import time

class BufferedLogger(BaseLogger):
    """带缓冲的Logger"""

    def __init__(self, buffer_size: int = 100, flush_interval: float = 5.0, **kwargs):
        super().__init__(**kwargs)
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self._buffer = deque()
        self._lock = threading.Lock()
        self._last_flush = time.time()

        # 启动后台刷新线程
        self._start_flush_thread()

    def log(self, level: LogLevel, message: str, **kwargs):
        """带缓冲的日志记录"""
        with self._lock:
            self._buffer.append((level, message, kwargs))

            # 检查是否需要刷新
            if len(self._buffer) >= self.buffer_size:
                self._flush_buffer()
            elif time.time() - self._last_flush >= self.flush_interval:
                self._flush_buffer()

    def _flush_buffer(self):
        """刷新缓冲区"""
        if not self._buffer:
            return

        # 批量处理日志
        batch = list(self._buffer)
        self._buffer.clear()
        self._last_flush = time.time()

        for level, message, kwargs in batch:
            super().log(level, message, **kwargs)
```

### 3. 内存管理

```python
import weakref

class MemoryEfficientLogger(BaseLogger):
    """内存高效的Logger"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._handlers_refs = []  # 使用弱引用

    def add_handler(self, handler):
        """使用弱引用添加处理器"""
        ref = weakref.ref(handler, self._cleanup_handler)
        self._handlers_refs.append(ref)

    def _cleanup_handler(self, ref):
        """清理失效的处理器引用"""
        self._handlers_refs = [r for r in self._handlers_refs if r() is not None]
```

## 扩展开发指南

### 1. 创建新的Logger类型

```python
from src.infrastructure.logging.core.interfaces import BaseLogger, LogCategory, LogFormat

class SecurityLogger(BaseLogger):
    """
    安全专用Logger

    专注于安全事件记录和威胁检测
    """

    def __init__(self, name: str = "security", **kwargs):
        super().__init__(
            name=name,
            category=LogCategory.SECURITY,
            format_type=LogFormat.STRUCTURED,
            **kwargs
        )

        # 安全特定配置
        self._threat_patterns = self._load_threat_patterns()

    def log_security_incident(self, incident_type: str, severity: str, details: Dict[str, Any]):
        """记录安全事件"""
        self.critical("安全事件检测", {
            'incident_type': incident_type,
            'severity': severity,
            'details': details,
            'timestamp': datetime.now().isoformat(),
            'alert_required': severity in ['high', 'critical']
        })

        # 触发告警
        if severity in ['high', 'critical']:
            self._trigger_alert(incident_type, details)

    def _load_threat_patterns(self) -> List[str]:
        """加载威胁模式"""
        return [
            'sql_injection',
            'xss_attempt',
            'brute_force',
            'unauthorized_access'
        ]

    def _trigger_alert(self, incident_type: str, details: Dict[str, Any]):
        """触发安全告警"""
        # 实现告警逻辑
        pass
```

### 2. 创建自定义处理器

```python
import logging
from typing import Dict, Any

class MetricsHandler(logging.Handler):
    """指标收集处理器"""

    def __init__(self):
        super().__init__()
        self.metrics = {
            'total_logs': 0,
            'logs_by_level': {},
            'errors': []
        }

    def emit(self, record):
        """处理日志记录"""
        self.metrics['total_logs'] += 1

        level_name = record.levelname
        if level_name not in self.metrics['logs_by_level']:
            self.metrics['logs_by_level'][level_name] = 0
        self.metrics['logs_by_level'][level_name] += 1

        # 收集错误信息
        if record.levelno >= logging.ERROR:
            self.metrics['errors'].append({
                'message': record.getMessage(),
                'level': level_name,
                'timestamp': record.created,
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            })

            # 限制错误历史长度
            if len(self.metrics['errors']) > 1000:
                self.metrics['errors'] = self.metrics['errors'][-500:]

    def get_metrics(self) -> Dict[str, Any]:
        """获取指标数据"""
        return self.metrics.copy()
```

### 3. 创建自定义格式化器

```python
import logging
import json
from datetime import datetime

class StructuredFormatter(logging.Formatter):
    """结构化日志格式化器"""

    def format(self, record) -> str:
        """格式化日志记录"""
        # 基础字段
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'process': record.process,
            'thread': record.thread
        }

        # 添加异常信息
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)

        # 添加自定义字段
        if hasattr(record, 'extra_data') and record.extra_data:
            log_entry.update(record.extra_data)

        return json.dumps(log_entry, ensure_ascii=False, default=str)
```

## 持续集成规范

### 1. 质量门禁

```yaml
# .github/workflows/quality-check.yml
name: Quality Check
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov mypy
    - name: Run tests
      run: python scripts/run_tests.py
    - name: Type check
      run: mypy src/infrastructure/logging --ignore-missing-imports
    - name: Interface compliance check
      run: python -c "from src.infrastructure.logging.core.interfaces import validate_interface_compliance; validate_interface_compliance()"
```

### 2. 代码质量检查

```yaml
# .pre-commit-config.yaml
repos:
- repo: https://github.com/pre-commit/pre-commit-hooks
  rev: v4.4.0
  hooks:
  - id: trailing-whitespace
  - id: end-of-file-fixer
  - id: check-yaml
  - id: check-added-large-files

- repo: https://github.com/psf/black
  rev: 22.3.0
  hooks:
  - id: black
    language_version: python3

- repo: https://github.com/pycqa/flake8
  rev: 4.0.1
  hooks:
  - id: flake8
    args: [--max-line-length=120, --extend-ignore=E203,W503]

- repo: https://github.com/pre-commit/mirrors-mypy
  rev: v0.991
  hooks:
  - id: mypy
    additional_dependencies: [types-all]
    args: [--ignore-missing-imports]
```

## 版本兼容性

### 1. API稳定性保证

- 保持向后兼容性
- 使用语义化版本控制
- 提供迁移指南

### 2. 弃用策略

```python
import warnings
from typing import Optional

class LegacyLogger(BaseLogger):
    """带向后兼容性的Logger"""

    def __init__(self, **kwargs):
        warnings.warn(
            "LegacyLogger is deprecated, use BaseLogger instead",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(**kwargs)

    def old_method(self, param: Optional[str] = None):
        """已弃用的方法"""
        warnings.warn(
            "old_method is deprecated, use new_method instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.new_method(param or "default")
```

## 监控和维护

### 1. 健康检查

```python
def health_check(self) -> Dict[str, Any]:
    """Logger健康检查"""
    return {
        'status': 'healthy',
        'handlers_count': len(self._handlers),
        'log_directory_writable': self._check_log_directory(),
        'buffer_size': len(getattr(self, '_buffer', [])),
        'last_log_time': getattr(self, '_last_log_time', None),
        'error_count': getattr(self, '_error_count', 0)
    }
```

### 2. 性能监控

```python
import time
from functools import wraps

def performance_monitor(func):
    """性能监控装饰器"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        try:
            result = func(self, *args, **kwargs)
            duration = time.time() - start_time

            # 记录慢操作
            if duration > 0.1:  # 超过100ms
                self._logger.warning("慢操作检测", {
                    'method': func.__name__,
                    'duration_seconds': duration,
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys())
                })

            return result
        except Exception as e:
            duration = time.time() - start_time
            self._logger.error("方法执行失败", {
                'method': func.__name__,
                'duration_seconds': duration,
                'error': str(e)
            })
            raise
    return wrapper
```

遵循这些规范，将确保日志系统的可维护性、可扩展性和可靠性。

