# RQA2025 基础设施层功能增强分析报告（续3）

## 2. 功能分析（续）

### 2.4 监控系统增强（续）

#### 2.4.2 应用监控（续）

**实现建议**（续）：

```python
    def get_custom_metrics(
        self,
        name: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> List[Dict]:
        """
        获取自定义指标
        
        Args:
            name: 指标名称
            start_time: 开始时间（ISO格式）
            end_time: 结束时间（ISO格式）
            tags: 标签过滤
            
        Returns:
            List[Dict]: 自定义指标列表
        """
        metrics = self.metrics['custom_metrics']
        
        if name:
            metrics = [m for m in metrics if m['name'] == name]
        
        if start_time:
            metrics = [m for m in metrics if m['timestamp'] >= start_time]
        
        if end_time:
            metrics = [m for m in metrics if m['timestamp'] <= end_time]
        
        if tags:
            metrics = [
                m for m in metrics if all(
                    m['tags'].get(k) == v for k, v in tags.items()
                )
            ]
        
        return metrics
    
    def get_function_summary(self, name: Optional[str] = None) -> Dict:
        """
        获取函数指标摘要
        
        Args:
            name: 函数名称
            
        Returns:
            Dict: 函数指标摘要
        """
        metrics = self.get_function_metrics(name=name)
        
        if not metrics:
            return {
                'total_calls': 0,
                'success_rate': 0,
                'avg_time': 0,
                'min_time': 0,
                'max_time': 0
            }
        
        total_calls = len(metrics)
        success_calls = sum(1 for m in metrics if m['success'])
        success_rate = success_calls / total_calls if total_calls > 0 else 0
        
        execution_times = [m['execution_time'] for m in metrics]
        avg_time = sum(execution_times) / len(execution_times) if execution_times else 0
        min_time = min(execution_times) if execution_times else 0
        max_time = max(execution_times) if execution_times else 0
        
        return {
            'total_calls': total_calls,
            'success_calls': success_calls,
            'success_rate': success_rate,
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time
        }
    
    def get_error_summary(self) -> Dict:
        """
        获取错误指标摘要
        
        Returns:
            Dict: 错误指标摘要
        """
        errors = self.metrics['errors']
        
        if not errors:
            return {
                'total_errors': 0,
                'sources': {}
            }
        
        # 按来源统计错误数量
        sources = {}
        for error in errors:
            source = error['source']
            if source not in sources:
                sources[source] = 0
            sources[source] += 1
        
        return {
            'total_errors': len(errors),
            'sources': sources
        }
```

### 2.5 错误处理增强

#### 2.5.1 统一异常处理

**现状分析**：
异常处理机制不够完善，缺乏统一的异常处理策略。

**实现建议**：
实现一个 `ErrorHandler` 类，提供统一的异常处理功能：

```python
import sys
import traceback
from typing import Dict, List, Optional, Callable, Any, Type
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ErrorHandler:
    """错误处理器"""
    
    def __init__(
        self,
        log_errors: bool = True,
        raise_unknown: bool = False,
        alert_callbacks: Optional[List[Callable[[str, Dict], None]]] = None
    ):
        """
        初始化错误处理器
        
        Args:
            log_errors: 是否记录错误日志
            raise_unknown: 是否抛出未处理的异常
            alert_callbacks: 告警回调函数列表
        """
        self.log_errors = log_errors
        self.raise_unknown = raise_unknown
        self.alert_callbacks = alert_callbacks or []
        
        # 异常处理器映射
        self.handlers: Dict[Type[Exception], Callable] = {}
        
        # 错误历史
        self.error_history: List[Dict] = []
    
    def register_handler(
        self,
        exception_type: Type[Exception],
        handler: Callable[[Exception], Any]
    ) -> None:
        """
        注册异常处理器
        
        Args:
            exception_type: 异常类型
            handler: 处理函数
        """
        self.handlers[exception_type] = handler
    
    def handle(self, exception: Exception) -> Any:
        """
        处理异常
        
        Args:
            exception: 异常对象
            
        Returns:
            Any: 处理结果
        """
        exception_type = type(exception)
        
        # 记录错误
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'type': exception_type.__name__,
            'message': str(exception),
            'stack_trace': traceback.format_exc()
        }
        self.error_history.append(error_info)
        
        # 限制错误历史数量
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]
        
        # 记录错误日志
        if self.log_errors:
            logger.error(
                f"Error: {exception_type.__name__}: {exception}",
                exc_info=True
            )
        
        # 发送告警
        self._send_alert('error', error_info)
        
        # 查找处理器
        handler = None
        for exc_type, hdlr in self.handlers.items():
            if isinstance(exception, exc_type):
                handler = hdlr
                break
        
        # 处理异常
        if handler:
            return handler(exception)
        elif self.raise_unknown:
            raise exception
        else:
            return None
    
    def _send_alert(self, level: str, alert_data: Dict) -> None:
        """
        发送告警
        
        Args:
            level: 告警级别
            alert_data: 告警数据
        """
        # 调用告警回调函数
        for callback in self.alert_callbacks:
            try:
                callback(level, alert_data)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def get_error_history(
        self,
        error_type: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> List[Dict]:
        """
        获取错误历史
        
        Args:
            error_type: 错误类型
            start_time: 开始时间（ISO格式）
            end_time: 结束时间（ISO格式）
            
        Returns:
            List[Dict]: 错误历史列表
        """
        errors = self.error_history
        
        if error_type:
            errors = [e for e in errors if e['type'] == error_type]
        
        if start_time:
            errors = [e for e in errors if e['timestamp'] >= start_time]
        
        if end_time:
            errors = [e for e in errors if e['timestamp'] <= end_time]
        
        return errors
    
    def get_error_summary(self) -> Dict:
        """
        获取错误摘要
        
        Returns:
            Dict: 错误摘要
        """
        if not self.error_history:
            return {
                'total_errors': 0,
                'types': {}
            }
        
        # 按类型统计错误数量
        types = {}
        for error in self.error_history:
            error_type = error['type']
            if error_type not in types:
                types[error_type] = 0
            types[error_type] += 1
        
        return {
            'total_errors': len(self.error_history),
            'types': types
        }
    
    def safe_execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        安全执行函数
        
        Args:
            func: 要执行的函数
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            Any: 函数返回值或异常处理结果
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return self.handle(e)
```

#### 2.5.2 重试机制

**现状分析**：
缺乏自动重试机制，导致临时性错误可能导致任务失败。

**实现建议**：
实现一个 `RetryHandler` 类，提供自动重试功能：

```python
import time
from typing import Dict, List, Optional, Callable, Any, Type, Union
import logging
from datetime import datetime
import functools
import random

logger = logging.getLogger(__name__)

class RetryHandler:
    """重试处理器"""
    
    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        backoff_factor: float = 2.0,
        jitter: float = 0.1,
        retry_exceptions: Optional[List[Type[Exception]]] = None
    ):
        """
        初始化重试处理器
        
        Args:
            max_retries: 最大重试次数
            retry_delay: 初始重试延迟（秒）
            backoff_factor: 退避因子
            jitter: 抖动因子
            retry_exceptions: 需要重试的异常类型列表
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.retry_exceptions = retry_exceptions or [Exception]
        
        # 重试历史
        self.retry_history: List[Dict] = []
    
    def with_retry(
        self,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
        backoff_factor: Optional[float] = None,
        jitter: Optional[float] = None,
        retry_exceptions: Optional[List[Type[Exception]]] = None
    ) -> Callable:
        """
        重试装饰器
        
        Args:
            max_retries: 最大重试次数
            retry_delay: 初始重试延迟（秒）
            backoff_factor: 退避因子
            jitter: 抖动因子
            retry_exceptions: 需要重试的异常类型列表
            
        Returns:
            Callable: 装饰器函数
        """
        _max_retries = max_retries if max_retries is not None else self.max_retries
        _retry_delay = retry_delay if retry_delay is not None else self.retry_delay
        _backoff_factor = backoff_factor if backoff_factor is not None else self.backoff_factor
        _jitter = jitter if jitter is not None else self.jitter
        _retry_exceptions = retry_exceptions if retry_exceptions is not None else self.retry_exceptions
        
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                retry_count = 0
                last_exception = None
                
                while retry_count <= _max_retries:
                    try:
                        if retry_count > 0:
                            # 计算重试延迟
                            delay = _retry_delay * (_backoff_factor ** (retry_count - 1))
                            # 添加随机抖动
                            delay = delay * (1 + random.uniform(-_jitter, _jitter))
                            
                            logger.info(f"Retrying {func.__name__} after {delay:.2f}s (attempt {retry_count}/{_max_retries})")
                            time.sleep(delay)
                        
                        # 执行函数
                        result = func(*args, **kwargs)
                        
                        # 如果成功且不是第一次尝试，记录重试成功
                        if retry_count > 0:
                            self._record_retry(
                                func.__name__,
                                retry_count,
                                True,
                                last_exception
                            )
                        
                        return result
                    
                    except Exception as e:
                        last_exception = e
                        
                        # 检查是否是需要重试的异常
                        should_retry = any(
                            isinstance(e, exc_type) for exc_type in _retry_exceptions
                        )
                        
                        if not should_retry or retry_count >= _max_retries:
                            # 记录重试失败
                            self._record_retry(
                                func.__name__,
                                retry_count,
                                False,
                                last_exception
                            )
                            raise e
                        
                        retry_count += 1
                
                # 不应该到达这里
                raise RuntimeError("Unexpected error in retry logic")
            
            return wrapper
        
        return decorator
    
    def _record_retry(
        self,
        function_name: str,
        retry_count: int,
        success: bool,
        exception: Optional[Exception] = None
    ) -> None:
        """
        记录重试
        
        Args:
            function_name: 函数名称
            retry_count: 重试次数
            success: 是否成功
            exception: 异常对象
        """
        retry_info = {
            'timestamp': datetime.now().isoformat(),
            'function': function_name,
            'retry_count': retry_count,
            'success': success
        }
        
        if exception:
            retry_info['exception'] = {
                'type': type(exception).__name__,
                'message': str(exception)
            }
        
        self.retry_history.append(retry_info)
        
        # 限制重