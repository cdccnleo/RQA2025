"""
advanced_logger 模块

提供 advanced_logger 相关功能和接口。
"""


import asyncio
import logging
import threading
import time

from ..core import UnifiedLogger, LogLevel
from .types import LogEntry, LogEntryPool
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Callable, Union
"""
基础设施层 - 高级日志器

提供高级日志功能，包括性能监控、异步处理等。
继承自UnifiedLogger，提供额外的企业级功能。
"""


class AdvancedLogger(UnifiedLogger):
    """
    高级日志器

    在UnifiedLogger基础上提供额外的企业级功能：
    - 异步日志处理
    - 性能监控
    - 智能过滤
    - 日志压缩和归档
    """

    def __init__(self, name: str = "AdvancedLogger", level: LogLevel = LogLevel.INFO,
                 enable_async: bool = True, enable_monitoring: bool = True):
        """
        初始化高级日志器

        Args:
            name: 日志器名称
            level: 日志级别
            enable_async: 是否启用异步处理
            enable_monitoring: 是否启用性能监控
        """
        super().__init__(name)
        self.name = name
        self.level = level

        self.enable_async = enable_async
        self.enable_monitoring = enable_monitoring

        # 异步处理组件
        self._async_executor = ThreadPoolExecutor(max_workers=4) if enable_async else None
        self._event_loop = None

        # 性能监控
        self._performance_stats = {
            'total_logs': 0,
            'log_count': 0,
            'error_count': 0,
            'avg_processing_time': 0.0,
            'total_processing_time': 0.0,
            'max_processing_time': 0.0,
            'min_processing_time': float('inf'),
            'start_time': time.time()
        }

        # 对象池
        self._entry_pool = LogEntryPool()

        # 智能过滤器
        self._filters = []

        # 日志队列
        self._log_queue = []
        self._async_queue = []

        # 兼容性别名
        self._executor = self._async_executor

        if enable_async:
            self._init_async_processing()

    def log_structured(self, level: Union[LogLevel, str, int], message: Any, **kwargs):
        """
        结构化日志记录

        Args:
            level: 日志级别
            message: 日志消息
            **kwargs: 额外参数
        """
        resolved_level = self._normalize_level(level)
        payload = kwargs.copy()
        if isinstance(message, dict):
            payload.update(message)
            message_text = str(message.get("message", message))
        else:
            message_text = str(message)
        self.log(resolved_level.value, message_text)
        return payload

    def _normalize_level(self, level: Union[LogLevel, str, int]) -> LogLevel:
        if isinstance(level, LogLevel):
            return level
        if isinstance(level, int):
            for member in LogLevel:
                if getattr(logging, member.value) == level:
                    return member
            return LogLevel.INFO
        level_str = str(level).upper()
        return LogLevel[level_str] if level_str in LogLevel.__members__ else LogLevel.INFO

    def shutdown(self):
        """关闭高级日志器"""
        # 关闭异步执行器 - 优先检查_executor（可能被测试覆盖）
        executor = getattr(self, '_executor', None) or getattr(self, '_async_executor', None)
        if executor and hasattr(executor, 'shutdown'):
            executor.shutdown(wait=True)

        if hasattr(self, '_event_thread') and self._event_thread and self._event_thread.is_alive():
            self._event_thread.join(timeout=5.0)

        # 清理队列
        self._log_queue.clear()

        # 清理对象池
        if hasattr(self._entry_pool, 'clear'):
            self._entry_pool.clear()

    def _init_async_processing(self):
        """初始化异步处理"""
        def run_event_loop():
            self._event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._event_loop)
            self._event_loop.run_forever()

        self._event_thread = threading.Thread(target=run_event_loop, daemon=True)
        self._event_thread.start()

    def log_async(self, level: LogLevel, message: str, **kwargs):
        """
        异步日志记录

        Args:
            level: 日志级别
            message: 日志消息
            **kwargs: 额外参数
        """
        if not self.enable_async or not self._async_executor:
            return self.log_structured(level, message, **kwargs)

        # 提交异步任务
        self._async_executor.submit(self._do_async_log, level, message, **kwargs)

    def _do_async_log(self, level: LogLevel, message: str, **kwargs):
        """执行异步日志记录"""
        try:
            start_time = time.time()
            self.log_structured(level, message, **kwargs)

            if self.enable_monitoring:
                processing_time = time.time() - start_time
                self._update_performance_stats(processing_time)

        except Exception as e:
            # 异步日志失败时的错误处理
            print(f"异步日志失败: {e}")

    def _update_performance_stats(self, processing_time: float):
        """更新性能统计"""
        self._performance_stats['log_count'] += 1
        self._performance_stats['total_logs'] += 1
        self._performance_stats['total_processing_time'] += processing_time

        # 更新最大/最小处理时间
        if processing_time > self._performance_stats['max_processing_time']:
            self._performance_stats['max_processing_time'] = processing_time
        if processing_time < self._performance_stats['min_processing_time']:
            self._performance_stats['min_processing_time'] = processing_time

        # 计算平均处理时间
        if self._performance_stats['log_count'] > 0:
            self._performance_stats['avg_processing_time'] = (
                self._performance_stats['total_processing_time'] / self._performance_stats['log_count']
            )

    def add_filter(self, filter_func: Callable[[LogEntry], bool]):
        """
        添加日志过滤器

        Args:
            filter_func: 过滤函数，接收LogEntry返回bool
        """
        self._filters.append(filter_func)

    def _should_filter(self, entry: LogEntry) -> bool:
        """检查是否应该过滤日志条目"""
        # 如果没有过滤器，不过滤
        if not self._filters:
            return False

        # 所有过滤器都必须返回True才能通过（不被过滤）
        # 任何一个过滤器返回False都会导致被过滤
        for filter_func in self._filters:
            if not filter_func(entry):
                return True  # 被过滤
        return False  # 不被过滤

    def log_with_performance_tracking(self, level: LogLevel, message: str, operation: str, **kwargs):
        """
        带性能跟踪的日志记录

        Args:
            level: 日志级别
            message: 日志消息
            operation: 操作名称
            **kwargs: 额外参数
        """
        start_time = time.time()

        # 执行操作（这里只是记录日志，实际使用时可以包装业务逻辑）
        if self.enable_async:
            self.log_async(level, message, operation=operation, **kwargs)
        else:
            self.log_structured(level, message, operation=operation, **kwargs)

        processing_time = time.time() - start_time

        # 更新性能统计
        if self.enable_monitoring:
            self._update_performance_stats(processing_time)

        return None

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        获取性能统计信息

        Returns:
            性能统计字典
        """
        stats = self._performance_stats.copy()
        stats['uptime'] = time.time() - stats['start_time']
        stats['logs_per_second'] = stats['log_count'] / \
            stats['uptime'] if stats['uptime'] > 0 else 0
        return stats

    def log_with_performance_tracking(self, level: LogLevel, message: str,
                                      operation: str, **kwargs):
        """
        带性能跟踪的日志记录

        Args:
            level: 日志级别
            message: 日志消息
            operation: 操作名称
            **kwargs: 额外参数
        """
        start_time = time.time()

        # 执行操作（这里只是记录日志，实际使用时可以包装业务逻辑）
        if self.enable_async:
            self.log_async(level, message, operation=operation, **kwargs)
        else:
            self.log_structured(level, message, operation=operation, **kwargs)

        processing_time = time.time() - start_time

        # 更新性能统计
        if self.enable_monitoring:
            self._update_performance_stats(processing_time)

    def shutdown(self):
        """关闭高级日志器"""

        if self.enable_async and self._async_executor:
            self._async_executor.shutdown(wait=True)

        if hasattr(self, '_event_loop') and self._event_loop:
            self._event_loop.stop()

    def set_context(self, context: Dict[str, Any]):
        """设置日志上下文"""
        self._context = context

    def update_config(self, config: Dict[str, Any]):
        """更新配置"""
        if 'level' in config:
            from logging import getLevelName
            if isinstance(config['level'], str):
                self.level = LogLevel(getLevelName(config['level']))
            else:
                self.level = LogLevel(config['level'])

    def log_batch(self, messages: list):
        """批量日志记录"""
        for msg in messages:
            if isinstance(msg, dict):
                level = msg.get('level', 'INFO')
                message = msg.get('msg', str(msg))
                self.log(level, message)
            else:
                self.info(str(msg))

    def _format_log(self, level, message, *args, **kwargs):
        """格式化日志"""
        return message

    def _log_performance(self, operation: str, duration: float, details: dict):
        """记录性能日志"""
        self.log_structured(LogLevel.INFO, f"Performance: {operation} took {duration}s", 
                           operation=operation, duration=duration, **details)

    def log_performance(self, operation: str, duration: float, details: dict):
        """记录性能日志的公共方法"""
        self._log_performance(operation, duration, details)

    def log_security_event(self, event_type: str, details: dict):
        """记录安全事件"""
        self.log(LogLevel.WARNING.value, f"Security event: {event_type}", extra={"security": details})

    def _rotate_logs(self):
        """日志轮转"""
        pass

    def _check_rotation(self):
        """检查是否需要轮转日志"""
        self._rotate_logs()

    def _log_async(self, message: str):
        """异步日志记录内部方法"""
        self._async_queue.append(message)

    def _log(self, level, message, *args, **kwargs):
        """内部日志方法，供测试Mock使用"""
        return self.log(level.value if hasattr(level, 'value') else level, message, *args, **kwargs)

    def __del__(self):
        """析构函数，确保资源清理"""
        try:
            self.shutdown()
        except BaseException:
            pass
