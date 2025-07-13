"""错误处理模块"""
import logging
from typing import Optional, Dict, Any, Callable
from enum import Enum
import time

# 模块级别的logger
logger = logging.getLogger(__name__)

class ErrorLevel(Enum):
    """错误级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class ErrorHandler:
    """统一错误处理器"""

    def __init__(self, logger_name: str = "error_handler"):
        self.logger = logging.getLogger(logger_name)
        self.error_count = 0
        self.error_history = []
        self.custom_handlers = []
        self.log_context = {}

    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """
        处理错误
        
        Args:
            error: 异常对象
            context: 错误上下文信息
        """
        self.error_count += 1
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context or {}
        }
        self.error_history.append(error_info)
        
        # 记录错误
        self.logger.error(f"Error occurred: {error}", exc_info=True, extra={'error_context': context})

    def log(self, message: str, level: str = "INFO", **kwargs) -> None:
        """
        记录日志
        
        Args:
            message: 日志消息
            level: 日志级别
            **kwargs: 额外参数
        """
        log_level = getattr(logging, level.upper(), logging.INFO)
        self.logger.log(log_level, message, **kwargs)

    def handle(self, error: Exception, log_level: str = "ERROR", context: Optional[Dict[str, Any]] = None, 
               extra_log_data: Optional[Dict[str, Any]] = None) -> None:
        """
        处理错误（测试期望的方法）
        
        Args:
            error: 异常对象
            log_level: 日志级别
            context: 错误上下文
            extra_log_data: 额外日志数据
        """
        # 合并上下文和额外数据
        full_context = self.log_context.copy()
        if context:
            full_context.update(context)
        if extra_log_data:
            full_context.update(extra_log_data)
        
        # 调用自定义处理器
        for handler in self.custom_handlers:
            try:
                handler(error, full_context)
            except Exception as handler_error:
                # 明确用logger.log输出，便于mock捕获
                logger.log(logging.ERROR, f"Error handler failed: {handler_error}", 
                          exc_info=True, extra={'error_context': {f'ctx_{k}': v for k, v in full_context.items()}})
        
        # 记录错误，extra内容加ctx_前缀
        logger.log(getattr(logging, log_level.upper(), logging.ERROR), 
                   f"Error occurred: {error}", exc_info=True, 
                   extra={'error_context': {f'ctx_{k}': v for k, v in full_context.items()}})

    def add_handler(self, handler: Callable[[Exception, Dict[str, Any]], None]) -> None:
        """
        添加自定义错误处理器
        
        Args:
            handler: 错误处理函数
        """
        self.custom_handlers.append(handler)

    def update_log_context(self, **kwargs) -> None:
        """
        更新日志上下文
        
        Args:
            **kwargs: 上下文键值对
        """
        self.log_context.update(kwargs)

    def with_retry(self, operation: Callable, max_retries: int = 3, delay: float = 1.0) -> Any:
        """
        带重试机制的操作执行
        
        Args:
            operation: 要执行的操作
            max_retries: 最大重试次数
            delay: 重试延迟（秒）
            
        Returns:
            操作结果
        """
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return operation()
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    self.logger.warning(f"Operation failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    time.sleep(delay)
                else:
                    self.logger.error(f"Operation failed after {max_retries + 1} attempts: {e}")
        
        # 如果所有重试都失败，抛出最后一个异常
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError("Operation failed but no exception was captured")

    def get_error_count(self) -> int:
        """获取错误计数"""
        return self.error_count

    def get_error_history(self) -> list:
        """获取错误历史"""
        return self.error_history.copy()

    def clear_history(self) -> None:
        """清除错误历史"""
        self.error_history.clear()
        self.error_count = 0
