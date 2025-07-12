from functools import wraps
import time
import random
import threading
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, Union, List, Type, Callable
import logging

logger = logging.getLogger(__name__)


class RetryHandler:
    """处理函数重试逻辑的装饰器类"""

    def __init__(self, initial_delay: float = 0.2, max_delay: float = 30.0,
                 backoff_factor: float = 2.0, jitter: float = 0.1,
                 max_attempts: int = 3, retry_exceptions: Optional[List[Type[Exception]]] = None,
                 # 兼容性参数
                 max_retries: Optional[int] = None, timeout: Optional[float] = None):
        """初始化重试处理器
        Args:
            initial_delay: 初始延迟时间(秒)
            max_delay: 最大延迟时间(秒)
            backoff_factor: 退避因子
            jitter: 抖动系数(0-1)
            max_attempts: 最大总尝试次数（包括初始尝试）
            retry_exceptions: 需要重试的异常类型列表
            max_retries: 兼容性参数，等同于max_attempts
            timeout: 兼容性参数，暂未使用
        Raises:
            ValueError: 如果参数无效
        """
        # 兼容性处理
        if max_retries is not None:
            max_attempts = max_retries
        
        if initial_delay <= 0 or max_delay <= 0:
            raise ValueError("延迟时间必须大于0")
        if backoff_factor < 1:
            raise ValueError("退避因子必须大于等于1")
        if not (0 <= jitter <= 1):
            raise ValueError("抖动系数必须在0-1之间")
        if max_attempts < 1:
            raise ValueError("最大尝试次数必须大于0")

        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.max_attempts = max_attempts
        self.retry_exceptions = retry_exceptions or []
        self._lock = threading.RLock()  # 可重入锁
        self._local_random = random.Random()  # 线程本地随机数生成器
        self._thread_local = threading.local()

    def _calculate_delay(self, attempt: int) -> float:
        """计算退避延迟时间，包含抖动
        Args:
            attempt: 当前尝试次数(1-based)
        Returns:
            计算后的延迟时间(秒)
        """
        # 基础延迟 = 初始延迟 * (退避因子)^(尝试次数-1)
        base_delay = self.initial_delay * (self.backoff_factor ** (attempt - 1))
        base_delay = min(base_delay, self.max_delay)

        if self.jitter:
            # 抖动基于初始延迟而非当前延迟
            jitter_amount = random.uniform(-self.jitter, self.jitter) * self.initial_delay
            jittered_delay = base_delay + jitter_amount
            # 确保延迟不小于0
            return max(0, jittered_delay)
        return base_delay

    def _get_thread_records(self):
        if not hasattr(self._thread_local, 'retry_records'):
            self._thread_local.retry_records = {}
        return self._thread_local.retry_records

    def _record_retry(self, func_id: str, attempt: int, success: bool, exception: Optional[Dict[str, Any]], delay: float = 0) -> None:
        """记录重试历史
        Args:
            func_id: 函数标识
            attempt: 尝试次数(0-based)
            success: 是否成功
            exception: 异常信息
            delay: 实际延迟时间(秒)，默认为0
        """
        with self._lock:
            retry_records = self._get_thread_records()
            if func_id not in retry_records:
                retry_records[func_id] = []

            record = {
                'function': func_id,
                'timestamp': datetime.now().isoformat(),
                'attempt': attempt + 1,  # 调整为1-based计数
                'retry_count': attempt,  # 实际重试次数(0-based)
                'success': success,
                'delay': delay,  # 使用传入的实际延迟时间
                'exception': None
            }

            if exception:
                record['exception'] = {
                    'type': exception['type'],
                    'message': exception['message']
                }

            retry_records[func_id].append(record)

    def _should_retry(self, exception: Exception, retry_exceptions: Optional[List[Type[Exception]]] = None) -> bool:
        """检查异常是否应该重试"""
        # 使用传入的重试异常列表或默认值
        effective_retry_exceptions = retry_exceptions if retry_exceptions is not None else self.retry_exceptions

        # 如果没有指定重试异常类型，默认重试所有异常
        if not effective_retry_exceptions:
            return True

        # 检查异常是否在重试列表中
        return any(isinstance(exception, exc_type) for exc_type in effective_retry_exceptions)

    def get_retry_history(self, func_id: Optional[str] = None, only_failures: bool = True) -> List[Dict[str, Any]]:
        """获取重试历史记录
        Args:
            func_id: 可选函数标识，如果提供则返回该函数的记录
            only_failures: 是否只返回失败记录，默认为True
        Returns:
            重试记录列表
        """
        retry_records = self._get_thread_records()
        if func_id:
            records = retry_records.get(func_id, [])
            return [r for r in records if not only_failures or not r['success']]
        return [r for records in retry_records.values()
               for r in records if not only_failures or not r['success']]

    def get_retry_stats(self, func_name: str) -> Dict[str, Any]:
        """获取指定函数的重试统计信息
        Args:
            func_name: 函数名称
        Returns:
            包含重试统计信息的字典
        """
        retry_records = self._get_thread_records()
        # 查找所有匹配函数名的记录(可能包含模块前缀)
        matched_records = []
        for func_id, records in retry_records.items():
            if func_name in func_id:  # 部分匹配函数名
                matched_records.extend(records)
        
        successful = any(r['success'] for r in matched_records) if matched_records else False
        return {
            'function': func_name,
            'total_attempts': len(matched_records),
            'successful_attempts': sum(1 for r in matched_records if r['success']),
            'failed_attempts': sum(1 for r in matched_records if not r['success']),
            'successful': successful,
            'last_attempt': matched_records[-1] if matched_records else None
        }

    def get_retry_summary(self) -> Dict[str, Any]:
        """获取重试统计摘要
        Returns:
            Dict: 包含总重试次数和各函数重试情况的字典
        """
        retry_records = self._get_thread_records()
        summary = {
            'total_retries': 0,
            'functions': {}
        }

        for func_id, records in retry_records.items():
            # 只计算失败的尝试作为重试
            retries = sum(1 for r in records if not r['success'])
            successes = sum(1 for r in records if r['success'])

            summary['functions'][func_id] = {
                'total_attempts': len(records),
                'successful_attempts': successes,
                'failed_retries': retries
            }
            summary['total_retries'] += retries

        return summary

    def __call__(self, func=None, **kwargs):
        """装饰器实现，支持两种调用方式：
        1. @retry_handler
        2. @retry_handler.with_retry()
        """
        if func is None:
            return lambda f: self._create_wrapper(f, **kwargs)
        return self._create_wrapper(func, **kwargs)

    def _create_wrapper(self, func, **kwargs):
        """创建实际的装饰器包装函数"""

        @wraps(func)
        def wrapped(*args, **wrapped_kwargs):
            last_exception = None
            # 使用模块名+函数名作为唯一标识
            module_name = func.__module__.split('.')[-1]  # 获取最后一级模块名
            import threading
            thread_id = threading.get_ident()
            func_id = f"{module_name}.{func.__name__}.thread{thread_id}"

            # 从装饰器参数中获取覆盖的重试异常列表
            override_retry_exceptions = kwargs.get('retry_exceptions', None)

            # 修正：总尝试次数为max_attempts（包括初始尝试）
            for attempt in range(self.max_attempts):
                try:
                    result = func(*args, **wrapped_kwargs)
                    self._record_retry(
                        func_id,
                        attempt,
                        success=True,
                        exception=None,
                        delay=0
                    )
                    return result
                except Exception as e:
                    last_exception = e
                    delay = self._calculate_delay(attempt) if attempt < self.max_attempts - 1 else 0
                    self._record_retry(
                        func_id,
                        attempt,
                        success=False,
                        exception={
                            'type': type(e).__name__,
                            'message': str(e)
                        },
                        delay=delay
                    )

                    # 使用覆盖的重试异常列表（如果存在）
                    current_retry_exceptions = (
                        override_retry_exceptions
                        if override_retry_exceptions is not None
                        else self.retry_exceptions
                    )

                    if not self._should_retry(e, current_retry_exceptions):
                        # 不可重试异常，直接抛出原始异常
                        raise e

                    if attempt < self.max_attempts - 1:
                        delay = self._calculate_delay(attempt)
                        logger.info(f"Retry attempt {attempt + 1}/{self.max_attempts} after {delay:.2f}s for {func_id}")
                        time.sleep(delay)
                        continue
                    
                    # 可重试异常但重试次数用完，抛出RetryError并设置异常链
                    from src.infrastructure.error.exceptions import RetryError
                    retry_error = RetryError(f"Function {func_id} failed after {self.max_attempts} attempts", 
                                           attempts=self.max_attempts, 
                                           max_retries=self.max_attempts)
                    retry_error.__cause__ = last_exception
                    raise retry_error

        return wrapped

    def with_retry(self, **kwargs):
        """支持@retry_handler.with_retry()语法"""
        return self.__call__(**kwargs)

    def clear_history(self) -> None:
        """清空所有重试历史记录"""
        self._thread_local.retry_records = {}

    def _get_stats_for_func(self, func_name: str) -> Dict[str, Any]:
        """获取或创建函数的统计记录
        Args:
            func_name: 函数名称
        Returns:
            包含统计信息的字典
        """
        if func_name not in self._get_thread_records():
            self._get_thread_records()[func_name] = []
        
        # 从历史记录中计算当前统计
        records = self._get_thread_records()[func_name]
        return {
            'total_attempts': len(records) + 1 if records else 1,
            'successful': any(r['success'] for r in records),
            'last_exception': records[-1]['exception'] if records and not records[-1]['success'] else None
        }

    def get_retry_count(self, func_name: str) -> int:
        """获取指定函数的实际重试次数（不包括初始尝试）
        Args:
            func_name: 函数名称
        Returns:
            实际重试次数
        """
        records = self._get_thread_records().get(func_name, [])
        # 实际重试次数 = 总尝试次数 - 1（减去初始尝试）
        return max(0, len(records) - 1) if records else 0

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """执行被包装的函数，应用重试逻辑"""
        # 直接使用装饰器逻辑
        wrapped_func = self._create_wrapper(func)
        return wrapped_func(*args, **kwargs)

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """execute方法作为call方法的别名"""
        return self.call(func, *args, **kwargs)
