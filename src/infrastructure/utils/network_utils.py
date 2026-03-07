"""
网络连接工具模块
提供网络连接的增强功能，包括重试机制、超时控制等
"""

import time
import logging
import asyncio
from typing import Any, Callable, Optional, Union
from functools import wraps

logger = logging.getLogger(__name__)


class NetworkRetryConfig:
    """网络重试配置"""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        backoff_factor: float = 2.0,
        timeout: float = 10.0
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.timeout = timeout


def retry_on_network_error(
    config: Optional[NetworkRetryConfig] = None,
    retry_exceptions: Optional[tuple] = None
):
    """
    网络错误重试装饰器

    Args:
        config: 重试配置
        retry_exceptions: 需要重试的异常类型
    """
    if config is None:
        config = NetworkRetryConfig()

    if retry_exceptions is None:
        retry_exceptions = (
            ConnectionError,
            TimeoutError,
            OSError,  # 包含ConnectionResetError, RemoteDisconnected等
        )

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            return await _retry_async(func, args, kwargs, config, retry_exceptions)

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            return _retry_sync(func, args, kwargs, config, retry_exceptions)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


async def _retry_async(
    func: Callable,
    args: tuple,
    kwargs: dict,
    config: NetworkRetryConfig,
    retry_exceptions: tuple
) -> Any:
    """异步重试逻辑"""
    last_exception = None

    for attempt in range(config.max_retries + 1):
        try:
            if attempt > 0:
                delay = min(config.base_delay * (config.backoff_factor ** (attempt - 1)), config.max_delay)
                logger.warning(f"网络请求失败，{delay:.1f}秒后重试 (尝试 {attempt}/{config.max_retries})")
                await asyncio.sleep(delay)

            # 设置超时
            return await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=config.timeout
            )

        except retry_exceptions as e:
            last_exception = e
            if attempt == config.max_retries:
                logger.error(f"网络请求失败，已达到最大重试次数: {e}")
                break
            else:
                logger.warning(f"网络请求失败，将重试: {e}")
                continue
        except asyncio.TimeoutError as e:
            last_exception = e
            if attempt == config.max_retries:
                logger.error(f"网络请求超时，已达到最大重试次数")
                break
            else:
                logger.warning(f"网络请求超时，将重试")
                continue

    raise last_exception


def _retry_sync(
    func: Callable,
    args: tuple,
    kwargs: dict,
    config: NetworkRetryConfig,
    retry_exceptions: tuple
) -> Any:
    """同步重试逻辑"""
    last_exception = None

    for attempt in range(config.max_retries + 1):
        try:
            if attempt > 0:
                delay = min(config.base_delay * (config.backoff_factor ** (attempt - 1)), config.max_delay)
                logger.warning(f"网络请求失败，{delay:.1f}秒后重试 (尝试 {attempt}/{config.max_retries})")
                time.sleep(delay)

            return func(*args, **kwargs)

        except retry_exceptions as e:
            last_exception = e
            if attempt == config.max_retries:
                logger.error(f"网络请求失败，已达到最大重试次数: {e}")
                break
            else:
                logger.warning(f"网络请求失败，将重试: {e}")
                continue

    raise last_exception


def create_akshare_retry_config() -> NetworkRetryConfig:
    """创建AKShare专用的重试配置"""
    return NetworkRetryConfig(
        max_retries=5,  # AKShare网络较不稳定，需要更多重试
        base_delay=2.0,
        max_delay=60.0,
        backoff_factor=1.5,
        timeout=30.0  # AKShare有时需要更长的超时时间
    )


def enhance_akshare_function(func: Callable) -> Callable:
    """
    增强AKShare函数，添加重试机制

    Args:
        func: AKShare函数

    Returns:
        增强后的函数
    """
    config = create_akshare_retry_config()

    @retry_on_network_error(config=config)
    def enhanced_sync(*args, **kwargs):
        """同步AKShare函数增强"""
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            logger.error(f"AKShare函数 {func.__name__} 执行失败: {e}")
            raise

    @retry_on_network_error(config=config)
    async def enhanced_async(*args, **kwargs):
        """异步AKShare函数增强"""
        try:
            # 在线程池中运行同步AKShare函数
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, func, *args, **kwargs)
            return result
        except Exception as e:
            logger.error(f"AKShare函数 {func.__name__} 异步执行失败: {e}")
            raise

    # 根据原始函数类型返回相应的增强版本
    if asyncio.iscoroutinefunction(func):
        return enhanced_async
    else:
        return enhanced_sync


# 全局网络监控
_network_stats = {
    'requests_total': 0,
    'requests_success': 0,
    'requests_failed': 0,
    'retries_total': 0,
    'timeouts_total': 0
}


def get_network_stats() -> dict:
    """获取网络统计信息"""
    total = _network_stats['requests_total']
    success = _network_stats['requests_success']
    failed = _network_stats['requests_failed']

    return {
        'total_requests': total,
        'success_rate': f"{(success / total * 100):.1f}%" if total > 0 else "0%",
        'failure_rate': f"{(failed / total * 100):.1f}%" if total > 0 else "0%",
        'retries': _network_stats['retries_total'],
        'timeouts': _network_stats['timeouts_total'],
        'details': _network_stats.copy()
    }


def record_network_request(success: bool = True, retried: bool = False, timeout: bool = False):
    """记录网络请求统计"""
    _network_stats['requests_total'] += 1

    if success:
        _network_stats['requests_success'] += 1
    else:
        _network_stats['requests_failed'] += 1

    if retried:
        _network_stats['retries_total'] += 1

    if timeout:
        _network_stats['timeouts_total'] += 1


# 默认的重试配置
DEFAULT_RETRY_CONFIG = NetworkRetryConfig()
AKSHARE_RETRY_CONFIG = create_akshare_retry_config()