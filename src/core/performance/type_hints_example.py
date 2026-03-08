"""
类型注解示例文件

展示如何在性能优化模块中使用类型注解
遵循PEP 484标准
"""

from typing import (
    Dict, List, Optional, Union, Any, Callable,
    TypeVar, Generic, Protocol, Tuple, AsyncIterator
)
from dataclasses import dataclass
from datetime import datetime
import asyncio

# 定义类型变量
T = TypeVar('T')
R = TypeVar('R')
K = TypeVar('K')
V = TypeVar('V')


# 定义协议（接口）
class Cacheable(Protocol):
    """可缓存对象协议"""
    def cache_key(self) -> str: ...
    def cache_ttl(self) -> int: ...


class Processable(Protocol[T, R]):
    """可处理对象协议"""
    async def process(self, item: T) -> R: ...


# 带类型注解的数据类
@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    timestamp: datetime
    cpu_percent: float
    memory_mb: float
    latency_ms: float
    throughput: float
    error_rate: float

    def to_dict(self) -> Dict[str, Union[str, float]]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_percent': self.cpu_percent,
            'memory_mb': self.memory_mb,
            'latency_ms': self.latency_ms,
            'throughput': self.throughput,
            'error_rate': self.error_rate,
        }


@dataclass
class BatchConfig:
    """批处理配置类"""
    batch_size: int
    max_wait_time: float
    retry_count: int = 3
    timeout: Optional[float] = None


# 泛型类示例
class TypedCache(Generic[K, V]):
    """
    类型安全的缓存实现

    类型参数:
        K: 键的类型
        V: 值的类型
    """

    def __init__(self, max_size: int = 1000) -> None:
        self._cache: Dict[K, V] = {}
        self._max_size = max_size

    def get(self, key: K) -> Optional[V]:
        """获取缓存值"""
        return self._cache.get(key)

    def set(self, key: K, value: V) -> None:
        """设置缓存值"""
        if len(self._cache) >= self._max_size:
            # 简单的LRU淘汰
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[key] = value

    def get_all(self) -> Dict[K, V]:
        """获取所有缓存项"""
        return self._cache.copy()


# 异步函数类型注解
async def process_batch(
    items: List[T],
    processor: Callable[[T], asyncio.Future[R]],
    config: BatchConfig
) -> List[R]:
    """
    异步批处理函数

    参数:
        items: 待处理项目列表
        processor: 异步处理函数
        config: 批处理配置

    返回:
        处理结果列表
    """
    results: List[R] = []

    for i in range(0, len(items), config.batch_size):
        batch = items[i:i + config.batch_size]

        # 并发处理批次
        batch_tasks = [processor(item) for item in batch]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

        for result in batch_results:
            if isinstance(result, Exception):
                # 处理错误
                continue
            results.append(result)

    return results


# 复杂类型注解示例
class PerformanceMonitor:
    """
    性能监控器

    展示复杂的类型注解用法
    """

    def __init__(self) -> None:
        # 带类型的字典
        self._metrics: Dict[str, List[PerformanceMetrics]] = {}

        # 回调函数字典
        self._callbacks: Dict[str, Callable[[PerformanceMetrics], None]] = {}

        # 可选的配置
        self._config: Optional[BatchConfig] = None

    def register_callback(
        self,
        name: str,
        callback: Callable[[PerformanceMetrics], None]
    ) -> None:
        """注册监控回调"""
        self._callbacks[name] = callback

    async def collect_metrics(
        self,
        duration_seconds: float
    ) -> AsyncIterator[PerformanceMetrics]:
        """
        异步收集指标

        返回异步迭代器
        """
        start_time = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start_time < duration_seconds:
            # 模拟指标收集
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_percent=50.0,
                memory_mb=1024.0,
                latency_ms=100.0,
                throughput=1000.0,
                error_rate=0.01
            )

            yield metrics
            await asyncio.sleep(1.0)

    def analyze_trends(
        self,
        metric_name: str
    ) -> Tuple[float, float, float]:
        """
        分析指标趋势

        返回:
            (平均值, 最小值, 最大值)
        """
        metrics = self._metrics.get(metric_name, [])

        if not metrics:
            return (0.0, 0.0, 0.0)

        values = [m.latency_ms for m in metrics]
        avg = sum(values) / len(values)
        min_val = min(values)
        max_val = max(values)

        return (avg, min_val, max_val)


# Union类型示例
def parse_config_value(value: Union[str, int, float, bool]) -> Any:
    """
    解析配置值

    参数可以是多种类型
    """
    if isinstance(value, str):
        # 尝试解析为数字
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            # 检查布尔值
            if value.lower() in ('true', 'yes', '1'):
                return True
            elif value.lower() in ('false', 'no', '0'):
                return False
            return value
    return value


# 装饰器类型注解
from functools import wraps

F = TypeVar('F', bound=Callable[..., Any])


def timed(func: F) -> F:
    """
    计时装饰器

    展示如何为装饰器添加类型注解
    """
    @wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        start = asyncio.get_event_loop().time()
        result = await func(*args, **kwargs)
        elapsed = asyncio.get_event_loop().time() - start
        print(f"{func.__name__} took {elapsed:.2f}s")
        return result

    @wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        import time
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} took {elapsed:.2f}s")
        return result

    if asyncio.iscoroutinefunction(func):
        return async_wrapper  # type: ignore
    return sync_wrapper  # type: ignore


# 类型别名
MetricValue = Union[int, float, str]
MetricDict = Dict[str, MetricValue]
ProcessorFunc = Callable[[T], R]


def aggregate_metrics(metrics: List[MetricDict]) -> MetricDict:
    """
    聚合指标

    使用类型别名简化注解
    """
    result: MetricDict = {}

    for metric in metrics:
        for key, value in metric.items():
            if key not in result:
                result[key] = value
            elif isinstance(value, (int, float)):
                current = result[key]
                if isinstance(current, (int, float)):
                    result[key] = current + value

    return result
