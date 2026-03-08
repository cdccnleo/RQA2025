"""
Prometheus性能监控系统

功能:
- Prometheus指标收集和暴露
- 自定义业务指标
- 性能仪表板集成
- 分布式追踪支持
- 告警规则配置

监控目标:
- 请求延迟 P99 < 200ms
- 错误率 < 0.1%
- 系统资源使用率
"""

import asyncio
import functools
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Optional,
    TypeVar, Union, Generator
)
from enum import Enum
import threading

logger = logging.getLogger(__name__)

# 尝试导入prometheus客户端
try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Info,
        CollectorRegistry, generate_latest,
        CONTENT_TYPE_LATEST, start_http_server
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client 未安装，使用模拟实现")


F = TypeVar('F', bound=Callable[..., Any])


class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"
    INFO = "info"


@dataclass
class MetricConfig:
    """指标配置"""
    name: str
    description: str
    metric_type: MetricType
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # 仅用于Histogram


class MetricsCollector:
    """
    Prometheus指标收集器

    统一管理所有性能指标的收集和暴露
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return

        self._initialized = True
        self._registry = CollectorRegistry() if PROMETHEUS_AVAILABLE else None
        self._metrics: Dict[str, Any] = {}
        self._server_started = False
        self._server_port = 9090

        # 默认指标
        if PROMETHEUS_AVAILABLE:
            self._init_default_metrics()

    def _init_default_metrics(self) -> None:
        """初始化默认指标"""
        # HTTP请求指标
        self.register_metric(MetricConfig(
            name="http_requests_total",
            description="Total HTTP requests",
            metric_type=MetricType.COUNTER,
            labels=["method", "endpoint", "status"]
        ))

        self.register_metric(MetricConfig(
            name="http_request_duration_seconds",
            description="HTTP request duration in seconds",
            metric_type=MetricType.HISTOGRAM,
            labels=["method", "endpoint"],
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        ))

        # 业务指标
        self.register_metric(MetricConfig(
            name="trades_total",
            description="Total number of trades",
            metric_type=MetricType.COUNTER,
            labels=["symbol", "side", "status"]
        ))

        self.register_metric(MetricConfig(
            name="trade_latency_seconds",
            description="Trade execution latency",
            metric_type=MetricType.HISTOGRAM,
            labels=["symbol"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
        ))

        # 系统指标
        self.register_metric(MetricConfig(
            name="active_connections",
            description="Number of active connections",
            metric_type=MetricType.GAUGE,
            labels=["service"]
        ))

        self.register_metric(MetricConfig(
            name="queue_size",
            description="Current queue size",
            metric_type=MetricType.GAUGE,
            labels=["queue_name"]
        ))

        # 模型推理指标
        self.register_metric(MetricConfig(
            name="inference_requests_total",
            description="Total inference requests",
            metric_type=MetricType.COUNTER,
            labels=["model", "status"]
        ))

        self.register_metric(MetricConfig(
            name="inference_duration_seconds",
            description="Inference duration in seconds",
            metric_type=MetricType.HISTOGRAM,
            labels=["model"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
        ))

        self.register_metric(MetricConfig(
            name="model_cache_hits_total",
            description="Total model cache hits",
            metric_type=MetricType.COUNTER,
            labels=["model"]
        ))

        # 批处理指标
        self.register_metric(MetricConfig(
            name="batch_size",
            description="Current batch size",
            metric_type=MetricType.GAUGE,
            labels=["processor"]
        ))

        self.register_metric(MetricConfig(
            name="batch_processing_seconds",
            description="Batch processing duration",
            metric_type=MetricType.HISTOGRAM,
            labels=["processor"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25]
        ))

        # 数据库指标
        self.register_metric(MetricConfig(
            name="db_connections_active",
            description="Active database connections",
            metric_type=MetricType.GAUGE,
            labels=["pool"]
        ))

        self.register_metric(MetricConfig(
            name="db_query_duration_seconds",
            description="Database query duration",
            metric_type=MetricType.HISTOGRAM,
            labels=["operation"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
        ))

        # 缓存指标
        self.register_metric(MetricConfig(
            name="cache_hits_total",
            description="Total cache hits",
            metric_type=MetricType.COUNTER,
            labels=["cache_level"]
        ))

        self.register_metric(MetricConfig(
            name="cache_misses_total",
            description="Total cache misses",
            metric_type=MetricType.COUNTER,
            labels=["cache_level"]
        ))

        # 应用信息
        if PROMETHEUS_AVAILABLE:
            info = Info("app_info", "Application information", registry=self._registry)
            info.info({"version": "1.0.0", "name": "RQA2025"})

    def register_metric(self, config: MetricConfig) -> Any:
        """
        注册指标

        Args:
            config: 指标配置

        Returns:
            指标对象
        """
        if not PROMETHEUS_AVAILABLE:
            return MockMetric()

        if config.name in self._metrics:
            return self._metrics[config.name]

        if config.metric_type == MetricType.COUNTER:
            metric = Counter(
                config.name,
                config.description,
                config.labels,
                registry=self._registry
            )
        elif config.metric_type == MetricType.HISTOGRAM:
            metric = Histogram(
                config.name,
                config.description,
                config.labels,
                buckets=config.buckets or Histogram.DEFAULT_BUCKETS,
                registry=self._registry
            )
        elif config.metric_type == MetricType.GAUGE:
            metric = Gauge(
                config.name,
                config.description,
                config.labels,
                registry=self._registry
            )
        elif config.metric_type == MetricType.INFO:
            metric = Info(
                config.name,
                config.description,
                registry=self._registry
            )
        else:
            raise ValueError(f"Unknown metric type: {config.metric_type}")

        self._metrics[config.name] = metric
        return metric

    def get_metric(self, name: str) -> Optional[Any]:
        """获取指标对象"""
        return self._metrics.get(name)

    def increment_counter(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
        value: float = 1.0
    ) -> None:
        """
        增加计数器

        Args:
            name: 指标名称
            labels: 标签值
            value: 增加值
        """
        if not PROMETHEUS_AVAILABLE:
            return

        metric = self._metrics.get(name)
        if metric and isinstance(metric, Counter):
            if labels:
                metric.labels(**labels).inc(value)
            else:
                metric.inc(value)

    def observe_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        观察直方图值

        Args:
            name: 指标名称
            value: 观察值
            labels: 标签值
        """
        if not PROMETHEUS_AVAILABLE:
            return

        metric = self._metrics.get(name)
        if metric and isinstance(metric, Histogram):
            if labels:
                metric.labels(**labels).observe(value)
            else:
                metric.observe(value)

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        设置仪表盘值

        Args:
            name: 指标名称
            value: 设置值
            labels: 标签值
        """
        if not PROMETHEUS_AVAILABLE:
            return

        metric = self._metrics.get(name)
        if metric and isinstance(metric, Gauge):
            if labels:
                metric.labels(**labels).set(value)
            else:
                metric.set(value)

    def start_server(self, port: int = 9090) -> None:
        """
        启动Prometheus HTTP服务器

        Args:
            port: 服务端口
        """
        if not PROMETHEUS_AVAILABLE:
            logger.warning("prometheus_client 未安装，无法启动服务器")
            return

        if self._server_started:
            logger.warning("Prometheus服务器已在运行")
            return

        try:
            start_http_server(port, registry=self._registry)
            self._server_port = port
            self._server_started = True
            logger.info(f"Prometheus指标服务器已启动，端口: {port}")
        except Exception as e:
            logger.exception(f"启动Prometheus服务器失败: {e}")

    def stop_server(self) -> None:
        """停止Prometheus服务器"""
        # prometheus_client不支持直接停止服务器
        self._server_started = False
        logger.info("Prometheus服务器标记为停止")

    def generate_metrics(self) -> bytes:
        """
        生成Prometheus格式的指标数据

        Returns:
            指标数据字节串
        """
        if not PROMETHEUS_AVAILABLE:
            return b"# prometheus_client not installed\n"

        return generate_latest(self._registry)

    def get_content_type(self) -> str:
        """获取Content-Type"""
        return CONTENT_TYPE_LATEST if PROMETHEUS_AVAILABLE else "text/plain"


class MockMetric:
    """模拟指标类（当prometheus_client不可用时使用）"""

    def inc(self, value: float = 1.0) -> None:
        pass

    def observe(self, value: float) -> None:
        pass

    def set(self, value: float) -> None:
        pass

    def labels(self, **kwargs) -> 'MockMetric':
        return self

    def info(self, info_dict: Dict[str, str]) -> None:
        pass


# 便捷装饰器

def timed(
    metric_name: str,
    labels: Optional[Dict[str, str]] = None
) -> Callable[[F], F]:
    """
    计时装饰器

    Args:
        metric_name: 直方图指标名称
        labels: 标签值

    Example:
        @timed("my_function_duration_seconds", {"module": "trading"})
        async def my_function():
            pass
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            collector = MetricsCollector()
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                collector.observe_histogram(metric_name, duration, labels)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            collector = MetricsCollector()
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                collector.observe_histogram(metric_name, duration, labels)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper  # type: ignore
    return decorator


def counted(
    metric_name: str,
    labels: Optional[Dict[str, str]] = None
) -> Callable[[F], F]:
    """
    计数装饰器

    Args:
        metric_name: 计数器指标名称
        labels: 标签值
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            collector = MetricsCollector()
            collector.increment_counter(metric_name, labels)
            return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            collector = MetricsCollector()
            collector.increment_counter(metric_name, labels)
            return func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper  # type: ignore
    return decorator


@contextmanager
def timed_block(
    metric_name: str,
    labels: Optional[Dict[str, str]] = None
) -> Generator[None, None, None]:
    """
    计时上下文管理器

    Example:
        with timed_block("db_query_duration_seconds", {"operation": "select"}):
            result = db.execute(query)
    """
    collector = MetricsCollector()
    start_time = time.time()

    try:
        yield
    finally:
        duration = time.time() - start_time
        collector.observe_histogram(metric_name, duration, labels)


class PerformanceMonitor:
    """
    性能监控器

    提供高级性能监控功能
    """

    def __init__(self):
        self._collector = MetricsCollector()
        self._active_timers: Dict[str, float] = {}

    def record_http_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration_seconds: float
    ) -> None:
        """记录HTTP请求"""
        self._collector.increment_counter(
            "http_requests_total",
            {"method": method, "endpoint": endpoint, "status": str(status_code)}
        )
        self._collector.observe_histogram(
            "http_request_duration_seconds",
            duration_seconds,
            {"method": method, "endpoint": endpoint}
        )

    def record_trade(
        self,
        symbol: str,
        side: str,
        status: str,
        latency_seconds: float
    ) -> None:
        """记录交易"""
        self._collector.increment_counter(
            "trades_total",
            {"symbol": symbol, "side": side, "status": status}
        )
        self._collector.observe_histogram(
            "trade_latency_seconds",
            latency_seconds,
            {"symbol": symbol}
        )

    def record_inference(
        self,
        model: str,
        status: str,
        duration_seconds: float,
        cached: bool = False
    ) -> None:
        """记录模型推理"""
        self._collector.increment_counter(
            "inference_requests_total",
            {"model": model, "status": status}
        )
        self._collector.observe_histogram(
            "inference_duration_seconds",
            duration_seconds,
            {"model": model}
        )

        if cached:
            self._collector.increment_counter(
                "model_cache_hits_total",
                {"model": model}
            )

    def record_batch_processing(
        self,
        processor: str,
        batch_size: int,
        duration_seconds: float
    ) -> None:
        """记录批处理"""
        self._collector.set_gauge(
            "batch_size",
            float(batch_size),
            {"processor": processor}
        )
        self._collector.observe_histogram(
            "batch_processing_seconds",
            duration_seconds,
            {"processor": processor}
        )

    def record_cache_access(
        self,
        cache_level: str,
        hit: bool
    ) -> None:
        """记录缓存访问"""
        if hit:
            self._collector.increment_counter(
                "cache_hits_total",
                {"cache_level": cache_level}
            )
        else:
            self._collector.increment_counter(
                "cache_misses_total",
                {"cache_level": cache_level}
            )

    def update_active_connections(self, service: str, count: int) -> None:
        """更新活跃连接数"""
        self._collector.set_gauge(
            "active_connections",
            float(count),
            {"service": service}
        )

    def update_queue_size(self, queue_name: str, size: int) -> None:
        """更新队列大小"""
        self._collector.set_gauge(
            "queue_size",
            float(size),
            {"queue_name": queue_name}
        )

    def update_db_connections(self, pool: str, count: int) -> None:
        """更新数据库连接数"""
        self._collector.set_gauge(
            "db_connections_active",
            float(count),
            {"pool": pool}
        )

    def start_timer(self, name: str) -> None:
        """启动计时器"""
        self._active_timers[name] = time.time()

    def stop_timer(self, name: str, metric_name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """停止计时器并记录"""
        if name not in self._active_timers:
            return 0.0

        duration = time.time() - self._active_timers[name]
        self._collector.observe_histogram(metric_name, duration, labels)
        del self._active_timers[name]
        return duration


# 全局监控器实例
monitor = PerformanceMonitor()


def get_metrics_endpoint() -> Tuple[bytes, str]:
    """
    获取Prometheus指标端点数据

    Returns:
        (数据, Content-Type)
    """
    collector = MetricsCollector()
    return collector.generate_metrics(), collector.get_content_type()


def start_metrics_server(port: int = 9090) -> None:
    """
    启动指标服务器

    Args:
        port: 服务端口
    """
    collector = MetricsCollector()
    collector.start_server(port)


def create_fastapi_middleware():
    """
    创建FastAPI中间件

    Returns:
        FastAPI中间件类
    """
    try:
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.requests import Request
        from starlette.responses import Response

        class PrometheusMiddleware(BaseHTTPMiddleware):
            """Prometheus监控中间件"""

            async def dispatch(self, request: Request, call_next):
                start_time = time.time()

                response = await call_next(request)

                duration = time.time() - start_time
                monitor.record_http_request(
                    method=request.method,
                    endpoint=request.url.path,
                    status_code=response.status_code,
                    duration_seconds=duration
                )

                return response

        return PrometheusMiddleware

    except ImportError:
        logger.warning("starlette not installed, cannot create FastAPI middleware")
        return None
