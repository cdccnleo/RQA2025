"""
监控指标采集模块

提供系统性能指标、业务指标和自定义指标的采集功能，
集成Prometheus客户端实现指标暴露。

作者: 李明
创建日期: 2026-03-28
版本: 1.0.0
"""

import time
import functools
from typing import Callable, Optional, Dict, Any, List
from contextlib import contextmanager
import logging

# 尝试导入prometheus客户端
# 如果未安装，提供降级方案
logger = logging.getLogger(__name__)

PROMETHEUS_AVAILABLE = False
try:
    from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST
    from prometheus_client.core import CollectorRegistry
    PROMETHEUS_AVAILABLE = True
    logger.info("✅ Prometheus客户端加载成功")
except ImportError:
    logger.warning("⚠️ Prometheus客户端未安装，使用降级方案")


class MetricsCollector:
    """
    指标采集器
    
    负责管理系统所有监控指标的采集、存储和暴露。
    支持系统指标、业务指标和自定义指标。
    
    Attributes:
        registry: Prometheus注册表
        metrics: 指标字典
        enabled: 是否启用监控
    """
    
    def __init__(self, enabled: bool = True, namespace: str = "rqa2025"):
        """
        初始化指标采集器
        
        Args:
            enabled: 是否启用监控
            namespace: 指标命名空间
        """
        self.enabled = enabled and PROMETHEUS_AVAILABLE
        self.namespace = namespace
        self.metrics: Dict[str, Any] = {}
        self.registry = CollectorRegistry() if PROMETHEUS_AVAILABLE else None
        
        if self.enabled:
            self._init_system_metrics()
            self._init_business_metrics()
            logger.info(f"✅ 指标采集器初始化完成，命名空间: {namespace}")
        else:
            logger.warning("⚠️ 指标采集器已禁用或Prometheus不可用")
    
    def _init_system_metrics(self):
        """初始化系统指标"""
        # HTTP请求指标
        self.metrics['http_requests_total'] = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status'],
            namespace=self.namespace,
            registry=self.registry
        )
        
        self.metrics['http_request_duration_seconds'] = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint'],
            buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            namespace=self.namespace,
            registry=self.registry
        )
        
        # 系统资源指标
        self.metrics['system_cpu_usage'] = Gauge(
            'system_cpu_usage',
            'System CPU usage percentage',
            namespace=self.namespace,
            registry=self.registry
        )
        
        self.metrics['system_memory_usage_bytes'] = Gauge(
            'system_memory_usage_bytes',
            'System memory usage in bytes',
            namespace=self.namespace,
            registry=self.registry
        )
        
        self.metrics['system_memory_total_bytes'] = Gauge(
            'system_memory_total_bytes',
            'System total memory in bytes',
            namespace=self.namespace,
            registry=self.registry
        )
        
        # 应用信息
        self.metrics['app_info'] = Info(
            'app',
            'Application information',
            namespace=self.namespace,
            registry=self.registry
        )
        self.metrics['app_info'].info({
            'version': '1.0.0',
            'name': 'RQA2025',
            'environment': 'production'
        })
    
    def _init_business_metrics(self):
        """初始化业务指标"""
        # 策略执行指标
        self.metrics['strategy_executions_total'] = Counter(
            'strategy_executions_total',
            'Total strategy executions',
            ['strategy_id', 'status'],
            namespace=self.namespace,
            registry=self.registry
        )
        
        self.metrics['strategy_execution_duration_seconds'] = Histogram(
            'strategy_execution_duration_seconds',
            'Strategy execution duration in seconds',
            ['strategy_id'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            namespace=self.namespace,
            registry=self.registry
        )
        
        # 交易指标
        self.metrics['trades_total'] = Counter(
            'trades_total',
            'Total trades executed',
            ['symbol', 'side', 'status'],
            namespace=self.namespace,
            registry=self.registry
        )
        
        self.metrics['trade_volume'] = Counter(
            'trade_volume',
            'Trade volume',
            ['symbol', 'side'],
            namespace=self.namespace,
            registry=self.registry
        )
        
        # 数据处理指标
        self.metrics['data_processing_records_total'] = Counter(
            'data_processing_records_total',
            'Total data records processed',
            ['source', 'operation'],
            namespace=self.namespace,
            registry=self.registry
        )
        
        self.metrics['data_processing_duration_seconds'] = Histogram(
            'data_processing_duration_seconds',
            'Data processing duration in seconds',
            ['source', 'operation'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
            namespace=self.namespace,
            registry=self.registry
        )
        
        # 特征计算指标
        self.metrics['feature_computations_total'] = Counter(
            'feature_computations_total',
            'Total feature computations',
            ['feature_name', 'status'],
            namespace=self.namespace,
            registry=self.registry
        )
        
        self.metrics['feature_computation_duration_seconds'] = Histogram(
            'feature_computation_duration_seconds',
            'Feature computation duration in seconds',
            ['feature_name'],
            buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1],
            namespace=self.namespace,
            registry=self.registry
        )
        
        # 模型推理指标
        self.metrics['model_inferences_total'] = Counter(
            'model_inferences_total',
            'Total model inferences',
            ['model_id', 'status'],
            namespace=self.namespace,
            registry=self.registry
        )
        
        self.metrics['model_inference_duration_seconds'] = Histogram(
            'model_inference_duration_seconds',
            'Model inference duration in seconds',
            ['model_id'],
            buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25],
            namespace=self.namespace,
            registry=self.registry
        )
        
        # 缓存指标
        self.metrics['cache_hits_total'] = Counter(
            'cache_hits_total',
            'Total cache hits',
            ['cache_name'],
            namespace=self.namespace,
            registry=self.registry
        )
        
        self.metrics['cache_misses_total'] = Counter(
            'cache_misses_total',
            'Total cache misses',
            ['cache_name'],
            namespace=self.namespace,
            registry=self.registry
        )
        
        # 数据库指标
        self.metrics['db_connections_active'] = Gauge(
            'db_connections_active',
            'Active database connections',
            ['pool_name'],
            namespace=self.namespace,
            registry=self.registry
        )
        
        self.metrics['db_query_duration_seconds'] = Histogram(
            'db_query_duration_seconds',
            'Database query duration in seconds',
            ['operation'],
            buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
            namespace=self.namespace,
            registry=self.registry
        )
    
    def record_http_request(self, method: str, endpoint: str, status: int, duration: float):
        """
        记录HTTP请求指标
        
        Args:
            method: HTTP方法
            endpoint: 请求端点
            status: HTTP状态码
            duration: 请求耗时（秒）
        """
        if not self.enabled:
            return
        
        status_label = str(status)
        self.metrics['http_requests_total'].labels(
            method=method,
            endpoint=endpoint,
            status=status_label
        ).inc()
        
        self.metrics['http_request_duration_seconds'].labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def record_strategy_execution(self, strategy_id: str, status: str, duration: float):
        """
        记录策略执行指标
        
        Args:
            strategy_id: 策略ID
            status: 执行状态
            duration: 执行耗时（秒）
        """
        if not self.enabled:
            return
        
        self.metrics['strategy_executions_total'].labels(
            strategy_id=strategy_id,
            status=status
        ).inc()
        
        self.metrics['strategy_execution_duration_seconds'].labels(
            strategy_id=strategy_id
        ).observe(duration)
    
    def record_trade(self, symbol: str, side: str, volume: float, status: str = "success"):
        """
        记录交易指标
        
        Args:
            symbol: 交易标的
            side: 交易方向
            volume: 交易量
            status: 交易状态
        """
        if not self.enabled:
            return
        
        self.metrics['trades_total'].labels(
            symbol=symbol,
            side=side,
            status=status
        ).inc()
        
        self.metrics['trade_volume'].labels(
            symbol=symbol,
            side=side
        ).inc(volume)
    
    def record_data_processing(self, source: str, operation: str, duration: float):
        """
        记录数据处理指标
        
        Args:
            source: 数据源
            operation: 操作类型
            duration: 处理耗时（秒）
        """
        if not self.enabled:
            return
        
        self.metrics['data_processing_records_total'].labels(
            source=source,
            operation=operation
        ).inc()
        
        self.metrics['data_processing_duration_seconds'].labels(
            source=source,
            operation=operation
        ).observe(duration)
    
    def record_feature_computation(self, feature_name: str, status: str, duration: float):
        """
        记录特征计算指标
        
        Args:
            feature_name: 特征名称
            status: 计算状态
            duration: 计算耗时（秒）
        """
        if not self.enabled:
            return
        
        self.metrics['feature_computations_total'].labels(
            feature_name=feature_name,
            status=status
        ).inc()
        
        self.metrics['feature_computation_duration_seconds'].labels(
            feature_name=feature_name
        ).observe(duration)
    
    def record_model_inference(self, model_id: str, status: str, duration: float):
        """
        记录模型推理指标
        
        Args:
            model_id: 模型ID
            status: 推理状态
            duration: 推理耗时（秒）
        """
        if not self.enabled:
            return
        
        self.metrics['model_inferences_total'].labels(
            model_id=model_id,
            status=status
        ).inc()
        
        self.metrics['model_inference_duration_seconds'].labels(
            model_id=model_id
        ).observe(duration)
    
    def record_cache_access(self, cache_name: str, hit: bool):
        """
        记录缓存访问指标
        
        Args:
            cache_name: 缓存名称
            hit: 是否命中
        """
        if not self.enabled:
            return
        
        if hit:
            self.metrics['cache_hits_total'].labels(cache_name=cache_name).inc()
        else:
            self.metrics['cache_misses_total'].labels(cache_name=cache_name).inc()
    
    def update_db_connections(self, pool_name: str, count: int):
        """
        更新数据库连接数
        
        Args:
            pool_name: 连接池名称
            count: 连接数
        """
        if not self.enabled:
            return
        
        self.metrics['db_connections_active'].labels(pool_name=pool_name).set(count)
    
    def record_db_query(self, operation: str, duration: float):
        """
        记录数据库查询指标
        
        Args:
            operation: 操作类型
            duration: 查询耗时（秒）
        """
        if not self.enabled:
            return
        
        self.metrics['db_query_duration_seconds'].labels(operation=operation).observe(duration)
    
    def update_system_metrics(self):
        """更新系统资源指标"""
        if not self.enabled:
            return
        
        try:
            import psutil
            
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics['system_cpu_usage'].set(cpu_percent)
            
            # 内存使用
            memory = psutil.virtual_memory()
            self.metrics['system_memory_usage_bytes'].set(memory.used)
            self.metrics['system_memory_total_bytes'].set(memory.total)
            
        except ImportError:
            logger.debug("psutil未安装，跳过系统指标采集")
        except Exception as e:
            logger.error(f"更新系统指标失败: {e}")
    
    def get_metrics_text(self) -> str:
        """
        获取指标文本格式（Prometheus格式）
        
        Returns:
            Prometheus格式的指标文本
        """
        if not self.enabled or not PROMETHEUS_AVAILABLE:
            return "# Prometheus metrics disabled\n"
        
        return generate_latest(self.registry).decode('utf-8')
    
    def get_metrics_content_type(self) -> str:
        """
        获取指标Content-Type
        
        Returns:
            Content-Type字符串
        """
        if PROMETHEUS_AVAILABLE:
            return CONTENT_TYPE_LATEST
        return "text/plain"


# 全局指标采集器实例
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector(enabled: bool = True, namespace: str = "rqa2025") -> MetricsCollector:
    """
    获取全局指标采集器实例
    
    Args:
        enabled: 是否启用
        namespace: 命名空间
        
    Returns:
        MetricsCollector实例
    """
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector(enabled=enabled, namespace=namespace)
    return _metrics_collector


def timed(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """
    计时装饰器
    
    用于自动记录函数执行时间
    
    Args:
        metric_name: 指标名称
        labels: 指标标签
        
    Returns:
        装饰器函数
        
    Example:
        @timed('operation_duration', {'type': 'calculation'})
        def my_function():
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            collector = get_metrics_collector()
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                status = "success"
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                # 这里可以根据metric_name记录到对应的指标
                # 简化实现，实际使用时需要更复杂的映射逻辑
                
        return wrapper
    return decorator


@contextmanager
def timed_context(metric_name: str, **labels):
    """
    计时上下文管理器
    
    用于记录代码块执行时间
    
    Args:
        metric_name: 指标名称
        **labels: 指标标签
        
    Example:
        with timed_context('operation_duration', type='calculation'):
            # 执行操作
            pass
    """
    start_time = time.time()
    status = "success"
    
    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        duration = time.time() - start_time
        # 记录指标（简化实现）
        logger.debug(f"{metric_name} completed in {duration:.4f}s with status {status}")


class MetricsMiddleware:
    """
    Web框架中间件基类
    
    用于自动记录HTTP请求指标
    """
    
    def __init__(self, app=None, collector: Optional[MetricsCollector] = None):
        self.app = app
        self.collector = collector or get_metrics_collector()
    
    def process_request(self, request):
        """处理请求开始"""
        request._metrics_start_time = time.time()
    
    def process_response(self, request, response):
        """处理请求结束"""
        if hasattr(request, '_metrics_start_time'):
            duration = time.time() - request._metrics_start_time
            
            # 提取请求信息
            method = getattr(request, 'method', 'UNKNOWN')
            endpoint = getattr(request, 'path', 'unknown')
            status = getattr(response, 'status_code', 200)
            
            self.collector.record_http_request(method, endpoint, status, duration)
        
        return response


# 便捷函数
def record_strategy_execution(strategy_id: str, status: str, duration: float):
    """记录策略执行"""
    get_metrics_collector().record_strategy_execution(strategy_id, status, duration)


def record_trade(symbol: str, side: str, volume: float, status: str = "success"):
    """记录交易"""
    get_metrics_collector().record_trade(symbol, side, volume, status)


def record_data_processing(source: str, operation: str, duration: float):
    """记录数据处理"""
    get_metrics_collector().record_data_processing(source, operation, duration)


def record_feature_computation(feature_name: str, status: str, duration: float):
    """记录特征计算"""
    get_metrics_collector().record_feature_computation(feature_name, status, duration)


def record_model_inference(model_id: str, status: str, duration: float):
    """记录模型推理"""
    get_metrics_collector().record_model_inference(model_id, status, duration)


def record_cache_access(cache_name: str, hit: bool):
    """记录缓存访问"""
    get_metrics_collector().record_cache_access(cache_name, hit)


def update_system_metrics():
    """更新系统指标"""
    get_metrics_collector().update_system_metrics()
