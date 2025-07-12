import time
from typing import Dict, Any, Optional, Callable, Type, List, Tuple
import threading
from enum import Enum
from collections import defaultdict, OrderedDict
from prometheus_client import Counter, Gauge, Histogram
from .security import SecurityFilter, SecurityContext, FilterResult

class TradingErrorType(Enum):
    """交易错误类型枚举"""
    CONNECTION = "connection"
    TIMEOUT = "timeout"
    LIMIT = "limit"
    VALIDATION = "validation"
    ORDER_REJECTED = "order_rejected"
    INVALID_PRICE = "invalid_price"
    UNKNOWN = "unknown"

class OrderRejectedError(Exception):
    """订单被拒绝异常"""
    def __init__(self, reason: str, order_id: Optional[str] = None):
        self.reason = reason
        self.order_id = order_id
        super().__init__(f"Order rejected: {reason}")

class InvalidPriceError(Exception):
    """无效价格异常"""
    def __init__(self, price: float, valid_range: tuple):
        self.price = price
        self.valid_range = valid_range
        super().__init__(
            f"Invalid price {price}, valid range: {valid_range[0]}-{valid_range[1]}"
        )

class CircuitBreaker:
    """熔断器实现"""
    
    STATE_METRICS = Gauge(
        'circuit_breaker_state', 
        'Current state (0=closed,1=half-open,2=open)', 
        ['breaker']
    )
    STATE_CHANGE_COUNTER = Counter(
        'circuit_breaker_state_changes', 
        'State transition count', 
        ['breaker', 'from', 'to']
    )
    
    def __init__(self, name: str, threshold: int = 5):
        self.name = name
        self.threshold = threshold
        self._failures = 0
        self.state = "closed"
        self.STATE_METRICS.labels(self.name).set(0)  # 0=closed
        
    def trip(self):
        """触发熔断"""
        old_state = self.state
        self.state = "open"
        self.STATE_METRICS.labels(self.name).set(2)  # 2=open
        self.STATE_CHANGE_COUNTER.labels(
            self.name, old_state, "open"
        ).inc()
        
    def reset(self):
        """重置熔断器"""
        old_state = self.state
        self.state = "closed"
        self._failures = 0
        self.STATE_METRICS.labels(self.name).set(0)
        self.STATE_CHANGE_COUNTER.labels(
            self.name, old_state, "closed"
        ).inc()
        
    def attempt_reset(self):
        """尝试半开状态"""
        old_state = self.state
        self.state = "half-open"
        self.STATE_METRICS.labels(self.name).set(1)  # 1=half-open
        self.STATE_CHANGE_COUNTER.labels(
            self.name, old_state, "half-open"
        ).inc()

class TradingErrorHandler:
    """增强版交易错误处理器"""

    _error_counter = Counter(
        'trading_errors_total',
        'Total trading errors by type',
        ['error_type']
    )
    _processing_time = Gauge(
        'trading_error_processing_seconds',
        'Error processing time in seconds'
    )
    _recovery_success = Counter(
        'trading_error_recovery_success',
        'Successful error recovery',
        ['error_type', 'strategy']
    )
    _recovery_time = Histogram(
        'trading_error_recovery_time',
        'Error recovery time in seconds',
        ['error_type']
    )

    def __init__(
        self,
        default_strategy: Optional[Callable] = None,
        error_classifiers: Optional[Dict[str, Callable]] = None,
        security_filter: Optional[Callable] = None,
        sensitive_fields: Optional[List[str]] = None,
        **kwargs
    ):
        """初始化交易错误处理器"""
        self.default_strategy = default_strategy or self._default_handle_strategy
        self.error_classifiers = error_classifiers or {}
        self._strategies = defaultdict(list)
        self._handlers = OrderedDict()
        self._lock = threading.Lock()
        
        # 初始化安全上下文
        self._security_ctx = SecurityContext()
        if security_filter is not None:
            self._security_ctx.add_filter(security_filter)
        else:
            # 添加默认安全过滤器
            self._security_ctx.add_filter(
                SecurityFilter(
                    sensitive_fields=sensitive_fields or [
                        'api_key', 'secret', 'password',
                        'token', 'credentials', 'private_key'
                    ],
                    patterns=[
                        r'[A-Za-z0-9]{32}',  # MD5哈希
                        r'sk_(live|test)_[A-Za-z0-9]{24}',  # Stripe密钥
                        r'AKIA[0-9A-Z]{16}',  # AWS访问密钥
                        r'[0-9]{12,19}',  # 长数字(可能为卡号)
                        r'eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*',  # JWT
                        r'sk-[a-zA-Z0-9]{32}',  # OpenAI密钥
                        r'gh[pousr]_[A-Za-z0-9]{36}',  # GitHub令牌
                        r'xoxb-[0-9]{11}-[0-9]{11}-[a-zA-Z0-9]{24}'  # Slack令牌
                    ]
                )
            )

        # 注册内置处理器和分类器
        self._register_trading_handlers()
        self._register_default_classifiers()

    def _register_trading_handlers(self):
        """注册内置交易错误处理器"""
        with self._lock:
            # 订单拒绝处理
            self.register_handler(
                OrderRejectedError,
                self._handle_order_rejection
            )

            # 无效价格处理
            self.register_handler(
                InvalidPriceError,
                self._handle_price_error
            )

            # 标准异常处理器
            self.register_handler(
                ConnectionError,
                lambda e,o,c: self._default_handle_strategy(e, o, c)
            )
            self.register_handler(
                TimeoutError,
                lambda e,o,c: self._default_handle_strategy(e, o, c)
            )

    def _handle_order_rejection(self, error: OrderRejectedError, order: Dict, context: Dict) -> Dict:
        """处理订单拒绝错误"""
        start_time = time.time()
        result = {'status': 'failed', 'error_type': 'order_rejected'}

        try:
            # 根据拒绝原因采取不同策略
            if error.reason == "INSUFFICIENT_FUNDS":
                result.update(self._fund_reallocation(order, context))
                self._recovery_success.labels(
                    error_type="order_rejected",
                    strategy="fund_reallocation"
                ).inc()

            elif error.reason == "POSITION_LIMIT":
                result.update(self._adjust_position(order, context))
                self._recovery_success.labels(
                    error_type="order_rejected",
                    strategy="position_adjustment"
                ).inc()

            else:
                result['message'] = f"Unhandled rejection reason: {error.reason}"

        except Exception as e:
            result['recovery_error'] = str(e)
        finally:
            self._recovery_time.labels(
                error_type="order_rejected"
            ).observe(time.time() - start_time)

        return result

    def _handle_price_error(self, error: InvalidPriceError, order: Dict, context: Dict) -> Dict:
        """处理无效价格错误"""
        start_time = time.time()
        result = {'status': 'failed', 'error_type': 'invalid_price'}

        try:
            # 尝试价格修正
            new_price = self._correct_price(
                error.price,
                error.valid_range
            )

            if new_price:
                result.update({
                    'status': 'corrected',
                    'original_price': error.price,
                    'corrected_price': new_price
                })
                self._recovery_success.labels(
                    error_type="invalid_price",
                    strategy="price_correction"
                ).inc()
            else:
                result['message'] = "Price correction failed"

        except Exception as e:
            result['recovery_error'] = str(e)
        finally:
            self._recovery_time.labels(
                error_type="invalid_price"
            ).observe(time.time() - start_time)

        return result

    def _fund_reallocation(self, order: Dict, context: Dict) -> Dict:
        """资金重分配策略"""
        # 实现资金重分配逻辑
        return {
            'strategy': 'fund_reallocation',
            'details': 'Attempting to reallocate funds'
        }

    def _adjust_position(self, order: Dict, context: Dict) -> Dict:
        """仓位调整策略"""
        # 实现仓位调整逻辑
        return {
            'strategy': 'position_adjustment',
            'details': 'Attempting to adjust positions'
        }

    def _correct_price(self, price: float, valid_range: tuple) -> Optional[float]:
        """价格修正策略"""
        # 实现价格修正逻辑
        min_price, max_price = valid_range
        if price < min_price:
            return min_price
        elif price > max_price:
            return max_price
        return None

    def handle_error(self, error: Exception, order: Dict, context: Dict) -> Dict:
        """处理交易错误并返回增强结果
        
        Args:
            error: 发生的异常
            order: 相关订单数据
            context: 处理上下文
            
        Returns:
            处理结果字典，包含:
            - status: 处理状态 (handled/failed/corrected)
            - error_type: 错误类型
            - filtered: 是否进行了敏感数据过滤
            - details: 处理详情
            - timestamp: 处理时间戳
        """
        error_type = self._classify_error(error)
        self._error_counter.labels(error_type=error_type).inc()
        
        start_time = time.time()
        result = {
            'status': 'failed',
            'error_type': error_type,
            'filtered': False,
            'details': {},
            'timestamp': time.time()
        }
        
        try:
            # 应用安全过滤器并记录过滤情况
            filter_result = self._security_ctx.filter_context(context)
            filtered_ctx = filter_result.filtered
            result['filtered'] = filter_result.modified
            result['details']['filtered_fields'] = filter_result.matched_fields
            
            # 查找匹配的处理策略
            handler = self._find_handler(error)
            if handler is not None:
                handler_result = handler(error, order, filtered_ctx)
                result['details'].update(handler_result)
            else:
                default_result = self.default_strategy(error, order, filtered_ctx)
                result['details'].update(default_result)
                
            result['status'] = 'handled'
            return result
        except Exception as e:
            result['details']['handler_error'] = str(e)
            return result
        finally:
            self._processing_time.set(time.time() - start_time)

    def _classify_error(self, error: Exception) -> str:
        """分类错误类型"""
        for error_type, classifier in self.error_classifiers.items():
            if classifier(error):
                return error_type
                
        if isinstance(error, OrderRejectedError):
            return "order_rejected"
        elif isinstance(error, InvalidPriceError):
            return "invalid_price"
        elif isinstance(error, ConnectionError):
            return "connection"
        elif isinstance(error, TimeoutError):
            return "timeout"
            
        return "unknown"

    def _find_handler(self, error: Exception) -> Optional[Callable]:
        """查找匹配的错误处理器"""
        error_type = self._classify_error(error)
        with self._lock:
            for handler_type, handler in self._handlers.items():
                if isinstance(error, handler_type):
                    return handler
            return None

    def _default_handle_strategy(self, error: Exception, order: Dict, context: Dict) -> Dict:
        """默认错误处理策略"""
        return {
            'status': 'failed',
            'message': f"No specific handler for {type(error).__name__}",
            'details': {'error': str(error)}
        }

    def register_handler(self, error_type: Type[Exception], handler: Callable):
        """注册错误处理器"""
        with self._lock:
            self._handlers[error_type] = handler

    def add_filter(self, filter_func: Callable):
        """添加安全过滤器"""
        self._security_ctx.add_filter(filter_func)
        
    def get_filter_stats(self) -> Dict[str, Any]:
        """获取过滤器统计信息"""
        return {
            'filter_count': len(self._security_ctx.filters),
            'sensitive_patterns': [
                p.pattern for f in self._security_ctx.filters 
                if isinstance(f, SecurityFilter) 
                for p in f.patterns
            ]
        }
