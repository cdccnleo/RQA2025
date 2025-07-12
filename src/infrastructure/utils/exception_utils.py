"""
异常处理工具模块
"""
import logging
import sys
from typing import Optional, Dict, Any
from functools import wraps
from datetime import datetime


class DataLoaderError(Exception):
    """数据加载异常基类

    属性：
        original_exception (Exception): 原始异常对象
        timestamp (datetime): 异常发生时间
    """

    def __init__(self, message: str, original_exception: Optional[BaseException] = None):
        super().__init__(message)
        self.original_exception = original_exception
        self.timestamp = datetime.now()

    def __str__(self):
        """返回包含时间戳、错误信息和原始异常的字符串表示"""
        timestamp_str = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")  # 使用标准时间格式
        base_msg = f"DataLoaderError[{timestamp_str}]: {super().__str__()}"
        if self.original_exception:
            return f"{base_msg} (原始异常: {type(self.original_exception).__name__}: {str(self.original_exception)})"
        return base_msg


class AShareException(Exception):
    """A股市场异常基类"""
    def __init__(self, message: str, symbol: Optional[str] = None):
        self.message = message
        self.symbol = symbol
        self.timestamp = datetime.now()
        super().__init__(self.formatted_message)

    @property
    def formatted_message(self) -> str:
        return f"[{self.timestamp}] {self.message}" + (
            f" (标的: {self.symbol})" if self.symbol else ""
        )

class LimitViolation(AShareException):
    """涨跌停规则违反异常"""
    def __init__(self, symbol: str, price: float, limit: float, is_upper: bool):
        self.limit = limit
        self.is_upper = is_upper
        limit_type = "涨停" if is_upper else "跌停"
        super().__init__(
            f"价格{price}违反{limit_type}限制{limit}",
            symbol
        )

class T1Violation(AShareException):
    """T+1违规异常"""
    def __init__(self, symbol: str, operation: str):
        super().__init__(
            f"尝试在T+1规则下{operation}尚未结算的{symbol}",
            symbol
        )

class DataValidationError(AShareException):
    """数据校验失败异常"""
    def __init__(self, message: str, data: Optional[Dict] = None):
        self.data = data
        super().__init__(message)

class BrokerAPIError(AShareException):
    """券商API异常"""
    def __init__(self, broker: str, api: str, error: Any):
        super().__init__(
            f"券商{broker}的API{api}调用失败: {str(error)}"
        )
        self.broker = broker
        self.api = api
        self.original_error = error

class ExceptionHandler:
    """异常处理器"""

    @staticmethod
    def handle(exception: Exception, context: Optional[Dict] = None, log_level: str = 'ERROR'):
        """
        处理异常并采取相应措施
        Args:
            exception: 捕获的异常
            context: 异常发生时的上下文信息
            log_level: 日志级别，默认为'ERROR'
        """
        # 初始化日志记录器
        logger = logging.getLogger("exception")
        log_method = getattr(logger, log_level.lower(), logger.error)

        # logging保留字段，不能出现在extra中
        RESERVED_LOG_KEYS = {
            'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename', 'module',
            'exc_info', 'exc_text', 'stack_info', 'lineno', 'funcName', 'created', 'msecs',
            'relativeCreated', 'thread', 'threadName', 'processName', 'process', 'message',
        }
        def filter_extra(ctx):
            if not ctx:
                return {}
            return {k: v for k, v in ctx.items() if k not in RESERVED_LOG_KEYS}

        extra = filter_extra(context)

        # 根据异常类型采取不同措施
        if isinstance(exception, LimitViolation):
            log_method(f"风控拦截: {str(exception)}", extra=extra)
            # 自动调整订单价格到涨跌停限制内
            if context and 'order' in context:
                adjusted = ExceptionHandler._adjust_order_to_limit(
                    context['order'],
                    exception
                )
                return adjusted

        elif isinstance(exception, T1Violation):
            log_method(f"交易规则违反: {str(exception)}", extra=extra)
            # 自动取消违规订单
            if context and 'order' in context:
                ExceptionHandler._cancel_order(context['order'])

        elif isinstance(exception, DataValidationError):
            log_method(f"数据校验失败: {str(exception)}", extra=extra)
            # 尝试使用备用数据源
            if context and 'data_source' in context:
                ExceptionHandler._switch_data_source(context['data_source'])

        elif isinstance(exception, BrokerAPIError):
            log_method(f"券商接口错误: {str(exception)}", extra=extra)
            # 切换备用券商通道
            if context and 'broker' in context:
                ExceptionHandler._switch_broker(context['broker'])

        else:
            # 通用异常处理
            logger.exception("未处理的异常发生", exc_info=True, extra=extra)

        # 默认返回None表示未处理或处理失败
        return None

    @staticmethod
    def retry(max_attempts: int = 3, delay: float = 1.0):
        """
        重试装饰器
        Args:
            max_attempts: 最大重试次数
            delay: 重试间隔(秒)
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                for attempt in range(1, max_attempts + 1):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        logging.warning(
                            f"尝试 {attempt}/{max_attempts} 失败: {str(e)}"
                        )
                        if attempt < max_attempts:
                            import time
                            time.sleep(delay)
                raise last_exception
            return wrapper
        return decorator

    @staticmethod
    def _adjust_order_to_limit(order: Dict, exception: LimitViolation) -> Dict:
        """调整订单价格到涨跌停限制内"""
        adjusted = order.copy()
        if isinstance(exception, LimitViolation):
            if exception.is_upper:
                adjusted['price'] = exception.limit - 0.01  # 低1分钱
            else:
                adjusted['price'] = exception.limit + 0.01  # 高1分钱
        return adjusted

    @staticmethod
    def _cancel_order(order: Dict):
        """取消订单"""
        # 实际实现需要调用交易引擎的取消接口
        logging.info(f"取消违规订单: {order.get('order_id')}")

    @staticmethod
    def _switch_data_source(source: str):
        """切换数据源"""
        logging.info(f"切换到备用数据源: {source}")

    @staticmethod
    def _switch_broker(broker: str):
        """切换券商通道"""
        logging.info(f"切换到备用券商: {broker}")

def is_timeout_error(exception: Exception) -> bool:
    """判断异常是否为超时错误
    
    Args:
        exception: 要检查的异常对象
        
    Returns:
        bool: 如果是超时错误返回True，否则返回False
    """
    # 检查Python内置超时异常
    if isinstance(exception, TimeoutError):
        return True
        
    # 检查requests库的超时异常
    try:
        from requests.exceptions import Timeout as RequestsTimeout
        if isinstance(exception, RequestsTimeout):
            return True
    except ImportError:
        pass
        
    # 检查socket超时异常
    try:
        from socket import timeout as SocketTimeout
        if isinstance(exception, SocketTimeout):
            return True
    except ImportError:
        pass
        
    # 检查异常类型名称中包含"Timeout"
    exception_type = type(exception).__name__
    if "Timeout" in exception_type:
        return True
        
    return False


class CircuitBreaker:
    """熔断机制控制器"""

    @staticmethod
    def check_market_condition() -> bool:
        """检查市场条件是否触发熔断"""
        # TODO: 实现实际的市场条件检查逻辑
        return False

    @staticmethod
    def activate() -> bool:
        """激活熔断机制"""
        logging.critical("市场熔断机制激活 - 暂停所有交易")
        # TODO: 实现实际的熔断逻辑
        return True

    @staticmethod
    def deactivate() -> bool:
        """解除熔断机制"""
        logging.info("市场熔断机制解除 - 恢复交易")
        # TODO: 实现实际的恢复逻辑
        return True
