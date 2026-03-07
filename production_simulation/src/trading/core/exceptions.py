"""
交易层异常处理
Trading Layer Exception Handling

定义交易相关的异常类和错误处理机制
"""


class TradingException(Exception):
    """交易基础异常类"""

    def __init__(self, message: str, error_code: int = -1):
        super().__init__(message)
        self.error_code = error_code
        self.message = message


class OrderException(TradingException):
    """订单异常"""

    def __init__(self, message: str, order_id: str = None):
        super().__init__(f"订单异常 - {order_id}: {message}")
        self.order_id = order_id


class ExecutionException(TradingException):
    """执行异常"""

    def __init__(self, message: str, execution_id: str = None):
        super().__init__(f"执行异常 - {execution_id}: {message}")
        self.execution_id = execution_id


class ConnectionException(TradingException):
    """连接异常"""

    def __init__(self, message: str, broker_name: str = None):
        super().__init__(f"连接异常 - {broker_name}: {message}")
        self.broker_name = broker_name


class InsufficientFundsException(TradingException):
    """资金不足异常"""

    def __init__(self, message: str, required_amount: float = None):
        super().__init__(f"资金不足 - 需要{required_amount}: {message}")
        self.required_amount = required_amount


class InvalidOrderException(TradingException):
    """无效订单异常"""

    def __init__(self, message: str, order_details: dict = None):
        super().__init__(f"无效订单 - {order_details}: {message}")
        self.order_details = order_details


class MarketDataException(TradingException):
    """市场数据异常"""

    def __init__(self, message: str, symbol: str = None):
        super().__init__(f"市场数据异常 - {symbol}: {message}")
        self.symbol = symbol


class RiskControlException(TradingException):
    """风险控制异常"""

    def __init__(self, message: str, risk_type: str = None):
        super().__init__(f"风险控制异常 - {risk_type}: {message}")
        self.risk_type = risk_type


class TimeoutException(TradingException):
    """超时异常"""

    def __init__(self, message: str, timeout_seconds: int = None):
        super().__init__(f"操作超时 - {timeout_seconds}秒: {message}")
        self.timeout_seconds = timeout_seconds


class BrokerException(TradingException):
    """券商异常"""

    def __init__(self, message: str, broker_code: str = None):
        super().__init__(f"券商异常 - {broker_code}: {message}")
        self.broker_code = broker_code


def handle_trading_exception(func):
    """
    装饰器：统一处理交易异常

    Args:
        func: 被装饰的函数

    Returns:
        包装后的函数
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TradingException:
            # 重新抛出交易异常
            raise
        except Exception as e:
            # 将其他异常包装为交易异常
            raise TradingException(f"意外交易错误: {str(e)}") from e

    return wrapper


def validate_order_params(order: dict):
    """
    验证订单参数

    Args:
        order: 订单字典

    Raises:
        InvalidOrderException: 订单参数无效
    """
    required_fields = ['symbol', 'side', 'quantity']
    missing_fields = [
        field for field in required_fields if field not in order or order[field] is None]

    if missing_fields:
        raise InvalidOrderException(f"缺少必需字段: {missing_fields}", order)

    if order['quantity'] <= 0:
        raise InvalidOrderException("订单数量必须大于0", order)

    if order['side'] not in ['BUY', 'SELL']:
        raise InvalidOrderException(f"无效订单方向: {order['side']}", order)


def validate_connection_status(is_connected: bool, broker_name: str):
    """
    验证连接状态

    Args:
        is_connected: 是否已连接
        broker_name: 券商名称

    Raises:
        ConnectionException: 连接异常
    """
    if not is_connected:
        raise ConnectionException(f"与券商{broker_name}的连接已断开", broker_name)


def validate_sufficient_funds(available_balance: float, required_amount: float):
    """
    验证资金充足性

    Args:
        available_balance: 可用余额
        required_amount: 所需金额

    Raises:
        InsufficientFundsException: 资金不足
    """
    if available_balance < required_amount:
        raise InsufficientFundsException(
            f"可用余额{available_balance}不足，所需{required_amount}",
            required_amount
        )


def check_order_timeout(order_timestamp, timeout_seconds: int):
    """
    检查订单是否超时

    Args:
        order_timestamp: 订单时间戳
        timeout_seconds: 超时秒数

    Returns:
        是否超时

    Raises:
        TimeoutException: 订单超时
    """
    from datetime import datetime, timedelta

    if isinstance(order_timestamp, str):
        order_timestamp = datetime.fromisoformat(order_timestamp.replace('Z', '+00:00'))

    if datetime.now() - order_timestamp > timedelta(seconds=timeout_seconds):
        raise TimeoutException(f"订单已超时{timeout_seconds}秒", timeout_seconds)

    return False
