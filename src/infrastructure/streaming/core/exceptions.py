"""
流处理层异常处理
Streaming Layer Exception Handling

定义流数据处理相关的异常类和错误处理机制
"""


class StreamingException(Exception):
    """流处理基础异常类"""

    def __init__(self, message: str, error_code: int = -1):
        super().__init__(message)
        self.error_code = error_code
        self.message = message


class StreamProcessingError(StreamingException):
    """流处理错误"""

    def __init__(self, message: str, stream_id: str = None):
        super().__init__(f"流处理失败 - {stream_id}: {message}")
        self.stream_id = stream_id


class DataIngestionError(StreamingException):
    """数据摄入错误"""

    def __init__(self, message: str, source_id: str = None):
        super().__init__(f"数据摄入失败 - {source_id}: {message}")
        self.source_id = source_id


class BufferOverflowError(StreamingException):
    """缓冲区溢出错误"""

    def __init__(self, message: str, buffer_id: str = None):
        super().__init__(f"缓冲区溢出 - {buffer_id}: {message}")
        self.buffer_id = buffer_id


class ProcessingLatencyError(StreamingException):
    """处理延迟错误"""

    def __init__(self, message: str, latency_ms: int = None):
        super().__init__(f"处理延迟过高 - {latency_ms}ms: {message}")
        self.latency_ms = latency_ms


class ConnectionError(StreamingException):
    """连接错误"""

    def __init__(self, message: str, connection_id: str = None):
        super().__init__(f"连接失败 - {connection_id}: {message}")
        self.connection_id = connection_id


class DataValidationError(StreamingException):
    """数据验证错误"""

    def __init__(self, message: str, data_field: str = None):
        super().__init__(f"数据验证失败 - {data_field}: {message}")
        self.data_field = data_field


class StateManagementError(StreamingException):
    """状态管理错误"""

    def __init__(self, message: str, state_key: str = None):
        super().__init__(f"状态管理失败 - {state_key}: {message}")
        self.state_key = state_key


class ResourceExhaustionError(StreamingException):
    """资源耗尽错误"""

    def __init__(self, message: str, resource_type: str = None):
        super().__init__(f"资源耗尽 - {resource_type}: {message}")
        self.resource_type = resource_type


class SerializationError(StreamingException):
    """序列化错误"""

    def __init__(self, message: str, data_type: str = None):
        super().__init__(f"序列化失败 - {data_type}: {message}")
        self.data_type = data_type


class AggregationError(StreamingException):
    """聚合错误"""

    def __init__(self, message: str, aggregation_key: str = None):
        super().__init__(f"聚合失败 - {aggregation_key}: {message}")
        self.aggregation_key = aggregation_key


def handle_streaming_exception(func):
    """
    装饰器：统一处理流处理异常

    Args:
        func: 被装饰的函数

    Returns:
        包装后的函数
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except StreamingException:
            # 重新抛出流处理异常
            raise
        except Exception as e:
            # 将其他异常包装为流处理异常
            raise StreamingException(f"意外流处理错误: {str(e)}") from e

    return wrapper


def validate_stream_data(data, required_fields: list = None):
    """
    验证流数据

    Args:
        data: 流数据
        required_fields: 必需字段列表

    Raises:
        DataValidationError: 数据验证失败
    """
    if data is None:
        raise DataValidationError("流数据不能为空")

    if required_fields:
        if isinstance(data, dict):
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                raise DataValidationError(f"缺少必需字段: {missing_fields}")
        else:
            raise DataValidationError("流数据必须是字典格式")


def validate_buffer_capacity(current_size: int, max_capacity: int, buffer_id: str):
    """
    验证缓冲区容量

    Args:
        current_size: 当前大小
        max_capacity: 最大容量
        buffer_id: 缓冲区ID

    Raises:
        BufferOverflowError: 缓冲区溢出
    """
    if current_size >= max_capacity:
        raise BufferOverflowError(
            f"缓冲区已满，当前大小: {current_size}, 最大容量: {max_capacity}",
            buffer_id
        )


def check_processing_latency(start_time, max_latency_ms: int, operation_id: str):
    """
    检查处理延迟

    Args:
        start_time: 开始时间
        max_latency_ms: 最大延迟(毫秒)
        operation_id: 操作ID

    Returns:
        是否超时

    Raises:
        ProcessingLatencyError: 处理延迟过高
    """
    from datetime import datetime

    if isinstance(start_time, str):
        start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))

    latency_ms = (datetime.now() - start_time).total_seconds() * 1000

    if latency_ms > max_latency_ms:
        raise ProcessingLatencyError(
            f"处理延迟超过阈值: {latency_ms:.2f}ms > {max_latency_ms}ms",
            int(latency_ms)
        )

    return False


def validate_event_rate(current_rate: float, max_rate: float, stream_id: str):
    """
    验证事件速率

    Args:
        current_rate: 当前速率
        max_rate: 最大速率
        stream_id: 流ID

    Raises:
        ResourceExhaustionError: 事件速率过高
    """
    if current_rate > max_rate:
        raise ResourceExhaustionError(
            f"事件速率过高: {current_rate:.2f} > {max_rate} 事件/秒",
            "event_rate"
        )
