"""
核心服务层异常处理
Core Services Layer Exception Handling

定义核心服务相关的异常类和错误处理机制
"""


class CoreServiceException(Exception):
    """核心服务基础异常类"""

    def __init__(self, message: str, error_code: int = -1):
        super().__init__(message)
        self.error_code = error_code
        self.message = message


class ServiceContainerError(CoreServiceException):
    """服务容器异常"""

    def __init__(self, message: str, service_name: str = None):
        super().__init__(f"服务容器错误 - {service_name}: {message}")
        self.service_name = service_name


class EventBusError(CoreServiceException):
    """事件总线异常"""

    def __init__(self, message: str, event_type: str = None):
        super().__init__(f"事件总线错误 - {event_type}: {message}")
        self.event_type = event_type


class EventBusException(EventBusError):
    """事件总线异常（兼容性别名）"""

    def __init__(self, message: str, event_type: str = None):
        super().__init__(message, event_type)


class BusinessProcessError(CoreServiceException):
    """业务流程异常"""

    def __init__(self, message: str, process_id: str = None):
        super().__init__(f"业务流程错误 - {process_id}: {message}")
        self.process_id = process_id


class OrchestratorException(BusinessProcessError):
    """编排器异常"""

    def __init__(self, message: str, orchestrator_id: str = None):
        super().__init__(message, orchestrator_id)
        self.orchestrator_id = orchestrator_id


class IntegrationError(CoreServiceException):
    """集成异常"""

    def __init__(self, message: str, adapter_name: str = None):
        super().__init__(f"集成错误 - {adapter_name}: {message}")
        self.adapter_name = adapter_name


class DatabaseError(CoreServiceException):
    """数据库异常"""

    def __init__(self, message: str, db_name: str = None, operation: str = None):
        super().__init__(f"数据库错误 - {db_name} - {operation}: {message}")
        self.db_name = db_name
        self.operation = operation


class ConnectionError(DatabaseError):
    """数据库连接异常"""

    def __init__(self, message: str, host: str = None, port: int = None):
        super().__init__(f"连接错误 - {host}:{port}: {message}", operation="connection")
        self.host = host
        self.port = port


class QueryError(DatabaseError):
    """数据库查询异常"""

    def __init__(self, message: str, query: str = None):
        super().__init__(f"查询错误 - {query}: {message}", operation="query")
        self.query = query


class SecurityError(CoreServiceException):
    """安全异常"""

    def __init__(self, message: str, security_context: str = None):
        super().__init__(f"安全错误 - {security_context}: {message}")
        self.security_context = security_context


class CommunicationError(CoreServiceException):
    """通信异常"""

    def __init__(self, message: str, endpoint: str = None):
        super().__init__(f"通信错误 - {endpoint}: {message}")
        self.endpoint = endpoint


class ResourceManagementError(CoreServiceException):
    """资源管理异常"""

    def __init__(self, message: str, resource_type: str = None):
        super().__init__(f"资源管理错误 - {resource_type}: {message}")
        self.resource_type = resource_type


class ConfigurationError(CoreServiceException):
    """配置异常"""

    def __init__(self, message: str, config_key: str = None):
        super().__init__(f"配置错误 - {config_key}: {message}")
        self.config_key = config_key


class PerformanceError(CoreServiceException):
    """性能异常"""

    def __init__(self, message: str, metric_name: str = None):
        super().__init__(f"性能错误 - {metric_name}: {message}")
        self.metric_name = metric_name


class AsyncProcessingError(CoreServiceException):
    """异步处理异常"""

    def __init__(self, message: str, task_id: str = None):
        super().__init__(f"异步处理错误 - {task_id}: {message}")
        self.task_id = task_id


class ValidationError(CoreServiceException):
    """验证错误"""

    def __init__(self, message: str, field: str = None):
        super().__init__(f"验证错误 - {field}: {message}" if field else message)
        self.field = field


class DataProcessingError(CoreServiceException):
    """数据处理错误"""

    def __init__(self, message: str, data_type: str = None):
        super().__init__(f"数据处理错误 - {data_type}: {message}" if data_type else message)
        self.data_type = data_type


class TradingEngineError(CoreServiceException):
    """交易引擎错误"""

    def __init__(self, message: str, order_id: str = None):
        super().__init__(f"交易引擎错误 - {order_id}: {message}" if order_id else message)
        self.order_id = order_id


class StrategyError(CoreServiceException):
    """策略错误"""

    def __init__(self, message: str, strategy_id: str = None):
        super().__init__(f"策略错误 - {strategy_id}: {message}" if strategy_id else message)
        self.strategy_id = strategy_id


class RiskError(CoreServiceException):
    """风险管理错误"""

    def __init__(self, message: str, risk_type: str = None):
        super().__init__(f"风险错误 - {risk_type}: {message}" if risk_type else message)
        self.risk_type = risk_type


class RiskManagementError(CoreServiceException):
    """风险管理错误（别名）"""

    def __init__(self, message: str, risk_type: str = None):
        super().__init__(f"风险管理错误 - {risk_type}: {message}" if risk_type else message)
        self.risk_type = risk_type


class MarketDataError(CoreServiceException):
    """市场数据错误"""

    def __init__(self, message: str, symbol: str = None):
        super().__init__(f"市场数据错误 - {symbol}: {message}" if symbol else message)
        self.symbol = symbol


class PortfolioError(CoreServiceException):
    """投资组合错误"""

    def __init__(self, message: str, portfolio_id: str = None):
        super().__init__(f"投资组合错误 - {portfolio_id}: {message}" if portfolio_id else message)
        self.portfolio_id = portfolio_id


class OrderManagementError(CoreServiceException):
    """订单管理错误"""

    def __init__(self, message: str, order_id: str = None):
        super().__init__(f"订单管理错误 - {order_id}: {message}" if order_id else message)
        self.order_id = order_id


class ContainerException(CoreServiceException):
    """依赖注入容器异常"""

    def __init__(self, message: str, container_name: str = None, service_name: str = None):
        super().__init__(f"容器错误 - {container_name} - {service_name}: {message}")
        self.container_name = container_name
        self.service_name = service_name


# 向后兼容性别名
CoreException = CoreServiceException


def handle_core_service_exception(func):
    """
    装饰器：统一处理核心服务异常

    Args:
        func: 被装饰的函数

    Returns:
        包装后的函数
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except CoreServiceException:
            # 重新抛出核心服务异常
            raise
        except Exception as e:
            # 将其他异常包装为核心服务异常
            raise CoreServiceException(f"意外核心服务错误: {str(e)}") from e

    return wrapper


def validate_service_registration(service_name: str, service_config: dict):
    """
    验证服务注册

    Args:
        service_name: 服务名称
        service_config: 服务配置

    Raises:
        ServiceContainerError: 服务注册验证失败
    """
    if not service_name:
        raise ServiceContainerError("服务名称不能为空")

    required_keys = ['class', 'config']
    missing_keys = [key for key in required_keys if key not in service_config]
    if missing_keys:
        raise ServiceContainerError(f"服务配置缺少必需字段: {missing_keys}", service_name)


def validate_event_structure(event: dict):
    """
    验证事件结构

    Args:
        event: 事件字典

    Raises:
        EventBusError: 事件结构验证失败
    """
    required_fields = ['type', 'data', 'timestamp']
    missing_fields = [field for field in required_fields if field not in event]
    if missing_fields:
        raise EventBusError(f"事件缺少必需字段: {missing_fields}")

    if not isinstance(event.get('data'), dict):
        raise EventBusError("事件数据必须是字典类型")


def validate_workflow_definition(workflow: dict):
    """
    验证工作流定义

    Args:
        workflow: 工作流字典

    Raises:
        BusinessProcessError: 工作流验证失败
    """
    if not workflow:
        raise BusinessProcessError("工作流定义不能为空")

    required_fields = ['id', 'name', 'steps']
    missing_fields = [field for field in required_fields if field not in workflow]
    if missing_fields:
        raise BusinessProcessError(f"工作流缺少必需字段: {missing_fields}")

    steps = workflow.get('steps', [])
    if not steps:
        raise BusinessProcessError("工作流必须包含至少一个步骤")


def validate_adapter_connection(adapter_name: str, connection_config: dict):
    """
    验证适配器连接

    Args:
        adapter_name: 适配器名称
        connection_config: 连接配置

    Raises:
        IntegrationError: 适配器连接验证失败
    """
    if not connection_config:
        raise IntegrationError("适配器连接配置不能为空", adapter_name)

    required_fields = ['host', 'port']
    missing_fields = [field for field in required_fields if field not in connection_config]
    if missing_fields:
        raise IntegrationError(f"连接配置缺少必需字段: {missing_fields}", adapter_name)


def validate_security_context(user_id: str, required_permissions: list = None):
    """
    验证安全上下文

    Args:
        user_id: 用户ID
        required_permissions: 必需权限列表

    Raises:
        SecurityError: 安全验证失败
    """
    if not user_id:
        raise SecurityError("用户ID不能为空")

    if required_permissions:
        # 这里应该实现具体的权限检查逻辑
        # 暂时简化处理
        pass


def check_service_health(service_name: str, health_metrics: dict) -> dict:
    """
    检查服务健康状态

    Args:
        service_name: 服务名称
        health_metrics: 健康指标字典

    Returns:
        健康检查结果字典
    """
    health_status = {
        'service_name': service_name,
        'overall_health': 'healthy',
        'warnings': [],
        'critical_issues': []
    }

    # 检查响应时间
    response_time = health_metrics.get('avg_response_time_ms', 0)
    if response_time > 1000:  # 1秒
        health_status['critical_issues'].append(f"响应时间过高: {response_time}ms")
    elif response_time > 500:  # 500ms
        health_status['warnings'].append(f"响应时间偏高: {response_time}ms")

    # 检查错误率
    error_rate = health_metrics.get('error_rate', 0)
    if error_rate > 0.05:  # 5%
        health_status['critical_issues'].append(f"错误率过高: {error_rate:.1%}")
    elif error_rate > 0.01:  # 1%
        health_status['warnings'].append(f"错误率偏高: {error_rate:.1%}")

    # 检查资源使用率
    cpu_usage = health_metrics.get('cpu_usage', 0)
    memory_usage = health_metrics.get('memory_usage', 0)

    if cpu_usage > 90 or memory_usage > 90:
        health_status['critical_issues'].append(f"资源使用率过高: CPU{cpu_usage}%, 内存{memory_usage}%")
    elif cpu_usage > 70 or memory_usage > 80:
        health_status['warnings'].append(f"资源使用率偏高: CPU{cpu_usage}%, 内存{memory_usage}%")

    # 确定整体健康状态
    if health_status['critical_issues']:
        health_status['overall_health'] = 'critical'
    elif health_status['warnings']:
        health_status['overall_health'] = 'warning'

    return health_status
