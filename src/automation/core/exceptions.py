"""
自动化层异常处理
Automation Layer Exception Handling

定义自动化相关的异常类和错误处理机制
"""


class AutomationException(Exception):
    """自动化基础异常类"""

    def __init__(self, message: str, error_code: int = -1):
        super().__init__(message)
        self.error_code = error_code
        self.message = message


class TaskExecutionError(AutomationException):
    """任务执行异常"""

    def __init__(self, message: str, task_id: str = None):
        super().__init__(f"任务执行失败 - {task_id}: {message}")
        self.task_id = task_id


class WorkflowExecutionError(AutomationException):
    """工作流执行异常"""

    def __init__(self, message: str, workflow_id: str = None):
        super().__init__(f"工作流执行失败 - {workflow_id}: {message}")
        self.workflow_id = workflow_id


class RuleExecutionError(AutomationException):
    """规则执行异常"""

    def __init__(self, message: str, rule_id: str = None):
        super().__init__(f"规则执行失败 - {rule_id}: {message}")
        self.rule_id = rule_id


class SchedulerError(AutomationException):
    """调度器异常"""

    def __init__(self, message: str, scheduler_id: str = None):
        super().__init__(f"调度器错误 - {scheduler_id}: {message}")
        self.scheduler_id = scheduler_id


class DeploymentError(AutomationException):
    """部署异常"""

    def __init__(self, message: str, deployment_id: str = None):
        super().__init__(f"部署失败 - {deployment_id}: {message}")
        self.deployment_id = deployment_id


class IntegrationError(AutomationException):
    """集成异常"""

    def __init__(self, message: str, service_name: str = None):
        super().__init__(f"集成失败 - {service_name}: {message}")
        self.service_name = service_name


class ResourceExhaustionError(AutomationException):
    """资源耗尽异常"""

    def __init__(self, message: str, resource_type: str = None):
        super().__init__(f"资源耗尽 - {resource_type}: {message}")
        self.resource_type = resource_type


class ConfigurationError(AutomationException):
    """配置异常"""

    def __init__(self, message: str, config_key: str = None):
        super().__init__(f"配置错误 - {config_key}: {message}")
        self.config_key = config_key


class TimeoutError(AutomationException):
    """超时异常"""

    def __init__(self, message: str, timeout_seconds: int = None):
        super().__init__(f"操作超时 - {timeout_seconds}秒: {message}")
        self.timeout_seconds = timeout_seconds


class CircuitBreakerError(AutomationException):
    """熔断器异常"""

    def __init__(self, message: str, service_name: str = None):
        super().__init__(f"服务熔断 - {service_name}: {message}")
        self.service_name = service_name


def handle_automation_exception(func):
    """
    装饰器：统一处理自动化异常

    Args:
        func: 被装饰的函数

    Returns:
        包装后的函数
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except AutomationException:
            # 重新抛出自动化异常
            raise
        except Exception as e:
            # 将其他异常包装为自动化异常
            raise AutomationException(f"意外自动化错误: {str(e)}") from e

    return wrapper


def validate_task_config(config: dict, required_keys: list):
    """
    验证任务配置

    Args:
        config: 配置字典
        required_keys: 必需的键列表

    Raises:
        ConfigurationError: 配置验证失败
    """
    if not config:
        raise ConfigurationError("任务配置不能为空")

    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ConfigurationError(f"缺少必需配置项: {missing_keys}")

    for key in required_keys:
        if config[key] is None:
            raise ConfigurationError(f"配置项值不能为空: {key}")


def validate_workflow_dependencies(workflow: dict):
    """
    验证工作流依赖关系

    Args:
        workflow: 工作流字典

    Raises:
        WorkflowExecutionError: 工作流验证失败
    """
    if not workflow:
        raise WorkflowExecutionError("工作流配置不能为空")

    tasks = workflow.get('tasks', [])
    if not tasks:
        raise WorkflowExecutionError("工作流必须包含至少一个任务")

    # 检查循环依赖
    task_ids = {task['id'] for task in tasks}
    for task in tasks:
        dependencies = task.get('dependencies', [])
        invalid_deps = [dep for dep in dependencies if dep not in task_ids]
        if invalid_deps:
            raise WorkflowExecutionError(f"任务 {task['id']} 依赖无效任务: {invalid_deps}")


def check_task_timeout(start_time, timeout_seconds: int, task_id: str):
    """
    检查任务是否超时

    Args:
        start_time: 任务开始时间
        timeout_seconds: 超时秒数
        task_id: 任务ID

    Returns:
        是否超时

    Raises:
        TimeoutError: 任务超时
    """
    from datetime import datetime, timedelta

    if isinstance(start_time, str):
        start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))

    if datetime.now() - start_time > timedelta(seconds=timeout_seconds):
        raise TimeoutError(f"任务 {task_id} 已超时", timeout_seconds)

    return False


def validate_resource_limits(current_usage: float, limit: float, resource_type: str):
    """
    验证资源限制

    Args:
        current_usage: 当前使用量
        limit: 限制值
        resource_type: 资源类型

    Raises:
        ResourceExhaustionError: 资源超限
    """
    if current_usage > limit:
        raise ResourceExhaustionError(
            f"{resource_type} 使用量 {current_usage} 超过限制 {limit}",
            resource_type
        )
