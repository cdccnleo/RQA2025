"""
策略层异常处理
Strategy Layer Exception Handling

定义策略相关的异常类和错误处理机制
"""


class StrategyException(Exception):
    """策略基础异常类"""

    def __init__(self, message: str, error_code: int = -1):
        super().__init__(message)
        self.error_code = error_code
        self.message = message


class StrategyInitializationError(StrategyException):
    """策略初始化错误"""

    def __init__(self, message: str, strategy_name: str = None):
        super().__init__(f"策略初始化失败 - {strategy_name}: {message}")
        self.strategy_name = strategy_name


class BacktestError(StrategyException):
    """回测错误"""

    def __init__(self, message: str, strategy_id: str = None):
        super().__init__(f"回测执行失败 - {strategy_id}: {message}")
        self.strategy_id = strategy_id


class SignalGenerationError(StrategyException):
    """信号生成错误"""

    def __init__(self, message: str, signal_type: str = None):
        super().__init__(f"信号生成失败 - {signal_type}: {message}")
        self.signal_type = signal_type


class ParameterOptimizationError(StrategyException):
    """参数优化错误"""

    def __init__(self, message: str, parameter_name: str = None):
        super().__init__(f"参数优化失败 - {parameter_name}: {message}")
        self.parameter_name = parameter_name


class RiskControlError(StrategyException):
    """风险控制错误"""

    def __init__(self, message: str, risk_type: str = None):
        super().__init__(f"风险控制失败 - {risk_type}: {message}")
        self.risk_type = risk_type


class DataValidationError(StrategyException):
    """数据验证错误"""

    def __init__(self, message: str, data_field: str = None):
        super().__init__(f"数据验证失败 - {data_field}: {message}")
        self.data_field = data_field


class PerformanceEvaluationError(StrategyException):
    """性能评估错误"""

    def __init__(self, message: str, metric_name: str = None):
        super().__init__(f"性能评估失败 - {metric_name}: {message}")
        self.metric_name = metric_name


class ResourceExhaustionError(StrategyException):
    """资源耗尽错误"""

    def __init__(self, message: str, resource_type: str = None):
        super().__init__(f"资源耗尽 - {resource_type}: {message}")
        self.resource_type = resource_type


class ConfigurationError(StrategyException):
    """配置错误"""

    def __init__(self, message: str, config_key: str = None):
        super().__init__(f"配置错误 - {config_key}: {message}")
        self.config_key = config_key


def handle_strategy_exception(func):
    """
    装饰器：统一处理策略异常

    Args:
        func: 被装饰的函数

    Returns:
        包装后的函数
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except StrategyException:
            # 重新抛出策略异常
            raise
        except Exception as e:
            # 将其他异常包装为策略异常
            raise StrategyException(f"意外策略错误: {str(e)}") from e

    return wrapper


def validate_strategy_config(config: dict, required_keys: list):
    """
    验证策略配置

    Args:
        config: 配置字典
        required_keys: 必需的键列表

    Raises:
        ConfigurationError: 配置验证失败
    """
    if not config:
        raise ConfigurationError("策略配置不能为空")

    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ConfigurationError(f"缺少必需配置项: {missing_keys}")

    for key in required_keys:
        if config[key] is None:
            raise ConfigurationError(f"配置项值不能为空: {key}")


def validate_market_data(data, required_columns: list = None):
    """
    验证市场数据

    Args:
        data: 市场数据
        required_columns: 必需的列名列表

    Raises:
        DataValidationError: 数据验证失败
    """
    if data is None or len(data) == 0:
        raise DataValidationError("市场数据不能为空")

    if required_columns:
        if hasattr(data, 'columns'):
            # DataFrame
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise DataValidationError(f"缺少必需数据列: {missing_columns}")
        elif isinstance(data, dict):
            # Dict
            missing_keys = [key for key in required_columns if key not in data]
            if missing_keys:
                raise DataValidationError(f"缺少必需数据字段: {missing_keys}")


def validate_position_size(position_size: float, max_size: float = None):
    """
    验证仓位大小

    Args:
        position_size: 仓位大小
        max_size: 最大允许仓位大小

    Raises:
        RiskControlError: 仓位验证失败
    """
    if position_size < 0:
        raise RiskControlError("仓位大小不能为负数", "position_size")

    if max_size and position_size > max_size:
        raise RiskControlError(f"仓位大小超过最大限制: {position_size} > {max_size}", "position_limit")


def check_strategy_health(strategy_metrics: dict) -> dict:
    """
    检查策略健康状态

    Args:
        strategy_metrics: 策略指标字典

    Returns:
        健康检查结果字典
    """
    health_status = {
        'overall_health': 'healthy',
        'warnings': [],
        'critical_issues': []
    }

    # 检查夏普比率
    sharpe_ratio = strategy_metrics.get('sharpe_ratio', 0)
    if sharpe_ratio < 0.5:
        health_status['critical_issues'].append(f"夏普比率过低: {sharpe_ratio}")
    elif sharpe_ratio < 1.0:
        health_status['warnings'].append(f"夏普比率偏低: {sharpe_ratio}")

    # 检查胜率
    win_rate = strategy_metrics.get('win_rate', 0)
    if win_rate < 0.4:
        health_status['critical_issues'].append(f"胜率过低: {win_rate:.1%}")
    elif win_rate < 0.5:
        health_status['warnings'].append(f"胜率偏低: {win_rate:.1%}")

    # 检查最大回撤
    max_drawdown = strategy_metrics.get('max_drawdown', 0)
    if max_drawdown > 0.3:
        health_status['critical_issues'].append(f"最大回撤过高: {max_drawdown:.1%}")
    elif max_drawdown > 0.2:
        health_status['warnings'].append(f"最大回撤偏高: {max_drawdown:.1%}")

    # 确定整体健康状态
    if health_status['critical_issues']:
        health_status['overall_health'] = 'critical'
    elif health_status['warnings']:
        health_status['overall_health'] = 'warning'

    return health_status
