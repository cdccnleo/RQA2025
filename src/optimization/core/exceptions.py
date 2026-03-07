"""
优化层异常处理
Optimization Layer Exception Handling

定义优化相关的异常类和错误处理机制
"""


class OptimizationException(Exception):
    """优化基础异常类"""

    def __init__(self, message: str, error_code: int = -1):
        super().__init__(message)
        self.error_code = error_code
        self.message = message


class ConvergenceError(OptimizationException):
    """收敛失败异常"""

    def __init__(self, message: str, algorithm: str = None):
        super().__init__(f"优化算法收敛失败 - {algorithm}: {message}")
        self.algorithm = algorithm


class ResourceExhaustionError(OptimizationException):
    """资源耗尽异常"""

    def __init__(self, message: str, resource_type: str = None):
        super().__init__(f"优化资源耗尽 - {resource_type}: {message}")
        self.resource_type = resource_type


class InvalidParameterError(OptimizationException):
    """无效参数异常"""

    def __init__(self, message: str, parameter: str = None):
        super().__init__(f"无效优化参数 - {parameter}: {message}")
        self.parameter = parameter


class OptimizationTimeoutError(OptimizationException):
    """优化超时异常"""

    def __init__(self, message: str, timeout_seconds: int = None):
        super().__init__(f"优化超时 - {timeout_seconds}s: {message}")
        self.timeout_seconds = timeout_seconds


class ConstraintViolationError(OptimizationException):
    """约束违反异常"""

    def __init__(self, message: str, constraint: str = None):
        super().__init__(f"优化约束违反 - {constraint}: {message}")
        self.constraint = constraint


class DataValidationError(OptimizationException):
    """数据验证异常"""

    def __init__(self, message: str, data_field: str = None):
        super().__init__(f"优化数据验证失败 - {data_field}: {message}")
        self.data_field = data_field


class AlgorithmError(OptimizationException):
    """算法错误异常"""

    def __init__(self, message: str, algorithm_name: str = None):
        super().__init__(f"优化算法错误 - {algorithm_name}: {message}")
        self.algorithm_name = algorithm_name


def handle_optimization_exception(func):
    """
    装饰器：统一处理优化异常

    Args:
        func: 被装饰的函数

    Returns:
        包装后的函数
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except OptimizationException:
            # 重新抛出优化异常
            raise
        except Exception as e:
            # 将其他异常包装为优化异常
            raise OptimizationException(f"意外优化错误: {str(e)}") from e

    return wrapper


def validate_optimization_params(params: dict, required_keys: list):
    """
    验证优化参数

    Args:
        params: 参数字典
        required_keys: 必需的键列表

    Raises:
        InvalidParameterError: 参数验证失败
    """
    if not params:
        raise InvalidParameterError("优化参数不能为空")

    missing_keys = [key for key in required_keys if key not in params]
    if missing_keys:
        raise InvalidParameterError(f"缺少必需参数: {missing_keys}")

    for key in required_keys:
        if params[key] is None:
            raise InvalidParameterError(f"参数值不能为空: {key}")


def validate_constraints(constraints: dict):
    """
    验证优化约束

    Args:
        constraints: 约束字典

    Raises:
        ConstraintViolationError: 约束验证失败
    """
    if not constraints:
        return

    # 验证权重约束
    if 'weights' in constraints:
        weights = constraints['weights']
        if sum(weights) > 1.0001:  # 允许小数点精度误差
            raise ConstraintViolationError("权重总和不能超过1", "weights_sum")

        if any(w < 0 for w in weights):
            raise ConstraintViolationError("权重不能为负数", "weights_negative")

    # 验证边界约束
    if 'bounds' in constraints:
        bounds = constraints['bounds']
        for i, (lower, upper) in enumerate(bounds):
            if lower > upper:
                raise ConstraintViolationError(f"边界约束无效: 下界{lower} > 上界{upper}", f"bounds_{i}")


def check_convergence(current_fitness: float, best_fitness: float,
                      threshold: float = 0.001, generations: int = 10) -> bool:
    """
    检查优化算法是否收敛

    Args:
        current_fitness: 当前适应度
        best_fitness: 最佳适应度
        threshold: 收敛阈值
        generations: 连续代数

    Returns:
        是否收敛

    Raises:
        ConvergenceError: 收敛检查失败
    """
    if current_fitness < 0 or best_fitness < 0:
        raise ConvergenceError("适应度值不能为负数")

    improvement = abs(current_fitness - best_fitness) / max(abs(best_fitness), 1e-10)
    return improvement < threshold
