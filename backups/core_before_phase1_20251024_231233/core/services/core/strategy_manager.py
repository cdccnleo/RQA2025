"""
RQA2025系统策略模式管理器
Strategy Pattern Manager for RQA2025 System

实现策略模式，统一管理各种算法和业务策略
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, List
from src.unified_exceptions import handle_business_exceptions, BusinessLogicError
import logging

logger = logging.getLogger(__name__)


class Strategy(ABC):
    """
    策略接口

    所有策略类必须实现的接口
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """策略名称"""

    @property
    @abstractmethod
    def description(self) -> str:
        """策略描述"""

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """
        执行策略

        Args:
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            执行结果
        """

    def validate_input(self, *args, **kwargs) -> bool:
        """
        验证输入参数

        Args:
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            输入是否有效
        """
        return True

    def get_metadata(self) -> Dict[str, Any]:
        """
        获取策略元数据

        Returns:
            元数据字典
        """
        return {
            'name': self.name,
            'description': self.description,
            'class': self.__class__.__name__,
            'module': self.__class__.__module__
        }


class StrategyManager:
    """
    策略管理器

    统一管理各种策略的注册、选择和执行
    """

    def __init__(self, name: str = "default"):
        self.name = name
        self._strategies: Dict[str, Strategy] = {}
        self._default_strategy: Optional[str] = None
        self._strategy_groups: Dict[str, List[str]] = {}
        self._execution_history: List[Dict[str, Any]] = []
        self._max_history_size = 1000

    @handle_business_exceptions
    def register_strategy(self, strategy: Strategy,
                          group: Optional[str] = None) -> None:
        """
        注册策略

        Args:
            strategy: 策略实例
            group: 策略分组
        """
        if not isinstance(strategy, Strategy):
            raise ValueError("策略必须实现Strategy接口")

        strategy_name = strategy.name
        if strategy_name in self._strategies:
            logger.warning(f"策略 '{strategy_name}' 已被覆盖")

        self._strategies[strategy_name] = strategy

        # 添加到分组
        if group:
            if group not in self._strategy_groups:
                self._strategy_groups[group] = []
            if strategy_name not in self._strategy_groups[group]:
                self._strategy_groups[group].append(strategy_name)

        logger.info(f"策略 '{strategy_name}' 已注册到策略管理器 '{self.name}'")

    def unregister_strategy(self, strategy_name: str) -> bool:
        """
        注销策略

        Args:
            strategy_name: 策略名称

        Returns:
            是否成功注销
        """
        if strategy_name not in self._strategies:
            return False

        del self._strategies[strategy_name]

        # 从分组中移除
        for group_strategies in self._strategy_groups.values():
            if strategy_name in group_strategies:
                group_strategies.remove(strategy_name)

        # 如果是默认策略，则清除默认策略
        if self._default_strategy == strategy_name:
            self._default_strategy = None

        logger.info(f"策略 '{strategy_name}' 已从策略管理器 '{self.name}' 注销")
        return True

    def set_default_strategy(self, strategy_name: str) -> None:
        """
        设置默认策略

        Args:
            strategy_name: 策略名称
        """
        if strategy_name not in self._strategies:
            raise ValueError(f"未注册的策略: {strategy_name}")
        self._default_strategy = strategy_name
        logger.info(f"默认策略已设置为: {strategy_name}")

    @handle_business_exceptions
    def execute_strategy(self, strategy_name: Optional[str] = None,
                         group: Optional[str] = None,
                         *args, **kwargs) -> Any:
        """
        执行策略

        Args:
            strategy_name: 策略名称
            group: 策略分组（如果指定，将从分组中选择策略）
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            执行结果
        """
        # 确定要执行的策略
        target_strategy_name = self._resolve_strategy_name(strategy_name, group)
        if not target_strategy_name:
            raise BusinessLogicError("未指定策略且无默认策略")

        strategy = self._strategies[target_strategy_name]

        # 验证输入
        if not strategy.validate_input(*args, **kwargs):
            raise BusinessLogicError(f"策略 '{target_strategy_name}' 输入验证失败")

        # 执行策略
        try:
            start_time = __import__('time').time()
            result = strategy.execute(*args, **kwargs)
            execution_time = __import__('time').time() - start_time

            # 记录执行历史
            self._record_execution(target_strategy_name, execution_time, result, args, kwargs)

            logger.debug(f"策略 '{target_strategy_name}' 执行成功，耗时: {execution_time:.3f}s")
            return result

        except Exception as e:
            # 记录失败执行
            self._record_execution(target_strategy_name, 0, None, args, kwargs, error=str(e))
            logger.error(f"策略 '{target_strategy_name}' 执行失败: {e}")
            raise BusinessLogicError(f"策略执行失败: {target_strategy_name}") from e

    def _resolve_strategy_name(self, strategy_name: Optional[str],
                               group: Optional[str]) -> Optional[str]:
        """
        解析策略名称

        Args:
            strategy_name: 指定的策略名称
            group: 策略分组

        Returns:
            解析后的策略名称
        """
        # 优先使用指定的策略名称
        if strategy_name:
            return strategy_name if strategy_name in self._strategies else None

        # 如果指定了分组，从分组中选择策略
        if group and group in self._strategy_groups:
            group_strategies = self._strategy_groups[group]
            if group_strategies:
                # 返回分组中的第一个策略（可以扩展为更复杂的选择逻辑）
                return group_strategies[0]

        # 使用默认策略
        return self._default_strategy

    def _record_execution(self, strategy_name: str, execution_time: float,
                          result: Any, args: tuple, kwargs: dict,
                          error: Optional[str] = None) -> None:
        """
        记录执行历史

        Args:
            strategy_name: 策略名称
            execution_time: 执行时间
            result: 执行结果
            args: 位置参数
            kwargs: 关键字参数
            error: 错误信息
        """
        execution_record = {
            'strategy_name': strategy_name,
            'execution_time': execution_time,
            'timestamp': __import__('datetime').datetime.now().isoformat(),
            'success': error is None,
            'error': error,
            'args_count': len(args),
            'kwargs_keys': list(kwargs.keys())
        }

        self._execution_history.append(execution_record)

        # 限制历史记录数量
        if len(self._execution_history) > self._max_history_size:
            self._execution_history.pop(0)

    def get_available_strategies(self, group: Optional[str] = None) -> List[str]:
        """
        获取可用策略列表

        Args:
            group: 策略分组（可选）

        Returns:
            策略名称列表
        """
        if group:
            return self._strategy_groups.get(group, []).copy()
        return list(self._strategies.keys())

    def get_strategy_info(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """
        获取策略信息

        Args:
            strategy_name: 策略名称

        Returns:
            策略信息字典
        """
        if strategy_name not in self._strategies:
            return None

        strategy = self._strategies[strategy_name]
        return strategy.get_metadata()

    def get_groups(self) -> List[str]:
        """
        获取所有策略分组

        Returns:
            分组名称列表
        """
        return list(self._strategy_groups.keys())

    def get_execution_stats(self) -> Dict[str, Any]:
        """
        获取执行统计信息

        Returns:
            统计信息字典
        """
        if not self._execution_history:
            return {'total_executions': 0}

        total_executions = len(self._execution_history)
        successful_executions = sum(1 for record in self._execution_history if record['success'])
        failed_executions = total_executions - successful_executions

        strategy_stats = {}
        for record in self._execution_history:
            strategy_name = record['strategy_name']
            if strategy_name not in strategy_stats:
                strategy_stats[strategy_name] = {
                    'executions': 0,
                    'successes': 0,
                    'failures': 0,
                    'total_time': 0.0
                }

            stats = strategy_stats[strategy_name]
            stats['executions'] += 1
            if record['success']:
                stats['successes'] += 1
            else:
                stats['failures'] += 1
            stats['total_time'] += record['execution_time']

        return {
            'total_executions': total_executions,
            'successful_executions': successful_executions,
            'failed_executions': failed_executions,
            'success_rate': successful_executions / total_executions if total_executions > 0 else 0,
            'strategy_stats': strategy_stats
        }

    def clear_execution_history(self) -> None:
        """清空执行历史"""
        self._execution_history.clear()

    def get_recent_executions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取最近的执行记录

        Args:
            limit: 返回记录数量限制

        Returns:
            执行记录列表
        """
        return self._execution_history[-limit:].copy()

    def create_strategy_selector(self, criteria: Dict[str, Any]) -> Callable:
        """
        创建策略选择器

        Args:
            criteria: 选择标准

        Returns:
            策略选择函数
        """
        def selector(*args, **kwargs) -> str:
            # 基于输入参数和标准选择最佳策略
            # 这里可以实现复杂的选择逻辑
            for strategy_name, strategy in self._strategies.items():
                # 简单的示例：选择第一个满足条件的策略
                if self._matches_criteria(strategy, criteria, args, kwargs):
                    return strategy_name

            return self._default_strategy or list(self._strategies.keys())[0]

        return selector

    def _matches_criteria(self, strategy: Strategy, criteria: Dict[str, Any],
                          args: tuple, kwargs: dict) -> bool:
        """
        检查策略是否匹配选择标准

        Args:
            strategy: 策略实例
            criteria: 选择标准
            args: 位置参数
            kwargs: 关键字参数

        Returns:
            是否匹配
        """
        # 简单的匹配逻辑，可以根据需要扩展
        strategy_info = strategy.get_metadata()

        for key, value in criteria.items():
            if key in strategy_info and strategy_info[key] == value:
                continue
            else:
                return False

        return True


# 全局策略管理器实例
global_strategy_manager = StrategyManager("global")


# 具体策略实现示例
class FastProcessingStrategy(Strategy):
    """快速处理策略"""

    @property
    def name(self) -> str:
        return "fast"

    @property
    def description(self) -> str:
        return "快速处理策略，优先考虑速度而非准确性"

    def execute(self, data: Any) -> Any:
        """快速处理逻辑"""
        # 模拟快速处理
        if isinstance(data, (int, float)):
            return data * 1.1  # 简单的近似计算
        elif isinstance(data, str):
            return data.upper()  # 简单的字符串处理
        else:
            return f"快速处理: {data}"


class AccurateProcessingStrategy(Strategy):
    """准确处理策略"""

    @property
    def name(self) -> str:
        return "accurate"

    @property
    def description(self) -> str:
        return "准确处理策略，优先考虑准确性而非速度"

    def execute(self, data: Any) -> Any:
        """准确处理逻辑"""
        # 模拟准确处理
        if isinstance(data, (int, float)):
            return data * 1.05  # 更精确的计算
        elif isinstance(data, str):
            return data.title()  # 更规范的字符串处理
        else:
            return f"准确处理: {data}"


class ConservativeStrategy(Strategy):
    """保守策略"""

    @property
    def name(self) -> str:
        return "conservative"

    @property
    def description(self) -> str:
        return "保守策略，优先考虑稳定性和风险控制"

    def execute(self, data: Any) -> Any:
        """保守处理逻辑"""
        # 模拟保守处理
        if isinstance(data, (int, float)):
            return data * 0.95  # 保守的调整
        elif isinstance(data, str):
            return data.lower()  # 保守的字符串处理
        else:
            return f"保守处理: {data}"


class AggressiveStrategy(Strategy):
    """激进策略"""

    @property
    def name(self) -> str:
        return "aggressive"

    @property
    def description(self) -> str:
        return "激进策略，追求最大收益，可能承担更高风险"

    def execute(self, data: Any) -> Any:
        """激进处理逻辑"""
        # 模拟激进处理
        if isinstance(data, (int, float)):
            return data * 1.2  # 激进的调整
        elif isinstance(data, str):
            return f"!!!{data.upper()}!!!"  # 激进的字符串处理
        else:
            return f"激进处理: {data}"


# 初始化默认策略
def initialize_default_strategies():
    """初始化默认策略"""
    global_strategy_manager.register_strategy(FastProcessingStrategy(), "processing")
    global_strategy_manager.register_strategy(AccurateProcessingStrategy(), "processing")
    global_strategy_manager.register_strategy(ConservativeStrategy(), "risk")
    global_strategy_manager.register_strategy(AggressiveStrategy(), "risk")

    global_strategy_manager.set_default_strategy("fast")


# 自动初始化
initialize_default_strategies()


# 便捷函数
def register_strategy(strategy: Strategy, group: Optional[str] = None) -> None:
    """
    注册策略到全局管理器

    Args:
        strategy: 策略实例
        group: 策略分组
    """
    global_strategy_manager.register_strategy(strategy, group)


def execute_strategy(strategy_name: Optional[str] = None,
                     group: Optional[str] = None, *args, **kwargs) -> Any:
    """
    执行策略

    Args:
        strategy_name: 策略名称
        group: 策略分组
        *args: 位置参数
        **kwargs: 关键字参数

    Returns:
        执行结果
    """
    return global_strategy_manager.execute_strategy(strategy_name, group, *args, **kwargs)


def get_strategy_manager() -> StrategyManager:
    """
    获取全局策略管理器

    Returns:
        策略管理器实例
    """
    return global_strategy_manager


def get_strategy_stats() -> Dict[str, Any]:
    """
    获取策略执行统计

    Returns:
        统计信息
    """
    return global_strategy_manager.get_execution_stats()
