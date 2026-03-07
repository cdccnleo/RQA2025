#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
策略工厂
Strategy Factory

统一管理各种策略的创建、注册和实例化。
"""

from typing import Dict, Type, Any, Optional, List
import logging
from ..interfaces.strategy_interfaces import StrategyConfig, StrategyType, IStrategyFactory, IStrategy
from .base_strategy import BaseStrategy
from .momentum_strategy import MomentumStrategy
from .mean_reversion_strategy import MeanReversionStrategy

logger = logging.getLogger(__name__)


class StrategyFactory(IStrategyFactory):

    """
    策略工厂
    Strategy Factory

    负责创建和管理不同类型的策略实例，提供统一的策略创建接口。
    """

    # 策略类注册表
    _strategy_registry: Dict[StrategyType, Type[BaseStrategy]] = {
        StrategyType.MOMENTUM: MomentumStrategy,
        StrategyType.MEAN_REVERSION: MeanReversionStrategy,
    }

    # 策略模板配置
    _strategy_templates: Dict[StrategyType, Dict[str, Any]] = {
        StrategyType.MOMENTUM: {
            "name": "动量策略模板",
            "description": "基于价格动量的趋势跟随策略",
            "parameters": {
                "lookback_period": 20,
                "momentum_threshold": 0.05,
                "volume_threshold": 1.5,
                "min_trend_period": 5,
                "max_hold_period": 10
            },
            "risk_limits": {
                "max_position": 1000,
                "max_drawdown": 0.1,
                "risk_per_trade": 0.02
            }
        },
        StrategyType.MEAN_REVERSION: {
            "name": "均值回归策略模板",
            "description": "基于价格均值回归的反转策略",
            "parameters": {
                "lookback_period": 20,
                "std_threshold": 2.0,
                "min_reversion_period": 3,
                "max_hold_period": 5,
                "profit_target": 0.05,
                "stop_loss": -0.03
            },
            "risk_limits": {
                "max_position": 1000,
                "max_drawdown": 0.1,
                "risk_per_trade": 0.02
            }
        }
    }

    def __init__(self):
        """初始化策略工厂"""
        self._custom_strategies: Dict[str, Type[BaseStrategy]] = {}
        logger.info("策略工厂初始化完成")

    def create_strategy_instance(self, config: StrategyConfig) -> BaseStrategy:
        """
        创建策略实例

        Args:
            config: 策略配置

        Returns:
            BaseStrategy: 策略实例

        Raises:
            ValueError: 当策略类型不支持或配置无效时
        """
        # 验证配置
        validation_result = self.validate_config(config)
        if not validation_result["is_valid"]:
            error_msg = f"策略配置无效: {validation_result['errors']}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # 获取策略类
        strategy_class = self._get_strategy_class(config.strategy_type)
        if not strategy_class:
            error_msg = f"不支持的策略类型: {config.strategy_type}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            # 创建策略实例
            strategy_instance = strategy_class(config)
            logger.info(f"策略实例创建成功: {config.strategy_name} ({config.strategy_type.value})")
            return strategy_instance

        except Exception as e:
            error_msg = f"策略实例创建失败: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def create_strategy(self, config: StrategyConfig) -> 'IStrategy':
        """
        创建策略实例 (实现IStrategyFactory接口)

        Args:
            config: 策略配置

        Returns:
            IStrategy: 策略实例
        """
        return self.create_strategy_instance(config)

    def _get_strategy_class(self, strategy_type: StrategyType) -> Optional[Type[BaseStrategy]]:
        """
        获取策略类

        Args:
            strategy_type: 策略类型

        Returns:
            Optional[Type[BaseStrategy]]: 策略类
        """
        # 先从内置策略中查找
        if strategy_type in self._strategy_registry:
            return self._strategy_registry[strategy_type]

        # 从自定义策略中查找
        for custom_type_name, custom_class in self._custom_strategies.items():
            if custom_type_name == strategy_type.value:
                return custom_class

        return None

    def get_supported_types(self) -> List[StrategyType]:
        """
        获取支持的策略类型

        Returns:
            List[StrategyType]: 支持的策略类型列表
        """
        supported_types = list(self._strategy_registry.keys())

        # 添加自定义策略类型
        for custom_type_name in self._custom_strategies.keys():
            try:
                custom_type = StrategyType(custom_type_name)
                if custom_type not in supported_types:
                    supported_types.append(custom_type)
            except ValueError:
                # 如果不是有效的枚举值，跳过
                continue

        return supported_types

    def validate_config(self, config: StrategyConfig) -> Dict[str, Any]:
        """
        验证策略配置

        Args:
            config: 策略配置

        Returns:
            Dict[str, Any]: 验证结果，包含'is_valid'和'errors'字段
        """
        errors = []

        # 基本字段验证
        if not config.strategy_id or not isinstance(config.strategy_id, str):
            errors.append("策略ID必须是非空字符串")

        if not config.strategy_name or not isinstance(config.strategy_name, str):
            errors.append("策略名称必须是非空字符串")

        if not isinstance(config.strategy_type, StrategyType):
            errors.append("策略类型必须是有效的StrategyType枚举值")

        if not isinstance(config.parameters, dict):
            errors.append("策略参数必须是字典类型")

        # 策略特定验证
        if config.strategy_type == StrategyType.MOMENTUM:
            errors.extend(self._validate_momentum_config(config))
        elif config.strategy_type == StrategyType.MEAN_REVERSION:
            errors.extend(self._validate_mean_reversion_config(config))

        return {
            "is_valid": len(errors) == 0,
            "errors": errors
        }

    def _validate_momentum_config(self, config: StrategyConfig) -> List[str]:
        """验证动量策略配置"""
        errors = []

        params = config.parameters

        if 'lookback_period' in params:
            if not isinstance(params['lookback_period'], int) or params['lookback_period'] <= 0:
                errors.append("lookback_period必须是正整数")

        if 'momentum_threshold' in params:
            if not isinstance(params['momentum_threshold'], (int, float)):
                errors.append("momentum_threshold必须是数值")

        if 'volume_threshold' in params:
            if not isinstance(params['volume_threshold'], (int, float)) or params['volume_threshold'] <= 0:
                errors.append("volume_threshold必须是正数值")

        return errors

    def _validate_mean_reversion_config(self, config: StrategyConfig) -> List[str]:
        """验证均值回归策略配置"""
        errors = []

        params = config.parameters

        if 'lookback_period' in params:
            if not isinstance(params['lookback_period'], int) or params['lookback_period'] <= 0:
                errors.append("lookback_period必须是正整数")

        if 'std_threshold' in params:
            if not isinstance(params['std_threshold'], (int, float)) or params['std_threshold'] <= 0:
                errors.append("std_threshold必须是正数值")

        if 'profit_target' in params and 'stop_loss' in params:
            if params['profit_target'] <= params['stop_loss']:
                errors.append("profit_target必须大于stop_loss")

        return errors

    def get_template_configs(self, strategy_type: StrategyType) -> List[StrategyConfig]:
        """
        获取策略模板配置

        Args:
            strategy_type: 策略类型

        Returns:
            List[StrategyConfig]: 模板配置列表
        """
        if strategy_type not in self._strategy_templates:
            return []

        template = self._strategy_templates[strategy_type]

        config = StrategyConfig(
            strategy_id=f"{strategy_type.value}_template",
            strategy_name=template["name"],
            strategy_type=strategy_type,
            parameters=template["parameters"].copy(),
            risk_limits=template["risk_limits"].copy(),
            description=template["description"]
        )

        return [config]

    def register_custom_strategy(self, strategy_type_name: str, strategy_class: Type[BaseStrategy]):
        """
        注册自定义策略

        Args:
            strategy_type_name: 策略类型名称
            strategy_class: 策略类

        Raises:
            ValueError: 当策略类无效时
        """
        # 验证策略类
        if not issubclass(strategy_class, BaseStrategy):
            raise ValueError(f"策略类 {strategy_class.__name__} 必须继承自BaseStrategy")

        self._custom_strategies[strategy_type_name] = strategy_class
        logger.info(f"自定义策略已注册: {strategy_type_name} -> {strategy_class.__name__}")

    def unregister_custom_strategy(self, strategy_type_name: str):
        """
        注销自定义策略

        Args:
            strategy_type_name: 策略类型名称
        """
        if strategy_type_name in self._custom_strategies:
            del self._custom_strategies[strategy_type_name]
            logger.info(f"自定义策略已注销: {strategy_type_name}")

    def get_registered_strategies(self) -> Dict[str, Type[BaseStrategy]]:
        """
        获取所有已注册的策略

        Returns:
            Dict[str, Type[BaseStrategy]]: 策略名称到策略类的映射
        """
        registered = {}

        # 添加内置策略
        for strategy_type, strategy_class in self._strategy_registry.items():
            registered[strategy_type.value] = strategy_class

        # 添加自定义策略
        registered.update(self._custom_strategies)

        return registered

    def create_strategy_from_template(self, strategy_type: StrategyType,


                                      custom_params: Optional[Dict[str, Any]] = None) -> BaseStrategy:
        """
        从模板创建策略

        Args:
            strategy_type: 策略类型
            custom_params: 自定义参数

        Returns:
            BaseStrategy: 策略实例

        Raises:
            ValueError: 当模板不存在时
        """
        templates = self.get_template_configs(strategy_type)
        if not templates:
            raise ValueError(f"策略类型 {strategy_type.value} 没有可用的模板")

        config = templates[0]

        # 应用自定义参数
        if custom_params:
            if 'parameters' in custom_params:
                config.parameters.update(custom_params['parameters'])
            if 'risk_limits' in custom_params:
                config.risk_limits.update(custom_params['risk_limits'])
            if 'strategy_name' in custom_params:
                config.strategy_name = custom_params['strategy_name']
            if 'description' in custom_params:
                config.description = custom_params['description']

        return self.create_strategy_instance(config)


# Global strategy factory instance - only created when needed
# 全局策略工厂实例 - 仅在需要时创建
_strategy_factory = None


def get_strategy_factory() -> StrategyFactory:
    """获取全局策略工厂实例"""
    global _strategy_factory
    if _strategy_factory is None:
        _strategy_factory = StrategyFactory()
    return _strategy_factory


# For backward compatibility
strategy_factory = None


# 便捷函数

def create_momentum_strategy(lookback_period: int = 20,


                             momentum_threshold: float = 0.05,
                             strategy_name: str = "Momentum Strategy") -> BaseStrategy:
    """
    创建动量策略

    Args:
        lookback_period: 回溯周期
        momentum_threshold: 动量阈值
        strategy_name: 策略名称

    Returns:
        BaseStrategy: 动量策略实例
    """
    return strategy_factory.create_strategy_from_template(
        StrategyType.MOMENTUM,
        {
            "strategy_name": strategy_name,
            "parameters": {
                "lookback_period": lookback_period,
                "momentum_threshold": momentum_threshold
            }
        }
    )


def create_mean_reversion_strategy(lookback_period: int = 20,


                                   std_threshold: float = 2.0,
                                   strategy_name: str = "Mean Reversion Strategy") -> BaseStrategy:
    """
    创建均值回归策略

    Args:
        lookback_period: 回溯周期
        std_threshold: 标准差阈值
        strategy_name: 策略名称

    Returns:
        BaseStrategy: 均值回归策略实例
    """
    return strategy_factory.create_strategy_from_template(
        StrategyType.MEAN_REVERSION,
        {
            "strategy_name": strategy_name,
            "parameters": {
                "lookback_period": lookback_period,
                "std_threshold": std_threshold
            }
        }
    )


# 导出类和函数
__all__ = [
    'StrategyFactory',
    'strategy_factory',
    'create_momentum_strategy',
    'create_mean_reversion_strategy'
]
