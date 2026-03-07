#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
策略工厂
Strategy Factory

基于业务流程驱动架构，实现策略的统一创建和管理。
"""

from typing import Dict, Type, Any, List, Optional
import logging
from ..interfaces.strategy_interfaces import (
    IStrategyFactory, IStrategy, StrategyConfig, StrategyType
)

logger = logging.getLogger(__name__)


class StrategyFactory(IStrategyFactory):

    """
    策略工厂实现
    Strategy Factory Implementation

    提供策略的统一创建和管理功能。
    """

    def __init__(self):

        self._strategy_registry: Dict[StrategyType, Type[IStrategy]] = {}
        self._register_builtin_strategies()

    def _register_builtin_strategies(self):
        """注册内置策略"""
        try:
            # 动量策略
            from .momentum_strategy import MomentumStrategy
            self._strategy_registry[StrategyType.MOMENTUM] = MomentumStrategy

            # 均值回归策略
            from .mean_reversion_strategy import MeanReversionStrategy
            self._strategy_registry[StrategyType.MEAN_REVERSION] = MeanReversionStrategy

            # 尝试注册其他策略（如果存在）
            try:
                from .trend_following_strategy import TrendFollowingStrategy
                self._strategy_registry[StrategyType.TREND_FOLLOWING] = TrendFollowingStrategy
            except ImportError:
                logger.info("TrendFollowingStrategy not available")

            try:
                from .cross_market_arbitrage import CrossMarketArbitrageStrategy
                self._strategy_registry[StrategyType.ARBITRAGE] = CrossMarketArbitrageStrategy
            except ImportError:
                logger.info("CrossMarketArbitrageStrategy not available")

            logger.info(f"Built - in strategies registered: {list(self._strategy_registry.keys())}")

        except ImportError as e:
            logger.warning(f"Some built - in strategies could not be imported: {e}")

    def register_strategy(self, strategy_type: StrategyType, strategy_class: Type[IStrategy]):
        """注册自定义策略"""
        self._strategy_registry[strategy_type] = strategy_class
        logger.info(f"Strategy {strategy_type.value} registered successfully")

    def create_strategy(self, strategy_type_or_config, config_or_params=None) -> IStrategy:
        """创建策略实例"""

        # 处理不同调用方式
        if isinstance(strategy_type_or_config, str):
            # 字符串调用方式: create_strategy("mean_reversion", config_dict)
            strategy_type_str = strategy_type_or_config
            config_dict = config_or_params or {}

            # 转换为StrategyType枚举
            strategy_type = StrategyType(strategy_type_str)

            # 创建StrategyConfig
            strategy_config = StrategyConfig(
                strategy_id=config_dict.get('strategy_id', f"{strategy_type_str}_001"),
                strategy_name=config_dict.get('name', config_dict.get('strategy_name', f"{strategy_type_str.title()} Strategy")),
                strategy_type=strategy_type,
                parameters=config_dict.get('parameters', {}),
                symbols=config_dict.get('symbols', ['AAPL']),
                risk_limits=config_dict.get('risk_limits', {})
            )
        else:
            # 原始调用方式: create_strategy(config)
            strategy_config = strategy_type_or_config
            strategy_type = strategy_config.strategy_type

        if strategy_type not in self._strategy_registry:
            raise ValueError(f"Strategy type {strategy_type.value} not supported")

        strategy_class = self._strategy_registry[strategy_type]

        try:
            # 创建策略实例（使用兼容的构造函数）
            strategy = strategy_class(
                strategy_config.strategy_id,
                strategy_config.strategy_name,  # 使用strategy_name而不是name
                strategy_type.value
            )
            logger.info(f"Strategy {strategy_config.strategy_id} created successfully")
            return strategy

        except Exception as e:
            logger.error(f"Failed to create strategy {strategy_config.strategy_id}: {e}")
            raise

    def get_supported_types(self) -> List[StrategyType]:
        """获取支持的策略类型"""
        return list(self._strategy_registry.keys())
    
    def get_available_strategies(self) -> List[str]:
        """获取可用策略列表（兼容性方法）"""
        return [strategy_type.value for strategy_type in self._strategy_registry.keys()]

    def get_strategy_info(self, strategy_type: StrategyType) -> Optional[Dict[str, Any]]:
        """获取策略信息"""
        if strategy_type not in self._strategy_registry:
            return None

        strategy_class = self._strategy_registry[strategy_type]
        return {
            "type": strategy_type.value,
            "class": strategy_class.__name__,
            "module": strategy_class.__module__,
            "doc": strategy_class.__doc__ or ""
        }


# 全局策略工厂实例
_strategy_factory = None


def get_strategy_factory() -> StrategyFactory:
    """获取全局策略工厂实例"""
    global _strategy_factory
    if _strategy_factory is None:
        _strategy_factory = StrategyFactory()
    return _strategy_factory


# 兼容性函数 - 为旧代码提供兼容性

def create_strategy(strategy_type: str, config: Dict[str, Any]) -> Any:
    """创建策略的兼容性函数"""
    try:
        # 转换策略类型
        strategy_type_enum = StrategyType(strategy_type)

        # 创建策略配置
        strategy_config = StrategyConfig(
            strategy_id=config.get("strategy_id", f"{strategy_type}_{id(config)}"),
            strategy_name=config.get("strategy_name", f"{strategy_type} Strategy"),
            strategy_type=strategy_type_enum,
            parameters=config.get("parameters", {}),
            risk_limits=config.get("risk_limits", {})
        )

        factory = get_strategy_factory()
        return factory.create_strategy(strategy_config)

    except Exception as e:
        logger.error(f"Failed to create strategy via compatibility function: {e}")
        return None


# 导出兼容性类名
StrategyFactoryImpl = StrategyFactory
