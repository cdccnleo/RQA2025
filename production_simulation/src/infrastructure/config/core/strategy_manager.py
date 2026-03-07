#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略管理器 - 增强版

管理配置加载策略，支持策略注册、查询、执行等操作
"""

import threading
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class StrategyManager:
    """策略管理器 - 增强版
    
    提供策略的注册、查询、执行、启用/禁用等功能
    """

    def __init__(self):
        """初始化策略管理器"""
        self._strategies = {}  # 策略存储
        self._lock = threading.RLock()  # 线程锁
        logger.debug("StrategyManager initialized")
        
    @property
    def strategies(self) -> Dict[str, Any]:
        """获取策略字典（向后兼容）"""
        return self._strategies

    def add_strategy(self, name: str, strategy: Any) -> None:
        """添加策略（别名：register_strategy）
        
        Args:
            name: 策略名称
            strategy: 策略实例
        """
        with self._lock:
            self._strategies[name] = strategy
            logger.debug(f"Strategy added: {name}")

    def register_strategy(self, strategy: Any) -> None:
        """注册策略
        
        Args:
            strategy: 策略实例（需要有name属性）
        """
        name = getattr(strategy, 'name', str(id(strategy)))
        self.add_strategy(name, strategy)

    def unregister_strategy(self, name: str) -> bool:
        """注销策略
        
        Args:
            name: 策略名称
            
        Returns:
            是否成功注销
        """
        with self._lock:
            if name in self._strategies:
                del self._strategies[name]
                logger.debug(f"Strategy unregistered: {name}")
                return True
            return False

    def get_strategy(self, name: str) -> Optional[Any]:
        """获取策略
        
        Args:
            name: 策略名称
            
        Returns:
            策略实例或None
        """
        with self._lock:
            return self._strategies.get(name)

    def get_all_strategies(self) -> Dict[str, Any]:
        """获取所有策略
        
        Returns:
            所有策略的字典
        """
        with self._lock:
            return self._strategies.copy()

    def get_strategies_by_type(self, strategy_type: str) -> List[Any]:
        """按类型获取策略
        
        Args:
            strategy_type: 策略类型
            
        Returns:
            匹配类型的策略列表
        """
        with self._lock:
            result = []
            for strategy in self._strategies.values():
                # 检查策略类型
                if hasattr(strategy, 'type'):
                    s_type = strategy.type
                    # 处理字符串比较
                    if isinstance(s_type, str):
                        if s_type == strategy_type:
                            result.append(strategy)
                    # 处理枚举比较
                    elif hasattr(s_type, 'value'):
                        if s_type.value == strategy_type or str(s_type) == strategy_type:
                            result.append(strategy)
                # 如果策略类型匹配_type属性
                elif hasattr(strategy, '_type'):
                    if strategy._type == strategy_type:
                        result.append(strategy)
            return result

    def enable_strategy(self, name: str) -> bool:
        """启用策略
        
        Args:
            name: 策略名称
            
        Returns:
            是否成功
        """
        with self._lock:
            strategy = self._strategies.get(name)
            if strategy and hasattr(strategy, 'enable'):
                strategy.enable()
                logger.debug(f"Strategy enabled: {name}")
                return True
            return False

    def disable_strategy(self, name: str) -> bool:
        """禁用策略
        
        Args:
            name: 策略名称
            
        Returns:
            是否成功
        """
        with self._lock:
            strategy = self._strategies.get(name)
            if strategy and hasattr(strategy, 'disable'):
                strategy.disable()
                logger.debug(f"Strategy disabled: {name}")
                return True
            return False

    def execute_loader_strategy(self, name: str, source: str) -> Optional[Dict[str, Any]]:
        """执行加载器策略
        
        Args:
            name: 策略名称
            source: 配置源
            
        Returns:
            加载结果或None
        """
        strategy = self.get_strategy(name)
        if not strategy:
            logger.warning(f"Strategy not found: {name}")
            return None
            
        if hasattr(strategy, 'load_config'):
            try:
                result = strategy.load_config(source)
                return result
            except Exception as e:
                logger.error(f"Strategy execution failed: {name}, {e}")
                return None
        return None

    def execute_validator_strategy(self, name: str, config: Dict[str, Any]) -> Optional[Any]:
        """执行验证器策略
        
        Args:
            name: 策略名称
            config: 配置数据
            
        Returns:
            验证结果或None
        """
        strategy = self.get_strategy(name)
        if not strategy:
            logger.warning(f"Validator strategy not found: {name}")
            return None
            
        if hasattr(strategy, 'validate'):
            try:
                result = strategy.validate(config)
                return result
            except Exception as e:
                logger.error(f"Validator execution failed: {name}, {e}")
                return None
        elif hasattr(strategy, 'execute'):
            try:
                result = strategy.execute(config=config)
                return result
            except Exception as e:
                logger.error(f"Strategy execution failed: {name}, {e}")
                return None
        return None
