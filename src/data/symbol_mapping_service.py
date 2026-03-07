#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票代码映射服务

功能：
- 策略到股票代码的映射
- 支持多对多关系
- 动态映射更新

作者: AI Assistant
创建日期: 2026-02-21
"""

import logging
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from datetime import datetime
import threading

logger = logging.getLogger(__name__)


@dataclass
class SymbolMapping:
    """股票代码映射数据类"""
    strategy_id: str
    symbols: List[str] = field(default_factory=list)
    priority: int = 0  # 优先级，数字越小优先级越高
    weight: float = 1.0  # 权重，用于信号合成
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class SymbolMappingService:
    """
    股票代码映射服务
    
    职责：
    1. 策略到股票代码的映射
    2. 支持多对多关系
    3. 动态映射更新
    """
    
    def __init__(self):
        """初始化股票代码映射服务"""
        # 策略ID到映射的映射
        self._strategy_to_symbols: Dict[str, SymbolMapping] = {}
        
        # 股票代码到策略ID的反向映射
        self._symbol_to_strategies: Dict[str, Set[str]] = {}
        
        # 线程锁
        self._lock = threading.RLock()
        
        logger.info("股票代码映射服务初始化完成")
    
    def register_mapping(
        self,
        strategy_id: str,
        symbols: List[str],
        priority: int = 0,
        weight: float = 1.0
    ) -> bool:
        """
        注册策略到股票代码的映射
        
        Args:
            strategy_id: 策略ID
            symbols: 股票代码列表
            priority: 优先级
            weight: 权重
            
        Returns:
            是否注册成功
        """
        try:
            with self._lock:
                # 创建映射
                mapping = SymbolMapping(
                    strategy_id=strategy_id,
                    symbols=list(set(symbols)),  # 去重
                    priority=priority,
                    weight=weight,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                
                # 更新策略到股票的映射
                self._strategy_to_symbols[strategy_id] = mapping
                
                # 更新股票到策略的反向映射
                for symbol in mapping.symbols:
                    if symbol not in self._symbol_to_strategies:
                        self._symbol_to_strategies[symbol] = set()
                    self._symbol_to_strategies[symbol].add(strategy_id)
                
                logger.info(f"映射注册成功: {strategy_id} -> {len(symbols)} 只股票")
                return True
                
        except Exception as e:
            logger.error(f"映射注册失败 {strategy_id}: {e}")
            return False
    
    def unregister_mapping(self, strategy_id: str) -> bool:
        """
        注销策略映射
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            是否注销成功
        """
        try:
            with self._lock:
                if strategy_id not in self._strategy_to_symbols:
                    logger.warning(f"映射不存在: {strategy_id}")
                    return False
                
                # 获取映射
                mapping = self._strategy_to_symbols[strategy_id]
                
                # 从反向映射中移除
                for symbol in mapping.symbols:
                    if symbol in self._symbol_to_strategies:
                        self._symbol_to_strategies[symbol].discard(strategy_id)
                        if not self._symbol_to_strategies[symbol]:
                            del self._symbol_to_strategies[symbol]
                
                # 从策略映射中移除
                del self._strategy_to_symbols[strategy_id]
                
                logger.info(f"映射注销成功: {strategy_id}")
                return True
                
        except Exception as e:
            logger.error(f"映射注销失败 {strategy_id}: {e}")
            return False
    
    def update_mapping(
        self,
        strategy_id: str,
        symbols: Optional[List[str]] = None,
        priority: Optional[int] = None,
        weight: Optional[float] = None
    ) -> bool:
        """
        更新策略映射
        
        Args:
            strategy_id: 策略ID
            symbols: 新的股票代码列表
            priority: 新的优先级
            weight: 新的权重
            
        Returns:
            是否更新成功
        """
        try:
            with self._lock:
                if strategy_id not in self._strategy_to_symbols:
                    logger.warning(f"映射不存在，将创建新映射: {strategy_id}")
                    return self.register_mapping(
                        strategy_id,
                        symbols or [],
                        priority or 0,
                        weight or 1.0
                    )
                
                # 获取现有映射
                mapping = self._strategy_to_symbols[strategy_id]
                
                # 如果更新了股票列表，需要更新反向映射
                if symbols is not None:
                    # 从反向映射中移除旧的股票
                    for symbol in mapping.symbols:
                        if symbol in self._symbol_to_strategies:
                            self._symbol_to_strategies[symbol].discard(strategy_id)
                    
                    # 更新股票列表
                    mapping.symbols = list(set(symbols))
                    
                    # 添加到新的反向映射
                    for symbol in mapping.symbols:
                        if symbol not in self._symbol_to_strategies:
                            self._symbol_to_strategies[symbol] = set()
                        self._symbol_to_strategies[symbol].add(strategy_id)
                
                # 更新其他属性
                if priority is not None:
                    mapping.priority = priority
                if weight is not None:
                    mapping.weight = weight
                
                mapping.updated_at = datetime.now()
                
                logger.info(f"映射更新成功: {strategy_id}")
                return True
                
        except Exception as e:
            logger.error(f"映射更新失败 {strategy_id}: {e}")
            return False
    
    def get_symbols_for_strategy(self, strategy_id: str) -> List[str]:
        """
        获取策略相关的股票代码
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            股票代码列表
        """
        with self._lock:
            mapping = self._strategy_to_symbols.get(strategy_id)
            if mapping:
                return mapping.symbols.copy()
            return []
    
    def get_strategies_for_symbol(self, symbol: str) -> List[str]:
        """
        获取股票相关的策略ID
        
        Args:
            symbol: 股票代码
            
        Returns:
            策略ID列表
        """
        with self._lock:
            strategies = self._symbol_to_strategies.get(symbol, set())
            return list(strategies)
    
    def get_all_symbols(self) -> List[str]:
        """
        获取所有股票代码
        
        Returns:
            股票代码列表
        """
        with self._lock:
            return list(self._symbol_to_strategies.keys())
    
    def get_all_strategies(self) -> List[str]:
        """
        获取所有策略ID
        
        Returns:
            策略ID列表
        """
        with self._lock:
            return list(self._strategy_to_symbols.keys())
    
    def get_mapping(self, strategy_id: str) -> Optional[SymbolMapping]:
        """
        获取策略映射
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            映射对象
        """
        with self._lock:
            return self._strategy_to_symbols.get(strategy_id)
    
    def get_all_mappings(self) -> Dict[str, SymbolMapping]:
        """
        获取所有映射
        
        Returns:
            策略ID到映射的映射
        """
        with self._lock:
            return self._strategy_to_symbols.copy()
    
    def get_symbols_for_strategies(self, strategy_ids: List[str]) -> Dict[str, List[str]]:
        """
        批量获取多个策略的股票代码
        
        Args:
            strategy_ids: 策略ID列表
            
        Returns:
            策略ID到股票代码列表的映射
        """
        result = {}
        with self._lock:
            for strategy_id in strategy_ids:
                mapping = self._strategy_to_symbols.get(strategy_id)
                if mapping:
                    result[strategy_id] = mapping.symbols.copy()
                else:
                    result[strategy_id] = []
        return result
    
    def get_strategies_for_symbols(self, symbols: List[str]) -> Dict[str, List[str]]:
        """
        批量获取多个股票的策略ID
        
        Args:
            symbols: 股票代码列表
            
        Returns:
            股票代码到策略ID列表的映射
        """
        result = {}
        with self._lock:
            for symbol in symbols:
                strategies = self._symbol_to_strategies.get(symbol, set())
                result[symbol] = list(strategies)
        return result
    
    def clear(self):
        """清空所有映射"""
        with self._lock:
            self._strategy_to_symbols.clear()
            self._symbol_to_strategies.clear()
            logger.info("所有映射已清空")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取映射统计信息
        
        Returns:
            统计信息字典
        """
        with self._lock:
            return {
                'total_strategies': len(self._strategy_to_symbols),
                'total_symbols': len(self._symbol_to_strategies),
                'avg_symbols_per_strategy': (
                    sum(len(m.symbols) for m in self._strategy_to_symbols.values()) /
                    len(self._strategy_to_symbols)
                    if self._strategy_to_symbols else 0
                ),
                'avg_strategies_per_symbol': (
                    sum(len(s) for s in self._symbol_to_strategies.values()) /
                    len(self._symbol_to_strategies)
                    if self._symbol_to_strategies else 0
                )
            }


# 单例实例
_service: Optional[SymbolMappingService] = None


def get_symbol_mapping_service() -> SymbolMappingService:
    """
    获取股票代码映射服务单例
    
    Returns:
        SymbolMappingService实例
    """
    global _service
    if _service is None:
        _service = SymbolMappingService()
    return _service
