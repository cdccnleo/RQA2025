#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多股票数据管理器

功能：
- 从策略配置获取股票代码
- 批量获取多股票数据
- 多级缓存支持

作者: AI Assistant
创建日期: 2026-02-21
"""

import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from .strategy_config_parser import get_strategy_config_parser, StrategyConfig
from .symbol_mapping_service import get_symbol_mapping_service

logger = logging.getLogger(__name__)


class MultiStockDataManager:
    """
    多股票数据管理器
    
    职责：
    1. 从策略配置获取股票代码
    2. 批量获取多股票数据
    3. 多级缓存支持
    """
    
    def __init__(self, max_workers: int = 4):
        """
        初始化多股票数据管理器
        
        Args:
            max_workers: 最大工作线程数
        """
        self.max_workers = max_workers
        self.config_parser = get_strategy_config_parser()
        self.symbol_mapper = get_symbol_mapping_service()
        
        # 简单内存缓存
        self._cache: Dict[str, Any] = {}
        self._cache_ttl: Dict[str, datetime] = {}
        self._cache_duration = timedelta(minutes=5)  # 默认缓存5分钟
        
        logger.info(f"多股票数据管理器初始化完成，最大工作线程数: {max_workers}")
    
    def get_symbols_from_strategy(self, strategy_id: str) -> List[str]:
        """
        从策略配置获取股票代码
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            股票代码列表
        """
        # 首先尝试从配置解析器获取
        symbols = self.config_parser.get_symbols_for_strategy(strategy_id)
        
        if symbols:
            logger.info(f"从策略配置获取到 {len(symbols)} 只股票: {strategy_id}")
            return symbols
        
        # 如果配置中没有，尝试从映射服务获取
        symbols = self.symbol_mapper.get_symbols_for_strategy(strategy_id)
        
        if symbols:
            logger.info(f"从映射服务获取到 {len(symbols)} 只股票: {strategy_id}")
            return symbols
        
        logger.warning(f"未找到策略的股票代码: {strategy_id}")
        return []
    
    def get_data_for_strategy(
        self,
        strategy_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        获取策略相关的股票数据
        
        Args:
            strategy_id: 策略ID
            start_date: 开始日期
            end_date: 结束日期
            use_cache: 是否使用缓存
            
        Returns:
            股票代码到DataFrame的映射
        """
        # 获取策略相关的股票代码
        symbols = self.get_symbols_from_strategy(strategy_id)
        
        if not symbols:
            logger.warning(f"策略没有配置股票代码: {strategy_id}")
            return {}
        
        logger.info(f"获取策略 {strategy_id} 的数据，共 {len(symbols)} 只股票")
        
        # 批量获取数据
        return self.get_batch_data(symbols, start_date, end_date, use_cache)
    
    def get_batch_data(
        self,
        symbols: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        批量获取多股票数据
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            use_cache: 是否使用缓存
            
        Returns:
            股票代码到DataFrame的映射
        """
        if not symbols:
            return {}
        
        # 去重
        symbols = list(set(symbols))
        
        results = {}
        symbols_to_fetch = []
        
        # 检查缓存
        if use_cache:
            for symbol in symbols:
                cache_key = self._get_cache_key(symbol, start_date, end_date)
                if cache_key in self._cache:
                    # 检查缓存是否过期
                    if datetime.now() < self._cache_ttl.get(cache_key, datetime.min):
                        results[symbol] = self._cache[cache_key]
                        logger.debug(f"从缓存获取数据: {symbol}")
                    else:
                        symbols_to_fetch.append(symbol)
                else:
                    symbols_to_fetch.append(symbol)
        else:
            symbols_to_fetch = symbols
        
        if not symbols_to_fetch:
            logger.info(f"所有数据从缓存获取，共 {len(results)} 只股票")
            return results
        
        logger.info(f"从数据库获取 {len(symbols_to_fetch)} 只股票的数据")
        
        # 并行获取数据
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_symbol = {
                executor.submit(
                    self._fetch_single_stock_data,
                    symbol,
                    start_date,
                    end_date
                ): symbol
                for symbol in symbols_to_fetch
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    df = future.result()
                    if df is not None and not df.empty:
                        results[symbol] = df
                        
                        # 更新缓存
                        if use_cache:
                            cache_key = self._get_cache_key(symbol, start_date, end_date)
                            self._cache[cache_key] = df
                            self._cache_ttl[cache_key] = datetime.now() + self._cache_duration
                    else:
                        logger.warning(f"未获取到数据: {symbol}")
                except Exception as e:
                    logger.error(f"获取数据失败 {symbol}: {e}")
        
        logger.info(f"批量获取完成，成功 {len(results)}/{len(symbols)} 只股票")
        return results
    
    def _fetch_single_stock_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """
        获取单只股票数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame或None
        """
        try:
            from src.gateway.web.postgresql_persistence import get_db_connection
            
            conn = get_db_connection()
            
            # 构建查询
            query = """
                SELECT symbol, date, open_price, high_price, low_price, close_price,
                       volume, amount, pct_change, change, turnover_rate, amplitude
                FROM akshare_stock_data
                WHERE symbol = %s
            """
            params = [symbol]
            
            if start_date:
                query += " AND date >= %s"
                params.append(start_date.strftime('%Y-%m-%d'))
            
            if end_date:
                query += " AND date <= %s"
                params.append(end_date.strftime('%Y-%m-%d'))
            
            query += " ORDER BY date ASC"
            
            # 执行查询
            df = pd.read_sql_query(query, conn, params=params)
            
            conn.close()
            
            if df.empty:
                return None
            
            # 转换列名
            column_mapping = {
                'open_price': 'open',
                'high_price': 'high',
                'low_price': 'low',
                'close_price': 'close',
                'turnover_rate': 'turnover'
            }
            df = df.rename(columns=column_mapping)
            
            return df
            
        except Exception as e:
            logger.error(f"获取股票数据失败 {symbol}: {e}")
            return None
    
    def _get_cache_key(
        self,
        symbol: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> str:
        """
        生成缓存键
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            缓存键
        """
        start_str = start_date.strftime('%Y%m%d') if start_date else 'none'
        end_str = end_date.strftime('%Y%m%d') if end_date else 'none'
        return f"{symbol}_{start_str}_{end_str}"
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        self._cache_ttl.clear()
        logger.info("缓存已清空")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            统计信息字典
        """
        total_keys = len(self._cache)
        expired_keys = sum(
            1 for key, ttl in self._cache_ttl.items()
            if datetime.now() > ttl
        )
        
        return {
            'total_keys': total_keys,
            'expired_keys': expired_keys,
            'valid_keys': total_keys - expired_keys
        }
    
    def register_strategy_mapping(
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
        return self.symbol_mapper.register_mapping(strategy_id, symbols, priority, weight)
    
    def create_strategy_config(
        self,
        strategy_id: str,
        symbols: List[str],
        **kwargs
    ) -> bool:
        """
        创建策略配置
        
        Args:
            strategy_id: 策略ID
            symbols: 股票代码列表
            **kwargs: 其他配置参数
            
        Returns:
            是否创建成功
        """
        try:
            config = self.config_parser.create_default_config(strategy_id, symbols)
            
            # 更新其他参数
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            # 保存配置
            return self.config_parser.save_config(config)
            
        except Exception as e:
            logger.error(f"创建策略配置失败 {strategy_id}: {e}")
            return False
    
    def get_all_strategy_symbols(self) -> Dict[str, List[str]]:
        """
        获取所有策略的股票代码
        
        Returns:
            策略ID到股票代码列表的映射
        """
        result = {}
        
        # 从配置解析器获取
        configs = self.config_parser.load_all_configs()
        for strategy_id, config in configs.items():
            result[strategy_id] = config.symbols
        
        # 从映射服务获取
        mappings = self.symbol_mapper.get_all_mappings()
        for strategy_id, mapping in mappings.items():
            if strategy_id not in result:
                result[strategy_id] = mapping.symbols
            else:
                # 合并股票代码
                result[strategy_id] = list(set(result[strategy_id] + mapping.symbols))
        
        return result


# 单例实例
_manager: Optional[MultiStockDataManager] = None


def get_multi_stock_data_manager(max_workers: int = 4) -> MultiStockDataManager:
    """
    获取多股票数据管理器单例
    
    Args:
        max_workers: 最大工作线程数
        
    Returns:
        MultiStockDataManager实例
    """
    global _manager
    if _manager is None:
        _manager = MultiStockDataManager(max_workers=max_workers)
    return _manager
