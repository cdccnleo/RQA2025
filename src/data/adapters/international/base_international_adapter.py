#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
国际数据源适配器基类

功能：
- 定义国际数据源适配器标准接口
- 支持美股、港股、期货等国际市场数据获取
- 统一数据格式和错误处理

作者: AI Assistant
创建日期: 2026-02-21
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class MarketType(Enum):
    """市场类型"""
    US_STOCK = "us_stock"           # 美股
    HK_STOCK = "hk_stock"           # 港股
    JP_STOCK = "jp_stock"           # 日股
    UK_STOCK = "uk_stock"           # 英股
    FUTURES = "futures"             # 期货
    FOREX = "forex"                 # 外汇
    CRYPTO = "crypto"               # 加密货币


class DataFrequency(Enum):
    """数据频率"""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    MINUTE_60 = "60m"
    DAILY = "1d"
    WEEKLY = "1wk"
    MONTHLY = "1mo"


@dataclass
class MarketDataRequest:
    """市场数据请求"""
    symbol: str
    market: MarketType
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    frequency: DataFrequency = DataFrequency.DAILY
    adjusted: bool = True


@dataclass
class RealtimeQuote:
    """实时行情数据"""
    symbol: str
    market: MarketType
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_volume: Optional[int] = None
    ask_volume: Optional[int] = None


@dataclass
class AdapterStatus:
    """适配器状态"""
    name: str
    is_available: bool
    last_check_time: datetime
    rate_limit_remaining: int
    rate_limit_reset_time: Optional[datetime] = None
    error_message: Optional[str] = None


class InternationalDataAdapter(ABC):
    """
    国际数据源适配器基类
    
    所有国际市场数据源适配器必须继承此类
    """
    
    def __init__(self, name: str, api_key: Optional[str] = None):
        """
        初始化适配器
        
        Args:
            name: 适配器名称
            api_key: API密钥
        """
        self.name = name
        self.api_key = api_key
        self._status = AdapterStatus(
            name=name,
            is_available=False,
            last_check_time=datetime.now(),
            rate_limit_remaining=0
        )
        self._session = None
        
        logger.info(f"国际数据源适配器 {name} 初始化完成")
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        连接数据源
        
        Returns:
            是否连接成功
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """
        断开数据源连接
        
        Returns:
            是否断开成功
        """
        pass
    
    @abstractmethod
    async def fetch_market_data(
        self,
        request: MarketDataRequest
    ) -> pd.DataFrame:
        """
        获取市场历史数据
        
        Args:
            request: 数据请求参数
            
        Returns:
            DataFrame包含OHLCV数据
            
        Raises:
            ConnectionError: 连接失败
            ValueError: 参数无效
            RateLimitError: 超出速率限制
        """
        pass
    
    @abstractmethod
    async def get_realtime_quote(
        self,
        symbol: str,
        market: MarketType
    ) -> RealtimeQuote:
        """
        获取实时行情
        
        Args:
            symbol: 股票代码
            market: 市场类型
            
        Returns:
            实时行情数据
        """
        pass
    
    @abstractmethod
    async def get_batch_realtime_quotes(
        self,
        symbols: List[str],
        market: MarketType
    ) -> List[RealtimeQuote]:
        """
        批量获取实时行情
        
        Args:
            symbols: 股票代码列表
            market: 市场类型
            
        Returns:
            实时行情数据列表
        """
        pass
    
    @abstractmethod
    async def search_symbols(
        self,
        keyword: str,
        market: Optional[MarketType] = None
    ) -> List[Dict[str, Any]]:
        """
        搜索股票代码
        
        Args:
            keyword: 搜索关键词
            market: 市场类型（可选）
            
        Returns:
            匹配的股票列表
        """
        pass
    
    @abstractmethod
    async def get_company_info(
        self,
        symbol: str,
        market: MarketType
    ) -> Dict[str, Any]:
        """
        获取公司信息
        
        Args:
            symbol: 股票代码
            market: 市场类型
            
        Returns:
            公司信息字典
        """
        pass
    
    @abstractmethod
    async def check_health(self) -> AdapterStatus:
        """
        检查适配器健康状态
        
        Returns:
            适配器状态
        """
        pass
    
    def get_status(self) -> AdapterStatus:
        """获取适配器状态"""
        return self._status
    
    def _standardize_symbol(
        self,
        symbol: str,
        market: MarketType
    ) -> str:
        """
        标准化股票代码格式
        
        Args:
            symbol: 原始股票代码
            market: 市场类型
            
        Returns:
            标准化后的股票代码
        """
        symbol = symbol.upper().strip()
        
        if market == MarketType.HK_STOCK and not symbol.endswith('.HK'):
            return f"{symbol}.HK"
        elif market == MarketType.US_STOCK:
            # 美股不需要后缀
            return symbol
        elif market == MarketType.JP_STOCK and not symbol.endswith('.T'):
            return f"{symbol}.T"
        
        return symbol
    
    def _convert_to_standard_df(
        self,
        df: pd.DataFrame,
        symbol: str,
        market: MarketType
    ) -> pd.DataFrame:
        """
        转换为标准格式DataFrame
        
        Args:
            df: 原始数据
            symbol: 股票代码
            market: 市场类型
            
        Returns:
            标准格式DataFrame
        """
        # 确保必要的列存在
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # 标准化列名
        column_mapping = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close'
        }
        
        df = df.rename(columns=column_mapping)
        
        # 添加元数据
        df['symbol'] = symbol
        df['market'] = market.value
        
        return df


class RateLimitError(Exception):
    """速率限制错误"""
    pass


class DataSourceError(Exception):
    """数据源错误"""
    pass
