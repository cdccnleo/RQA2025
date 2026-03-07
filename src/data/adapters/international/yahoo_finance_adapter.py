#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Yahoo Finance 国际数据源适配器

功能：
- 获取美股、港股等国际市场数据
- 支持实时行情和历史数据
- 自动处理速率限制

作者: AI Assistant
创建日期: 2026-02-21
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from .base_international_adapter import (
    InternationalDataAdapter,
    MarketDataRequest,
    RealtimeQuote,
    AdapterStatus,
    MarketType,
    DataFrequency,
    RateLimitError,
    DataSourceError
)

logger = logging.getLogger(__name__)


class YahooFinanceAdapter(InternationalDataAdapter):
    """
    Yahoo Finance 数据适配器
    
    支持市场：
    - 美股 (US_STOCK)
    - 港股 (HK_STOCK)
    - 日股 (JP_STOCK)
    - 英股 (UK_STOCK)
    - 期货 (FUTURES)
    - 外汇 (FOREX)
    - 加密货币 (CRYPTO)
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化Yahoo Finance适配器
        
        Args:
            api_key: API密钥（可选，Yahoo Finance通常不需要）
        """
        super().__init__("YahooFinance", api_key)
        self._yf = None
        self._rate_limit_delay = 0.1  # 速率限制延迟（秒）
        self._last_request_time = None
        
    async def connect(self) -> bool:
        """
        连接Yahoo Finance
        
        Returns:
            是否连接成功
        """
        try:
            # 延迟导入yfinance库
            import yfinance as yf
            self._yf = yf
            
            # 测试连接 - 获取AAPL数据
            ticker = self._yf.Ticker("AAPL")
            info = ticker.info
            
            self._status.is_available = True
            self._status.last_check_time = datetime.now()
            self._status.rate_limit_remaining = 1000  # Yahoo Finance限制较宽松
            
            logger.info("Yahoo Finance 连接成功")
            return True
            
        except Exception as e:
            self._status.is_available = False
            self._status.error_message = str(e)
            logger.error(f"Yahoo Finance 连接失败: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """
        断开Yahoo Finance连接
        
        Returns:
            是否断开成功
        """
        self._yf = None
        self._status.is_available = False
        logger.info("Yahoo Finance 已断开")
        return True
    
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
        """
        if not self._yf:
            raise DataSourceError("Yahoo Finance 未连接")
        
        # 速率限制控制
        await self._rate_limit_control()
        
        try:
            # 标准化股票代码
            yahoo_symbol = self._to_yahoo_symbol(request.symbol, request.market)
            
            # 设置默认日期范围
            end_date = request.end_date or datetime.now()
            start_date = request.start_date or (end_date - timedelta(days=365))
            
            # 获取数据
            ticker = self._yf.Ticker(yahoo_symbol)
            
            # 转换频率
            interval = self._convert_frequency(request.frequency)
            
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=request.adjusted
            )
            
            if df.empty:
                logger.warning(f"未找到数据: {yahoo_symbol}")
                return pd.DataFrame()
            
            # 转换为标准格式
            df = self._convert_to_standard_df(df, request.symbol, request.market)
            
            logger.info(f"成功获取 {request.symbol} 的历史数据，共 {len(df)} 条")
            return df
            
        except Exception as e:
            logger.error(f"获取数据失败 {request.symbol}: {e}")
            raise DataSourceError(f"获取数据失败: {e}")
    
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
        if not self._yf:
            raise DataSourceError("Yahoo Finance 未连接")
        
        await self._rate_limit_control()
        
        try:
            yahoo_symbol = self._to_yahoo_symbol(symbol, market)
            ticker = self._yf.Ticker(yahoo_symbol)
            
            # 获取实时数据
            info = ticker.info
            history = ticker.history(period="1d", interval="1m")
            
            if history.empty:
                raise DataSourceError(f"无法获取 {symbol} 的实时数据")
            
            latest = history.iloc[-1]
            
            return RealtimeQuote(
                symbol=symbol,
                market=market,
                timestamp=datetime.now(),
                open=latest['Open'],
                high=latest['High'],
                low=latest['Low'],
                close=latest['Close'],
                volume=int(latest['Volume']),
                bid=info.get('bid'),
                ask=info.get('ask'),
                bid_volume=info.get('bidSize'),
                ask_volume=info.get('askSize')
            )
            
        except Exception as e:
            logger.error(f"获取实时行情失败 {symbol}: {e}")
            raise DataSourceError(f"获取实时行情失败: {e}")
    
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
        quotes = []
        
        # 使用并发获取
        tasks = [
            self.get_realtime_quote(symbol, market)
            for symbol in symbols
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, RealtimeQuote):
                quotes.append(result)
            elif isinstance(result, Exception):
                logger.warning(f"批量获取中某个股票失败: {result}")
        
        return quotes
    
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
        if not self._yf:
            raise DataSourceError("Yahoo Finance 未连接")
        
        await self._rate_limit_control()
        
        try:
            # 使用yfinance的搜索功能
            # 注意：yfinance本身没有搜索API，这里使用简单的匹配
            # 实际生产环境可能需要调用其他API
            
            # 模拟搜索结果
            results = []
            
            # 常见股票映射
            common_stocks = {
                'AAPL': {'name': 'Apple Inc.', 'market': MarketType.US_STOCK},
                'MSFT': {'name': 'Microsoft Corporation', 'market': MarketType.US_STOCK},
                'GOOGL': {'name': 'Alphabet Inc.', 'market': MarketType.US_STOCK},
                'AMZN': {'name': 'Amazon.com Inc.', 'market': MarketType.US_STOCK},
                'TSLA': {'name': 'Tesla Inc.', 'market': MarketType.US_STOCK},
                '0700.HK': {'name': 'Tencent Holdings', 'market': MarketType.HK_STOCK},
                '9988.HK': {'name': 'Alibaba Group', 'market': MarketType.HK_STOCK},
            }
            
            keyword_upper = keyword.upper()
            for symbol, info in common_stocks.items():
                if keyword_upper in symbol or keyword_upper in info['name'].upper():
                    if market is None or info['market'] == market:
                        results.append({
                            'symbol': symbol,
                            'name': info['name'],
                            'market': info['market'].value,
                            'exchange': self._get_exchange(info['market'])
                        })
            
            return results
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    
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
        if not self._yf:
            raise DataSourceError("Yahoo Finance 未连接")
        
        await self._rate_limit_control()
        
        try:
            yahoo_symbol = self._to_yahoo_symbol(symbol, market)
            ticker = self._yf.Ticker(yahoo_symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'market': market.value,
                'name': info.get('longName', info.get('shortName', '')),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'pb_ratio': info.get('priceToBook'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                'average_volume': info.get('averageVolume'),
                'website': info.get('website', ''),
                'description': info.get('longBusinessSummary', '')
            }
            
        except Exception as e:
            logger.error(f"获取公司信息失败 {symbol}: {e}")
            return {}
    
    async def check_health(self) -> AdapterStatus:
        """
        检查适配器健康状态
        
        Returns:
            适配器状态
        """
        try:
            if not self._yf:
                self._status.is_available = False
                return self._status
            
            # 测试获取AAPL数据
            ticker = self._yf.Ticker("AAPL")
            info = ticker.info
            
            self._status.is_available = True
            self._status.last_check_time = datetime.now()
            self._status.error_message = None
            
        except Exception as e:
            self._status.is_available = False
            self._status.error_message = str(e)
        
        return self._status
    
    def _to_yahoo_symbol(self, symbol: str, market: MarketType) -> str:
        """
        转换为Yahoo Finance股票代码格式
        
        Args:
            symbol: 原始股票代码
            market: 市场类型
            
        Returns:
            Yahoo Finance格式的股票代码
        """
        symbol = symbol.upper().strip()
        
        if market == MarketType.HK_STOCK:
            # 港股格式: 0700.HK
            if not symbol.endswith('.HK'):
                # 确保是4位数字代码
                symbol = symbol.zfill(4)
                return f"{symbol}.HK"
        elif market == MarketType.JP_STOCK:
            # 日股格式: 7203.T
            if not symbol.endswith('.T'):
                return f"{symbol}.T"
        elif market == MarketType.UK_STOCK:
            # 英股格式: VOD.L
            if not symbol.endswith('.L'):
                return f"{symbol}.L"
        elif market == MarketType.CRYPTO:
            # 加密货币格式: BTC-USD
            if '-' not in symbol:
                return f"{symbol}-USD"
        
        # 美股不需要后缀
        return symbol
    
    def _convert_frequency(self, frequency: DataFrequency) -> str:
        """
        转换频率为Yahoo Finance格式
        
        Args:
            frequency: 数据频率
            
        Returns:
            Yahoo Finance频率字符串
        """
        mapping = {
            DataFrequency.MINUTE_1: "1m",
            DataFrequency.MINUTE_5: "5m",
            DataFrequency.MINUTE_15: "15m",
            DataFrequency.MINUTE_30: "30m",
            DataFrequency.MINUTE_60: "60m",
            DataFrequency.DAILY: "1d",
            DataFrequency.WEEKLY: "1wk",
            DataFrequency.MONTHLY: "1mo"
        }
        return mapping.get(frequency, "1d")
    
    def _get_exchange(self, market: MarketType) -> str:
        """获取交易所名称"""
        mapping = {
            MarketType.US_STOCK: "NASDAQ/NYSE",
            MarketType.HK_STOCK: "Hong Kong Stock Exchange",
            MarketType.JP_STOCK: "Tokyo Stock Exchange",
            MarketType.UK_STOCK: "London Stock Exchange"
        }
        return mapping.get(market, "Unknown")
    
    async def _rate_limit_control(self):
        """速率限制控制"""
        if self._last_request_time:
            elapsed = (datetime.now() - self._last_request_time).total_seconds()
            if elapsed < self._rate_limit_delay:
                await asyncio.sleep(self._rate_limit_delay - elapsed)
        
        self._last_request_time = datetime.now()


# 全局适配器实例
_yahoo_adapter: Optional[YahooFinanceAdapter] = None


async def get_yahoo_finance_adapter() -> YahooFinanceAdapter:
    """
    获取Yahoo Finance适配器实例（单例模式）
    
    Returns:
        YahooFinanceAdapter实例
    """
    global _yahoo_adapter
    
    if _yahoo_adapter is None:
        _yahoo_adapter = YahooFinanceAdapter()
        await _yahoo_adapter.connect()
    
    return _yahoo_adapter
