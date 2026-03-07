#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alpha Vantage 国际数据源适配器

功能：
- 获取美股、外汇、加密货币数据
- 支持技术指标计算
- 专业级金融数据API

作者: AI Assistant
创建日期: 2026-02-21
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import aiohttp

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


class AlphaVantageAdapter(InternationalDataAdapter):
    """
    Alpha Vantage 数据适配器
    
    支持市场：
    - 美股 (US_STOCK)
    - 外汇 (FOREX)
    - 加密货币 (CRYPTO)
    
    特点：
    - 提供技术指标数据
    - 支持多种时间序列
    - 专业级金融数据
    """
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    def __init__(self, api_key: str):
        """
        初始化Alpha Vantage适配器
        
        Args:
            api_key: Alpha Vantage API密钥
        """
        super().__init__("AlphaVantage", api_key)
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limit_per_minute = 5  # 免费版限制
        self._request_count = 0
        self._last_reset_time = datetime.now()
        
    async def connect(self) -> bool:
        """
        连接Alpha Vantage API
        
        Returns:
            是否连接成功
        """
        try:
            if not self.api_key:
                raise ValueError("Alpha Vantage需要API密钥")
            
            # 创建HTTP会话
            self._session = aiohttp.ClientSession()
            
            # 测试连接 - 获取IBM数据
            test_data = await self._make_request({
                "function": "TIME_SERIES_INTRADAY",
                "symbol": "IBM",
                "interval": "5min",
                "apikey": self.api_key
            })
            
            if "Time Series (5min)" in test_data or "Meta Data" in test_data:
                self._status.is_available = True
                self._status.last_check_time = datetime.now()
                self._status.rate_limit_remaining = self._rate_limit_per_minute
                
                logger.info("Alpha Vantage 连接成功")
                return True
            else:
                error_msg = test_data.get("Note", test_data.get("Information", "未知错误"))
                raise ConnectionError(f"API测试失败: {error_msg}")
                
        except Exception as e:
            self._status.is_available = False
            self._status.error_message = str(e)
            logger.error(f"Alpha Vantage 连接失败: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """
        断开Alpha Vantage连接
        
        Returns:
            是否断开成功
        """
        if self._session:
            await self._session.close()
            self._session = None
        
        self._status.is_available = False
        logger.info("Alpha Vantage 已断开")
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
        if not self._session:
            raise DataSourceError("Alpha Vantage 未连接")
        
        # 检查速率限制
        await self._check_rate_limit()
        
        try:
            # 根据市场类型选择API函数
            if request.market == MarketType.US_STOCK:
                data = await self._fetch_stock_data(request)
            elif request.market == MarketType.FOREX:
                data = await self._fetch_forex_data(request)
            elif request.market == MarketType.CRYPTO:
                data = await self._fetch_crypto_data(request)
            else:
                raise ValueError(f"不支持的市场类型: {request.market}")
            
            # 转换为DataFrame
            df = self._parse_time_series(data, request.symbol, request.market)
            
            logger.info(f"成功获取 {request.symbol} 的历史数据，共 {len(df)} 条")
            return df
            
        except Exception as e:
            logger.error(f"获取数据失败 {request.symbol}: {e}")
            raise DataSourceError(f"获取数据失败: {e}")
    
    async def _fetch_stock_data(self, request: MarketDataRequest) -> Dict:
        """获取股票数据"""
        # 根据频率选择函数
        if request.frequency in [DataFrequency.MINUTE_1, DataFrequency.MINUTE_5, 
                                  DataFrequency.MINUTE_15, DataFrequency.MINUTE_30,
                                  DataFrequency.MINUTE_60]:
            interval = request.frequency.value.replace('m', 'min')
            params = {
                "function": "TIME_SERIES_INTRADAY",
                "symbol": request.symbol,
                "interval": interval,
                "apikey": self.api_key,
                "outputsize": "full"
            }
        elif request.frequency == DataFrequency.DAILY:
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": request.symbol,
                "apikey": self.api_key,
                "outputsize": "full"
            }
        elif request.frequency == DataFrequency.WEEKLY:
            params = {
                "function": "TIME_SERIES_WEEKLY",
                "symbol": request.symbol,
                "apikey": self.api_key
            }
        else:  # MONTHLY
            params = {
                "function": "TIME_SERIES_MONTHLY",
                "symbol": request.symbol,
                "apikey": self.api_key
            }
        
        return await self._make_request(params)
    
    async def _fetch_forex_data(self, request: MarketDataRequest) -> Dict:
        """获取外汇数据"""
        # 解析货币对 (如 EURUSD)
        if len(request.symbol) == 6:
            from_currency = request.symbol[:3]
            to_currency = request.symbol[3:]
        else:
            from_currency = request.symbol
            to_currency = "USD"
        
        params = {
            "function": "FX_DAILY",
            "from_symbol": from_currency,
            "to_symbol": to_currency,
            "apikey": self.api_key,
            "outputsize": "full"
        }
        
        return await self._make_request(params)
    
    async def _fetch_crypto_data(self, request: MarketDataRequest) -> Dict:
        """获取加密货币数据"""
        symbol = request.symbol.replace("-USD", "").replace("-USDT", "")
        
        params = {
            "function": "DIGITAL_CURRENCY_DAILY",
            "symbol": symbol,
            "market": "USD",
            "apikey": self.api_key
        }
        
        return await self._make_request(params)
    
    def _parse_time_series(
        self,
        data: Dict,
        symbol: str,
        market: MarketType
    ) -> pd.DataFrame:
        """
        解析时间序列数据
        
        Args:
            data: API返回数据
            symbol: 股票代码
            market: 市场类型
            
        Returns:
            DataFrame
        """
        # 查找时间序列数据
        time_series_key = None
        for key in data.keys():
            if "Time Series" in key or "Time Series (Digital Currency)" in key:
                time_series_key = key
                break
        
        if not time_series_key:
            raise DataSourceError(f"未找到时间序列数据: {data.keys()}")
        
        time_series = data[time_series_key]
        
        # 转换为DataFrame
        records = []
        for timestamp, values in time_series.items():
            record = {"timestamp": pd.to_datetime(timestamp)}
            
            # 标准化列名
            for key, value in values.items():
                key_lower = key.lower()
                if "open" in key_lower and "b" not in key_lower:
                    record["open"] = float(value)
                elif "high" in key_lower and "b" not in key_lower:
                    record["high"] = float(value)
                elif "low" in key_lower and "b" not in key_lower:
                    record["low"] = float(value)
                elif "close" in key_lower and "b" not in key_lower:
                    record["close"] = float(value)
                elif "volume" in key_lower:
                    record["volume"] = float(value)
            
            records.append(record)
        
        df = pd.DataFrame(records)
        df = df.sort_values("timestamp")
        df = df.reset_index(drop=True)
        
        # 添加元数据
        df["symbol"] = symbol
        df["market"] = market.value
        
        return df
    
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
        # Alpha Vantage的实时数据通过intraday API获取最新数据
        request = MarketDataRequest(
            symbol=symbol,
            market=market,
            frequency=DataFrequency.MINUTE_1
        )
        
        df = await self.fetch_market_data(request)
        
        if df.empty:
            raise DataSourceError(f"无法获取 {symbol} 的实时数据")
        
        latest = df.iloc[-1]
        
        return RealtimeQuote(
            symbol=symbol,
            market=market,
            timestamp=latest["timestamp"],
            open=latest["open"],
            high=latest["high"],
            low=latest["low"],
            close=latest["close"],
            volume=int(latest["volume"])
        )
    
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
        
        # Alpha Vantage速率限制较严格，需要串行处理
        for symbol in symbols:
            try:
                quote = await self.get_realtime_quote(symbol, market)
                quotes.append(quote)
                await asyncio.sleep(12)  # 免费版限制：每分钟5次请求
            except Exception as e:
                logger.warning(f"获取 {symbol} 实时行情失败: {e}")
        
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
        if not self._session:
            raise DataSourceError("Alpha Vantage 未连接")
        
        await self._check_rate_limit()
        
        params = {
            "function": "SYMBOL_SEARCH",
            "keywords": keyword,
            "apikey": self.api_key
        }
        
        try:
            data = await self._make_request(params)
            
            results = []
            if "bestMatches" in data:
                for match in data["bestMatches"]:
                    symbol = match.get("1. symbol", "")
                    
                    # 根据市场类型过滤
                    if market:
                        asset_type = match.get("3. type", "").upper()
                        if market == MarketType.US_STOCK and asset_type != "EQUITY":
                            continue
                        if market == MarketType.FOREX and asset_type != "FOREX":
                            continue
                    
                    results.append({
                        "symbol": symbol,
                        "name": match.get("2. name", ""),
                        "type": match.get("3. type", ""),
                        "region": match.get("4. region", ""),
                        "market": self._detect_market_type(match),
                        "currency": match.get("8. currency", "")
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
        if not self._session:
            raise DataSourceError("Alpha Vantage 未连接")
        
        await self._check_rate_limit()
        
        params = {
            "function": "OVERVIEW",
            "symbol": symbol,
            "apikey": self.api_key
        }
        
        try:
            data = await self._make_request(params)
            
            return {
                "symbol": symbol,
                "market": market.value,
                "name": data.get("Name", ""),
                "description": data.get("Description", ""),
                "sector": data.get("Sector", ""),
                "industry": data.get("Industry", ""),
                "market_cap": data.get("MarketCapitalization"),
                "pe_ratio": data.get("PERatio"),
                "pb_ratio": data.get("PriceToBookRatio"),
                "dividend_yield": data.get("DividendYield"),
                "beta": data.get("Beta"),
                "52_week_high": data.get("52WeekHigh"),
                "52_week_low": data.get("52WeekLow"),
                "avg_volume": data.get("AverageVolume"),
                "exchange": data.get("Exchange", "")
            }
            
        except Exception as e:
            logger.error(f"获取公司信息失败 {symbol}: {e}")
            return {}
    
    async def get_technical_indicator(
        self,
        symbol: str,
        indicator: str,
        interval: str = "daily",
        time_period: int = 14
    ) -> pd.DataFrame:
        """
        获取技术指标
        
        Args:
            symbol: 股票代码
            indicator: 指标名称 (SMA, EMA, RSI, MACD等)
            interval: 时间间隔
            time_period: 时间周期
            
        Returns:
            指标数据DataFrame
        """
        if not self._session:
            raise DataSourceError("Alpha Vantage 未连接")
        
        await self._check_rate_limit()
        
        params = {
            "function": indicator.upper(),
            "symbol": symbol,
            "interval": interval,
            "time_period": time_period,
            "series_type": "close",
            "apikey": self.api_key
        }
        
        data = await self._make_request(params)
        
        # 解析技术指标数据
        indicator_key = None
        for key in data.keys():
            if "Technical Analysis" in key:
                indicator_key = key
                break
        
        if not indicator_key:
            raise DataSourceError(f"未找到技术指标数据: {data.keys()}")
        
        tech_data = data[indicator_key]
        
        records = []
        for timestamp, values in tech_data.items():
            record = {"timestamp": pd.to_datetime(timestamp)}
            record.update(values)
            records.append(record)
        
        df = pd.DataFrame(records)
        df = df.sort_values("timestamp")
        
        return df
    
    async def check_health(self) -> AdapterStatus:
        """
        检查适配器健康状态
        
        Returns:
            适配器状态
        """
        try:
            if not self._session:
                self._status.is_available = False
                return self._status
            
            # 测试API
            params = {
                "function": "TIME_SERIES_INTRADAY",
                "symbol": "IBM",
                "interval": "5min",
                "apikey": self.api_key
            }
            
            data = await self._make_request(params)
            
            if "Time Series (5min)" in data or "Meta Data" in data:
                self._status.is_available = True
                self._status.last_check_time = datetime.now()
                self._status.error_message = None
            else:
                self._status.is_available = False
                self._status.error_message = "API返回异常数据"
                
        except Exception as e:
            self._status.is_available = False
            self._status.error_message = str(e)
        
        return self._status
    
    async def _make_request(self, params: Dict[str, Any]) -> Dict:
        """
        发送API请求
        
        Args:
            params: 请求参数
            
        Returns:
            API响应数据
        """
        if not self._session:
            raise DataSourceError("会话未创建")
        
        async with self._session.get(self.BASE_URL, params=params) as response:
            if response.status == 429:
                raise RateLimitError("超出Alpha Vantage速率限制")
            
            response.raise_for_status()
            data = await response.json()
            
            # 检查错误信息
            if "Error Message" in data:
                raise DataSourceError(data["Error Message"])
            
            if "Note" in data:
                # 通常是速率限制警告
                logger.warning(f"Alpha Vantage警告: {data['Note']}")
            
            self._request_count += 1
            
            return data
    
    async def _check_rate_limit(self):
        """检查速率限制"""
        now = datetime.now()
        
        # 每分钟重置计数
        if (now - self._last_reset_time).total_seconds() >= 60:
            self._request_count = 0
            self._last_reset_time = now
        
        # 检查是否超出限制
        if self._request_count >= self._rate_limit_per_minute:
            wait_time = 60 - (now - self._last_reset_time).total_seconds()
            if wait_time > 0:
                logger.warning(f"达到Alpha Vantage速率限制，等待 {wait_time:.1f} 秒")
                await asyncio.sleep(wait_time)
                self._request_count = 0
                self._last_reset_time = datetime.now()
    
    def _detect_market_type(self, match: Dict) -> str:
        """检测市场类型"""
        asset_type = match.get("3. type", "").upper()
        region = match.get("4. region", "").upper()
        
        if asset_type == "EQUITY":
            if "UNITED STATES" in region:
                return MarketType.US_STOCK.value
            else:
                return MarketType.US_STOCK.value  # 默认为美股
        elif asset_type == "FOREX":
            return MarketType.FOREX.value
        elif asset_type == "CRYPTO":
            return MarketType.CRYPTO.value
        
        return MarketType.US_STOCK.value


# 全局适配器实例
_alpha_vantage_adapter: Optional[AlphaVantageAdapter] = None


async def get_alpha_vantage_adapter(api_key: Optional[str] = None) -> AlphaVantageAdapter:
    """
    获取Alpha Vantage适配器实例
    
    Args:
        api_key: API密钥（可选，如果不提供则尝试从环境变量获取）
        
    Returns:
        AlphaVantageAdapter实例
    """
    global _alpha_vantage_adapter
    
    if _alpha_vantage_adapter is None:
        if api_key is None:
            import os
            api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        
        if not api_key:
            raise ValueError("需要提供Alpha Vantage API密钥")
        
        _alpha_vantage_adapter = AlphaVantageAdapter(api_key)
        await _alpha_vantage_adapter.connect()
    
    return _alpha_vantage_adapter
