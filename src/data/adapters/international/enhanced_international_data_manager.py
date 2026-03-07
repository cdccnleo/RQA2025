"""
增强型国际数据管理器 - 多源数据整合

本模块提供统一的国际市场数据管理，支持：
1. 多数据源整合（Yahoo Finance、Alpha Vantage等）
2. 智能数据源选择
3. 数据质量校验和融合
4. 跨市场数据对齐
5. 实时和历史数据统一管理

作者: 数据团队
创建日期: 2026-02-21
版本: 2.0.0
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np

from .base_international_adapter import (
    InternationalDataAdapter,
    MarketDataRequest,
    RealtimeQuote,
    AdapterStatus,
    MarketType,
    DataFrequency,
    DataSourceError
)
from .yahoo_finance_adapter import YahooFinanceAdapter
from .alpha_vantage_adapter import AlphaVantageAdapter


# 配置日志
logger = logging.getLogger(__name__)


class DataSourcePriority(Enum):
    """数据源优先级"""
    PRIMARY = 1      # 主数据源
    SECONDARY = 2    # 备用数据源
    FALLBACK = 3     # 降级数据源


@dataclass
class DataSourceConfig:
    """数据源配置"""
    adapter: InternationalDataAdapter
    priority: DataSourcePriority
    markets: List[MarketType]
    max_requests_per_minute: int
    is_active: bool = True
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None


@dataclass
class DataQualityReport:
    """数据质量报告"""
    symbol: str
    market: MarketType
    completeness: float           # 完整度 0-1
    timeliness: float             # 及时性 0-1
    accuracy_score: float         # 准确性 0-1
    source_reliability: float     # 源可靠性 0-1
    overall_score: float          # 综合评分 0-1
    issues: List[str] = field(default_factory=list)


class EnhancedInternationalDataManager:
    """
    增强型国际数据管理器
    
    功能:
    1. 多数据源整合和管理
    2. 智能数据源选择和故障转移
    3. 数据质量校验和评分
    4. 跨市场数据对齐和标准化
    5. 统一的实时和历史数据接口
    
    使用示例:
        manager = EnhancedInternationalDataManager()
        
        # 注册数据源
        await manager.register_data_source(
            YahooFinanceAdapter(),
            priority=DataSourcePriority.PRIMARY,
            markets=[MarketType.US_STOCK, MarketType.HK_STOCK]
        )
        
        # 获取数据
        data = await manager.fetch_market_data(
            symbol="AAPL",
            market=MarketType.US_STOCK,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31)
        )
    """
    
    def __init__(self):
        """初始化数据管理器"""
        self._data_sources: Dict[str, DataSourceConfig] = {}
        self._market_data_cache: Dict[str, pd.DataFrame] = {}
        self._cache_ttl_seconds = 300  # 缓存5分钟
        self._cache_timestamps: Dict[str, datetime] = {}
        
        # 统计信息
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "fallback_uses": 0
        }
        
        logger.info("增强型国际数据管理器已初始化")
    
    async def register_data_source(
        self,
        adapter: InternationalDataAdapter,
        priority: DataSourcePriority = DataSourcePriority.PRIMARY,
        markets: Optional[List[MarketType]] = None,
        max_requests_per_minute: int = 100
    ) -> bool:
        """
        注册数据源
        
        参数:
            adapter: 数据适配器
            priority: 数据源优先级
            markets: 支持的市场列表
            max_requests_per_minute: 每分钟最大请求数
            
        返回:
            bool: 注册是否成功
        """
        try:
            # 测试连接
            if not await adapter.connect():
                logger.warning(f"数据源连接失败: {adapter.name}")
                return False
            
            # 创建配置
            config = DataSourceConfig(
                adapter=adapter,
                priority=priority,
                markets=markets or list(MarketType),
                max_requests_per_minute=max_requests_per_minute
            )
            
            self._data_sources[adapter.name] = config
            
            logger.info(f"数据源已注册: {adapter.name} (优先级: {priority.name})")
            return True
            
        except Exception as e:
            logger.error(f"注册数据源失败 {adapter.name}: {e}")
            return False
    
    async def fetch_market_data(
        self,
        symbol: str,
        market: MarketType,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        frequency: DataFrequency = DataFrequency.DAILY,
        adjusted: bool = True,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        获取市场历史数据
        
        参数:
            symbol: 股票代码
            market: 市场类型
            start_date: 开始日期
            end_date: 结束日期
            frequency: 数据频率
            adjusted: 是否复权
            use_cache: 是否使用缓存
            
        返回:
            DataFrame: OHLCV数据
        """
        self._stats["total_requests"] += 1
        
        # 生成缓存键
        cache_key = f"{market.value}:{symbol}:{frequency.value}:{start_date}:{end_date}"
        
        # 检查缓存
        if use_cache and cache_key in self._market_data_cache:
            cache_time = self._cache_timestamps.get(cache_key)
            if cache_time and (datetime.now() - cache_time).seconds < self._cache_ttl_seconds:
                self._stats["cache_hits"] += 1
                logger.debug(f"缓存命中: {cache_key}")
                return self._market_data_cache[cache_key].copy()
        
        # 创建请求
        request = MarketDataRequest(
            symbol=symbol,
            market=market,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            adjusted=adjusted
        )
        
        # 获取排序后的数据源
        sorted_sources = self._get_sorted_data_sources(market)
        
        last_error = None
        
        # 尝试从各个数据源获取数据
        for source_name, config in sorted_sources:
            if not config.is_active:
                continue
            
            try:
                logger.info(f"尝试从 {source_name} 获取数据: {symbol}")
                
                data = await config.adapter.fetch_market_data(request)
                
                if not data.empty:
                    # 数据质量检查
                    quality_report = self._check_data_quality(data, symbol, market)
                    
                    if quality_report.overall_score >= 0.7:  # 质量阈值
                        # 缓存数据
                        if use_cache:
                            self._market_data_cache[cache_key] = data.copy()
                            self._cache_timestamps[cache_key] = datetime.now()
                        
                        self._stats["successful_requests"] += 1
                        
                        logger.info(f"成功从 {source_name} 获取数据: {symbol}, "
                                  f"质量评分: {quality_report.overall_score:.2f}")
                        
                        return data
                    else:
                        logger.warning(f"{source_name} 数据质量不佳: {quality_report.overall_score:.2f}")
                        
            except Exception as e:
                logger.warning(f"从 {source_name} 获取数据失败: {e}")
                last_error = e
                
                # 记录失败
                config.failure_count += 1
                config.last_failure_time = datetime.now()
                
                # 如果失败次数过多，暂时停用
                if config.failure_count >= 3:
                    config.is_active = False
                    logger.warning(f"数据源 {source_name} 已停用（失败次数过多）")
        
        # 所有数据源都失败
        self._stats["failed_requests"] += 1
        
        if last_error:
            raise DataSourceError(f"所有数据源都无法获取 {symbol} 的数据: {last_error}")
        else:
            raise DataSourceError(f"无法获取 {symbol} 的数据，所有数据源都不可用")
    
    async def get_realtime_quote(
        self,
        symbol: str,
        market: MarketType,
        timeout: float = 5.0
    ) -> RealtimeQuote:
        """
        获取实时行情
        
        参数:
            symbol: 股票代码
            market: 市场类型
            timeout: 超时时间（秒）
            
        返回:
            RealtimeQuote: 实时行情数据
        """
        sorted_sources = self._get_sorted_data_sources(market)
        
        for source_name, config in sorted_sources:
            if not config.is_active:
                continue
            
            try:
                # 使用超时
                quote = await asyncio.wait_for(
                    config.adapter.get_realtime_quote(symbol, market),
                    timeout=timeout
                )
                
                if quote:
                    logger.debug(f"从 {source_name} 获取实时行情: {symbol}")
                    return quote
                    
            except asyncio.TimeoutError:
                logger.warning(f"{source_name} 获取实时行情超时: {symbol}")
            except Exception as e:
                logger.warning(f"从 {source_name} 获取实时行情失败: {e}")
        
        raise DataSourceError(f"无法获取 {symbol} 的实时行情")
    
    async def get_batch_realtime_quotes(
        self,
        symbols: List[str],
        market: MarketType,
        max_concurrent: int = 10
    ) -> Dict[str, RealtimeQuote]:
        """
        批量获取实时行情
        
        参数:
            symbols: 股票代码列表
            market: 市场类型
            max_concurrent: 最大并发数
            
        返回:
            Dict[str, RealtimeQuote]: 实时行情字典
        """
        results = {}
        
        # 分批处理
        for i in range(0, len(symbols), max_concurrent):
            batch = symbols[i:i + max_concurrent]
            
            tasks = [
                self._get_quote_with_fallback(symbol, market)
                for symbol in batch
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for symbol, result in zip(batch, batch_results):
                if isinstance(result, RealtimeQuote):
                    results[symbol] = result
                elif isinstance(result, Exception):
                    logger.warning(f"获取 {symbol} 实时行情失败: {result}")
        
        return results
    
    async def get_cross_market_data(
        self,
        symbols_by_market: Dict[MarketType, List[str]],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        frequency: DataFrequency = DataFrequency.DAILY
    ) -> Dict[MarketType, Dict[str, pd.DataFrame]]:
        """
        获取跨市场数据
        
        参数:
            symbols_by_market: 按市场分组的股票代码
            start_date: 开始日期
            end_date: 结束日期
            frequency: 数据频率
            
        返回:
            Dict: 跨市场数据
        """
        results = {}
        
        for market, symbols in symbols_by_market.items():
            market_data = {}
            
            for symbol in symbols:
                try:
                    data = await self.fetch_market_data(
                        symbol=symbol,
                        market=market,
                        start_date=start_date,
                        end_date=end_date,
                        frequency=frequency
                    )
                    
                    if not data.empty:
                        market_data[symbol] = data
                        
                except Exception as e:
                    logger.warning(f"获取跨市场数据失败 {market.value}:{symbol}: {e}")
            
            if market_data:
                results[market] = market_data
        
        return results
    
    async def align_cross_market_data(
        self,
        data_by_market: Dict[MarketType, Dict[str, pd.DataFrame]],
        timezone: str = "UTC"
    ) -> pd.DataFrame:
        """
        对齐跨市场数据
        
        参数:
            data_by_market: 跨市场数据
            timezone: 目标时区
            
        返回:
            DataFrame: 对齐后的数据
        """
        aligned_data = []
        
        for market, symbols_data in data_by_market.items():
            for symbol, df in symbols_data.items():
                if df.empty:
                    continue
                
                # 添加市场标识
                df_copy = df.copy()
                df_copy['market'] = market.value
                df_copy['symbol'] = symbol
                
                # 时区转换（简化处理）
                # 实际生产环境需要使用时区库进行转换
                
                aligned_data.append(df_copy)
        
        if not aligned_data:
            return pd.DataFrame()
        
        # 合并所有数据
        combined = pd.concat(aligned_data, ignore_index=True)
        
        # 按时间排序
        if 'timestamp' in combined.columns:
            combined = combined.sort_values('timestamp')
        elif 'date' in combined.columns:
            combined = combined.sort_values('date')
        
        return combined
    
    def get_data_source_status(self) -> Dict[str, AdapterStatus]:
        """
        获取所有数据源状态
        
        返回:
            Dict[str, AdapterStatus]: 数据源状态字典
        """
        status = {}
        
        for name, config in self._data_sources.items():
            adapter_status = config.adapter.get_status()
            adapter_status.is_available = config.is_active
            status[name] = adapter_status
        
        return status
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        返回:
            Dict: 统计信息
        """
        total = self._stats["total_requests"]
        success_rate = (self._stats["successful_requests"] / total * 100) if total > 0 else 0
        
        return {
            "total_requests": total,
            "successful_requests": self._stats["successful_requests"],
            "failed_requests": self._stats["failed_requests"],
            "success_rate": f"{success_rate:.1f}%",
            "cache_hits": self._stats["cache_hits"],
            "fallback_uses": self._stats["fallback_uses"],
            "active_data_sources": sum(1 for c in self._data_sources.values() if c.is_active),
            "total_data_sources": len(self._data_sources),
            "cache_size": len(self._market_data_cache)
        }
    
    def clear_cache(self):
        """清除缓存"""
        self._market_data_cache.clear()
        self._cache_timestamps.clear()
        logger.info("数据缓存已清除")
    
    async def health_check(self) -> Dict[str, bool]:
        """
        健康检查
        
        返回:
            Dict[str, bool]: 各数据源健康状态
        """
        health_status = {}
        
        for name, config in self._data_sources.items():
            try:
                # 简单的健康检查 - 尝试获取状态
                status = config.adapter.get_status()
                health_status[name] = status.is_available and config.is_active
            except Exception:
                health_status[name] = False
        
        return health_status
    
    def _get_sorted_data_sources(
        self,
        market: MarketType
    ) -> List[Tuple[str, DataSourceConfig]]:
        """
        获取按优先级排序的数据源
        
        参数:
            market: 市场类型
            
        返回:
            List[Tuple[str, DataSourceConfig]]: 排序后的数据源列表
        """
        # 过滤支持该市场且活跃的数据源
        applicable_sources = [
            (name, config)
            for name, config in self._data_sources.items()
            if market in config.markets and config.is_active
        ]
        
        # 按优先级排序
        return sorted(
            applicable_sources,
            key=lambda x: x[1].priority.value
        )
    
    async def _get_quote_with_fallback(
        self,
        symbol: str,
        market: MarketType
    ) -> Optional[RealtimeQuote]:
        """获取行情（带降级）"""
        sorted_sources = self._get_sorted_data_sources(market)
        
        for source_name, config in sorted_sources:
            try:
                quote = await config.adapter.get_realtime_quote(symbol, market)
                if quote:
                    return quote
            except Exception:
                continue
        
        return None
    
    def _check_data_quality(
        self,
        data: pd.DataFrame,
        symbol: str,
        market: MarketType
    ) -> DataQualityReport:
        """
        检查数据质量
        
        参数:
            data: 数据DataFrame
            symbol: 股票代码
            market: 市场类型
            
        返回:
            DataQualityReport: 质量报告
        """
        issues = []
        
        # 检查完整度
        if data.empty:
            completeness = 0.0
            issues.append("数据为空")
        else:
            # 检查关键字段
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                completeness = 1.0 - (len(missing_columns) / len(required_columns))
                issues.append(f"缺少字段: {missing_columns}")
            else:
                # 检查数据缺失
                null_counts = data[required_columns].isnull().sum()
                total_cells = len(data) * len(required_columns)
                null_cells = null_counts.sum()
                completeness = 1.0 - (null_cells / total_cells) if total_cells > 0 else 0.0
                
                if null_cells > 0:
                    issues.append(f"存在 {null_cells} 个空值")
        
        # 检查及时性（简化处理）
        timeliness = 1.0  # 假设数据是及时的
        
        # 检查准确性（简化处理）
        accuracy = 1.0
        if not data.empty and 'close' in data.columns:
            # 检查价格是否为正
            if (data['close'] <= 0).any():
                accuracy -= 0.2
                issues.append("存在非正价格")
            
            # 检查high >= low
            if 'high' in data.columns and 'low' in data.columns:
                if (data['high'] < data['low']).any():
                    accuracy -= 0.3
                    issues.append("存在high < low的数据")
        
        # 计算综合评分
        overall = (completeness * 0.4 + timeliness * 0.3 + accuracy * 0.3)
        
        return DataQualityReport(
            symbol=symbol,
            market=market,
            completeness=completeness,
            timeliness=timeliness,
            accuracy_score=accuracy,
            source_reliability=0.9,  # 简化处理
            overall_score=overall,
            issues=issues
        )


# 便捷函数
async def create_default_international_data_manager() -> EnhancedInternationalDataManager:
    """
    创建默认的国际数据管理器
    
    返回:
        EnhancedInternationalDataManager: 配置好的数据管理器
    """
    manager = EnhancedInternationalDataManager()
    
    # 注册Yahoo Finance（主数据源）
    await manager.register_data_source(
        YahooFinanceAdapter(),
        priority=DataSourcePriority.PRIMARY,
        markets=[
            MarketType.US_STOCK,
            MarketType.HK_STOCK,
            MarketType.JP_STOCK,
            MarketType.UK_STOCK
        ],
        max_requests_per_minute=200
    )
    
    # 注册Alpha Vantage（备用数据源）
    # 注意：需要API密钥
    # await manager.register_data_source(
    #     AlphaVantageAdapter(api_key="YOUR_API_KEY"),
    #     priority=DataSourcePriority.SECONDARY,
    #     markets=[MarketType.US_STOCK],
    #     max_requests_per_minute=5
    # )
    
    return manager
