#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
另类数据适配器基类

功能：
- 定义另类数据适配器标准接口
- 支持社交媒体情绪、新闻情绪等另类数据
- 统一数据格式和处理流程

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
import numpy as np

logger = logging.getLogger(__name__)


class AlternativeDataType(Enum):
    """另类数据类型"""
    SOCIAL_MEDIA_SENTIMENT = "social_media_sentiment"  # 社交媒体情绪
    NEWS_SENTIMENT = "news_sentiment"                  # 新闻情绪
    SEARCH_TREND = "search_trend"                      # 搜索趋势
    SATELLITE_DATA = "satellite_data"                  # 卫星数据
    CREDIT_CARD_DATA = "credit_card_data"             # 信用卡数据
    APP_USAGE_DATA = "app_usage_data"                  # App使用数据
    WEB_TRAFFIC_DATA = "web_traffic_data"             # 网站流量数据
    EARNINGS_CALL = "earnings_call"                   # 财报电话会议
    SEC_FILING = "sec_filing"                         # SEC文件


class SentimentPolarity(Enum):
    """情绪极性"""
    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2


@dataclass
class SentimentDataPoint:
    """情绪数据点"""
    timestamp: datetime
    symbol: str
    data_type: AlternativeDataType
    polarity: SentimentPolarity
    score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    volume: int  # 数据量（帖子数、文章数等）
    source: str
    raw_text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TrendDataPoint:
    """趋势数据点"""
    timestamp: datetime
    symbol: str
    data_type: AlternativeDataType
    value: float
    change_percent: float
    volume: int
    source: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AlternativeDataRequest:
    """另类数据请求"""
    symbol: str
    data_type: AlternativeDataType
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    granularity: str = "1d"  # 1m, 5m, 15m, 30m, 1h, 1d
    sources: Optional[List[str]] = None
    min_confidence: float = 0.5


@dataclass
class AdapterMetrics:
    """适配器指标"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_latency_ms: float = 0.0
    last_request_time: Optional[datetime] = None
    data_points_collected: int = 0


class AlternativeDataAdapter(ABC):
    """
    另类数据适配器基类
    
    所有另类数据源适配器必须继承此类
    """
    
    def __init__(
        self,
        name: str,
        data_types: List[AlternativeDataType],
        api_key: Optional[str] = None
    ):
        """
        初始化适配器
        
        Args:
            name: 适配器名称
            data_types: 支持的数据类型列表
            api_key: API密钥
        """
        self.name = name
        self.data_types = data_types
        self.api_key = api_key
        self._metrics = AdapterMetrics()
        self._is_connected = False
        
        logger.info(f"另类数据适配器 {name} 初始化完成，支持类型: {[t.value for t in data_types]}")
    
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
    async def fetch_sentiment_data(
        self,
        request: AlternativeDataRequest
    ) -> pd.DataFrame:
        """
        获取情绪数据
        
        Args:
            request: 数据请求参数
            
        Returns:
            DataFrame包含情绪数据
        """
        pass
    
    @abstractmethod
    async def fetch_trend_data(
        self,
        request: AlternativeDataRequest
    ) -> pd.DataFrame:
        """
        获取趋势数据
        
        Args:
            request: 数据请求参数
            
        Returns:
            DataFrame包含趋势数据
        """
        pass
    
    @abstractmethod
    async def get_realtime_sentiment(
        self,
        symbol: str,
        data_type: AlternativeDataType
    ) -> Optional[SentimentDataPoint]:
        """
        获取实时情绪数据
        
        Args:
            symbol: 股票代码
            data_type: 数据类型
            
        Returns:
            最新情绪数据点
        """
        pass
    
    @abstractmethod
    async def analyze_text_sentiment(
        self,
        text: str,
        symbol: Optional[str] = None
    ) -> SentimentDataPoint:
        """
        分析文本情绪
        
        Args:
            text: 待分析文本
            symbol: 相关股票代码（可选）
            
        Returns:
            情绪分析结果
        """
        pass
    
    @abstractmethod
    async def get_data_sources(
        self,
        data_type: AlternativeDataType
    ) -> List[str]:
        """
        获取数据源列表
        
        Args:
            data_type: 数据类型
            
        Returns:
            数据源名称列表
        """
        pass
    
    @abstractmethod
    async def check_health(self) -> bool:
        """
        检查适配器健康状态
        
        Returns:
            是否健康
        """
        pass
    
    def get_metrics(self) -> AdapterMetrics:
        """获取适配器指标"""
        return self._metrics
    
    def supports_data_type(self, data_type: AlternativeDataType) -> bool:
        """
        检查是否支持特定数据类型
        
        Args:
            data_type: 数据类型
            
        Returns:
            是否支持
        """
        return data_type in self.data_types
    
    def _update_metrics(self, success: bool, latency_ms: float, data_points: int = 0):
        """
        更新指标
        
        Args:
            success: 是否成功
            latency_ms: 延迟（毫秒）
            data_points: 数据点数量
        """
        self._metrics.total_requests += 1
        self._metrics.last_request_time = datetime.now()
        
        if success:
            self._metrics.successful_requests += 1
            self._metrics.data_points_collected += data_points
        else:
            self._metrics.failed_requests += 1
        
        # 更新平均延迟
        if self._metrics.total_requests == 1:
            self._metrics.average_latency_ms = latency_ms
        else:
            self._metrics.average_latency_ms = (
                (self._metrics.average_latency_ms * (self._metrics.total_requests - 1) + latency_ms)
                / self._metrics.total_requests
            )
    
    def _calculate_sentiment_score(
        self,
        positive_count: int,
        negative_count: int,
        neutral_count: int
    ) -> tuple:
        """
        计算情绪分数
        
        Args:
            positive_count: 正面数量
            negative_count: 负面数量
            neutral_count: 中性数量
            
        Returns:
            (score, polarity, confidence)
        """
        total = positive_count + negative_count + neutral_count
        
        if total == 0:
            return 0.0, SentimentPolarity.NEUTRAL, 0.0
        
        # 计算基础分数
        score = (positive_count - negative_count) / total
        
        # 确定极性
        if score > 0.6:
            polarity = SentimentPolarity.VERY_POSITIVE
        elif score > 0.2:
            polarity = SentimentPolarity.POSITIVE
        elif score < -0.6:
            polarity = SentimentPolarity.VERY_NEGATIVE
        elif score < -0.2:
            polarity = SentimentPolarity.NEGATIVE
        else:
            polarity = SentimentPolarity.NEUTRAL
        
        # 计算置信度（基于数据量）
        confidence = min(total / 100, 1.0)  # 100条数据达到最大置信度
        
        return score, polarity, confidence
    
    def _standardize_sentiment_df(
        self,
        df: pd.DataFrame,
        symbol: str,
        data_type: AlternativeDataType
    ) -> pd.DataFrame:
        """
        标准化情绪数据DataFrame
        
        Args:
            df: 原始数据
            symbol: 股票代码
            data_type: 数据类型
            
        Returns:
            标准格式DataFrame
        """
        # 确保必要的列存在
        required_columns = ['timestamp', 'score', 'volume']
        
        for col in required_columns:
            if col not in df.columns:
                logger.warning(f"缺失必要列: {col}")
        
        # 添加元数据
        df['symbol'] = symbol
        df['data_type'] = data_type.value
        df['adapter'] = self.name
        
        return df


class DataFusionEngine:
    """
    数据融合引擎
    
    融合多个另类数据源的数据
    """
    
    def __init__(self):
        """初始化数据融合引擎"""
        self.adapters: List[AlternativeDataAdapter] = []
        
    def register_adapter(self, adapter: AlternativeDataAdapter):
        """
        注册适配器
        
        Args:
            adapter: 另类数据适配器
        """
        self.adapters.append(adapter)
        logger.info(f"注册另类数据适配器: {adapter.name}")
    
    async def fetch_fused_sentiment_data(
        self,
        symbol: str,
        data_type: AlternativeDataType,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        获取融合后的情绪数据
        
        Args:
            symbol: 股票代码
            data_type: 数据类型
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            融合后的情绪数据
        """
        request = AlternativeDataRequest(
            symbol=symbol,
            data_type=data_type,
            start_date=start_date,
            end_date=end_date
        )
        
        # 从所有适配器获取数据
        all_data = []
        for adapter in self.adapters:
            if adapter.supports_data_type(data_type):
                try:
                    df = await adapter.fetch_sentiment_data(request)
                    if not df.empty:
                        all_data.append(df)
                except Exception as e:
                    logger.warning(f"从 {adapter.name} 获取数据失败: {e}")
        
        if not all_data:
            return pd.DataFrame()
        
        # 融合数据
        fused_df = self._fuse_sentiment_data(all_data)
        
        return fused_df
    
    def _fuse_sentiment_data(
        self,
        dataframes: List[pd.DataFrame]
    ) -> pd.DataFrame:
        """
        融合情绪数据
        
        Args:
            dataframes: 多个数据源的DataFrame
            
        Returns:
            融合后的DataFrame
        """
        # 合并所有数据
        combined = pd.concat(dataframes, ignore_index=True)
        
        # 按时间戳分组
        combined['timestamp'] = pd.to_datetime(combined['timestamp'])
        combined = combined.sort_values('timestamp')
        
        # 计算加权平均分数（基于置信度）
        if 'confidence' in combined.columns:
            combined['weighted_score'] = combined['score'] * combined['confidence']
            
            fused = combined.groupby('timestamp').apply(
                lambda x: pd.Series({
                    'score': x['weighted_score'].sum() / x['confidence'].sum(),
                    'confidence': x['confidence'].mean(),
                    'volume': x['volume'].sum(),
                    'sources': x['adapter'].nunique()
                })
            ).reset_index()
        else:
            # 简单平均
            fused = combined.groupby('timestamp').agg({
                'score': 'mean',
                'volume': 'sum',
                'adapter': 'nunique'
            }).reset_index()
            fused = fused.rename(columns={'adapter': 'sources'})
            fused['confidence'] = 0.5  # 默认置信度
        
        return fused


class AlternativeDataError(Exception):
    """另类数据错误"""
    pass
