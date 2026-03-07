#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据模型基类定义
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json
import pandas as pd


class DataModel(ABC):

    """数据模型基类"""

    def __init__(self, raw_data=None, metadata=None, validation_status=False, **kwargs):
        """初始化数据模型

        Args:
            raw_data: 原始数据
            metadata: 元数据
            validation_status: 验证状态
            **kwargs: 其他参数
        """
        self.raw_data = raw_data
        self.metadata = metadata if metadata is not None else {}
        self.validation_status = validation_status
        self.data = None
        self.frequency = None
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

        # 处理其他参数
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # 如果raw_data是DataFrame，直接赋值给data
        if isinstance(raw_data, pd.DataFrame):
            self.data = raw_data
            # 允许空DataFrame通过验证（用于测试和初始化）
            self.validation_status = True
        elif raw_data is not None:
            # 尝试将raw_data转换为DataFrame
            try:
                if isinstance(raw_data, dict):
                    self.data = pd.DataFrame([raw_data])
                    self.validation_status = True
                elif isinstance(raw_data, list):
                    self.data = pd.DataFrame(raw_data)
                    self.validation_status = True
                else:
                    self.data = pd.DataFrame()
                    self.validation_status = False
            except Exception:
                self.data = pd.DataFrame()
                self.validation_status = False
        else:
            self.data = pd.DataFrame()
            self.validation_status = False  # 无数据时设置为False

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""

    @abstractmethod
    def from_dict(self, data: Dict[str, Any]) -> 'DataModel':
        """从字典创建实例"""

    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> 'DataModel':
        """从JSON字符串创建实例"""
        data = json.loads(json_str)
        return cls().from_dict(data)

    def set_metadata(self, metadata: Dict[str, Any]):
        """设置元数据"""
        if metadata is None:
            metadata = {}
        if self.metadata is None:
            self.metadata = {}
        self.metadata.update(metadata)
        self.updated_at = datetime.now()

    def get_metadata(self) -> Dict[str, Any]:
        """获取元数据"""
        return self.metadata.copy()

    def get_data(self) -> pd.DataFrame:
        """获取数据"""
        return self.data.copy() if self.data is not None else pd.DataFrame()

    def get_frequency(self) -> Optional[str]:
        """获取频率"""
        return self.frequency

    def validate(self) -> bool:
        """验证数据"""
        # 如果明确设置了validation_status，使用它
        if hasattr(self, 'validation_status') and self.validation_status is not None:
            return self.validation_status

        # 否则根据数据状态判断
        if self.data is not None and hasattr(self.data, 'empty'):
            return not self.data.empty
        elif self.raw_data is not None:
            return True
        else:
            return False

    def __repr__(self):
        """字符串表示"""
        return f"DataModel(data_shape={self.data.shape if self.data is not None else (0, 0)}, validation_status={self.validation_status})"


class SimpleDataModel(DataModel):

    """简单的数据模型实现，用于测试和占位"""

    def __init__(self, data=None, metadata=None, is_valid=True, frequency=None):

        # 支持多种参数顺序
        if isinstance(data, str) and metadata is None:
            # 如果第一个参数是字符串，可能是频率
            frequency = data
            data = None
        elif isinstance(metadata, str) and frequency is None:
            # 如果第二个参数是字符串，可能是频率
            frequency = metadata
            metadata = None

        # 确保metadata是字典
        if metadata is None:
            metadata = {}
        elif not isinstance(metadata, dict):
            metadata = {}

        # 调用父类构造函数
        super().__init__(raw_data=data, metadata=metadata, validation_status=is_valid)

        # 设置子类特有的属性
        self.is_valid = is_valid
        self._frequency = frequency  # 不设置默认值，保持为None

        # 确保metadata被正确设置（修复位置参数问题）
        if metadata and isinstance(metadata, dict):
            self.metadata = metadata.copy()

        # 不自动添加额外元数据字段，保持测试兼容性

    @property
    def frequency(self) -> Optional[str]:
        """频率属性"""
        return self._frequency

    @frequency.setter
    def frequency(self, value: Optional[str]):
        """设置频率"""
        self._frequency = value

    def get_frequency(self) -> Optional[str]:
        """获取数据频率"""
        return self._frequency

    def get_metadata(self, user_only: bool = False) -> Dict[str, Any]:
        """获取元数据信息"""
        if not isinstance(self.metadata, dict):
            return {}
        if user_only:
            # 返回用户原始元数据（不含自动补充字段）
            return {k: v for k, v in self.metadata.items()
                    if k not in ['data_shape', 'data_columns', 'created_at']}
        return self.metadata.copy()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        # 如果data为空DataFrame，返回None
        data_value = None if (self.data is None or (
            hasattr(self.data, 'empty') and self.data.empty)) else self.data
        return {
            'data': data_value,
            'frequency': self._frequency,
            'metadata': self.metadata,
            'is_valid': self.validation_status
        }

    def from_dict(self, data: Dict[str, Any]) -> 'SimpleDataModel':
        """从字典创建实例"""
        self.data = data.get('data')
        self._frequency = data.get('frequency')  # 不设置默认值
        self.metadata = data.get('metadata', {})
        self.is_valid = data.get('is_valid', True)
        return self


class DataType(Enum):

    """数据类型枚举"""
    STOCK = "stock"
    INDEX = "index"
    FUND = "fund"
    BOND = "bond"
    FUTURES = "futures"
    OPTIONS = "options"
    FOREX = "forex"
    CRYPTO = "crypto"
    NEWS = "news"
    SENTIMENT = "sentiment"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"


@dataclass
class MarketData(DataModel):

    """市场数据模型"""
    symbol: str
    timestamp: datetime
    data_type: DataType
    price: Optional[float] = None
    volume: Optional[float] = None
    open_price: Optional[float] = None
    high_price: Optional[float] = None
    low_price: Optional[float] = None
    close_price: Optional[float] = None
    change: Optional[float] = None
    change_pct: Optional[float] = None
    turnover: Optional[float] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    source: str = "unknown"
    exchange: str = "unknown"
    currency: str = "CNY"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后调用父类构造函数"""
        super().__init__(metadata=self.metadata)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'data_type': self.data_type.value,
            'price': self.price,
            'volume': self.volume,
            'open_price': self.open_price,
            'high_price': self.high_price,
            'low_price': self.low_price,
            'close_price': self.close_price,
            'change': self.change,
            'change_pct': self.change_pct,
            'turnover': self.turnover,
            'market_cap': self.market_cap,
            'pe_ratio': self.pe_ratio,
            'pb_ratio': self.pb_ratio,
            'dividend_yield': self.dividend_yield,
            'source': self.source,
            'exchange': self.exchange,
            'currency': self.currency,
            'metadata': self.metadata
        }

    def from_dict(self, data: Dict[str, Any]) -> 'MarketData':
        """从字典创建实例"""
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if 'data_type' in data and isinstance(data['data_type'], str):
            data['data_type'] = DataType(data['data_type'])

        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


@dataclass
class NewsData(DataModel):

    """新闻数据模型"""
    id: str
    title: str
    content: str
    summary: Optional[str] = None
    author: Optional[str] = None
    source: str = "unknown"
    url: Optional[str] = None
    published_at: Optional[datetime] = None
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    related_symbols: List[str] = field(default_factory=list)
    importance_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后调用父类构造函数"""
        super().__init__(metadata=self.metadata)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'id': self.id,
            'title': self.title,
            'content': self.content,
            'summary': self.summary,
            'author': self.author,
            'source': self.source,
            'url': self.url,
            'published_at': self.published_at.isoformat() if self.published_at else None,
            'sentiment_score': self.sentiment_score,
            'sentiment_label': self.sentiment_label,
            'tags': self.tags,
            'related_symbols': self.related_symbols,
            'importance_score': self.importance_score,
            'metadata': self.metadata
        }

    def from_dict(self, data: Dict[str, Any]) -> 'NewsData':
        """从字典创建实例"""
        if 'published_at' in data and isinstance(data['published_at'], str):
            data['published_at'] = datetime.fromisoformat(data['published_at'])

        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


@dataclass
class TechnicalIndicator(DataModel):

    """技术指标模型"""
    symbol: str
    timestamp: datetime
    indicator_name: str
    value: float
    parameters: Dict[str, Any] = field(default_factory=dict)
    signal: Optional[str] = None  # buy, sell, hold
    strength: Optional[float] = None  # 0 - 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后调用父类构造函数"""
        super().__init__(metadata=self.metadata)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'indicator_name': self.indicator_name,
            'value': self.value,
            'parameters': self.parameters,
            'signal': self.signal,
            'strength': self.strength,
            'metadata': self.metadata
        }

    def from_dict(self, data: Dict[str, Any]) -> 'TechnicalIndicator':
        """从字典创建实例"""
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])

        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


@dataclass
class FundamentalData(DataModel):

    """基本面数据模型"""
    symbol: str
    report_date: datetime
    data_type: str  # income, balance, cash_flow, ratios
    revenue: Optional[float] = None
    net_income: Optional[float] = None
    eps: Optional[float] = None
    total_assets: Optional[float] = None
    total_liabilities: Optional[float] = None
    equity: Optional[float] = None
    cash_flow_operations: Optional[float] = None
    cash_flow_investing: Optional[float] = None
    cash_flow_financing: Optional[float] = None
    free_cash_flow: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    gross_margin: Optional[float] = None
    net_margin: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后调用父类构造函数"""
        super().__init__(metadata=self.metadata)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'symbol': self.symbol,
            'report_date': self.report_date.isoformat() if self.report_date else None,
            'data_type': self.data_type,
            'revenue': self.revenue,
            'net_income': self.net_income,
            'eps': self.eps,
            'total_assets': self.total_assets,
            'total_liabilities': self.total_liabilities,
            'equity': self.equity,
            'cash_flow_operations': self.cash_flow_operations,
            'cash_flow_investing': self.cash_flow_investing,
            'cash_flow_financing': self.cash_flow_financing,
            'free_cash_flow': self.free_cash_flow,
            'debt_to_equity': self.debt_to_equity,
            'current_ratio': self.current_ratio,
            'quick_ratio': self.quick_ratio,
            'roe': self.roe,
            'roa': self.roa,
            'gross_margin': self.gross_margin,
            'net_margin': self.net_margin,
            'metadata': self.metadata
        }

    def from_dict(self, data: Dict[str, Any]) -> 'FundamentalData':
        """从字典创建实例"""
        if 'report_date' in data and isinstance(data['report_date'], str):
            data['report_date'] = datetime.fromisoformat(data['report_date'])

        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


class DataModelFactory:

    """数据模型工厂类"""

    @staticmethod
    def create_market_data(**kwargs) -> MarketData:
        """创建市场数据模型"""
        return MarketData(**kwargs)

    @staticmethod
    def create_news_data(**kwargs) -> NewsData:
        """创建新闻数据模型"""
        return NewsData(**kwargs)

    @staticmethod
    def create_technical_indicator(**kwargs) -> TechnicalIndicator:
        """创建技术指标模型"""
        return TechnicalIndicator(**kwargs)

    @staticmethod
    def create_fundamental_data(**kwargs) -> FundamentalData:
        """创建基本面数据模型"""
        return FundamentalData(**kwargs)

    @staticmethod
    def create_from_dict(data_type: str, data: Dict[str, Any]) -> DataModel:
        """根据数据类型从字典创建模型"""
        if data_type == "market":
            return MarketData().from_dict(data)
        elif data_type == "news":
            return NewsData().from_dict(data)
        elif data_type == "technical":
            return TechnicalIndicator().from_dict(data)
        elif data_type == "fundamental":
            return FundamentalData().from_dict(data)
        else:
            raise ValueError(f"Unknown data type: {data_type}")


# 导出主要类
__all__ = [
    'DataModel',
    'DataType',
    'MarketData',
    'NewsData',
    'TechnicalIndicator',
    'FundamentalData',
    'DataModelFactory'
]
