# -*- coding: utf-8 -*-
"""
RQA2025 流处理层核心数据模型
Stream Processing Layer Core Data Models

定义流处理层使用的数据模型和事件结构。
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json


class StreamEventType(Enum):

    """流事件类型"""
    MARKET_DATA = "market_data"
    ORDER_UPDATE = "order_update"
    TRADE_EXECUTION = "trade_execution"
    RISK_SIGNAL = "risk_signal"
    STRATEGY_SIGNAL = "strategy_signal"
    SYSTEM_METRICS = "system_metrics"


class ProcessingStatus(Enum):

    """处理状态"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DROPPED = "dropped"


@dataclass
class StreamEvent:

    """
    流事件基础类
    """
    event_id: str
    event_type: StreamEventType
    timestamp: datetime
    source: str
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'data': self.data,
            'metadata': self.metadata,
            'correlation_id': self.correlation_id,
            'version': self.version
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StreamEvent':
        """从字典创建事件"""
        return cls(
            event_id=data['event_id'],
            event_type=StreamEventType(data['event_type']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            source=data['source'],
            data=data['data'],
            metadata=data.get('metadata', {}),
            correlation_id=data.get('correlation_id'),
            version=data.get('version', '1.0')
        )

    def to_json(self) -> str:
        """序列化为JSON字符串"""
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_json(cls, json_str: str) -> 'StreamEvent':
        """从JSON字符串反序列化"""
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class MarketDataEvent(StreamEvent):

    """
    市场数据事件
    """

    def __init__(self, event_id: str, symbol: str, price: float,
                 volume: int, timestamp: datetime, **kwargs):
        super().__init__(
            event_id=event_id,
            event_type=StreamEventType.MARKET_DATA,
            timestamp=timestamp,
            source="market_data_feed",
            data={
                'symbol': symbol,
                'price': price,
                'volume': volume,
                'timestamp': timestamp.isoformat()
            },
            **kwargs
        )

        # 添加市场数据特有的字段
        self.symbol = symbol
        self.price = price
        self.volume = volume


@dataclass
class OrderEvent(StreamEvent):

    """
    订单事件
    """

    def __init__(self, event_id: str, order_id: str, symbol: str,
                 order_type: str, quantity: int, price: float,
                 timestamp: datetime, **kwargs):
        super().__init__(
            event_id=event_id,
            event_type=StreamEventType.ORDER_UPDATE,
            timestamp=timestamp,
            source="order_management",
            data={
                'order_id': order_id,
                'symbol': symbol,
                'order_type': order_type,
                'quantity': quantity,
                'price': price,
                'timestamp': timestamp.isoformat()
            },
            **kwargs
        )

        # 添加订单特有的字段
        self.order_id = order_id
        self.symbol = symbol
        self.order_type = order_type
        self.quantity = quantity
        self.price = price


@dataclass
class TradeEvent(StreamEvent):

    """
    交易执行事件
    """

    def __init__(self, event_id: str, trade_id: str, order_id: str,
                 symbol: str, quantity: int, price: float,
                 timestamp: datetime, **kwargs):
        super().__init__(
            event_id=event_id,
            event_type=StreamEventType.TRADE_EXECUTION,
            timestamp=timestamp,
            source="trade_execution",
            data={
                'trade_id': trade_id,
                'order_id': order_id,
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'timestamp': timestamp.isoformat()
            },
            **kwargs
        )

        # 添加交易特有的字段
        self.trade_id = trade_id
        self.order_id = order_id
        self.symbol = symbol
        self.quantity = quantity
        self.price = price


@dataclass
class StreamProcessingResult:

    """
    流处理结果
    """
    event_id: str
    processing_status: ProcessingStatus
    processed_data: Dict[str, Any]
    processing_time_ms: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'event_id': self.event_id,
            'processing_status': self.processing_status.value,
            'processed_data': self.processed_data,
            'processing_time_ms': self.processing_time_ms,
            'error_message': self.error_message,
            'metadata': self.metadata
        }


@dataclass
class WindowedData:

    """
    窗口数据结构
    """
    window_id: str
    window_start: datetime
    window_end: datetime
    data_points: List[Dict[str, Any]] = field(default_factory=list)
    aggregations: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_data_point(self, data: Dict[str, Any]):
        """添加数据点"""
        self.data_points.append(data)

    def update_aggregation(self, key: str, value: Any):
        """更新聚合值"""
        self.aggregations[key] = value

    def get_aggregation(self, key: str) -> Any:
        """获取聚合值"""
        return self.aggregations.get(key)

    def is_expired(self, current_time: datetime) -> bool:
        """检查窗口是否过期"""
        return current_time > self.window_end


@dataclass
class StreamState:

    """
    流状态数据结构
    """
    state_id: str
    state_data: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)
    ttl_seconds: int = 3600  # 1小时TTL
    version: int = 1

    def update(self, updates: Dict[str, Any]):
        """更新状态"""
        self.state_data.update(updates)
        self.last_updated = datetime.now()
        self.version += 1

    def get(self, key: str, default: Any = None) -> Any:
        """获取状态值"""
        return self.state_data.get(key, default)

    def set(self, key: str, value: Any):
        """设置状态值"""
        self.state_data[key] = value
        self.last_updated = datetime.now()
        self.version += 1

    def is_expired(self) -> bool:
        """检查状态是否过期"""
        return (datetime.now() - self.last_updated).total_seconds() > self.ttl_seconds


@dataclass
class StreamMetrics:

    """
    流处理指标
    """
    processor_id: str
    events_processed: int = 0
    events_failed: int = 0
    processing_time_avg: float = 0.0
    throughput_per_second: float = 0.0
    queue_size: int = 0
    error_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

    def update_metrics(self, processing_time: float, success: bool):
        """更新指标"""
        self.events_processed += 1
        if not success:
            self.events_failed += 1

        # 更新平均处理时间
        if self.events_processed == 1:
            self.processing_time_avg = processing_time
        else:
            self.processing_time_avg = (
                (self.processing_time_avg * (self.events_processed - 1))
                + processing_time
            ) / self.events_processed

        # 更新错误率
        self.error_rate = self.events_failed / self.events_processed

        # 更新吞吐量（简化的计算）
        time_window = 60  # 60秒窗口
        self.throughput_per_second = self.events_processed / time_window

        self.last_updated = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'processor_id': self.processor_id,
            'events_processed': self.events_processed,
            'events_failed': self.events_failed,
            'processing_time_avg': self.processing_time_avg,
            'throughput_per_second': self.throughput_per_second,
            'queue_size': self.queue_size,
            'error_rate': self.error_rate,
            'last_updated': self.last_updated.isoformat()
        }


# 事件工厂函数

def create_market_data_event(symbol: str, price: float, volume: int,


                             timestamp: Optional[datetime] = None) -> MarketDataEvent:
    """创建市场数据事件"""
    event_id = f"market_{symbol}_{int(timestamp.timestamp()) if timestamp else int(datetime.now().timestamp())}"
    return MarketDataEvent(
        event_id=event_id,
        symbol=symbol,
        price=price,
        volume=volume,
        timestamp=timestamp or datetime.now()
    )


def create_order_event(order_id: str, symbol: str, order_type: str,


                       quantity: int, price: float,
                       timestamp: Optional[datetime] = None) -> OrderEvent:
    """创建订单事件"""
    event_id = f"order_{order_id}_{int(timestamp.timestamp()) if timestamp else int(datetime.now().timestamp())}"
    return OrderEvent(
        event_id=event_id,
        order_id=order_id,
        symbol=symbol,
        order_type=order_type,
        quantity=quantity,
        price=price,
        timestamp=timestamp or datetime.now()
    )


def create_trade_event(trade_id: str, order_id: str, symbol: str,


                       quantity: int, price: float,
                       timestamp: Optional[datetime] = None) -> TradeEvent:
    """创建交易事件"""
    event_id = f"trade_{trade_id}_{int(timestamp.timestamp()) if timestamp else int(datetime.now().timestamp())}"
    return TradeEvent(
        event_id=event_id,
        trade_id=trade_id,
        order_id=order_id,
        symbol=symbol,
        quantity=quantity,
        price=price,
        timestamp=timestamp or datetime.now()
    )


__all__ = [
    'StreamEventType',
    'ProcessingStatus',
    'StreamEvent',
    'MarketDataEvent',
    'OrderEvent',
    'TradeEvent',
    'StreamProcessingResult',
    'WindowedData',
    'StreamState',
    'StreamMetrics',
    'create_market_data_event',
    'create_order_event',
    'create_trade_event'
]
