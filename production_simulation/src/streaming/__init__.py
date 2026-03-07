#!/usr/bin/env python3
"""
RQA2025 流处理层
Streaming Processing Layer

提供高性能的实时数据流处理能力。
"""

from .core.stream_models import (
    StreamEvent, StreamEventType, MarketDataEvent, OrderEvent,
    TradeEvent, create_market_data_event, create_order_event,
    create_trade_event
)
from .core.base_processor import (
    StreamProcessorBase, StreamProcessingResult, ProcessingStatus, StreamMetrics
)
from .core.aggregator import (
    RealTimeAggregator, WindowedData
)
from .core.state_manager import (
    StateManager, StreamState
)
from .core.data_pipeline import (
    DataPipeline, PipelineRule, PipelineStage, PipelineMetrics
)
from .core.stream_engine import (
    StreamProcessingEngine, StreamTopology, create_stream_engine
)

__version__ = "1.0.0"
__author__ = "RQA2025 Team"

__all__ = [
    # 流事件模型
    'StreamEvent', 'StreamEventType', 'MarketDataEvent', 'OrderUpdateEvent',
    'StrategySignalEvent', 'RiskAlertEvent', 'create_stream_event',
    'create_market_data_event',

    # 基础处理器
    'StreamProcessorBase', 'StreamProcessingResult', 'ProcessingStatus', 'StreamMetrics',

    # 聚合器
    'RealTimeAggregator', 'WindowedData',

    # 状态管理器
    'StateManager', 'StreamState',

    # 数据管道
    'DataPipeline', 'PipelineRule', 'PipelineStage', 'PipelineMetrics',

    # 流处理引擎
    'StreamProcessingEngine', 'StreamTopology', 'create_stream_engine'
]
