#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
流处理模型质量测试
测试覆盖 stream_models.py 的所有数据模型
"""

import pytest
from datetime import datetime
from typing import Dict, Any

from src.streaming.core.stream_models import (
    StreamEvent, StreamEventType, ProcessingStatus,
    MarketDataEvent, OrderEvent, TradeEvent
)


class TestStreamEvent:
    """StreamEvent测试类"""

    def test_stream_event_creation(self):
        """测试创建StreamEvent"""
        event = StreamEvent(
            event_id="test_001",
            event_type=StreamEventType.MARKET_DATA,
            timestamp=datetime.now(),
            source="test_source",
            data={"key": "value"}
        )
        assert event.event_id == "test_001"
        assert event.event_type == StreamEventType.MARKET_DATA
        assert event.source == "test_source"
        assert event.data == {"key": "value"}

    def test_stream_event_with_metadata(self):
        """测试带元数据的StreamEvent"""
        event = StreamEvent(
            event_id="test_002",
            event_type=StreamEventType.ORDER_UPDATE,
            timestamp=datetime.now(),
            source="test_source",
            data={"key": "value"},
            metadata={"meta": "data"},
            correlation_id="corr_001"
        )
        assert event.metadata == {"meta": "data"}
        assert event.correlation_id == "corr_001"

    def test_stream_event_to_dict(self):
        """测试转换为字典"""
        event = StreamEvent(
            event_id="test_003",
            event_type=StreamEventType.MARKET_DATA,
            timestamp=datetime.now(),
            source="test_source",
            data={"key": "value"}
        )
        result = event.to_dict()
        assert result['event_id'] == "test_003"
        assert result['event_type'] == "market_data"
        assert result['source'] == "test_source"
        assert result['data'] == {"key": "value"}

    def test_stream_event_from_dict(self):
        """测试从字典创建"""
        data = {
            'event_id': 'test_004',
            'event_type': 'market_data',
            'timestamp': datetime.now().isoformat(),
            'source': 'test_source',
            'data': {'key': 'value'},
            'metadata': {'meta': 'data'},
            'correlation_id': 'corr_002',
            'version': '1.0'
        }
        event = StreamEvent.from_dict(data)
        assert event.event_id == 'test_004'
        assert event.event_type == StreamEventType.MARKET_DATA
        assert event.metadata == {'meta': 'data'}
        assert event.correlation_id == 'corr_002'

    def test_stream_event_to_json(self):
        """测试序列化为JSON"""
        event = StreamEvent(
            event_id="test_005",
            event_type=StreamEventType.MARKET_DATA,
            timestamp=datetime.now(),
            source="test_source",
            data={"key": "value"}
        )
        json_str = event.to_json()
        assert isinstance(json_str, str)
        assert "test_005" in json_str

    def test_stream_event_from_json(self):
        """测试从JSON反序列化"""
        event = StreamEvent(
            event_id="test_006",
            event_type=StreamEventType.MARKET_DATA,
            timestamp=datetime.now(),
            source="test_source",
            data={"key": "value"}
        )
        json_str = event.to_json()
        new_event = StreamEvent.from_json(json_str)
        assert new_event.event_id == "test_006"
        assert new_event.event_type == StreamEventType.MARKET_DATA


class TestMarketDataEvent:
    """MarketDataEvent测试类"""

    def test_market_data_event_creation(self):
        """测试创建MarketDataEvent"""
        event = MarketDataEvent(
            event_id="md_001",
            symbol="AAPL",
            price=150.0,
            volume=1000,
            timestamp=datetime.now()
        )
        assert event.symbol == "AAPL"
        assert event.price == 150.0
        assert event.volume == 1000
        assert event.event_type == StreamEventType.MARKET_DATA
        assert event.source == "market_data_feed"

    def test_market_data_event_with_kwargs(self):
        """测试带额外参数的MarketDataEvent"""
        event = MarketDataEvent(
            event_id="md_002",
            symbol="AAPL",
            price=150.0,
            volume=1000,
            timestamp=datetime.now(),
            metadata={"exchange": "NASDAQ"},
            correlation_id="corr_003"
        )
        assert event.metadata == {"exchange": "NASDAQ"}
        assert event.correlation_id == "corr_003"


class TestOrderEvent:
    """OrderEvent测试类"""

    def test_order_event_creation(self):
        """测试创建OrderEvent"""
        event = OrderEvent(
            event_id="order_001",
            order_id="ord_123",
            symbol="AAPL",
            order_type="buy",
            quantity=100,
            price=150.0,
            timestamp=datetime.now()
        )
        assert event.order_id == "ord_123"
        assert event.symbol == "AAPL"
        assert event.order_type == "buy"
        assert event.quantity == 100
        assert event.price == 150.0
        assert event.event_type == StreamEventType.ORDER_UPDATE
        assert event.source == "order_management"

    def test_order_event_with_kwargs(self):
        """测试带额外参数的OrderEvent"""
        event = OrderEvent(
            event_id="order_002",
            order_id="ord_124",
            symbol="AAPL",
            order_type="sell",
            quantity=50,
            price=151.0,
            timestamp=datetime.now(),
            metadata={"strategy": "momentum"},
            correlation_id="corr_004"
        )
        assert event.metadata == {"strategy": "momentum"}
        assert event.correlation_id == "corr_004"


class TestTradeEvent:
    """TradeEvent测试类"""

    def test_trade_event_creation(self):
        """测试创建TradeEvent"""
        event = TradeEvent(
            event_id="trade_001",
            trade_id="trd_123",
            order_id="ord_123",
            symbol="AAPL",
            quantity=100,
            price=150.0,
            timestamp=datetime.now()
        )
        assert event.trade_id == "trd_123"
        assert event.order_id == "ord_123"
        assert event.symbol == "AAPL"
        assert event.quantity == 100
        assert event.price == 150.0
        assert event.event_type == StreamEventType.TRADE_EXECUTION
        assert event.source == "trade_execution"

    def test_trade_event_with_kwargs(self):
        """测试带额外参数的TradeEvent"""
        event = TradeEvent(
            event_id="trade_002",
            trade_id="trd_124",
            order_id="ord_124",
            symbol="AAPL",
            quantity=50,
            price=151.0,
            timestamp=datetime.now(),
            metadata={"venue": "NYSE"},
            correlation_id="corr_005"
        )
        assert event.metadata == {"venue": "NYSE"}
        assert event.correlation_id == "corr_005"


class TestStreamEventType:
    """StreamEventType测试类"""

    def test_all_event_types(self):
        """测试所有事件类型"""
        assert StreamEventType.MARKET_DATA.value == "market_data"
        assert StreamEventType.ORDER_UPDATE.value == "order_update"
        assert StreamEventType.TRADE_EXECUTION.value == "trade_execution"
        assert StreamEventType.RISK_SIGNAL.value == "risk_signal"
        assert StreamEventType.STRATEGY_SIGNAL.value == "strategy_signal"
        assert StreamEventType.SYSTEM_METRICS.value == "system_metrics"


class TestProcessingStatus:
    """ProcessingStatus测试类"""

    def test_all_processing_statuses(self):
        """测试所有处理状态"""
        assert ProcessingStatus.PENDING.value == "pending"
        assert ProcessingStatus.PROCESSING.value == "processing"
        assert ProcessingStatus.COMPLETED.value == "completed"
        assert ProcessingStatus.FAILED.value == "failed"
        assert ProcessingStatus.DROPPED.value == "dropped"


class TestStreamProcessingResult:
    """StreamProcessingResult测试类"""

    def test_stream_processing_result_creation(self):
        """测试创建StreamProcessingResult"""
        from src.streaming.core.stream_models import StreamProcessingResult
        
        result = StreamProcessingResult(
            event_id="event_001",
            processing_status=ProcessingStatus.COMPLETED,
            processed_data={"result": "success"},
            processing_time_ms=10.5
        )
        assert result.event_id == "event_001"
        assert result.processing_status == ProcessingStatus.COMPLETED
        assert result.processed_data == {"result": "success"}
        assert result.processing_time_ms == 10.5

    def test_stream_processing_result_with_error(self):
        """测试带错误的StreamProcessingResult"""
        from src.streaming.core.stream_models import StreamProcessingResult
        
        result = StreamProcessingResult(
            event_id="event_002",
            processing_status=ProcessingStatus.FAILED,
            processed_data={},
            processing_time_ms=5.0,
            error_message="Processing failed"
        )
        assert result.processing_status == ProcessingStatus.FAILED
        assert result.error_message == "Processing failed"

    def test_stream_processing_result_to_dict(self):
        """测试转换为字典"""
        from src.streaming.core.stream_models import StreamProcessingResult
        
        result = StreamProcessingResult(
            event_id="event_003",
            processing_status=ProcessingStatus.COMPLETED,
            processed_data={"result": "success"},
            processing_time_ms=10.5
        )
        result_dict = result.to_dict()
        assert result_dict['event_id'] == "event_003"
        assert result_dict['processing_status'] == "completed"
        assert result_dict['processed_data'] == {"result": "success"}


class TestWindowedData:
    """WindowedData测试类"""

    def test_windowed_data_creation(self):
        """测试创建WindowedData"""
        from src.streaming.core.stream_models import WindowedData
        
        window = WindowedData(
            window_id="win_001",
            window_start=datetime.now(),
            window_end=datetime.now()
        )
        assert window.window_id == "win_001"
        assert len(window.data_points) == 0
        assert len(window.aggregations) == 0

    def test_windowed_data_add_data_point(self):
        """测试添加数据点"""
        from src.streaming.core.stream_models import WindowedData
        
        window = WindowedData(
            window_id="win_002",
            window_start=datetime.now(),
            window_end=datetime.now()
        )
        window.add_data_point({"value": 100})
        assert len(window.data_points) == 1
        assert window.data_points[0] == {"value": 100}

    def test_windowed_data_update_aggregation(self):
        """测试更新聚合值"""
        from src.streaming.core.stream_models import WindowedData
        
        window = WindowedData(
            window_id="win_003",
            window_start=datetime.now(),
            window_end=datetime.now()
        )
        window.update_aggregation("sum", 1000)
        assert window.get_aggregation("sum") == 1000

    def test_windowed_data_get_aggregation(self):
        """测试获取聚合值"""
        from src.streaming.core.stream_models import WindowedData
        
        window = WindowedData(
            window_id="win_004",
            window_start=datetime.now(),
            window_end=datetime.now()
        )
        assert window.get_aggregation("nonexistent") is None
        window.update_aggregation("count", 10)
        assert window.get_aggregation("count") == 10

    def test_windowed_data_is_expired(self):
        """测试检查窗口是否过期"""
        from src.streaming.core.stream_models import WindowedData
        from datetime import timedelta
        
        now = datetime.now()
        window = WindowedData(
            window_id="win_005",
            window_start=now - timedelta(seconds=10),
            window_end=now - timedelta(seconds=5)
        )
        assert window.is_expired(now) is True
        
        window2 = WindowedData(
            window_id="win_006",
            window_start=now,
            window_end=now + timedelta(seconds=10)
        )
        assert window2.is_expired(now) is False


class TestStreamState:
    """StreamState测试类"""

    def test_stream_state_creation(self):
        """测试创建StreamState"""
        from src.streaming.core.stream_models import StreamState
        
        state = StreamState(state_id="state_001")
        assert state.state_id == "state_001"
        assert len(state.state_data) == 0
        assert state.version == 1

    def test_stream_state_update(self):
        """测试更新状态"""
        from src.streaming.core.stream_models import StreamState
        
        state = StreamState(state_id="state_002")
        state.update({"key1": "value1", "key2": "value2"})
        assert state.state_data["key1"] == "value1"
        assert state.state_data["key2"] == "value2"
        assert state.version == 2

    def test_stream_state_get_set(self):
        """测试获取和设置状态值"""
        from src.streaming.core.stream_models import StreamState
        
        state = StreamState(state_id="state_003")
        state.set("key", "value")
        assert state.get("key") == "value"
        assert state.get("nonexistent", "default") == "default"
        assert state.version == 2  # set方法会增加version

    def test_stream_state_is_expired(self):
        """测试检查状态是否过期"""
        from src.streaming.core.stream_models import StreamState
        from datetime import timedelta
        
        state = StreamState(state_id="state_004", ttl_seconds=1)
        assert state.is_expired() is False
        
        # 模拟过期
        state.last_updated = datetime.now() - timedelta(seconds=2)
        assert state.is_expired() is True


class TestStreamMetrics:
    """StreamMetrics测试类"""

    def test_stream_metrics_creation(self):
        """测试创建StreamMetrics"""
        from src.streaming.core.stream_models import StreamMetrics
        
        metrics = StreamMetrics(processor_id="proc_001")
        assert metrics.processor_id == "proc_001"
        assert metrics.events_processed == 0
        assert metrics.events_failed == 0

    def test_stream_metrics_update_success(self):
        """测试更新指标（成功）"""
        from src.streaming.core.stream_models import StreamMetrics
        
        metrics = StreamMetrics(processor_id="proc_002")
        metrics.update_metrics(processing_time=10.0, success=True)
        assert metrics.events_processed == 1
        assert metrics.events_failed == 0
        assert metrics.processing_time_avg == 10.0

    def test_stream_metrics_update_failure(self):
        """测试更新指标（失败）"""
        from src.streaming.core.stream_models import StreamMetrics
        
        metrics = StreamMetrics(processor_id="proc_003")
        metrics.update_metrics(processing_time=5.0, success=False)
        assert metrics.events_processed == 1
        assert metrics.events_failed == 1
        assert metrics.error_rate == 1.0

    def test_stream_metrics_update_average(self):
        """测试更新平均处理时间"""
        from src.streaming.core.stream_models import StreamMetrics
        
        metrics = StreamMetrics(processor_id="proc_004")
        metrics.update_metrics(processing_time=10.0, success=True)
        metrics.update_metrics(processing_time=20.0, success=True)
        assert metrics.processing_time_avg == 15.0

    def test_stream_metrics_to_dict(self):
        """测试转换为字典"""
        from src.streaming.core.stream_models import StreamMetrics
        
        metrics = StreamMetrics(processor_id="proc_005")
        metrics.update_metrics(processing_time=10.0, success=True)
        metrics_dict = metrics.to_dict()
        assert metrics_dict['processor_id'] == "proc_005"
        assert metrics_dict['events_processed'] == 1
        assert 'last_updated' in metrics_dict


class TestEventFactoryFunctions:
    """事件工厂函数测试类"""

    def test_create_market_data_event(self):
        """测试创建市场数据事件"""
        from src.streaming.core.stream_models import create_market_data_event
        
        event = create_market_data_event("AAPL", 150.0, 1000)
        assert event.symbol == "AAPL"
        assert event.price == 150.0
        assert event.volume == 1000

    def test_create_order_event(self):
        """测试创建订单事件"""
        from src.streaming.core.stream_models import create_order_event
        
        event = create_order_event("ord_123", "AAPL", "buy", 100, 150.0)
        assert event.order_id == "ord_123"
        assert event.symbol == "AAPL"
        assert event.order_type == "buy"
        assert event.quantity == 100

    def test_create_trade_event(self):
        """测试创建交易事件"""
        from src.streaming.core.stream_models import create_trade_event
        
        event = create_trade_event("trd_123", "ord_123", "AAPL", 100, 150.0)
        assert event.trade_id == "trd_123"
        assert event.order_id == "ord_123"
        assert event.symbol == "AAPL"
        assert event.quantity == 100

