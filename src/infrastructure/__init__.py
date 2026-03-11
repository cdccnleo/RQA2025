"""
基础设施模块

提供存储、通知和流处理等基础设施组件
"""

# 流处理层组件（从src/streaming合并）
try:
    from .streaming.core.stream_engine import StreamEngine
    from .streaming.core.data_pipeline import DataPipeline
    from .streaming.core.event_processor import EventProcessor
    _streaming_available = True
except ImportError:
    _streaming_available = False

    class StreamEngine:
        """流引擎基础实现"""
        pass

    class DataPipeline:
        """数据管道基础实现"""
        pass

    class EventProcessor:
        """事件处理器基础实现"""
        pass

__all__ = [
    # 流处理层组件（合并后）
    "StreamEngine",
    "DataPipeline",
    "EventProcessor"
]
