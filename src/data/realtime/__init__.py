"""
实时数据处理引擎
包含以下核心组件：
- engine: 实时引擎主模块
- decoder: 行情解码器
- dispatcher: 数据分发器
- buffer: 零拷贝缓冲区
"""
from .engine import RealTimeEngine
from .decoder import MarketDataDecoder
from .dispatcher import DataDispatcher
from .buffer import ZeroCopyBuffer

__all__ = [
    'RealTimeEngine',
    'MarketDataDecoder',
    'DataDispatcher',
    'ZeroCopyBuffer'
]
