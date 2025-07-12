"""RQA2025实时引擎 - 高性能事件处理核心

核心组件:
- buffers: 零拷贝缓冲区
- level2: Level2行情处理器
- dispatcher: 事件分发器
- production: 生产部署组件

使用示例:
    from src.engine import RealTimeEngine
    from src.engine.level2 import Level2Processor

    # 初始化实时引擎
    engine = RealTimeEngine()
    engine.start()

    # 处理Level2数据
    processor = Level2Processor()
    processor.process(market_data)

主要功能:
- 超低延迟事件处理
- Level2行情解析
- 实时数据分发
- 生产环境监控

性能指标:
- 事件处理延迟 <1ms
- 吞吐量 50,000+ events/sec
- 99.99%可用性

版本历史:
- v1.0 (2024-03-15): 初始版本
- v1.1 (2024-04-20): 添加Level2支持
"""

from .realtime import RealTimeEngine
from .dispatcher import EventDispatcher
from .level2 import Level2Processor
from .buffers import RingBuffer

__all__ = [
    'RealTimeEngine',
    'EventDispatcher',
    'Level2Processor',
    'RingBuffer',
    # 子模块
    'buffers',
    'level2',
    'dispatcher',
    'production'
]
