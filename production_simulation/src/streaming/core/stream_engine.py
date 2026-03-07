# -*- coding: utf-8 -*-
"""
RQA2025 流处理层流处理引擎
Stream Processing Layer Stream Processing Engine

实现完整的流处理引擎，整合所有组件。
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field

from .base_processor import StreamProcessorBase
from .stream_models import StreamEvent
from .aggregator import RealTimeAggregator
from .state_manager import StateManager
from .data_pipeline import DataPipeline

# 获取统一基础设施集成层的日志适配器
try:
    from src.core.integration import get_models_adapter
    models_adapter = get_models_adapter()
    from src.infrastructure.logging.core.interfaces import get_logger
    logger = get_logger(__name__)
except Exception:
    logger = logging.getLogger(__name__)


@dataclass
class StreamTopology:

    """流拓扑结构"""
    topology_id: str
    processors: List[str] = field(default_factory=list)
    connections: Dict[str, List[str]] = field(default_factory=dict)  # 处理器 -> 下游处理器列表
    config: Dict[str, Any] = field(default_factory=dict)


class StreamProcessingEngine:

    """
    流处理引擎
    整合所有流处理组件，提供完整的流处理能力
    """

    def __init__(self, engine_id: str, config: Optional[Dict[str, Any]] = None):

        self.engine_id = engine_id
        self.config = config or {}

        # 引擎配置
        self.enable_kafka = self.config.get('enable_kafka', True)
        self.enable_redis = self.config.get('enable_redis', True)
        self.enable_clickhouse = self.config.get('enable_clickhouse', True)

        # 组件存储
        self.processors: Dict[str, StreamProcessorBase] = {}
        self.topologies: Dict[str, StreamTopology] = {}

        # 事件队列
        self.event_queue = asyncio.Queue(maxsize=self.config.get('queue_size', 10000))

        # 引擎状态
        self.is_running = False
        self.start_time: Optional[datetime] = None

        # 性能指标
        self.engine_metrics = {
            'total_events_processed': 0,
            'total_processing_time_ms': 0,
            'active_processors': 0,
            'queue_size': 0
        }

        logger.info(f"流处理引擎 {engine_id} 已初始化")

    async def start_engine(self):
        """启动流处理引擎"""
        if self.is_running:
            logger.warning(f"引擎 {self.engine_id} 已在运行中")
            return

        self.is_running = True
        self.start_time = datetime.now()

        logger.info(f"启动流处理引擎 {self.engine_id}")

        # 启动所有处理器
        for processor in self.processors.values():
            await processor.start_processing()

        # 启动引擎处理循环
        asyncio.create_task(self._engine_processing_loop())

        # 启动指标收集
        asyncio.create_task(self._metrics_collection_loop())

        logger.info(f"流处理引擎 {self.engine_id} 启动完成")

    async def stop_engine(self):
        """停止流处理引擎"""
        if not self.is_running:
            logger.info(f"引擎 {self.engine_id} 已停止")
            return

        self.is_running = False
        logger.info(f"正在停止流处理引擎 {self.engine_id}")

        # 停止所有处理器
        stop_tasks = []
        for processor in self.processors.values():
            stop_tasks.append(processor.stop_processing())

        await asyncio.gather(*stop_tasks, return_exceptions=True)

        # 清空队列
        while not self.event_queue.empty():
            try:
                self.event_queue.get_nowait()
                self.event_queue.task_done()
            except asyncio.QueueEmpty:
                break

        logger.info(f"流处理引擎 {self.engine_id} 已停止")

    async def add_processor(self, processor: StreamProcessorBase):
        """添加处理器"""
        self.processors[processor.processor_id] = processor
        logger.info(f"添加处理器: {processor.processor_id}")

    async def remove_processor(self, processor_id: str) -> bool:
        """删除处理器"""
        if processor_id in self.processors:
            processor = self.processors[processor_id]
            await processor.stop_processing()
            del self.processors[processor_id]
            logger.info(f"删除处理器: {processor_id}")
            return True
        return False

    async def create_topology(self, topology: StreamTopology):
        """创建流拓扑"""
        self.topologies[topology.topology_id] = topology

        # 创建处理器实例
        for processor_id in topology.processors:
            if processor_id not in self.processors:
                await self._create_processor_from_config(processor_id, topology.config)

        logger.info(f"创建流拓扑: {topology.topology_id}")

    async def submit_event(self, event: StreamEvent) -> bool:
        """提交事件到引擎"""
        if not self.is_running:
            logger.warning(f"引擎 {self.engine_id} 未运行，无法提交事件")
            return False

        try:
            await asyncio.wait_for(
                self.event_queue.put(event),
                timeout=5.0
            )
            return True
        except asyncio.TimeoutError:
            logger.error(f"提交事件 {event.event_id} 超时")
            return False

    async def submit_events_batch(self, events: List[StreamEvent]) -> int:
        """批量提交事件"""
        submitted_count = 0
        for event in events:
            if await self.submit_event(event):
                submitted_count += 1
            else:
                break
        return submitted_count

    async def _engine_processing_loop(self):
        """引擎处理循环"""
        while self.is_running:
            try:
                # 获取待处理事件
                event = await self.event_queue.get()

                # 查找适用的拓扑
                applicable_topologies = await self._find_applicable_topologies(event)

                # 在每个适用的拓扑中处理事件
                for topology in applicable_topologies:
                    await self._process_event_in_topology(event, topology)

                self.event_queue.task_done()
                self.engine_metrics['total_events_processed'] += 1

            except Exception as e:
                logger.error(f"引擎处理循环异常: {str(e)}")

    async def _find_applicable_topologies(self, event: StreamEvent) -> List[StreamTopology]:
        """查找适用的拓扑"""
        applicable = []

        for topology in self.topologies.values():
            # 检查拓扑配置是否匹配事件
            if await self._topology_matches_event(topology, event):
                applicable.append(topology)

        return applicable

    async def _topology_matches_event(self, topology: StreamTopology, event: StreamEvent) -> bool:
        """检查拓扑是否匹配事件"""
        # 简单的匹配逻辑，可以根据需要扩展
        topology_config = topology.config

        # 检查事件类型匹配
        if 'event_types' in topology_config:
            if event.event_type.value not in topology_config['event_types']:
                return False

        # 检查来源匹配
        if 'sources' in topology_config:
            if event.source not in topology_config['sources']:
                return False

        return True

    async def _process_event_in_topology(self, event: StreamEvent, topology: StreamTopology):
        """在拓扑中处理事件"""
        current_event = event

        # 按照拓扑连接顺序处理
        for processor_id in topology.processors:
            if processor_id in self.processors:
                processor = self.processors[processor_id]

                # 提交事件到处理器
                success = await processor.submit_event(current_event)
                if not success:
                    logger.error(f"处理器 {processor_id} 处理事件失败")
                    break

                # 获取处理结果（这里简化了，实际可能需要等待结果）
                # 在实际实现中，可能需要更复杂的同步机制

    async def _create_processor_from_config(self, processor_id: str, config: Dict[str, Any]):
        """根据配置创建处理器"""
        processor_type = config.get('type', 'data_pipeline')

        if processor_type == 'aggregator':
            processor = RealTimeAggregator(processor_id, config)
        elif processor_type == 'state_manager':
            processor = StateManager(processor_id, config)
        elif processor_type == 'data_pipeline':
            processor = DataPipeline(processor_id, config)
        else:
            raise ValueError(f"不支持的处理器类型: {processor_type}")

        await self.add_processor(processor)

    async def _metrics_collection_loop(self):
        """指标收集循环"""
        while self.is_running:
            try:
                await asyncio.sleep(30)  # 每30秒收集一次

                # 更新引擎指标
                self.engine_metrics['queue_size'] = self.event_queue.qsize()
                self.engine_metrics['active_processors'] = len([
                    p for p in self.processors.values() if p.is_running
                ])

                # 可以在这里发送指标到监控系统

            except Exception as e:
                logger.error(f"指标收集循环异常: {str(e)}")

    def get_engine_status(self) -> Dict[str, Any]:
        """获取引擎状态"""
        return {
            'engine_id': self.engine_id,
            'is_running': self.is_running,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            'active_processors': len([p for p in self.processors.values() if p.is_running]),
            'total_processors': len(self.processors),
            'active_topologies': len(self.topologies),
            'queue_size': self.event_queue.qsize(),
            'engine_metrics': self.engine_metrics
        }

    def get_topology_status(self, topology_id: str) -> Optional[Dict[str, Any]]:
        """获取拓扑状态"""
        if topology_id not in self.topologies:
            return None

        topology = self.topologies[topology_id]
        processor_statuses = {}

        for processor_id in topology.processors:
            if processor_id in self.processors:
                processor_statuses[processor_id] = self.processors[processor_id].get_processor_status()

        return {
            'topology_id': topology_id,
            'processors': topology.processors,
            'connections': topology.connections,
            'processor_statuses': processor_statuses
        }


# 创建默认的流处理引擎工厂函数
async def create_stream_engine(engine_id: str, config: Optional[Dict[str, Any]] = None) -> StreamProcessingEngine:
    """创建流处理引擎"""
    engine = StreamProcessingEngine(engine_id, config)

    # 创建默认拓扑

    default_topology = StreamTopology(
        topology_id="default_market_data_topology",
        processors=["market_data_pipeline", "realtime_aggregator", "market_state_manager"],
        connections={
            "market_data_pipeline": ["realtime_aggregator"],
            "realtime_aggregator": ["market_state_manager"]
        },
        config={
            'event_types': ['market_data', 'order_update', 'trade_execution'],
            'sources': ['exchange_feed', 'order_system', 'trading_engine']
        }
    )

    await engine.create_topology(default_topology)
    return engine


__all__ = [
    'StreamProcessingEngine',
    'StreamTopology',
    'create_stream_engine'
]
