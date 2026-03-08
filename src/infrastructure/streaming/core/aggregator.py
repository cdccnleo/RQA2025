# -*- coding: utf-8 -*-
"""
RQA2025 流处理层实时聚合器
Stream Processing Layer Real - Time Aggregator

实现对流数据的实时聚合计算功能。
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from .base_processor import StreamProcessorBase, StreamProcessingResult, ProcessingStatus
from .stream_models import StreamEvent

# 获取统一基础设施集成层的日志适配器
try:
    from src.infrastructure.integration import get_models_adapter
    models_adapter = get_models_adapter()
    logger = logging.getLogger(__name__)
except Exception:
    try:
        from src.infrastructure.logging.core.interfaces import get_logger
        logger = get_logger(__name__)
    except Exception:
        # 如果都失败了，使用标准logging
        logger = logging.getLogger(__name__)


@dataclass
class WindowedData:

    """窗口数据"""
    window_id: str
    window_start: datetime
    window_end: datetime
    data_points: List[Dict[str, Any]] = field(default_factory=list)
    aggregations: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_data_point(self, data: Dict[str, Any]):
        """添加数据点"""
        self.data_points.append(data)

    def update_aggregation(self, agg_name: str, value: Any):
        """更新聚合结果"""
        self.aggregations[agg_name] = value

    def is_expired(self, current_time: datetime) -> bool:
        """检查窗口是否过期"""
        return current_time > self.window_end

    def get_data_summary(self) -> Dict[str, Any]:
        """获取数据摘要"""
        return {
            'window_id': self.window_id,
            'window_start': self.window_start.isoformat(),
            'window_end': self.window_end.isoformat(),
            'data_points_count': len(self.data_points),
            'aggregations': self.aggregations,
            'metadata': self.metadata
        }


class RealTimeAggregator(StreamProcessorBase):

    """
    实时聚合器
    负责对流数据进行实时聚合计算
    """

    def __init__(self, processor_id: str, config: Optional[Dict[str, Any]] = None):

        super().__init__(processor_id, config)

        # 聚合配置
        self.window_duration = timedelta(seconds=self.config.get('window_duration_seconds', 60))
        self.slide_interval = timedelta(seconds=self.config.get('slide_interval_seconds', 10))

        # 窗口存储
        self.active_windows: Dict[str, WindowedData] = {}
        self.window_cleanup_interval = self.config.get('window_cleanup_interval', 300)  # 5分钟

        # 聚合函数
        self.aggregation_functions = self._load_aggregation_functions()

        logger.info(f"实时聚合器 {processor_id} 已初始化")

    async def process_event(self, event: StreamEvent) -> StreamProcessingResult:
        """处理事件并进行聚合"""
        start_time = datetime.now()

        try:
            # 获取或创建窗口
            window = await self._get_window_for_event(event)

            # 添加数据到窗口
            window.add_data_point(event.data)

            # 执行聚合计算
            await self._perform_aggregations(window)

            # 检查是否需要触发输出
            should_emit = await self._should_emit_window(window)

            result_data = {
                'window_id': window.window_id,
                'aggregations': window.aggregations,
                'data_points_count': len(window.data_points)
            }

            if should_emit:
                result_data['emit_triggered'] = True
                # 重置窗口以进行滑动
                await self._slide_window(window)

            return StreamProcessingResult(
                event_id=event.event_id,
                processing_status=ProcessingStatus.COMPLETED,
                processed_data=result_data,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )

        except Exception as e:
            return StreamProcessingResult(
                event_id=event.event_id,
                processing_status=ProcessingStatus.FAILED,
                processed_data={},
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                error_message=str(e)
            )

    async def _get_window_for_event(self, event: StreamEvent) -> WindowedData:
        """获取事件对应的窗口"""
        # 简单的基于时间的窗口分配
        # 在实际应用中可能需要更复杂的窗口策略

        window_start = event.timestamp.replace(second=0, microsecond=0)
        window_id = f"{event.data.get('symbol', 'unknown')}_{window_start.isoformat()}"

        if window_id not in self.active_windows:
            window = WindowedData(
                window_id=window_id,
                window_start=window_start,
                window_end=window_start + self.window_duration,
                metadata={'symbol': event.data.get('symbol', 'unknown')}
            )
            self.active_windows[window_id] = window

            # 启动窗口清理任务
            asyncio.create_task(self._schedule_window_cleanup(window))

        return self.active_windows[window_id]

    async def _perform_aggregations(self, window: WindowedData):
        """执行聚合计算"""
        if not window.data_points:
            return

        # 执行配置的聚合函数
        for agg_name, agg_config in self.aggregation_functions.items():
            try:
                result = await self._execute_aggregation(agg_config, window.data_points)
                window.update_aggregation(agg_name, result)
            except Exception as e:
                logger.error(f"聚合计算 {agg_name} 失败: {str(e)}")

    async def _execute_aggregation(self, agg_config: Dict[str, Any],
                                   data_points: List[Dict[str, Any]]) -> Any:
        """执行单个聚合计算"""
        agg_type = agg_config.get('type', 'count')
        field = agg_config.get('field', 'value')

        # 提取字段值
        values = []
        for point in data_points:
            if field in point:
                values.append(point[field])

        if not values:
            return 0

        # 执行聚合
        if agg_type == 'count':
            return len(values)
        elif agg_type == 'sum':
            return sum(values)
        elif agg_type == 'avg':
            return sum(values) / len(values)
        elif agg_type == 'max':
            return max(values)
        elif agg_type == 'min':
            return min(values)
        elif agg_type == 'std':
            # 简化的标准差计算
            avg = sum(values) / len(values)
            variance = sum((x - avg) ** 2 for x in values) / len(values)
            return variance ** 0.5
        else:
            raise ValueError(f"不支持的聚合类型: {agg_type}")

    async def _should_emit_window(self, window: WindowedData) -> bool:
        """检查是否应该发射窗口"""
        # 简单的基于时间和数据量的发射策略
        current_time = datetime.now()

        # 时间到达窗口结束时间
        if current_time >= window.window_end:
            return True

        # 数据量达到阈值
        min_data_points = self.config.get('min_data_points_for_emit', 10)
        if len(window.data_points) >= min_data_points:
            return True

        return False

    async def _slide_window(self, window: WindowedData):
        """滑动窗口"""
        # 创建新窗口
        new_window_start = window.window_end
        new_window = WindowedData(
            window_id=f"{window.metadata.get('symbol', 'unknown')}_{new_window_start.isoformat()}",
            window_start=new_window_start,
            window_end=new_window_start + self.window_duration,
            metadata=window.metadata
        )

        # 可以在这里实现更复杂的滑动逻辑
        # 例如：保留部分历史数据用于计算趋势

        # 清理旧窗口
        del self.active_windows[window.window_id]

        # 添加新窗口
        self.active_windows[new_window.window_id] = new_window

    async def _schedule_window_cleanup(self, window: WindowedData):
        """调度窗口清理"""
        await asyncio.sleep(self.window_cleanup_interval)

        # 检查窗口是否仍然存在且已过期
        if (window.window_id in self.active_windows
                and window.is_expired(datetime.now())):
            del self.active_windows[window.window_id]
            logger.info(f"清理过期窗口: {window.window_id}")

    async def _cleanup_expired_windows(self):
        """清理过期窗口"""
        current_time = datetime.now()
        expired_window_ids = [
            window_id
            for window_id, window in self.active_windows.items()
            if window.is_expired(current_time)
        ]
        
        for window_id in expired_window_ids:
            del self.active_windows[window_id]
            logger.info(f"清理过期窗口: {window_id}")
        
        return len(expired_window_ids)

    def _load_aggregation_functions(self) -> Dict[str, Dict[str, Any]]:
        """加载聚合函数配置"""
        # 默认聚合配置
        return {
            'price_avg': {
                'type': 'avg',
                'field': 'price'
            },
            'volume_sum': {
                'type': 'sum',
                'field': 'volume'
            },
            'price_max': {
                'type': 'max',
                'field': 'price'
            },
            'price_min': {
                'type': 'min',
                'field': 'price'
            },
            'data_count': {
                'type': 'count',
                'field': 'price'
            }
        }

    def get_active_windows(self) -> Dict[str, Dict[str, Any]]:
        """获取活跃窗口信息"""
        return {
            window_id: window.get_data_summary()
            for window_id, window in self.active_windows.items()
        }


__all__ = [
    'RealTimeAggregator',
    'WindowedData'
]
