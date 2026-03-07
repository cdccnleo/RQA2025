# -*- coding: utf-8 -*-
"""
RQA2025 流处理层数据管道
Stream Processing Layer Data Pipeline

实现数据流的处理管道，包括数据路由、转换和分发。
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from .base_processor import StreamProcessorBase, StreamProcessingResult, ProcessingStatus
from .stream_models import StreamEvent

# 获取统一基础设施集成层的日志适配器
try:
    from src.core.integration import get_models_adapter
    models_adapter = get_models_adapter()
    from src.infrastructure.logging.core.interfaces import get_logger
    logger = get_logger(__name__)
except Exception:
    logger = logging.getLogger(__name__)


class PipelineStage(Enum):

    """管道阶段"""
    INGESTION = "ingestion"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    ENRICHMENT = "enrichment"
    ROUTING = "routing"
    OUTPUT = "output"


@dataclass
class PipelineRule:

    """管道规则"""
    rule_id: str
    stage: PipelineStage
    conditions: Dict[str, Any]
    actions: List[Dict[str, Any]]
    priority: int = 0
    enabled: bool = True


@dataclass
class PipelineMetrics:

    """管道指标"""
    pipeline_id: str
    total_events: int = 0
    processed_events: int = 0
    failed_events: int = 0
    avg_processing_time_ms: float = 0.0
    stage_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class DataPipeline(StreamProcessorBase):

    """
    数据管道
    负责数据的流式处理，包括验证、转换、路由和分发
    """

    def __init__(self, pipeline_id: str, config: Optional[Dict[str, Any]] = None):

        super().__init__(pipeline_id, config)

        # 管道配置
        self.pipeline_id = pipeline_id
        self.stages = [stage.value for stage in PipelineStage]

        # 规则存储
        self.rules: Dict[str, List[PipelineRule]] = {}
        for stage in self.stages:
            self.rules[stage] = []

        # 路由表
        self.routing_table: Dict[str, List[str]] = {}  # 目标ID -> 处理器列表

        # 指标收集
        self.pipeline_metrics = PipelineMetrics(pipeline_id)

        # 加载默认规则
        self._load_default_rules()

        logger.info(f"数据管道 {pipeline_id} 已初始化")

    async def process_event(self, event: StreamEvent) -> StreamProcessingResult:
        """处理事件通过管道"""
        start_time = datetime.now()
        self.pipeline_metrics.total_events += 1

        try:
            current_event = event

            # 依次执行每个阶段
            for stage in self.stages:
                current_event = await self._execute_stage(stage, current_event)
                if current_event is None:
                    # 事件被过滤或丢弃
                    break

            if current_event is not None:
                self.pipeline_metrics.processed_events += 1

            return StreamProcessingResult(
                event_id=event.event_id,
                processing_status=ProcessingStatus.COMPLETED,
                processed_data={
                    'pipeline_id': self.pipeline_id,
                    'stages_completed': len(self.stages) if current_event else 0,
                    'final_event_id': current_event.event_id if current_event else None
                },
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )

        except Exception as e:
            self.pipeline_metrics.failed_events += 1
            return StreamProcessingResult(
                event_id=event.event_id,
                processing_status=ProcessingStatus.FAILED,
                processed_data={},
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                error_message=str(e)
            )

    async def add_rule(self, rule: PipelineRule):
        """添加管道规则"""
        stage_rules = self.rules[rule.stage.value]
        stage_rules.append(rule)
        # 按优先级排序
        stage_rules.sort(key=lambda r: r.priority, reverse=True)

        logger.info(f"添加管道规则: {rule.rule_id} 到阶段 {rule.stage.value}")

    async def remove_rule(self, rule_id: str, stage: PipelineStage) -> bool:
        """删除管道规则"""
        stage_rules = self.rules[stage.value]
        for i, rule in enumerate(stage_rules):
            if rule.rule_id == rule_id:
                del stage_rules[i]
                logger.info(f"删除管道规则: {rule_id}")
                return True
        return False

    async def add_routing_rule(self, source_pattern: str, target_processors: List[str]):
        """添加路由规则"""
        self.routing_table[source_pattern] = target_processors
        logger.info(f"添加路由规则: {source_pattern} -> {target_processors}")

    async def _execute_stage(self, stage_name: str, event: StreamEvent) -> Optional[StreamEvent]:
        """执行管道阶段"""
        stage_start = datetime.now()

        try:
            # 获取阶段规则
            stage_rules = self.rules[stage_name]

            # 应用规则
            processed_event = event
            for rule in stage_rules:
                if not rule.enabled:
                    continue

                if await self._evaluate_conditions(rule.conditions, processed_event):
                    processed_event = await self._execute_actions(rule.actions, processed_event)
                    if processed_event is None:
                        # 规则要求丢弃事件
                        break

            # 更新阶段指标
            stage_time = (datetime.now() - stage_start).total_seconds() * 1000
            if stage_name not in self.pipeline_metrics.stage_metrics:
                self.pipeline_metrics.stage_metrics[stage_name] = {
                    'total_events': 0,
                    'avg_processing_time_ms': 0.0
                }

            stage_metric = self.pipeline_metrics.stage_metrics[stage_name]
            stage_metric['total_events'] += 1
            stage_metric['avg_processing_time_ms'] = (
                (stage_metric['avg_processing_time_ms'] *
                 (stage_metric['total_events'] - 1)) + stage_time
            ) / stage_metric['total_events']

            return processed_event

        except Exception as e:
            logger.error(f"执行管道阶段 {stage_name} 异常: {str(e)}")
            return None

    async def _evaluate_conditions(self, conditions: Dict[str, Any], event: StreamEvent) -> bool:
        """评估条件"""
        for condition_key, condition_value in conditions.items():
            if condition_key == 'event_type':
                if event.event_type.value != condition_value:
                    return False
            elif condition_key == 'source':
                if event.source != condition_value:
                    return False
            elif condition_key == 'payload_field':
                field_name = condition_value.get('field')
                expected_value = condition_value.get('value')
                if event.data and event.data.get(field_name) != expected_value:
                    return False
            elif condition_key == 'payload_range':
                field_name = condition_value.get('field')
                min_val = condition_value.get('min')
                max_val = condition_value.get('max')
                field_value = event.data.get(field_name) if event.data else None
                if field_value is None or not (min_val <= field_value <= max_val):
                    return False

        return True

    async def _execute_actions(self, actions: List[Dict[str, Any]], event: StreamEvent) -> Optional[StreamEvent]:
        """执行动作"""
        processed_event = event

        for action in actions:
            action_type = action.get('type')

            if action_type == 'transform':
                processed_event = await self._execute_transform_action(action, processed_event)
            elif action_type == 'enrich':
                processed_event = await self._execute_enrich_action(action, processed_event)
            elif action_type == 'filter':
                if not await self._execute_filter_action(action, processed_event):
                    return None  # 过滤掉事件
            elif action_type == 'route':
                await self._execute_route_action(action, processed_event)
            elif action_type == 'drop':
                return None  # 丢弃事件

        return processed_event

    async def _execute_transform_action(self, action: Dict[str, Any], event: StreamEvent) -> StreamEvent:
        """执行转换动作"""
        transform_type = action.get('transform_type')

        if not event.data:
            event.data = {}

        if transform_type == 'add_field':
            field_name = action.get('field_name')
            field_value = action.get('field_value')
            event.data[field_name] = field_value

        elif transform_type == 'rename_field':
            old_name = action.get('old_name')
            new_name = action.get('new_name')
            if old_name in event.data:
                event.data[new_name] = event.data.pop(old_name)

        elif transform_type == 'calculate':
            expression = action.get('expression')
            # 简单的字段计算
            if expression == 'price * volume':
                price = event.data.get('price', 0)
                volume = event.data.get('volume', 0)
                event.data['total_value'] = price * volume

        return event

    async def _execute_enrich_action(self, action: Dict[str, Any], event: StreamEvent) -> StreamEvent:
        """执行丰富动作"""
        enrich_type = action.get('enrich_type')

        if not event.data:
            event.data = {}

        if enrich_type == 'add_timestamp':
            event.data['processed_at'] = datetime.now().isoformat()

        elif enrich_type == 'add_metadata':
            metadata = action.get('metadata', {})
            if 'metadata' not in event.data:
                event.data['metadata'] = {}
            event.data['metadata'].update(metadata)

        return event

    async def _execute_filter_action(self, action: Dict[str, Any], event: StreamEvent) -> bool:
        """执行过滤动作"""
        filter_type = action.get('filter_type')

        if not event.data:
            event.data = {}

        if filter_type == 'value_range':
            field_name = action.get('field_name')
            min_val = action.get('min_value')
            max_val = action.get('max_value')
            field_value = event.data.get(field_name)

            if field_value is None:
                return False

            return min_val <= field_value <= max_val

        elif filter_type == 'required_fields':
            required_fields = action.get('fields', [])
            for req_field in required_fields:
                if req_field not in event.data:
                    return False
            return True

        return True

    async def _execute_route_action(self, action: Dict[str, Any], event: StreamEvent):
        """执行路由动作"""
        # 这里可以实现事件路由逻辑
        # 例如：根据路由表将事件发送到指定的处理器

    def _load_default_rules(self):
        """加载默认规则"""
        # 默认验证规则
        validation_rule = PipelineRule(
            rule_id="default_validation",
            stage=PipelineStage.VALIDATION,
            conditions={},
            actions=[
                {
                    'type': 'filter',
                    'filter_type': 'required_fields',
                    'fields': ['event_type', 'timestamp']
                }
            ]
        )
        self.rules[PipelineStage.VALIDATION.value].append(validation_rule)

        # 默认丰富规则
        enrichment_rule = PipelineRule(
            rule_id="default_enrichment",
            stage=PipelineStage.ENRICHMENT,
            conditions={},
            actions=[
                {
                    'type': 'enrich',
                    'enrich_type': 'add_timestamp'
                }
            ]
        )
        self.rules[PipelineStage.ENRICHMENT.value].append(enrichment_rule)

    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """获取管道指标"""
        return {
            'pipeline_id': self.pipeline_id,
            'total_events': self.pipeline_metrics.total_events,
            'processed_events': self.pipeline_metrics.processed_events,
            'failed_events': self.pipeline_metrics.failed_events,
            'success_rate': self.pipeline_metrics.processed_events / max(self.pipeline_metrics.total_events, 1),
            'stage_metrics': self.pipeline_metrics.stage_metrics
        }


__all__ = [
    'DataPipeline',
    'PipelineRule',
    'PipelineStage',
    'PipelineMetrics'
]
