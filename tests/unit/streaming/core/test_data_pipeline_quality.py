#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据管道质量测试
测试覆盖 DataPipeline 的核心功能
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch

from tests.unit.streaming.conftest import import_data_pipeline, import_stream_models


@pytest.fixture
def data_pipeline():
    """创建数据管道实例"""
    DataPipeline = import_data_pipeline()
    if DataPipeline is None:
        try:
            from src.streaming.core.data_pipeline import DataPipeline
        except ImportError:
            pytest.skip("DataPipeline不可用")
    return DataPipeline('test_pipeline')


@pytest.fixture
def sample_event():
    """创建示例事件"""
    stream_models = import_stream_models()
    if stream_models is None:
        try:
            from src.streaming.core.stream_models import StreamEvent, StreamEventType
            stream_models = (StreamEvent, StreamEventType)
        except ImportError:
            pytest.skip("StreamEvent不可用")
    StreamEvent = stream_models[0]
    StreamEventType = stream_models[1]

    return StreamEvent(
        event_id='test_event_1',
        event_type=StreamEventType.MARKET_DATA,
        timestamp=datetime.now(),
        source='test_source',
        data={'symbol': 'AAPL', 'price': 150.0, 'volume': 1000}
    )


@pytest.fixture
def pipeline_rule():
    """创建管道规则"""
    from src.streaming.core.data_pipeline import PipelineRule, PipelineStage
    return PipelineRule(
        rule_id='test_rule',
        stage=PipelineStage.VALIDATION,
        conditions={'event_type': 'market_data'},
        actions=[]
    )


class TestDataPipeline:
    """DataPipeline测试类"""

    def test_initialization(self, data_pipeline):
        """测试初始化"""
        assert data_pipeline.pipeline_id == 'test_pipeline'
        assert len(data_pipeline.stages) > 0
        assert isinstance(data_pipeline.rules, dict)
        assert isinstance(data_pipeline.routing_table, dict)

    def test_add_rule(self, data_pipeline, pipeline_rule):
        """测试添加规则"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            await data_pipeline.add_rule(pipeline_rule)
            return len(data_pipeline.rules[pipeline_rule.stage.value])
        
        rule_count = loop.run_until_complete(test_async())
        assert rule_count > 0

    def test_remove_rule(self, data_pipeline, pipeline_rule):
        """测试移除规则"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            # 先添加规则
            await data_pipeline.add_rule(pipeline_rule)
            # 然后移除规则
            result = await data_pipeline.remove_rule(pipeline_rule.rule_id, pipeline_rule.stage)
            return result
        
        result = loop.run_until_complete(test_async())
        assert result is True

    def test_add_routing_rule(self, data_pipeline):
        """测试添加路由规则"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            await data_pipeline.add_routing_rule('source_pattern', ['processor1', 'processor2'])
            return 'source_pattern' in data_pipeline.routing_table
        
        result = loop.run_until_complete(test_async())
        assert result is True

    def test_process_event(self, data_pipeline, sample_event):
        """测试处理事件"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            result = await data_pipeline.process_event(sample_event)
            return result
        
        result = loop.run_until_complete(test_async())
        assert result is not None
        assert hasattr(result, 'event_id')
        assert hasattr(result, 'processing_status')

    def test_get_pipeline_metrics(self, data_pipeline):
        """测试获取管道指标"""
        metrics = data_pipeline.get_pipeline_metrics()
        assert isinstance(metrics, dict)
        assert 'pipeline_id' in metrics
        assert 'total_events' in metrics
        assert 'processed_events' in metrics
        assert 'failed_events' in metrics

    def test_pipeline_stages(self, data_pipeline):
        """测试管道阶段"""
        from src.streaming.core.data_pipeline import PipelineStage
        assert PipelineStage.INGESTION in PipelineStage
        assert PipelineStage.VALIDATION in PipelineStage
        assert PipelineStage.TRANSFORMATION in PipelineStage
        assert PipelineStage.ENRICHMENT in PipelineStage
        assert PipelineStage.ROUTING in PipelineStage
        assert PipelineStage.OUTPUT in PipelineStage

    def test_pipeline_rule_creation(self):
        """测试管道规则创建"""
        from src.streaming.core.data_pipeline import PipelineRule, PipelineStage
        rule = PipelineRule(
            rule_id='test_rule_2',
            stage=PipelineStage.TRANSFORMATION,
            conditions={'field': 'value'},
            actions=[{'action': 'transform'}],
            priority=1,
            enabled=True
        )
        assert rule.rule_id == 'test_rule_2'
        assert rule.stage == PipelineStage.TRANSFORMATION
        assert rule.priority == 1
        assert rule.enabled is True

    def test_pipeline_metrics_creation(self):
        """测试管道指标创建"""
        from src.streaming.core.data_pipeline import PipelineMetrics
        metrics = PipelineMetrics(
            pipeline_id='test_pipeline',
            total_events=100,
            processed_events=95,
            failed_events=5,
            avg_processing_time_ms=10.5
        )
        assert metrics.pipeline_id == 'test_pipeline'
        assert metrics.total_events == 100
        assert metrics.processed_events == 95
        assert metrics.failed_events == 5
        assert metrics.avg_processing_time_ms == 10.5

    def test_process_event_with_rules(self, data_pipeline, sample_event, pipeline_rule):
        """测试使用规则处理事件"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            # 添加规则
            await data_pipeline.add_rule(pipeline_rule)
            # 处理事件
            result = await data_pipeline.process_event(sample_event)
            return result
        
        result = loop.run_until_complete(test_async())
        assert result is not None

    def test_process_event_multiple_stages(self, data_pipeline, sample_event):
        """测试多阶段处理事件"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            # 添加多个阶段的规则
            from src.streaming.core.data_pipeline import PipelineRule, PipelineStage
            
            validation_rule = PipelineRule(
                rule_id='validation_rule',
                stage=PipelineStage.VALIDATION,
                conditions={},
                actions=[]
            )
            await data_pipeline.add_rule(validation_rule)
            
            transformation_rule = PipelineRule(
                rule_id='transformation_rule',
                stage=PipelineStage.TRANSFORMATION,
                conditions={},
                actions=[]
            )
            await data_pipeline.add_rule(transformation_rule)
            
            # 处理事件
            result = await data_pipeline.process_event(sample_event)
            return result
        
        result = loop.run_until_complete(test_async())
        assert result is not None

    def test_execute_stage(self, data_pipeline, sample_event):
        """测试执行管道阶段"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            result = await data_pipeline._execute_stage('validation', sample_event)
            return result
        
        result = loop.run_until_complete(test_async())
        # 可能返回None或事件
        assert result is None or hasattr(result, 'event_id')

    def test_evaluate_conditions(self, data_pipeline, sample_event):
        """测试评估条件"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            # 测试event_type条件
            result1 = await data_pipeline._evaluate_conditions({'event_type': 'market_data'}, sample_event)
            result2 = await data_pipeline._evaluate_conditions({'event_type': 'invalid'}, sample_event)
            return result1, result2
        
        result1, result2 = loop.run_until_complete(test_async())
        assert result1 is True or result2 is False  # 至少一个条件应该匹配

    def test_execute_transform_action(self, data_pipeline, sample_event):
        """测试执行转换动作"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            action = {
                'type': 'transform',
                'transform_type': 'add_field',
                'field_name': 'new_field',
                'field_value': 'new_value'
            }
            result = await data_pipeline._execute_transform_action(action, sample_event)
            return result
        
        result = loop.run_until_complete(test_async())
        assert result is not None
        assert result.data.get('new_field') == 'new_value'

    def test_execute_enrich_action(self, data_pipeline, sample_event):
        """测试执行丰富动作"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            action = {
                'type': 'enrich',
                'enrich_type': 'add_timestamp'
            }
            result = await data_pipeline._execute_enrich_action(action, sample_event)
            return result
        
        result = loop.run_until_complete(test_async())
        assert result is not None
        assert 'processed_at' in result.data

    def test_execute_filter_action(self, data_pipeline, sample_event):
        """测试执行过滤动作"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            action = {
                'type': 'filter',
                'filter_type': 'value_range',
                'field_name': 'price',
                'min_value': 100,
                'max_value': 200
            }
            result = await data_pipeline._execute_filter_action(action, sample_event)
            return result
        
        result = loop.run_until_complete(test_async())
        assert isinstance(result, bool)

    def test_execute_actions(self, data_pipeline, sample_event):
        """测试执行动作列表"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            actions = [
                {
                    'type': 'transform',
                    'transform_type': 'add_field',
                    'field_name': 'test_field',
                    'field_value': 'test_value'
                }
            ]
            result = await data_pipeline._execute_actions(actions, sample_event)
            return result
        
        result = loop.run_until_complete(test_async())
        assert result is not None

    def test_process_event_exception(self, data_pipeline, sample_event):
        """测试处理事件异常"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            # Mock _execute_stage抛出异常
            with patch.object(data_pipeline, '_execute_stage', side_effect=Exception("Test error")):
                result = await data_pipeline.process_event(sample_event)
                return result
        
        result = loop.run_until_complete(test_async())
        assert result.processing_status == 'failed'

    def test_remove_rule_not_found(self, data_pipeline):
        """测试移除不存在的规则"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            from src.streaming.core.data_pipeline import PipelineStage
            result = await data_pipeline.remove_rule('nonexistent', PipelineStage.VALIDATION)
            return result
        
        result = loop.run_until_complete(test_async())
        assert result is False

    def test_execute_stage_with_disabled_rule(self, data_pipeline, sample_event):
        """测试执行阶段时禁用规则"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            from src.streaming.core.data_pipeline import PipelineRule, PipelineStage
            # 清空默认规则，避免干扰
            data_pipeline.rules[PipelineStage.VALIDATION.value] = []
            
            disabled_rule = PipelineRule(
                rule_id='disabled_rule',
                stage=PipelineStage.VALIDATION,
                conditions={},
                actions=[],
                enabled=False
            )
            await data_pipeline.add_rule(disabled_rule)
            result = await data_pipeline._execute_stage('validation', sample_event)
            return result
        
        result = loop.run_until_complete(test_async())
        # 禁用规则应该被跳过，事件应该正常返回
        assert result is not None

    def test_execute_stage_exception(self, data_pipeline, sample_event):
        """测试执行阶段异常"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            # Mock _evaluate_conditions抛出异常
            with patch.object(data_pipeline, '_evaluate_conditions', side_effect=Exception("Test error")):
                result = await data_pipeline._execute_stage('validation', sample_event)
                return result
        
        result = loop.run_until_complete(test_async())
        assert result is None

    def test_evaluate_conditions_source(self, data_pipeline, sample_event):
        """测试评估source条件"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            result1 = await data_pipeline._evaluate_conditions({'source': 'test_source'}, sample_event)
            result2 = await data_pipeline._evaluate_conditions({'source': 'wrong_source'}, sample_event)
            return result1, result2
        
        result1, result2 = loop.run_until_complete(test_async())
        assert result1 is True
        assert result2 is False

    def test_evaluate_conditions_payload_field(self, data_pipeline, sample_event):
        """测试评估payload_field条件"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            conditions = {
                'payload_field': {
                    'field': 'symbol',
                    'value': 'AAPL'
                }
            }
            result1 = await data_pipeline._evaluate_conditions(conditions, sample_event)
            
            conditions2 = {
                'payload_field': {
                    'field': 'symbol',
                    'value': 'MSFT'
                }
            }
            result2 = await data_pipeline._evaluate_conditions(conditions2, sample_event)
            return result1, result2
        
        result1, result2 = loop.run_until_complete(test_async())
        assert result1 is True
        assert result2 is False

    def test_evaluate_conditions_payload_range(self, data_pipeline, sample_event):
        """测试评估payload_range条件"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            conditions = {
                'payload_range': {
                    'field': 'price',
                    'min': 100,
                    'max': 200
                }
            }
            result1 = await data_pipeline._evaluate_conditions(conditions, sample_event)
            
            conditions2 = {
                'payload_range': {
                    'field': 'price',
                    'min': 200,
                    'max': 300
                }
            }
            result2 = await data_pipeline._evaluate_conditions(conditions2, sample_event)
            return result1, result2
        
        result1, result2 = loop.run_until_complete(test_async())
        assert result1 is True
        assert result2 is False

    def test_execute_actions_enrich(self, data_pipeline, sample_event):
        """测试执行enrich动作"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            actions = [
                {
                    'type': 'enrich',
                    'enrich_type': 'add_timestamp'
                }
            ]
            result = await data_pipeline._execute_actions(actions, sample_event)
            return result
        
        result = loop.run_until_complete(test_async())
        assert result is not None

    def test_execute_actions_route(self, data_pipeline, sample_event):
        """测试执行route动作"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            actions = [
                {
                    'type': 'route',
                    'target': 'processor1'
                }
            ]
            result = await data_pipeline._execute_actions(actions, sample_event)
            return result
        
        result = loop.run_until_complete(test_async())
        assert result is not None

    def test_execute_actions_drop(self, data_pipeline, sample_event):
        """测试执行drop动作"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            actions = [
                {
                    'type': 'drop'
                }
            ]
            result = await data_pipeline._execute_actions(actions, sample_event)
            return result
        
        result = loop.run_until_complete(test_async())
        assert result is None

    def test_execute_transform_action_rename_field(self, data_pipeline, sample_event):
        """测试执行rename_field转换"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            action = {
                'type': 'transform',
                'transform_type': 'rename_field',
                'old_name': 'symbol',
                'new_name': 'ticker'
            }
            result = await data_pipeline._execute_transform_action(action, sample_event)
            return result
        
        result = loop.run_until_complete(test_async())
        assert result.data.get('ticker') == 'AAPL'
        assert 'symbol' not in result.data

    def test_execute_transform_action_calculate(self, data_pipeline, sample_event):
        """测试执行calculate转换"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            action = {
                'type': 'transform',
                'transform_type': 'calculate',
                'expression': 'price * volume'
            }
            result = await data_pipeline._execute_transform_action(action, sample_event)
            return result
        
        result = loop.run_until_complete(test_async())
        assert result.data.get('total_value') == 150000.0

    def test_execute_transform_action_no_data(self, data_pipeline):
        """测试转换动作（无数据）"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            from src.streaming.core.stream_models import StreamEvent, StreamEventType
            event = StreamEvent(
                event_id='test',
                event_type=StreamEventType.MARKET_DATA,
                timestamp=datetime.now(),
                source='test',
                data=None
            )
            action = {
                'type': 'transform',
                'transform_type': 'add_field',
                'field_name': 'new_field',
                'field_value': 'new_value'
            }
            result = await data_pipeline._execute_transform_action(action, event)
            return result
        
        result = loop.run_until_complete(test_async())
        assert result.data.get('new_field') == 'new_value'

    def test_execute_enrich_action_add_metadata(self, data_pipeline, sample_event):
        """测试执行add_metadata丰富动作"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            action = {
                'type': 'enrich',
                'enrich_type': 'add_metadata',
                'metadata': {'source': 'test', 'version': '1.0'}
            }
            result = await data_pipeline._execute_enrich_action(action, sample_event)
            return result
        
        result = loop.run_until_complete(test_async())
        assert 'metadata' in result.data
        assert result.data['metadata']['source'] == 'test'

    def test_execute_filter_action_required_fields(self, data_pipeline, sample_event):
        """测试执行required_fields过滤动作"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            action = {
                'type': 'filter',
                'filter_type': 'required_fields',
                'fields': ['symbol', 'price']
            }
            result1 = await data_pipeline._execute_filter_action(action, sample_event)
            
            action2 = {
                'type': 'filter',
                'filter_type': 'required_fields',
                'fields': ['nonexistent_field']
            }
            result2 = await data_pipeline._execute_filter_action(action2, sample_event)
            return result1, result2
        
        result1, result2 = loop.run_until_complete(test_async())
        assert result1 is True
        assert result2 is False

    def test_execute_filter_action_value_range_none(self, data_pipeline):
        """测试执行value_range过滤（字段为None）"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            from src.streaming.core.stream_models import StreamEvent, StreamEventType
            event = StreamEvent(
                event_id='test',
                event_type=StreamEventType.MARKET_DATA,
                timestamp=datetime.now(),
                source='test',
                data={'symbol': 'AAPL'}  # 没有price字段
            )
            action = {
                'type': 'filter',
                'filter_type': 'value_range',
                'field_name': 'price',
                'min_value': 100,
                'max_value': 200
            }
            result = await data_pipeline._execute_filter_action(action, event)
            return result
        
        result = loop.run_until_complete(test_async())
        assert result is False

    def test_process_event_with_dropped_event(self, data_pipeline, sample_event):
        """测试处理被丢弃的事件"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            from src.streaming.core.data_pipeline import PipelineRule, PipelineStage
            drop_rule = PipelineRule(
                rule_id='drop_rule',
                stage=PipelineStage.VALIDATION,
                conditions={},
                actions=[{'type': 'drop'}]
            )
            await data_pipeline.add_rule(drop_rule)
            result = await data_pipeline.process_event(sample_event)
            return result
        
        result = loop.run_until_complete(test_async())
        assert result is not None
        assert result.processing_status == 'completed'
