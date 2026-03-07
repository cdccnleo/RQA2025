#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
流处理层测试配置和导入辅助
提供统一的导入逻辑，解决pytest-xdist并发环境下的导入问题
"""

import sys
import os
import importlib
from typing import Optional, Any
import pytest

# 确保项目根目录在路径中
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')

# 在模块加载时确保路径正确
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# 添加pytest hook确保在测试运行前路径正确
def pytest_configure(config):
    """pytest配置钩子，确保路径正确"""
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    if SRC_PATH not in sys.path:
        sys.path.insert(0, SRC_PATH)


def import_streaming_module(module_path: str, class_name: Optional[str] = None) -> Any:
    """
    导入流处理模块（增强版，支持更多容错机制）

    Args:
        module_path: 模块路径，如 'src.streaming.core.realtime_analyzer' 或 'streaming.core.realtime_analyzer'
        class_name: 要导入的类名，如果为None则返回模块

    Returns:
        导入的类或模块，失败返回None
    """
    # 确保路径在sys.path中（每次调用都检查）
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    if SRC_PATH not in sys.path:
        sys.path.insert(0, SRC_PATH)

    # 尝试多种导入路径
    import_paths = [
        module_path,
        module_path.replace('src.streaming', 'streaming') if 'src.streaming' in module_path else None,
        module_path.replace('streaming', 'src.streaming') if 'streaming' in module_path and 'src.streaming' not in module_path else None,
    ]

    # 移除None值并去重
    import_paths = list(dict.fromkeys([p for p in import_paths if p is not None]))

    # 如果模块路径包含相对导入，尝试绝对导入
    if not any(path.startswith('src.') for path in import_paths):
        import_paths.insert(0, f'src.{module_path}' if not module_path.startswith('src.') else module_path)

    for path in import_paths:
        try:
            # 尝试导入模块
            module = importlib.import_module(path)

            if class_name:
                # 尝试多种方式获取类
                result = getattr(module, class_name, None)

                # 如果直接获取失败，尝试从__all__中查找
                if result is None and hasattr(module, '__all__') and class_name in module.__all__:
                    try:
                        result = getattr(module, class_name)
                    except AttributeError:
                        pass

                # 如果还是失败，尝试从模块字典中查找（处理动态导入）
                if result is None and hasattr(module, '__dict__') and class_name in module.__dict__:
                    result = module.__dict__[class_name]

                # 如果还是失败，尝试直接访问模块属性（处理私有属性）
                if result is None:
                    try:
                        result = module.__getattribute__(class_name)
                    except AttributeError:
                        pass

                # 如果还是失败，尝试从子模块中查找
                if result is None and '.' in class_name:
                    parts = class_name.split('.')
                    current = module
                    for part in parts:
                        current = getattr(current, part, None)
                        if current is None:
                            break
                    result = current

                if result is not None:
                    return result
            else:
                return module

        except (ImportError, AttributeError, ModuleNotFoundError, TypeError, ValueError) as e:
            # 继续尝试下一个路径
            continue
        except Exception as e:
            # 记录其他异常但不中断，继续尝试
            continue

    # 如果所有路径都失败，尝试重新加载模块（处理循环导入问题）
    if class_name:
        try:
            # 尝试从已导入的模块中查找
            for module_name in sys.modules:
                if 'streaming' in module_name and module_name.endswith(module_path.split('.')[-1]):
                    module = sys.modules[module_name]
                    result = getattr(module, class_name, None)
                    if result is not None:
                        return result
        except Exception:
            pass

    return None


# 便捷导入函数
def import_realtime_analyzer():
    """导入RealTimeAnalyzer"""
    return import_streaming_module('src.streaming.core.realtime_analyzer', 'RealTimeAnalyzer')


def import_data_processor():
    """导入DataProcessor"""
    return import_streaming_module('src.streaming.core.data_processor', 'DataProcessor')


def import_stream_processor():
    """导入StreamProcessor"""
    return import_streaming_module('src.streaming.core.stream_processor', 'StreamProcessor')


def import_data_stream_processor():
    """导入DataStreamProcessor"""
    return import_streaming_module('src.streaming.core.data_stream_processor', 'DataStreamProcessor')


def import_stream_engine():
    """导入StreamProcessingEngine"""
    return import_streaming_module('src.streaming.core.stream_engine', 'StreamProcessingEngine')


def import_base_processor():
    """导入StreamProcessorBase"""
    return import_streaming_module('src.streaming.core.base_processor', 'StreamProcessorBase')


def import_realtime_component_factory():
    """导入RealtimeComponentFactory"""
    return import_streaming_module('src.streaming.engine.realtime_components', 'RealtimeComponentFactory')


def import_performance_optimizer():
    """导入PerformanceOptimizer"""
    return import_streaming_module('src.streaming.optimization.performance_optimizer', 'PerformanceOptimizer')


def import_stream_models():
    """导入StreamEvent和StreamEventType"""
    StreamEvent = import_streaming_module('src.streaming.core.stream_models', 'StreamEvent')
    StreamEventType = import_streaming_module('src.streaming.core.stream_models', 'StreamEventType')
    return StreamEvent, StreamEventType


def import_memory_optimizer():
    """导入MemoryOptimizer"""
    return import_streaming_module('src.streaming.optimization.memory_optimizer', 'MemoryOptimizer')


def import_throughput_optimizer():
    """导入ThroughputOptimizer"""
    return import_streaming_module('src.streaming.optimization.throughput_optimizer', 'ThroughputOptimizer')


def import_in_memory_stream():
    """导入InMemoryStream"""
    return import_streaming_module('src.streaming.data.in_memory_stream', 'InMemoryStream')


def import_streaming_optimizer():
    """导入StreamingOptimizer"""
    return import_streaming_module('src.streaming.data.streaming_optimizer', 'StreamingOptimizer')


def import_data_pipeline():
    """导入DataPipeline"""
    return import_streaming_module('src.streaming.core.data_pipeline', 'DataPipeline')


def import_aggregator():
    """导入RealTimeAggregator和WindowedData"""
    RealTimeAggregator = import_streaming_module('src.streaming.core.aggregator', 'RealTimeAggregator')
    WindowedData = import_streaming_module('src.streaming.core.aggregator', 'WindowedData')
    return {'RealTimeAggregator': RealTimeAggregator, 'WindowedData': WindowedData}


def import_event_processor():
    """导入EventProcessor"""
    return import_streaming_module('src.streaming.core.event_processor', 'EventProcessor')


def import_state_manager():
    """导入StateManager和StreamState"""
    StateManager = import_streaming_module('src.streaming.core.state_manager', 'StateManager')
    StreamState = import_streaming_module('src.streaming.core.state_manager', 'StreamState')
    return {'StateManager': StateManager, 'StreamState': StreamState}


def import_stream_topology():
    """导入StreamTopology"""
    return import_streaming_module('src.streaming.core.stream_engine', 'StreamTopology')


def import_create_stream_engine():
    """导入create_stream_engine函数"""
    return import_streaming_module('src.streaming.core.stream_engine', 'create_stream_engine')


def import_pipeline_rule():
    """导入PipelineRule和PipelineStage"""
    PipelineRule = import_streaming_module('src.streaming.core.data_pipeline', 'PipelineRule')
    PipelineStage = import_streaming_module('src.streaming.core.data_pipeline', 'PipelineStage')
    result = {}
    if PipelineRule:
        result['PipelineRule'] = PipelineRule
    if PipelineStage:
        result['PipelineStage'] = PipelineStage
    return result


def import_processing_result():
    """导入StreamProcessingResult和ProcessingStatus"""
    StreamProcessingResult = import_streaming_module('src.streaming.core.base_processor', 'StreamProcessingResult')
    ProcessingStatus = import_streaming_module('src.streaming.core.base_processor', 'ProcessingStatus')
    return {'StreamProcessingResult': StreamProcessingResult, 'ProcessingStatus': ProcessingStatus}


def import_stream_component_factory():
    """导入StreamComponentFactory"""
    return import_streaming_module('src.streaming.engine.stream_components', 'StreamComponentFactory')


def import_engine_component_factory():
    """导入EngineComponentFactory"""
    return import_streaming_module('src.streaming.engine.engine_components', 'EngineComponentFactory')


def import_live_component_factory():
    """导入LiveComponentFactory"""
    return import_streaming_module('src.streaming.engine.live_components', 'LiveComponentFactory')


def import_streaming_exceptions():
    """导入流处理异常类"""
    exceptions = {}
    exception_classes = [
        'StreamingException', 'StreamProcessingError', 'StreamTimeoutError',
        'StreamValidationError', 'StreamConfigurationError', 'StreamConnectionError',
        'StreamDataError', 'StreamStateError', 'StreamResourceError',
        'StreamSecurityError', 'StreamCompatibilityError'
    ]
    for exc_name in exception_classes:
        exc_class = import_streaming_module('src.streaming.core.exceptions', exc_name)
        if exc_class:
            exceptions[exc_name] = exc_class
    return exceptions


def import_all_streaming_modules():
    """导入所有流处理模块（用于测试）"""
    modules = {}

    # 核心模块
    modules['StreamProcessor'] = import_stream_processor()
    modules['DataProcessor'] = import_data_processor()
    modules['RealTimeAnalyzer'] = import_realtime_analyzer()
    modules['StreamProcessingEngine'] = import_stream_engine()
    modules['DataPipeline'] = import_data_pipeline()
    modules['EventProcessor'] = import_event_processor()
    modules['StreamProcessorBase'] = import_base_processor()
    modules['DataStreamProcessor'] = import_data_stream_processor()

    # 聚合器和状态管理
    aggregator_dict = import_aggregator()
    modules.update(aggregator_dict)

    state_dict = import_state_manager()
    modules.update(state_dict)

    # 流模型
    StreamEvent, StreamEventType = import_stream_models()
    modules['StreamEvent'] = StreamEvent
    modules['StreamEventType'] = StreamEventType

    # 引擎组件
    modules['StreamComponentFactory'] = import_stream_component_factory()
    modules['EngineComponentFactory'] = import_engine_component_factory()
    modules['LiveComponentFactory'] = import_live_component_factory()
    modules['RealtimeComponentFactory'] = import_realtime_component_factory()

    # 优化器
    modules['MemoryOptimizer'] = import_memory_optimizer()
    modules['ThroughputOptimizer'] = import_throughput_optimizer()
    modules['PerformanceOptimizer'] = import_performance_optimizer()
    modules['StreamingOptimizer'] = import_streaming_optimizer()

    # 数据流
    modules['InMemoryStream'] = import_in_memory_stream()

    # 异常类
    exceptions = import_streaming_exceptions()
    modules.update(exceptions)

    return modules


# pytest fixture: 确保模块可用
@pytest.fixture(scope="session", autouse=True)
def ensure_streaming_paths():
    """确保流处理模块路径在sys.path中"""
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    if SRC_PATH not in sys.path:
        sys.path.insert(0, SRC_PATH)
    yield
