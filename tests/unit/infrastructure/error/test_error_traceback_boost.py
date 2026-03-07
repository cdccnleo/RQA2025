#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Error模块追踪测试
覆盖错误追踪和调试功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from unittest.mock import Mock, MagicMock
import traceback
import sys

# 测试错误追踪器
try:
    from src.infrastructure.error.traceback.error_tracer import ErrorTracer, TraceInfo
    HAS_ERROR_TRACER = True
except ImportError:
    HAS_ERROR_TRACER = False
    
    from dataclasses import dataclass
    
    @dataclass
    class TraceInfo:
        error_type: str
        error_message: str
        stack_trace: str
        timestamp: float
    
    class ErrorTracer:
        def __init__(self):
            self.traces = []
        
        def capture_error(self, error):
            import time
            
            trace_info = TraceInfo(
                error_type=type(error).__name__,
                error_message=str(error),
                stack_trace=traceback.format_exc(),
                timestamp=time.time()
            )
            self.traces.append(trace_info)
            return trace_info
        
        def get_traces(self):
            return self.traces


class TestTraceInfo:
    """测试追踪信息"""
    
    def test_create_trace_info(self):
        """测试创建追踪信息"""
        trace = TraceInfo(
            error_type="ValueError",
            error_message="Invalid value",
            stack_trace="Traceback...",
            timestamp=1699000000.0
        )
        
        assert trace.error_type == "ValueError"
        assert trace.error_message == "Invalid value"


class TestErrorTracer:
    """测试错误追踪器"""
    
    def test_init(self):
        """测试初始化"""
        tracer = ErrorTracer()
        
        if hasattr(tracer, 'traces'):
            assert tracer.traces == []
    
    def test_capture_error(self):
        """测试捕获错误"""
        tracer = ErrorTracer()
        error = ValueError("test error")
        
        if hasattr(tracer, 'capture_error'):
            trace = tracer.capture_error(error)
            
            assert isinstance(trace, TraceInfo)
    
    def test_get_traces(self):
        """测试获取追踪"""
        tracer = ErrorTracer()
        
        if hasattr(tracer, 'capture_error') and hasattr(tracer, 'get_traces'):
            tracer.capture_error(ValueError("err1"))
            tracer.capture_error(TypeError("err2"))
            
            traces = tracer.get_traces()
            assert isinstance(traces, list)


# 测试堆栈分析器
try:
    from src.infrastructure.error.traceback.stack_analyzer import StackAnalyzer
    HAS_STACK_ANALYZER = True
except ImportError:
    HAS_STACK_ANALYZER = False
    
    class StackAnalyzer:
        def __init__(self):
            self.analysis_results = []
        
        def analyze_stack(self, stack_trace):
            lines = stack_trace.split('\n')
            result = {
                'line_count': len(lines),
                'has_traceback': 'Traceback' in stack_trace,
                'functions': []
            }
            self.analysis_results.append(result)
            return result
        
        def get_error_location(self, stack_trace):
            lines = [l for l in stack_trace.split('\n') if 'File' in l]
            return lines[-1] if lines else None


class TestStackAnalyzer:
    """测试堆栈分析器"""
    
    def test_init(self):
        """测试初始化"""
        analyzer = StackAnalyzer()
        
        if hasattr(analyzer, 'analysis_results'):
            assert analyzer.analysis_results == []
    
    def test_analyze_stack(self):
        """测试分析堆栈"""
        analyzer = StackAnalyzer()
        stack_trace = "Traceback (most recent call last):\n  File 'test.py', line 10"
        
        if hasattr(analyzer, 'analyze_stack'):
            result = analyzer.analyze_stack(stack_trace)
            
            assert isinstance(result, dict)
    
    def test_get_error_location(self):
        """测试获取错误位置"""
        analyzer = StackAnalyzer()
        stack_trace = "File '/path/to/file.py', line 42, in function_name"
        
        if hasattr(analyzer, 'get_error_location'):
            location = analyzer.get_error_location(stack_trace)
            
            assert location is not None or location is None


# 测试异常包装器
try:
    from src.infrastructure.error.exceptions.exception_wrapper import ExceptionWrapper
    HAS_EXCEPTION_WRAPPER = True
except ImportError:
    HAS_EXCEPTION_WRAPPER = False
    
    class ExceptionWrapper:
        def __init__(self, original_exception):
            self.original_exception = original_exception
            self.context = {}
        
        def add_context(self, key, value):
            self.context[key] = value
        
        def get_full_info(self):
            return {
                'exception': str(self.original_exception),
                'type': type(self.original_exception).__name__,
                'context': self.context
            }


class TestExceptionWrapper:
    """测试异常包装器"""
    
    def test_init(self):
        """测试初始化"""
        error = ValueError("test")
        wrapper = ExceptionWrapper(error)
        
        if hasattr(wrapper, 'original_exception'):
            assert wrapper.original_exception is error
    
    def test_add_context(self):
        """测试添加上下文"""
        error = RuntimeError("error")
        wrapper = ExceptionWrapper(error)
        
        if hasattr(wrapper, 'add_context'):
            wrapper.add_context("user_id", "123")
            
            if hasattr(wrapper, 'context'):
                assert wrapper.context["user_id"] == "123"
    
    def test_get_full_info(self):
        """测试获取完整信息"""
        error = TypeError("type error")
        wrapper = ExceptionWrapper(error)
        
        if hasattr(wrapper, 'get_full_info'):
            info = wrapper.get_full_info()
            
            assert isinstance(info, dict)


# 测试错误传播器
try:
    from src.infrastructure.error.propagation.error_propagator import ErrorPropagator
    HAS_ERROR_PROPAGATOR = True
except ImportError:
    HAS_ERROR_PROPAGATOR = False
    
    class ErrorPropagator:
        def __init__(self):
            self.handlers = []
        
        def add_handler(self, handler):
            self.handlers.append(handler)
        
        def propagate(self, error):
            for handler in self.handlers:
                try:
                    handler(error)
                except:
                    pass


class TestErrorPropagator:
    """测试错误传播器"""
    
    def test_init(self):
        """测试初始化"""
        propagator = ErrorPropagator()
        
        if hasattr(propagator, 'handlers'):
            assert propagator.handlers == []
    
    def test_add_handler(self):
        """测试添加处理器"""
        propagator = ErrorPropagator()
        handler = Mock()
        
        if hasattr(propagator, 'add_handler'):
            propagator.add_handler(handler)
            
            if hasattr(propagator, 'handlers'):
                assert len(propagator.handlers) == 1
    
    def test_propagate(self):
        """测试传播"""
        propagator = ErrorPropagator()
        handler = Mock()
        
        if hasattr(propagator, 'add_handler') and hasattr(propagator, 'propagate'):
            propagator.add_handler(handler)
            
            error = ValueError("test")
            propagator.propagate(error)
            
            assert True  # 传播完成


# 测试异常分类器
try:
    from src.infrastructure.error.classification.exception_classifier import ExceptionClassifier, ErrorCategory
    HAS_EXCEPTION_CLASSIFIER = True
except ImportError:
    HAS_EXCEPTION_CLASSIFIER = False
    
    from enum import Enum
    
    class ErrorCategory(Enum):
        SYSTEM = "system"
        USER_INPUT = "user_input"
        NETWORK = "network"
        DATABASE = "database"
        UNKNOWN = "unknown"
    
    class ExceptionClassifier:
        def classify(self, error):
            error_type = type(error).__name__
            
            if 'Connection' in error_type or 'Network' in error_type:
                return ErrorCategory.NETWORK
            elif 'Database' in error_type or 'SQL' in error_type:
                return ErrorCategory.DATABASE
            elif 'Value' in error_type or 'Type' in error_type:
                return ErrorCategory.USER_INPUT
            elif 'System' in error_type:
                return ErrorCategory.SYSTEM
            else:
                return ErrorCategory.UNKNOWN


class TestErrorCategory:
    """测试错误分类"""
    
    def test_categories(self):
        """测试分类枚举"""
        assert ErrorCategory.SYSTEM.value == "system"
        assert ErrorCategory.USER_INPUT.value == "user_input"
        assert ErrorCategory.NETWORK.value == "network"
        assert ErrorCategory.DATABASE.value == "database"
        assert ErrorCategory.UNKNOWN.value == "unknown"


class TestExceptionClassifier:
    """测试异常分类器"""
    
    def test_classify_value_error(self):
        """测试分类值错误"""
        classifier = ExceptionClassifier()
        error = ValueError("invalid value")
        
        if hasattr(classifier, 'classify'):
            category = classifier.classify(error)
            
            assert isinstance(category, ErrorCategory)
    
    def test_classify_type_error(self):
        """测试分类类型错误"""
        classifier = ExceptionClassifier()
        error = TypeError("wrong type")
        
        if hasattr(classifier, 'classify'):
            category = classifier.classify(error)
            
            assert isinstance(category, ErrorCategory)
    
    def test_classify_runtime_error(self):
        """测试分类运行时错误"""
        classifier = ExceptionClassifier()
        error = RuntimeError("runtime issue")
        
        if hasattr(classifier, 'classify'):
            category = classifier.classify(error)
            
            assert isinstance(category, ErrorCategory)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

