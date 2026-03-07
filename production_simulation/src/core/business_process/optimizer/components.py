"""
优化器组件统一导出模块
"""

try:
    from .components.process_monitor import ProcessMonitor, ProcessMetrics
    from .components.recommendation_generator import RecommendationGenerator, Recommendation
    from .components.performance_analyzer import PerformanceAnalyzer
except ImportError:
    # 提供基础实现
    class ProcessMonitor:
        pass
    
    class ProcessMetrics:
        pass
    
    class RecommendationGenerator:
        pass
    
    class Recommendation:
        pass
    
    class PerformanceAnalyzer:
        pass

__all__ = [
    'ProcessMonitor',
    'ProcessMetrics',
    'RecommendationGenerator',
    'Recommendation',
    'PerformanceAnalyzer'
]

