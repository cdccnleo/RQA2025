"""
Short Term Optimizations - 短期优化组件

本模块将原 short_term_optimizations.py (1,651行) 拆分为8个职责单一的组件。

重构日期: 2025-10-25
重构原因: 消除超大文件，分离业务和测试代码，统一目录结构
"""

# 导入数据模型
from .models import FeedbackItem, PerformanceMetric

# 导入业务组件
from .feedback_collector import FeedbackCollector
from .feedback_analyzer import FeedbackAnalyzer
from .performance_monitor import PerformanceMonitorService
from .documentation_enhancer import DocumentationEnhancer
from .testing_enhancer import TestingEnhancer
from .memory_optimizer import MemoryOptimizer

# 导入协调器
from .short_term_strategy import ShortTermStrategy

# 向后兼容的别名（保持旧API可用）
UserFeedbackCollector = FeedbackCollector  # 旧名称
PerformanceMonitor = PerformanceMonitorService  # 旧名称

# 导出所有组件
__all__ = [
    # 数据模型
    'FeedbackItem',
    'PerformanceMetric',
    
    # 业务组件
    'FeedbackCollector',
    'FeedbackAnalyzer',
    'PerformanceMonitorService',
    'DocumentationEnhancer',
    'TestingEnhancer',
    'MemoryOptimizer',
    
    # 协调器
    'ShortTermStrategy',
    
    # 向后兼容别名
    'UserFeedbackCollector',
    'PerformanceMonitor',
]

