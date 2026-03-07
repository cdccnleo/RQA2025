"""
AI Performance Optimizer - 重构后的组件化实现

本模块将原 ai_performance_optimizer.py (1,118行) 拆分为4个职责单一的组件。

重构日期: 2025-10-25
重构原因: 消除超长函数，提升可维护性
"""

# 导入数据模型（保持兼容）
from .models import (
    OptimizationMode,
    PerformanceMetric,
    PerformanceData,
    OptimizationAction,
    PerformanceInsight
)

# 导入组件（新的实现）
from .performance_analyzer import PerformanceAnalyzer
from .optimization_strategy import OptimizationStrategy
from .reactive_optimizer import ReactiveOptimizer
from .performance_monitor import PerformanceMonitorService

# 导入协调器（向后兼容）
from .ai_performance_optimizer import AIPerformanceOptimizer

# 向后兼容的别名
PerformanceOptimizer = AIPerformanceOptimizer  # 保持原API
IntelligentPerformanceMonitor = AIPerformanceOptimizer  # 统一到协调器

# 导出
__all__ = [
    # 数据模型
    'OptimizationMode',
    'PerformanceMetric',
    'PerformanceData',
    'OptimizationAction',
    'PerformanceInsight',
    
    # 组件
    'PerformanceAnalyzer',
    'OptimizationStrategy',
    'ReactiveOptimizer',
    'PerformanceMonitorService',
    
    # 协调器
    'AIPerformanceOptimizer',
    
    # 向后兼容别名
    'PerformanceOptimizer',
    'IntelligentPerformanceMonitor',
]

