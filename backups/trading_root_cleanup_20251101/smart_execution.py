"""
智能执行模块（别名模块）
"""

try:
    from .execution.smart_execution import SmartExecutor, SmartExecutionEngine
except ImportError:
    # 提供基础实现
    class SmartExecutor:
        pass
    
    SmartExecutionEngine = SmartExecutor

__all__ = ['SmartExecutor', 'SmartExecutionEngine']

