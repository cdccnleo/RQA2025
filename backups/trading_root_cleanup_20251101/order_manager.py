"""
订单管理器模块（别名模块）
提供向后兼容的导入路径

实际实现在 execution/order_manager.py 中
"""

try:
    from .execution.order_manager import OrderManager
except ImportError:
    # 提供基础实现
    class OrderManager:
        pass

__all__ = ['OrderManager']

