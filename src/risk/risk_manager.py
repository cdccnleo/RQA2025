"""
风险管理器模块（别名模块）
提供向后兼容的导入路径

实际实现在 models/risk_manager.py 中
"""

try:
    from .models.risk_manager import RiskManager
except ImportError:
    # 提供基础实现
    class RiskManager:
        pass

__all__ = ['RiskManager']

