"""
实时交易模块（别名模块）
"""

try:
    from .core.live_trading import LiveTradingEngine
except ImportError:
    # 提供基础实现
    class LiveTradingEngine:
        pass

__all__ = ['LiveTradingEngine']

