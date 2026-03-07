"""
FPGA加速模块
提供FPGA硬件加速功能

主要类:
- FPGAManager: FPGA管理器
- FPGAAccelerator: FPGA加速器
- FPGARiskEngine: FPGA风险引擎
- FpgaOrderOptimizer: FPGA订单优化器
- FpgaSentimentAnalyzer: FPGA情感分析器
- FPGAOptimizer: FPGA优化器
- FPGAPerformanceMonitor: FPGA性能监视器
- FPGAFallbackManager: FPGA降级管理器
- FPGADashboard: FPGA仪表板
- FPGAOrderbookOptimizer: FPGA订单簿优化器

使用示例:
    from src.acceleration.fpga import FPGAManager, FPGARiskEngine

    manager = FPGAManager()
    risk_engine = FPGARiskEngine(manager)
    result = risk_engine.check_risks(order)
        """

from .fpga_manager import FPGAManager
from .fpga_accelerator import FPGAAccelerator
# from .fpga_risk_engine import FPGARiskEngine, FpgaRiskEngine
# from .fpga_order_optimizer import FpgaOrderOptimizer
# from .fpga_sentiment_analyzer import FpgaSentimentAnalyzer
# from .fpga_optimizer import FPGAOptimizer
# from .fpga_performance_monitor import FPGAPerformanceMonitor
# from .fpga_fallback_manager import FPGAFallbackManager
# from .fpga_dashboard import FPGADashboard
# from .fpga_orderbook_optimizer import FPGAOrderbookOptimizer

__all__ = [
    'FPGAManager',
    'FPGAAccelerator',
    # 'FPGARiskEngine',
    # 'FpgaRiskEngine',
    # 'FpgaOrderOptimizer',
    # 'FpgaSentimentAnalyzer',
    # 'FPGAOptimizer',
    # 'FPGAPerformanceMonitor',
    # 'FPGAFallbackManager',
    # 'FPGADashboard',
    # 'FPGAOrderbookOptimizer',
    # 添加别名以保持向后兼容
    'FpgaManager',  # 别名
    'FpgaAccelerator'  # 别名
]

# 添加别名
FpgaManager = FPGAManager
FpgaAccelerator = FPGAAccelerator
