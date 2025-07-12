"""RQA2025 FPGA加速层 - 硬件加速关键算法

核心组件:
- accelerator: 加速器核心实现
- manager: 设备状态管理
- risk_engine: 硬件风控引擎
- order_optimizer: 订单优化加速

使用示例:
    from src.fpga import FpgaManager
    from src.fpga.risk_engine import FpgaRiskChecker

    # 初始化FPGA设备
    fpga = FpgaManager()
    if fpga.initialize():
        # 使用FPGA加速风控检查
        risk_checker = FpgaRiskChecker()
        result = risk_checker.check(order)
    else:
        # 降级到软件实现
        from src.trading.risk import SoftwareRiskChecker
        risk_checker = SoftwareRiskChecker()

主要功能:
- 情感分析硬件加速
- 风控规则硬件实现
- 订单簿优化加速
- 自动降级管理

注意事项:
1. 必须提供软件降级路径
2. 设备状态需要实时监控
3. 初始化失败应优雅降级
4. 保持驱动版本与硬件兼容

版本历史:
- v1.0 (2024-05-10): 初始版本
- v1.1 (2024-06-15): 添加多设备支持
"""

from .manager import FpgaManager
from .risk_engine import FpgaRiskChecker
from .order_optimizer import FpgaOrderOptimizer
from .sentiment import FpgaSentimentAnalyzer

__all__ = [
    'FpgaManager',
    'FpgaRiskChecker',
    'FpgaOrderOptimizer',
    'FpgaSentimentAnalyzer',
    # 子模块
    'accelerator',
    'manager',
    'risk_engine',
    'order_optimizer',
    'sentiment'
]
