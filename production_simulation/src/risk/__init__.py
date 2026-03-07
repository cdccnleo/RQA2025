"""
风险管理模块
"""

import logging
logger = logging.getLogger(__name__)

# 从models.risk_manager导入
try:
    from .models.risk_manager import RiskManager, RiskLevel, RiskCheck, RiskManagerConfig, RiskManagerStatus
except ImportError:
    class RiskManager:
        pass
    class RiskLevel:
        pass
    class RiskCheck:
        pass
    class RiskManagerConfig:
        pass
    class RiskManagerStatus:
        pass

# 导入monitor相关组件
try:
    from .monitor.realtime_risk_monitor import RealtimeRiskMonitor
except Exception as e:
    logger.warning(f"Failed to import RealtimeRiskMonitor: {e}")

    class RealtimeRiskMonitor:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("RealtimeRiskMonitor is not available") from e

try:
    from .monitor.real_time_monitor import RealTimeMonitor
except Exception as e:
    logger.warning(f"Failed to import RealTimeMonitor: {e}")

    class RealTimeMonitor(RealtimeRiskMonitor):
        pass

# 导入compliance相关组件
try:
    from .compliance.cross_border_compliance_manager import CrossBorderComplianceManager
except ImportError:
    logger.warning("Failed to import CrossBorderComplianceManager")
    class CrossBorderComplianceManager:
        pass

__all__ = [
    'RiskManager', 'RiskLevel', 'RiskCheck', 'RiskManagerConfig', 'RiskManagerStatus',
    'RealtimeRiskMonitor', 'RealTimeMonitor', 'CrossBorderComplianceManager'
]
