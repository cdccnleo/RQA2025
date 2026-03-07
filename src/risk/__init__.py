"""
风险管理模块
"""

import logging
logger = logging.getLogger(__name__)

# 导入子模块
try:
    from . import models
except ImportError:
    pass

try:
    from . import monitor
except ImportError:
    pass

try:
    from . import alert
except ImportError:
    pass

try:
    from . import checker
except ImportError:
    pass

try:
    from . import compliance
except ImportError:
    pass

try:
    from . import analysis
except ImportError:
    pass

try:
    from . import infrastructure
except ImportError:
    pass

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
