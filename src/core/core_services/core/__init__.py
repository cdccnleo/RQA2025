"""
核心服务组件

提供基础业务服务：
- BusinessService: 业务服务
- DatabaseService: 数据库服务
- StrategyManager: 策略管理器
"""

try:
    from .business_service import BusinessService
    from .database_service import DatabaseService
    from .strategy_manager import StrategyManager
except ImportError:
    # 如果导入失败，提供基础实现
    class BusinessService:
        pass
    class DatabaseService:
        pass
    class StrategyManager:
        pass

__all__ = [
    "BusinessService",
    "DatabaseService",
    "StrategyManager"
]
