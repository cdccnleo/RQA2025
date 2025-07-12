"""
配置模块接口定义
版本更新记录：
2024-04-02 v3.6.0 - 重构接口定义
    - 统一接口命名规范
    - 新增版本控制接口
2024-04-03 v3.6.1 - 接口完善
    - 添加IVersionComparator详细定义
    - 增强IConfigEventSystem文档
    - 补充ISecurityValidator方法
    - 更新IVersionManager类型提示
"""

from .config_loader import IConfigLoader
from .config_validator import IConfigValidator
from .version_controller import IVersionManager as IVersionController
from ..exceptions import (
    SecurityError,
    ConfigValidationError, 
    ConfigLoadError,
    TradingConfigError
)
from .version_manager import IVersionManager
from src.infrastructure.utils.audit import audit_log
from .diff_service import IVersionComparator
from .event_system import IConfigEventSystem
from .security_service import ISecurityValidator
from .version_controller import IVersionManager
from .version_storage import IVersionStorage

__all__ = [
    'IConfigLoader',
    'IConfigValidator',
    'IVersionComparator',
    'IConfigEventSystem',
    'ISecurityValidator',
    'IVersionManager',
    'IVersionStorage'
]
