import logging
"""
RQA2025 统一安全模块
提供完整的统一安全接口和实现

整合内容：
- 基础设施层安全模块 (src / infrastructure / security/)
- 数据层安全模块 (src / data / security/)
- 核心安全模块 (src / core / security/)

职责：
- 定义核心安全接口和实现
- 提供完整的认证、授权、审计、加密功能
- 支持多层次的安全管理
- 提供统一的安全API接口
"""

# 基础安全模块
from .base_security import ISecurity, SecurityLevel
from .base import BaseSecurityComponent
from .interfaces import ISecurityComponent, IAuthManager, IEncryptor, IAuditor

# 核心安全实现
from .unified_security import UnifiedSecurity, get_security, set_security
from .authentication_service import MultiFactorAuthenticationService, IAuthenticator
from .access_control import get_access_control_system, AccessControlSystem
from .audit_system import get_audit_system, AuditSystem
from .encryption_service import get_encryption_service, EncryptionService

# 安全工具和工厂
from .security_utils import SecurityUtils
from .security_factory import SecurityFactory, create_security_manager, get_security_factory_info
from .security import SecurityManager

# 数据层安全管理器
from .access_control_manager import AccessControlManager
from .audit_logging_manager import AuditLoggingManager
from .data_encryption_manager import DataEncryptionManager

# 服务模块
from .data_protection_service import DataProtectionService
# from .config_encryption_service import ConfigEncryptionService  # 暂时禁用，有语法错误
# from .web_management_service import WebManagementService  # 暂时禁用，有语法错误

# 组件模块
from .security_components import SecurityComponent
from .audit_components import AuditComponent
from .auth_components import AuthComponent
from .encrypt_components import EncryptComponent
from .policy_components import PolicyComponent

__all__ = [
    # 基础接口和实现
    'ISecurity', 'BaseSecurityComponent', 'SecurityLevel',
    'BaseSecurity', 'ISecurityComponent', 'IAuthManager', 'IEncryptor', 'IAuditor',

    # 核心安全服务
    'UnifiedSecurity', 'get_security', 'set_security',
    'MultiFactorAuthenticationService', 'IAuthenticator',
    'get_access_control_system', 'AccessControlSystem',
    'get_audit_system', 'AuditSystem',
    'get_encryption_service', 'EncryptionService',

    # 工具和工厂
    'SecurityUtils', 'SecurityFactory', 'create_security_manager', 'get_security_factory_info',
    'SecurityManager', 'SecurityFilter',

    # 数据层安全管理器
    'AccessControlManager', 'AuditLoggingManager', 'DataEncryptionManager',

    # 服务模块
    'DataProtectionService',

    # 组件模块
    'SecurityComponent', 'AuditComponent', 'AuthComponent', 'EncryptComponent', 'PolicyComponent'
]
