import logging
from typing import Any, TYPE_CHECKING

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

# 类型检查时预导入所有类型，避免循环导入问题
if TYPE_CHECKING:
    from .core.types import (
        AuditEventParams, UserCreationParams, AccessCheckParams,
        PolicyCreationParams, QueryFilterParams, ConfigOperationParams,
        AuthenticationParams, ReportGenerationParams, EncryptionParams,
        HealthCheckParams, EventType, EventSeverity
    )
    from .config.security_config import SecurityConfigManager, AuditConfigManager
    from .auth.role_manager import RoleManager, Role, UserRole, Permission, RoleDefinition
    from .access.permission_checker import PermissionChecker, AccessRequest, AccessResult, AccessDecision
    from .auth.session_manager import SessionManager, UserSession
    from .auth.user_manager import UserManager, PermissionManager
    from .audit.audit_manager import AuditManager
    from .audit.audit_events import AuditEventManager, AuditEventBuilder, AuditEventFilter, AuditEvent, AuditEventType, AuditSeverity
    from .audit.audit_storage import AuditStorageManager
    from .audit.audit_rules import AuditRuleEngine, AuditRule, RuleCondition, RuleAction, RuleConditionType, AuditRuleTemplates
    from .audit.audit_reporting import AuditReportGenerator, ComplianceReport
    from .crypto.algorithms import EncryptionAlgorithm, AESGCMAlgorithm, AESCBCAlgorithm, RSAOAEPAlgorithm, ChaCha20Algorithm, EncryptionKey, EncryptionResult
    from .crypto.key_management import KeyManager
    from .access.access_control import AccessControlManager
    from .data_protection_service import DataProtectionService
    from .components.audit_component import AuditComponent
    from .components.auth_component import AuthComponent
    from .components.encrypt_component import EncryptComponent
    from .components.policy_component import PolicyComponent
    from .components.base_security_component import BaseSecurityComponent
    from .components.security_component import SecurityComponent


# 懒加载导入函数
def _lazy_import(module_name: str, class_name: str) -> Any:
    """懒加载导入函数"""
    def _getattr(name: str) -> Any:
        if name == class_name:
            try:
                module = __import__(f"src.infrastructure.security.{module_name}", fromlist=[class_name])
                return getattr(module, class_name)
            except ImportError as e:
                logging.warning(f"懒加载导入失败: {module_name}.{class_name} - {e}")
                raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    return _getattr


# 创建懒加载的模块属性
_lazy_imports = {
    # ========== 核心类型 ==========
    'UserRole': _lazy_import('core.types', 'UserRole'),
    'Permission': _lazy_import('core.types', 'Permission'),
    'User': _lazy_import('core.types', 'User'),
    'UserSession': _lazy_import('core.types', 'UserSession'),
    'AccessPolicy': _lazy_import('core.types', 'AccessPolicy'),
    'EventType': _lazy_import('core.types', 'EventType'),
    'EventSeverity': _lazy_import('core.types', 'EventSeverity'),

    # 参数对象
    'AuditEventParams': _lazy_import('core.types', 'AuditEventParams'),
    'UserCreationParams': _lazy_import('core.types', 'UserCreationParams'),
    'AccessCheckParams': _lazy_import('core.types', 'AccessCheckParams'),
    'PolicyCreationParams': _lazy_import('core.types', 'PolicyCreationParams'),
    'QueryFilterParams': _lazy_import('core.types', 'QueryFilterParams'),
    'ConfigOperationParams': _lazy_import('core.types', 'ConfigOperationParams'),
    'AuthenticationParams': _lazy_import('core.types', 'AuthenticationParams'),
    'ReportGenerationParams': _lazy_import('core.types', 'ReportGenerationParams'),
    'EncryptionParams': _lazy_import('core.types', 'EncryptionParams'),
    'HealthCheckParams': _lazy_import('core.types', 'HealthCheckParams'),

    # ========== 认证模块 ==========
    'UserManager': _lazy_import('auth.user_manager', 'UserManager'),
    'PermissionManager': _lazy_import('auth.user_manager', 'PermissionManager'),
    'SessionManager': _lazy_import('auth.session_manager', 'SessionManager'),
    'RoleManager': _lazy_import('auth.role_manager', 'RoleManager'),
    'Role': _lazy_import('auth.role_manager', 'Role'),
    'RoleDefinition': _lazy_import('auth.role_manager', 'RoleDefinition'),

    # ========== 访问控制模块 ==========
    'PermissionChecker': _lazy_import('access.permission_checker', 'PermissionChecker'),
    'AccessRequest': _lazy_import('access.permission_checker', 'AccessRequest'),
    'AccessResult': _lazy_import('access.permission_checker', 'AccessResult'),
    'AccessDecision': _lazy_import('access.permission_checker', 'AccessDecision'),
    'PolicyManager': _lazy_import('access.policy_manager', 'PolicyManager'),
    'AccessControlManager': _lazy_import('access.access_control', 'AccessControlManager'),

    # ========== 审计模块 ==========
    'AuditManager': _lazy_import('audit.audit_manager', 'AuditManager'),
    'AuditEventManager': _lazy_import('audit.audit_events', 'AuditEventManager'),
    'AuditEventBuilder': _lazy_import('audit.audit_events', 'AuditEventBuilder'),
    'AuditEventFilter': _lazy_import('audit.audit_events', 'AuditEventFilter'),
    'AuditEvent': _lazy_import('audit.audit_events', 'AuditEvent'),
    'AuditEventType': _lazy_import('audit.audit_events', 'AuditEventType'),
    'AuditSeverity': _lazy_import('audit.audit_events', 'AuditSeverity'),
    'AuditStorageManager': _lazy_import('audit.audit_storage', 'AuditStorageManager'),
    'AuditRuleEngine': _lazy_import('audit.audit_rules', 'AuditRuleEngine'),
    'AuditRule': _lazy_import('audit.audit_rules', 'AuditRule'),
    'RuleCondition': _lazy_import('audit.audit_rules', 'RuleCondition'),
    'RuleAction': _lazy_import('audit.audit_rules', 'RuleAction'),
    'RuleConditionType': _lazy_import('audit.audit_rules', 'RuleConditionType'),
    'AuditRuleTemplates': _lazy_import('audit.audit_rules', 'AuditRuleTemplates'),
    'AuditReportGenerator': _lazy_import('audit.audit_reporting', 'AuditReportGenerator'),
    'ComplianceReport': _lazy_import('audit.audit_reporting', 'ComplianceReport'),

    # ========== 加密模块 ==========
    'DataEncryptionManager': _lazy_import('crypto.encryption', 'DataEncryptionManager'),
    'KeyManager': _lazy_import('crypto.key_management', 'KeyManager'),
    'EncryptionAlgorithm': _lazy_import('crypto.algorithms', 'EncryptionAlgorithm'),
    'AESGCMAlgorithm': _lazy_import('crypto.algorithms', 'AESGCMAlgorithm'),
    'AESCBCAlgorithm': _lazy_import('crypto.algorithms', 'AESCBCAlgorithm'),
    'RSAOAEPAlgorithm': _lazy_import('crypto.algorithms', 'RSAOAEPAlgorithm'),
    'ChaCha20Algorithm': _lazy_import('crypto.algorithms', 'ChaCha20Algorithm'),
    'EncryptionKey': _lazy_import('crypto.algorithms', 'EncryptionKey'),
    'EncryptionResult': _lazy_import('crypto.algorithms', 'EncryptionResult'),

    # ========== 配置模块 ==========
    'SecurityConfigManager': _lazy_import('config.security_config', 'SecurityConfigManager'),
    'AuditConfigManager': _lazy_import('config.security_config', 'AuditConfigManager'),

    # ========== 插件系统 ==========
    'PluginManager': _lazy_import('plugins.plugin_system', 'PluginManager'),

    # ========== 过滤器模块 ==========
    'IEventFilter': _lazy_import('filters.event_filters', 'IEventFilter'),
    'IEventFilterComponent': _lazy_import('filters.event_filters', 'IEventFilterComponent'),
    'EventTypeFilter': _lazy_import('filters.event_filters', 'EventTypeFilter'),
    'SensitiveDataFilter': _lazy_import('filters.event_filters', 'SensitiveDataFilter'),
    'PatternFilter': _lazy_import('filters.event_filters', 'PatternFilter'),
    'CompositeFilter': _lazy_import('filters.event_filters', 'CompositeFilter'),
    'FilterType': _lazy_import('filters.event_filters', 'FilterType'),

    # ========== 从core层迁移的安全模块 ==========
    # JWT认证
    'JWTAuth': _lazy_import('auth.jwt_auth_core', 'JWTAuth'),
    'JWTConfig': _lazy_import('auth.jwt_auth_core', 'JWTConfig'),
    'TokenPayload': _lazy_import('auth.jwt_auth_core', 'TokenPayload'),
    'require_auth': _lazy_import('auth.jwt_auth_core', 'require_auth'),
    'create_token': _lazy_import('auth.jwt_auth_core', 'create_token'),
    'verify_token': _lazy_import('auth.jwt_auth_core', 'verify_token'),

    # 日志脱敏
    'LogSanitizer': _lazy_import('filters.log_sanitizer', 'LogSanitizer'),
    'SanitizerConfig': _lazy_import('filters.log_sanitizer', 'SanitizerConfig'),
    'sanitize_log_message': _lazy_import('filters.log_sanitizer', 'sanitize_log_message'),
    'SensitivePattern': _lazy_import('filters.log_sanitizer', 'SensitivePattern'),

    # 安全HTTP头
    'SecurityHeadersMiddleware': _lazy_import('components.security_headers_core', 'SecurityHeadersMiddleware'),
    'SecurityConfig': _lazy_import('components.security_headers_core', 'SecurityConfig'),
    'create_security_headers': _lazy_import('components.security_headers_core', 'create_security_headers'),

    # ========== 监控模块 ==========
    'PerformanceMonitor': _lazy_import('monitoring.performance_monitor', 'PerformanceMonitor'),

    # ========== 核心组件 ==========
    'BaseSecurityComponent': _lazy_import('core.base', 'BaseSecurityComponent'),
    'ISecurityComponent': _lazy_import('core.base', 'ISecurityComponent'),
    'IAuthManager': _lazy_import('core.base', 'IAuthManager'),
    'IEncryptor': _lazy_import('core.base', 'IEncryptor'),
    'IAuditor': _lazy_import('core.base', 'IAuditor'),
    'SecurityService': _lazy_import('services.security', 'SecurityService'),
    'EncryptionManager': _lazy_import('services.encryption_manager', 'EncryptionManager'),
}


def __getattr__(name: str) -> Any:
    """实现懒加载的__getattr__"""
    if name in _lazy_imports:
        importer = _lazy_imports[name]
        return importer(name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# 保留基础导入（这些不是懒加载的）
# 注意：所有组件现在都通过懒加载机制按需导入，提高启动性能

# 注意：DataProtectionService 和所有组件现在都通过懒加载机制按需导入

__all__ = [
    # 重构后的组件和参数对象
    'AuditEventParams', 'UserCreationParams', 'AccessCheckParams',
    'PolicyCreationParams', 'QueryFilterParams', 'ConfigOperationParams',
    'AuthenticationParams', 'ReportGenerationParams', 'EncryptionParams',
    'HealthCheckParams', 'EventType', 'EventSeverity',

    # 审计事件管理
    'AuditEventManager', 'AuditEventBuilder', 'AuditEventFilter',
    'AuditEvent', 'AuditEventType', 'AuditSeverity',

    # 审计存储管理
    'AuditStorageManager',

    # 审计规则引擎
    'AuditRuleEngine', 'AuditRule', 'RuleCondition', 'RuleAction',
    'RuleConditionType', 'AuditRuleTemplates',

    # 审计报告生成
    'AuditReportGenerator', 'ComplianceReport',

    # 管理器组件
    'SecurityConfigManager', 'AuditConfigManager',
    'UserManager', 'PermissionManager',
    'AuditManager', 'RealtimeAuditMonitor',
    'RoleManager', 'PermissionChecker', 'SessionManager',

    # 加密组件
    'EncryptionAlgorithm', 'AESGCMAlgorithm', 'AESCBCAlgorithm',
    'RSAOAEPAlgorithm', 'ChaCha20Algorithm', 'EncryptionKey', 'EncryptionResult',
    'KeyManager',

    # 重构后的管理器
    'RefactoredAccessControlManager', 'RefactoredAuditLoggingManager', 'RefactoredDataEncryptionManager',

    # 访问控制组件
    'Role', 'UserRole', 'Permission', 'RoleDefinition',
    'AccessRequest', 'AccessResult', 'AccessDecision',
    'UserSession',

    # 原始管理器（向后兼容）
    'AccessControlManager', 'AuditLoggingManager', 'DataEncryptionManager',
    'DataProtectionService',

    # 组件模块
    'SecurityComponent', 'AuditComponent', 'AuthComponent', 'EncryptComponent', 'PolicyComponent',
    'BaseSecurityComponent', 'SecurityService', 'EncryptionManager',

    # 从core层迁移的安全模块
    'JWTAuth', 'JWTConfig', 'TokenPayload', 'require_auth', 'create_token', 'verify_token',
    'LogSanitizer', 'SanitizerConfig', 'sanitize_log_message', 'SensitivePattern',
    'SecurityHeadersMiddleware', 'SecurityConfig', 'create_security_headers',
]
