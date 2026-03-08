import logging
#!/usr/bin/env python3
"""
RQA2025 安全层基础设施适配器

专门为安全层提供基础设施服务访问接口，
基于统一业务层适配器架构，实现安全层的特定需求。
"""

from typing import Dict, Any, Optional
from src.infrastructure.integration.unified_business_adapters import UnifiedBusinessAdapter, BusinessLayerType
from datetime import datetime, timedelta
import threading


logger = logging.getLogger(__name__)


class SecurityLayerAdapter(UnifiedBusinessAdapter):

    """安全层适配器"""

    def __init__(self):

        super().__init__(BusinessLayerType.RISK)  # 使用RISK类型作为安全层
        self._init_security_specific_services()

    def _init_security_specific_services(self):
        """初始化安全层特定的基础设施服务"""
        try:
            # 安全层特定的服务桥接器
            self._service_bridges = {
                'security_infrastructure_bridge': self._create_security_bridge()
            }

            logger.info("安全层特定服务桥接器初始化完成")

        except Exception as e:
            logger.warning(f"安全层特定服务桥接器初始化失败，使用基础服务: {e}")

    def _create_security_bridge(self):
        """创建安全层专用的基础设施桥接器"""
        # 使用完善的安全基础设施桥接器
        return SecurityInfrastructureBridge()

    def get_audit_system(self):
        """获取审计系统"""
        try:
            from src.security.audit_system import get_audit_system
            logger.info("使用审计系统")
            return get_audit_system()
        except ImportError:
            logger.warning("审计系统导入失败")
            return None

    def get_monitoring_system(self):
        """获取监控系统"""
        try:
            from src.monitoring.monitoring_system import get_monitoring_system
            logger.info("使用监控系统")
            return get_monitoring_system()
        except ImportError:
            logger.warning("监控系统导入失败")
            return None

    def get_encryption_service(self):
        """获取加密服务"""
        try:
            from src.security.encryption_service import get_encryption_service
            logger.info("使用加密服务")
            return get_encryption_service()
        except ImportError:
            logger.warning("加密服务导入失败")
            return None

    def get_access_control_system(self):
        """获取访问控制系统"""
        try:
            from src.security.access_control import get_access_control_system
            logger.info("使用访问控制系统")
            return get_access_control_system()
        except ImportError:
            logger.warning("访问控制系统导入失败")
            return None

    def audit_security_event(self, event_type: str, severity: str,


                             source_ip: Optional[str] = None, user_id: Optional[str] = None,
                             description: str = "", details: Dict[str, Any] = None):
        """审计安全事件"""
        audit_system = self.get_audit_system()
        if audit_system:
            from src.security.audit_system import SecurityLevel
            severity_map = {
                "low": SecurityLevel.LOW,
                "medium": SecurityLevel.MEDIUM,
                "high": SecurityLevel.HIGH,
                "critical": SecurityLevel.CRITICAL
            }
            severity_level = severity_map.get(severity.lower(), SecurityLevel.MEDIUM)
            audit_system.log_security_event(
                event_type, severity_level, source_ip, user_id, description, details)
            return True
        return False

    def record_security_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """记录安全指标"""
        monitoring_system = self.get_monitoring_system()
        if monitoring_system:
            monitoring_system.record_business_metric(name, value, labels)
            return True
        return False

    def check_security_status(self, ip_address: str) -> Dict[str, Any]:
        """检查安全状态"""
        audit_system = self.get_audit_system()
        if audit_system:
            return audit_system.check_security_status(ip_address)
        return {"ip_blocked": False, "failed_login_attempts": 0, "suspicious_score": 0}

    def encrypt_data(self, data: str, key_id: Optional[str] = None) -> Optional[str]:
        """加密数据"""
        encryption_service = self.get_encryption_service()
        if encryption_service:
            return encryption_service.encrypt(data, key_id)
        return None

    def decrypt_data(self, encrypted_data: str, key_id: Optional[str] = None) -> Optional[str]:
        """解密数据"""
        encryption_service = self.get_encryption_service()
        if encryption_service:
            return encryption_service.decrypt(encrypted_data, key_id)
        return None

    def check_access_permission(self, user_id: str, resource: str, action: str) -> bool:
        """检查访问权限"""
        access_control = self.get_access_control_system()
        if access_control:
            return access_control.check_permission(user_id, resource, action)
        return False

    def authenticate_user(self, user_id: str, credentials: Dict[str, Any]) -> bool:
        """用户认证"""
        access_control = self.get_access_control_system()
        if access_control:
            return access_control.authenticate(user_id, credentials)
        return False

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health_info = super().health_check()
        health_info['component'] = 'SecurityLayerAdapter'

        # 检查安全模块状态
        security_modules = {
            'audit_system': self.get_audit_system() is not None,
            'monitoring_system': self.get_monitoring_system() is not None,
            'encryption_service': self.get_encryption_service() is not None,
            'access_control_system': self.get_access_control_system() is not None
        }

        health_info['security_modules'] = security_modules

        # 检查安全事件
        audit_system = self.get_audit_system()
        if audit_system:
            security_events = audit_system.security_monitor.get_security_events(hours=1)
            health_info['recent_security_events'] = len(security_events)

            # 检查是否有未解决的严重安全事件
            critical_events = [
                event for event in security_events
                if not event.resolved and event.severity.name in ['HIGH', 'CRITICAL']
            ]
            if critical_events:
                health_info['status'] = 'critical'
                health_info['critical_security_events'] = [
                    {
                        'event_type': event.event_type,
                        'severity': event.severity.name,
                        'description': event.description,
                        'timestamp': event.timestamp.isoformat()
                    } for event in critical_events
                ]

        return health_info


class SecurityInfrastructureBridge:

    """安全层基础设施桥接器"""

    def __init__(self):

        self._services = {}
        self._fallback_services = {}
        self._init_security_services()
        self._init_fallback_services()

    def _init_security_services(self):
        """初始化安全基础设施服务"""
        # 整合统一安全模块的服务
        try:
            from src.security.unified_security import get_security
            self._services['unified_security'] = get_security()
            logger.info("已整合统一安全服务")
        except ImportError:
            logger.warning("统一安全服务不可用")

        # 整合数据安全服务（已迁移到统一安全模块）
        try:
            from src.security.services.data_access_control import AccessControlManager
            self._services['data_access_control'] = AccessControlManager()
            logger.info("已整合数据访问控制管理器")
        except ImportError:
            logger.warning("数据访问控制管理器不可用")

        try:
            from src.security.services.data_audit_manager import AuditLoggingManager
            self._services['data_audit_logging'] = AuditLoggingManager()
            logger.info("已整合数据审计日志管理器")
        except ImportError:
            logger.warning("数据审计日志管理器不可用")

        try:
            from src.security.services.data_encryption_service import DataEncryptionManager
            self._services['data_encryption'] = DataEncryptionManager()
            logger.info("已整合数据加密管理器")
        except ImportError:
            logger.warning("数据加密管理器不可用")

    def _init_fallback_services(self):
        """初始化降级服务"""
        try:
            from .fallback_services import FallbackSecurityService
            self._fallback_services['security'] = FallbackSecurityService()
        except ImportError:
            logger.warning("降级安全服务初始化失败")

    def get_service(self, service_name: str):
        """获取服务"""
        return self._services.get(service_name, self._fallback_services.get(service_name))

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            'status': 'healthy',
            'bridge_type': 'security_infrastructure_bridge',
            'services': list(self._services.keys()),
            'fallback_services': list(self._fallback_services.keys())
        }


class SecurityIntegrationManager:

    """安全层集成管理器"""

    def __init__(self):

        self.adapter = SecurityLayerAdapter()
        self._integration_cache = {}
        self._last_health_check = datetime.now()
        self._health_check_interval = timedelta(minutes=5)

    def get_security_services(self) -> Dict[str, Any]:
        """获取安全服务字典"""
        return {
            'audit_system': self.adapter.get_audit_system(),
            'monitoring_system': self.adapter.get_monitoring_system(),
            'encryption_service': self.adapter.get_encryption_service(),
            'access_control_system': self.adapter.get_access_control_system(),
            'security_adapter': self.adapter
        }

    def audit_operation(self, operation: str, user_id: str, resource: str,


                        result: str, details: Dict[str, Any] = None):
        """审计操作"""
        audit_system = self.adapter.get_audit_system()
        if audit_system:
            if operation == "login":
                audit_system.log_user_authentication(
                    user_id=user_id,
                    action="login",
                    result=result,
                    session_id=details.get("session_id") if details else None,
                    ip_address=details.get("ip_address") if details else None,
                    user_agent=details.get("user_agent") if details else None
                )
            elif operation == "trade":
                audit_system.log_trade_execution(
                    user_id=user_id,
                    trade_details=details or {},
                    result=result,
                    session_id=details.get("session_id") if details else None,
                    ip_address=details.get("ip_address") if details else None
                )
            elif operation == "order":
                audit_system.log_order_operation(
                    user_id=user_id,
                    order_type=details.get("order_type", "unknown") if details else "unknown",
                    order_details=details or {},
                    operation=details.get("operation", "unknown") if details else "unknown",
                    result=result,
                    session_id=details.get("session_id") if details else None,
                    ip_address=details.get("ip_address") if details else None
                )

    def record_security_event(self, event_type: str, severity: str,


                              source_ip: Optional[str] = None, user_id: Optional[str] = None,
                              description: str = "", details: Dict[str, Any] = None):
        """记录安全事件"""
        return self.adapter.audit_security_event(event_type, severity, source_ip, user_id, description, details)

    def check_security_clearance(self, user_id: str, resource: str, action: str) -> Dict[str, Any]:
        """检查安全许可"""
        result = {
            'authorized': False,
            'security_status': {},
            'warnings': []
        }

        # 检查访问权限
        if self.adapter.check_access_permission(user_id, resource, action):
            result['authorized'] = True
        else:
            result['warnings'].append("访问权限不足")

        # 检查安全状态
        security_status = self.adapter.check_security_status("")
        result['security_status'] = security_status

        if security_status.get('ip_blocked'):
            result['authorized'] = False
            result['warnings'].append("IP地址被封禁")

        return result

    def encrypt_sensitive_data(self, data: str, context: str = "general") -> Optional[str]:
        """加密敏感数据"""
        return self.adapter.encrypt_data(data, context)

    def decrypt_sensitive_data(self, encrypted_data: str, context: str = "general") -> Optional[str]:
        """解密敏感数据"""
        return self.adapter.decrypt_data(encrypted_data, context)

    def get_security_report(self, report_type: str = "summary") -> Dict[str, Any]:
        """生成安全报告"""
        audit_system = self.adapter.get_audit_system()
        if not audit_system:
            return {"error": "审计系统不可用"}

        if report_type == "summary":
            return audit_system.get_audit_report(days=7)
        elif report_type == "security_events":
            security_events = audit_system.security_monitor.get_security_events(hours=24)
            return {
                "total_events": len(security_events),
                "events": [
                    {
                        "event_type": event.event_type,
                        "severity": event.severity.name,
                        "description": event.description,
                        "timestamp": event.timestamp.isoformat(),
                        "resolved": event.resolved
                    } for event in security_events[-50:]  # 最近50个事件
                ]
            }
        else:
            return {"error": "未知的报告类型"}

    def perform_security_health_check(self) -> Dict[str, Any]:
        """执行安全健康检查"""
        # 检查是否需要更新缓存的健康状态
        if datetime.now() - self._last_health_check > self._health_check_interval:
            self._integration_cache['health_status'] = self.adapter.health_check()
            self._last_health_check = datetime.now()

        return self._integration_cache.get('health_status', {"status": "unknown"})


# 全局安全集成管理器实例
_security_integration_manager = None
_security_integration_manager_lock = threading.Lock()


def get_security_integration_manager() -> SecurityIntegrationManager:
    """获取全局安全集成管理器实例"""
    global _security_integration_manager

    if _security_integration_manager is None:
        with _security_integration_manager_lock:
            if _security_integration_manager is None:
                _security_integration_manager = SecurityIntegrationManager()

    return _security_integration_manager


# 便捷函数

def audit_security_operation(operation: str, user_id: str, resource: str,


                             result: str, details: Dict[str, Any] = None):
    """审计安全操作"""
    manager = get_security_integration_manager()
    manager.audit_operation(operation, user_id, resource, result, details)


def check_security_clearance(user_id: str, resource: str, action: str) -> Dict[str, Any]:
    """检查安全许可"""
    manager = get_security_integration_manager()
    return manager.check_security_clearance(user_id, resource, action)


def encrypt_data(data: str, context: str = "general") -> Optional[str]:
    """加密数据"""
    manager = get_security_integration_manager()
    return manager.encrypt_sensitive_data(data, context)


def decrypt_data(encrypted_data: str, context: str = "general") -> Optional[str]:
    """解密数据"""
    manager = get_security_integration_manager()
    return manager.decrypt_sensitive_data(encrypted_data, context)

    def get_unified_security(self):
        """获取统一安全服务"""
        try:
            from src.security.unified_security import get_security
            logger.info("使用统一安全服务")
            return get_security()
        except ImportError:
            logger.warning("统一安全服务导入失败")
            return None

    def get_authentication_service(self):
        """获取认证服务"""
        try:
            from src.security.authentication_service import MultiFactorAuthenticationService
            logger.info("使用认证服务")
            return MultiFactorAuthenticationService()
        except ImportError:
            logger.warning("认证服务导入失败")
            return None
