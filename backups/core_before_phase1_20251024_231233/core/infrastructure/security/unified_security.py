"""
统一安全模块
提供统一的安全接口和实现
"""

from typing import Any, Dict, Optional, List
from datetime import datetime, timedelta
import threading

from .base_security import BaseSecurityComponent, SecurityLevel


class UnifiedSecurity(BaseSecurityComponent):

    """
unified_security - 安全管理

职责说明：
负责系统安全、权限控制、加密解密和安全审计

核心职责：
- 用户认证和授权
- 数据加密和解密
- 权限控制和访问
- 安全审计和监控
- 安全策略管理
- 安全事件处理

相关接口：
- ISecurityComponent
- IAuthManager
- IEncryptor
""" """统一安全实现"""

    def __init__(self, secret_key: Optional[str] = None):

        super().__init__(secret_key)
        self.security_level = SecurityLevel.HIGH
        self._rate_limit = {}
        self._blacklist = set()
        self._whitelist = set()
        self._audit_log = []
        self._lock = threading.RLock()  # 使用可重入锁避免死锁

    def encrypt(self, data: str) -> str:
        """加密数据"""
        try:
            # 使用基础实现，避免依赖问题
            return super().encrypt(data)
        except Exception:
            return data

    def decrypt(self, encrypted_data: str) -> str:
        """解密数据"""
        try:
            return super().decrypt(encrypted_data)
        except Exception:
            return encrypted_data

    def check_rate_limit(self, identifier: str, max_attempts: int = 5, window: int = 300) -> bool:
        """检查速率限制"""
        try:
            with self._lock:
                now = datetime.now()

                if identifier not in self._rate_limit:
                    self._rate_limit[identifier] = []

                # 清理过期的尝试
                self._rate_limit[identifier] = [
                    attempt for attempt in self._rate_limit[identifier]
                    if now - attempt < timedelta(seconds=window)
                ]

                # 检查是否超过限制
                if len(self._rate_limit[identifier]) >= max_attempts:
                    self._log_audit("rate_limit_exceeded", identifier=identifier)
                    return False

                # 记录当前尝试
                self._rate_limit[identifier].append(now)
                return True
        except Exception as e:
            # 如果出现异常，记录并返回True（允许访问）
            print(f"速率限制检查异常: {e}")
            return True

    def add_to_blacklist(self, identifier: str, reason: str = "") -> None:
        """添加到黑名单"""
        try:
            with self._lock:
                self._blacklist.add(identifier)
                self._log_audit("blacklist_add", identifier=identifier, reason=reason)
        except Exception as e:
            print(f"添加到黑名单异常: {e}")

    def is_blacklisted(self, identifier: str) -> bool:
        """检查是否在黑名单中"""
        try:
            with self._lock:
                return identifier in self._blacklist
        except Exception as e:
            print(f"检查黑名单异常: {e}")
            return False

    def remove_from_blacklist(self, identifier: str) -> None:
        """从黑名单移除"""
        try:
            with self._lock:
                self._blacklist.discard(identifier)
                self._log_audit("blacklist_remove", identifier=identifier)
        except Exception as e:
            print(f"从黑名单移除异常: {e}")

    def add_to_whitelist(self, identifier: str) -> None:
        """添加到白名单"""
        try:
            with self._lock:
                self._whitelist.add(identifier)
                self._log_audit("whitelist_add", identifier=identifier)
        except Exception as e:
            print(f"添加到白名单异常: {e}")

    def is_whitelisted(self, identifier: str) -> bool:
        """检查是否在白名单中"""
        try:
            with self._lock:
                return identifier in self._whitelist
        except Exception as e:
            print(f"检查白名单异常: {e}")
            return False

    def remove_from_whitelist(self, identifier: str) -> None:
        """从白名单移除"""
        try:
            with self._lock:
                self._whitelist.discard(identifier)
                self._log_audit("whitelist_remove", identifier=identifier)
        except Exception as e:
            print(f"从白名单移除异常: {e}")

    def _log_audit(self, event: str, **kwargs) -> None:
        """记录审计日志"""
        try:
            audit_entry = {
                "timestamp": datetime.now().isoformat(),
                "event": event,
                "data": kwargs
            }

            self._audit_log.append(audit_entry)

            # 限制审计日志大小
            if len(self._audit_log) > 10000:
                self._audit_log = self._audit_log[-5000:]
        except Exception as e:
            print(f"记录审计日志异常: {e}")

    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取审计日志"""
        try:
            with self._lock:
                return self._audit_log[-limit:].copy()
        except Exception as e:
            print(f"获取审计日志异常: {e}")
            return []

    def clear_audit_log(self) -> None:
        """清空审计日志"""
        try:
            with self._lock:
                self._audit_log.clear()
        except Exception as e:
            print(f"清空审计日志异常: {e}")

    def get_security_stats(self) -> Dict[str, Any]:
        """获取安全统计信息"""
        try:
            with self._lock:
                return {
                    "security_level": self.security_level.value,
                    "blacklist_count": len(self._blacklist),
                    "whitelist_count": len(self._whitelist),
                    "audit_log_count": len(self._audit_log),
                    "rate_limit_entries": len(self._rate_limit),
                    "blacklist_size": len(self._blacklist),
                    "whitelist_size": len(self._whitelist),
                    "audit_log_size": len(self._audit_log)
                }

        except Exception as e:
            print(f"获取安全统计异常: {e}")
            return {}

    def validate_access(self, identifier: str, resource: str, action: str) -> bool:
        """验证访问权限"""
        try:
            with self._lock:
                # 检查黑名单
                if self.is_blacklisted(identifier):
                    self._log_audit("access_denied_blacklist", identifier=identifier,
                                    resource=resource, action=action)
                    return False

                # 检查白名单
                if self.is_whitelisted(identifier):
                    self._log_audit("access_granted_whitelist", identifier=identifier,
                                    resource=resource, action=action)
                    return True

                # 检查速率限制（使用默认参数）
                if not self.check_rate_limit(identifier, max_attempts=5, window=300):
                    self._log_audit("access_denied_rate_limit", identifier=identifier,
                                    resource=resource, action=action)
                    return False

                # 默认允许访问
                self._log_audit("access_granted_default", identifier=identifier,
                                resource=resource, action=action)
                return True
        except Exception as e:
            print(f"访问验证异常: {e}")
            # 如果出现异常，记录并返回False（拒绝访问）
            return False

    def generate_session_token(self, user_id: str, permissions: List[str] = None) -> str:
        """生成会话令牌"""
        try:
            session_data = {
                "user_id": user_id,
                "permissions": permissions or [],
                "created_at": datetime.now().isoformat(),
                "expires_at": (datetime.now() + timedelta(hours=24)).isoformat()
            }

            return self.generate_token(session_data, expires_in=86400)
        except Exception as e:
            print(f"生成会话令牌异常: {e}")
            return ""

    def verify_session_token(self, token: str) -> Optional[Dict[str, Any]]:
        """验证会话令牌"""
        try:
            return self.verify_token(token)
        except Exception as e:
            print(f"验证会话令牌异常: {e}")
            return None

    def sanitize_sql_input(self, input_data: str) -> str:
        """清理SQL输入"""
        try:
            with self._lock:
                # 移除SQL注入字符
                dangerous_patterns = [
                    "'", "'''", ";", "--", "/*", "*/", "DROP", "DELETE", "UPDATE",
                    "INSERT", "SELECT", "UNION", "EXEC", "EXECUTE", "SCRIPT", "JAVASCRIPT"
                ]

                sanitized = input_data.upper()
                for pattern in dangerous_patterns:
                    sanitized = sanitized.replace(pattern, '')

                return sanitized.lower()
        except Exception as e:
            print(f"SQL输入清理异常: {e}")
            return input_data

    def sanitize_html_input(self, input_data: str) -> str:
        """清理HTML输入"""
        try:
            with self._lock:
                # 移除HTML / XSS字符
                html_patterns = [
                    '<', '>', 'script', 'javascript', 'onload', 'onerror', 'onclick',
                    'iframe', 'object', 'embed', 'form', 'input', 'button'
                ]

                sanitized = input_data.lower()
                for pattern in html_patterns:
                    sanitized = sanitized.replace(pattern, '')

                return sanitized
        except Exception as e:
            print(f"HTML输入清理异常: {e}")
            return input_data

    def validate_file_upload(self, filename: str, file_size: int, allowed_extensions: List[str] = None) -> Dict[str, Any]:
        """验证文件上传"""
        try:
            result = {
                "valid": True,
                "issues": []
            }

            # 检查文件扩展名
            if allowed_extensions:
                file_ext = '.' + filename.lower().split('.')[-1] if '.' in filename else ''
                if file_ext not in allowed_extensions:
                    result["valid"] = False
                    result["issues"].append(f"不允许的文件类型: {file_ext}")

            # 检查文件大小（默认最大10MB）
            max_size = 10 * 1024 * 1024  # 10MB
            if file_size > max_size:
                result["valid"] = False
                result["issues"].append(f"文件大小超过限制: {file_size} bytes")

            # 检查文件名安全性
            dangerous_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
            for char in dangerous_chars:
                if char in filename:
                    result["valid"] = False
                    result["issues"].append(f"文件名包含危险字符: {char}")
                    break

            return result
        except Exception as e:
            print(f"文件上传验证异常: {e}")
            return {"valid": False, "issues": [f"验证异常: {e}"]}


# 全局安全实例
_global_security = None


def get_security() -> UnifiedSecurity:
    """获取安全实例"""
    global _global_security
    if _global_security is None:
        _global_security = UnifiedSecurity()
    return _global_security


def set_security(security: UnifiedSecurity) -> None:
    """设置安全实例"""
    global _global_security
    _global_security = security
