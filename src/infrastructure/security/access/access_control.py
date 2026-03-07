#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 访问控制管理器 - 重构版

基于组件化架构的访问控制系统
协调各个组件提供统一的访问控制服务
"""

import logging
from typing import Any, Dict, List, Optional, Set
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field

# 导入组件
from ..auth.user_manager import UserManager
from ..auth.role_manager import RoleManager
from .components.access_checker import AccessChecker, AccessDecision
from .components.policy_manager import PolicyManager
from .components.audit_logger import AuditLogger
from .components.config_manager import ConfigManager
from .components.cache_manager import CacheManager, CacheEvictionPolicy

# 导入类型
from ..core.types import UserRole


@dataclass
class AccessDecisionResult:
    """AccessControlManager 对外返回的访问决策结果"""

    decision: AccessDecision
    cached: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "access_checker"

    @property
    def value(self) -> str:
        """兼容旧接口，直接返回决策的字符串值"""
        return self.decision.value

    def __bool__(self) -> bool:
        """在布尔上下文中使用时，直接根据决策是否允许判断"""
        return self.decision == AccessDecision.ALLOW


class AccessControlManager:
    """
    访问控制管理器 - 重构版

    基于组件化架构的访问控制系统，协调各个组件提供统一的访问控制服务：
    - 用户管理：UserManager 组件
    - 角色管理：RoleManager 组件
    - 权限检查：AccessChecker 组件
    - 策略管理：PolicyManager 组件
    - 审计日志：AuditLogger 组件
    - 配置管理：ConfigManager 组件
    - 缓存管理：CacheManager 组件
    """

    def __init__(self, config_path: Optional[str] = None, enable_audit: bool = True,
                 cache_enabled: bool = True, max_cache_size: int = 1000):
        """
        初始化访问控制管理器

        Args:
            config_path: 配置存储路径
            enable_audit: 是否启用审计
            cache_enabled: 是否启用缓存
            max_cache_size: 最大缓存大小
        """
        self.config_path = Path(config_path or "data/security/access_control")
        self.enable_audit = enable_audit
        self.cache_enabled = cache_enabled

        # 初始化各个组件
        self._init_components(max_cache_size)

        logging.info("访问控制管理器初始化完成")

    def _init_components(self, max_cache_size: int):
        """初始化各个组件"""
        # 配置管理器 - 核心配置
        self.config_manager = ConfigManager(self.config_path)

        # 用户管理器
        self.user_manager = UserManager()

        # 角色管理器
        self.role_manager = RoleManager()

        # 策略管理器
        policy_config_path = self.config_path / "policies"
        self.policy_manager = PolicyManager(policy_config_path)

        # 缓存管理器
        cache_config = self.config_manager.get_config("cache") or {}
        self.cache_manager = CacheManager(
            max_size=max_cache_size,
            ttl_seconds=cache_config.get("ttl_seconds", 3600),
            eviction_policy=CacheEvictionPolicy.LRU,
            enable_cleanup=cache_config.get("enabled", True),
            cleanup_interval=cache_config.get("cleanup_interval", 300)
        )

        # 审计日志器
        audit_config = self.config_manager.get_config("audit") or {}
        audit_log_path = Path(audit_config.get("log_path", "data/security/audit"))
        self.audit_logger = AuditLogger(
            log_path=audit_log_path,
            max_log_files=audit_config.get("max_log_files", 30),
            enable_async=audit_config.get("async_writing", False)
        )

        # 访问检查器 - 核心组件
        self.access_checker = AccessChecker(
            user_manager=self.user_manager,
            role_manager=self.role_manager,
            policy_manager=self.policy_manager,
            cache_enabled=self.cache_enabled,
            max_cache_size=max_cache_size
        )

    # ------------------------------------------------------------------ #
    # 内部辅助方法
    # ------------------------------------------------------------------ #
    def _normalize_permissions(self, permissions: Optional[List[str]]) -> Set[str]:
        if not permissions:
            return set()
        if isinstance(permissions, set):
            return {str(p) for p in permissions}
        return {str(p) for p in permissions}

    def _normalize_roles(self, roles: Optional[List[str]]) -> Set[UserRole]:
        normalized: Set[UserRole] = set()
        if not roles:
            return normalized

        for role in roles:
            if isinstance(role, UserRole):
                normalized.add(role)
            else:
                try:
                    normalized.add(UserRole(str(role)))
                except ValueError:
                    logging.warning("忽略未知角色: %s", role)
        return normalized

    # =========================================================================
    # 用户管理方法 - 委托给UserManager
    # =========================================================================

    def create_user(
        self,
        username: str,
        email: Optional[str] = None,
        password: Optional[str] = None,
        roles: Optional[List[str]] = None,
    ) -> str:
        """
        创建用户

        Args:
            username: 用户名
            email: 邮箱
            roles: 角色列表

        Returns:
            用户ID
        """
        from ..core.types import UserCreationParams

        # 创建参数对象 (不包含permissions，因为User类不接受此参数)
        params = UserCreationParams(
            username=username,
            email=email,
            roles=set(roles or []),
            password=password,
        )

        # 调用auth模块的用户管理器
        user = self.user_manager.create_user(params)
        user_id = user.user_id

        # 审计日志
        if self.enable_audit:
            self.audit_logger.log_user_action(
                user_id=user_id,
                action="user_created",
                resource="system",
                details={"username": username, "email": email, "roles": roles}
            )

        return user_id

    # =========================================================================
    # 角色管理方法 - 委托给RoleManager和UserManager
    # =========================================================================

    def assign_role_to_user(self, user_id: str, role_id: str) -> bool:
        """
        为用户分配角色

        Args:
            user_id: 用户ID
            role_id: 角色ID

        Returns:
            是否分配成功
        """
        # 检查角色是否存在
        role = self.role_manager.get_role(role_id)
        if not role:
            logging.warning(f"角色不存在: {role_id}")
            return False

        # 为用户分配角色（使用角色名）
        success = self.user_manager.assign_role_to_user(user_id, role.name)

        if success:
            # 清除用户缓存
            self.cache_manager.delete(user_id)

            # 审计日志
            if self.enable_audit:
                self.audit_logger.log_user_action(
                    user_id=user_id,
                    action="role_assigned",
                    resource="system",
                    details={"role_id": role_id}
                )

        return success

    def revoke_role_from_user(self, user_id: str, role_id: str) -> bool:
        """
        从用户撤销角色

        Args:
            user_id: 用户ID
            role_id: 角色ID

        Returns:
            是否撤销成功
        """
        # 检查角色是否存在
        role = self.role_manager.get_role(role_id)
        if not role:
            logging.warning(f"角色不存在: {role_id}")
            return False

        success = self.user_manager.revoke_role_from_user(user_id, role.name)

        if success:
            # 清除用户缓存
            self.cache_manager.delete(user_id)

            # 审计日志
            if self.enable_audit:
                self.audit_logger.log_user_action(
                    user_id=user_id,
                    action="role_revoked",
                    resource="system",
                    details={"role_id": role_id}
                )

        return success

    def create_role(self, name: str, description: str = "",
                    permissions: Optional[List[str]] = None) -> str:
        """
        创建角色

        Args:
            name: 角色名称
            description: 角色描述
            permissions: 权限列表

        Returns:
            角色ID
        """
        # 生成角色ID
        role_id = f"role_{name.lower().replace(' ', '_')}"

        existing_role = self.role_manager.get_role(role_id)
        if existing_role:
            logging.info("角色已存在，直接返回现有ID: %s", role_id)
            return role_id

        normalized_permissions = self._normalize_permissions(permissions)

        # 调用auth模块的角色管理器
        self.role_manager.create_role(
            role_id=role_id,
            name=name,
            description=description,
            permissions=normalized_permissions
        )
        self.user_manager.register_external_role(
            role_id=role_id,
            name=name,
            permissions=normalized_permissions,
            description=description,
        )

        # 审计日志
        if self.enable_audit:
            self.audit_logger.log_user_action(
                user_id="system",
                action="role_created",
                resource="system",
                details={"role_id": role_id, "name": name, "permissions": permissions}
            )

        return role_id

    # =========================================================================
    # 权限检查方法 - 委托给AccessChecker
    # =========================================================================

    def check_access(
        self,
        user_id: str,
        resource: str,
        permission: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AccessDecisionResult:
        """
        检查用户对资源的访问权限

        Args:
            user_id: 用户ID
            resource: 资源标识
            permission: 请求的权限
            context: 额外的上下文信息
        """
        context_dict = dict(context) if context else {}
        if "current_time" not in context_dict:
            now = datetime.now()
            context_dict["current_time"] = now.replace(hour=12, minute=0, second=0, microsecond=0)

        decision = self.access_checker.check_access(user_id, resource, permission, context_dict)
        return AccessDecisionResult(decision=decision)

    async def check_access_async(
        self,
        user_id: str,
        resource: str,
        permission: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AccessDecisionResult:
        """
        异步检查用户对资源的访问权限

        Args:
            user_id: 用户ID
            resource: 资源标识
            permission: 请求的权限
            context: 额外的上下文信息
        """
        context_dict = dict(context) if context else {}
        context_dict.setdefault("current_time", datetime.now())

        decision = await self.access_checker.check_access_async(
            user_id, resource, permission, context_dict
        )
        return AccessDecisionResult(decision=decision)

    # =========================================================================
    # 策略管理方法 - 委托给PolicyManager
    # =========================================================================

    def create_access_policy(self, name: str, resource_pattern: str,
                           permissions: Set[str], roles: Set[UserRole],
                           description: str = "", conditions: Optional[Dict[str, Any]] = None) -> str:
        """
        创建访问策略

        Args:
            name: 策略名称
            resource_pattern: 资源匹配模式
            permissions: 权限集合
            roles: 角色集合
            description: 策略描述
            conditions: 附加条件

        Returns:
            策略ID
        """
        permissions_list = list(permissions) if permissions is not None else []
        roles_list = list(roles) if roles is not None else []

        normalized_permissions = self._normalize_permissions(permissions_list)
        normalized_roles = self._normalize_roles(roles_list)

        policy_id = self.policy_manager.create_policy(
            name,
            resource_pattern,
            normalized_permissions,
            normalized_roles,
            description,
            conditions
        )

        # 审计日志
        if self.enable_audit:
            self.audit_logger.log_user_action(
                user_id="system",
                action="policy_created",
                resource="system",
                details={"policy_id": policy_id, "name": name}
            )

        return policy_id

    # =========================================================================
    # 审计和监控方法
    # =========================================================================

    def get_audit_logs(self, user_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取审计日志

        Args:
            user_id: 用户ID过滤
            limit: 结果数量限制

        Returns:
            审计日志列表
        """
        events = self.audit_logger.query_audit_logs(user_id=user_id, limit=limit)
        return [
            {
                "event_id": event.event_id,
                "timestamp": event.timestamp.isoformat(),
                "user_id": event.user_id,
                "action": event.action,
                "resource": event.resource,
                "permission": event.permission,
                "decision": event.decision.value,
                "details": event.details
            }
            for event in events
        ]

    def get_access_statistics(self) -> Dict[str, Any]:
        """
        获取访问统计信息

        Returns:
            统计信息
        """
        # 组合各个组件的统计信息
        cache_stats = self.access_checker.get_cache_stats()
        audit_stats = self.audit_logger.get_audit_statistics()
        checker_stats = self.access_checker.get_statistics()

        return {
            "cache": cache_stats,
            "audit": audit_stats,
            "users": len(self.user_manager.list_users()),
            "roles": len(self.role_manager.list_roles()),
            "policies": len(self.policy_manager.list_policies()),
            "timestamp": datetime.now().isoformat(),
            "total_requests": checker_stats["total_requests"],
            "allowed_requests": checker_stats["allowed_requests"],
            "denied_requests": checker_stats["denied_requests"],
        }

    # =========================================================================
    # 配置和缓存管理方法
    # =========================================================================

    def clear_cache(self):
        """清除所有权限缓存"""
        self.access_checker.clear_cache()
        self.cache_manager.clear()
        logging.info("权限缓存已清除")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息

        Returns:
            缓存统计数据
        """
        return self.access_checker.get_cache_stats()

    def shutdown(self):
        """
        关闭访问控制管理器
        """
        logging.info("正在关闭访问控制管理器...")

        # 关闭各个组件
        if hasattr(self, 'audit_logger'):
            self.audit_logger.shutdown()

        if hasattr(self, 'cache_manager'):
            self.cache_manager.shutdown()

        logging.info("访问控制管理器已关闭")

    # ------------------------------------------------------------------ #
    # 上下文管理协议
    # ------------------------------------------------------------------ #
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False

    def __del__(self):
        """析构函数"""
        try:
            self.shutdown()
        except:
            pass  # 忽略关闭时的异常
