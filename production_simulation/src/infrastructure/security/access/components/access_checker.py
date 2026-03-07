#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 访问控制组件 - 访问检查器

负责权限检查的核心逻辑和访问决策
"""

import logging
import asyncio
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import sys

from ...core.types import User

# 保持模块别名兼容，支持同时通过 `src.infrastructure...` 和 `infrastructure...` 引用
if __name__.startswith("infrastructure."):
    sys.modules.setdefault(f"src.{__name__}", sys.modules[__name__])
elif __name__.startswith("src."):
    sys.modules.setdefault(__name__[4:], sys.modules[__name__])
logger = logging.getLogger(__name__)


class AccessDecision(Enum):
    """访问决策枚举"""
    ALLOW = "allow"
    DENY = "deny"
    ABSTAIN = "abstain"


@dataclass
class AccessRequest:
    """访问请求(兼容旧版 action 参数)"""
    user_id: str
    resource: str
    permission: Optional[str] = None
    action: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.context is None:
            self.context = {}

        # 兼容旧版 action 参数
        if self.permission is None and self.action is not None:
            self.permission = self.action
        if self.action is None and self.permission is not None:
            self.action = self.permission

        # 确保字符串
        self.permission = self.permission or ""
        self.action = self.action or ""


@dataclass
class AccessResult:
    """访问结果（兼容旧接口）"""
    decision: AccessDecision
    reason: str = ""
    evaluated_policies: List[str] = field(default_factory=list)
    risk_score: float = 0.0
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class AccessChecker:
    """
    访问检查器

    负责权限检查的核心逻辑，包括用户权限验证、策略评估和缓存管理
    """

    def __init__(
        self,
        user_manager=None,
        role_manager=None,
        policy_manager=None,
        audit_manager=None,
        cache_enabled: bool = True,
        max_cache_size: int = 10000,
        cache_ttl_seconds: int = 300,
    ):
        """
        初始化访问检查器

        Args:
            user_manager: 用户管理器
            role_manager: 角色管理器
            policy_manager: 策略管理器
            audit_manager: 审计管理器
            cache_enabled: 是否启用缓存
            max_cache_size: 最大缓存条目数
            cache_ttl_seconds: 缓存过期时间（秒）
        """
        self.user_manager = user_manager
        self.role_manager = role_manager
        self.policy_manager = policy_manager
        self.audit_manager = audit_manager

        self.cache_enabled = cache_enabled
        self._max_cache_size = max_cache_size
        self._cache_ttl_seconds = cache_ttl_seconds

        # 缓存结构：单层键 -> AccessDecision
        self._access_cache: Dict[str, AccessDecision] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_lock = threading.Lock()

        self._total_checks = 0
        self._allow_count = 0
        self._deny_count = 0

        logger.info("访问检查器初始化完成")
        self._logger = logger
        self._username_cache: Dict[str, User] = {}

    def check_access(self, request_or_user_id, *args, **kwargs) -> AccessDecision:
        """
        兼容旧接口的访问检查。

        支持两种调用方式：
            1. check_access(user_id, resource, permission, context=None)
            2. check_access(AccessRequest(...), user_permissions)
        """
        if isinstance(request_or_user_id, AccessRequest):
            request = request_or_user_id
            provided_permissions: Set[str] = set()
            if args:
                provided_permissions = {str(p) for p in args[0]}
            elif "user_permissions" in kwargs and kwargs["user_permissions"] is not None:
                provided_permissions = {str(p) for p in kwargs["user_permissions"]}

            if provided_permissions and self._matches_permissions(request, provided_permissions):
                self._allow_count += 1
                self._log_access_check(request, AccessDecision.ALLOW)
                return AccessResult(
                    decision=AccessDecision.ALLOW,
                    reason="用户拥有直接权限",
                    risk_score=self._calculate_risk_score(request),
                    metadata={"source": "direct_permissions"},
                )

            decision = self.check_access_request(request)
            fallback_reason = (
                "无匹配权限" if decision == AccessDecision.DENY else "访问检查已通过"
            )
            return AccessResult(
                decision=decision,
                reason=fallback_reason,
                risk_score=self._calculate_risk_score(request),
                metadata={"source": "access_checker"},
            )

        user_id = request_or_user_id
        resource = None
        permission = None
        context = None

        if args:
            if len(args) >= 1:
                resource = args[0]
            if len(args) >= 2:
                permission = args[1]
            if len(args) >= 3:
                context = args[2]

        if resource is None:
            resource = kwargs.get("resource", "")
        if permission is None:
            permission = kwargs.get("permission", "")
        if context is None:
            context = kwargs.get("context", {})

        request = AccessRequest(
            user_id=user_id,
            resource=resource,
            permission=permission,
            context=context or {},
        )
        return self.check_access_request(request)

    async def check_access_async(
        self,
        user_id: str,
        resource: str,
        permission: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AccessDecision:
        """
        异步检查用户对资源的访问权限
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.check_access, user_id, resource, permission, context
        )

    def check_access_request(self, request: AccessRequest) -> AccessDecision:
        """
        检查访问请求
        """
        self._total_checks += 1

        if not request.permission:
            if self.cache_enabled:
                self._update_cache(request, AccessDecision.DENY)
            self._deny_count += 1
            self._log_access_check(request, AccessDecision.DENY)
            return AccessDecision.DENY

        try:
            if self.cache_enabled:
                cached = self._check_cache(request)
                if cached is not None:
                    return cached

            user = self._resolve_user(request.user_id)
            if not user:
                logger.warning("访问检查失败: 用户不存在 %s", request.user_id)
                decision = AccessDecision.DENY
            elif not getattr(user, "is_active", True):
                logger.warning("访问检查失败: 用户未激活 %s", request.user_id)
                decision = AccessDecision.DENY
            else:
                decision = self._check_user_permissions(user, request)
                policy_decision = self._evaluate_access_policies(user, request)
                if policy_decision != AccessDecision.ABSTAIN:
                    decision = policy_decision

            if decision == AccessDecision.ABSTAIN:
                decision = AccessDecision.DENY

            if decision == AccessDecision.ALLOW:
                self._allow_count += 1
            elif decision == AccessDecision.DENY:
                self._deny_count += 1

            if self.cache_enabled:
                self._update_cache(request, decision)

            self._log_access_check(request, decision)
            return decision

        except Exception as exc:  # pragma: no cover - 防御性
            logger.exception("访问检查异常: %s", exc)
            return AccessDecision.DENY

    def batch_check_access(self, requests: List[AccessRequest]) -> List[AccessDecision]:
        """批量检查访问权限"""
        return [self.check_access_request(req) for req in requests]

    async def batch_check_access_async(self, requests: List[AccessRequest]) -> List[AccessDecision]:
        """异步批量检查访问权限"""
        tasks = [
            self.check_access_async(req.user_id, req.resource, req.permission, req.context)
            for req in requests
        ]
        return await asyncio.gather(*tasks)

    # --------------------------------------------------------------------- #
    # 缓存相关逻辑
    # --------------------------------------------------------------------- #
    def _generate_cache_key(self, request: AccessRequest) -> str:
        """生成缓存键"""
        return f"{request.user_id}:{request.resource}:{request.permission}"

    def _check_cache(self, request: AccessRequest) -> Optional[AccessDecision]:
        """检查缓存命中情况"""
        cache_key = self._generate_cache_key(request)

        with self._cache_lock:
            decision = self._access_cache.get(cache_key)
            timestamp = self._cache_timestamps.get(cache_key)

            if decision is None or timestamp is None:
                self._cache_misses += 1
                return None

            if datetime.now() - timestamp > timedelta(seconds=self._cache_ttl_seconds):
                # 缓存过期，删除该项
                del self._access_cache[cache_key]
                del self._cache_timestamps[cache_key]
                self._cache_misses += 1
                return None

            self._cache_hits += 1
            return decision

    def _update_cache(self, request: AccessRequest, decision: AccessDecision) -> None:
        """更新缓存"""
        cache_key = self._generate_cache_key(request)

        with self._cache_lock:
            self._access_cache[cache_key] = decision
            self._cache_timestamps[cache_key] = datetime.now()
            self._enforce_cache_size_limit()

    def _enforce_cache_size_limit(self) -> None:
        """保证缓存大小不超过限制"""
        if len(self._access_cache) <= self._max_cache_size:
            return

        # 根据时间戳从旧到新排序，删除最旧的条目
        sorted_items = sorted(
            self._cache_timestamps.items(), key=lambda item: item[1]
        )
        while len(self._access_cache) > self._max_cache_size and sorted_items:
            key, _ = sorted_items.pop(0)
            self._access_cache.pop(key, None)
            self._cache_timestamps.pop(key, None)

    def clear_cache(self) -> None:
        """清空缓存"""
        with self._cache_lock:
            self._access_cache.clear()
            self._cache_timestamps.clear()
            self._cache_hits = 0
            self._cache_misses = 0

        logger.info("权限缓存已清除")

    def invalidate_user_cache(self, user_id: str) -> None:
        """使指定用户的缓存失效"""
        prefix = f"{user_id}:"

        with self._cache_lock:
            keys_to_remove = [key for key in self._access_cache if key.startswith(prefix)]
            for key in keys_to_remove:
                self._access_cache.pop(key, None)
                self._cache_timestamps.pop(key, None)

        if keys_to_remove:
            logger.info("用户缓存已失效: %s", user_id)

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._cache_lock:
            total_entries = len(self._access_cache)
            hits = self._cache_hits
            misses = self._cache_misses
            total = hits + misses
            hit_rate = (hits / total) * 100 if total else 0.0

            if self._cache_timestamps:
                oldest_timestamp = min(self._cache_timestamps.values())
                oldest_age = (datetime.now() - oldest_timestamp).total_seconds()
            else:
                oldest_age = 0.0

        return {
            "total_entries": total_entries,
            "entries": total_entries,
            "cache_hits": hits,
            "cache_misses": misses,
            "hits": hits,
            "misses": misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "oldest_entry_age_seconds": oldest_age,
            "cache_enabled": self.cache_enabled,
            "max_size": self._max_cache_size,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """获取访问检查统计信息"""
        return {
            "total_requests": self._total_checks,
            "allowed_requests": self._allow_count,
            "denied_requests": self._deny_count,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
        }

    # --------------------------------------------------------------------- #
    # 权限与策略逻辑
    # --------------------------------------------------------------------- #
    def _get_user_permissions(self, user: User) -> Set[str]:
        """汇总用户的所有权限"""
        permissions: Set[str] = set()

        # 用户直接权限
        user_permissions = getattr(user, "permissions", None)
        if isinstance(user_permissions, (set, list, tuple)):
            permissions.update(str(p) for p in user_permissions)

        # 角色权限
        roles = getattr(user, "roles", None) or []
        if self.role_manager:
            for role in roles:
                role_name = role.value if hasattr(role, "value") else str(role)

                # 支持两种角色权限查询方式
                get_role_permissions = getattr(self.role_manager, "get_role_permissions", None)
                get_role_by_name = getattr(self.role_manager, "get_role_by_name", None)

                if callable(get_role_permissions):
                    try:
                        role_perms = get_role_permissions(role_name, include_inherited=True)
                        if role_perms:
                            permissions.update(str(p) for p in role_perms)
                    except TypeError:
                        # 某些实现可能需要role_id
                        if callable(get_role_by_name):
                            role_obj = get_role_by_name(role_name)
                            if role_obj is not None:
                                role_id = getattr(role_obj, "role_id", role_name)
                                role_perms = get_role_permissions(role_id, include_inherited=True)
                                if role_perms:
                                    permissions.update(str(p) for p in role_perms)
                elif callable(get_role_by_name):
                    role_obj = get_role_by_name(role_name)
                    if role_obj is not None:
                        role_perms = getattr(role_obj, "permissions", None)
                        if role_perms:
                            permissions.update(str(p) for p in role_perms)

        return permissions

    def _resolve_user(self, identifier: str) -> Optional[User]:
        """根据用户ID或用户名解析用户"""
        if not self.user_manager:
            return None

        user = self.user_manager.get_user(identifier)
        if user:
            return user

        cached = self._username_cache.get(identifier)
        if cached:
            return cached

        users_map = getattr(self.user_manager, "users", {})
        if isinstance(users_map, dict):
            for candidate in users_map.values():
                if getattr(candidate, "username", None) == identifier:
                    self._username_cache[identifier] = candidate
                    try:
                        # 建立别名，便于后续快速获取
                        self.user_manager.users[identifier] = candidate  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    return candidate

        return None

    def _check_user_permissions(self, user: User, request: AccessRequest) -> AccessDecision:
        """校验用户权限"""
        permissions = self._get_user_permissions(user)
        if not permissions:
            return AccessDecision.ABSTAIN

        if self._matches_permissions(request, permissions):
            return AccessDecision.ALLOW

        # 超级权限
        for permission in permissions:
            if permission == "*" or permission.lower() == "admin":
                return AccessDecision.ALLOW

        return AccessDecision.DENY if permissions else AccessDecision.ABSTAIN

    def _matches_permissions(self, request: AccessRequest, permissions: Set[str]) -> bool:
        """检查请求是否匹配给定权限集合"""
        if not permissions:
            return False

        permission_key = request.permission or request.action or ""
        resource_variants = {request.resource}
        if "." in request.resource:
            resource_variants.add(request.resource.split(".", 1)[0])
        if "/" in request.resource:
            segments = request.resource.split("/")
            prefix = []
            for segment in segments:
                prefix.append(segment)
                resource_variants.add("/".join(prefix))

        perm_variants = {permission_key}

        for resource_variant in list(resource_variants):
            perm_variants.add(f"{resource_variant}:{permission_key}")
            perm_variants.add(f"{permission_key}:{resource_variant}")

        if any(variant in permissions for variant in perm_variants):
            return True

        for permission in permissions:
            if permission.endswith("*"):
                prefix = permission[:-1]
                if any(
                    variant.startswith(prefix)
                    for variant in (f"{res}:{permission_key}" for res in resource_variants)
                ) or any(
                    variant.startswith(prefix)
                    for variant in (f"{permission_key}:{res}" for res in resource_variants)
                ) or any(variant.startswith(prefix) for variant in perm_variants):
                    return True

        return False

    def _calculate_risk_score(self, request: AccessRequest) -> float:
        """估算访问风险"""
        risk_score = 0.0
        resource = (request.resource or "").lower()
        if "admin" in resource or "config" in resource:
            risk_score += 0.8
        elif "sensitive" in resource or "private" in resource:
            risk_score += 0.6
        elif "public" in resource:
            risk_score += 0.1

        action = (request.permission or request.action or "").lower()
        if "delete" in action or "drop" in action:
            risk_score += 0.7
        elif "write" in action or "update" in action:
            risk_score += 0.4
        elif "execute" in action or "run" in action:
            risk_score += 0.5

        hour = request.timestamp.hour if request.timestamp else datetime.now().hour
        if hour < 6 or hour > 22:
            risk_score += 0.3

        return min(1.0, max(0.0, risk_score))

    def _evaluate_access_policies(self, user: User, request: AccessRequest) -> AccessDecision:
        """评估访问策略"""
        if not self.policy_manager:
            return AccessDecision.ABSTAIN

        evaluate = getattr(self.policy_manager, "evaluate_policies", None)
        if not callable(evaluate):
            return AccessDecision.ABSTAIN

        user_permissions = self._get_user_permissions(user)

        try:
            return evaluate(user, request)
        except (TypeError, AttributeError):
            try:
                return evaluate(request, user_permissions)
            except Exception as exc:  # pragma: no cover - 防御性
                logger.exception("策略评估失败: %s", exc)
                return AccessDecision.DENY
        except Exception as exc:  # pragma: no cover - 防御性
            logger.exception("策略评估失败: %s", exc)
            return AccessDecision.DENY

    # --------------------------------------------------------------------- #
    # 日志与审计
    # --------------------------------------------------------------------- #
    def _log_access_check(self, request: AccessRequest, decision: AccessDecision) -> None:
        """记录访问检查日志"""
        log_message = (
            f"访问检查: 用户={request.user_id}, 资源={request.resource}, "
            f"权限={request.permission}, 决策={decision.name}"
        )
        active_logger = globals().get("logger")
        if active_logger is not None:
            self._logger = active_logger

        log_target = self.logger

        if decision == AccessDecision.ALLOW:
            log_target.info(log_message)
        else:
            log_target.warning(log_message)

        if self.audit_manager:
            try:
                from ...core.types import AuditEventParams, EventType, EventSeverity

                severity = (
                    EventSeverity.HIGH if decision == AccessDecision.DENY else EventSeverity.MEDIUM
                )
                audit_params = AuditEventParams(
                    event_type=EventType.SECURITY,
                    severity=severity,
                    user_id=request.user_id,
                    resource=request.resource,
                    action=f"access_check:{request.permission}",
                    result="allowed" if decision == AccessDecision.ALLOW else "denied",
                    details={
                        "decision": decision.value,
                        "permission": request.permission,
                        "context": request.context,
                    },
                )
                self.audit_manager.log_event(audit_params)
            except Exception as exc:  # pragma: no cover - 防御性
                logger.exception("审计记录失败: %s", exc)

    @property
    def logger(self):
        active_logger = globals().get("logger")
        if active_logger is not None:
            return active_logger
        return self._logger

    @logger.setter
    def logger(self, value):
        self._logger = value
