#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 策略管理器

负责访问控制策略的管理
分离了AccessControlManager的策略职责
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
from src.infrastructure.security.core.types import PolicyCreationParams, AccessPolicy


class PolicyManager:
    """
    策略管理器

    职责：专门管理访问控制策略
    包括策略创建、评估、匹配等功能
    """

    def __init__(self):
        self.policies: Dict[str, 'AccessPolicy'] = {}
        self.policy_cache: Dict[str, List['AccessPolicy']] = {}

    def create_policy(self, params: PolicyCreationParams) -> 'AccessPolicy':
        """
        创建访问策略

        Args:
            params: 策略创建参数

        Returns:
            创建的策略对象
        """
        policy = AccessPolicy(
            policy_id=self._generate_policy_id(),
            name=params.name,
            resource_type=params.resource_type,
            resource_pattern=params.resource_pattern,
            permissions=params.permissions.copy(),
            roles=params.roles.copy(),
            conditions=params.conditions.copy(),
            priority=params.priority,
            description=params.description,
            is_active=params.is_active,
            expiry_date=params.expiry_date,
            metadata=params.metadata.copy(),
        )

        self.policies[policy.policy_id] = policy

        # 清除相关缓存
        self._clear_policy_cache()

        logging.info(f"创建策略: {policy.name} ({policy.policy_id})")
        return policy

    def evaluate_policies(self, user: 'User', resource: str, permission: str) -> List['AccessPolicy']:
        """
        评估适用的策略

        Args:
            user: 用户对象
            resource: 资源
            permission: 权限

        Returns:
            适用的策略列表
        """
        cache_key = f"{user.user_id}:{resource}:{permission}"
        if cache_key in self.policy_cache:
            return self.policy_cache[cache_key]

        applicable_policies = []

        for policy in self.policies.values():
            if self._policy_applies(policy, user, resource, permission):
                applicable_policies.append(policy)

        # 按优先级排序
        applicable_policies.sort(key=lambda p: p.priority, reverse=True)

        # 缓存结果
        self.policy_cache[cache_key] = applicable_policies

        return applicable_policies

    def check_policy_access(self, policies: List['AccessPolicy'], permission: str, context: Dict) -> bool:
        """
        检查策略访问权限

        Args:
            policies: 适用的策略列表
            permission: 请求的权限
            context: 访问上下文

        Returns:
            是否允许访问
        """
        if not policies:
            return False

        # 默认拒绝，除非有明确允许的策略
        for policy in policies:
            if self._evaluate_policy_conditions(policy, context):
                if any(
                    self._permission_matches(str(policy_perm), permission, context.get("resource", ""))
                    for policy_perm in policy.permissions
                ):
                    return True

        return False

    def update_policy(self, policy_id: str, **kwargs) -> bool:
        """更新策略"""
        if policy_id not in self.policies:
            return False

        policy = self.policies[policy_id]

        for key, value in kwargs.items():
            if hasattr(policy, key):
                setattr(policy, key, value)

        # 清除缓存
        self._clear_policy_cache()

        return True

    def delete_policy(self, policy_id: str) -> bool:
        """删除策略"""
        if policy_id not in self.policies:
            return False

        del self.policies[policy_id]
        self._clear_policy_cache()

        return True

    def _policy_applies(self, policy: 'AccessPolicy', user: 'User', resource: str, permission: str) -> bool:
        """检查策略是否适用"""
        # 检查策略是否激活
        if not policy.is_active:
            return False

        # 检查是否过期
        if policy.expiry_date and policy.expiry_date < datetime.now():
            return False

        # 检查资源匹配
        if not self._matches_resource_pattern(policy.resource_pattern, resource):
            return False

        # 检查权限匹配
        permission_values = policy.permission_values()
        if not any(
            self._permission_matches(policy_perm, permission, resource)
            for policy_perm in permission_values
        ):
            return False

        return True

    def _matches_resource_pattern(self, pattern: str, resource: str) -> bool:
        """检查资源是否匹配模式"""
        # 简化实现：支持通配符匹配
        if pattern == "*":
            return True

        if pattern.endswith("*"):
            prefix = pattern[:-1]
            return resource.startswith(prefix)

        if pattern.startswith("*"):
            suffix = pattern[1:]
            return resource.endswith(suffix)

        return pattern == resource

    def _permission_matches(self, policy_permission: str, request_permission: str, resource: str) -> bool:
        """策略权限匹配判定"""
        policy_permission = policy_permission or ""
        request_permission = request_permission or ""
        resource = resource or ""

        if policy_permission == "*" or policy_permission == request_permission:
            return True

        resource_variants = {resource}
        if "." in resource:
            resource_variants.add(resource.split(".", 1)[0])
        if "/" in resource:
            segments = resource.split("/")
            prefix = []
            for segment in segments:
                prefix.append(segment)
                resource_variants.add("/".join(prefix))

        perm_variants = {request_permission}
        for resource_variant in list(resource_variants):
            perm_variants.add(f"{resource_variant}:{request_permission}")
            perm_variants.add(f"{request_permission}:{resource_variant}")

        if policy_permission in perm_variants:
            return True

        if policy_permission.endswith("*"):
            prefix = policy_permission[:-1]
            return any(variant.startswith(prefix) for variant in perm_variants)

        if ":" in policy_permission:
            left, right = policy_permission.split(":", 1)
            if request_permission in {left, right}:
                return True
            combinations = (
                (left, right),
                (right, left),
            )
            for first, second in combinations:
                first_matches = first in resource_variants or first in {request_permission, "*"}
                second_matches = second in resource_variants or second in {request_permission, "*"}
                if first_matches and second_matches:
                    return True

        alias_map = {
            "read": {"read", "data:read"},
            "write": {"write", "data:write"},
            "delete": {"delete", "write", "data:write", "trade:execute"},
        }
        normalized = request_permission.lower()
        if normalized in alias_map:
            if any(policy_permission.lower() == alias.lower() for alias in alias_map[normalized]):
                return True

        return False

    def _evaluate_policy_conditions(self, policy: 'AccessPolicy', context: Dict) -> bool:
        """评估策略条件"""
        # 简化实现：检查所有条件是否满足
        for condition_key, condition_value in policy.conditions.items():
            if condition_key not in context:
                return False

            context_value = context[condition_key]

            # 简单的条件匹配
            if isinstance(condition_value, (list, set, tuple)):
                if context_value not in condition_value:
                    return False
            else:
                if context_value != condition_value:
                    return False

        return True

    def _generate_policy_id(self) -> str:
        """生成策略ID"""
        import uuid
        return f"policy_{uuid.uuid4().hex[:8]}"

    def _clear_policy_cache(self) -> None:
        """清除策略缓存"""
        self.policy_cache.clear()

    def list_policies(self, active_only: bool = True) -> List['AccessPolicy']:
        """列出策略"""
        policies = list(self.policies.values())
        if active_only:
            policies = [p for p in policies if p.is_active]
        return policies

    def get_policy(self, policy_id: str) -> Optional['AccessPolicy']:
        """获取策略"""
        return self.policies.get(policy_id)


class SessionManager:
    """
    会话管理器

    职责：专门管理用户会话
    """

    def __init__(self):
        self.sessions: Dict[str, 'UserSession'] = {}
        self.session_timeout = 3600  # 默认1小时

    def create_session(self, user_id: str, **kwargs) -> str:
        """创建会话"""
        session_id = self._generate_session_id()

        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(seconds=self.session_timeout),
            **kwargs
        )

        self.sessions[session_id] = session
        return session_id

    def get_session(self, session_id: str) -> Optional['UserSession']:
        """获取会话"""
        session = self.sessions.get(session_id)
        if session and session.is_expired():
            del self.sessions[session_id]
            return None
        return session

    def invalidate_session(self, session_id: str) -> bool:
        """使会话失效"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    def _generate_session_id(self) -> str:
        """生成会话ID"""
        import uuid
        return f"sess_{uuid.uuid4().hex[:12]}"


class CacheManager:
    """
    缓存管理器

    职责：专门管理权限检查缓存
    """

    def __init__(self, max_size: int = 10000):
        self.cache: Dict[str, Dict] = {}
        self.max_size = max_size
        self.access_times: Dict[str, datetime] = {}

    def get(self, key: str) -> Optional[Dict]:
        """从缓存获取值"""
        if key in self.cache:
            self.access_times[key] = datetime.now()
            return self.cache[key]
        return None

    def set(self, key: str, value: Dict) -> None:
        """设置缓存值"""
        self.cache[key] = value
        self.access_times[key] = datetime.now()

        # 清理过期缓存
        self._cleanup_expired_cache()

    def clear(self) -> None:
        """清除所有缓存"""
        self.cache.clear()
        self.access_times.clear()

    def _cleanup_expired_cache(self) -> None:
        """清理过期缓存"""
        if len(self.cache) > self.max_size:
            # 清理最旧的缓存项
            oldest_keys = sorted(self.access_times.items(), key=lambda x: x[1])[:1000]
            for key, _ in oldest_keys:
                self.cache.pop(key, None)
                self.access_times.pop(key, None)


# 导入必要的类型
from datetime import timedelta
from src.infrastructure.security.core.types import AccessPolicy, User, UserSession
