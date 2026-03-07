#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 权限检查器

专门负责权限验证和访问控制逻辑
从AccessControlManager中分离出来，提高代码组织性
"""

import logging
import time
import asyncio
from typing import Dict, List, Optional, Set, Tuple, Any
from src.infrastructure.security.core.types import AccessPolicy
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class AccessDecision(Enum):
    """访问决定"""
    ALLOW = "allow"
    DENY = "deny"
    ABSTAIN = "abstain"


@dataclass
class AccessRequest:
    """访问请求"""
    user_id: str
    resource: str
    action: str
    permission: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        if self.permission is None:
            self.permission = self.action

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'user_id': self.user_id,
            'resource': self.resource,
            'action': self.action,
            'permission': self.permission,
            'context': self.context,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class AccessResult:
    """访问结果"""
    decision: AccessDecision
    reason: str = ""
    evaluated_policies: List[str] = field(default_factory=list)
    risk_score: float = 0.0
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'decision': self.decision.value,
            'reason': self.reason,
            'evaluated_policies': self.evaluated_policies,
            'risk_score': self.risk_score,
            'processing_time': self.processing_time,
            'metadata': self.metadata
        }


class PermissionChecker:
    """权限检查器"""

    def __init__(self, cache_enabled: bool = True, cache_ttl: int = 300) -> None:
        self.cache_enabled = cache_enabled
        self.cache_ttl = cache_ttl
        self._permission_cache: Dict[str, Tuple[AccessResult, datetime]] = {}
        self._stats = {
            'total_checks': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'average_processing_time': 0.0
        }

    def check_access(self, request: AccessRequest, user_permissions: Set[str],
                    policies: Optional[List['AccessPolicy']] = None) -> AccessResult:
        """检查访问权限"""
        start_time = time.time()

        # 尝试从缓存获取
        cache_key = self._generate_cache_key(request)
        if self.cache_enabled:
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                self._stats['cache_hits'] += 1
                self._stats['total_checks'] += 1
                processing_time = time.time() - start_time
                self._update_processing_stats(processing_time)
                return cached_result

        self._stats['cache_misses'] += 1
        self._stats['total_checks'] += 1

        # 执行权限检查
        result = self._evaluate_access(request, user_permissions, policies)

        # 缓存结果
        if self.cache_enabled:
            self._cache_result(cache_key, result)

        # 更新统计信息
        processing_time = time.time() - start_time
        result.processing_time = processing_time
        self._update_processing_stats(processing_time)

        return result

    def _evaluate_access(self, request: AccessRequest, user_permissions: Set[str],
                        policies: Optional[List['AccessPolicy']] = None) -> AccessResult:
        """评估访问权限"""
        # 基本权限检查
        if self._has_direct_permission(request, user_permissions):
            return AccessResult(
                decision=AccessDecision.ALLOW,
                reason="用户拥有直接权限",
                evaluated_policies=[],
                risk_score=self._calculate_risk_score(request)
            )

        # 策略评估
        if policies:
            policy_result = self._evaluate_policies(request, policies)
            if policy_result.decision != AccessDecision.ABSTAIN:
                return policy_result

        # 基于资源的权限检查
        if self._check_resource_permission(request):
            return AccessResult(
                decision=AccessDecision.ALLOW,
                reason="资源权限检查通过",
                evaluated_policies=[],
                risk_score=self._calculate_risk_score(request)
            )

        # 默认拒绝
        return AccessResult(
            decision=AccessDecision.DENY,
            reason="无匹配权限",
            evaluated_policies=[],
            risk_score=self._calculate_risk_score(request)
        )

    def _has_direct_permission(self, request: AccessRequest, user_permissions: Set[str]) -> bool:
        """检查是否拥有直接权限"""
        # 构造完整的权限字符串: resource:permission
        full_permission = f"{request.resource}:{request.permission}"

        # 直接匹配完整权限
        if full_permission in user_permissions:
            return True

        # 匹配权限部分（向后兼容）
        if request.permission in user_permissions:
            return True

        # 通配符匹配
        for permission in user_permissions:
            if permission.endswith("*"):
                if full_permission.startswith(permission[:-1]):
                    return True

        return False

    def _evaluate_policies(self, request: AccessRequest, policies: List['AccessPolicy']) -> AccessResult:
        """评估策略"""
        evaluated_policies = []

        for policy in policies:
            evaluated_policies.append(policy.policy_id)

        if self._policy_applies_to_request(policy, request):
                permission_values = policy.permission_values()
                if (request.permission or request.action) in permission_values:
                    return AccessResult(
                        decision=AccessDecision.ALLOW,
                        reason=f"策略 {policy.name} 允许访问",
                        evaluated_policies=evaluated_policies,
                        risk_score=self._calculate_risk_score(request)
                    )

        return AccessResult(
            decision=AccessDecision.ABSTAIN,
            reason="无适用的策略",
            evaluated_policies=evaluated_policies,
            risk_score=self._calculate_risk_score(request)
        )

    def _policy_applies_to_request(self, policy: 'AccessPolicy', request: AccessRequest) -> bool:
        """检查策略是否适用于请求"""
        # 资源匹配
        if not self._matches_resource_pattern(policy.resource_pattern, request.resource):
            return False

        # 条件检查
        if policy.conditions:
            for condition_key, condition_value in policy.conditions.items():
                request_value = request.context.get(condition_key)
                if request_value != condition_value:
                    return False

        return True

    def _matches_resource_pattern(self, pattern: str, resource: str) -> bool:
        """检查资源是否匹配模式"""
        import re

        if pattern == "*":
            return True

        if pattern.endswith("*"):
            return resource.startswith(pattern[:-1])

        if pattern.startswith("*"):
            return resource.endswith(pattern[1:])

        # 正则表达式匹配
        try:
            return bool(re.match(pattern, resource))
        except re.error:
            return pattern == resource

    def _check_resource_permission(self, request: AccessRequest) -> bool:
        """检查基于资源的权限"""
        # 这里可以实现更复杂的资源权限逻辑
        # 例如：基于所有者、部门、项目等

        # 简单的实现：检查上下文中的权限信息
        context_permissions = request.context.get('permissions', [])
        target_permission = request.permission or request.action
        return target_permission in context_permissions

    def _calculate_risk_score(self, request: AccessRequest) -> float:
        """计算风险分数"""
        risk_score = 0.0

        # 基于资源类型的风险评估
        resource = request.resource.lower()
        if 'admin' in resource or 'config' in resource:
            risk_score += 0.8
        elif 'sensitive' in resource or 'private' in resource:
            risk_score += 0.6
        elif 'public' in resource:
            risk_score += 0.1

        # 基于操作类型的风险评估
        action = (request.permission or request.action).lower()
        if 'delete' in action or 'drop' in action:
            risk_score += 0.7
        elif 'write' in action or 'update' in action:
            risk_score += 0.4
        elif 'execute' in action or 'run' in action:
            risk_score += 0.5

        # 基于时间的风险评估
        hour = request.timestamp.hour
        if hour < 6 or hour > 22:  # 非工作时间
            risk_score += 0.3

        # 限制在0-1之间
        return min(1.0, max(0.0, risk_score))

    def _generate_cache_key(self, request: AccessRequest) -> str:
        """生成缓存键"""
        context_str = str(sorted(request.context.items()))
        permission_key = request.permission or request.action
        return f"{request.user_id}:{request.resource}:{permission_key}:{context_str}"

    def _get_cached_result(self, cache_key: str) -> Optional[AccessResult]:
        """从缓存获取结果"""
        if cache_key in self._permission_cache:
            result, timestamp = self._permission_cache[cache_key]
            if datetime.now() - timestamp < timedelta(seconds=self.cache_ttl):
                return result
            else:
                # 缓存过期，删除
                del self._permission_cache[cache_key]
        return None

    def _cache_result(self, cache_key: str, result: AccessResult) -> None:
        """缓存结果"""
        self._permission_cache[cache_key] = (result, datetime.now())

        # 清理过期缓存
        self._cleanup_expired_cache()

    def _cleanup_expired_cache(self) -> None:
        """清理过期缓存"""
        current_time = datetime.now()
        expired_keys = []

        for key, (_, timestamp) in self._permission_cache.items():
            if current_time - timestamp > timedelta(seconds=self.cache_ttl):
                expired_keys.append(key)

        for key in expired_keys:
            del self._permission_cache[key]

    def _update_processing_stats(self, processing_time: float) -> None:
        """更新处理统计信息"""
        # 简单的移动平均
        alpha = 0.1
        self._stats['average_processing_time'] = (
            self._stats['average_processing_time'] * (1 - alpha) +
            processing_time * alpha
        )

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        cache_hit_rate = (
            self._stats['cache_hits'] / max(self._stats['total_checks'], 1) * 100
        )

        return {
            'total_checks': self._stats['total_checks'],
            'cache_hits': self._stats['cache_hits'],
            'cache_misses': self._stats['cache_misses'],
            'cache_hit_rate': f"{cache_hit_rate:.1f}%",
            'average_processing_time': f"{self._stats['average_processing_time']:.4f}s",
            'cache_size': len(self._permission_cache),
            'cache_enabled': self.cache_enabled
        }

    def clear_cache(self) -> None:
        """清除缓存"""
        self._permission_cache.clear()
        logging.info("权限检查缓存已清除")

    def set_cache_enabled(self, enabled: bool) -> None:
        """设置缓存启用状态"""
        self.cache_enabled = enabled
        if not enabled:
            self.clear_cache()
        logging.info(f"权限检查缓存已{'启用' if enabled else '禁用'}")

    async def check_access_async(self, request: AccessRequest, user_permissions: Set[str],
                                 policies: Optional[List['AccessPolicy']] = None) -> AccessResult:
        """异步检查访问权限"""
        # 在线程池中执行同步检查，避免阻塞事件循环
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self.check_access, request, user_permissions, policies
        )
        return result

    async def batch_check_access_async(self, requests: List[AccessRequest],
                                      user_permissions: Dict[str, Set[str]],
                                      policies: Optional[List['AccessPolicy']] = None,
                                      max_concurrency: int = 10) -> List[AccessResult]:
        """异步批量检查访问权限"""
        semaphore = asyncio.Semaphore(max_concurrency)

        async def check_single(request: AccessRequest) -> AccessResult:
            async with semaphore:
                user_perms = user_permissions.get(request.user_id, set())
                return await self.check_access_async(request, user_perms, policies)

        # 创建所有检查任务
        tasks = [check_single(request) for request in requests]

        # 并发执行所有任务
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常结果，将异常转换为错误结果
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # 创建错误结果
                error_result = AccessResult(
                    decision=AccessDecision.DENY,
                    reason=f"检查失败: {str(result)}",
                    risk_score=1.0,
                    processing_time=0.0,
                    metadata={'error': str(result), 'request_index': i}
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)

        return processed_results

    def batch_check_access(self, requests: List[AccessRequest],
                          user_permissions: Dict[str, Set[str]],
                          policies: Optional[List['AccessPolicy']] = None) -> List[AccessResult]:
        """批量检查访问权限"""
        results = []
        for request in requests:
            user_perms = user_permissions.get(request.user_id, set())
            result = self.check_access(request, user_perms, policies)
            results.append(result)

        return results
