#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 访问控制组件 - 策略管理器

负责访问策略的定义、管理和评估
"""

import logging
import re
from typing import Dict, List, Optional, Set, Any
from pathlib import Path
from datetime import datetime

from ...core.types import AccessPolicy, UserRole
from .access_checker import AccessDecision, AccessRequest


class PolicyManager:
    """
    策略管理器

    负责访问策略的创建、更新、删除和评估
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        初始化策略管理器

        Args:
            config_path: 配置存储路径
        """
        self.config_path = config_path or Path("data/security/policies")
        self.config_path.mkdir(parents=True, exist_ok=True)

        self.policies: Dict[str, AccessPolicy] = {}
        self._load_policies()

        logging.info("策略管理器初始化完成")

    def create_policy(self, name: str, resource_pattern: str,
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
        # 生成策略ID
        policy_id = f"policy_{name.lower().replace(' ', '_')}_{len(self.policies)}"

        policy = AccessPolicy(
            policy_id=policy_id,
            name=name,
            description=description,
            resource_pattern=resource_pattern,
            permissions=permissions,
            roles=roles,
            conditions=conditions or {}
        )

        self.policies[policy_id] = policy
        self._save_policies()

        logging.info(f"策略创建成功: {name} (ID: {policy_id})")
        return policy_id

    def get_policy(self, policy_id: str) -> Optional[AccessPolicy]:
        """
        获取策略信息

        Args:
            policy_id: 策略ID

        Returns:
            策略对象或None
        """
        return self.policies.get(policy_id)

    def update_policy(self, policy_id: str, updates: Dict[str, Any]) -> bool:
        """
        更新策略信息

        Args:
            policy_id: 策略ID
            updates: 更新内容

        Returns:
            是否更新成功
        """
        policy = self.policies.get(policy_id)
        if not policy:
            return False

        # 更新策略信息
        for key, value in updates.items():
            if hasattr(policy, key):
                setattr(policy, key, value)

        self._save_policies()
        logging.info(f"策略更新成功: {policy_id}")
        return True

    def delete_policy(self, policy_id: str) -> bool:
        """
        删除策略

        Args:
            policy_id: 策略ID

        Returns:
            是否删除成功
        """
        if policy_id not in self.policies:
            return False

        del self.policies[policy_id]
        self._save_policies()

        logging.info(f"策略删除成功: {policy_id}")
        return True

    def evaluate_policies(self, request: AccessRequest, user_permissions: Set[str]) -> AccessDecision:
        """
        评估访问策略

        Args:
            request: 访问请求
            user_permissions: 用户权限集合

        Returns:
            访问决策
        """
        applicable_policies = []

        # 找到所有适用的策略
        for policy in self.policies.values():
            if self._policy_matches(policy, request, user_permissions):
                applicable_policies.append(policy)

        if not applicable_policies:
            return AccessDecision.ABSTAIN

        # 评估策略 - 优先级处理
        # 这里使用简单的策略：如果有任何策略允许，则允许
        # 在更复杂的系统中，可能需要考虑策略优先级和冲突解决

        allow_count = 0
        deny_count = 0

        for policy in applicable_policies:
            if self._evaluate_policy_conditions(policy, request):
                permission_values = policy.permission_values()
                if any(self._permission_matches(perm, request) for perm in permission_values):
                    allow_count += 1
                else:
                    deny_count += 1

        # 决策逻辑：多数原则，如果允许多于拒绝，则允许
        if allow_count > deny_count:
            return AccessDecision.ALLOW
        elif deny_count > allow_count:
            return AccessDecision.DENY
        else:
            # 平局情况下，安全优先，返回拒绝
            return AccessDecision.DENY

    def list_policies(self) -> List[AccessPolicy]:
        """
        获取所有策略列表

        Returns:
            策略列表
        """
        return list(self.policies.values())

    def get_policies_for_resource(self, resource: str) -> List[AccessPolicy]:
        """
        获取适用于指定资源的策略

        Args:
            resource: 资源标识

        Returns:
            策略列表
        """
        applicable_policies = []

        for policy in self.policies.values():
            if self._resource_matches_pattern(resource, policy.resource_pattern):
                applicable_policies.append(policy)

        return applicable_policies

    def _policy_matches(self, policy: AccessPolicy, request: AccessRequest,
                       user_permissions: Set[str]) -> bool:
        """
        检查策略是否适用于请求

        Args:
            policy: 访问策略
            request: 访问请求
            user_permissions: 用户权限集合

        Returns:
            是否匹配
        """
        # 检查资源匹配
        if not self._resource_matches_pattern(request.resource, policy.resource_pattern):
            return False

        # 检查权限匹配
        permission_values = policy.permission_values()
        if not any(self._permission_matches(perm, request) for perm in permission_values):
            return False

        return True

    def _resource_matches_pattern(self, resource: str, pattern: str) -> bool:
        """
        检查资源是否匹配模式

        Args:
            resource: 资源标识
            pattern: 匹配模式

        Returns:
            是否匹配
        """
        try:
            # 支持通配符模式，如 /api/data/* 或 /api/data/**
            regex_pattern = pattern.replace('*', '.*')
            if pattern.endswith("/*") and resource == pattern[:-2]:
                return True
            return bool(re.match(f"^{regex_pattern}$", resource))
        except re.error:
            # 如果正则表达式无效，则使用简单字符串匹配
            return resource == pattern or pattern in resource

    def _permission_matches(self, policy_permission: str, request: AccessRequest) -> bool:
        """
        检查策略权限是否匹配访问请求
        """
        if policy_permission == "*":
            return True

        resource_variants = {request.resource}
        if "." in request.resource:
            resource_variants.add(request.resource.split(".", 1)[0])
        if "/" in request.resource:
            segments = request.resource.split("/")
            prefix = []
            for segment in segments:
                prefix.append(segment)
                resource_variants.add("/".join(prefix))

        basic_variants = {request.permission}
        for res in resource_variants:
            basic_variants.add(f"{res}:{request.permission}")
            basic_variants.add(f"{request.permission}:{res}")

        if policy_permission in basic_variants:
            return True

        if policy_permission.endswith("*"):
            prefix = policy_permission[:-1]
            return any(variant.startswith(prefix) for variant in basic_variants)

        if ":" in policy_permission:
            left, right = policy_permission.split(":", 1)
            combinations = (
                (left, right),
                (right, left),
            )

            for first, second in combinations:
                first_matches = first in resource_variants or first in {request.permission, "*"}
                second_matches = second in resource_variants or second in {request.permission, "*"}
                if first_matches and second_matches:
                    return True

        return False

    def _evaluate_policy_conditions(self, policy: AccessPolicy, request: AccessRequest) -> bool:
        """
        评估策略条件

        Args:
            policy: 访问策略
            request: 访问请求

        Returns:
            条件是否满足
        """
        if not policy.conditions:
            return True

        # 评估每个条件
        for condition_key, condition_value in policy.conditions.items():
            if condition_key == "time_range":
                current_time = request.context.get("current_time")
                if not self._is_within_time_range(str(condition_value), current_time):
                    return False
                continue

            if condition_key not in request.context:
                return False

            context_value = request.context[condition_key]
            if not self._evaluate_condition(context_value, condition_value):
                return False

        return True

    def _evaluate_condition(self, context_value: Any, condition_value: Any) -> bool:
        """
        评估单个条件

        Args:
            context_value: 上下文中的值
            condition_value: 条件中的值

        Returns:
            条件是否满足
        """
        # 支持多种条件类型
        if isinstance(condition_value, dict):
            # 复杂条件，如 {"operator": "in", "values": ["admin", "user"]}
            operator = condition_value.get("operator", "eq")
            values = condition_value.get("values", [condition_value])

            if operator == "in":
                return context_value in values
            elif operator == "not_in":
                return context_value not in values
            elif operator == "eq":
                return context_value == values[0] if values else False
            elif operator == "ne":
                return context_value != values[0] if values else True
            elif operator == "gt":
                return context_value > values[0] if values else False
            elif operator == "lt":
                return context_value < values[0] if values else False

        # 简单相等条件
        return context_value == condition_value

    def _is_within_time_range(self, time_range: str, current_time: Optional[datetime]) -> bool:
        """判断当前时间是否落在策略提供的时间范围内"""
        try:
            start_str, end_str = [part.strip() for part in time_range.split("-", 1)]
            start_hour, start_minute = map(int, start_str.split(":"))
            end_hour, end_minute = map(int, end_str.split(":"))
        except (ValueError, AttributeError):
            return True  # 非法配置视为通过，避免误拒绝

        now = current_time or datetime.now()
        start = now.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)
        end = now.replace(hour=end_hour, minute=end_minute, second=0, microsecond=0)

        if start <= end:
            return start <= now <= end

        # 跨天场景，例如 22:00-06:00
        return now >= start or now <= end

    def _load_policies(self):
        """从文件加载策略数据"""
        policy_file = self.config_path / "policies.json"
        if policy_file.exists():
            try:
                import json
                with open(policy_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for policy_data in data.values():
                        # 转换角色字符串为枚举
                        roles = set(policy_data.get('roles', []))

                        policy = AccessPolicy(
                            policy_id=policy_data['policy_id'],
                            name=policy_data['name'],
                            description=policy_data.get('description', ''),
                            resource_pattern=policy_data['resource_pattern'],
                            permissions=set(policy_data.get('permissions', [])),
                            roles=roles,
                            conditions=policy_data.get('conditions', {})
                        )
                        self.policies[policy.policy_id] = policy
            except Exception as e:
                logging.error(f"加载策略数据失败: {e}")

    def _save_policies(self):
        """保存策略数据到文件"""
        try:
            import json
            policy_file = self.config_path / "policies.json"
            data = {}
            for policy_id, policy in self.policies.items():
                data[policy_id] = {
                    'policy_id': policy.policy_id,
                    'name': policy.name,
                    'description': policy.description,
                    'resource_pattern': policy.resource_pattern,
                    'permissions': list(policy.permission_values()),
                    'roles': list(policy.role_values()),
                    'conditions': policy.conditions
                }

            with open(policy_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"保存策略数据失败: {e}")
