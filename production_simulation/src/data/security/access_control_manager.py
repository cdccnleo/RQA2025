#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 访问控制管理器

实现基于角色的访问控制(RBAC)系统
提供细粒度的权限管理和安全访问控制
"""

import json
import logging
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
import threading


class Permission(Enum):

    """权限枚举"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"
    AUDIT = "audit"


class ResourceType(Enum):

    """资源类型枚举"""
    DATA = "data"
    CACHE = "cache"
    CONFIG = "config"
    LOG = "log"
    METADATA = "metadata"
    SYSTEM = "system"


@dataclass
class User:

    """用户"""
    user_id: str
    username: str
    email: Optional[str] = None
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    roles: Set[str] = field(default_factory=set)
    permissions: Set[str] = field(default_factory=set)  # 直接权限
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_role(self, role: str) -> bool:
        """检查是否有指定角色"""
        return role in self.roles

    def has_permission(self, permission: str) -> bool:
        """检查是否有指定权限"""
        return permission in self.permissions

    def add_role(self, role: str):
        """添加角色"""
        self.roles.add(role)

    def remove_role(self, role: str):
        """移除角色"""
        self.roles.discard(role)

    def add_permission(self, permission: str):
        """添加直接权限"""
        self.permissions.add(permission)

    def remove_permission(self, permission: str):
        """移除直接权限"""
        self.permissions.discard(permission)


@dataclass
class Role:

    """角色"""
    role_id: str
    name: str
    description: str = ""
    permissions: Set[str] = field(default_factory=set)
    parent_roles: Set[str] = field(default_factory=set)  # 父角色（继承）
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_permission(self, permission: str):
        """添加权限"""
        self.permissions.add(permission)

    def remove_permission(self, permission: str):
        """移除权限"""
        self.permissions.discard(permission)

    def add_parent_role(self, role_id: str):
        """添加父角色"""
        self.parent_roles.add(role_id)

    def remove_parent_role(self, role_id: str):
        """移除父角色"""
        self.parent_roles.discard(role_id)

    def get_all_permissions(self, role_registry: Dict[str, 'Role']) -> Set[str]:
        """获取所有权限（包括继承的）"""
        all_permissions = self.permissions.copy()

        # 递归获取父角色的权限
        for parent_id in self.parent_roles:
            if parent_id in role_registry:
                parent_permissions = role_registry[parent_id].get_all_permissions(role_registry)
                all_permissions.update(parent_permissions)

        return all_permissions


@dataclass
class AccessPolicy:

    """访问策略"""
    policy_id: str
    name: str
    resource_type: ResourceType
    resource_pattern: str  # 资源匹配模式，如 "data:stock:*", "config:*"
    permissions: Set[str]
    conditions: Dict[str, Any] = field(default_factory=dict)  # 访问条件
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def matches_resource(self, resource: str) -> bool:
        """检查资源是否匹配模式"""
        # 简单的通配符匹配
        if self.resource_pattern == "*":
            return True

        # 精确匹配
        if self.resource_pattern == resource:
            return True

        # 前缀匹配
        if self.resource_pattern.endswith("*"):
            prefix = self.resource_pattern[:-1]
            return resource.startswith(prefix)

        # 类型匹配
        if ":" in self.resource_pattern and ":" in resource:
            policy_type, policy_pattern = self.resource_pattern.split(":", 1)
            resource_type, resource_pattern = resource.split(":", 1)
            if policy_type == resource_type:
                if policy_pattern == "*" or policy_pattern == resource_pattern:
                    return True

        return False

    def check_conditions(self, context: Dict[str, Any]) -> bool:
        """检查访问条件"""
        for condition_key, condition_value in self.conditions.items():
            if condition_key not in context:
                return False

            context_value = context[condition_key]

            # 时间范围条件
            if condition_key == "time_range":
                if isinstance(condition_value, dict):
                    start_time = condition_value.get("start")
                    end_time = condition_value.get("end")
                    current_time = context_value

                    if start_time and current_time < start_time:
                        return False
                    if end_time and current_time > end_time:
                        return False

            # IP地址条件
            elif condition_key == "ip_range":
                # 简化的IP检查
                if context_value not in condition_value:
                    return False

            # 其他条件
            elif context_value != condition_value:
                return False

        return True


@dataclass
class AccessRequest:

    """访问请求"""
    user_id: str
    resource: str
    permission: str
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AccessDecision:

    """访问决策"""
    request: AccessRequest
    allowed: bool
    reason: str
    applied_policies: List[str] = field(default_factory=list)
    decision_time: datetime = field(default_factory=datetime.now)


class AccessControlManager:

    """
    访问控制管理器

    提供完整的RBAC访问控制功能：
    - 用户和角色管理
    - 权限分配和继承
    - 访问策略控制
    - 审计日志记录
    - 动态权限检查
    """

    def __init__(self, config_path: Optional[str] = None, enable_audit: bool = True):
        """
        初始化访问控制管理器

        Args:
            config_path: 配置存储路径
            enable_audit: 是否启用审计
        """
        self.config_path = Path(config_path or "data/security/access_control")
        self.config_path.mkdir(parents=True, exist_ok=True)

        self.enable_audit = enable_audit
        self.audit_log_path = self.config_path / "access_audit.log"

        # 用户、角色和策略存储
        self.users: Dict[str, User] = {}
        self.roles: Dict[str, Role] = {}
        self.policies: Dict[str, AccessPolicy] = {}

        # 缓存
        self._permission_cache: Dict[str, Dict[str, bool]] = {}
        self._cache_lock = threading.Lock()

        # 默认角色
        self._initialize_default_roles()

        # 加载配置
        self._load_config()

        logging.info("访问控制管理器初始化完成")

    def _initialize_default_roles(self):
        """初始化默认角色"""
        # 系统管理员
        admin_role = Role(
            role_id="admin",
            name="System Administrator",
            description="系统管理员，具有所有权限",
            permissions={"admin", "read", "write", "delete", "execute", "audit"}
        )
        self.roles["admin"] = admin_role

        # 数据分析师
        analyst_role = Role(
            role_id="analyst",
            name="Data Analyst",
            description="数据分析师，具有读取和分析权限",
            permissions={"read", "execute"}
        )
        self.roles["analyst"] = analyst_role

        # 数据操作员
        operator_role = Role(
            role_id="operator",
            name="Data Operator",
            description="数据操作员，具有读取和写入权限",
            permissions={"read", "write"}
        )
        self.roles["operator"] = operator_role

        # 审计员
        auditor_role = Role(
            role_id="auditor",
            name="Auditor",
            description="审计员，具有审计和读取权限",
            permissions={"read", "audit"}
        )
        self.roles["auditor"] = auditor_role

        # 建立角色继承关系
        operator_role.add_parent_role("analyst")  # 操作员继承分析师权限

    # =========================================================================
    # 用户管理
    # =========================================================================

    def create_user(self, username: str, email: Optional[str] = None,


                    roles: Optional[List[str]] = None) -> str:
        """
        创建用户

        Args:
            username: 用户名
            email: 邮箱
            roles: 角色列表

        Returns:
            用户ID
        """
        # 检查用户名是否已存在
        for user in self.users.values():
            if user.username == username:
                raise ValueError(f"用户名已存在: {username}")

        user_id = f"user_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

        user = User(
            user_id=user_id,
            username=username,
            email=email,
            roles=set(roles or [])
        )

        self.users[user_id] = user

        # 清除相关缓存
        self._clear_user_cache(user_id)

        # 审计日志
        if self.enable_audit:
            self._audit_log('create_user', {
                'user_id': user_id,
                'username': username,
                'roles': list(user.roles)
            })

        logging.info(f"创建用户: {username} ({user_id})")
        return user_id

    def assign_role_to_user(self, user_id: str, role_id: str):
        """
        为用户分配角色

        Args:
            user_id: 用户ID
            role_id: 角色ID
        """
        if user_id not in self.users:
            raise ValueError(f"用户不存在: {user_id}")

        if role_id not in self.roles:
            raise ValueError(f"角色不存在: {role_id}")

        user = self.users[user_id]
        user.add_role(role_id)

        # 清除缓存
        self._clear_user_cache(user_id)

        # 审计日志
        if self.enable_audit:
            self._audit_log('assign_role', {
                'user_id': user_id,
                'role_id': role_id
            })

        logging.info(f"为用户 {user.username} 分配角色: {role_id}")

    def revoke_role_from_user(self, user_id: str, role_id: str):
        """
        从用户撤销角色

        Args:
            user_id: 用户ID
            role_id: 角色ID
        """
        if user_id not in self.users:
            raise ValueError(f"用户不存在: {user_id}")

        user = self.users[user_id]
        user.remove_role(role_id)

        # 清除缓存
        self._clear_user_cache(user_id)

        # 审计日志
        if self.enable_audit:
            self._audit_log('revoke_role', {
                'user_id': user_id,
                'role_id': role_id
            })

        logging.info(f"从用户 {user.username} 撤销角色: {role_id}")

    def grant_permission_to_user(self, user_id: str, permission: str):
        """
        为用户授予直接权限

        Args:
            user_id: 用户ID
            permission: 权限
        """
        if user_id not in self.users:
            raise ValueError(f"用户不存在: {user_id}")

        user = self.users[user_id]
        user.add_permission(permission)

        # 清除缓存
        self._clear_user_cache(user_id)

        # 审计日志
        if self.enable_audit:
            self._audit_log('grant_permission', {
                'user_id': user_id,
                'permission': permission,
                'type': 'direct'
            })

        logging.info(f"为用户 {user.username} 授予直接权限: {permission}")

    # =========================================================================
    # 角色管理
    # =========================================================================

    def create_role(self, name: str, description: str = "",


                    permissions: Optional[List[str]] = None,
                    parent_roles: Optional[List[str]] = None) -> str:
        """
        创建角色

        Args:
            name: 角色名称
            description: 描述
            permissions: 权限列表
            parent_roles: 父角色列表

        Returns:
            角色ID
        """
        # 检查角色名是否已存在
        for role in self.roles.values():
            if role.name == name:
                raise ValueError(f"角色名已存在: {name}")

        role_id = f"role_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

        role = Role(
            role_id=role_id,
            name=name,
            description=description,
            permissions=set(permissions or []),
            parent_roles=set(parent_roles or [])
        )

        self.roles[role_id] = role

        # 审计日志
        if self.enable_audit:
            self._audit_log('create_role', {
                'role_id': role_id,
                'name': name,
                'permissions': list(role.permissions)
            })

        logging.info(f"创建角色: {name} ({role_id})")
        return role_id

    def add_permission_to_role(self, role_id: str, permission: str):
        """
        为角色添加权限

        Args:
            role_id: 角色ID
            permission: 权限
        """
        if role_id not in self.roles:
            raise ValueError(f"角色不存在: {role_id}")

        role = self.roles[role_id]
        role.add_permission(permission)

        # 清除相关用户的缓存
        self._clear_role_cache(role_id)

        # 审计日志
        if self.enable_audit:
            self._audit_log('add_role_permission', {
                'role_id': role_id,
                'permission': permission
            })

        logging.info(f"为角色 {role.name} 添加权限: {permission}")

    def set_role_inheritance(self, child_role_id: str, parent_role_id: str):
        """
        设置角色继承关系

        Args:
            child_role_id: 子角色ID
            parent_role_id: 父角色ID
        """
        if child_role_id not in self.roles:
            raise ValueError(f"子角色不存在: {child_role_id}")

        if parent_role_id not in self.roles:
            raise ValueError(f"父角色不存在: {parent_role_id}")

        if child_role_id == parent_role_id:
            raise ValueError("不能设置角色自继承")

        child_role = self.roles[child_role_id]
        child_role.add_parent_role(parent_role_id)

        # 清除相关缓存
        self._clear_role_cache(child_role_id)

        # 审计日志
        if self.enable_audit:
            self._audit_log('set_inheritance', {
                'child_role_id': child_role_id,
                'parent_role_id': parent_role_id
            })

        logging.info(f"设置角色继承: {child_role.name} -> {self.roles[parent_role_id].name}")

    # =========================================================================
    # 访问控制
    # =========================================================================

    def check_access(self, user_id: str, resource: str, permission: str,


                     context: Optional[Dict[str, Any]] = None) -> AccessDecision:
        """
        检查访问权限

        Args:
            user_id: 用户ID
            resource: 资源
            permission: 权限
            context: 访问上下文

        Returns:
            访问决策
        """
        request = AccessRequest(
            user_id=user_id,
            resource=resource,
            permission=permission,
            context=context or {}
        )

        # 检查缓存
        cache_key = f"{user_id}:{resource}:{permission}"
        with self._cache_lock:
            if cache_key in self._permission_cache:
                cached_result = self._permission_cache[cache_key]
                return AccessDecision(
                    request=request,
                    allowed=cached_result,
                    reason="cached_result",
                    applied_policies=[]
                )

        # 获取用户
        if user_id not in self.users:
            return AccessDecision(
                request=request,
                allowed=False,
                reason="user_not_found"
            )

        user = self.users[user_id]

        # 检查用户是否激活
        if not user.is_active:
            return AccessDecision(
                request=request,
                allowed=False,
                reason="user_inactive"
            )

        # 获取用户所有权限
        user_permissions = self._get_user_permissions(user)

        # 检查直接权限
        if permission in user_permissions:
            decision = AccessDecision(
                request=request,
                allowed=True,
                reason="direct_permission",
                applied_policies=[]
            )
        else:
            # 检查访问策略
            decision = self._check_access_policies(request, user_permissions)

        # 更新缓存
        with self._cache_lock:
            self._permission_cache[cache_key] = decision.allowed

        # 审计日志
        if self.enable_audit:
            self._audit_log('access_check', {
                'user_id': user_id,
                'resource': resource,
                'permission': permission,
                'allowed': decision.allowed,
                'reason': decision.reason
            })

        return decision

    def _get_user_permissions(self, user: User) -> Set[str]:
        """获取用户的所有权限"""
        permissions = user.permissions.copy()

        # 添加角色权限
        for role_id in user.roles:
            if role_id in self.roles:
                role_permissions = self.roles[role_id].get_all_permissions(self.roles)
                permissions.update(role_permissions)

        return permissions

    def _check_access_policies(self, request: AccessRequest, user_permissions: Set[str]) -> AccessDecision:
        """检查访问策略"""
        applied_policies = []

        for policy in self.policies.values():
            if not policy.is_active:
                continue

            # 检查资源匹配
            if not policy.matches_resource(request.resource):
                continue

            # 检查权限
            if request.permission not in policy.permissions:
                continue

            # 检查访问条件
            if not policy.check_conditions(request.context):
                continue

            applied_policies.append(policy.policy_id)

        allowed = len(applied_policies) > 0

        return AccessDecision(
            request=request,
            allowed=allowed,
            reason="policy_check" if allowed else "no_matching_policy",
            applied_policies=applied_policies
        )

    # =========================================================================
    # 访问策略管理
    # =========================================================================

    def create_access_policy(self, name: str, resource_type: ResourceType,


                             resource_pattern: str, permissions: List[str],
                             conditions: Optional[Dict[str, Any]] = None) -> str:
        """
        创建访问策略

        Args:
            name: 策略名称
            resource_type: 资源类型
            resource_pattern: 资源模式
            permissions: 权限列表
            conditions: 访问条件

        Returns:
            策略ID
        """
        policy_id = f"policy_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

        policy = AccessPolicy(
            policy_id=policy_id,
            name=name,
            resource_type=resource_type,
            resource_pattern=resource_pattern,
            permissions=set(permissions),
            conditions=conditions or {}
        )

        self.policies[policy_id] = policy

        # 审计日志
        if self.enable_audit:
            self._audit_log('create_policy', {
                'policy_id': policy_id,
                'name': name,
                'resource_pattern': resource_pattern,
                'permissions': permissions
            })

        logging.info(f"创建访问策略: {name} ({policy_id})")
        return policy_id

    def update_access_policy(self, policy_id: str, updates: Dict[str, Any]):
        """
        更新访问策略

        Args:
            policy_id: 策略ID
            updates: 更新内容
        """
        if policy_id not in self.policies:
            raise ValueError(f"策略不存在: {policy_id}")

        policy = self.policies[policy_id]

        for key, value in updates.items():
            if hasattr(policy, key):
                setattr(policy, key, value)

        # 审计日志
        if self.enable_audit:
            self._audit_log('update_policy', {
                'policy_id': policy_id,
                'updates': updates
            })

        logging.info(f"更新访问策略: {policy_id}")

    # =========================================================================
    # 缓存管理
    # =========================================================================

    def _clear_user_cache(self, user_id: str):
        """清除用户相关缓存"""
        with self._cache_lock:
            keys_to_remove = [k for k in self._permission_cache.keys()
                              if k.startswith(f"{user_id}:")]
            for key in keys_to_remove:
                del self._permission_cache[key]

    def _clear_role_cache(self, role_id: str):
        """清除角色相关缓存"""
        # 清除所有用户的缓存（因为角色权限可能影响多个用户）
        with self._cache_lock:
            self._permission_cache.clear()

    def clear_permission_cache(self):
        """清除所有权限缓存"""
        with self._cache_lock:
            self._permission_cache.clear()
        logging.info("权限缓存已清除")

    # =========================================================================
    # 审计和监控
    # =========================================================================

    def _audit_log(self, operation: str, details: Dict[str, Any]):
        """审计日志"""
        if not self.enable_audit:
            return

        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'details': details
        }

        try:
            with open(self.audit_log_path, 'a') as f:
                f.write(json.dumps(audit_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            logging.error(f"审计日志写入失败: {e}")

    def get_audit_logs(self, user_id: Optional[str] = None,


                       operation: Optional[str] = None,
                       limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取审计日志

        Args:
            user_id: 用户ID过滤
            operation: 操作类型过滤
            limit: 限制条数

        Returns:
            审计日志列表
        """
        if not self.audit_log_path.exists():
            return []

        logs = []
        try:
            with open(self.audit_log_path, 'r') as f:
                for line in f:
                    if len(logs) >= limit:
                        break
                    try:
                        entry = json.loads(line.strip())

                        # 应用过滤器
                        if user_id and entry.get('details', {}).get('user_id') != user_id:
                            continue
                        if operation and entry.get('operation') != operation:
                            continue

                        logs.append(entry)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logging.error(f"读取审计日志失败: {e}")

        return logs[::-1]  # 返回最新的日志

    def get_access_statistics(self) -> Dict[str, Any]:
        """
        获取访问统计信息

        Returns:
            统计信息
        """
        audit_logs = self.get_audit_logs(operation='access_check', limit=1000)

        total_checks = len(audit_logs)
        allowed_count = sum(1 for log in audit_logs
                            if log.get('details', {}).get('allowed', False))
        denied_count = total_checks - allowed_count

        # 按资源类型统计
        resource_stats = {}
        for log in audit_logs:
            resource = log.get('details', {}).get('resource', '')
            if ':' in resource:
                resource_type = resource.split(':')[0]
                if resource_type not in resource_stats:
                    resource_stats[resource_type] = {'allowed': 0, 'denied': 0}

                if log.get('details', {}).get('allowed', False):
                    resource_stats[resource_type]['allowed'] += 1
                else:
                    resource_stats[resource_type]['denied'] += 1

        return {
            'total_access_checks': total_checks,
            'allowed_access': allowed_count,
            'denied_access': denied_count,
            'allow_rate': allowed_count / total_checks if total_checks > 0 else 0,
            'resource_statistics': resource_stats,
            'cache_size': len(self._permission_cache),
            'timestamp': datetime.now().isoformat()
        }

    # =========================================================================
    # 配置持久化
    # =========================================================================

    def _load_config(self):
        """加载配置"""
        try:
            # 加载用户
            users_file = self.config_path / "users.json"
            if users_file.exists():
                with open(users_file, 'r') as f:
                    users_data = json.load(f)
                    for user_data in users_data.get('users', []):
                        user = User(
                            user_id=user_data['user_id'],
                            username=user_data['username'],
                            email=user_data.get('email'),
                            is_active=user_data.get('is_active', True),
                            created_at=datetime.fromisoformat(user_data['created_at']),
                            last_login=datetime.fromisoformat(
                                user_data['last_login']) if user_data.get('last_login') else None,
                            roles=set(user_data.get('roles', [])),
                            permissions=set(user_data.get('permissions', []))
                        )
                        self.users[user.user_id] = user

            # 加载角色
            roles_file = self.config_path / "roles.json"
            if roles_file.exists():
                with open(roles_file, 'r') as f:
                    roles_data = json.load(f)
                    for role_data in roles_data.get('roles', []):
                        role = Role(
                            role_id=role_data['role_id'],
                            name=role_data['name'],
                            description=role_data.get('description', ''),
                            permissions=set(role_data.get('permissions', [])),
                            parent_roles=set(role_data.get('parent_roles', [])),
                            is_active=role_data.get('is_active', True),
                            created_at=datetime.fromisoformat(role_data['created_at'])
                        )
                        self.roles[role.role_id] = role

            # 加载策略
            policies_file = self.config_path / "policies.json"
            if policies_file.exists():
                with open(policies_file, 'r') as f:
                    policies_data = json.load(f)
                    for policy_data in policies_data.get('policies', []):
                        policy = AccessPolicy(
                            policy_id=policy_data['policy_id'],
                            name=policy_data['name'],
                            resource_type=ResourceType(policy_data['resource_type']),
                            resource_pattern=policy_data['resource_pattern'],
                            permissions=set(policy_data.get('permissions', [])),
                            conditions=policy_data.get('conditions', {}),
                            is_active=policy_data.get('is_active', True),
                            created_at=datetime.fromisoformat(policy_data['created_at'])
                        )
                        self.policies[policy.policy_id] = policy

        except Exception as e:
            logging.error(f"加载配置失败: {e}")

    def _save_config(self):
        """保存配置"""
        try:
            # 保存用户
            users_data = {
                'users': [
                    {
                        'user_id': user.user_id,
                        'username': user.username,
                        'email': user.email,
                        'is_active': user.is_active,
                        'created_at': user.created_at.isoformat(),
                        'last_login': user.last_login.isoformat() if user.last_login else None,
                        'roles': list(user.roles),
                        'permissions': list(user.permissions)
                    }
                    for user in self.users.values()
                ]
            }

            with open(self.config_path / "users.json", 'w') as f:
                json.dump(users_data, f, indent=2, ensure_ascii=False)

            # 保存角色
            roles_data = {
                'roles': [
                    {
                        'role_id': role.role_id,
                        'name': role.name,
                        'description': role.description,
                        'permissions': list(role.permissions),
                        'parent_roles': list(role.parent_roles),
                        'is_active': role.is_active,
                        'created_at': role.created_at.isoformat()
                    }
                    for role in self.roles.values()
                ]
            }

            with open(self.config_path / "roles.json", 'w') as f:
                json.dump(roles_data, f, indent=2, ensure_ascii=False)

            # 保存策略
            policies_data = {
                'policies': [
                    {
                        'policy_id': policy.policy_id,
                        'name': policy.name,
                        'resource_type': policy.resource_type.value,
                        'resource_pattern': policy.resource_pattern,
                        'permissions': list(policy.permissions),
                        'conditions': policy.conditions,
                        'is_active': policy.is_active,
                        'created_at': policy.created_at.isoformat()
                    }
                    for policy in self.policies.values()
                ]
            }

            with open(self.config_path / "policies.json", 'w') as f:
                json.dump(policies_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logging.error(f"保存配置失败: {e}")

    def shutdown(self):
        """关闭访问控制管理器"""
        # 保存配置
        self._save_config()

        # 清除缓存
        self.clear_permission_cache()

        logging.info("访问控制管理器已关闭")

    def __del__(self):
        """析构函数"""
        self.shutdown()
