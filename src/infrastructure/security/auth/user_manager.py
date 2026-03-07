#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 用户管理器

负责用户和角色的管理
分离了AccessControlManager的用户管理职责
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

from src.infrastructure.security.core.types import AccessCheckParams, UserCreationParams


def _to_str_set(values: Optional[Iterable[Union[str, Enum]]]) -> Set[str]:
    result: Set[str] = set()
    if not values:
        return result
    for value in values:
        result.add(value.value if hasattr(value, "value") else str(value))
    return result


@dataclass
class Role:
    """用户角色定义（UserManager内部使用）"""

    role_id: str
    name: str
    description: str = ""
    permissions: Set[str] = field(default_factory=set)
    parent_roles: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)

    def add_permission(self, permission: str) -> None:
        self.permissions.add(permission)

    def remove_permission(self, permission: str) -> None:
        self.permissions.discard(permission)

    def add_parent_role(self, role_name: str) -> None:
        self.parent_roles.add(role_name)

    def remove_parent_role(self, role_name: str) -> None:
        self.parent_roles.discard(role_name)


@dataclass
class User:
    """用户实体"""

    user_id: str
    username: str
    email: Optional[str] = None
    roles: Set[str] = field(default_factory=set)
    permissions: Set[str] = field(default_factory=set)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    password_hash: Optional[str] = None
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None

    def add_role(self, role_name: str) -> None:
        self.roles.add(role_name)

    def remove_role(self, role_name: str) -> None:
        self.roles.discard(role_name)

    def add_permission(self, permission: str) -> None:
        self.permissions.add(permission)

    def remove_permission(self, permission: str) -> None:
        self.permissions.discard(permission)


class UserManager:
    """
    用户管理器

    职责：专门管理用户和角色相关的操作
    包括用户创建、权限分配、角色管理等
    """

    def __init__(self) -> None:
        self.users: Dict[str, User] = {}
        self.roles: Dict[str, Role] = {}
        self._username_index: Dict[str, str] = {}
        self._role_name_index: Dict[str, str] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    # 用户管理
    # ------------------------------------------------------------------ #
    def create_user(
        self,
        params: Optional[UserCreationParams] = None,
        **overrides: Any,
    ) -> User:
        """
        创建或更新用户。

        兼容两种调用方式：
        1. create_user(UserCreationParams(...))
        2. create_user(user_id='u1', username='name', ...)
        """
        with self._lock:
            prepared_params, user_id = self._prepare_user_params(params, overrides)
            roles = _to_str_set(prepared_params.roles)
            permissions = _to_str_set(prepared_params.permissions)

            if user_id and user_id in self.users:
                user = self.users[user_id]
                self._update_user_fields(
                    user,
                    username=prepared_params.username,
                    email=prepared_params.email,
                    roles=roles,
                    permissions=permissions,
                    is_active=prepared_params.is_active,
                    metadata=prepared_params.metadata,
                    password=prepared_params.password,
                )
            else:
                user_id = user_id or self._generate_user_id()
                user = User(
                    user_id=user_id,
                    username=prepared_params.username,
                    email=prepared_params.email,
                    roles=roles or set(),
                    permissions=permissions or set(),
                    is_active=prepared_params.is_active,
                    metadata=dict(prepared_params.metadata),
                )
                if prepared_params.password:
                    user.password_hash = self._hash_password(prepared_params.password)

                self.users[user_id] = user
                self._username_index[user.username] = user_id

            # 记录审计日志
            self._audit_user_creation(user, prepared_params.created_by)
            return user

    def get_user(self, identifier: str) -> Optional[User]:
        """根据用户ID或用户名获取用户"""
        with self._lock:
            user = self.users.get(identifier)
            if user:
                return user

            user_id = self._username_index.get(identifier)
            if user_id:
                return self.users.get(user_id)
            return None

    def update_user(self, user_id: str, **kwargs: any) -> bool:
        """更新用户信息"""
        with self._lock:
            user = self.users.get(user_id)
            if not user:
                return False

            self._update_user_fields(user, **kwargs)
            return True

    def delete_user(self, user_id: str) -> bool:
        """删除用户"""
        with self._lock:
            user = self.users.pop(user_id, None)
            if not user:
                return False

            if user.username in self._username_index:
                del self._username_index[user.username]
            return True

    def list_users(self, active_only: bool = True) -> List[User]:
        """列出用户"""
        with self._lock:
            users = list(self.users.values())
            if active_only:
                users = [u for u in users if u.is_active]
            return users

    # ------------------------------------------------------------------ #
    # 角色管理
    # ------------------------------------------------------------------ #
    def create_role(
        self,
        name: str,
        permissions: Iterable[Union[str, Enum]],
        description: str = "",
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Role:
        """创建或更新角色"""
        with self._lock:
            role_id = self._role_name_index.get(name)
            permissions_set = _to_str_set(permissions)

            if role_id and role_id in self.roles:
                role = self.roles[role_id]
                role.permissions = permissions_set or role.permissions
                role.description = description or role.description
                if metadata:
                    role.metadata.update(metadata)
                return role

            role = Role(
                role_id=self._generate_role_id(),
                name=name,
                description=description,
                permissions=permissions_set,
                metadata=dict(metadata or {}),
            )

            self.roles[role.role_id] = role
            self._role_name_index[role.name] = role.role_id
            return role

    def register_external_role(
        self,
        role_id: str,
        name: str,
        permissions: Iterable[Union[str, Enum]],
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Role:
        """注册外部系统创建的角色，保持与角色管理器同步"""
        with self._lock:
            if role_id in self.roles:
                role = self.roles[role_id]
                role.name = name
                role.permissions = _to_str_set(permissions)
                if description:
                    role.metadata["description"] = description
                if metadata:
                    role.metadata.update(metadata)
                return role

            role = Role(
                role_id=role_id,
                name=name,
                description=description,
                permissions=_to_str_set(permissions),
                metadata=dict(metadata or {}),
            )
            self.roles[role_id] = role
            self._role_name_index[name] = role_id
            return role

    def assign_role_to_user(self, user_id: str, role_identifier: str) -> bool:
        """为用户分配角色（支持角色ID或名称）"""
        with self._lock:
            user = self.users.get(user_id)
            if not user:
                return False

            role = self._resolve_role(role_identifier)
            if role is None:
                return False

            role_name = role.name
            user.add_role(role_name)
            return True

    def revoke_role_from_user(self, user_id: str, role_identifier: str) -> bool:
        """撤销用户角色（支持角色ID或名称）"""
        with self._lock:
            user = self.users.get(user_id)
            if not user:
                return False

            role = self._resolve_role(role_identifier)
            if role is None:
                return False

            role_name = role.name
            if role_name not in user.roles:
                return False

            user.remove_role(role_name)
            return True

    def get_role(self, role_id: str) -> Optional[Role]:
        with self._lock:
            return self.roles.get(role_id)

    def get_role_by_name(self, role_name: str) -> Optional[Role]:
        with self._lock:
            role_id = self._role_name_index.get(role_name)
            if role_id:
                return self.roles.get(role_id)
            return None

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    def _prepare_user_params(
        self,
        params: Optional[UserCreationParams],
        overrides: Dict[str, Any],
    ) -> tuple[UserCreationParams, Optional[str]]:
        if params is None:
            if not overrides:
                raise ValueError("必须提供用户创建参数")
            params = self._build_params_from_kwargs(overrides)
        else:
            metadata_override = overrides.get("metadata")
            base_metadata = dict(params.metadata or {})
            if metadata_override:
                base_metadata.update(metadata_override)
            # 使用 overrides 覆盖 dataclass 字段
            params = UserCreationParams(
                username=overrides.get("username", params.username),
                email=overrides.get("email", params.email),
                roles=_to_str_set(overrides.get("roles", params.roles)),
                permissions=_to_str_set(overrides.get("permissions", params.permissions)),
                is_active=overrides.get("is_active", params.is_active),
                metadata=base_metadata,
                created_by=overrides.get("created_by", params.created_by),
                password=overrides.get("password", params.password),
                require_password_change=overrides.get(
                    "require_password_change", params.require_password_change
                ),
                expiry_date=overrides.get("expiry_date", params.expiry_date),
            )

        user_id = overrides.get("user_id")
        return params, user_id

    def _build_params_from_kwargs(self, data: Dict[str, Any]) -> UserCreationParams:
        data = dict(data)  # 复制以避免污染调用方
        username = data.pop("username")
        email = data.pop("email", None)
        roles = _to_str_set(data.pop("roles", set()))
        permissions = _to_str_set(data.pop("permissions", set()))
        is_active = data.pop("is_active", True)
        metadata_value = data.pop("metadata", None)
        metadata = dict(metadata_value) if metadata_value else {}
        created_by = data.pop("created_by", None)
        password = data.pop("password", None)
        require_password_change = data.pop("require_password_change", False)
        expiry_date = data.pop("expiry_date", None)

        return UserCreationParams(
            username=username,
            email=email,
            roles=roles,
            permissions=permissions,
            is_active=is_active,
            metadata=metadata,
            created_by=created_by,
            password=password,
            require_password_change=require_password_change,
            expiry_date=expiry_date,
        )

    def _update_user_fields(self, user: User, **fields: Any) -> None:
        if "username" in fields and fields["username"]:
            new_username = fields["username"]
            if new_username != user.username:
                if user.username in self._username_index:
                    del self._username_index[user.username]
                self._username_index[new_username] = user.user_id
            user.username = new_username

        if "email" in fields and fields["email"] is not None:
            user.email = fields["email"]

        if "is_active" in fields and fields["is_active"] is not None:
            user.is_active = bool(fields["is_active"])

        if "roles" in fields and fields["roles"] is not None:
            user.roles = _to_str_set(fields["roles"])

        if "permissions" in fields and fields["permissions"] is not None:
            user.permissions = _to_str_set(fields["permissions"])

        if "metadata" in fields and fields["metadata"]:
            user.metadata.update(fields["metadata"])

        if "password" in fields and fields["password"]:
            user.password_hash = self._hash_password(fields["password"])

    def _resolve_role(self, identifier: str) -> Optional[Role]:
        role = self.roles.get(identifier)
        if role:
            return role

        role_id = self._role_name_index.get(identifier)
        if role_id:
            return self.roles.get(role_id)

        # 直接遍历（容错），用于外部直接写入parent_roles的情况
        for role_obj in self.roles.values():
            if role_obj.name == identifier:
                return role_obj
        return None

    def _generate_user_id(self) -> str:
        import uuid

        return f"user_{uuid.uuid4().hex[:8]}"

    def _generate_role_id(self) -> str:
        import uuid

        return f"role_{uuid.uuid4().hex[:8]}"

    def _hash_password(self, password: str) -> str:
        import hashlib

        return hashlib.sha256(password.encode("utf-8")).hexdigest()

    def _audit_user_creation(self, user: User, created_by: Optional[str]) -> None:
        logging.info(
            f"用户创建: {user.username} (ID: {user.user_id}) by {created_by or 'system'}"
        )


class PermissionManager:
    """
    权限管理器

    职责：专门管理权限检查和授权逻辑
    """

    def __init__(self, user_manager: UserManager):
        self.user_manager = user_manager
        self.permission_cache: Dict[str, bool] = {}

    def check_permission(self, params: AccessCheckParams) -> bool:
        """
        检查权限

        Args:
            params: 访问检查参数

        Returns:
            是否有权限
        """
        cache_key = self._get_cache_key(params)
        if params.check_cache and cache_key in self.permission_cache:
            return self.permission_cache[cache_key]

        user = self.user_manager.get_user(params.user_id)
        if not user:
            result = False
        else:
            result = self._check_user_permission(user, params)

        # 缓存结果
        if params.check_cache:
            self.permission_cache[cache_key] = result

        # 记录访问检查日志
        self._log_access_check(params, result)

        return result

    def _check_user_permission(self, user: 'User', params: AccessCheckParams) -> bool:
        """检查用户权限"""
        # 检查角色权限（User类没有直接permissions字段，只通过角色）
        for role_name in user.roles:
            # 查找具有匹配名称的角色
            role = None
            for r in self.user_manager.roles.values():
                if r.name == role_name:
                    role = r
                    break
            if role and params.permission in role.permissions:
                return True

        # 检查继承权限
        if params.include_inherited:
            return self._check_inherited_permissions(user, params)

        return False

    def _check_inherited_permissions(self, user: 'User', params: AccessCheckParams) -> bool:
        """检查继承权限（支持多层继承）"""
        def check_role_permissions(role_name: str, visited: Set[str] = None) -> bool:
            """递归检查角色及其父角色的权限"""
            if visited is None:
                visited = set()
            if role_name in visited:
                return False  # 防止循环继承
            visited.add(role_name)

            # 查找具有匹配名称的角色
            role = None
            for r in self.user_manager.roles.values():
                if r.name == role_name:
                    role = r
                    break

            if role:
                # 检查当前角色的权限
                if params.permission in role.permissions:
                    return True

                # 递归检查父角色的权限
                for parent_role_name in role.parent_roles:
                    if check_role_permissions(parent_role_name, visited.copy()):
                        return True

            return False

        for role_name in user.roles:
            if check_role_permissions(role_name):
                return True

        return False

    def _get_cache_key(self, params: AccessCheckParams) -> str:
        """生成缓存键"""
        return f"{params.user_id}:{params.resource}:{params.permission}:{hash(str(params.context))}"

    def _log_access_check(self, params: AccessCheckParams, result: bool) -> None:
        """记录访问检查日志"""
        # 这里应该调用审计服务
        logging.info(f"权限检查: 用户{params.user_id} 资源{params.resource} 权限{params.permission} 结果:{result}")


