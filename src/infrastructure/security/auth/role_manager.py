#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 角色管理器

专门负责角色的创建、管理和权限分配
从AccessControlManager中分离出来，提高代码组织性
"""

import logging
from enum import Enum
from typing import Dict, Iterable, List, Optional, Set, Any, Union
def _to_str_set(values: Optional[Iterable[Union[str, Enum]]]) -> Set[str]:
    result: Set[str] = set()
    if not values:
        return result
    for value in values:
        result.add(value.value if hasattr(value, "value") else str(value))
    return result

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class UserRole(Enum):
    """用户角色枚举"""
    ADMIN = "admin"
    TRADER = "trader"
    ANALYST = "analyst"
    AUDITOR = "auditor"
    GUEST = "guest"


class Permission(Enum):
    """权限枚举"""
    # 系统权限
    SYSTEM_READ = "system:read"
    SYSTEM_WRITE = "system:write"
    SYSTEM_ADMIN = "system:admin"

    # 用户管理权限
    USER_READ = "user:read"
    USER_WRITE = "user:write"
    USER_ADMIN = "user:admin"

    # 交易权限
    TRADE_EXECUTE = "trade:execute"
    TRADE_CANCEL = "trade:cancel"
    ORDER_PLACE = "order:place"
    ORDER_CANCEL = "order:cancel"

    # 数据权限
    DATA_READ = "data:read"
    DATA_WRITE = "data:write"
    DATA_EXPORT = "data:export"

    # 审计权限
    AUDIT_READ = "audit:read"
    AUDIT_WRITE = "audit:write"
    AUDIT_EXPORT = "audit:export"

    # 配置权限
    CONFIG_READ = "config:read"
    CONFIG_WRITE = "config:write"

    # 监控权限
    SYSTEM_MONITOR = "system:monitor"


@dataclass
class RoleDefinition:
    """角色定义"""
    role: UserRole
    name: str
    description: str
    permissions: Set[Permission]
    parent_roles: Set[UserRole]


@dataclass
class Role:
    """角色"""
    role_id: str
    name: str
    description: str = ""
    permissions: Set[str] = field(default_factory=set)
    parent_roles: Set[str] = field(default_factory=set)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_permission(self, permission: str) -> None:
        """添加权限"""
        self.permissions.add(permission)

    def remove_permission(self, permission: str) -> None:
        """移除权限"""
        self.permissions.discard(permission)

    def add_parent_role(self, role_id: str) -> None:
        """添加父角色"""
        self.parent_roles.add(role_id)

    def remove_parent_role(self, role_id: str) -> None:
        """移除父角色"""
        self.parent_roles.discard(role_id)

    def get_all_permissions(self, all_roles: Dict[str, 'Role'], visited: Optional[Set[str]] = None) -> Set[str]:
        """获取所有权限（包括继承的权限）"""
        if visited is None:
            visited = set()

        # 检测循环依赖
        if self.role_id in visited:
            return set()  # 返回空集合避免无限递归

        visited.add(self.role_id)
        all_permissions = self.permissions.copy()

        # 添加父角色的权限
        for parent_role_id in self.parent_roles:
            if parent_role_id in all_roles:
                parent_permissions = all_roles[parent_role_id].get_all_permissions(all_roles, visited)
                all_permissions.update(parent_permissions)

        visited.remove(self.role_id)  # 回溯时移除
        return all_permissions


class RoleManager:
    """角色管理器"""

    def __init__(self) -> None:
        self.roles: Dict[str, Role] = {}
        self.role_definitions: Dict[UserRole, RoleDefinition] = {}
        self._initialize_default_roles()

    def _initialize_default_roles(self) -> None:
        """初始化默认角色"""
        role_definitions = [
            RoleDefinition(
                role=UserRole.ADMIN,
                name="管理员",
                description="系统管理员，拥有所有权限",
                permissions=set(Permission),
                parent_roles=set()
            ),
            RoleDefinition(
                role=UserRole.TRADER,
                name="交易员",
                description="专业交易员，可以执行交易操作",
                permissions={
                    Permission.TRADE_EXECUTE,
                    Permission.TRADE_CANCEL,
                    Permission.ORDER_PLACE,
                    Permission.ORDER_CANCEL,
                    Permission.DATA_READ
                },
                parent_roles=set()
            ),
            RoleDefinition(
                role=UserRole.ANALYST,
                name="分析师",
                description="数据分析师，可以查看和导出数据",
                permissions={
                    Permission.DATA_READ,
                    Permission.DATA_EXPORT,
                    Permission.SYSTEM_MONITOR
                },
                parent_roles=set()
            ),
            RoleDefinition(
                role=UserRole.AUDITOR,
                name="审计员",
                description="审计员，可以查看审计日志",
                permissions={
                    Permission.AUDIT_READ,
                    Permission.AUDIT_EXPORT,
                    Permission.DATA_READ
                },
                parent_roles=set()
            ),
            RoleDefinition(
                role=UserRole.GUEST,
                name="访客",
                description="访客用户，只读权限",
                permissions={Permission.DATA_READ},
                parent_roles=set()
            )
        ]

        for role_def in role_definitions:
            self.role_definitions[role_def.role] = role_def
            # 同时创建实际的Role对象
            role = Role(
                role_id=role_def.role.value,
                name=role_def.name,
                description=role_def.description,
                permissions=_to_str_set({p.value for p in role_def.permissions})
            )
            if role.role_id == UserRole.ADMIN.value:
                role.permissions.update({"admin", "admin:*", "admin:manage"})
            role.metadata["system_role"] = True
            self.roles[role.role_id] = role

    def create_role(self, role_id: str, name: str, description: str = "",
                   permissions: Optional[Set[str]] = None,
                   parent_roles: Optional[Set[str]] = None,
                   metadata: Optional[Dict[str, Any]] = None,
                   allow_replace: bool = False) -> Role:
        """创建角色"""
        if role_id in self.roles:
            if not allow_replace:
                raise ValueError(f"角色已存在: {role_id}")
            role = self.roles[role_id]
            role.name = name
            role.description = description
            if permissions is not None:
                role.permissions = _to_str_set(permissions)
            if parent_roles is not None:
                role.parent_roles = _to_str_set(parent_roles)
            if metadata:
                role.metadata.update(metadata)
            role.is_active = True
            logging.info(f"更新已存在角色: {name} ({role_id})")
            return role

        role = Role(
            role_id=role_id,
            name=name,
            description=description,
            permissions=_to_str_set(permissions or set()),
            parent_roles=_to_str_set(parent_roles or set()),
            metadata=dict(metadata or {})
        )

        self.roles[role_id] = role
        logging.info(f"创建角色: {name} ({role_id})")
        return role

    def update_role(self, role_id: str, **kwargs: Any) -> bool:
        """更新角色"""
        if role_id not in self.roles:
            return False

        role = self.roles[role_id]
        for key, value in kwargs.items():
            if hasattr(role, key):
                setattr(role, key, value)

        logging.info(f"更新角色: {role_id}")
        return True

    def delete_role(self, role_id: str) -> bool:
        """删除角色"""
        if role_id not in self.roles:
            return False

        del self.roles[role_id]
        logging.info(f"删除角色: {role_id}")
        return True

    def get_role(self, role_id: str) -> Optional[Role]:
        """获取角色"""
        return self.roles.get(role_id)

    def get_role_by_name(self, role_name: str) -> Optional[Role]:
        """根据角色名获取角色"""
        for role in self.roles.values():
            if role.name == role_name:
                return role
        return None

    def list_roles(self, active_only: bool = True) -> List[Role]:
        """列出角色"""
        roles = list(self.roles.values())
        if active_only:
            roles = [r for r in roles if r.is_active]
        return roles

    def assign_role_to_user(self, user_id: str, role_id: str) -> bool:
        """为用户分配角色"""
        # 这里只是接口，实际实现需要在UserManager中
        logging.info(f"为用户 {user_id} 分配角色 {role_id}")
        return True

    def revoke_role_from_user(self, user_id: str, role_id: str) -> bool:
        """撤销用户的角色"""
        # 这里只是接口，实际实现需要在UserManager中
        logging.info(f"撤销用户 {user_id} 的角色 {role_id}")
        return True

    def get_user_roles(self, user_id: str) -> List[str]:
        """获取用户的角色"""
        # 这里只是接口，实际实现需要在UserManager中
        return []

    def get_role_permissions(self, role_id: str, include_inherited: bool = True) -> Set[str]:
        """获取角色的所有权限

        Args:
            role_id: 角色ID
            include_inherited: 是否包含继承的权限，默认False

        Returns:
            权限集合
        """
        role = self.get_role(role_id)
        if not role:
            return set()

        if include_inherited:
            return role.get_all_permissions(self.roles)
        else:
            return role.permissions.copy()

    def check_role_permission(self, role_id: str, permission: str) -> bool:
        """检查角色是否有指定权限"""
        role_permissions = self.get_role_permissions(role_id, include_inherited=True)
        return permission in role_permissions

    def get_roles_with_permission(self, permission: str) -> List[str]:
        """获取拥有指定权限的所有角色"""
        roles_with_perm = []
        for role_id, role in self.roles.items():
            if role.metadata.get("system_role"):
                continue
            if self.check_role_permission(role_id, permission):
                roles_with_perm.append(role_id)
        return roles_with_perm

    def _find_role_definition(self, identifier: Any) -> Optional[RoleDefinition]:
        for enum_key, role_def in self.role_definitions.items():
            if identifier == enum_key:
                return role_def
            value = getattr(identifier, "value", None)
            if value and value == enum_key.value:
                return role_def
            if isinstance(identifier, str) and identifier == enum_key.value:
                return role_def
        return None

    def create_role_from_template(self, role_enum: Any) -> Optional[Role]:
        """从模板创建角色"""
        role_def = self._find_role_definition(role_enum)
        if role_def is None:
            logging.warning("未找到角色模板: %s", role_enum)
            return None

        role_id = role_def.role.value
        return self.create_role(
            role_id=role_id,
            name=role_def.name,
            description=role_def.description,
            permissions={p.value for p in role_def.permissions},
            metadata={"system_role": True},
            allow_replace=True
        )

    def get_role_hierarchy(self) -> Dict[str, List[str]]:
        """获取角色层次结构"""
        hierarchy = {}
        for role_id, role in self.roles.items():
            hierarchy[role_id] = list(role.parent_roles)
        return hierarchy

    def validate_role_hierarchy(self) -> List[str]:
        """验证角色层次结构的有效性"""
        issues = []

        # 检查循环依赖
        visited = set()
        recursion_stack = set()

        def check_cycle(role_id: str) -> bool:
            visited.add(role_id)
            recursion_stack.add(role_id)

            role = self.roles.get(role_id)
            if role:
                for parent_id in role.parent_roles:
                    if parent_id not in visited:
                        if check_cycle(parent_id):
                            return True
                    elif parent_id in recursion_stack:
                        return True

            recursion_stack.remove(role_id)
            return False

        for role_id in self.roles:
            if role_id not in visited:
                if check_cycle(role_id):
                    issues.append(f"检测到角色循环依赖: {role_id}")

        # 检查不存在的父角色
        for role_id, role in self.roles.items():
            for parent_id in role.parent_roles:
                if parent_id not in self.roles:
                    issues.append(f"角色 {role_id} 引用不存在的父角色: {parent_id}")

        return issues

    def get_role_stats(self) -> Dict[str, any]:
        """获取角色统计信息"""
        total_roles = len(self.roles)
        active_roles = len([r for r in self.roles.values() if r.is_active])

        permissions_count = {}
        for role in self.roles.values():
            for permission in role.permissions:
                permissions_count[permission] = permissions_count.get(permission, 0) + 1

        return {
            'total_roles': total_roles,
            'active_roles': active_roles,
            'inactive_roles': total_roles - active_roles,
            'permissions_distribution': permissions_count,
            'hierarchy_issues': self.validate_role_hierarchy()
        }
