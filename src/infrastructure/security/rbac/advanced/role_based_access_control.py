"""
基于角色的细粒度权限控制(RBAC)模块

功能：
- 用户-角色-权限三级模型
- 资源级别的权限控制
- 动态权限分配与撤销
- 权限继承与组合
- 上下文感知访问控制
- 权限审计与报告

技术栈：
- dataclasses: 数据结构定义
- functools: 装饰器实现
- typing: 类型提示

作者: Claude
创建日期: 2026-02-21
"""

import functools
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Union
from collections import defaultdict
import hashlib
import uuid

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PermissionType(Enum):
    """权限类型"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"
    EXPORT = "export"
    IMPORT = "import"
    SHARE = "share"


class ResourceType(Enum):
    """资源类型"""
    USER = "user"
    ROLE = "role"
    DATA = "data"
    REPORT = "report"
    STRATEGY = "strategy"
    SYSTEM = "system"
    API = "api"
    FILE = "file"
    DATABASE = "database"


@dataclass
class Permission:
    """权限定义"""
    resource: ResourceType
    action: PermissionType
    conditions: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash((self.resource.value, self.action.value))
    
    def __eq__(self, other):
        if isinstance(other, Permission):
            return self.resource == other.resource and self.action == other.action
        return False
    
    def to_string(self) -> str:
        """转换为字符串表示"""
        return f"{self.resource.value}:{self.action.value}"
    
    @classmethod
    def from_string(cls, permission_str: str) -> 'Permission':
        """从字符串解析"""
        parts = permission_str.split(':')
        if len(parts) != 2:
            raise ValueError(f"无效的权限字符串: {permission_str}")
        return cls(
            resource=ResourceType(parts[0]),
            action=PermissionType(parts[1])
        )


@dataclass
class Role:
    """角色定义"""
    id: str
    name: str
    description: str
    permissions: Set[Permission] = field(default_factory=set)
    parent_roles: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    
    def add_permission(self, permission: Permission) -> None:
        """添加权限"""
        self.permissions.add(permission)
        logger.info(f"角色 {self.name} 添加权限: {permission.to_string()}")
    
    def remove_permission(self, permission: Permission) -> None:
        """移除权限"""
        self.permissions.discard(permission)
        logger.info(f"角色 {self.name} 移除权限: {permission.to_string()}")
    
    def has_permission(self, permission: Permission) -> bool:
        """检查是否有权限"""
        return permission in self.permissions
    
    def inherit_permissions(self, parent_role: 'Role') -> None:
        """继承父角色权限"""
        self.permissions.update(parent_role.permissions)
        self.parent_roles.add(parent_role.id)
        logger.info(f"角色 {self.name} 继承角色 {parent_role.name} 的权限")


@dataclass
class User:
    """用户定义"""
    id: str
    username: str
    email: str
    roles: Set[str] = field(default_factory=set)
    direct_permissions: Set[Permission] = field(default_factory=set)
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    last_login: Optional[datetime] = None
    
    def assign_role(self, role_id: str) -> None:
        """分配角色"""
        self.roles.add(role_id)
        logger.info(f"用户 {self.username} 分配角色: {role_id}")
    
    def revoke_role(self, role_id: str) -> None:
        """撤销角色"""
        self.roles.discard(role_id)
        logger.info(f"用户 {self.username} 撤销角色: {role_id}")
    
    def grant_permission(self, permission: Permission) -> None:
        """授予直接权限"""
        self.direct_permissions.add(permission)
        logger.info(f"用户 {self.username} 获得权限: {permission.to_string()}")
    
    def revoke_permission(self, permission: Permission) -> None:
        """撤销直接权限"""
        self.direct_permissions.discard(permission)
        logger.info(f"用户 {self.username} 撤销权限: {permission.to_string()}")


@dataclass
class AccessContext:
    """访问上下文"""
    user_id: str
    resource_id: str
    resource_type: ResourceType
    action: PermissionType
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    additional_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccessDecision:
    """访问决策"""
    allowed: bool
    reason: str
    context: AccessContext
    evaluated_permissions: List[str] = field(default_factory=list)
    decision_time_ms: float = 0.0


class RBACManager:
    """
    RBAC管理器
    
    管理用户、角色、权限的完整生命周期
    """
    
    def __init__(self):
        """初始化RBAC管理器"""
        self.users: Dict[str, User] = {}
        self.roles: Dict[str, Role] = {}
        self.permissions: Dict[str, Permission] = {}
        self.access_logs: List[Dict[str, Any]] = []
        
        # 初始化默认角色
        self._init_default_roles()
    
    def _init_default_roles(self) -> None:
        """初始化默认角色"""
        # 超级管理员
        admin_role = Role(
            id="role_admin",
            name="超级管理员",
            description="拥有所有权限"
        )
        for resource in ResourceType:
            for action in PermissionType:
                admin_role.add_permission(Permission(resource, action))
        self.roles[admin_role.id] = admin_role
        
        # 数据分析师
        analyst_role = Role(
            id="role_analyst",
            name="数据分析师",
            description="可以读取和分析数据"
        )
        analyst_role.add_permission(Permission(ResourceType.DATA, PermissionType.READ))
        analyst_role.add_permission(Permission(ResourceType.REPORT, PermissionType.READ))
        analyst_role.add_permission(Permission(ResourceType.REPORT, PermissionType.CREATE))
        analyst_role.add_permission(Permission(ResourceType.REPORT, PermissionType.EXPORT))
        self.roles[analyst_role.id] = analyst_role
        
        # 策略交易员
        trader_role = Role(
            id="role_trader",
            name="策略交易员",
            description="可以执行交易策略"
        )
        trader_role.add_permission(Permission(ResourceType.STRATEGY, PermissionType.READ))
        trader_role.add_permission(Permission(ResourceType.STRATEGY, PermissionType.EXECUTE))
        trader_role.add_permission(Permission(ResourceType.DATA, PermissionType.READ))
        self.roles[trader_role.id] = trader_role
        
        # 普通用户
        user_role = Role(
            id="role_user",
            name="普通用户",
            description="基本访问权限"
        )
        user_role.add_permission(Permission(ResourceType.DATA, PermissionType.READ))
        user_role.add_permission(Permission(ResourceType.REPORT, PermissionType.READ))
        self.roles[user_role.id] = user_role
        
        logger.info("默认角色初始化完成")
    
    # ============ 用户管理 ============
    
    def create_user(self, username: str, email: str, 
                   user_id: Optional[str] = None) -> User:
        """
        创建用户
        
        Args:
            username: 用户名
            email: 邮箱
            user_id: 用户ID，如果不提供则自动生成
            
        Returns:
            创建的用户
        """
        user_id = user_id or str(uuid.uuid4())
        user = User(
            id=user_id,
            username=username,
            email=email
        )
        
        # 默认分配普通用户角色
        user.assign_role("role_user")
        
        self.users[user_id] = user
        logger.info(f"创建用户: {username} ({user_id})")
        return user
    
    def get_user(self, user_id: str) -> Optional[User]:
        """获取用户"""
        return self.users.get(user_id)
    
    def delete_user(self, user_id: str) -> bool:
        """删除用户"""
        if user_id in self.users:
            del self.users[user_id]
            logger.info(f"删除用户: {user_id}")
            return True
        return False
    
    # ============ 角色管理 ============
    
    def create_role(self, name: str, description: str,
                   role_id: Optional[str] = None,
                   parent_role_ids: Optional[List[str]] = None) -> Role:
        """
        创建角色
        
        Args:
            name: 角色名
            description: 角色描述
            role_id: 角色ID
            parent_role_ids: 父角色ID列表
            
        Returns:
            创建的角色
        """
        role_id = role_id or f"role_{str(uuid.uuid4())[:8]}"
        role = Role(
            id=role_id,
            name=name,
            description=description
        )
        
        # 继承父角色权限
        if parent_role_ids:
            for parent_id in parent_role_ids:
                parent_role = self.roles.get(parent_id)
                if parent_role:
                    role.inherit_permissions(parent_role)
        
        self.roles[role_id] = role
        logger.info(f"创建角色: {name} ({role_id})")
        return role
    
    def get_role(self, role_id: str) -> Optional[Role]:
        """获取角色"""
        return self.roles.get(role_id)
    
    def delete_role(self, role_id: str) -> bool:
        """删除角色"""
        if role_id in self.roles:
            del self.roles[role_id]
            logger.info(f"删除角色: {role_id}")
            return True
        return False
    
    # ============ 权限检查 ============
    
    def check_access(self, user_id: str, resource_type: ResourceType,
                    action: PermissionType, resource_id: Optional[str] = None,
                    context: Optional[Dict[str, Any]] = None) -> AccessDecision:
        """
        检查访问权限
        
        Args:
            user_id: 用户ID
            resource_type: 资源类型
            action: 操作类型
            resource_id: 资源ID
            context: 附加上下文
            
        Returns:
            访问决策
        """
        import time
        start_time = time.time()
        
        user = self.users.get(user_id)
        if not user:
            return AccessDecision(
                allowed=False,
                reason="用户不存在",
                context=AccessContext(
                    user_id=user_id,
                    resource_id=resource_id or "",
                    resource_type=resource_type,
                    action=action
                )
            )
        
        if not user.is_active:
            return AccessDecision(
                allowed=False,
                reason="用户已被禁用",
                context=AccessContext(
                    user_id=user_id,
                    resource_id=resource_id or "",
                    resource_type=resource_type,
                    action=action
                )
            )
        
        # 构建需要检查的权限
        required_permission = Permission(resource_type, action)
        evaluated_permissions = [required_permission.to_string()]
        
        # 检查直接权限
        if required_permission in user.direct_permissions:
            decision = AccessDecision(
                allowed=True,
                reason="用户有直接权限",
                context=AccessContext(
                    user_id=user_id,
                    resource_id=resource_id or "",
                    resource_type=resource_type,
                    action=action,
                    additional_context=context or {}
                ),
                evaluated_permissions=evaluated_permissions,
                decision_time_ms=(time.time() - start_time) * 1000
            )
            self._log_access(decision)
            return decision
        
        # 检查角色权限
        for role_id in user.roles:
            role = self.roles.get(role_id)
            if role and role.is_active and role.has_permission(required_permission):
                decision = AccessDecision(
                    allowed=True,
                    reason=f"角色 {role.name} 拥有权限",
                    context=AccessContext(
                        user_id=user_id,
                        resource_id=resource_id or "",
                        resource_type=resource_type,
                        action=action,
                        additional_context=context or {}
                    ),
                    evaluated_permissions=evaluated_permissions,
                    decision_time_ms=(time.time() - start_time) * 1000
                )
                self._log_access(decision)
                return decision
        
        # 无权限
        decision = AccessDecision(
            allowed=False,
            reason="无权限访问",
            context=AccessContext(
                user_id=user_id,
                resource_id=resource_id or "",
                resource_type=resource_type,
                action=action,
                additional_context=context or {}
            ),
            evaluated_permissions=evaluated_permissions,
            decision_time_ms=(time.time() - start_time) * 1000
        )
        self._log_access(decision)
        return decision
    
    def _log_access(self, decision: AccessDecision) -> None:
        """记录访问日志"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_id': decision.context.user_id,
            'resource_type': decision.context.resource_type.value,
            'action': decision.context.action.value,
            'resource_id': decision.context.resource_id,
            'allowed': decision.allowed,
            'reason': decision.reason,
            'decision_time_ms': decision.decision_time_ms
        }
        self.access_logs.append(log_entry)
        
        if not decision.allowed:
            logger.warning(f"访问被拒绝: {log_entry}")
    
    # ============ 权限查询 ============
    
    def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """
        获取用户的所有权限
        
        Args:
            user_id: 用户ID
            
        Returns:
            权限集合
        """
        user = self.users.get(user_id)
        if not user:
            return set()
        
        all_permissions = set(user.direct_permissions)
        
        for role_id in user.roles:
            role = self.roles.get(role_id)
            if role and role.is_active:
                all_permissions.update(role.permissions)
        
        return all_permissions
    
    def get_user_permissions_string(self, user_id: str) -> List[str]:
        """获取用户权限字符串列表"""
        permissions = self.get_user_permissions(user_id)
        return [p.to_string() for p in permissions]
    
    def get_role_permissions(self, role_id: str) -> Set[Permission]:
        """获取角色的所有权限"""
        role = self.roles.get(role_id)
        if role:
            return set(role.permissions)
        return set()


# ============ 装饰器 ============

def require_permission(resource_type: ResourceType, action: PermissionType):
    """
    权限检查装饰器
    
    Args:
        resource_type: 资源类型
        action: 操作类型
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 从参数中获取用户ID（假设第一个参数是self，包含rbac_manager和current_user）
            instance = args[0] if args else None
            if not instance:
                raise PermissionError("无法获取实例")
            
            rbac_manager = getattr(instance, 'rbac_manager', None)
            current_user = getattr(instance, 'current_user', None)
            
            if not rbac_manager or not current_user:
                raise PermissionError("RBAC管理器或当前用户未配置")
            
            # 检查权限
            decision = rbac_manager.check_access(
                user_id=current_user.id,
                resource_type=resource_type,
                action=action
            )
            
            if not decision.allowed:
                raise PermissionError(f"权限不足: {decision.reason}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ============ 便捷函数 ============

def create_rbac_manager() -> RBACManager:
    """创建RBAC管理器"""
    return RBACManager()


def generate_permission_matrix() -> Dict[str, List[str]]:
    """
    生成权限矩阵
    
    Returns:
        权限矩阵字典
    """
    matrix = {}
    for resource in ResourceType:
        matrix[resource.value] = [action.value for action in PermissionType]
    return matrix


# 单例实例
_rbac_instance: Optional[RBACManager] = None


def get_rbac_manager() -> RBACManager:
    """
    获取RBAC管理器单例
    
    Returns:
        RBACManager实例
    """
    global _rbac_instance
    if _rbac_instance is None:
        _rbac_instance = RBACManager()
    return _rbac_instance
