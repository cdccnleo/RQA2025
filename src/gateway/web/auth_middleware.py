"""
权限控制系统中间件

本模块实现基于RBAC的权限控制系统，满足量化交易系统合规要求：
- QTS-015: 权限控制
- QTS-016: 操作日志记录

功能特性：
- 基于角色的访问控制(RBAC)
- 资源级权限管理
- 操作级权限控制
- 策略数据隔离
- 审计日志集成
"""

import functools
import re
from enum import Enum
from typing import Dict, List, Optional, Set, Callable, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import threading
import jwt
from flask import request, g, jsonify

from .audit_logger import get_audit_logger, AuditCategory, AuditLevel


class Permission(Enum):
    """权限枚举"""
    # 策略执行权限
    STRATEGY_VIEW = "strategy:view"              # 查看策略
    STRATEGY_CREATE = "strategy:create"          # 创建策略
    STRATEGY_UPDATE = "strategy:update"          # 修改策略
    STRATEGY_DELETE = "strategy:delete"          # 删除策略
    STRATEGY_START = "strategy:start"            # 启动策略
    STRATEGY_STOP = "strategy:stop"              # 停止策略
    
    # 交易权限
    TRADING_VIEW = "trading:view"                # 查看交易
    TRADING_EXECUTE = "trading:execute"          # 执行交易
    ORDER_MANAGE = "order:manage"                # 订单管理
    
    # 监控权限
    MONITOR_VIEW = "monitor:view"                # 查看监控
    MONITOR_CONFIGURE = "monitor:configure"      # 配置监控
    
    # 告警权限
    ALERT_VIEW = "alert:view"                    # 查看告警
    ALERT_ACKNOWLEDGE = "alert:acknowledge"      # 确认告警
    ALERT_CONFIGURE = "alert:configure"          # 配置告警
    
    # 风险管理权限
    RISK_VIEW = "risk:view"                      # 查看风险
    RISK_CONFIGURE = "risk:configure"            # 配置风险
    RISK_OVERRIDE = "risk:override"              # 覆盖风险限制
    
    # 系统管理权限
    SYSTEM_VIEW = "system:view"                  # 查看系统
    SYSTEM_CONFIGURE = "system:configure"        # 配置系统
    USER_MANAGE = "user:manage"                  # 用户管理
    AUDIT_VIEW = "audit:view"                    # 查看审计日志
    
    # 数据权限
    DATA_VIEW = "data:view"                      # 查看数据
    DATA_EXPORT = "data:export"                  # 导出数据
    DATA_DELETE = "data:delete"                  # 删除数据


class Role(Enum):
    """角色枚举"""
    ADMIN = "admin"                              # 管理员
    TRADER = "trader"                            # 交易员
    ANALYST = "analyst"                          # 分析师
    VIEWER = "viewer"                            # 只读用户
    RISK_MANAGER = "risk_manager"                # 风控经理
    OPERATOR = "operator"                        # 运维人员


# 角色权限映射
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.ADMIN: set(Permission),  # 管理员拥有所有权限
    
    Role.TRADER: {
        Permission.STRATEGY_VIEW,
        Permission.STRATEGY_CREATE,
        Permission.STRATEGY_UPDATE,
        Permission.STRATEGY_DELETE,
        Permission.STRATEGY_START,
        Permission.STRATEGY_STOP,
        Permission.TRADING_VIEW,
        Permission.TRADING_EXECUTE,
        Permission.ORDER_MANAGE,
        Permission.MONITOR_VIEW,
        Permission.ALERT_VIEW,
        Permission.ALERT_ACKNOWLEDGE,
        Permission.RISK_VIEW,
        Permission.DATA_VIEW,
        Permission.DATA_EXPORT,
    },
    
    Role.ANALYST: {
        Permission.STRATEGY_VIEW,
        Permission.MONITOR_VIEW,
        Permission.ALERT_VIEW,
        Permission.RISK_VIEW,
        Permission.DATA_VIEW,
        Permission.DATA_EXPORT,
    },
    
    Role.VIEWER: {
        Permission.STRATEGY_VIEW,
        Permission.MONITOR_VIEW,
        Permission.ALERT_VIEW,
        Permission.RISK_VIEW,
        Permission.DATA_VIEW,
    },
    
    Role.RISK_MANAGER: {
        Permission.STRATEGY_VIEW,
        Permission.TRADING_VIEW,
        Permission.MONITOR_VIEW,
        Permission.ALERT_VIEW,
        Permission.ALERT_ACKNOWLEDGE,
        Permission.ALERT_CONFIGURE,
        Permission.RISK_VIEW,
        Permission.RISK_CONFIGURE,
        Permission.RISK_OVERRIDE,
        Permission.DATA_VIEW,
        Permission.DATA_EXPORT,
        Permission.AUDIT_VIEW,
    },
    
    Role.OPERATOR: {
        Permission.STRATEGY_VIEW,
        Permission.STRATEGY_START,
        Permission.STRATEGY_STOP,
        Permission.TRADING_VIEW,
        Permission.MONITOR_VIEW,
        Permission.MONITOR_CONFIGURE,
        Permission.ALERT_VIEW,
        Permission.ALERT_ACKNOWLEDGE,
        Permission.SYSTEM_VIEW,
        Permission.SYSTEM_CONFIGURE,
        Permission.DATA_VIEW,
    },
}


@dataclass
class User:
    """用户实体"""
    user_id: str
    username: str
    roles: List[Role] = field(default_factory=list)
    permissions: Set[Permission] = field(default_factory=set)
    allowed_strategies: Optional[Set[str]] = None  # None表示所有策略
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    
    def has_permission(self, permission: Permission) -> bool:
        """检查用户是否有指定权限"""
        if not self.is_active:
            return False
        
        # 直接权限
        if permission in self.permissions:
            return True
        
        # 角色权限
        for role in self.roles:
            if permission in ROLE_PERMISSIONS.get(role, set()):
                return True
        
        return False
    
    def has_any_permission(self, permissions: List[Permission]) -> bool:
        """检查用户是否有任一权限"""
        return any(self.has_permission(p) for p in permissions)
    
    def has_all_permissions(self, permissions: List[Permission]) -> bool:
        """检查用户是否有所有权限"""
        return all(self.has_permission(p) for p in permissions)
    
    def can_access_strategy(self, strategy_id: str) -> bool:
        """检查用户是否可以访问指定策略"""
        if not self.is_active:
            return False
        
        if self.allowed_strategies is None:
            return True
        
        return strategy_id in self.allowed_strategies
    
    def get_all_permissions(self) -> Set[Permission]:
        """获取用户所有权限（包括角色权限）"""
        all_perms = set(self.permissions)
        for role in self.roles:
            all_perms.update(ROLE_PERMISSIONS.get(role, set()))
        return all_perms


@dataclass
class Resource:
    """资源实体"""
    resource_type: str  # strategy, alert, monitor, etc.
    resource_id: str
    owner_id: Optional[str] = None
    allowed_users: Set[str] = field(default_factory=set)
    allowed_roles: Set[Role] = field(default_factory=set)
    is_public: bool = False


class AuthManager:
    """
    认证管理器
    
    单例模式，管理用户认证和权限验证
    """
    
    _instance = None
    _lock = threading.Lock()
    
    JWT_SECRET = "your-secret-key-change-in-production"  # 生产环境需要更改
    JWT_ALGORITHM = "HS256"
    JWT_EXPIRATION_HOURS = 24
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        
        # 用户存储
        self._users: Dict[str, User] = {}
        self._users_lock = threading.RLock()
        
        # 资源存储
        self._resources: Dict[str, Resource] = {}
        self._resources_lock = threading.RLock()
        
        # API权限映射
        self._api_permissions: Dict[str, Permission] = {}
        
        # 初始化默认用户
        self._init_default_users()
    
    def _init_default_users(self):
        """初始化默认用户"""
        # 管理员
        self._users["admin"] = User(
            user_id="admin",
            username="admin",
            roles=[Role.ADMIN],
            is_active=True
        )
        
        # 示例交易员
        self._users["trader1"] = User(
            user_id="trader1",
            username="trader1",
            roles=[Role.TRADER],
            is_active=True
        )
        
        # 示例风控经理
        self._users["risk1"] = User(
            user_id="risk1",
            username="risk1",
            roles=[Role.RISK_MANAGER],
            is_active=True
        )
    
    def register_user(self, user: User) -> bool:
        """注册用户"""
        with self._users_lock:
            if user.user_id in self._users:
                return False
            self._users[user.user_id] = user
            return True
    
    def get_user(self, user_id: str) -> Optional[User]:
        """获取用户"""
        with self._users_lock:
            return self._users.get(user_id)
    
    def authenticate(self, username: str, password: str) -> Optional[str]:
        """
        用户认证
        
        Args:
            username: 用户名
            password: 密码
        
        Returns:
            JWT token或None
        """
        # 简化实现，实际应该验证密码哈希
        user = self.get_user(username)
        if user and user.is_active:
            # 更新最后登录时间
            user.last_login = datetime.now()
            
            # 生成JWT
            token = self._generate_token(user)
            
            # 记录审计日志
            audit = get_audit_logger()
            audit.log(
                level=AuditLevel.INFO,
                category=AuditCategory.USER_ACTION,
                action="user_login",
                message=f"用户登录: {username}",
                user_id=user.user_id,
                result="success"
            )
            
            return token
        
        # 记录失败日志
        audit = get_audit_logger()
        audit.log(
            level=AuditLevel.WARNING,
            category=AuditCategory.USER_ACTION,
            action="user_login",
            message=f"登录失败: {username}",
            result="failure",
            error_message="Invalid credentials"
        )
        
        return None
    
    def _generate_token(self, user: User) -> str:
        """生成JWT token"""
        payload = {
            'user_id': user.user_id,
            'username': user.username,
            'roles': [r.value for r in user.roles],
            'exp': datetime.utcnow().timestamp() + (self.JWT_EXPIRATION_HOURS * 3600),
            'iat': datetime.utcnow().timestamp()
        }
        return jwt.encode(payload, self.JWT_SECRET, algorithm=self.JWT_ALGORITHM)
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """验证JWT token"""
        try:
            payload = jwt.decode(token, self.JWT_SECRET, algorithms=[self.JWT_ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def register_resource(self, resource: Resource):
        """注册资源"""
        with self._resources_lock:
            key = f"{resource.resource_type}:{resource.resource_id}"
            self._resources[key] = resource
    
    def get_resource(self, resource_type: str, resource_id: str) -> Optional[Resource]:
        """获取资源"""
        with self._resources_lock:
            key = f"{resource_type}:{resource_id}"
            return self._resources.get(key)
    
    def check_permission(
        self,
        user: User,
        permission: Permission,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        检查权限
        
        Args:
            user: 用户
            permission: 权限
            resource_type: 资源类型
            resource_id: 资源ID
        
        Returns:
            (是否有权限, 拒绝原因)
        """
        # 检查用户是否激活
        if not user.is_active:
            return False, "User is inactive"
        
        # 检查权限
        if not user.has_permission(permission):
            return False, f"Missing permission: {permission.value}"
        
        # 检查资源访问权限
        if resource_type and resource_id:
            resource = self.get_resource(resource_type, resource_id)
            if resource and not resource.is_public:
                # 检查是否是所有者
                if resource.owner_id == user.user_id:
                    return True, ""
                
                # 检查是否在允许列表中
                if user.user_id not in resource.allowed_users:
                    # 检查角色
                    user_roles = set(user.roles)
                    if not user_roles.intersection(resource.allowed_roles):
                        return False, "Access denied to resource"
        
        return True, ""
    
    def register_api_permission(self, endpoint_pattern: str, permission: Permission):
        """注册API端点权限映射"""
        self._api_permissions[endpoint_pattern] = permission
    
    def get_api_permission(self, endpoint: str) -> Optional[Permission]:
        """获取API端点所需权限"""
        for pattern, permission in self._api_permissions.items():
            if re.match(pattern, endpoint):
                return permission
        return None


# 全局认证管理器实例
_auth_manager: Optional[AuthManager] = None


def get_auth_manager() -> AuthManager:
    """获取全局认证管理器实例"""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
    return _auth_manager


class AuthMiddleware:
    """
    Flask权限控制中间件
    
    提供请求级别的权限验证
    """
    
    def __init__(self, app=None, exempt_routes: Optional[List[str]] = None):
        self.app = app
        self.exempt_routes = exempt_routes or [
            '/api/auth/login',
            '/api/auth/register',
            '/health',
            '/api/public/'
        ]
        
        if app:
            self.init_app(app)
    
    def init_app(self, app):
        """初始化Flask应用"""
        app.before_request(self._before_request)
        app.after_request(self._after_request)
    
    def _before_request(self):
        """请求前处理"""
        # 跳过豁免路由
        for route in self.exempt_routes:
            if request.path.startswith(route) or re.match(route, request.path):
                return
        
        # 获取token
        token = self._extract_token()
        if not token:
            return jsonify({
                'success': False,
                'error': 'Unauthorized',
                'message': 'Missing authentication token'
            }), 401
        
        # 验证token
        auth_manager = get_auth_manager()
        payload = auth_manager.verify_token(token)
        if not payload:
            return jsonify({
                'success': False,
                'error': 'Unauthorized',
                'message': 'Invalid or expired token'
            }), 401
        
        # 获取用户
        user = auth_manager.get_user(payload['user_id'])
        if not user or not user.is_active:
            return jsonify({
                'success': False,
                'error': 'Unauthorized',
                'message': 'User not found or inactive'
            }), 401
        
        # 存储用户信息到请求上下文
        g.user = user
        g.user_id = user.user_id
        g.token_payload = payload
        
        # 检查API权限
        permission = auth_manager.get_api_permission(request.path)
        if permission:
            has_perm, reason = auth_manager.check_permission(user, permission)
            if not has_perm:
                # 记录审计日志
                audit = get_audit_logger()
                audit.log(
                    level=AuditLevel.WARNING,
                    category=AuditCategory.USER_ACTION,
                    action="access_denied",
                    message=f"访问被拒绝: {request.path}",
                    user_id=user.user_id,
                    result="failure",
                    error_message=reason,
                    details={
                        'path': request.path,
                        'method': request.method,
                        'required_permission': permission.value
                    }
                )
                
                return jsonify({
                    'success': False,
                    'error': 'Forbidden',
                    'message': reason
                }), 403
    
    def _after_request(self, response):
        """请求后处理"""
        # 可以在这里添加响应头或日志
        return response
    
    def _extract_token(self) -> Optional[str]:
        """从请求中提取token"""
        # 从Header提取
        auth_header = request.headers.get('Authorization')
        if auth_header:
            parts = auth_header.split()
            if len(parts) == 2 and parts[0].lower() == 'bearer':
                return parts[1]
        
        # 从Cookie提取
        token = request.cookies.get('auth_token')
        if token:
            return token
        
        # 从Query参数提取
        token = request.args.get('token')
        if token:
            return token
        
        return None


# 装饰器

def require_permission(permission: Permission):
    """要求指定权限的装饰器"""
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            user = getattr(g, 'user', None)
            if not user:
                return jsonify({
                    'success': False,
                    'error': 'Unauthorized',
                    'message': 'Authentication required'
                }), 401
            
            auth_manager = get_auth_manager()
            has_perm, reason = auth_manager.check_permission(user, permission)
            
            if not has_perm:
                # 记录审计日志
                audit = get_audit_logger()
                audit.log(
                    level=AuditLevel.WARNING,
                    category=AuditCategory.USER_ACTION,
                    action="permission_denied",
                    message=f"权限不足: {permission.value}",
                    user_id=user.user_id,
                    result="failure",
                    error_message=reason
                )
                
                return jsonify({
                    'success': False,
                    'error': 'Forbidden',
                    'message': reason
                }), 403
            
            return f(*args, **kwargs)
        return wrapper
    return decorator


def require_any_permission(permissions: List[Permission]):
    """要求任一权限的装饰器"""
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            user = getattr(g, 'user', None)
            if not user:
                return jsonify({
                    'success': False,
                    'error': 'Unauthorized',
                    'message': 'Authentication required'
                }), 401
            
            if not user.has_any_permission(permissions):
                perm_names = [p.value for p in permissions]
                return jsonify({
                    'success': False,
                    'error': 'Forbidden',
                    'message': f'Requires any of permissions: {perm_names}'
                }), 403
            
            return f(*args, **kwargs)
        return wrapper
    return decorator


def require_all_permissions(permissions: List[Permission]):
    """要求所有权限的装饰器"""
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            user = getattr(g, 'user', None)
            if not user:
                return jsonify({
                    'success': False,
                    'error': 'Unauthorized',
                    'message': 'Authentication required'
                }), 401
            
            if not user.has_all_permissions(permissions):
                perm_names = [p.value for p in permissions]
                return jsonify({
                    'success': False,
                    'error': 'Forbidden',
                    'message': f'Requires all permissions: {perm_names}'
                }), 403
            
            return f(*args, **kwargs)
        return wrapper
    return decorator


def require_resource_access(resource_type: str, resource_id_param: str = 'id'):
    """要求资源访问权限的装饰器"""
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            user = getattr(g, 'user', None)
            if not user:
                return jsonify({
                    'success': False,
                    'error': 'Unauthorized',
                    'message': 'Authentication required'
                }), 401
            
            resource_id = kwargs.get(resource_id_param)
            if not resource_id:
                return jsonify({
                    'success': False,
                    'error': 'Bad Request',
                    'message': 'Resource ID required'
                }), 400
            
            # 检查策略访问权限
            if resource_type == 'strategy':
                if not user.can_access_strategy(resource_id):
                    return jsonify({
                        'success': False,
                        'error': 'Forbidden',
                        'message': 'Access denied to strategy'
                    }), 403
            
            return f(*args, **kwargs)
        return wrapper
    return decorator


def audit_log(action: str, category: AuditCategory = AuditCategory.USER_ACTION):
    """审计日志装饰器"""
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            user = getattr(g, 'user', None)
            user_id = user.user_id if user else None
            
            start_time = datetime.now()
            result = "success"
            error_message = None
            
            try:
                response = f(*args, **kwargs)
                return response
            except Exception as e:
                result = "failure"
                error_message = str(e)
                raise
            finally:
                duration = (datetime.now() - start_time).total_seconds() * 1000
                
                audit = get_audit_logger()
                audit.log(
                    level=AuditLevel.INFO,
                    category=category,
                    action=action,
                    message=f"API调用: {request.path}",
                    user_id=user_id,
                    result=result,
                    error_message=error_message,
                    duration_ms=duration,
                    details={
                        'path': request.path,
                        'method': request.method,
                        'args': {k: v for k, v in request.args.items()}
                    }
                )
        return wrapper
    return decorator


# 初始化API权限映射
def init_api_permissions():
    """初始化API权限映射"""
    auth_manager = get_auth_manager()
    
    # 策略相关
    auth_manager.register_api_permission(r"/api/strategies$", Permission.STRATEGY_VIEW)
    auth_manager.register_api_permission(r"/api/strategies/\w+$", Permission.STRATEGY_VIEW)
    auth_manager.register_api_permission(r"/api/strategies/create", Permission.STRATEGY_CREATE)
    auth_manager.register_api_permission(r"/api/strategies/\w+/update", Permission.STRATEGY_UPDATE)
    auth_manager.register_api_permission(r"/api/strategies/\w+/delete", Permission.STRATEGY_DELETE)
    auth_manager.register_api_permission(r"/api/strategies/\w+/start", Permission.STRATEGY_START)
    auth_manager.register_api_permission(r"/api/strategies/\w+/stop", Permission.STRATEGY_STOP)
    
    # 监控相关
    auth_manager.register_api_permission(r"/api/monitor", Permission.MONITOR_VIEW)
    auth_manager.register_api_permission(r"/api/monitor/configure", Permission.MONITOR_CONFIGURE)
    
    # 告警相关
    auth_manager.register_api_permission(r"/api/alerts", Permission.ALERT_VIEW)
    auth_manager.register_api_permission(r"/api/alerts/\w+/ack", Permission.ALERT_ACKNOWLEDGE)
    auth_manager.register_api_permission(r"/api/alerts/configure", Permission.ALERT_CONFIGURE)
    
    # 风险相关
    auth_manager.register_api_permission(r"/api/risk", Permission.RISK_VIEW)
    auth_manager.register_api_permission(r"/api/risk/configure", Permission.RISK_CONFIGURE)
    
    # 系统相关
    auth_manager.register_api_permission(r"/api/system", Permission.SYSTEM_VIEW)
    auth_manager.register_api_permission(r"/api/system/configure", Permission.SYSTEM_CONFIGURE)
    auth_manager.register_api_permission(r"/api/users", Permission.USER_MANAGE)
    auth_manager.register_api_permission(r"/api/audit", Permission.AUDIT_VIEW)
    
    # 数据相关
    auth_manager.register_api_permission(r"/api/data", Permission.DATA_VIEW)
    auth_manager.register_api_permission(r"/api/data/export", Permission.DATA_EXPORT)


# 初始化
init_api_permissions()
