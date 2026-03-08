#!/usr/bin/env python3
"""
JWT认证模块
提供API认证和授权功能
"""

import os
import jwt
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
from functools import wraps
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


# 安全提示：这些配置应该从环境变量读取
JWT_SECRET = os.environ.get('JWT_SECRET', '')
JWT_ALGORITHM = os.environ.get('JWT_ALGORITHM', 'HS256')
JWT_EXPIRATION_HOURS = int(os.environ.get('JWT_EXPIRATION_HOURS', '24'))
API_KEY = os.environ.get('API_KEY', '')

# 安全令牌模式
security = HTTPBearer()


class JWTAuth:
    """
    JWT认证管理器
    处理Token的生成、验证和刷新
    """
    
    def __init__(self):
        self.secret = JWT_SECRET
        self.algorithm = JWT_ALGORITHM
        self.expiration_hours = JWT_EXPIRATION_HOURS
        
        if not self.secret:
            raise ValueError("JWT_SECRET environment variable is not set!")
    
    def generate_token(
        self,
        user_id: str,
        username: str,
        roles: Optional[list] = None,
        extra_claims: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        生成JWT Token
        
        Args:
            user_id: 用户ID
            username: 用户名
            roles: 用户角色列表
            extra_claims: 额外声明
            
        Returns:
            JWT Token字符串
        """
        now = datetime.utcnow()
        expiration = now + timedelta(hours=self.expiration_hours)
        
        payload = {
            'sub': user_id,  # subject
            'username': username,
            'iat': now,  # issued at
            'exp': expiration,  # expiration
            'type': 'access'
        }
        
        if roles:
            payload['roles'] = roles
        
        if extra_claims:
            payload.update(extra_claims)
        
        token = jwt.encode(payload, self.secret, algorithm=self.algorithm)
        return token
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """
        验证JWT Token
        
        Args:
            token: JWT Token字符串
            
        Returns:
            Token payload
            
        Raises:
            HTTPException: Token无效或过期
        """
        try:
            payload = jwt.decode(
                token,
                self.secret,
                algorithms=[self.algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=401,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"}
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=401,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"}
            )
    
    def refresh_token(self, token: str) -> str:
        """
        刷新JWT Token
        
        Args:
            token: 当前有效的Token
            
        Returns:
            新的Token
        """
        payload = self.verify_token(token)
        
        # 移除过期时间和签发时间
        payload.pop('exp', None)
        payload.pop('iat', None)
        
        # 生成新Token
        return self.generate_token(
            user_id=payload['sub'],
            username=payload['username'],
            roles=payload.get('roles'),
            extra_claims={k: v for k, v in payload.items() 
                         if k not in ['sub', 'username', 'roles', 'iat', 'exp', 'type']}
        )


class APIKeyAuth:
    """
    API密钥认证
    用于服务间通信和机器对机器认证
    """
    
    def __init__(self):
        self.api_key = API_KEY
        
        if not self.api_key:
            raise ValueError("API_KEY environment variable is not set!")
    
    def verify_api_key(self, api_key: str) -> bool:
        """
        验证API密钥
        
        Args:
            api_key: 提供的API密钥
            
        Returns:
            是否有效
        """
        # 使用恒定时间比较防止时序攻击
        return secrets.compare_digest(api_key, self.api_key)
    
    def generate_api_key(self) -> str:
        """
        生成新的API密钥
        
        Returns:
            新的API密钥
        """
        return secrets.token_urlsafe(32)


# 全局认证实例
jwt_auth = None
api_key_auth = None

def init_auth():
    """初始化认证模块"""
    global jwt_auth, api_key_auth
    
    if JWT_SECRET:
        jwt_auth = JWTAuth()
        print("✅ JWT认证已启用")
    else:
        print("⚠️  JWT_SECRET未设置，JWT认证未启用")
    
    if API_KEY:
        api_key_auth = APIKeyAuth()
        print("✅ API密钥认证已启用")
    else:
        print("⚠️  API_KEY未设置，API密钥认证未启用")


def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)) -> Dict[str, Any]:
    """
    获取当前用户（依赖注入）
    
    用法:
        @app.get("/protected")
        def protected_route(user: dict = Depends(get_current_user)):
            return {"message": f"Hello {user['username']}"}
    """
    if not jwt_auth:
        raise HTTPException(
            status_code=503,
            detail="Authentication service not available"
        )
    
    token = credentials.credentials
    return jwt_auth.verify_token(token)


def require_roles(required_roles: list):
    """
    角色要求装饰器
    
    用法:
        @app.get("/admin")
        @require_roles(["admin"])
        def admin_route(user: dict = Depends(get_current_user)):
            return {"message": "Admin only"}
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 获取当前用户
            user = kwargs.get('user') or args[0] if args else None
            
            if not user:
                raise HTTPException(
                    status_code=401,
                    detail="Authentication required"
                )
            
            user_roles = user.get('roles', [])
            
            # 检查是否有必需角色
            if not any(role in user_roles for role in required_roles):
                raise HTTPException(
                    status_code=403,
                    detail="Insufficient permissions"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def verify_api_key_header(x_api_key: str = Header(..., alias="X-API-Key")) -> bool:
    """
    验证API密钥头部
    
    用法:
        @app.get("/api/service")
        def service_route(api_key_valid: bool = Depends(verify_api_key_header)):
            return {"message": "Service access granted"}
    """
    if not api_key_auth:
        raise HTTPException(
            status_code=503,
            detail="API key authentication not available"
        )
    
    if not api_key_auth.verify_api_key(x_api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    return True


# 密码哈希工具
class PasswordHasher:
    """密码哈希工具类"""
    
    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """
        哈希密码
        
        Args:
            password: 明文密码
            salt: 盐值（可选，自动生成）
            
        Returns:
            (哈希值, 盐值)
        """
        if not salt:
            salt = secrets.token_hex(16)
        
        # 使用PBKDF2进行哈希
        hash_value = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # 迭代次数
        ).hex()
        
        return hash_value, salt
    
    @staticmethod
    def verify_password(password: str, hash_value: str, salt: str) -> bool:
        """
        验证密码
        
        Args:
            password: 明文密码
            hash_value: 哈希值
            salt: 盐值
            
        Returns:
            是否匹配
        """
        computed_hash, _ = PasswordHasher.hash_password(password, salt)
        return secrets.compare_digest(computed_hash, hash_value)


# 示例用法
if __name__ == "__main__":
    # 设置测试环境变量
    os.environ['JWT_SECRET'] = 'test-secret-key-for-development-only'
    os.environ['API_KEY'] = 'test-api-key-for-development-only'
    
    # 初始化认证
    init_auth()
    
    print("\n=== JWT认证测试 ===\n")
    
    # 生成Token
    token = jwt_auth.generate_token(
        user_id="12345",
        username="test_user",
        roles=["user", "admin"]
    )
    print(f"生成的Token: {token[:50]}...")
    
    # 验证Token
    payload = jwt_auth.verify_token(token)
    print(f"验证结果: {payload}")
    
    print("\n=== API密钥测试 ===\n")
    
    # 验证API密钥
    is_valid = api_key_auth.verify_api_key("test-api-key-for-development-only")
    print(f"API密钥验证: {is_valid}")
    
    # 生成新API密钥
    new_api_key = api_key_auth.generate_api_key()
    print(f"新生成的API密钥: {new_api_key}")
    
    print("\n=== 密码哈希测试 ===\n")
    
    # 哈希密码
    password = "my_secure_password123"
    hash_value, salt = PasswordHasher.hash_password(password)
    print(f"密码哈希: {hash_value[:30]}...")
    print(f"盐值: {salt}")
    
    # 验证密码
    is_valid = PasswordHasher.verify_password(password, hash_value, salt)
    print(f"密码验证: {is_valid}")
    
    print("\n=== 测试完成 ===")
