"""认证管理器模块"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# 检查JWT可用性
try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    logger.warning("JWT不可用，认证功能将受限")


class AuthenticationManager:
    """认证管理器
    
    提供JWT令牌认证和授权功能
    """
    
    def __init__(self, jwt_secret: str, jwt_algorithm: str = "HS256"):
        """初始化认证管理器
        
        Args:
            jwt_secret: JWT密钥
            jwt_algorithm: JWT算法，默认HS256
        """
        self.jwt_secret = jwt_secret
        self.jwt_algorithm = jwt_algorithm
        
        if not JWT_AVAILABLE:
            logger.warning("JWT库不可用，认证功能将无法正常工作")
    
    def authenticate(self, token: str) -> Optional[Dict[str, Any]]:
        """验证JWT令牌
        
        Args:
            token: JWT令牌字符串
            
        Returns:
            如果验证成功返回用户信息字典，否则返回None
        """
        if not JWT_AVAILABLE:
            logger.error("JWT库不可用，无法验证令牌")
            return None
        
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            
            # 检查令牌是否过期
            if 'exp' in payload:
                if datetime.fromtimestamp(payload['exp']) < datetime.now():
                    logger.warning("JWT令牌已过期")
                    return None
            
            return payload
        
        except jwt.ExpiredSignatureError:
            logger.warning("JWT令牌已过期")
            return None
        except jwt.InvalidTokenError:
            logger.warning("无效的JWT令牌")
            return None
        except Exception as e:
            logger.error(f"JWT验证失败: {e}")
            return None
    
    def authorize(self, user_info: Dict[str, Any], required_permissions: List[str]) -> bool:
        """授权检查
        
        Args:
            user_info: 用户信息字典
            required_permissions: 所需权限列表
            
        Returns:
            如果有权限返回True，否则返回False
        """
        if not user_info:
            return False
        
        user_permissions = user_info.get('permissions', [])
        
        # 检查是否有任一所需权限
        return any(perm in user_permissions for perm in required_permissions)
    
    def generate_token(self, user_info: Dict[str, Any], expires_in: int = 3600) -> str:
        """生成JWT令牌
        
        Args:
            user_info: 用户信息字典
            expires_in: 过期时间（秒），默认1小时
            
        Returns:
            生成的JWT令牌字符串
        """
        if not JWT_AVAILABLE:
            logger.error("JWT库不可用，无法生成令牌")
            return ""
        
        try:
            payload = {
                **user_info,
                'iat': datetime.now().timestamp(),
                'exp': (datetime.now() + timedelta(seconds=expires_in)).timestamp()
            }
            
            return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
        
        except Exception as e:
            logger.error(f"生成JWT令牌失败: {e}")
            return ""
    
    def refresh_token(self, token: str, expires_in: int = 3600) -> Optional[str]:
        """刷新JWT令牌
        
        Args:
            token: 原JWT令牌
            expires_in: 新令牌过期时间（秒）
            
        Returns:
            新的JWT令牌，如果刷新失败返回None
        """
        user_info = self.authenticate(token)
        if user_info:
            # 移除旧的时间戳
            user_info.pop('iat', None)
            user_info.pop('exp', None)
            return self.generate_token(user_info, expires_in)
        return None
    
    def validate_and_refresh(self, token: str, refresh_threshold: int = 300) -> tuple:
        """验证令牌，如果快过期则刷新
        
        Args:
            token: JWT令牌
            refresh_threshold: 刷新阈值（秒），令牌剩余时间小于此值时刷新
            
        Returns:
            (is_valid, new_token_or_user_info)
            - 如果需要刷新：(True, new_token)
            - 如果有效但不需要刷新：(True, user_info)
            - 如果无效：(False, None)
        """
        user_info = self.authenticate(token)
        if not user_info:
            return False, None
        
        # 检查是否需要刷新
        if 'exp' in user_info:
            exp_time = datetime.fromtimestamp(user_info['exp'])
            time_remaining = (exp_time - datetime.now()).total_seconds()
            
            if time_remaining < refresh_threshold:
                new_token = self.refresh_token(token)
                if new_token:
                    return True, new_token
        
        return True, user_info


__all__ = ['AuthenticationManager']

