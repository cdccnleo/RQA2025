#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安全工具集

提供安全相关的工具函数和常量
"""

import hashlib
import secrets
from typing import Optional, Dict, Any


class SecurityConstants:
    """安全常量"""
    MIN_PASSWORD_LENGTH = 8
    MAX_PASSWORD_LENGTH = 128
    DEFAULT_HASH_ALGORITHM = "sha256"
    DEFAULT_SALT_LENGTH = 16
    TOKEN_EXPIRY_SECONDS = 3600


class SecurityUtils:
    """安全工具类"""
    
    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None) -> Dict[str, str]:
        """哈希密码"""
        if salt is None:
            salt = secrets.token_hex(SecurityConstants.DEFAULT_SALT_LENGTH)
        
        hashed = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        ).hex()
        
        return {
            "hash": hashed,
            "salt": salt
        }
    
    @staticmethod
    def verify_password(password: str, hashed: str, salt: str) -> bool:
        """验证密码"""
        result = SecurityUtils.hash_password(password, salt)
        return result["hash"] == hashed
    
    @staticmethod
    def generate_token(length: int = 32) -> str:
        """生成随机token"""
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def validate_password_strength(password: str) -> Dict[str, Any]:
        """验证密码强度"""
        length_ok = len(password) >= SecurityConstants.MIN_PASSWORD_LENGTH
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(not c.isalnum() for c in password)
        
        score = sum([length_ok, has_upper, has_lower, has_digit, has_special])
        
        return {
            "is_strong": score >= 4,
            "score": score,
            "length_ok": length_ok,
            "has_upper": has_upper,
            "has_lower": has_lower,
            "has_digit": has_digit,
            "has_special": has_special
        }


__all__ = [
    "SecurityConstants",
    "SecurityUtils"
]

