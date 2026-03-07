# 基础设施层安全模块 API 文档

## 概述

本文档详细描述了基础设施层安全模块的API接口、使用方法和最佳实践。

## 1. 核心安全模块 API

### 1.1 BaseSecurity 类

#### 主要方法
```python
def encrypt(self, data: str) -> str
def decrypt(self, encrypted_data: str) -> str
def hash(self, data: str) -> str
def verify_hash(self, data: str, hash_value: str) -> bool
def generate_token(self, data: dict, expires_in: int = 3600) -> str
def verify_token(self, token: str) -> dict
```

### 1.2 SecurityUtils 类

#### 主要方法
```python
@staticmethod
def generate_secure_password(length: int = 12) -> str
@staticmethod
def validate_password_strength(password: str) -> dict
@staticmethod
def hash_password(password: str, salt: str) -> str
@staticmethod
def verify_password(password: str, salt: str, hashed: str) -> bool
@staticmethod
def generate_api_key(length: int = 32) -> str
@staticmethod
def generate_otp(length: int = 6) -> str
```

### 1.3 SecurityFactory 类

#### 主要方法
```python
@classmethod
def create_security_component(cls, component_type: str, config: dict = None) -> object
@classmethod
def create_default_security_stack(cls) -> dict
```

## 2. 服务层安全组件 API

### 2.1 DataSanitizer 类
### 2.2 AuthManager 类
### 2.3 EnhancedSecurityManager 类
### 2.4 SecurityAuditor 类

## 3. 配置层安全组件 API

### 3.1 SecurityManager 类

#### 主要方法
```python
def add_user(self, username: str, password: str, role: str = "user") -> bool
def authenticate_user(self, username: str, password: str) -> bool
def grant_permission(self, username: str, resource: str, action: str) -> bool
def check_permission(self, username: str, resource: str, action: str) -> bool
```

---

**文档版本**: 1.0  
**更新时间**: 2025-01-27
