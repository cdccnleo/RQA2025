#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
访问控制测试
测试权限验证、身份认证、用户角色管理等访问控制功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
from unittest.mock import patch, MagicMock
from typing import Dict, List, Optional
import hashlib
import hmac
import secrets

from src.infrastructure.config.security.enhanced_secure_config import (
    ConfigAccessControl,
    SecurityConfig,
    AccessRecord
)

# 设置测试超时，避免死锁和无限等待
pytestmark = [
    pytest.mark.timeout(30),  # 30秒超时
    pytest.mark.deadlock_risk,  # 标记为可能存在死锁风险
    pytest.mark.concurrent,  # 并发测试
    pytest.mark.infinite_loop_risk  # 可能存在无限循环风险
]


class TestUserAuthentication:
    """测试用户身份认证"""

    def test_password_hashing(self):
        """测试密码哈希"""
        def hash_password(password: str, salt: str = None) -> tuple:
            """哈希密码"""
            if salt is None:
                salt = secrets.token_hex(16)

            # 使用PBKDF2哈希
            key = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode(),
                salt.encode(),
                100000
            )

            return salt, key.hex()

        # 测试密码哈希
        password = "MySecurePassword123!"
        salt1, hash1 = hash_password(password)
        salt2, hash2 = hash_password(password)

        # 相同密码不同盐应该产生不同哈希
        assert salt1 != salt2
        assert hash1 != hash2

        # 相同密码相同盐应该产生相同哈希
        _, hash1_same_salt = hash_password(password, salt1)
        assert hash1 == hash1_same_salt

    def test_password_validation(self):
        """测试密码强度验证"""
        def validate_password_strength(password: str) -> List[str]:
            """验证密码强度"""
            errors = []

            if len(password) < 8:
                errors.append("密码长度至少8位")

            if not re.search(r'[A-Z]', password):
                errors.append("密码必须包含大写字母")

            if not re.search(r'[a-z]', password):
                errors.append("密码必须包含小写字母")

            if not re.search(r'[0-9]', password):
                errors.append("密码必须包含数字")

            if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
                errors.append("密码必须包含特殊字符")

            # 检查常见弱密码
            weak_passwords = ['password', '123456', 'qwerty', 'admin', 'password123']
            if password.lower() in weak_passwords:
                errors.append("密码过于简单")

            return errors

        import re

        # 测试弱密码
        weak_passwords = [
            "123",
            "password",
            "PASSWORD",
            "password123",
            "123456789"
        ]

        for weak_pwd in weak_passwords:
            errors = validate_password_strength(weak_pwd)
            assert len(errors) > 0, f"弱密码未被检测: {weak_pwd}"

        # 测试强密码
        strong_passwords = [
            "MySecurePass123!",
            "Complex_P@ssw0rd",
            "Str0ng#Pass123"
        ]

        for strong_pwd in strong_passwords:
            errors = validate_password_strength(strong_pwd)
            assert len(errors) == 0, f"强密码被误报: {strong_pwd}, 错误: {errors}"

    def test_session_token_generation(self):
        """测试会话令牌生成"""
        def generate_session_token(user_id: str, expiry_hours: int = 24) -> str:
            """生成会话令牌"""
            timestamp = str(int(time.time()) + expiry_hours * 3600)
            random_part = secrets.token_hex(16)

            # 创建令牌数据
            token_data = f"{user_id}:{timestamp}:{random_part}"

            # 使用HMAC签名
            secret_key = "my_secret_key"  # 实际应用中应该从配置获取
            signature = hmac.new(
                secret_key.encode(),
                token_data.encode(),
                hashlib.sha256
            ).hexdigest()

            return f"{token_data}:{signature}"

        def validate_session_token(token: str) -> Optional[str]:
            """验证会话令牌"""
            try:
                parts = token.split(':')
                if len(parts) != 4:
                    return None

                user_id, timestamp_str, random_part, signature = parts

                # 检查过期
                current_time = int(time.time())
                if current_time > int(timestamp_str):
                    return None

                # 验证签名
                token_data = f"{user_id}:{timestamp_str}:{random_part}"
                secret_key = "my_secret_key"
                expected_signature = hmac.new(
                    secret_key.encode(),
                    token_data.encode(),
                    hashlib.sha256
                ).hexdigest()

                if signature != expected_signature:
                    return None

                return user_id

            except:
                return None

        # 测试令牌生成和验证
        user_id = "user123"
        token = generate_session_token(user_id)

        # 验证有效令牌
        validated_user = validate_session_token(token)
        assert validated_user == user_id

        # 测试无效令牌
        invalid_tokens = [
            "invalid:token:format",
            token.replace(':', ''),  # 修改格式
            token[:-1],  # 修改签名
        ]

        for invalid_token in invalid_tokens:
            validated_user = validate_session_token(invalid_token)
            assert validated_user is None, f"无效令牌被接受: {invalid_token}"

        # 测试过期令牌
        expired_token = generate_session_token(user_id, -1)  # 立即过期
        time.sleep(1)  # 等待过期
        validated_user = validate_session_token(expired_token)
        assert validated_user is None


class TestAccessControl:
    """测试访问控制"""

    def setup_method(self):
        """测试前准备"""
        self.security_config = SecurityConfig()
        self.access_control = ConfigAccessControl(self.security_config)

    def test_permission_checking(self):
        """测试权限检查"""
        # 测试管理员权限
        assert self.access_control.check_access("admin", "read", "config") == True
        assert self.access_control.check_access("admin", "write", "config") == True
        assert self.access_control.check_access("admin", "delete", "config") == True
        assert self.access_control.check_access("admin", "audit", "config") == True

        # 测试操作员权限
        assert self.access_control.check_access("operator", "read", "config") == True
        assert self.access_control.check_access("operator", "write", "config") == True
        assert self.access_control.check_access("operator", "delete", "config") == False
        assert self.access_control.check_access("operator", "audit", "config") == False

        # 测试查看者权限
        assert self.access_control.check_access("viewer", "read", "config") == True
        assert self.access_control.check_access("viewer", "write", "config") == False
        assert self.access_control.check_access("viewer", "delete", "config") == False
        assert self.access_control.check_access("viewer", "audit", "config") == False

    def test_access_logging(self):
        """测试访问日志"""
        # 执行一些访问操作
        self.access_control.check_access("admin", "read", "test_config")
        self.access_control.check_access("viewer", "write", "test_config")  # 应该失败
        self.access_control.check_access("operator", "read", "test_config")

        # 获取访问日志
        logs = self.access_control.get_access_logs()

        assert len(logs) == 3

        # 验证日志内容
        read_log = logs[0]
        assert read_log.user == "admin"
        assert read_log.action == "read"
        assert read_log.resource == "test_config"
        assert read_log.success == True

        write_log = logs[1]
        assert write_log.user == "viewer"
        assert write_log.action == "write"
        assert write_log.success == False

    def test_failed_attempt_lockout(self):
        """测试失败尝试锁定"""
        # 配置锁定参数
        self.security_config.max_access_attempts = 3
        self.security_config.lockout_duration = 2  # 2秒锁定

        test_user = "test_user"

        # 多次失败尝试
        for i in range(3):
            result = self.access_control.check_access(test_user, "write", "config")
            assert result == False

        # 第4次尝试应该被锁定
        result = self.access_control.check_access(test_user, "read", "config")
        assert result == False

        # 等待锁定期结束
        time.sleep(3)

        # 锁定期结束后应该可以访问
        result = self.access_control.check_access(test_user, "read", "config")
        assert result == True

    def test_role_based_access(self):
        """测试基于角色的访问控制"""
        def check_role_access(user: str, resource: str, action: str,
                            roles: Dict[str, List[str]]) -> bool:
            """检查角色访问权限"""
            if user not in roles:
                return False

            user_roles = roles[user]
            required_permissions = {
                "config": ["admin", "operator"],
                "data": ["admin", "operator", "analyst"],
                "reports": ["admin", "operator", "analyst", "viewer"]
            }

            allowed_roles = required_permissions.get(resource, [])
            return any(role in allowed_roles for role in user_roles)

        # 定义用户角色
        user_roles = {
            "alice": ["admin"],
            "bob": ["operator"],
            "charlie": ["analyst"],
            "diana": ["viewer"],
            "eve": []  # 无角色用户
        }

        # 测试管理员访问
        assert check_role_access("alice", "config", "write", user_roles) == True
        assert check_role_access("alice", "data", "read", user_roles) == True
        assert check_role_access("alice", "reports", "read", user_roles) == True

        # 测试操作员访问
        assert check_role_access("bob", "config", "write", user_roles) == True
        assert check_role_access("bob", "data", "read", user_roles) == True
        assert check_role_access("bob", "reports", "read", user_roles) == True

        # 测试分析师访问
        assert check_role_access("charlie", "config", "write", user_roles) == False
        assert check_role_access("charlie", "data", "read", user_roles) == True
        assert check_role_access("charlie", "reports", "read", user_roles) == True

        # 测试查看者访问
        assert check_role_access("diana", "config", "write", user_roles) == False
        assert check_role_access("diana", "data", "read", user_roles) == False
        assert check_role_access("diana", "reports", "read", user_roles) == True

        # 测试无角色用户
        assert check_role_access("eve", "reports", "read", user_roles) == False


class TestResourceProtection:
    """测试资源保护"""

    def test_file_access_control(self):
        """测试文件访问控制"""
        def check_file_access(user: str, file_path: str,
                            permissions: Dict[str, List[str]]) -> bool:
            """检查文件访问权限"""
            # 解析文件路径
            path_parts = file_path.strip('/').split('/')

            # 检查各级目录权限
            current_path = ""
            for part in path_parts:
                current_path += "/" + part
                if current_path in permissions:
                    allowed_users = permissions[current_path]
                    if user not in allowed_users:
                        return False

            return True

        # 定义文件权限
        file_permissions = {
            "/config": ["admin", "operator"],
            "/config/database": ["admin"],
            "/data": ["admin", "operator", "analyst"],
            "/data/stocks": ["admin", "operator", "analyst"],
            "/reports": ["admin", "operator", "analyst", "viewer"],
            "/logs": ["admin"]
        }

        # 测试管理员访问
        assert check_file_access("admin", "/config/database", file_permissions) == True
        assert check_file_access("admin", "/logs", file_permissions) == True

        # 测试操作员访问
        assert check_file_access("operator", "/config", file_permissions) == True
        assert check_file_access("operator", "/config/database", file_permissions) == False  # 无数据库权限
        assert check_file_access("operator", "/logs", file_permissions) == False  # 无日志权限

        # 测试分析师访问
        assert check_file_access("analyst", "/data/stocks", file_permissions) == True
        assert check_file_access("analyst", "/config", file_permissions) == False

        # 测试查看者访问
        assert check_file_access("viewer", "/reports", file_permissions) == True
        assert check_file_access("viewer", "/data", file_permissions) == False

    def test_api_rate_limiting(self):
        """测试API速率限制"""
        def check_rate_limit(user: str, endpoint: str,
                           requests: Dict[str, List[float]],
                           limits: Dict[str, int],
                           window_seconds: int = 60) -> bool:
            """检查速率限制"""
            current_time = time.time()
            key = f"{user}:{endpoint}"

            if key not in requests:
                requests[key] = []

            # 清理过期请求
            requests[key] = [
                t for t in requests[key]
                if current_time - t < window_seconds
            ]

            # 检查是否超过限制
            limit = limits.get(endpoint, 100)
            if len(requests[key]) >= limit:
                return False

            # 记录新请求
            requests[key].append(current_time)
            return True

        # 请求记录
        requests = {}
        # 速率限制
        limits = {
            "/api/config": 10,
            "/api/data": 100,
            "/api/reports": 50
        }

        # 测试正常访问
        for i in range(5):
            assert check_rate_limit("user1", "/api/config", requests, limits) == True

        # 测试达到限制
        for i in range(6):  # 再请求6次，超过10次限制
            result = check_rate_limit("user1", "/api/config", requests, limits)
            if i < 5:
                assert result == True
            else:
                assert result == False

        # 测试不同端点
        assert check_rate_limit("user1", "/api/data", requests, limits) == True

    def test_data_encryption_at_rest(self):
        """测试数据静态加密"""
        def encrypt_data(data: str, key: str) -> str:
            """加密数据"""
            import base64
            from cryptography.fernet import Fernet

            # 使用密钥派生
            key_bytes = hashlib.sha256(key.encode()).digest()
            fernet_key = base64.urlsafe_b64encode(key_bytes)
            cipher = Fernet(fernet_key)

            encrypted = cipher.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted).decode()

        def decrypt_data(encrypted_data: str, key: str) -> str:
            """解密数据"""
            import base64
            from cryptography.fernet import Fernet

            key_bytes = hashlib.sha256(key.encode()).digest()
            fernet_key = base64.urlsafe_b64encode(key_bytes)
            cipher = Fernet(fernet_key)

            encrypted = base64.urlsafe_b64decode(encrypted_data)
            decrypted = cipher.decrypt(encrypted)
            return decrypted.decode()

        test_data = "sensitive financial data"
        encryption_key = "my_secret_key"

        # 加密数据
        encrypted = encrypt_data(test_data, encryption_key)
        assert encrypted != test_data

        # 解密数据
        decrypted = decrypt_data(encrypted, encryption_key)
        assert decrypted == test_data

        # 测试错误密钥
        with pytest.raises(Exception):
            decrypt_data(encrypted, "wrong_key")


class TestAuditTrail:
    """测试审计追踪"""

    def test_audit_log_integrity(self):
        """测试审计日志完整性"""
        def create_audit_entry(action: str, user: str, resource: str,
                             details: Dict = None) -> Dict:
            """创建审计条目"""
            entry = {
                "timestamp": time.time(),
                "action": action,
                "user": user,
                "resource": resource,
                "details": details or {},
                "checksum": ""
            }

            # 计算校验和
            content = f"{entry['timestamp']}|{entry['action']}|{entry['user']}|{entry['resource']}"
            if entry['details']:
                content += f"|{str(sorted(entry['details'].items()))}"

            entry["checksum"] = hashlib.sha256(content.encode()).hexdigest()
            return entry

        def verify_audit_entry(entry: Dict) -> bool:
            """验证审计条目完整性"""
            # 重新计算校验和
            content = f"{entry['timestamp']}|{entry['action']}|{entry['user']}|{entry['resource']}"
            if entry.get('details'):
                content += f"|{str(sorted(entry['details'].items()))}"

            expected_checksum = hashlib.sha256(content.encode()).hexdigest()
            return entry.get("checksum") == expected_checksum

        # 创建审计条目
        entry = create_audit_entry(
            "read",
            "alice",
            "/config/database",
            {"query": "SELECT * FROM users"}
        )

        # 验证完整性
        assert verify_audit_entry(entry) == True

        # 测试篡改检测
        tampered_entry = entry.copy()
        tampered_entry["action"] = "write"  # 篡改动作

        assert verify_audit_entry(tampered_entry) == False

    def test_audit_log_chain(self):
        """测试审计日志链"""
        def create_log_chain(entries: List[Dict]) -> List[Dict]:
            """创建日志链"""
            chained_entries = []

            for i, entry in enumerate(entries):
                entry_copy = entry.copy()
                if i > 0:
                    # 链接到前一个条目
                    prev_entry = chained_entries[-1]
                    entry_copy["previous_hash"] = hashlib.sha256(
                        str(prev_entry).encode()
                    ).hexdigest()

                entry_copy["chain_hash"] = hashlib.sha256(
                    str(entry_copy).encode()
                ).hexdigest()

                chained_entries.append(entry_copy)

            return chained_entries

        def verify_log_chain(chain: List[Dict]) -> bool:
            """验证日志链"""
            for i, entry in enumerate(chain):
                # 验证当前条目哈希
                expected_hash = hashlib.sha256(str({
                    k: v for k, v in entry.items() if k != "chain_hash"
                }).encode()).hexdigest()

                if entry["chain_hash"] != expected_hash:
                    return False

                # 验证链链接
                if i > 0:
                    prev_entry = chain[i-1]
                    expected_prev_hash = hashlib.sha256(str(prev_entry).encode()).hexdigest()
                    if entry.get("previous_hash") != expected_prev_hash:
                        return False

            return True

        # 创建日志条目
        entries = [
            {"action": "login", "user": "alice", "timestamp": 1000},
            {"action": "read", "user": "alice", "resource": "/config", "timestamp": 1001},
            {"action": "write", "user": "alice", "resource": "/config", "timestamp": 1002}
        ]

        # 创建日志链
        chain = create_log_chain(entries)

        # 验证链完整性
        assert verify_log_chain(chain) == True

        # 测试篡改检测
        tampered_chain = chain.copy()
        tampered_chain[1]["action"] = "delete"  # 篡改中间条目

        assert verify_log_chain(tampered_chain) == False


if __name__ == "__main__":
    pytest.main([__file__])
