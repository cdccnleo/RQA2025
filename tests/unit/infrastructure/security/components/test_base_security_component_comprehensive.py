#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础安全组件综合测试
测试BaseSecurityComponent和AdvancedSecurity的核心功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from src.infrastructure.security.components.base_security_component import (
    BaseSecurityComponent,
    AdvancedSecurity,
    ISecurityComponent,
    SecurityLevel
)


@pytest.fixture
def base_security():
    """创建基础安全组件实例"""
    return BaseSecurityComponent()


@pytest.fixture
def advanced_security():
    """创建高级安全组件实例"""
    return AdvancedSecurity()


class TestSecurityLevel:
    """测试安全级别枚举"""

    def test_security_levels_exist(self):
        """测试安全级别定义"""
        assert SecurityLevel.LOW.value == "low"
        assert SecurityLevel.MEDIUM.value == "medium"
        assert SecurityLevel.HIGH.value == "high"
        assert SecurityLevel.CRITICAL.value == "critical"

    def test_security_level_values_unique(self):
        """测试安全级别值唯一"""
        values = [level.value for level in SecurityLevel]
        assert len(values) == len(set(values))


class TestISecurityComponent:
    """测试安全组件接口"""

    def test_interface_is_abstract(self):
        """测试接口是抽象的"""
        # 不能直接实例化抽象类
        with pytest.raises(TypeError):
            ISecurityComponent()


class TestBaseSecurityComponentInitialization:
    """测试基础安全组件初始化"""

    def test_initialization_with_default_params(self):
        """测试默认参数初始化"""
        component = BaseSecurityComponent()

        assert component.secret_key is not None
        assert len(component.secret_key) > 0
        assert component.security_level == SecurityLevel.MEDIUM

    def test_initialization_with_custom_key(self):
        """测试自定义密钥初始化"""
        custom_key = "test_secret_key_123"
        component = BaseSecurityComponent(secret_key=custom_key)

        assert component.secret_key == custom_key


class TestBaseSecurityComponentEncryption:
    """测试基础安全组件加密功能"""

    def test_encrypt_decrypt_roundtrip(self, base_security):
        """测试加密解密往返"""
        component = base_security
        test_data = "Hello, World! This is a test message."

        # 加密
        encrypted = component.encrypt(test_data)
        assert encrypted != test_data
        assert isinstance(encrypted, str)

        # 解密
        decrypted = component.decrypt(encrypted)
        assert decrypted == test_data

    def test_encrypt_different_inputs_produce_different_outputs(self, base_security):
        """测试不同输入产生不同输出"""
        component = base_security

        encrypted1 = component.encrypt("test1")
        encrypted2 = component.encrypt("test2")

        assert encrypted1 != encrypted2

    def test_encrypt_empty_string(self, base_security):
        """测试加密空字符串"""
        component = base_security

        encrypted = component.encrypt("")
        decrypted = component.decrypt(encrypted)

        assert decrypted == ""

    def test_encrypt_unicode_string(self, base_security):
        """测试加密Unicode字符串"""
        component = base_security
        test_data = "Unicode测试数据: 你好世界 🌍 αβγδε"

        encrypted = component.encrypt(test_data)
        decrypted = component.decrypt(encrypted)

        assert decrypted == test_data


class TestBaseSecurityComponentHashing:
    """测试基础安全组件哈希功能"""

    def test_hash_consistent(self, base_security):
        """测试哈希一致性"""
        component = base_security
        test_data = "test data for hashing"

        hash1 = component.hash(test_data)
        hash2 = component.hash(test_data)

        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) > 0

    def test_hash_different_inputs(self, base_security):
        """测试不同输入的哈希不同"""
        component = base_security

        hash1 = component.hash("data1")
        hash2 = component.hash("data2")

        assert hash1 != hash2

    def test_verify_hash_valid(self, base_security):
        """测试验证有效哈希"""
        component = base_security
        test_data = "verification test data"

        hash_value = component.hash(test_data)
        result = component.verify_hash(test_data, hash_value)

        assert result is True

    def test_verify_hash_invalid(self, base_security):
        """测试验证无效哈希"""
        component = base_security
        test_data = "verification test data"

        # 使用不同数据的哈希
        wrong_hash = component.hash("different data")
        result = component.verify_hash(test_data, wrong_hash)

        assert result is False

    def test_verify_hash_empty_data(self, base_security):
        """测试验证空数据哈希"""
        component = base_security
        test_data = ""

        hash_value = component.hash(test_data)
        result = component.verify_hash(test_data, hash_value)

        assert result is True


class TestBaseSecurityComponentToken:
    """测试基础安全组件令牌功能"""

    def test_generate_verify_token_roundtrip(self, base_security):
        """测试令牌生成验证往返"""
        component = base_security
        test_data = {"user_id": "user123", "role": "admin", "permissions": ["read", "write"]}

        # 生成令牌
        token = component.generate_token(test_data)
        assert isinstance(token, str)
        assert len(token) > 0

        # 验证令牌
        verified_data = component.verify_token(token)
        assert verified_data is not None
        assert verified_data["user_id"] == test_data["user_id"]
        assert "exp" in verified_data  # 验证包含过期时间

    def test_generate_token_with_expiry(self, base_security):
        """测试生成带过期时间的令牌"""
        component = base_security
        test_data = {"user_id": "user123"}

        # 生成1秒过期的令牌
        token = component.generate_token(test_data, expires_in=1)

        # 立即验证应该成功
        verified_data = component.verify_token(token)
        assert verified_data is not None

        # 等待过期
        time.sleep(2)

        # 验证应该失败
        verified_data = component.verify_token(token)
        assert verified_data is None

    def test_verify_invalid_token(self, base_security):
        """测试验证无效令牌"""
        component = base_security

        result = component.verify_token("invalid_token")
        assert result is None

    def test_verify_expired_token(self, base_security):
        """测试验证过期令牌"""
        component = base_security
        test_data = {"user_id": "user123"}

        # 生成已过期令牌（-1秒）
        token = component.generate_token(test_data, expires_in=-1)

        result = component.verify_token(token)
        assert result is None


class TestBaseSecurityComponentSanitization:
    """测试基础安全组件输入清理功能"""

    def test_sanitize_input_basic(self, base_security):
        """测试基本输入清理"""
        component = base_security

        # 测试包含潜在危险字符的输入
        dangerous_input = "<script>alert('xss')</script>Hello World"
        sanitized = component.sanitize_input(dangerous_input)

        # 应该移除或转义危险字符
        assert "<script>" not in sanitized
        assert "Hello World" in sanitized

    def test_sanitize_input_normal_text(self, base_security):
        """测试正常文本输入清理"""
        component = base_security

        normal_input = "This is normal text with spaces and punctuation!?"
        sanitized = component.sanitize_input(normal_input)

        assert sanitized == normal_input

    def test_sanitize_input_empty(self, base_security):
        """测试空输入清理"""
        component = base_security

        sanitized = component.sanitize_input("")
        assert sanitized == ""


class TestBaseSecurityComponentSecurityLevel:
    """测试基础安全组件安全级别功能"""

    def test_set_security_level(self, base_security):
        """测试设置安全级别"""
        component = base_security

        component.set_security_level(SecurityLevel.HIGH)
        assert component.get_security_level() == SecurityLevel.HIGH

        component.set_security_level(SecurityLevel.CRITICAL)
        assert component.get_security_level() == SecurityLevel.CRITICAL

    def test_get_security_level_default(self, base_security):
        """测试获取默认安全级别"""
        component = base_security

        assert component.get_security_level() == SecurityLevel.MEDIUM


class TestBaseSecurityComponentPasswordValidation:
    """测试基础安全组件密码验证功能"""

    def test_validate_password_strong(self, base_security):
        """测试验证强密码"""
        component = base_security

        strong_password = "MyStr0ngP@ssw0rd123!"
        result = component.validate_password(strong_password)

        assert result["is_valid"] is True
        assert result["score"] == 5  # 满分5分
        assert len(result["issues"]) == 0

    def test_validate_password_weak(self, base_security):
        """测试验证弱密码"""
        component = base_security

        weak_password = "123"
        result = component.validate_password(weak_password)

        assert result["is_valid"] is False
        assert result["score"] == 1  # 只有长度分（虽然短但至少有1位）
        assert len(result["issues"]) > 0

    def test_validate_password_medium(self, base_security):
        """测试验证中等强度密码"""
        component = base_security

        medium_password = "Password123"
        result = component.validate_password(medium_password)

        assert result["is_valid"] is True
        assert result["score"] == 4  # 大写、小写、数字，但缺少特殊字符
        assert len(result["issues"]) == 1

    def test_validate_password_empty(self, base_security):
        """测试验证空密码"""
        component = base_security

        result = component.validate_password("")

        assert result["is_valid"] is False
        assert result["score"] == 0


class TestBaseSecurityComponentPasswordGeneration:
    """测试基础安全组件密码生成功能"""

    def test_generate_secure_password_default_length(self, base_security):
        """测试生成默认长度安全密码"""
        component = base_security

        password = component.generate_secure_password()

        assert isinstance(password, str)
        assert len(password) == 12

        # 验证密码强度
        validation = component.validate_password(password)
        assert validation["is_valid"] is True

    def test_generate_secure_password_custom_length(self, base_security):
        """测试生成自定义长度安全密码"""
        component = base_security

        password = component.generate_secure_password(length=16)

        assert isinstance(password, str)
        assert len(password) == 16

    def test_generate_secure_password_various_lengths(self, base_security):
        """测试生成各种长度安全密码"""
        component = base_security

        for length in [8, 12, 16, 20]:
            password = component.generate_secure_password(length=length)
            assert len(password) == length

            # 验证密码强度
            validation = component.validate_password(password)
            assert validation["is_valid"] is True

    def test_generate_secure_password_uniqueness(self, base_security):
        """测试生成密码的唯一性"""
        component = base_security

        passwords = set()
        for _ in range(10):
            password = component.generate_secure_password()
            passwords.add(password)

        # 所有密码应该唯一（虽然理论上可能有碰撞，但概率极低）
        assert len(passwords) == 10


class TestAdvancedSecurity:
    """测试高级安全组件"""

    def test_advanced_security_inheritance(self, advanced_security):
        """测试高级安全组件继承"""
        component = advanced_security

        # 应该继承基础功能
        assert hasattr(component, 'encrypt')
        assert hasattr(component, 'decrypt')
        assert hasattr(component, 'hash')

    def test_advanced_encrypt_decrypt(self, advanced_security):
        """测试高级加密解密"""
        component = advanced_security
        test_data = "Advanced security test data"

        encrypted = component.encrypt(test_data)
        decrypted = component.decrypt(encrypted)

        assert decrypted == test_data

    def test_rate_limit_check(self, advanced_security):
        """测试速率限制检查"""
        component = advanced_security

        identifier = "test_user"

        # 前5次应该允许
        for i in range(5):
            assert component.check_rate_limit(identifier, max_attempts=5) is True

        # 第6次应该拒绝
        assert component.check_rate_limit(identifier, max_attempts=5) is False

    def test_rate_limit_different_identifiers(self, advanced_security):
        """测试不同标识符的速率限制"""
        component = advanced_security

        # 不同用户应该独立计数
        assert component.check_rate_limit("user1", max_attempts=2) is True
        assert component.check_rate_limit("user1", max_attempts=2) is True
        assert component.check_rate_limit("user1", max_attempts=2) is False  # user1达到限制

        assert component.check_rate_limit("user2", max_attempts=2) is True  # user2仍然允许

    def test_blacklist_management(self, advanced_security):
        """测试黑名单管理"""
        component = advanced_security

        identifier = "suspicious_user"

        # 初始不在黑名单中
        assert not component.is_blacklisted(identifier)

        # 添加到黑名单
        component.add_to_blacklist(identifier)
        assert component.is_blacklisted(identifier)

        # 从黑名单移除
        component.remove_from_blacklist(identifier)
        assert not component.is_blacklisted(identifier)


class TestIntegrationScenarios:
    """测试集成场景"""

    def test_full_security_workflow(self, advanced_security):
        """测试完整安全工作流"""
        component = advanced_security

        # 1. 生成安全密码
        password = component.generate_secure_password()
        assert len(password) >= 12

        # 2. 验证密码强度
        validation = component.validate_password(password)
        assert validation["is_valid"] is True

        # 3. 加密敏感数据
        sensitive_data = f"user_credentials:{password}"
        encrypted = component.encrypt(sensitive_data)

        # 4. 生成访问令牌
        token_data = {"user_id": "test_user", "permissions": ["read", "write"]}
        token = component.generate_token(token_data)

        # 5. 验证令牌
        verified = component.verify_token(token)
        assert verified is not None
        assert verified["user_id"] == "test_user"

        # 6. 解密数据
        decrypted = component.decrypt(encrypted)
        assert decrypted == sensitive_data

    def test_rate_limiting_and_blacklist_integration(self, advanced_security):
        """测试速率限制和黑名单集成"""
        component = advanced_security

        suspicious_ip = "192.168.1.100"

        # 模拟多次失败尝试
        for i in range(10):
            allowed = component.check_rate_limit(suspicious_ip, max_attempts=5)
            if i < 5:
                assert allowed is True
            else:
                assert allowed is False

        # 将IP加入黑名单
        component.add_to_blacklist(suspicious_ip)
        assert component.is_blacklisted(suspicious_ip)

    def test_multi_component_workflow(self, base_security, advanced_security):
        """测试多组件工作流"""
        base_comp = base_security
        advanced_comp = advanced_security

        # 使用基础组件生成和验证令牌
        token_data = {"session_id": "abc123", "user": "test"}
        token = base_comp.generate_token(token_data)
        verified = base_comp.verify_token(token)
        assert verified is not None

        # 使用高级组件进行速率限制检查
        assert advanced_comp.check_rate_limit("test_user", max_attempts=3) is True

        # 跨组件数据一致性
        test_message = "Cross-component test"
        encrypted_by_base = base_comp.encrypt(test_message)
        decrypted_by_advanced = advanced_comp.decrypt(encrypted_by_base)
        assert decrypted_by_advanced == test_message


class TestErrorHandling:
    """测试错误处理"""

    def test_decrypt_invalid_data(self, base_security):
        """测试解密无效数据"""
        component = base_security

        # 解密无效数据应该返回原始数据（安全处理）
        result = component.decrypt("invalid_encrypted_data")
        assert result == "invalid_encrypted_data"

    def test_verify_token_malformed(self, base_security):
        """测试验证格式错误的令牌"""
        component = base_security

        result = component.verify_token("malformed.token.here")
        assert result is None



class TestPerformance:
    """测试性能"""

    def test_encryption_performance(self, base_security):
        """测试加密性能"""
        component = base_security

        test_data = "A" * 1000  # 1KB数据

        start_time = time.time()
        for _ in range(100):
            encrypted = component.encrypt(test_data)
            component.decrypt(encrypted)
        end_time = time.time()

        total_time = end_time - start_time
        avg_time_per_operation = total_time / 200  # 100加密 + 100解密

        # 性能检查（根据环境调整阈值）
        assert avg_time_per_operation < 0.01  # 每操作10ms内

    def test_token_generation_performance(self, base_security):
        """测试令牌生成性能"""
        component = base_security

        token_data = {"user_id": "perf_test", "permissions": ["read", "write", "delete"]}

        start_time = time.time()
        for _ in range(100):
            token = component.generate_token(token_data)
            component.verify_token(token)
        end_time = time.time()

        total_time = end_time - start_time
        avg_time_per_operation = total_time / 200  # 100生成 + 100验证

        # 性能检查
        assert avg_time_per_operation < 0.01  # 每操作10ms内

    def test_password_validation_performance(self, base_security):
        """测试密码验证性能"""
        component = base_security

        passwords = [
            "weak",
            "Medium123",
            "Str0ngP@ssw0rd!2024",
            "V3ryStr0ng@ndL0ngP@ssw0rdWithSp3c1alCh@rs!2024"
        ]

        start_time = time.time()
        for _ in range(100):
            for password in passwords:
                component.validate_password(password)
        end_time = time.time()

        total_time = end_time - start_time
        avg_time_per_validation = total_time / (100 * len(passwords))

        # 性能检查
        assert avg_time_per_validation < 0.001  # 每验证1ms内
