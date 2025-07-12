import pytest
import hmac
import hashlib
import sys
import os

# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..')))

from src.infrastructure.config import SecurityService

class TestSecurityService:
    """安全服务测试"""

    def test_sign_and_verify_config(self):
        """测试配置签名和验证"""
        service = SecurityService()
        config = {"key": "value"}
        
        # 签名配置
        signed = service.sign_config(config)
        assert "config" in signed
        assert "signature" in signed
        assert signed["config"] == config
        
        # 验证签名
        assert service.verify_signature(signed)
        
        # 篡改签名后验证应失败
        signed["signature"] = "invalid"
        assert not service.verify_signature(signed)

    def test_sensitive_operations(self):
        """测试敏感操作检测"""
        service = SecurityService()
        
        # 测试敏感键检测
        assert service.is_sensitive_operation("database.password")
        assert service.is_sensitive_operation("api_keys.main")
        assert not service.is_sensitive_operation("general.setting")
        
        # 测试2FA要求
        assert service.require_2fa("database.password")
        assert not service.require_2fa("general.setting")

    def test_filter_sensitive_data(self):
        """测试敏感数据过滤"""
        service = SecurityService()
        config = {
            "database": {
                "password": "secret",
                "host": "localhost"
            },
            "api_keys": ["key1", "key2"],
            "general": "setting"
        }
        
        filtered = service.filter_sensitive_data(config)
        assert filtered["database"]["password"] == "***"
        assert filtered["database"]["host"] == "localhost"
        assert filtered["api_keys"] == ["***", "***"]
        assert filtered["general"] == "setting"

    def test_detect_malicious_input(self):
        """测试恶意输入检测"""
        service = SecurityService()
        
        # 测试恶意模式
        assert service.detect_malicious_input({"key": "value; DROP TABLE users"})
        assert service.detect_malicious_input({"key": "../etc/passwd"})
        assert service.detect_malicious_input({"key": "<script>alert(1)</script>"})
        
        # 测试正常输入
        assert not service.detect_malicious_input({"key": "normal_value"})
        assert not service.detect_malicious_input({"key": 123})

    def test_audit_and_validation_levels(self):
        """测试审计和验证级别"""
        service = SecurityService()
        assert service.audit_level in ["standard", "strict", "none"]
        assert service.validation_level in ["basic", "full", "none"]
