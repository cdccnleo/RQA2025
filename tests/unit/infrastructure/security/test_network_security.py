#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
网络安全测试
测试API安全、HTTPS通信、请求验证等网络安全功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import re
import time
import hashlib
import hmac
import secrets
from unittest.mock import patch, MagicMock
from typing import Dict, Any, Optional, List
import json



# 设置测试超时，避免死锁和无限等待
pytestmark = [
    pytest.mark.timeout(30),  # 30秒超时
    pytest.mark.deadlock_risk,  # 标记为可能存在死锁风险
    pytest.mark.concurrent,  # 并发测试
    pytest.mark.infinite_loop_risk  # 可能存在无限循环风险
]

class TestAPIKeySecurity:
    """测试API密钥安全"""

    def test_api_key_generation(self):
        """测试API密钥生成"""
        def generate_api_key(prefix: str = "sk", length: int = 32) -> str:
            """生成API密钥"""
            random_part = secrets.token_hex(length)
            return f"{prefix}-{random_part}"

        # 测试密钥生成
        key1 = generate_api_key()
        key2 = generate_api_key()

        assert key1.startswith("sk-")
        assert len(key1) == 2 + 1 + 64  # prefix + dash + hex
        assert key1 != key2  # 应该不同

    def test_api_key_validation(self):
        """测试API密钥验证"""
        def validate_api_key_format(api_key: str) -> bool:
            """验证API密钥格式"""
            pattern = r'^(sk|pk)-[a-f0-9]{64}$'
            return bool(re.match(pattern, api_key))

        # 测试有效密钥
        valid_keys = [
            "sk-1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
            "pk-fedcba0987654321fedcba0987654321fedcba0987654321fedcba0987654321"
        ]

        for key in valid_keys:
            assert validate_api_key_format(key) == True, f"有效密钥被拒绝: {key}"

        # 测试无效密钥
        invalid_keys = [
            "invalid-key",
            "sk-invalid",
            "sk-123",  # 太短
            "sk-gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg",  # 非十六进制
            "pk-1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcde",  # 太短
        ]

        for key in invalid_keys:
            assert validate_api_key_format(key) == False, f"无效密钥被接受: {key}"

    def test_api_key_rotation(self):
        """测试API密钥轮转"""
        def rotate_api_key(current_key: str, grace_period: int = 3600) -> Dict[str, Any]:
            """轮转API密钥"""
            # 生成新密钥
            new_key = f"sk-{secrets.token_hex(32)}"

            return {
                "new_key": new_key,
                "old_key": current_key,
                "rotation_time": time.time(),
                "grace_period_end": time.time() + grace_period
            }

        # 测试密钥轮转
        old_key = "sk-1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
        rotation_info = rotate_api_key(old_key)

        assert rotation_info["old_key"] == old_key
        assert rotation_info["new_key"] != old_key
        # 验证新密钥格式
        import re
        assert re.match(r'^(sk|pk)-[a-f0-9]{64}$', rotation_info["new_key"])
        assert rotation_info["grace_period_end"] > rotation_info["rotation_time"]


class TestRequestValidation:
    """测试请求验证"""

    def test_request_signature_verification(self):
        """测试请求签名验证"""
        def create_request_signature(method: str, path: str, body: str,
                                   timestamp: str, secret_key: str) -> str:
            """创建请求签名"""
            message = f"{method}{path}{body}{timestamp}"
            signature = hmac.new(
                secret_key.encode(),
                message.encode(),
                hashlib.sha256
            ).hexdigest()
            return signature

        def verify_request_signature(method: str, path: str, body: str,
                                   timestamp: str, signature: str, secret_key: str,
                                   max_age: int = 300) -> bool:
            """验证请求签名"""
            # 检查时间戳
            current_time = int(time.time())
            request_time = int(timestamp)

            if abs(current_time - request_time) > max_age:
                return False

            # 验证签名
            expected_signature = create_request_signature(
                method, path, body, timestamp, secret_key
            )

            return hmac.compare_digest(signature, expected_signature)

        secret_key = "my_secret_key"

        # 测试有效请求
        method = "POST"
        path = "/api/config"
        body = json.dumps({"key": "value"})
        timestamp = str(int(time.time()))

        signature = create_request_signature(method, path, body, timestamp, secret_key)
        is_valid = verify_request_signature(method, path, body, timestamp, signature, secret_key)

        assert is_valid == True

        # 测试过期请求
        old_timestamp = str(int(time.time()) - 400)  # 400秒前
        old_signature = create_request_signature(method, path, body, old_timestamp, secret_key)
        is_valid = verify_request_signature(method, path, body, old_timestamp, old_signature, secret_key)

        assert is_valid == False

        # 测试篡改的请求
        tampered_body = json.dumps({"key": "tampered"})
        is_valid = verify_request_signature(method, path, tampered_body, timestamp, signature, secret_key)

        assert is_valid == False

    def test_input_sanitization(self):
        """测试输入消毒"""
        def sanitize_request_input(input_data: Dict[str, Any]) -> Dict[str, Any]:
            """消毒请求输入"""
            sanitized = {}

            for key, value in input_data.items():
                # 消毒键名
                clean_key = re.sub(r'[^\w\-_]', '', str(key))

                if isinstance(value, str):
                    # 消毒字符串值
                    clean_value = value.replace('<', '&lt;')
                    clean_value = clean_value.replace('>', '&gt;')
                    clean_value = clean_value.replace('"', '&quot;')
                    clean_value = clean_value.replace("'", '&#x27;')
                    sanitized[clean_key] = clean_value
                elif isinstance(value, dict):
                    # 递归消毒嵌套字典
                    sanitized[clean_key] = sanitize_request_input(value)
                elif isinstance(value, list):
                    # 消毒列表
                    sanitized[clean_key] = [
                        sanitize_request_input({"item": item})["item"] if isinstance(item, dict)
                        else str(item).replace('<', '&lt;').replace('>', '&gt;')
                        for item in value
                    ]
                else:
                    sanitized[clean_key] = value

            return sanitized

        # 测试XSS消毒
        malicious_input = {
            "username": "<script>alert('XSS')</script>",
            "email": "test@example.com",
            "nested": {
                "content": "<img src=x onerror=alert('XSS')>"
            }
        }

        sanitized = sanitize_request_input(malicious_input)

        assert "<script>" not in sanitized["username"]
        assert "<img" not in sanitized["nested"]["content"]
        assert sanitized["email"] == "test@example.com"

    def test_rate_limiting(self):
        """测试速率限制"""
        def check_rate_limit(client_ip: str, endpoint: str,
                           requests: Dict[str, List[float]],
                           limits: Dict[str, int],
                           window_seconds: int = 60) -> tuple:
            """检查速率限制"""
            current_time = time.time()
            key = f"{client_ip}:{endpoint}"

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
                return False, limit - len(requests[key])

            # 记录新请求
            requests[key].append(current_time)
            remaining = limit - len(requests[key])

            return True, remaining

        # 请求记录存储
        requests = {}
        limits = {
            "/api/auth": 5,
            "/api/data": 100,
            "/api/config": 10
        }

        client_ip = "192.168.1.100"

        # 测试正常请求
        for i in range(3):
            allowed, remaining = check_rate_limit(client_ip, "/api/auth", requests, limits)
            assert allowed == True
            assert remaining == (5 - i - 1)

        # 测试达到限制
        for i in range(3):  # 再请求3次，达到5次限制
            allowed, remaining = check_rate_limit(client_ip, "/api/auth", requests, limits)
            if i < 2:  # 前2次仍然允许（总共第4、5次请求）
                assert allowed == True
            else:  # 第3次被拒绝（第6次请求）
                assert allowed == False
                assert remaining == 0  # 达到限制，remaining为0

        # 测试不同端点
        allowed, remaining = check_rate_limit(client_ip, "/api/data", requests, limits)
        assert allowed == True
        assert remaining == 99

    def test_request_size_limits(self):
        """测试请求大小限制"""
        def validate_request_size(content: str, max_size: int = 1024 * 1024) -> bool:
            """验证请求大小"""
            return len(content.encode('utf-8')) <= max_size

        # 测试正常大小请求
        normal_content = "Normal request content"
        assert validate_request_size(normal_content) == True

        # 测试超大请求
        large_content = "x" * (1024 * 1024 + 1)  # 1MB + 1字节
        assert validate_request_size(large_content) == False

        # 测试自定义大小限制
        small_limit = 100
        medium_content = "x" * 50
        large_for_small_limit = "x" * 150

        assert validate_request_size(medium_content, small_limit) == True
        assert validate_request_size(large_for_small_limit, small_limit) == False


class TestHTTPSSecurity:
    """测试HTTPS安全"""

    def test_ssl_certificate_validation(self):
        """测试SSL证书验证"""
        def validate_ssl_certificate(hostname: str, certificate_info: Dict[str, Any]) -> bool:
            """验证SSL证书"""
            # 检查证书是否过期
            current_time = time.time()
            not_before = certificate_info.get("not_before", 0)
            not_after = certificate_info.get("not_after", 0)

            if current_time < not_before or current_time > not_after:
                return False

            # 检查主机名匹配
            cert_hostname = certificate_info.get("subject", {}).get("common_name", "")
            alt_names = certificate_info.get("subject_alt_names", [])

            # 检查是否匹配
            if hostname == cert_hostname:
                return True

            for alt_name in alt_names:
                if hostname == alt_name:
                    return True

            return False

        # 测试有效证书
        valid_cert = {
            "not_before": time.time() - 3600,
            "not_after": time.time() + 3600 * 24 * 365,
            "subject": {"common_name": "api.example.com"},
            "subject_alt_names": ["www.example.com", "secure.example.com"]
        }

        assert validate_ssl_certificate("api.example.com", valid_cert) == True
        assert validate_ssl_certificate("www.example.com", valid_cert) == True

        # 测试过期证书
        expired_cert = {
            "not_before": time.time() - 7200,
            "not_after": time.time() - 3600,  # 已过期
            "subject": {"common_name": "api.example.com"}
        }

        assert validate_ssl_certificate("api.example.com", expired_cert) == False

        # 测试主机名不匹配
        mismatched_cert = {
            "not_before": time.time() - 3600,
            "not_after": time.time() + 3600,
            "subject": {"common_name": "wrong.example.com"}
        }

        assert validate_ssl_certificate("api.example.com", mismatched_cert) == False

    def test_hsts_headers(self):
        """测试HSTS头"""
        def validate_hsts_header(header_value: str) -> Dict[str, Any]:
            """验证HSTS头"""
            result = {
                "valid": False,
                "max_age": 0,
                "include_subdomains": False,
                "preload": False
            }

            if not header_value or not header_value.startswith("max-age="):
                return result

            parts = header_value.split(";")
            for part in parts:
                part = part.strip()
                if part.startswith("max-age="):
                    try:
                        max_age = int(part.split("=")[1])
                        result["max_age"] = max_age
                        result["valid"] = max_age > 0
                    except ValueError:
                        return result
                elif part == "includeSubDomains":
                    result["include_subdomains"] = True
                elif part == "preload":
                    result["preload"] = True

            return result

        # 测试有效HSTS头
        valid_hsts = "max-age=31536000; includeSubDomains; preload"
        result = validate_hsts_header(valid_hsts)

        assert result["valid"] == True
        assert result["max_age"] == 31536000
        assert result["include_subdomains"] == True
        assert result["preload"] == True

        # 测试无效HSTS头
        invalid_hsts = [
            "",
            "invalid-header",
            "max-age=abc",
            "max-age=-1"
        ]

        for invalid in invalid_hsts:
            result = validate_hsts_header(invalid)
            assert result["valid"] == False

    def test_secure_headers(self):
        """测试安全头"""
        def validate_security_headers(headers: Dict[str, str]) -> Dict[str, bool]:
            """验证安全头"""
            security_checks = {
                "content_security_policy": "Content-Security-Policy" in headers,
                "x_frame_options": "X-Frame-Options" in headers,
                "x_content_type_options": "X-Content-Type-Options" in headers,
                "strict_transport_security": "Strict-Transport-Security" in headers,
                "x_xss_protection": "X-XSS-Protection" in headers,
                "referrer_policy": "Referrer-Policy" in headers
            }

            # 验证关键安全头的值
            if "X-Frame-Options" in headers:
                frame_options = headers["X-Frame-Options"].upper()
                security_checks["x_frame_options_valid"] = frame_options in ["DENY", "SAMEORIGIN"]

            if "X-Content-Type-Options" in headers:
                security_checks["x_content_type_options_valid"] = headers["X-Content-Type-Options"] == "nosniff"

            return security_checks

        # 测试完整的安全头
        secure_headers = {
            "Content-Security-Policy": "default-src 'self'",
            "X-Frame-Options": "DENY",
            "X-Content-Type-Options": "nosniff",
            "Strict-Transport-Security": "max-age=31536000",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin"
        }

        result = validate_security_headers(secure_headers)

        assert result["content_security_policy"] == True
        assert result["x_frame_options"] == True
        assert result["x_frame_options_valid"] == True
        assert result["x_content_type_options"] == True
        assert result["x_content_type_options_valid"] == True

        # 测试缺失的安全头
        insecure_headers = {"Content-Type": "application/json"}
        result = validate_security_headers(insecure_headers)

        assert result["content_security_policy"] == False
        assert result["x_frame_options"] == False


class TestWebSocketSecurity:
    """测试WebSocket安全"""

    def test_websocket_origin_validation(self):
        """测试WebSocket源验证"""
        def validate_websocket_origin(origin: str, allowed_origins: List[str]) -> bool:
            """验证WebSocket连接源"""
            try:
                from urllib.parse import urlparse

                parsed = urlparse(origin)
                if not parsed.scheme or not parsed.netloc:
                    return False

                # 只允许HTTPS协议
                if parsed.scheme.lower() != 'https':
                    return False

                # 检查是否在允许的源列表中
                for allowed in allowed_origins:
                    if allowed == "*" or origin == allowed:
                        return True

                    # 支持通配符匹配
                    if allowed.startswith("*."):
                        domain_suffix = allowed[1:]  # 移除*.前缀
                        if parsed.netloc.endswith(domain_suffix):
                            return True

                return False

            except:
                return False

        allowed_origins = [
            "https://app.example.com",
            "*.example.com",
            "https://trusted-site.net"
        ]

        # 测试有效源
        valid_origins = [
            "https://app.example.com",
            "https://api.example.com",
            "https://sub.example.com",
            "https://trusted-site.net"
        ]

        for origin in valid_origins:
            assert validate_websocket_origin(origin, allowed_origins) == True

        # 测试无效源
        invalid_origins = [
            "http://app.example.com",  # 协议不匹配
            "https://malicious.com",   # 不在允许列表
            "file://localhost",        # 无效协议
            ""                         # 空值
        ]

        for origin in invalid_origins:
            assert validate_websocket_origin(origin, allowed_origins) == False

    def test_websocket_message_validation(self):
        """测试WebSocket消息验证"""
        def validate_websocket_message(message: Dict[str, Any],
                                     max_size: int = 1024 * 1024) -> tuple:
            """验证WebSocket消息"""
            # 检查消息大小
            message_size = len(json.dumps(message).encode('utf-8'))
            if message_size > max_size:
                return False, "消息大小超过限制"

            # 检查必需字段
            required_fields = ["type", "timestamp"]
            for field in required_fields:
                if field not in message:
                    return False, f"缺少必需字段: {field}"

            # 验证消息类型
            valid_types = ["subscribe", "unsubscribe", "ping", "pong", "data"]
            if message.get("type") not in valid_types:
                return False, f"无效的消息类型: {message.get('type')}"

            # 验证时间戳
            timestamp = message.get("timestamp", 0)
            current_time = time.time()
            if abs(current_time - timestamp) > 300:  # 5分钟容差
                return False, "时间戳无效"

            return True, "验证通过"

        # 测试有效消息
        valid_message = {
            "type": "subscribe",
            "timestamp": time.time(),
            "channel": "stock_data",
            "symbol": "AAPL"
        }

        is_valid, message = validate_websocket_message(valid_message)
        assert is_valid == True
        assert message == "验证通过"

        # 测试无效消息
        invalid_messages = [
            {"type": "invalid"},  # 缺少时间戳
            {"timestamp": time.time()},  # 缺少类型
            {"type": "subscribe", "timestamp": time.time() - 400},  # 时间戳过期
            {"type": "hack", "timestamp": time.time()},  # 无效类型
            {"type": "subscribe", "timestamp": time.time(), "data": "x" * (1024 * 1024 + 1)}  # 超大消息
        ]

        for invalid_msg in invalid_messages:
            is_valid, error_msg = validate_websocket_message(invalid_msg)
            assert is_valid == False


if __name__ == "__main__":
    pytest.main([__file__])
