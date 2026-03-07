#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Security异常深度测试"""

import pytest


def test_sensitive_data_access_denied_basic():
    """测试SensitiveDataAccessDenied基础创建"""
    from src.infrastructure.security.plugins.security_error_plugin import SensitiveDataAccessDenied
    
    error = SensitiveDataAccessDenied(
        resource="user_data",
        required_role="admin",
        current_role="user"
    )
    
    assert error.resource == "user_data"
    assert error.required_role == "admin"
    assert error.current_role == "user"


def test_sensitive_data_access_denied_message():
    """测试SensitiveDataAccessDenied错误消息"""
    from src.infrastructure.security.plugins.security_error_plugin import SensitiveDataAccessDenied
    
    error = SensitiveDataAccessDenied(
        resource="confidential_data",
        required_role="superadmin",
        current_role="guest"
    )
    
    error_msg = str(error)
    assert "Access denied" in error_msg
    assert "confidential_data" in error_msg
    assert "superadmin" in error_msg
    assert "guest" in error_msg


def test_sensitive_data_access_denied_is_exception():
    """测试SensitiveDataAccessDenied是Exception子类"""
    from src.infrastructure.security.plugins.security_error_plugin import SensitiveDataAccessDenied
    
    error = SensitiveDataAccessDenied("res", "admin", "user")
    assert isinstance(error, Exception)


def test_sensitive_data_access_denied_can_be_raised():
    """测试SensitiveDataAccessDenied可以被raise"""
    from src.infrastructure.security.plugins.security_error_plugin import SensitiveDataAccessDenied
    
    with pytest.raises(SensitiveDataAccessDenied) as exc_info:
        raise SensitiveDataAccessDenied("secret", "admin", "user")
    
    assert exc_info.value.resource == "secret"
    assert exc_info.value.required_role == "admin"
    assert exc_info.value.current_role == "user"


def test_sensitive_data_access_denied_with_special_chars():
    """测试SensitiveDataAccessDenied处理特殊字符"""
    from src.infrastructure.security.plugins.security_error_plugin import SensitiveDataAccessDenied
    
    error = SensitiveDataAccessDenied(
        resource="data/path/to/file.txt",
        required_role="role:admin:level5",
        current_role="role:user:level1"
    )
    
    assert "data/path/to/file.txt" in str(error)


def test_security_violation_error_basic():
    """测试SecurityViolationError基础创建"""
    from src.infrastructure.security.plugins.security_error_plugin import SecurityViolationError
    
    error = SecurityViolationError(
        violation_type="unauthorized_access",
        details="Attempted to access restricted API"
    )
    
    assert error.violation_type == "unauthorized_access"
    assert error.details == "Attempted to access restricted API"
    assert error.user_id is None


def test_security_violation_error_with_user_id():
    """测试SecurityViolationError包含用户ID"""
    from src.infrastructure.security.plugins.security_error_plugin import SecurityViolationError
    
    error = SecurityViolationError(
        violation_type="sql_injection_attempt",
        details="Malicious SQL detected in input",
        user_id="user_12345"
    )
    
    assert error.violation_type == "sql_injection_attempt"
    assert error.details == "Malicious SQL detected in input"
    assert error.user_id == "user_12345"


def test_security_violation_error_message():
    """测试SecurityViolationError错误消息"""
    from src.infrastructure.security.plugins.security_error_plugin import SecurityViolationError
    
    error = SecurityViolationError(
        violation_type="xss_attack",
        details="Script tag detected"
    )
    
    error_msg = str(error)
    assert "Security violation" in error_msg
    assert "xss_attack" in error_msg
    assert "Script tag detected" in error_msg


def test_security_violation_error_message_with_user():
    """测试SecurityViolationError消息包含用户"""
    from src.infrastructure.security.plugins.security_error_plugin import SecurityViolationError
    
    error = SecurityViolationError(
        violation_type="brute_force",
        details="Multiple failed login attempts",
        user_id="attacker_001"
    )
    
    error_msg = str(error)
    assert "Security violation" in error_msg
    assert "brute_force" in error_msg
    assert "Multiple failed login attempts" in error_msg
    assert "attacker_001" in error_msg


def test_security_violation_error_is_exception():
    """测试SecurityViolationError是Exception子类"""
    from src.infrastructure.security.plugins.security_error_plugin import SecurityViolationError
    
    error = SecurityViolationError("type", "details")
    assert isinstance(error, Exception)


def test_security_violation_error_can_be_raised():
    """测试SecurityViolationError可以被raise"""
    from src.infrastructure.security.plugins.security_error_plugin import SecurityViolationError
    
    with pytest.raises(SecurityViolationError) as exc_info:
        raise SecurityViolationError("csrf", "Invalid token", "user_99")
    
    assert exc_info.value.violation_type == "csrf"
    assert exc_info.value.details == "Invalid token"
    assert exc_info.value.user_id == "user_99"


def test_both_exceptions_independent():
    """测试两个异常类互相独立"""
    from src.infrastructure.security.plugins.security_error_plugin import (
        SensitiveDataAccessDenied,
        SecurityViolationError
    )
    
    error1 = SensitiveDataAccessDenied("res", "admin", "user")
    error2 = SecurityViolationError("type", "details")
    
    assert not isinstance(error1, SecurityViolationError)
    assert not isinstance(error2, SensitiveDataAccessDenied)


def test_exception_in_try_except():
    """测试异常在try-except中的使用"""
    from src.infrastructure.security.plugins.security_error_plugin import (
        SensitiveDataAccessDenied,
        SecurityViolationError
    )
    
    # 测试捕获SensitiveDataAccessDenied
    try:
        raise SensitiveDataAccessDenied("data", "admin", "user")
    except SensitiveDataAccessDenied as e:
        assert e.resource == "data"
    
    # 测试捕获SecurityViolationError
    try:
        raise SecurityViolationError("attack", "detected")
    except SecurityViolationError as e:
        assert e.violation_type == "attack"

