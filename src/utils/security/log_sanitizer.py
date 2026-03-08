#!/usr/bin/env python3
"""
日志脱敏模块
用于过滤和脱敏日志中的敏感信息
"""

import re
import logging
import json
from typing import Any, Dict, List, Optional


class SensitiveDataFilter(logging.Filter):
    """
    敏感数据过滤器
    自动检测并脱敏日志中的敏感信息
    """
    
    # 敏感字段模式
    SENSITIVE_PATTERNS = {
        'password': re.compile(r'(password|passwd|pwd)\s*[=:]\s*[^\s&]+', re.IGNORECASE),
        'token': re.compile(r'(token|access_token|refresh_token)\s*[=:]\s*[^\s&]+', re.IGNORECASE),
        'secret': re.compile(r'(secret|api_secret|client_secret)\s*[=:]\s*[^\s&]+', re.IGNORECASE),
        'key': re.compile(r'(api_key|key)\s*[=:]\s*[^\s&]+', re.IGNORECASE),
        'credit_card': re.compile(r'\b(?:\d[ -]*?){13,16}\b'),
        'phone': re.compile(r'\b1[3-9]\d{9}\b'),
        'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        'id_card': re.compile(r'\b\d{17}[\dXx]\b'),
    }
    
    # 需要脱敏的字段名
    SENSITIVE_FIELDS = {
        'password', 'passwd', 'pwd', 'secret', 'token', 'api_key', 'apikey',
        'access_token', 'refresh_token', 'client_secret', 'client_id',
        'private_key', 'secret_key', 'auth_token', 'session_id',
        'credit_card', 'card_number', 'cvv', 'ssn', 'phone', 'mobile',
        'email', 'id_card', 'identity_card', 'bank_account'
    }
    
    def __init__(self, name: str = ""):
        super().__init__(name)
        self.masked_count = 0
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        过滤日志记录中的敏感信息
        """
        # 处理日志消息
        if isinstance(record.msg, str):
            record.msg = self._sanitize_string(record.msg)
        
        # 处理日志参数
        if record.args:
            record.args = self._sanitize_args(record.args)
        
        # 处理extra字段
        if hasattr(record, 'extra') and record.extra:
            record.extra = self._sanitize_dict(record.extra)
        
        return True
    
    def _sanitize_string(self, text: str) -> str:
        """脱敏字符串中的敏感信息"""
        if not isinstance(text, str):
            return text
        
        original_text = text
        
        # 应用所有脱敏模式
        for pattern_name, pattern in self.SENSITIVE_PATTERNS.items():
            if pattern_name in ['credit_card', 'phone', 'email', 'id_card']:
                # 完全脱敏
                text = pattern.sub('[REDACTED]', text)
            else:
                # 保留字段名，脱敏值
                def mask_value(match):
                    full_match = match.group(0)
                    if '=' in full_match:
                        key, value = full_match.split('=', 1)
                        return f"{key}=[MASKED]"
                    elif ':' in full_match:
                        key, value = full_match.split(':', 1)
                        return f"{key}:[MASKED]"
                    return '[MASKED]'
                
                text = pattern.sub(mask_value, text)
        
        # 检测JSON中的敏感字段
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                sanitized_data = self._sanitize_dict(data)
                text = json.dumps(sanitized_data)
        except json.JSONDecodeError:
            pass
        
        if text != original_text:
            self.masked_count += 1
        
        return text
    
    def _sanitize_args(self, args: tuple) -> tuple:
        """脱敏日志参数"""
        sanitized = []
        for arg in args:
            if isinstance(arg, str):
                sanitized.append(self._sanitize_string(arg))
            elif isinstance(arg, dict):
                sanitized.append(self._sanitize_dict(arg))
            elif isinstance(arg, (list, tuple)):
                sanitized.append(self._sanitize_list(arg))
            else:
                sanitized.append(arg)
        return tuple(sanitized)
    
    def _sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """脱敏字典中的敏感字段"""
        if not isinstance(data, dict):
            return data
        
        sanitized = {}
        for key, value in data.items():
            key_lower = key.lower()
            
            # 检查是否为敏感字段
            if any(sensitive in key_lower for sensitive in self.SENSITIVE_FIELDS):
                if isinstance(value, str):
                    # 保留前3位和后3位，中间用***代替
                    if len(value) > 6:
                        sanitized[key] = value[:3] + '***' + value[-3:]
                    else:
                        sanitized[key] = '***'
                else:
                    sanitized[key] = '[MASKED]'
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_dict(value)
            elif isinstance(value, (list, tuple)):
                sanitized[key] = self._sanitize_list(value)
            elif isinstance(value, str):
                sanitized[key] = self._sanitize_string(value)
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _sanitize_list(self, data: List[Any]) -> List[Any]:
        """脱敏列表中的敏感信息"""
        return [self._sanitize_dict(item) if isinstance(item, dict) else item for item in data]


class SecureFormatter(logging.Formatter):
    """
    安全日志格式化器
    在格式化时自动脱敏敏感信息
    """
    
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None):
        super().__init__(fmt, datefmt)
        self.sanitizer = SensitiveDataFilter()
    
    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录"""
        # 先进行脱敏处理
        self.sanitizer.filter(record)
        
        # 然后调用父类的格式化方法
        return super().format(record)


def setup_secure_logging(config_path: Optional[str] = None):
    """
    设置安全的日志配置
    
    Args:
        config_path: 日志配置文件路径
    """
    # 创建敏感数据过滤器
    sensitive_filter = SensitiveDataFilter()
    
    # 获取根日志记录器
    root_logger = logging.getLogger()
    root_logger.addFilter(sensitive_filter)
    
    # 为所有现有处理器添加过滤器
    for handler in root_logger.handlers:
        handler.addFilter(sensitive_filter)
    
    # 设置安全格式化器
    formatter = SecureFormatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    for handler in root_logger.handlers:
        handler.setFormatter(formatter)
    
    print("✅ 安全日志配置已启用")
    print(f"   敏感字段过滤器: {len(sensitive_filter.SENSITIVE_FIELDS)} 个字段")
    print(f"   脱敏模式: {len(sensitive_filter.SENSITIVE_PATTERNS)} 种")


# 示例用法
if __name__ == "__main__":
    # 设置安全日志
    setup_secure_logging()
    
    # 创建测试日志
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # 添加控制台处理器
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    
    # 测试脱敏功能
    print("\n=== 日志脱敏测试 ===\n")
    
    # 测试1: 密码脱敏
    logger.info("User login attempt with password=secret123456")
    
    # 测试2: Token脱敏
    logger.info("API call with token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...")
    
    # 测试3: JSON数据脱敏
    user_data = {
        "username": "john_doe",
        "password": "my_secret_password",
        "email": "john@example.com",
        "phone": "13800138000",
        "api_key": "sk-1234567890abcdef"
    }
    logger.info(f"User data: {user_data}")
    
    # 测试4: 信用卡脱敏
    logger.info("Payment processed with card 4532123456789012")
    
    print("\n=== 测试完成 ===")
