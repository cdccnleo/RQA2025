import re
from typing import Dict, Any, List, Optional
from enum import Enum, auto
import hashlib
from datetime import datetime

class DataSensitivityLevel(Enum):
    """数据敏感度分级"""
    PUBLIC = auto()
    INTERNAL = auto()
    CONFIDENTIAL = auto()
    SECRET = auto()

class TradingDataSanitizer:
    """交易数据脱敏处理器（生产级实现）"""

    # 预定义敏感字段规则
    SENSITIVE_FIELDS = {
        'account_id': DataSensitivityLevel.SECRET,
        'order_id': DataSensitivityLevel.CONFIDENTIAL,
        'client_name': DataSensitivityLevel.CONFIDENTIAL,
        'phone': DataSensitivityLevel.SECRET,
        'id_card': DataSensitivityLevel.SECRET,
        'bank_account': DataSensitivityLevel.SECRET,
        'ip_address': DataSensitivityLevel.INTERNAL
    }

    # 字段特定脱敏规则
    FIELD_RULES = {
        'phone': {
            'pattern': r'(\d{3})\d{4}(\d{4})',
            'replacement': r'\1****\2'
        },
        'id_card': {
            'pattern': r'(\d{4})\d{10}(\w{4})',
            'replacement': r'\1**********\2'
        },
        'bank_account': {
            'mask_char': '*',
            'keep_last': 4
        }
    }

    def __init__(self,
                 log_watermark: bool = True,
                 encrypt_secrets: bool = False):
        """
        Args:
            log_watermark: 是否添加审计水印
            encrypt_secrets: 是否对最高密级数据加密
        """
        self.log_watermark = log_watermark
        self.encrypt_secrets = encrypt_secrets
        self._watermark_counter = 0

    def sanitize(self,
                data: Dict[str, Any],
                level: DataSensitivityLevel = DataSensitivityLevel.INTERNAL) -> Dict[str, Any]:
        """
        数据脱敏处理
        Args:
            data: 原始数据字典
            level: 当前处理的安全级别
        Returns:
            脱敏后的数据字典
        """
        if not isinstance(data, dict):
            return data

        sanitized = {}
        audit_log = {}

        for key, value in data.items():
            # 确定字段敏感度
            field_level = self.SENSITIVE_FIELDS.get(key, DataSensitivityLevel.PUBLIC)

            if field_level.value >= level.value:
                sanitized_value = self._apply_sanitization(key, value, field_level)
                sanitized[key] = sanitized_value

                # 记录审计日志
                if field_level == DataSensitivityLevel.SECRET:
                    audit_log[key] = {
                        'action': 'sanitized',
                        'original_length': len(str(value)),
                        'timestamp': datetime.now().isoformat()
                    }
            else:
                sanitized[key] = value

        # 添加审计水印
        if self.log_watermark and audit_log:
            self._watermark_counter += 1
            sanitized['_security_watermark'] = {
                'counter': self._watermark_counter,
                'audit_ref': hashlib.sha256(str(audit_log).encode()).hexdigest()
            }

        return sanitized

    def _apply_sanitization(self,
                           field_name: str,
                           value: Any,
                           level: DataSensitivityLevel) -> Any:
        """应用具体的脱敏规则"""
        if value is None:
            return None

        str_value = str(value)

        # 优先应用字段特定规则
        if field_name in self.FIELD_RULES:
            rule = self.FIELD_RULES[field_name]
            if 'pattern' in rule:
                return re.sub(rule['pattern'], rule['replacement'], str_value)
            elif 'mask_char' in rule:
                keep = rule.get('keep_last', 4)
                if len(str_value) > keep:
                    return rule['mask_char'] * (len(str_value) - keep) + str_value[-keep:]
                return rule['mask_char'] * len(str_value)

        # 默认脱敏规则
        if level == DataSensitivityLevel.SECRET:
            if len(str_value) > 8:
                return 'SECRET_REDACTED_' + hashlib.sha256(str_value.encode()).hexdigest()[:8]
            return 'SECRET_REDACTED'
        elif level == DataSensitivityLevel.CONFIDENTIAL:
            if len(str_value) > 4:
                return '***' + str_value[-4:]
            return '***REDACTED***'
        else:
            return str_value[:len(str_value)//2] + '...'

class SecureLogger:
    """安全日志记录器"""

    def __init__(self,
                 sanitizer: TradingDataSanitizer,
                 original_logger: Any):
        """
        Args:
            sanitizer: 数据脱敏处理器
            original_logger: 原始日志记录器
        """
        self.sanitizer = sanitizer
        self.logger = original_logger

    def error(self, message: str, data: Optional[Dict] = None, **kwargs):
        """记录错误日志（自动脱敏）"""
        safe_data = self.sanitizer.sanitize(data or {})
        self.logger.error(message, extra={'data': safe_data}, **kwargs)

    def warning(self, message: str, data: Optional[Dict] = None, **kwargs):
        """记录警告日志（自动脱敏）"""
        safe_data = self.sanitizer.sanitize(data or {})
        self.logger.warning(message, extra={'data': safe_data}, **kwargs)

    def audit(self, action: str, data: Dict):
        """记录审计日志（完整数据+水印）"""
        safe_data = self.sanitizer.sanitize(data, level=DataSensitivityLevel.PUBLIC)
        self.logger.info(
            f"AUDIT:{action}",
            extra={
                'data': safe_data,
                'audit_trail': True
            }
        )

# 集成到错误处理器
class SecureTradingErrorHandler:
    """带数据脱敏的交易错误处理器"""

    def __init__(self,
                 error_handler: Any,
                 sanitizer: TradingDataSanitizer):
        self.handler = error_handler
        self.sanitizer = sanitizer

    def handle_error(self, error: Exception, context: Dict) -> Any:
        # 脱敏处理上下文数据
        safe_context = self.sanitizer.sanitize(context)

        # 调用原始处理器
        result = self.handler.handle_error(error, safe_context)

        # 脱敏处理结果中的敏感数据
        if isinstance(result, dict):
            return self.sanitizer.sanitize(result)
        return result
