"""量化专用日志过滤器"""
import logging
import re


class QuantFilter(logging.Filter):
    """为日志记录添加量化交易专用字段"""

    SENSITIVE_PATTERNS = [
        r'\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b',  # 信用卡号
        r'\b\d{6}\d{2}[01]\d[0-3]\d\d{4}\b',  # 身份证号
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # 邮箱
        r'\b1[3-9]\d{9}\b',  # 手机号
        r'\b\d{17}[\dXx]\b'  # 二代身份证号
    ]
    
    def filter(self, record):
        """过滤方法，添加量化专用字段并过滤敏感信息"""
        # 保持原有signal字段处理
        if not hasattr(record, 'signal'):
            record.signal = None  # 默认信号值
            
        # 新增敏感信息过滤
        msg = str(record.msg)
        for pattern in self.SENSITIVE_PATTERNS:
            if re.search(pattern, msg):
                record.msg = "[REDACTED] " + re.sub(pattern, "****", msg)
                break
                
        return True

    def filter_message(self, message: str) -> str:
        """过滤消息中的敏感信息"""
        for pattern in self.SENSITIVE_PATTERNS:
            if re.search(pattern, message):
                return "[REDACTED] " + re.sub(pattern, "****", message)
        return message
