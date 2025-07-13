import re
import logging
from typing import Dict, Any

class SecurityFilter(logging.Filter):
    """增强版敏感信息过滤器"""

    # 预定义敏感模式
    DEFAULT_PATTERNS = {
        'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        'account_id': r'\bacc(_?id)?[:=][\'\"]?(\w+)[\'\"]?\b',
        'order_amount': r'\bamount[:=][\'\"]?(\d+\.?\d*)[\'\"]?\b',
        'api_key': r'\b(?:api|access)_?key[:=][\'\"]?(\w+)[\'\"]?\b',
        'password': r'\bpassword[:=][\'\"]?(\w+)[\'\"]?\b'
    }

    def __init__(self, custom_patterns: Dict[str, str] = None):
        """
        Args:
            custom_patterns: 自定义敏感模式 {name: regex}
        """
        super().__init__()
        self.patterns = self.DEFAULT_PATTERNS.copy()
        if custom_patterns:
            self.patterns.update(custom_patterns)

        self.compiled = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.patterns.items()
        }

    def filter(self, record: logging.LogRecord) -> bool:
        """过滤敏感信息"""
        if not super().filter(record):
            return False

        # 处理消息内容
        record.msg = self._sanitize(record.msg)

        # 处理额外参数
        if hasattr(record, 'context'):
            record.context = {
                k: self._sanitize(v) if isinstance(v, str) else v
                for k, v in record.context.items()
            }

        return True

    def _sanitize(self, text: str) -> str:
        """替换敏感信息为[REDACTED]"""
        if not isinstance(text, str):
            return text

        for name, pattern in self.compiled.items():
            if name == 'account_id':
                # 保留部分信息用于调试
                text = pattern.sub(r'Account ID: [REDACTED]', text)
            elif name == 'order_amount':
                text = pattern.sub(r'amount=[REDACTED]', text)
            elif name == 'password':
                text = pattern.sub(r'password=****', text)
            elif name == 'credit_card':
                text = pattern.sub(r'[REDACTED]', text)
            else:
                text = pattern.sub('[REDACTED]', text)
        return text

    def sanitize_account_id(self, text: str) -> str:
        """专门处理账户ID"""
        pattern = self.compiled.get('account_id')
        if pattern:
            return pattern.sub(r'Account ID: [REDACTED]', text)
        return text

    def sanitize_order_amount(self, text: str) -> str:
        """专门处理订单金额"""
        pattern = self.compiled.get('order_amount')
        if pattern:
            return pattern.sub(r'amount=[REDACTED]', text)
        return text

class AuditLogger:
    """安全审计日志器"""

    def __init__(self):
        self.logger = logging.getLogger('security.audit')
        self.logger.setLevel(logging.INFO)

        # 确保审计日志独立存储
        handler = logging.FileHandler('logs/security_audit.log')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s'
        ))
        self.logger.addHandler(handler)

    def log_sensitive_operation(self,
                              operation: str,
                              user: str,
                              metadata: Dict[str, Any]):
        """记录敏感操作"""
        self.logger.info(
            "敏感操作审计 | 操作类型: %s | 用户: %s | 元数据: %s",
            operation,
            user,
            {k: '[REDACTED]' if 'key' in k else v
             for k, v in metadata.items()}
        )
