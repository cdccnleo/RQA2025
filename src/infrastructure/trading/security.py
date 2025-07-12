import re
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from functools import lru_cache
from prometheus_client import Counter

@dataclass
class FilterResult:
    """过滤结果数据类"""
    filtered: Dict[str, Any]  # 过滤后的上下文
    modified: bool  # 是否进行了修改
    matched_fields: List[str]  # 匹配的敏感字段

class SecurityFilter:
    """增强版安全数据过滤器"""

    _filter_counter = Counter(
        'security_filter_operations',
        'Total security filter operations',
        ['action']
    )

    def __init__(
        self,
        sensitive_fields: Optional[List[str]] = None,
        patterns: Optional[List[str]] = None,
        replace_with: str = "***",
        case_sensitive: bool = False
    ):
        """
        初始化安全过滤器

        Args:
            sensitive_fields: 敏感字段名列表
            patterns: 敏感数据正则模式列表
            replace_with: 替换文本
            case_sensitive: 是否区分大小写
        """
        self.sensitive_fields = set(sensitive_fields or [])
        self.patterns = [re.compile(p) for p in (patterns or [
            r'[A-Za-z0-9]{32}',  # MD5哈希
            r'sk_(live|test)_[A-Za-z0-9]{24}',  # Stripe密钥
            r'AKIA[0-9A-Z]{16}',  # AWS访问密钥
            r'[0-9]{12,19}',  # 长数字(可能为卡号)
            r'eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*',  # JWT
            r'sk-[a-zA-Z0-9]{32}',  # OpenAI密钥
            r'gh[pousr]_[A-Za-z0-9]{36}',  # GitHub令牌
            r'xoxb-[0-9]{11}-[0-9]{11}-[a-zA-Z0-9]{24}'  # Slack令牌
        ])]
        self.replace_with = replace_with
        self.case_sensitive = case_sensitive
        self._field_cache = {}

        # 添加默认敏感字段
        self._add_default_fields()

    def _add_default_fields(self):
        """添加默认敏感字段"""
        default_fields = [
            'password', 'passwd', 'pwd',
            'api_key', 'api_secret',
            'token', 'access_token', 'refresh_token',
            'auth', 'authorization',
            'secret', 'private_key'
        ]
        self.sensitive_fields.update(default_fields)

    @lru_cache(maxsize=1024)
    def _is_sensitive_field(self, field_name: str) -> bool:
        """检查字段名是否敏感(带缓存)"""
        if not self.case_sensitive:
            field_name = field_name.lower()

        # 检查精确匹配
        if field_name in self.sensitive_fields:
            return True

        # 检查包含关系
        for sensitive in self.sensitive_fields:
            if sensitive in field_name:
                return True

        return False

    def _is_sensitive_value(self, value: Any) -> bool:
        """检查字段值是否敏感"""
        if not isinstance(value, str):
            return False

        for pattern in self.patterns:
            if pattern.search(value):
                return True

        return False

    def filter_context(self, context: Dict[str, Any]) -> FilterResult:
        """过滤上下文中的敏感数据"""
        if not context:
            return FilterResult({}, False, [])

        filtered = {}
        modified = False
        matched_fields = []

        for key, value in context.items():
            if self._is_sensitive_field(key) or self._is_sensitive_value(str(value)):
                filtered[key] = self.replace_with
                modified = True
                matched_fields.append(key)
                self._filter_counter.labels(action="redact").inc()
            else:
                filtered[key] = value

        return FilterResult(filtered, modified, matched_fields)

    def __call__(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """使实例可调用"""
        return self.filter_context(context).filtered

class SecurityContext:
    """安全上下文管理器"""

    def __init__(self, *filters: Callable):
        self.filters = list(filters)
        self._filter_chain = None

    def add_filter(self, filter: Callable):
        """添加过滤器"""
        self.filters.append(filter)
        self._filter_chain = None

    def _build_filter_chain(self):
        """构建过滤链"""
        if not self.filters:
            return lambda x: x

        def chain(context):
            for f in self.filters:
                context = f(context)
            return context

        return chain

    def __call__(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """应用所有过滤器"""
        if self._filter_chain is None:
            self._filter_chain = self._build_filter_chain()
        return self._filter_chain(context)
